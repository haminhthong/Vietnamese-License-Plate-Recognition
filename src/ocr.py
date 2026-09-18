"""Tiền xử lý ảnh biển số, nhận dạng EasyOCR và sắp xếp thứ tự ký tự theo hình học.

Quy trình:
1. Fast path: Phóng đại ảnh + Chuyển xám + Lọc nhiễu + Cân bằng tương phản cục bộ (CLAHE).
2. Fallback: Nắn thẳng phối cảnh (Rectification) hoặc phân ngưỡng (Threshold) khi ảnh nghiêng / tương phản kém.
3. Nhận dạng EasyOCR trên vùng biển số.
4. Phân tích hình học token: gom hàng và sắp xếp từ trên xuống dưới, từ trái sang phải cho biển 1 dòng / 2 dòng.
5. Kiểm tra cú pháp biển số Việt Nam và đề xuất sửa lỗi ký tự (nếu có).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import RecognitionConfig
from .grammar import normalize_plate_text, validate_and_correct_plate
from .rectification import rectify_plate

ASCII_DIGITS = "0123456789"
ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OCR_ALLOWLIST = f"{ASCII_DIGITS}{ASCII_LETTERS}-."
VALID_LAYOUTS = {"1_line", "2_line"}


def _token_geometry(bbox: list[list[float]]) -> dict[str, float]:
    """Trích xuất tọa độ tâm và chiều cao của bounding box token OCR."""
    points = np.asarray(bbox, dtype=float)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return {
        "center_x": float((min_x + max_x) / 2),
        "center_y": float((min_y + max_y) / 2),
        "height": float(max(1.0, max_y - min_y)),
    }


def order_ocr_tokens(
    ocr_results: list, layout: str, minimum_confidence: float = 0.20
) -> tuple[str, float, list[dict[str, Any]]]:
    """Lọc token nhiễu và sắp xếp theo thứ tự hình học dựa vào layout 1 dòng hoặc 2 dòng.

    Returns:
        tuple[str, float, list[dict]]: (chuỗi_ký_tự, độ_tin_cậy_trung_bình, danh_sách_tokens).
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence phải nằm trong khoảng [0, 1]")

    tokens = []
    for bbox, text, confidence in ocr_results:
        normalized = normalize_plate_text(text)
        if normalized and float(confidence) >= minimum_confidence:
            tokens.append(
                {
                    "text": normalized,
                    "confidence": float(confidence),
                    **_token_geometry(bbox),
                }
            )

    if not tokens:
        return "", 0.0, []

    median_height = float(np.median([token["height"] for token in tokens]))
    tokens = [token for token in tokens if token["height"] >= 0.45 * median_height]
    if not tokens:
        return "", 0.0, []

    if layout == "2_line" and len(tokens) >= 2:
        ordered = _order_two_line_tokens(tokens, median_height)
    else:
        ordered = sorted(tokens, key=lambda token: token["center_x"])

    text = "".join(token["text"] for token in ordered)
    confidence = float(
        np.average(
            [token["confidence"] for token in ordered],
            weights=[max(1, len(token["text"])) for token in ordered],
        )
    )
    return text, confidence, ordered


def _order_two_line_tokens(tokens: list[dict[str, Any]], median_height: float) -> list[dict[str, Any]]:
    """Phân tách tokens thành 2 dòng (trên / dưới) và sắp xếp từng dòng từ trái sang phải."""
    sorted_by_y = sorted(tokens, key=lambda token: token["center_y"])
    gaps = [
        sorted_by_y[index + 1]["center_y"] - sorted_by_y[index]["center_y"]
        for index in range(len(sorted_by_y) - 1)
    ]
    largest_gap = max(gaps, default=0.0)
    if largest_gap >= 0.25 * median_height:
        split_index = int(np.argmax(gaps)) + 1
        rows = [sorted_by_y[:split_index], sorted_by_y[split_index:]]
    else:
        median_y = float(np.median([token["center_y"] for token in tokens]))
        rows = [
            [token for token in tokens if token["center_y"] <= median_y],
            [token for token in tokens if token["center_y"] > median_y],
        ]
    non_empty_rows = [row for row in rows if row]
    non_empty_rows.sort(key=lambda row: np.mean([token["center_y"] for token in row]))
    return [token for row in non_empty_rows for token in sorted(row, key=lambda token: token["center_x"])]


def infer_plate_layout(
    crop_bgr: np.ndarray,
    wide_ratio_threshold: float = 2.20,
    tokens: list[dict[str, Any]] | None = None,
) -> str:
    """Xác định bố cục biển số: 1_line (biển dài ô tô) hoặc 2_line (xe máy / biển vuông ô tô)."""
    if wide_ratio_threshold <= 0:
        raise ValueError("wide_ratio_threshold phải lớn hơn 0")
    if crop_bgr is None or crop_bgr.size == 0:
        return "1_line"
    height, width = crop_bgr.shape[:2]
    initial_layout = "1_line" if width / max(1, height) >= wide_ratio_threshold else "2_line"

    if tokens and len(tokens) >= 2:
        sorted_by_y = sorted(tokens, key=lambda t: t["center_y"])
        y_gaps = [
            sorted_by_y[i + 1]["center_y"] - sorted_by_y[i]["center_y"] for i in range(len(sorted_by_y) - 1)
        ]
        median_h = float(np.median([t["height"] for t in tokens]))
        if max(y_gaps, default=0.0) >= 0.25 * median_h:
            return "2_line"

    return initial_layout


def preprocess_plate(crop_bgr: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Tiền xử lý ảnh crop biển số: phóng đại, làm xám, làm mượt và tăng tương phản CLAHE."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    height, width = crop_bgr.shape[:2]
    scale = float(np.clip(180 / max(height, width), 2.0, 4.0))
    enlarged = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    denoised = cv2.bilateralFilter(gray, 9, 55, 55)

    if method == "gray":
        return denoised
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    if method == "adaptive":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
        return cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    return denoised


def read_plate(
    reader: Any,
    crop_bgr: np.ndarray,
    layout: str = "auto",
    config: RecognitionConfig | None = None,
    detection_confidence: float = 1.0,
) -> dict[str, Any]:
    """Quy trình OCR hoàn chỉnh trên vùng ảnh crop biển số:

    1. Fast path: CLAHE preprocessing -> EasyOCR.
    2. Fallback: Nếu không đọc được hoặc confidence quá thấp -> Thử rectification hoặc adaptive threshold.
    3. Phân loại hình học tokens 1 dòng / 2 dòng.
    4. Kiểm tra cú pháp biển số Việt Nam và đề xuất sửa lỗi (nếu có).
    """
    if layout != "auto" and layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Crop biển số bị rỗng.")
    cfg = config or RecognitionConfig()

    initial_layout = (
        layout if layout in VALID_LAYOUTS else infer_plate_layout(crop_bgr, cfg.wide_ratio_threshold)
    )

    # 1. Fast path: CLAHE
    preprocessed = preprocess_plate(crop_bgr, method="clahe")
    ocr_results = reader.readtext(preprocessed, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST)
    selected_ocr_results = ocr_results
    raw_text, ocr_conf, tokens = order_ocr_tokens(
        ocr_results, initial_layout, minimum_confidence=cfg.ocr_minimum_confidence
    )

    rectified = False
    # 2. Fallback: Nếu text rỗng hoặc ocr_conf thấp và có bật rectification/variants
    if (not raw_text or ocr_conf < cfg.ocr_threshold) and cfg.enable_preprocessing_variants:
        fallback_crop = crop_bgr
        if cfg.enable_rectification:
            rectified_crop, was_rectified = rectify_plate(crop_bgr)
            if was_rectified:
                fallback_crop = rectified_crop
                rectified = True

        fallback_img = preprocess_plate(fallback_crop, method="adaptive")
        fallback_results = reader.readtext(fallback_img, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST)
        fb_text, fb_conf, fb_tokens = order_ocr_tokens(
            fallback_results, initial_layout, minimum_confidence=cfg.ocr_minimum_confidence
        )

        # Nếu fallback cho kết quả tốt hơn, chọn fallback
        if fb_text and (not raw_text or fb_conf > ocr_conf):
            raw_text, ocr_conf, tokens = fb_text, fb_conf, fb_tokens
            selected_ocr_results = fallback_results
            rectified = was_rectified

    # Xác định layout cuối cùng dựa trên tokens
    final_layout = infer_plate_layout(crop_bgr, cfg.wide_ratio_threshold, tokens=tokens)
    if final_layout != initial_layout and tokens:
        raw_text, ocr_conf, tokens = order_ocr_tokens(
            selected_ocr_results,
            final_layout,
            minimum_confidence=cfg.ocr_minimum_confidence,
        )

    # 3. Kiểm tra cú pháp biển số Việt Nam và đề xuất sửa lỗi
    syntax_result = validate_and_correct_plate(
        raw_text,
        enable_correction=cfg.enable_template_correction,
    )

    needs_review = (
        (not syntax_result["format_valid"])
        or (ocr_conf < cfg.ocr_threshold)
        or (detection_confidence < cfg.auto_accept_detector_threshold)
        or (syntax_result["suggested_text"] is not None)
    )

    return {
        "text": syntax_result["text"],
        "raw_text": syntax_result["raw_text"],
        "ocr_confidence": round(ocr_conf, 4),
        "format_valid": syntax_result["format_valid"],
        "suggested_text": syntax_result["suggested_text"],
        "needs_review": needs_review,
        "layout": final_layout,
        "rectified": rectified,
        "tokens": tokens,
    }
