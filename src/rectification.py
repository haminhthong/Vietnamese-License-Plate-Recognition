"""Thuật toán chỉnh phối cảnh (Perspective Rectification / Deskew) cho vùng ảnh biển số bị nghiêng.

Module này hỗ trợ:
1. Sắp xếp 4 điểm góc của tứ giác biển số theo chiều kim đồng hồ: Góc trên-trái, trên-phải, dưới-phải, dưới-trái.
2. Biến đổi ma trận phối cảnh (Perspective Transform) thành ảnh hình chữ nhật phẳng góc nhìn thẳng.
3. Tự động tìm đường viền tứ giác (Contour) bằng Canny Edge Detection & Gaussian Blur, có cơ chế dự phòng (fallback) giữ nguyên ảnh gốc nếu không tìm thấy tứ giác hợp lệ.
"""

from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 đỉnh của một tứ giác theo chiều kim đồng hồ: [trên-trái, trên-phải, dưới-phải, dưới-trái].

    Giải thuật:
    - Tổng tọa độ (x + y) nhỏ nhất ở góc trên-trái (top-left), lớn nhất ở góc dưới-phải (bottom-right).
    - Hiệu tọa độ (y - x) nhỏ nhất ở góc trên-phải (top-right), lớn nhất ở góc dưới-trái (bottom-left).

    Args:
        points (np.ndarray): Mảng 4 điểm tọa độ dạng (4, 2).

    Returns:
        np.ndarray: Mảng 4 điểm đã được sắp xếp chuẩn hóa định dạng float32.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def warp_quad(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Thực hiện biến đổi phối cảnh nắn phẳng một hình tứ giác bất kỳ thành hình chữ nhật nhìn thẳng.

    Args:
        image (np.ndarray): Ảnh BGR gốc.
        points (np.ndarray): 4 điểm đỉnh của vùng biển số nghiêng.

    Returns:
        np.ndarray: Ảnh mới đã được biến đổi ma trận perspective transform.
    """
    top_left, top_right, bottom_right, bottom_left = order_points(points)
    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    if width < 2 or height < 2:
        return image
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left]),
        destination,
    )
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def rectify_plate(crop_bgr: np.ndarray, minimum_area_ratio: float = 0.35) -> tuple[np.ndarray, bool]:
    """Phát hiện biên tứ giác biển số nghiêng và chỉnh thẳng phối cảnh (Deskew).

    Nếu không phát hiện được đường viền tứ giác thỏa mãn diện tích tối thiểu hoặc tỷ lệ khung hình không hợp lệ,
    hàm sẽ kích hoạt cơ chế dự phòng (fallback) và trả về crop gốc.

    Args:
        crop_bgr (np.ndarray): Ảnh crop biển số BGR.
        minimum_area_ratio (float): Diện tích đường viền tối thiểu so với diện tích crop (mặc định: 0.35).

    Returns:
        tuple[np.ndarray, bool]: Cặp (ảnh_đã_nắn_hoặc_ảnh_gốc, cờ_đã_nắn_thành_công).

    Raises:
        ValueError: Nếu minimum_area_ratio nằm ngoài khoảng (0, 1].
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr, False
    if not 0 < minimum_area_ratio <= 1:
        raise ValueError("minimum_area_ratio phải nằm trong khoảng (0, 1]")
    height, width = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Ảnh phẳng hoặc ảnh một màu không thể chứa đường viền tứ giác hữu ích.
    # Trả về sớm cũng giúp fallback không phụ thuộc vào lỗi xử lý Canny của backend OpenCV.
    if gray.size == 0 or float(gray.std()) == 0.0:
        return crop_bgr, False
    try:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 180)
    except cv2.error:
        return crop_bgr, False
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(contour) < minimum_area_ratio * height * width:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        rectified = warp_quad(crop_bgr, polygon.reshape(4, 2))
        out_height, out_width = rectified.shape[:2]
        if 1.0 <= out_width / max(1, out_height) <= 6.5:
            return rectified, True
    return crop_bgr, False
