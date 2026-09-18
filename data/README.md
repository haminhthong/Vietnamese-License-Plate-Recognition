# Hướng Dẫn Dữ Liệu (Data Guide)

Repository không phân phối ảnh hoặc biển số xe thực tế vì lý do bản quyền và quyền riêng tư. Dữ liệu huấn luyện và đánh giá cần được chuẩn bị theo cấu trúc sau:

## 1. Dữ liệu huấn luyện YOLOv8

Dữ liệu phát hiện biển số tuân theo định dạng chuẩn của Ultralytics YOLO. Thư mục nguồn truyền cho `prepare_dataset.py` phải có ba split `train/`, `valid/`, `test/`; mỗi split có `images/` và `labels/`:

```text
data/raw/
├── train/images/    # JPG, JPEG, PNG, BMP hoặc WebP
├── train/labels/    # Tệp .txt cùng tên ảnh
├── valid/images/
├── valid/labels/
├── test/images/
└── test/labels/
```

Mỗi dòng nhãn có định dạng `class_id x_center y_center width height`; tọa độ phải được chuẩn hóa trong `[0, 1]`. Lớp duy nhất là `0: license_plate`. Ảnh không có tệp label tương ứng hoặc label sai định dạng sẽ làm script dừng để tránh đưa dữ liệu lỗi vào pipeline.

```bash
python scripts/prepare_dataset.py \
  --source data/raw \
  --metadata data/raw/metadata.csv \
  --output dataset/grouped \
  --audit-output artifacts/dataset_audit.json
```

`--metadata` là tùy chọn. CSV metadata có thể dùng một trong các khóa ảnh `image_path`, `image_name`, `image_id` và các cột `capture_group`, `plate_identity`, `plate_text`. Script phát hiện ảnh trùng MD5, gộp ảnh cùng nhóm chụp hoặc cùng identity, sau đó tạo Group-Safe Split. Dữ liệu đầu ra dùng `train/`, `val/`, `test/`, tạo `data.yaml`, `split_manifest.csv` và báo cáo audit JSON.

## 2. Dữ liệu đánh giá OCR và End-to-End

- **Đánh giá OCR:** CSV phải có `crop_path`, `plate_text` và tùy chọn `layout` (`1_line` hoặc `2_line`). Đường dẫn tương đối được giải quyết theo thư mục chứa CSV. Xem `data/ocr_annotations.example.csv`.
- **Đánh giá End-to-End:** CSV phải có `image_path`, `x1`, `y1`, `x2`, `y2`, `plate_text`. Nhiều dòng có thể cùng trỏ tới một ảnh để biểu diễn nhiều biển số. Xem `data/end_to_end_annotations.example.csv`.

Hai file `.example.csv` chỉ là hợp đồng cột; các ảnh `sample/` không được phân phối trong repository. Cần thay đường dẫn bằng dữ liệu thật trước khi chạy evaluator.
