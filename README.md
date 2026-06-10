---
title: AIChamcong
emoji: 🧑‍💼
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Hệ thống Chấm công Tự động bằng Nhận diện Khuôn mặt (Face Attendance System)

> Hệ thống chấm công tự động sử dụng camera thời gian thực, ứng dụng các mô hình nhận diện khuôn mặt tiên tiến như **InsightFace (iResNet-100)**, **MobileNetV2**, và **VGG / ResNet** với hàm loss **ArcFace** nhằm xác thực nhân viên và ghi nhận thời gian làm việc một cách nhanh chóng, chính xác.

---

## 1. Giới thiệu dự án

Dự án xây dựng một hệ thống chấm công tự động bằng nhận diện khuôn mặt, hướng tới việc thay thế các phương pháp chấm công truyền thống như ký tên, quẹt thẻ hoặc vân tay.
Hệ thống sử dụng camera/webcam để thu nhận hình ảnh khuôn mặt theo thời gian thực. Sau đó, mô hình AI sẽ phát hiện khuôn mặt, trích xuất đặc trưng, so sánh với dữ liệu nhân viên đã đăng ký và tự động ghi nhận kết quả chấm công vào cơ sở dữ liệu.

Đặc biệt, dự án hỗ trợ cả việc **sử dụng pretrained models (InsightFace)** để triển khai nhanh, lẫn **huấn luyện (train/fine-tune)** các mô hình nhẹ hơn như **MobileNetV2** để tối ưu hóa hiệu năng trên thiết bị có tài nguyên hạn chế.

---

## 2. Tính năng nổi bật

* **Nhận diện khuôn mặt Real-time:** Xử lý luồng video từ webcam để phát hiện và nhận diện khuôn mặt với độ trễ thấp.
* **Hỗ trợ đa mô hình (Multi-model Support):**
  * **InsightFace (buffalo_l):** Backbone iResNet-100 cho độ chính xác cực cao.
  * **MobileNetV2 + ArcFace:** Trọng số nhẹ, tốc độ cao, phù hợp chạy trên CPU.
  * **VGG-Face / Custom ResNet:** Phục vụ nghiên cứu và đánh giá so sánh.
* **Huấn luyện mô hình (Training):** Tích hợp pipeline huấn luyện hoàn chỉnh với ArcFace, tự động checkpointing và logging (TensorBoard / CSV).
* **Đánh giá chuẩn (Benchmark Evaluation):** Đánh giá các mô hình trên các tập dữ liệu chuẩn: **LFW, CALFW, CPLFW, AgeDB-30**.
* **Quản lý dữ liệu nhân viên:** Công cụ tự động thu thập ảnh khuôn mặt từ camera để đăng ký nhân viên mới.
* **Chống giả mạo (Anti-spoofing):** (Đang phát triển) Hỗ trợ phát hiện hình ảnh/video giả mạo.

---

## 3. Kiến trúc kỹ thuật

| Thành phần              | Công nghệ sử dụng        | Vai trò                                                 |
| ----------------------- | ------------------------ | ------------------------------------------------------- |
| Lập trình               | Python 3.10+             | Xây dựng core logic, xử lý ảnh, AI và API               |
| Computer Vision         | OpenCV, Pillow           | Đọc luồng video, tiền xử lý, căn chỉnh khuôn mặt        |
| Face Detection          | SCRFD (InsightFace)      | Phát hiện và cắt khuôn mặt chuẩn xác                    |
| Feature Extraction      | MobileNetV2 / ResNet-50  | Mạng Backbone trích xuất vector đặc trưng (512-dim)     |
| Loss Function           | ArcFace Margin Loss      | Tăng khoảng cách giữa các class (identities)            |
| Deep Learning           | PyTorch, ONNX Runtime    | Huấn luyện (Train/Val) và Inference mô hình             |
| Đánh giá (Evaluation)   | Scikit-learn, Matplotlib | Tính toán Accuracy, AUC, EER, vẽ ROC/Confusion Matrix   |
| Database                | SQLite                   | Lưu trữ thông tin nhân viên, logs chấm công             |

---

## 4. Pipeline xử lý

```text
Camera / Webcam
        |
        v
Phát hiện khuôn mặt (Face Detection - SCRFD)
        |
        v
Căn chỉnh khuôn mặt (Face Alignment)
        |
        v
Trích xuất Embedding (MobileNetV2 / InsightFace) -> Vector 512D
        |
        v
Tính Cosine Similarity với Vector gốc trong Database
        |
        v
Vượt ngưỡng (Threshold) -> Xác thực danh tính
        |
        v
Ghi nhận chấm công vào Database
```

---

## 5. Cấu trúc thư mục

```text
face_attendance/
├── checkpoints/              # Thư mục lưu các model checkpoints (.pth) khi train
├── dataset/                  # Dữ liệu ảnh phục vụ training và evaluation
│   ├── benchmark/            # LFW, CALFW, CPLFW, AgeDB-30
│   └── vggface2_hq/          # Tập dữ liệu train
├── data/                     # Dữ liệu hệ thống chấm công (inhouse data)
│   ├── inhouse/              # Ảnh nhân viên đã đăng ký
│   └── embeddings/           # Vector đặc trưng đã lưu (.pkl)
├── logs/                     # File CSV lưu log quá trình train
├── outputs/                  # Kết quả evaluation (biểu đồ ROC, CM, CSV)
├── src/                      # Source code chính
│   ├── attendance/           # Logic chấm công và database
│   ├── capture/              # Scripts thu thập khuôn mặt từ camera
│   ├── datasets/             # PyTorch Dataset Loaders
│   ├── evaluation/           # Scripts đánh giá (LFW, đa tập, so sánh models)
│   ├── models/               # Định nghĩa kiến trúc mạng (MobileNetV2, ResNet, ArcFace)
│   ├── training/             # Scripts huấn luyện mô hình (train_mobilenetv2, vgg)
│   └── utils/                # Hàm tiện ích (logging, transforms)
├── notebooks/                # Jupyter Notebooks phục vụ EDA và thử nghiệm
├── requirements.txt          # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 6. Hướng dẫn cài đặt

### 6.1 Yêu cầu hệ thống
* CPU: Intel Core i5 trở lên
* RAM: 8GB+ (Khuyến nghị 16GB để train model)
* GPU: NVIDIA GPU (Khuyến nghị VRAM 8GB+ nếu train mô hình)
* Python: 3.10 trở lên

### 6.2 Cài đặt môi trường

Clone repository và di chuyển vào thư mục dự án:
```bash
git clone https://github.com/QDat18/-Group4-TriTueNhanTao.git
cd -Group4-TriTueNhanTao
```

Tạo và kích hoạt môi trường ảo (Conda/Venv):
```bash
conda create -n deepfake python=3.10
conda activate deepfake
```

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Nếu chạy Inference với CPU, cài đặt ONNX Runtime:
```bash
pip install onnxruntime
```
Nếu có GPU NVIDIA, cài đặt bản GPU:
```bash
pip install onnxruntime-gpu
```

---

## 7. Hướng dẫn sử dụng

### 7.1 Huấn luyện mô hình (Training)

**1. Huấn luyện mô hình MobileNetV2 (Nhẹ, tốc độ cao):**
Để tự huấn luyện mô hình MobileNetV2 từ đầu với ArcFace:
```bash
python -m src.training.train_mobilenetv2_arcface
```

**2. Huấn luyện mô hình Custom ResNet-50 (Độ chính xác cao):**
Mô hình này được huấn luyện trên các tập dữ liệu lớn như VGGFace2 để tối ưu hóa khả năng trích xuất đặc trưng khuôn mặt:
```bash
python -m src.training.train_vggface2
```

Quá trình huấn luyện sẽ tự động lưu lại các checkpoints tại `checkpoints/` và lịch sử training (loss, accuracy) vào thư mục `logs/`.

### 7.2 Đánh giá mô hình (Evaluation)

**Đánh giá MobileNetV2** trên tất cả các tập benchmark (LFW, CALFW, CPLFW, AgeDB-30):
```bash
python -m src.evaluation.evaluate_mobilenetv2 --dataset all
```

**Đánh giá Custom ResNet-50 (VGG Checkpoints)** trên các tập benchmark:
```bash
python -m src.evaluation.evaluate_vggface2 --dataset all
```

**Đánh giá InsightFace (Baseline)** trên các tập benchmark để so sánh:
```bash
python -m src.evaluation.evaluate_benchmark --model-type baseline --dataset all
```
Kết quả đánh giá, biểu đồ ROC, và Confusion Matrix sẽ được lưu tại thư mục `outputs/evaluation/`.

### 7.3 Sử dụng hệ thống chấm công (Inference/Realtime)

**Bước 1: Thu thập khuôn mặt nhân viên mới**
```bash
python -m src.capture.capture_dataset --employee_id NV001 --max_images 50
```

**Bước 2: Cập nhật Embedding cho toàn bộ hệ thống**
```bash
python -m src.attendance.build_embeddings
```

**Bước 3: Chạy ứng dụng chấm công Real-time**
```bash
python -m src.attendance.realtime_recognition
```
Hệ thống sẽ mở webcam, tự động nhận diện và cập nhật vào file/database lịch sử.

---

## 8. Các chỉ số đánh giá (Metrics)

Hệ thống sử dụng các tiêu chuẩn đo lường khắt khe trong nhận diện sinh trắc học:
* **Accuracy:** Tỷ lệ nhận diện đúng trên toàn bộ cặp (Pair matching).
* **AUC (Area Under ROC Curve):** Khả năng phân biệt của mô hình.
* **EER (Equal Error Rate):** Điểm tối ưu mà ở đó tỷ lệ từ chối sai (FRR) bằng tỷ lệ chấp nhận sai (FAR). Càng thấp càng tốt.
* **FAR (False Acceptance Rate):** Tỷ lệ nhận diện nhầm người lạ thành người quen.
* **FRR (False Rejection Rate):** Tỷ lệ không nhận ra nhân viên.

---

## 9. Đội ngũ phát triển

**Tên nhóm:** Nhóm 4 (Lớp Trí Tuệ Nhân Tạo)
**Trường:** Học viện Ngân hàng

---

## 10. Giấy phép (License)
Dự án được xây dựng nhằm mục đích học tập, nghiên cứu môn học. Việc sử dụng các mô hình, dataset (LFW, VGGFace2) tuân thủ giấy phép nguồn mở tương ứng. Không sử dụng cho mục đích thương mại khi chưa có sự cho phép của chủ sở hữu bộ dữ liệu.
