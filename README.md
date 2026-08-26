---
title: AIChamcong - Face Attendance System
emoji: 🧑‍💼
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🎯 Face Attendance System - Hệ thống Chấm công Tự động

**Nhận diện khuôn mặt AI-powered cho quản lý chấm công hiệu quả**

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5c3ee1?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🌐 Live Demo](https://aichamcong.vercel.app) • [📖 Tài liệu](#tài-liệu) • [🚀 Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)

</div>

---

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng nổi bật](#tính-năng-nổi-bật)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Đánh giá hiệu năng](#đánh-giá-hiệu-năng)
- [Đội phát triển](#đội-phát-triển)
- [Giấy phép](#giấy-phép)

---

## 🎯 Giới thiệu

**AIChamcong** là một hệ thống chấm công tự động sử dụng công nghệ nhận diện khuôn mặt AI tiên tiến, thay thế các phương pháp chấm công truyền thống như ký tên hay thẻ chấm công.

### 🔑 Điểm nổi bật:
- 📸 **Xử lý real-time** từ webcam/camera
- 🧠 **Nhiều mô hình AI** (InsightFace, MobileNetV2, ResNet-50)
- ⚡ **Tối ưu cho CPU & GPU** - chạy mượt trên nhiều nền tảng
- 📊 **Đánh giá chuẩn quốc tế** trên LFW, CALFW, CPLFW, AgeDB-30
- 🔐 **Chống giả mạo** - phát hiện ảnh/video giả mạo
- 💾 **Quản lý cơ sở dữ liệu** - lưu trữ thông tin nhân viên và lịch chấm công

---

## ✨ Tính năng nổi bật

| Tính năng | Mô tả |
|-----------|--------|
| 🎥 **Real-time Face Recognition** | Xử lý luồng video từ webcam với độ trễ thấp |
| 🧠 **Multi-Model Support** | InsightFace, MobileNetV2, ResNet-50, VGG-Face |
| 🏋️ **Model Training** | Pipeline huấn luyện hoàn chỉnh với ArcFace Loss |
| 📈 **Benchmark Evaluation** | Đánh giá trên 4 tập dữ liệu chuẩn quốc tế |
| 👤 **Employee Management** | Tự động thu thập ảnh khuôn mặt từ camera |
| 🛡️ **Anti-Spoofing** | Phát hiện hình ảnh/video giả mạo |
| 💾 **Database Integration** | SQLite cho lưu trữ nhân viên và log chấm công |
| 📊 **Analytics Dashboard** | Thống kê và báo cáo chấm công |

---

## 🛠️ Công nghệ sử dụng

### Backend & AI
- **Python 3.10+** - Ngôn ngữ lập trình chính
- **PyTorch** - Framework deep learning cho training & inference
- **ONNX Runtime** - Tối ưu hóa inference trên CPU/GPU
- **InsightFace** - State-of-the-art face detection & recognition
- **OpenCV** - Xử lý video và tiền xử lý hình ảnh
- **Scikit-learn** - Tính toán metrics và evaluation

### Frontend
- **React 18** + **Vite** - UI framework hiện đại
- **JavaScript/TypeScript** - Frontend logic
- **CSS3** - Styling responsive

### Infrastructure
- **SQLite** - Database nhẹ cho hệ thống
- **Docker** - Containerization
- **TensorBoard** - Visualization training

---

## 🏗️ Kiến trúc hệ thống

### Luồng xử lý chính

```
┌─────────────────────────┐
│  Camera / Webcam        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Face Detection (SCRFD)          │ ← InsightFace
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Face Alignment & Normalization  │ ← OpenCV
└────────────┬────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Feature Extraction (Embedding)       │ ← 512D Vector
│ MobileNetV2 / InsightFace / ResNet-50│
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Cosine Similarity Matching           │
│ Compare với Database Embeddings      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Threshold Verification & Validation  │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Attendance Recording                 │
│ Save to Database & Update Dashboard  │
└──────────────────────────────────────┘
```

### Model Architecture

| Thành phần | Công nghệ | Chi tiết |
|-----------|-----------|---------|
| **Face Detection** | SCRFD | Nhanh, chính xác cao |
| **Backbone (Feature Extraction)** | MobileNetV2 / ResNet-50 | 512D embedding vector |
| **Loss Function** | ArcFace Margin Loss | Tối ưu hóa khoảng cách giữa classes |
| **Deep Learning Framework** | PyTorch + ONNX | CPU/GPU inference support |
| **Evaluation Metrics** | Scikit-learn | Accuracy, AUC, EER, FAR, FRR |

---

## 📁 Cấu trúc thư mục

```
Attendance_tracking/
│
├── 📂 src/                          # Source code chính
│   ├── attendance/                  # Logic chấm công & database
│   │   ├── realtime_recognition.py  # Chạy hệ thống real-time
│   │   ├── build_embeddings.py      # Tạo embeddings từ dữ liệu
│   │   └── database.py              # SQLite database functions
│   │
│   ├── capture/                     # Thu thập dữ liệu khuôn mặt
│   │   └── capture_dataset.py       # Script capture từ camera
│   │
│   ├── models/                      # Định nghĩa kiến trúc mạng
│   │   ├── mobilenetv2.py           # MobileNetV2 + ArcFace
│   │   ├── resnet.py                # Custom ResNet-50
│   │   └── arcface.py               # ArcFace Loss implementation
│   │
│   ├── training/                    # Training scripts
│   │   ├── train_mobilenetv2_arcface.py  # Train MobileNetV2
│   │   └── train_vggface2.py             # Train ResNet-50
│   │
│   ├── evaluation/                  # Evaluation & Benchmark
│   │   ├── evaluate_mobilenetv2.py  # Evaluate MobileNetV2
│   │   ├── evaluate_vggface2.py     # Evaluate ResNet-50
│   │   └── evaluate_benchmark.py    # InsightFace baseline
│   │
│   ├── datasets/                    # PyTorch Dataset loaders
│   │   ├── face_dataset.py          # Custom FaceDataset
│   │   └── benchmark_dataset.py     # LFW, CALFW, etc.
│   │
│   └── utils/                       # Utility functions
│       ├── transforms.py            # Data augmentation
│       ├── logging.py               # Custom logging
│       └── metrics.py               # Evaluation metrics
│
├── 📂 frontend_web/                 # React frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── 📂 dataset/                      # Training & Benchmark datasets
│   ├── benchmark/                   # LFW, CALFW, CPLFW, AgeDB-30
│   └── vggface2_hq/                 # VGGFace2 training data
│
├── 📂 data/                         # System data (in-house)
│   ├── inhouse/                     # Registered employee faces
│   └── embeddings/                  # Pre-computed embeddings (.pkl)
│
├── 📂 checkpoints/                  # Model checkpoints (.pth)
│   ├── mobilenetv2_best.pth
│   ├── resnet50_best.pth
│   └── ...
│
├── 📂 logs/                         # Training logs (CSV, TensorBoard)
├── 📂 outputs/                      # Evaluation results & plots
├── 📂 notebooks/                    # Jupyter notebooks for EDA
│
├── requirements.txt                 # Python dependencies
├── .dockerignore
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker compose
└── README.md                        # This file
```

---

## 🚀 Hướng dẫn cài đặt

### 📋 Yêu cầu hệ thống

```
✓ CPU: Intel Core i5 / AMD Ryzen 5 trở lên
✓ RAM: 8GB+ (16GB+ nếu train model)
✓ GPU: NVIDIA GPU với VRAM 8GB+ (tùy chọn, khuyến nghị)
✓ Python: 3.10+
✓ OS: Linux / macOS / Windows
```

### 🔧 Bước 1: Clone repository

```bash
git clone https://github.com/QDat18/Attendance_tracking.git
cd Attendance_tracking
```

### 📦 Bước 2: Tạo môi trường ảo

**Với Conda:**
```bash
conda create -n attendance python=3.10
conda activate attendance
```

**Với Venv (Python):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc: venv\Scripts\activate  # Windows
```

### ⚙️ Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Cài đặt ONNX Runtime (tùy chọn):**

```bash
# Cho CPU
pip install onnxruntime

# Cho GPU NVIDIA
pip install onnxruntime-gpu
```

### 🐳 Bước 4 (Tùy chọn): Chạy với Docker

```bash
docker-compose up -d
```

---

## 📖 Hướng dẫn sử dụng

### 1️⃣ Thu thập khuôn mặt nhân viên mới

```bash
# Chụp 50 ảnh cho nhân viên ID = NV001
python -m src.capture.capture_dataset --employee_id NV001 --max_images 50
```

**Tham số:**
- `--employee_id`: ID nhân viên (bắt buộc)
- `--max_images`: Số ảnh thu thập (mặc định: 30)
- `--output_dir`: Thư mục lưu ảnh (mặc định: `data/inhouse/`)

### 2️⃣ Tạo embeddings từ dữ liệu nhân viên

```bash
# Xử lý tất cả ảnh và tạo embeddings
python -m src.attendance.build_embeddings
```

Điều này sẽ:
- Phát hiện khuôn mặt từ ảnh nhân viên
- Trích xuất 512D embedding vector
- Lưu vào `data/embeddings/employee_embeddings.pkl`

### 3️⃣ Chạy hệ thống chấm công real-time

```bash
# Mở webcam và bắt đầu nhận diện
python -m src.attendance.realtime_recognition
```

**Các tính năng:**
- 📹 Hiển thị video real-time từ webcam
- 👤 Nhận diện tự động khuôn mặt
- ⏱️ Ghi nhận thời gian chấm công
- 💾 Lưu vào database SQLite

**Điều khiển:**
- Nhấn `Q` để thoát
- Nhấn `S` để lưu ảnh snapshot

### 4️⃣ Huấn luyện mô hình (Optional)

#### **Huấn luyện MobileNetV2 (Nhẹ & nhanh)**

```bash
python -m src.training.train_mobilenetv2_arcface \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --device cuda  # hoặc 'cpu'
```

#### **Huấn luyện ResNet-50 Custom (Độ chính xác cao)**

```bash
python -m src.training.train_vggface2 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --device cuda
```

### 5️⃣ Đánh giá hiệu năng mô hình

#### **Đánh giá MobileNetV2**

```bash
python -m src.evaluation.evaluate_mobilenetv2 --dataset all
```

#### **Đánh giá ResNet-50**

```bash
python -m src.evaluation.evaluate_vggface2 --dataset all
```

#### **So sánh với InsightFace Baseline**

```bash
python -m src.evaluation.evaluate_benchmark \
    --model-type baseline \
    --dataset all
```

**Kết quả sẽ lưu tại:** `outputs/evaluation/`
- ROC curves
- Confusion matrices
- CSV reports
- Performance plots

---

## 📊 Đánh giá hiệu năng

### Các chỉ số (Metrics)

| Chỉ số | Giải thích | Ý nghĩa |
|--------|-----------|--------|
| **Accuracy** | Tỷ lệ nhận diện đúng | Càng cao càng tốt (%) |
| **Precision** | Tỷ lệ dương tính đúng | Độ tin cậy khi nhận diện |
| **Recall** | Tỷ lệ phát hiện | Khả năng bắt được tất cả |
| **AUC** | Diện tích dưới ROC curve | 0.0 → 1.0 (1.0 là tốt nhất) |
| **EER** | Equal Error Rate | Điểm cân bằng FAR=FRR (% thấp hơn tốt) |
| **FAR** | False Acceptance Rate | Nhận nhầm người lạ (% thấp hơn tốt) |
| **FRR** | False Rejection Rate | Không nhận ra nhân viên (% thấp hơn tốt) |

### Benchmark Results

Các mô hình được đánh giá trên 4 tập chuẩn quốc tế:

- **LFW** (Labeled Faces in the Wild) - 6,000 cặp ảnh
- **CALFW** (Cross-Age LFW) - 4,025 cặp ảnh
- **CPLFW** (Celebs in Profiles in the Wild) - 6,000 cặp ảnh
- **AgeDB-30** (Age Database) - 12,000 cặp ảnh

---

## 🤝 Đội phát triển

**Tên nhóm:** Nhóm 4 - Lớp Trí Tuệ Nhân Tạo

**Trường:** Học viện Ngân hàng

**Đơn vị:** Khoa Công Nghệ Thông Tin

---

## 📝 Giấy phép (License)

Dự án này được xây dựng cho mục đích **học tập và nghiên cứu**.

Việc sử dụng các mô hình, dataset tuân thủ giấy phép nguồn mở:
- 📚 **LFW Dataset** - [License](http://vis-www.cs.umass.edu/lfw/)
- 🎬 **VGGFace2** - [License](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- 🧠 **InsightFace** - [MIT License](https://github.com/deepinsight/insightface)
- 🔧 **PyTorch** - [BSD License](https://github.com/pytorch/pytorch)
- 📷 **OpenCV** - [Apache 2 License](https://github.com/opencv/opencv)

---

## 🌐 Liên kết hữu ích

- 🚀 [Live Demo](https://aichamcong.vercel.app)
- 📖 [InsightFace GitHub](https://github.com/deepinsight/insightface)
- 🔗 [PyTorch Documentation](https://pytorch.org)
- 📚 [OpenCV Documentation](https://docs.opencv.org)

---

## 💡 Ghi chú

- Hệ thống hoạt động tốt nhất trong điều kiện ánh sáng tốt
- Khuyến nghị sử dụng khoảng 50+ ảnh để đăng ký mỗi nhân viên
- GPU NVIDIA đáng khuyến nghị cho việc huấn luyện model
- Thư mục `data/inhouse/` chứa dữ liệu nhạy cảm - hãy bảo vệ nó tốt!

---

<div align="center">

**Made with ❤️ by Group 4**

⭐ Nếu bạn thích dự án này, hãy cho nó một star!

</div>
