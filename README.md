---
title: AIChamcong
emoji: 🧑‍💼
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Hệ thống Chấm công Tự động bằng Nhận diện Khuôn mặt

> Hệ thống chấm công tự động sử dụng camera thời gian thực, ứng dụng mô hình nhận diện khuôn mặt dựa trên InsightFace, ArcFace và iResNet-100 nhằm xác thực nhân viên và ghi nhận thời gian làm việc một cách nhanh chóng, chính xác và hạn chế gian lận.

---

## 1. Giới thiệu dự án

Dự án xây dựng một hệ thống chấm công tự động bằng nhận diện khuôn mặt, hướng tới việc thay thế các phương pháp chấm công truyền thống như ký tên, quẹt thẻ hoặc vân tay.

Hệ thống sử dụng camera/webcam để thu nhận hình ảnh khuôn mặt theo thời gian thực. Sau đó, mô hình trí tuệ nhân tạo sẽ phát hiện khuôn mặt, trích xuất đặc trưng, so sánh với dữ liệu nhân viên đã đăng ký và tự động ghi nhận kết quả chấm công vào cơ sở dữ liệu.

Dự án phù hợp với các môi trường như doanh nghiệp, trường học, phòng lab, văn phòng, nhà máy hoặc các khu vực cần kiểm soát ra vào.

---

## 2. Mục tiêu dự án

* Xây dựng hệ thống chấm công tự động bằng camera thời gian thực.
* Ứng dụng mô hình nhận diện khuôn mặt hiện đại trong bài toán quản lý nhân sự.
* Sử dụng pretrained model của InsightFace để rút ngắn thời gian triển khai.
* Hỗ trợ thu thập dữ liệu khuôn mặt nhân viên bằng công cụ Auto Capture.
* Tạo embedding khuôn mặt cho từng nhân viên và lưu trữ phục vụ nhận diện.
* Tự động ghi nhận thời gian check-in/check-out.
* Hỗ trợ phát hiện một số trường hợp giả mạo cơ bản.

---

## 3. Tính năng nổi bật

* Nhận diện khuôn mặt theo thời gian thực từ webcam.
* Tự động phát hiện và căn chỉnh khuôn mặt.
* Trích xuất đặc trưng khuôn mặt dưới dạng vector embedding.
* So khớp nhân viên bằng độ tương đồng cosine similarity.
* Công cụ tự động thu thập dữ liệu khuôn mặt nhân viên.
* Kiểm tra chất lượng ảnh đầu vào: ảnh mờ, thiếu sáng, khuôn mặt quá nhỏ.
* Lưu lịch sử chấm công vào cơ sở dữ liệu.
* Hỗ trợ mở rộng cho nhiều nhân viên.
* Có thể tích hợp thêm anti-spoofing để phát hiện ảnh/video giả mạo.
* Cấu trúc mã nguồn rõ ràng, dễ bảo trì và mở rộng.

---

## 4. Kiến trúc kỹ thuật

| Thành phần              | Công nghệ sử dụng        | Vai trò                                                 |
| ----------------------- | ------------------------ | ------------------------------------------------------- |
| Ngôn ngữ lập trình      | Python                   | Xây dựng toàn bộ hệ thống xử lý ảnh, AI và chấm công    |
| Computer Vision         | OpenCV                   | Đọc webcam, xử lý ảnh, hiển thị kết quả realtime        |
| Face Detection          | SCRFD / InsightFace      | Phát hiện khuôn mặt trong khung hình                    |
| Face Recognition        | InsightFace              | Trích xuất đặc trưng khuôn mặt                          |
| Backbone                | iResNet-100              | Mạng học đặc trưng khuôn mặt                            |
| Loss Function           | ArcFace                  | Tối ưu khả năng phân biệt các khuôn mặt khác nhau       |
| Embedding               | 512-dimensional vector   | Biểu diễn đặc trưng khuôn mặt                           |
| Matching                | Cosine Similarity        | So sánh khuôn mặt mới với dữ liệu đã đăng ký            |
| Deep Learning Framework | PyTorch                  | Hỗ trợ huấn luyện và fine-tuning mô hình                |
| Inference Runtime       | ONNX Runtime             | Chạy mô hình pretrained InsightFace                     |
| Database                | SQLite / Supabase        | Lưu thông tin nhân viên và lịch sử chấm công            |
| Backend/API             | FastAPI / Python Service | Xử lý nghiệp vụ hệ thống                                |
| Frontend/Demo UI        | Streamlit / Web UI       | Hiển thị camera, kết quả nhận diện và lịch sử chấm công |

---

## 5. Pipeline xử lý hệ thống

Quy trình hoạt động chính của hệ thống:

```text
Camera / Webcam
        |
        v
Phát hiện khuôn mặt
        |
        v
Căn chỉnh khuôn mặt
        |
        v
Trích xuất embedding
        |
        v
So khớp với dữ liệu nhân viên
        |
        v
Xác thực danh tính
        |
        v
Ghi nhận chấm công
        |
        v
Lưu vào cơ sở dữ liệu
```

---

## 6. Cấu trúc thư mục

```text
face_attendance/
│
├── data/
│   ├── inhouse/
│   │   ├── NV001/
│   │   ├── NV002/
│   │   └── ...
│   │
│   ├── embeddings/
│   │   └── employee_embeddings.pkl
│   │
│   └── attendance.db
│
├── models/
│   ├── insightface/
│   └── anti_spoof/
│
├── src/
│   ├── config.py
│   ├── capture_dataset.py
│   ├── build_embeddings.py
│   ├── recognize_attendance.py
│   ├── anti_spoof.py
│   ├── database.py
│   └── utils.py
│
├── notebooks/
│   ├── data_analysis.ipynb
│   └── model_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   └── results/
│
├── requirements.txt
├── main.py
└── README.md
```

---

## 7. Mô tả các module chính

| File                      | Chức năng                                               |
| ------------------------- | ------------------------------------------------------- |
| `capture_dataset.py`      | Thu thập dữ liệu khuôn mặt nhân viên bằng webcam        |
| `build_embeddings.py`     | Tạo embedding đại diện cho từng nhân viên               |
| `recognize_attendance.py` | Nhận diện nhân viên realtime và ghi nhận chấm công      |
| `anti_spoof.py`           | Kiểm tra giả mạo khuôn mặt                              |
| `database.py`             | Quản lý kết nối và thao tác với cơ sở dữ liệu           |
| `config.py`               | Lưu các tham số cấu hình của hệ thống                   |
| `utils.py`                | Chứa các hàm hỗ trợ xử lý ảnh, tính similarity, logging |

---

## 8. Yêu cầu hệ thống

### 8.1 Phần cứng khuyến nghị

| Thành phần | Cấu hình khuyến nghị                           |
| ---------- | ---------------------------------------------- |
| CPU        | Intel Core i5 trở lên                          |
| RAM        | Tối thiểu 8GB                                  |
| GPU        | NVIDIA GPU hỗ trợ CUDA nếu muốn chạy nhanh hơn |
| Camera     | Webcam HD hoặc camera IP                       |
| Ổ cứng     | Tối thiểu 10GB trống                           |

### 8.2 Phần mềm

* Python 3.10+
* pip hoặc conda
* Git
* CUDA Toolkit nếu sử dụng GPU
* Visual Studio Code hoặc PyCharm

---

## 9. Hướng dẫn cài đặt

### 9.1 Clone project

```bash
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system
```

### 9.2 Tạo môi trường ảo

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 9.3 Cài đặt thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 9.4 Cài đặt ONNX Runtime

Nếu sử dụng CPU:

```bash
pip install onnxruntime
```

Nếu sử dụng GPU NVIDIA:

```bash
pip install onnxruntime-gpu
```

---

## 10. File `requirements.txt` tham khảo

```txt
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
torch>=2.2.0
torchvision>=0.17.0
insightface>=0.7.3
onnxruntime>=1.17.0
scikit-learn>=1.3.0
scipy>=1.11.0
tqdm>=4.66.0
matplotlib>=3.8.0
sqlalchemy>=2.0.0
streamlit>=1.35.0
loguru>=0.7.2
```

Nếu dùng GPU, có thể thay:

```txt
onnxruntime>=1.17.0
```

bằng:

```txt
onnxruntime-gpu>=1.17.0
```

---

## 11. Hướng dẫn chạy hệ thống

### 11.1 Thu thập dữ liệu khuôn mặt nhân viên

```bash
python src/capture_dataset.py --employee_id NV001 --max_images 50 --ctx_id -1
```

Trong đó:

| Tham số         | Ý nghĩa                           |
| --------------- | --------------------------------- |
| `--employee_id` | Mã nhân viên cần thu thập dữ liệu |
| `--max_images`  | Số ảnh cần chụp                   |
| `--ctx_id`      | `-1` để chạy CPU, `0` để chạy GPU |

Dữ liệu sau khi thu thập sẽ được lưu tại:

```text
data/inhouse/NV001/
```

---

### 11.2 Tạo embedding cho nhân viên

```bash
python src/build_embeddings.py
```

Sau khi chạy xong, hệ thống sẽ tạo file:

```text
data/embeddings/employee_embeddings.pkl
```

File này chứa vector đặc trưng đại diện cho từng nhân viên.

---

### 11.3 Chạy nhận diện và chấm công realtime

```bash
python src/recognize_attendance.py
```

Hệ thống sẽ:

* Mở webcam.
* Phát hiện khuôn mặt.
* Trích xuất embedding.
* So khớp với dữ liệu nhân viên.
* Hiển thị kết quả nhận diện.
* Ghi nhận thời gian chấm công vào cơ sở dữ liệu.

---

### 11.4 Chạy giao diện demo nếu sử dụng Streamlit

```bash
streamlit run main.py
```

---

## 12. Cơ sở dữ liệu

Hệ thống sử dụng cơ sở dữ liệu để lưu thông tin nhân viên và lịch sử chấm công.

### 12.1 Bảng `employees`

| Trường        | Kiểu dữ liệu | Mô tả         |
| ------------- | ------------ | ------------- |
| `employee_id` | TEXT         | Mã nhân viên  |
| `full_name`   | TEXT         | Họ và tên     |
| `department`  | TEXT         | Phòng ban     |
| `created_at`  | DATETIME     | Thời gian tạo |

### 12.2 Bảng `attendance_logs`

| Trường        | Kiểu dữ liệu | Mô tả                   |
| ------------- | ------------ | ----------------------- |
| `id`          | INTEGER      | Khóa chính              |
| `employee_id` | TEXT         | Mã nhân viên            |
| `check_time`  | DATETIME     | Thời gian chấm công     |
| `status`      | TEXT         | Trạng thái nhận diện    |
| `similarity`  | FLOAT        | Độ tương đồng khuôn mặt |

---

## 13. Phương pháp nhận diện

Hệ thống sử dụng pretrained model của InsightFace để trích xuất đặc trưng khuôn mặt.

Mỗi khuôn mặt được biểu diễn thành một vector đặc trưng có 512 chiều. Khi người dùng xuất hiện trước camera, hệ thống sẽ tạo vector mới và so sánh với các vector đã lưu trong cơ sở dữ liệu.

Công thức so khớp sử dụng cosine similarity:

```text
similarity = dot(A, B) / (||A|| * ||B||)
```

Nếu giá trị similarity lớn hơn ngưỡng xác định, hệ thống sẽ xác nhận danh tính nhân viên.

---

## 14. Dataset sử dụng

| Dataset          | Mục đích                                       |
| ---------------- | ---------------------------------------------- |
| VGGFace2         | Fine-tuning mô hình nhận diện khuôn mặt        |
| In-house Dataset | Dữ liệu thực tế phục vụ kiểm thử và triển khai |
| RMFRD            | Bổ sung dữ liệu khuôn mặt đeo khẩu trang       |
| CelebA-Spoof     | Huấn luyện hoặc kiểm thử chống giả mạo         |
| LFW              | Đánh giá hiệu năng nhận diện                   |
| AgeDB-30         | Đánh giá khả năng nhận diện theo độ tuổi       |

---

## 15. Đánh giá mô hình

Các độ đo dự kiến sử dụng:

| Độ đo     | Ý nghĩa                                         |
| --------- | ----------------------------------------------- |
| Accuracy  | Tỷ lệ nhận diện đúng                            |
| Precision | Tỷ lệ dự đoán đúng trong các mẫu được nhận diện |
| Recall    | Tỷ lệ nhận diện đúng trên tổng số mẫu thực tế   |
| F1-score  | Trung bình điều hòa giữa Precision và Recall    |
| FAR       | Tỷ lệ chấp nhận nhầm người lạ                   |
| FRR       | Tỷ lệ từ chối nhầm người hợp lệ                 |
| EER       | Điểm cân bằng giữa FAR và FRR                   |

---

## 16. Hướng phát triển

Trong tương lai, hệ thống có thể được mở rộng theo các hướng sau:

* Tích hợp camera IP.
* Triển khai trên nền tảng web.
* Kết nối Supabase hoặc PostgreSQL.
* Thêm dashboard thống kê chấm công.
* Tích hợp anti-spoofing nâng cao.
* Hỗ trợ nhiều chi nhánh.
* Tối ưu mô hình để chạy trên thiết bị nhúng.
* Tích hợp xuất báo cáo Excel/PDF.

---

## 17. Tác giả

**Tên nhóm:** ErrorAtLine1
**Thành viên:** Hoàng Gia Khiêm và cộng sự
**Đơn vị:** Học viện Ngân hàng
**Môn học:** Trí tuệ nhân tạo

---

## 18. License

Dự án được xây dựng nhằm mục đích học tập, nghiên cứu và demo trong phạm vi môn học. Việc sử dụng các mô hình, dataset hoặc thư viện bên thứ ba cần tuân thủ giấy phép tương ứng của từng nguồn.

---

## 19. Ghi chú

Dự án sử dụng pretrained model của InsightFace để rút ngắn thời gian triển khai và tăng độ chính xác ban đầu. Các kết quả thực nghiệm có thể thay đổi tùy thuộc vào chất lượng dữ liệu, điều kiện ánh sáng, thiết bị camera và cấu hình phần cứng sử dụng.
