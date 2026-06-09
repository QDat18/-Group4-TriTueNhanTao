import sys
import os

# Thêm thư mục gốc của dự án vào sys.path để python tìm được thư mục 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
import argparse
import numpy as np
from insightface.app import FaceAnalysis

# Import hàm align_face từ project
from src.attendance.align_face import align_face
from src.config import DET_SIZE

def main():
    parser = argparse.ArgumentParser(description="Tạo ảnh minh họa Trước và Sau tiền xử lý")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn đến ảnh đầu vào (nếu không có sẽ dùng webcam)")
    args = parser.parse_args()

    # Khởi tạo bộ phát hiện khuôn mặt buffalo_l
    print("Đang khởi tạo mô hình phát hiện khuôn mặt...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    frame = None

    if args.image is not None:
        if not os.path.exists(args.image):
            print(f"Lỗi: Không tìm thấy file ảnh {args.image}")
            return
        frame = cv2.imread(args.image)
        print(f"Đã đọc ảnh từ: {args.image}")
    else:
        # Sử dụng Webcam để chụp
        print("\n=== SỬ DỤNG WEBCAM ===")
        print("Mở camera, chuẩn bị chụp ảnh trong 3 giây. Hãy nhìn vào camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Lỗi: Không thể mở webcam.")
            return
        
        # Cho camera khởi động và phơi sáng ổn định
        for i in range(3, 0, -1):
            print(f"Chụp sau: {i}...")
            ret, frame = cap.read()
            time.sleep(1)
            
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("Lỗi: Không chụp được ảnh từ camera.")
            return
        print("Đã chụp ảnh thành công!")

    # Tạo bản sao để vẽ ảnh "Trước tiền xử lý"
    before_img = frame.copy()
    
    # Phát hiện khuôn mặt
    faces = app.get(frame)
    if len(faces) == 0:
        print("Lỗi: Không tìm thấy khuôn mặt nào trong ảnh để xử lý.")
        # Lưu ảnh gốc để đối chiếu
        cv2.imwrite("scratch/truoc_tien_xu_ly.jpg", before_img)
        print("Đã lưu ảnh gốc vào scratch/truoc_tien_xu_ly.jpg")
        return

    # Lấy khuôn mặt đầu tiên phát hiện được
    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    kps = face.kps.astype(int)

    # 1. Vẽ bounding box và landmark lên ảnh "Trước xử lý"
    # Vẽ hộp giới hạn màu đỏ
    cv2.rectangle(before_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Vẽ 5 điểm mốc (landmarks) màu xanh lá
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255)] # Mỗi điểm một màu
    for i, kp in enumerate(kps):
        cv2.circle(before_img, tuple(kp), 5, colors[i % len(colors)], -1)
        
    # Ghi text giải thích
    cv2.putText(before_img, "Raw Frame + Bounding Box + 5 Landmarks", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 2. Thực hiện căn chỉnh và cắt khuôn mặt (Sau xử lý)
    after_img = align_face(frame, face.kps)

    # Đảm bảo thư mục scratch tồn tại
    os.makedirs("scratch", exist_ok=True)

    # Lưu 2 ảnh kết quả
    cv2.imwrite("scratch/truoc_tien_xu_ly.jpg", before_img)
    cv2.imwrite("scratch/sau_tien_xu_ly.jpg", after_img)

    print("\n" + "="*50)
    print("Đã tạo thành công các file ảnh minh họa trong thư mục 'scratch/':")
    print("1. [Trước tiền xử lý]: scratch/truoc_tien_xu_ly.jpg (Ảnh gốc + Khung quét + Điểm mốc)")
    print("2. [Sau tiền xử lý]:   scratch/sau_tien_xu_ly.jpg   (Ảnh 112x112 đã xoay thẳng và cắt gọn)")
    print("="*50)

if __name__ == "__main__":
    main()
