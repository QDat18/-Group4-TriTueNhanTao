import sys
import os

# Thêm thư mục gốc vào sys.path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import argparse
import numpy as np
from insightface.app import FaceAnalysis

from src.attendance.align_face import align_face
from src.config import DET_SIZE

def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def main():
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng khuôn mặt của một bức ảnh")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh cần kiểm tra chất lượng")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Lỗi: Không tìm thấy file ảnh tại đường dẫn: {args.image}")
        return

    # Đọc ảnh đầu vào
    img = cv2.imread(args.image)
    if img is None:
        print("Lỗi: Không thể đọc file ảnh (định dạng không hỗ trợ hoặc file hỏng).")
        return

    print("Đang khởi tạo mô hình phát hiện khuôn mặt...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    print(f"Đang phân tích chất lượng ảnh: {args.image}")
    faces = app.get(img)

    if len(faces) == 0:
        print("\n[THẤT BẠI] Không phát hiện thấy khuôn mặt nào trong khung hình.")
        return
    elif len(faces) > 1:
        print(f"\n[CẢNH BÁO] Phát hiện {len(faces)} khuôn mặt. Hệ thống tiến hành đánh giá khuôn mặt lớn nhất.")
        # Sắp xếp lấy khuôn mặt có diện tích bbox lớn nhất
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)

    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = img.shape[:2]
    
    # Giới hạn tọa độ trong kích thước ảnh
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face_width = x2 - x1
    face_height = y2 - y1

    # Thực hiện căn chỉnh và cắt khuôn mặt
    face_crop = align_face(img, face.kps)

    # Tính toán các chỉ số
    curr_blur = blur_score(face_crop)
    curr_brightness = brightness_score(face_crop)

    # Các ngưỡng quy định trong dự án
    min_blur = 80
    min_brightness = 50
    max_brightness = 220
    min_face_size = 90

    # Đánh giá các tiêu chí
    pass_blur = curr_blur >= min_blur
    pass_brightness = min_brightness <= curr_brightness <= max_brightness
    pass_size = face_width >= min_face_size and face_height >= min_face_size
    all_passed = pass_blur and pass_brightness and pass_size

    # Hiển thị báo cáo kết quả chi tiết
    print("\n" + "="*50)
    print("      BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG KHUÔN MẶT")
    print("="*50)
    print(f"Kích thước ảnh gốc: {w}x{h} px")
    print(f"Kích thước khuôn mặt phát hiện: {face_width}x{face_height} px")
    print(f"Độ sắc nét (Blur Score): {curr_blur:.2f}")
    print(f"Độ sáng (Brightness Score): {curr_brightness:.2f}")
    print("-"*50)
    
    # Trạng thái từng tiêu chí
    status_str = lambda flag: "ĐẠT" if flag else "KHÔNG ĐẠT"
    
    print(f"1. Kích thước tối thiểu (>= {min_face_size}x{min_face_size} px): {status_str(pass_size)}")
    print(f"2. Độ sắc nét (Laplacian Variance >= {min_blur}):   {status_str(pass_blur)}")
    print(f"3. Độ sáng an toàn ({min_brightness} <= light <= {max_brightness}): {status_str(pass_brightness)}")
    print("-"*50)
    
    # Kết luận chung
    if all_passed:
        print("KẾT LUẬN: [HỢP LỆ] Ảnh đủ tiêu chuẩn để lưu trữ / nhận diện.")
        # Lưu ảnh đã crop phục vụ kiểm tra
        os.makedirs("scratch", exist_ok=True)
        crop_path = "scratch/test_crop.jpg"
        cv2.imwrite(crop_path, face_crop)
        print(f"-> Đã lưu ảnh khuôn mặt sau xử lý tại: {crop_path}")
    else:
        reasons = []
        if not pass_size:
            reasons.append("Khuôn mặt quá nhỏ (cần đứng gần camera hơn)")
        if not pass_blur:
            reasons.append("Ảnh bị mờ/nhòe (giữ yên khi chụp)")
        if not pass_brightness:
            if curr_brightness < min_brightness:
                reasons.append("Ảnh quá tối (thiếu sáng/ngược sáng)")
            else:
                reasons.append("Ảnh quá sáng (cháy sáng/lóa đèn)")
        print(f"KẾT LUẬN: [KHÔNG HỢP LỆ] Lý do: {', '.join(reasons)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
