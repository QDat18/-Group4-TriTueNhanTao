import sys
import os

# Thêm thư mục gốc vào sys.path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import argparse
from insightface.app import FaceAnalysis
from src.config import DET_SIZE

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra phát hiện và cắt khuôn mặt")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh đầu vào")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Lỗi: Không tìm thấy file ảnh tại: {args.image}")
        return

    img = cv2.imread(args.image)
    if img is None:
        print("Lỗi: Không thể đọc file ảnh.")
        return

    print("Đang khởi tạo mô hình phát hiện khuôn mặt...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    print("Đang chạy mô hình phát hiện khuôn mặt SCRFD...")
    faces = app.get(img)

    print("\n" + "="*50)
    print("      KẾT QUẢ PHÁT HIỆN VÀ CẮT KHUÔN MẶT")
    print("="*50)
    print(f"Số lượng khuôn mặt phát hiện được: {len(faces)}")
    
    # Kiểm tra ràng buộc đăng ký (chỉ chấp nhận đúng 1 khuôn mặt)
    is_valid_for_registration = (len(faces) == 1)
    print(f"Hợp lệ để đăng ký nhân viên (chỉ duy nhất 1 mặt): {'ĐẠT' if is_valid_for_registration else 'KHÔNG ĐẠT'}")
    print("-"*50)

    if len(faces) == 0:
        print("Kết luận: Không tìm thấy khuôn mặt nào để cắt.")
        return

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img.shape[:2]
        
        # In tọa độ trước khi giới hạn (clipping)
        print(f"Khuôn mặt #{idx+1}:")
        print(f"  - Tọa độ BBox gốc: ({x1}, {y1}) -> ({x2}, {y2})")
        
        # Thực hiện clipping tọa độ để tránh lỗi tràn biên
        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(w, x2)
        y2_clip = min(h, y2)
        
        print(f"  - Tọa độ BBox sau giới hạn (Clipping): ({x1_clip}, {y1_clip}) -> ({x2_clip}, {y2_clip})")
        
        # Thực hiện cắt ảnh khuôn mặt
        cropped = img[y1_clip:y2_clip, x1_clip:x2_clip]
        
        # Lưu kết quả
        os.makedirs("scratch", exist_ok=True)
        out_name = f"scratch/face_{idx+1}_cropped.jpg"
        cv2.imwrite(out_name, cropped)
        print(f"  - Đã cắt và lưu khuôn mặt tại: {out_name}")
        print(f"  - Kích thước ảnh cắt: {cropped.shape[1]}x{cropped.shape[0]} px")
        print("-"*50)
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
