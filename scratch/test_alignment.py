import sys
import os

# Thêm thư mục gốc vào sys.path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import argparse
from insightface.app import FaceAnalysis
from src.attendance.align_face import align_face
from src.config import DET_SIZE

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra căn chỉnh khuôn mặt (Face Alignment)")
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

    print("Đang chạy mô hình phát hiện khuôn mặt và điểm mốc...")
    faces = app.get(img)

    if len(faces) == 0:
        print("\n[THẤT BẠI] Không tìm thấy khuôn mặt nào để căn chỉnh.")
        return

    face = faces[0]
    
    # 1. Cắt thô không căn chỉnh (chỉ resize về 112x112)
    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    raw_crop = img[y1:y2, x1:x2]
    raw_resized = cv2.resize(raw_crop, (112, 112))
    
    # 2. Căn chỉnh bằng thuật toán Affine Transform (align_face)
    print("Đang thực hiện căn chỉnh khuôn mặt bằng phép biến đổi Affine...")
    aligned_img = align_face(img, face.kps)

    # Lưu cả 2 ảnh để so sánh trước sau
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite("scratch/truoc_can_chinh.jpg", raw_resized)
    cv2.imwrite("scratch/sau_can_chinh.jpg", aligned_img)

    print("\n" + "="*50)
    print("        KẾT QUẢ CĂN CHỈNH KHUÔN MẶT")
    print("="*50)
    print("Đã lưu 2 ảnh so sánh tại thư mục 'scratch/':")
    print("1. [Chưa căn chỉnh] (Chỉ resize): scratch/truoc_can_chinh.jpg")
    print("2. [Đã căn chỉnh]   (Affine):     scratch/sau_can_chinh.jpg")
    print("-"*50)
    print("Chi tiết điểm mốc phát hiện được (5 Keypoints):")
    kps_labels = ["Mắt trái", "Mắt phải", "Mũi     ", "Mũi trái ", "Mũi phải "]
    # Lấy nhãn chuẩn của 5 điểm mốc
    labels = ["Mắt trái", "Mắt phải", "Đỉnh mũi", "Khóe miệng trái", "Khóe miệng phải"]
    for idx, kp in enumerate(face.kps):
        print(f"  - {labels[idx]:<18}: ({kp[0]:.2f}, {kp[1]:.2f})")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
