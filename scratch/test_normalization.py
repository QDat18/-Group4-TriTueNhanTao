import sys
import os

# Thêm thư mục gốc vào sys.path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import argparse
from PIL import Image
from torchvision import transforms

from src.utils.transforms import get_val_transform

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chuẩn hóa ảnh và pixel (Normalization)")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh khuôn mặt (đã crop)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Lỗi: Không tìm thấy file ảnh tại: {args.image}")
        return

    # 1. Đọc ảnh gốc bằng OpenCV (BGR)
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print("Lỗi: Không thể đọc file ảnh.")
        return

    # Chuyển đổi sang RGB và ảnh PIL vì torchvision transforms hoạt động tốt nhất trên PIL Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 2. Thống kê trước khi xử lý
    print("\n" + "="*50)
    print("      THỐNG KÊ ẢNH GỐC (TRƯỚC CHUẨN HÓA)")
    print("="*50)
    print(f"Kích thước ảnh gốc: {pil_img.size[0]}x{pil_img.size[1]} px")
    print(f"Kiểu dữ liệu: {type(pil_img)}")
    img_np = np = torch.from_numpy(img_rgb).float()
    print(f"Miền giá trị pixel RGB gốc: [{img_np.min().item():.1f}, {img_np.max().item():.1f}]")
    print(f"Giá trị trung bình pixel: {img_np.mean().item():.2f}")
    print("-"*50)

    # 3. Thực hiện ToTensor độc lập để xem miền giá trị [0, 1]
    to_tensor_transform = transforms.ToTensor()
    tensor_0_1 = to_tensor_transform(pil_img)
    print("\n" + "="*50)
    print("      SAU KHI CHUYỂN SANG TENSOR (ToTensor)")
    print("="*50)
    print(f"Kích thước Tensor: {list(tensor_0_1.shape)} (Channels x Height x Width)")
    print(f"Miền giá trị pixel: [{tensor_0_1.min().item():.4f}, {tensor_0_1.max().item():.4f}]")
    print(f"Giá trị trung bình: {tensor_0_1.mean().item():.4f}")
    print("-"*50)

    # 4. Áp dụng toàn bộ quy trình chuẩn hóa (Resize -> ToTensor -> Normalize)
    transform = get_val_transform()
    normalized_tensor = transform(pil_img)

    print("\n" + "="*50)
    print("      SAU KHI CHUẨN HÓA (transforms.Normalize)")
    print("="*50)
    print(f"Kích thước Tensor đầu ra: {list(normalized_tensor.shape)}")
    print(f"Miền giá trị pixel đầu ra: [{normalized_tensor.min().item():.4f}, {normalized_tensor.max().item():.4f}]")
    print(f"Giá trị trung bình đầu ra: {normalized_tensor.mean().item():.4f}")
    print(f"Độ lệch chuẩn (Std) đầu ra: {normalized_tensor.std().item():.4f}")
    print("-"*50)

    # Kiểm định xem giá trị có nằm trong khoảng [-1, 1] không
    in_range = (normalized_tensor.min().item() >= -1.05 and normalized_tensor.max().item() <= 1.05)
    print(f"Kiểm chứng miền giá trị [-1, 1]: {'ĐẠT' if in_range else 'KHÔNG ĐẠT'}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
