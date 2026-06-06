import cv2
import numpy as np

def align_face(image, kps):
    """
    Căn chỉnh khuôn mặt (Face Alignment) về chuẩn kích thước 112x112 dựa trên 5 điểm mốc (keypoints).
    Phương pháp này sử dụng phép biến đổi Affine (Similarity Transform) để căn thẳng hai mắt, mũi và khóe miệng.
    
    Tham số:
        image: Ảnh gốc (BGR OpenCV)
        kps: Mảng 5 điểm mốc [[x_le, y_le], [x_re, y_re], [x_n, y_n], [x_lm, y_lm], [x_rm, y_rm]]
    Trả về:
        Ảnh khuôn mặt đã căn chỉnh và resize về kích thước 112x112.
    """
    if kps is None or len(kps) != 5:
        # Fallback về resize trực tiếp nếu không có keypoints
        return cv2.resize(image, (112, 112))

    # Tọa độ mốc chuẩn (reference landmarks) cho ảnh 112x112 của ArcFace
    reference_kps = np.array([
        [30.2946, 51.6963],  # Mắt trái
        [65.5318, 51.5014],  # Mắt phải
        [48.0252, 71.7366],  # Mũi
        [33.5493, 92.3655],  # Khóe miệng trái
        [62.7299, 92.2041]   # Khóe miệng phải
    ], dtype=np.float32)

    src = np.array(kps, dtype=np.float32)

    # Ước lượng ma trận biến đổi Affine (similarity transform)
    matrix, _ = cv2.estimateAffinePartial2D(src, reference_kps)

    if matrix is None:
        return cv2.resize(image, (112, 112))

    # Áp dụng ma trận biến đổi để căn thẳng khuôn mặt về kích thước 112x112
    aligned = cv2.warpAffine(image, matrix, (112, 112))
    return aligned
