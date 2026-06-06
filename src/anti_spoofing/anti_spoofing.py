import cv2
import numpy as np

class LivenessDetector:
    """
    Module kiểm tra chống giả mạo khuôn mặt (Anti-Spoofing / Liveness Detection)
    Giúp phát hiện việc sử dụng ảnh in hoặc màn hình điện thoại để chấm công giả mạo.
    """
    def __init__(self):
        pass

    def check_liveness(self, face_image):
        """
        Kiểm tra độ liveness của khuôn mặt dựa trên phân tích texture và độ mờ (blur).
        """
        if face_image is None or face_image.size == 0:
            return {
                "is_live": False,
                "score": 0.0,
                "reason": "No face image"
            }

        # 1. Phân tích độ sắc nét (Laplacian variance)
        # Ảnh chụp lại từ màn hình hoặc ảnh in thường có độ mờ/nhiễu cao hơn ảnh thực tế.
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Phân tích phân bổ kênh màu V (HSV) để nhận dạng độ chói sáng phản xạ từ màn hình
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        _, max_val, _, _ = cv2.minMaxLoc(v_channel)

        # Tính toán phân bố histogram để phát hiện bất thường ánh sáng
        hist_v = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
        peak_ratio = hist_v.max() / hist_v.sum()

        # Quy đổi thành score liveness (0.0 -> 1.0)
        # Giá trị blur_score lý tưởng của một khuôn mặt thật trước camera HD là > 100.
        # Nếu peak_ratio quá cao nghĩa là có phản xạ sáng mạnh (đặc trưng của màn hình LCD).
        
        is_live = True
        reason = "Authentic Face"
        score = 0.90

        if blur_score < 50.0:
            is_live = False
            reason = "Low quality/Blur (Possible paper print)"
            score = 0.35
        elif peak_ratio > 0.08:
            is_live = False
            reason = "Reflection detected (Possible screen replay)"
            score = 0.20
        elif max_val > 250 and blur_score < 80.0:
            is_live = False
            reason = "Overexposure & low contrast"
            score = 0.40

        return {
            "is_live": is_live,
            "score": float(score),
            "reason": reason,
            "blur": float(blur_score),
            "peak_ratio": float(peak_ratio)
        }
