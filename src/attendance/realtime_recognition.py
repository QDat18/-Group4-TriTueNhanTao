import os
import cv2
import time
import torch
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from src.config import DET_SIZE, RECOGNITION_THRESHOLD
from src.models.face_recognition_model import FaceRecognitionModel
from src.attendance.attendance_service import AttendanceService
from src.utils.transforms import get_val_transform
from src.anti_spoofing.anti_spoofing import LivenessDetector
from src.attendance.align_face import align_face

class RealtimeRecognition:
    """
    Hệ thống nhận diện chấm công thời gian thực kết hợp kiểm tra chống giả mạo (Anti-Spoofing).
    """
    def __init__(self):
        print("Initializing Realtime Recognition System...")
        
        # 1. Khởi tạo detector khuôn mặt từ InsightFace (Buffalo_L)
        self.detector = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.detector.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
        
        # 2. Khởi tạo custom model nhận diện (ResNet50 + ArcFace)
        self.recognition_model = FaceRecognitionModel()
        
        # 3. Khởi tạo dịch vụ chấm công (Supabase Integration + Cooldown)
        self.attendance_service = AttendanceService()
        
        # 4. Khởi tạo module chống giả mạo
        self.liveness_detector = LivenessDetector()
        
        # 5. Khởi tạo transform chuẩn hóa ảnh đầu vào cho model nhận diện
        self.transform = get_val_transform()

    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("\n============================================================")
        print("  HỆ THỐNG CHẤM CÔNG THỜI GIAN THỰC (Bấm ESC hoặc 'q' để thoát)")
        print("============================================================\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            display = frame.copy()
            
            # 1. Phát hiện các khuôn mặt trong khung hình
            faces = self.detector.get(frame)
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # 2. Kiểm tra chống giả mạo (Liveness Checking)
                liveness_res = self.liveness_detector.check_liveness(face_crop)
                is_live = liveness_res["is_live"]
                liveness_score = liveness_res["score"]
                liveness_reason = liveness_res["reason"]

                if not is_live:
                    # Giả mạo phát hiện
                    color = (0, 0, 255) # Đỏ
                    label = f"SPOOF ({liveness_reason})"
                else:
                    # 3. Trích xuất vector đặc trưng (Embedding) sử dụng Custom ArcFace Model
                    try:
                        aligned_face = align_face(frame, face.kps)
                        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(face_rgb)
                        img_tensor = self.transform(pil_img).unsqueeze(0)
                        
                        emb = self.recognition_model.get_embedding(img_tensor).squeeze(0).numpy()
                        
                        # 4. Tìm kiếm nhân viên khớp nhất
                        match = self.attendance_service.find_best_match(emb)
                        
                        if match:
                            emp_id = match["employee_id"]
                            full_name = match["full_name"]
                            similarity = match["similarity"]
                            
                            # 5. Lưu chấm công (nếu đã qua cooldown)
                            logged = self.attendance_service.save_attendance(emp_id, similarity)
                            if logged:
                                color = (0, 255, 0) # Xanh lá - Chấm công thành công
                                label = f"{full_name} ({similarity:.2f}) - SUCCESS"
                            else:
                                color = (255, 255, 0) # Vàng/Xanh lam - Đã chấm trước đó (Cooldown)
                                label = f"{full_name} ({similarity:.2f}) - COOLDOWN"
                        else:
                            color = (0, 165, 255) # Cam - Không xác định
                            label = "Unknown"
                    except Exception as e:
                        print(f"Error during recognition loop: {e}")
                        color = (0, 0, 255)
                        label = "Error"

                # Vẽ bounding box và nhãn
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display, f"Liveness: {liveness_score:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.imshow("Realtime Face Attendance & Anti-Spoofing", display)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealtimeRecognition()
    # Mặc định sử dụng webcam chính (0)
    app.run(camera_id=0)
