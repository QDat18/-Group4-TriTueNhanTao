import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import cv2
import time
import torch
torch.set_num_threads(2)
import numpy as np
from PIL import Image
import threading

from insightface.app import FaceAnalysis
from src.config import DET_SIZE, RECOGNITION_THRESHOLD
from src.models.face_recognition_model import FaceRecognitionModel
from src.attendance.attendance_service import AttendanceService
from src.utils.transforms import get_val_transform
from src.anti_spoofing.anti_spoofing import LivenessDetector
from src.attendance.align_face import align_face


class CameraStream:
    """
    Background thread to continuously grab frames from the camera.
    This prevents OpenCV buffer accumulation and eliminates camera lag.
    """
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.frame_id = 0  # Track unique frame ID

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        # Use DirectShow on Windows for local webcams to prevent MSMF grab frame errors
        try:
            if isinstance(self.camera_id, int):
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        except Exception as e:
            print(f"Warning: Failed to open camera via CAP_DSHOW: {e}")
            self.cap = None

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            self.ret = False
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.ret = True
                    self.frame_id += 1
            else:
                time.sleep(0.01)

        if self.cap:
            self.cap.release()

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy(), self.frame_id
            return False, None, 0

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


class InferenceWorker:
    """
    Background worker to run face detection and recognition asynchronously.
    This prevents PyTorch and InsightFace model inference from blocking the streaming FPS.
    """
    def __init__(self, system, camera_stream):
        self.system = system
        self.camera_stream = camera_stream
        self.running = False
        self.thread = None
        self.draw_data = []
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        while self.running:
            ret, frame, _ = self.camera_stream.read()
            if not ret:
                time.sleep(0.03)
                continue

            # Run detection
            faces = self.system.detector.get(frame)
            new_draw_data = []

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

                color = (0, 165, 255)  # Orange (Unknown)
                label = "Unknown"
                liveness_score = 0.0

                # Check liveness
                liveness_res = self.system.liveness_detector.check_liveness(face_crop)
                is_live = liveness_res["is_live"]
                liveness_score = liveness_res["score"]
                liveness_reason = liveness_res["reason"]

                if not is_live:
                    color = (0, 0, 255)  # Red (Spoof)
                    label = f"SPOOF ({liveness_reason})"
                else:
                    try:
                        aligned_face = align_face(frame, face.kps)
                        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(face_rgb)
                        img_tensor = self.system.transform(pil_img).unsqueeze(0)

                        emb = self.system.recognition_model.get_embedding(img_tensor).squeeze(0).numpy()
                        match = self.system.attendance_service.find_best_match(emb)

                        if match:
                            emp_id = match["employee_id"]
                            full_name = match["full_name"]
                            similarity = match["similarity"]

                            res = self.system.attendance_service.save_attendance(emp_id, similarity)
                            if res["success"]:
                                if res["status"] == "LATE":
                                    color = (0, 165, 255)
                                    label = f"{full_name} - LATE (+{res['late_minutes']}m)"
                                else:
                                    color = (0, 255, 0)
                                    label = f"{full_name} - SUCCESS"
                            else:
                                color = (0, 255, 255)
                                label = f"{full_name} - COOLDOWN ({res['remaining']}s)"
                        else:
                            color = (0, 165, 255)
                            label = "Unknown"
                    except Exception as e:
                        print(f"Error during recognition worker: {e}")
                        color = (0, 0, 255)
                        label = "Error"

                new_draw_data.append({
                    "bbox": (x1, y1, x2, y2),
                    "color": color,
                    "label": label,
                    "liveness_score": liveness_score
                })

            with self.lock:
                self.draw_data = new_draw_data

            # Dynamic sleep to optimize CPU usage:
            # If no faces are detected, sleep longer (150ms) to save CPU.
            # If faces are detected, sleep shorter (80ms) for high responsiveness.
            if not faces:
                time.sleep(0.15)
            else:
                time.sleep(0.08)

    def get_draw_data(self):
        with self.lock:
            return list(self.draw_data)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


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

        # Shared camera streams to avoid reopening hardware camera on refresh or tab switch
        self.active_streams = {}
        self.stream_lock = threading.Lock()

    def _delayed_stop_stream(self, camera_id, delay=5.0):
        time.sleep(delay)
        with self.stream_lock:
            if camera_id in self.active_streams:
                stream_info = self.active_streams[camera_id]
                if stream_info["ref_count"] <= 0:
                    print(f"[CAMERA] Stopping idle stream for camera {camera_id}...")
                    stream_info["worker"].stop()
                    stream_info["stream"].stop()
                    self.active_streams.pop(camera_id, None)

    def run(self, camera_id=0):
        cap = None
        try:
            if isinstance(camera_id, int):
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        except Exception as e:
            print(f"Warning: Failed to open camera via CAP_DSHOW: {e}")
            cap = None
            
        if cap is None or not cap.isOpened():
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
                            res = self.attendance_service.save_attendance(emp_id, similarity)
                            if res["success"]:
                                if res["status"] == "LATE":
                                    color = (0, 165, 255) # Màu cam - Đi muộn
                                    label = f"{full_name} - LATE (+{res['late_minutes']}m)"
                                else:
                                    color = (0, 255, 0) # Xanh lá - Đi đúng giờ
                                    label = f"{full_name} - SUCCESS"
                            else:
                                color = (0, 255, 255) # Màu vàng/neon - Đã chấm trước đó (Cooldown)
                                label = f"{full_name} - COOLDOWN ({res['remaining']}s)"
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

    def run_gen(self, camera_id=0):
        # Convert string to int if it represents a digit
        if isinstance(camera_id, str):
            if camera_id.isdigit():
                camera_id = int(camera_id)
            else:
                import re
                # Prepend http:// if no protocol prefix is present
                if not re.match(r'^(http|https|rtsp|rtmp)://', camera_id, re.IGNORECASE):
                    camera_id = "http://" + camera_id

        # Get or initialize the shared stream & worker
        with self.stream_lock:
            if camera_id not in self.active_streams:
                print(f"[CAMERA] Opening camera {camera_id}...")
                stream = CameraStream(camera_id)
                stream.start()

                # Give it up to 1.5s to capture the first frame
                init_start = time.time()
                success = False
                while time.time() - init_start < 1.5:
                    ret, _, _ = stream.read()
                    if ret:
                        success = True
                        break
                    time.sleep(0.05)

                if not success:
                    print(f"Error: Could not open camera {camera_id}. Yielding placeholder offline frame.")
                    stream.stop()
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "CAMERA OFFLINE / IN USE", (130, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    ret, jpeg = cv2.imencode('.jpg', placeholder)
                    if ret:
                        yield jpeg.tobytes()
                    return

                worker = InferenceWorker(self, stream)
                worker.start()

                self.active_streams[camera_id] = {
                    "stream": stream,
                    "worker": worker,
                    "ref_count": 0
                }

            stream_info = self.active_streams[camera_id]
            stream_info["ref_count"] += 1
            stream = stream_info["stream"]
            worker = stream_info["worker"]

        last_frame_id = -1
        try:
            while True:
                ret, frame, fid = stream.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                if fid == last_frame_id:
                    # Sleep a tiny bit to not hog CPU while waiting for a new frame
                    time.sleep(0.005)
                    continue

                last_frame_id = fid
                display = frame.copy()
                draw_data = worker.get_draw_data()

                # Draw the boxes and text computed asynchronously by the InferenceWorker
                for item in draw_data:
                    bx1, by1, bx2, by2 = item["bbox"]
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), item["color"], 2)
                    cv2.putText(display, item["label"], (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, item["color"], 2)
                    cv2.putText(display, f"Liveness: {item['liveness_score']:.2f}", (bx1, by2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, item["color"], 1)

                ret, jpeg = cv2.imencode('.jpg', display)
                if not ret:
                    continue
                yield jpeg.tobytes()
        finally:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    stream_info = self.active_streams[camera_id]
                    stream_info["ref_count"] -= 1
                    if stream_info["ref_count"] <= 0:
                        # Schedule shutdown in 5 seconds to prevent camera closure during page refresh
                        threading.Thread(target=self._delayed_stop_stream, args=(camera_id,), daemon=True).start()

if __name__ == "__main__":
    app = RealtimeRecognition()
    # Mặc định sử dụng webcam chính (0)
    app.run(camera_id=0)
