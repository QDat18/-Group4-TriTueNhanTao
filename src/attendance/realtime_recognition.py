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
import threading

from insightface.app import FaceAnalysis

from src.config import DET_SIZE
from src.models.insightface_model import InsightFaceModel
from src.attendance.attendance_service import AttendanceService
from src.anti_spoofing.anti_spoofing import LivenessDetector


class CameraStream:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.frame_id = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )
        self.thread.start()

    def _update(self):
        try:
            if isinstance(self.camera_id, int):
                self.cap = cv2.VideoCapture(
                    self.camera_id,
                    cv2.CAP_DSHOW
                )
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


_shared_camera_streams = {}
_shared_camera_lock = threading.Lock()

def get_shared_camera_stream(camera_id=0):
    global _shared_camera_streams
    with _shared_camera_lock:
        if camera_id not in _shared_camera_streams:
            stream = CameraStream(camera_id)
            stream.start()
            _shared_camera_streams[camera_id] = stream
        return _shared_camera_streams[camera_id]


class InferenceWorker:
    def __init__(self, system, camera_stream):
        self.system = system
        self.camera_stream = camera_stream
        self.running = False
        self.thread = None
        self.draw_data = []
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._work,
            daemon=True
        )
        self.thread.start()

    def _work(self):
        while self.running:
            ret, frame, _ = self.camera_stream.read()

            if not ret:
                time.sleep(0.03)
                continue

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

                color = (0, 165, 255)
                label = "Unknown"
                liveness_score = 0.0
                similarity = 0.0

                liveness_res = (
                    self.system
                    .liveness_detector
                    .check_liveness(face_crop)
                )

                is_live = liveness_res["is_live"]
                liveness_score = liveness_res["score"]
                liveness_reason = liveness_res["reason"]

                if not is_live:
                    color = (0, 0, 255)
                    label = f"SPOOF ({liveness_reason})"

                else:
                    try:
                        emb = (
                            self.system
                            .recognition_model
                            .get_embedding_from_face(face)
                        )

                        if emb is None:
                            color = (0, 165, 255)
                            label = "Unknown"
                        else:
                            match = (
                                self.system
                                .attendance_service
                                .find_best_match(emb)
                            )

                            if match:
                                emp_id = match["employee_id"]
                                full_name = match["full_name"]
                                similarity = match["similarity"]

                                res = (
                                    self.system
                                    .attendance_service
                                    .save_attendance(
                                        emp_id,
                                        similarity
                                    )
                                )

                                if res["success"]:
                                    status = res["status"]
                                    if status == "SUCCESS":
                                        color = (0, 255, 0)
                                        label = f"{full_name} - SUCCESS"
                                    elif status == "LATE":
                                        color = (0, 165, 255)
                                        label = f"{full_name} - LATE (+{res['late_minutes']}m)"
                                    elif status == "CHECK_OUT":
                                        color = (255, 0, 0)
                                        label = f"{full_name} - CHECK_OUT"
                                    elif status == "EARLY_LEAVE":
                                        color = (0, 165, 255)
                                        label = f"{full_name} - EARLY_LEAVE (-{res['early_minutes']}m)"
                                    elif status == "REJECTED_CHECK_IN":
                                        color = (0, 0, 255)
                                        label = f"{full_name} - REJECTED_CHECK_IN"
                                    elif status == "REJECTED_CHECK_OUT":
                                        color = (0, 0, 255)
                                        label = f"{full_name} - REJECTED_CHECK_OUT"
                                    elif status == "COMPLETED":
                                        color = (0, 255, 0)
                                        label = f"{full_name} - COMPLETED"
                                    else:
                                        color = (0, 255, 0)
                                        label = f"{full_name} - {status}"
                                else:
                                    color = (0, 255, 255)
                                    label = (
                                        f"{full_name} - "
                                        f"COOLDOWN ({res['remaining']}s)"
                                    )
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
                    "liveness_score": liveness_score,
                    "similarity": similarity
                })

            with self.lock:
                self.draw_data = new_draw_data

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
    def __init__(self):
        print("Initializing Realtime Recognition System...")

        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        self.detector = FaceAnalysis(
            name="buffalo_l",
            providers=providers
        )

        ctx_id = 0 if torch.cuda.is_available() else -1

        self.detector.prepare(
            ctx_id=ctx_id,
            det_size=DET_SIZE
        )

        self.recognition_model = InsightFaceModel()

        self.attendance_service = AttendanceService()

        self.liveness_detector = LivenessDetector()

        self.active_streams = {}
        self.stream_lock = threading.Lock()

    def _delayed_stop_stream(self, camera_id, delay=5.0):
        time.sleep(delay)

        with self.stream_lock:
            if camera_id in self.active_streams:
                stream_info = self.active_streams[camera_id]

                if stream_info["ref_count"] <= 0:
                    print(
                        f"[CAMERA] Stopping idle stream for camera "
                        f"{camera_id}..."
                    )

                    stream_info["worker"].stop()
                    stream_info["stream"].stop()

                    self.active_streams.pop(camera_id, None)

    def run(self, camera_id=0):
        cap = None

        try:
            if isinstance(camera_id, int):
                cap = cv2.VideoCapture(
                    camera_id,
                    cv2.CAP_DSHOW
                )
        except Exception as e:
            print(f"Warning: Failed to open camera via CAP_DSHOW: {e}")
            cap = None

        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print()
        print("=" * 60)
        print("HỆ THỐNG CHẤM CÔNG THỜI GIAN THỰC")
        print("Bấm ESC hoặc q để thoát")
        print("=" * 60)
        print()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame.")
                break

            display = frame.copy()

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

                liveness_res = (
                    self.liveness_detector
                    .check_liveness(face_crop)
                )

                is_live = liveness_res["is_live"]
                liveness_score = liveness_res["score"]
                liveness_reason = liveness_res["reason"]

                if not is_live:
                    color = (0, 0, 255)
                    label = f"SPOOF ({liveness_reason})"

                else:
                    try:
                        emb = (
                            self.recognition_model
                            .get_embedding_from_face(face)
                        )

                        if emb is None:
                            color = (0, 165, 255)
                            label = "Unknown"
                        else:
                            match = (
                                self.attendance_service
                                .find_best_match(emb)
                            )

                            if match:
                                emp_id = match["employee_id"]
                                full_name = match["full_name"]
                                similarity = match["similarity"]

                                res = (
                                    self.attendance_service
                                    .save_attendance(
                                        emp_id,
                                        similarity
                                    )
                                )

                                if res["success"]:
                                    status = res["status"]
                                    if status == "SUCCESS":
                                        color = (0, 255, 0)
                                        label = f"{full_name} - SUCCESS"
                                    elif status == "LATE":
                                        color = (0, 165, 255)
                                        label = f"{full_name} - LATE (+{res['late_minutes']}m)"
                                    elif status == "CHECK_OUT":
                                        color = (255, 0, 0)
                                        label = f"{full_name} - CHECK_OUT"
                                    elif status == "EARLY_LEAVE":
                                        color = (0, 165, 255)
                                        label = f"{full_name} - EARLY_LEAVE (-{res['early_minutes']}m)"
                                    elif status == "REJECTED_CHECK_IN":
                                        color = (0, 0, 255)
                                        label = f"{full_name} - REJECTED_CHECK_IN"
                                    elif status == "REJECTED_CHECK_OUT":
                                        color = (0, 0, 255)
                                        label = f"{full_name} - REJECTED_CHECK_OUT"
                                    elif status == "COMPLETED":
                                        color = (0, 255, 0)
                                        label = f"{full_name} - COMPLETED"
                                    else:
                                        color = (0, 255, 0)
                                        label = f"{full_name} - {status}"
                                else:
                                    color = (0, 255, 255)
                                    label = (
                                        f"{full_name} - "
                                        f"COOLDOWN ({res['remaining']}s)"
                                    )
                            else:
                                color = (0, 165, 255)
                                label = "Unknown"

                    except Exception as e:
                        print(f"Error during recognition loop: {e}")
                        color = (0, 0, 255)
                        label = "Error"

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    display,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                cv2.putText(
                    display,
                    f"Liveness: {liveness_score:.2f}",
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )

            cv2.imshow(
                "Realtime Face Attendance & Anti-Spoofing",
                display
            )

            key = cv2.waitKey(1)

            if key == 27 or key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_gen(self, camera_id=0):
        if isinstance(camera_id, str):
            if camera_id.isdigit():
                camera_id = int(camera_id)
            else:
                import re

                if not re.match(
                    r"^(http|https|rtsp|rtmp)://",
                    camera_id,
                    re.IGNORECASE
                ):
                    camera_id = "http://" + camera_id

        with self.stream_lock:
            if camera_id not in self.active_streams:
                print(f"[CAMERA] Opening camera {camera_id}...")

                stream = get_shared_camera_stream(camera_id)

                init_start = time.time()
                success = False

                while time.time() - init_start < 1.5:
                    ret, _, _ = stream.read()

                    if ret:
                        success = True
                        break

                    time.sleep(0.05)

                if not success:
                    print(
                        f"Error: Could not open camera {camera_id}. "
                        f"Yielding placeholder offline frame."
                    )

                    placeholder = np.zeros(
                        (480, 640, 3),
                        dtype=np.uint8
                    )

                    cv2.putText(
                        placeholder,
                        "CAMERA OFFLINE / IN USE",
                        (130, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2
                    )

                    ret, jpeg = cv2.imencode(
                        ".jpg",
                        placeholder
                    )

                    if ret:
                        yield jpeg.tobytes()

                    return

                worker = InferenceWorker(
                    self,
                    stream
                )

                self.active_streams[camera_id] = {
                    "stream": stream,
                    "worker": worker,
                    "ref_count": 0
                }

            stream_info = self.active_streams[camera_id]
            stream_info["ref_count"] += 1

            stream = stream_info["stream"]
            worker = stream_info["worker"]

            if not worker.running:
                worker.start()

        last_frame_id = -1

        try:
            while True:
                ret, frame, fid = stream.read()

                if not ret:
                    time.sleep(0.03)
                    continue

                if fid == last_frame_id:
                    time.sleep(0.005)
                    continue

                last_frame_id = fid

                display = frame.copy()
                draw_data = worker.get_draw_data()

                for item in draw_data:
                    bx1, by1, bx2, by2 = item["bbox"]

                    cv2.rectangle(
                        display,
                        (bx1, by1),
                        (bx2, by2),
                        item["color"],
                        2
                    )

                    cv2.putText(
                        display,
                        item["label"],
                        (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        item["color"],
                        2
                    )

                    cv2.putText(
                        display,
                        f"Liveness: {item['liveness_score']:.2f}",
                        (bx1, by2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        item["color"],
                        1
                    )

                ret, jpeg = cv2.imencode(
                    ".jpg",
                    display
                )

                if not ret:
                    continue

                yield jpeg.tobytes()

        finally:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    stream_info = self.active_streams[camera_id]
                    stream_info["ref_count"] -= 1

                    if stream_info["ref_count"] <= 0:
                        print(f"[CAMERA] Releasing camera {camera_id} because ref_count reached 0.")
                        stream_info["worker"].stop()
                        stream_info["stream"].stop()
                        
                        global _shared_camera_streams
                        with _shared_camera_lock:
                            _shared_camera_streams.pop(camera_id, None)
                            
                        self.active_streams.pop(camera_id, None)


if __name__ == "__main__":
    app = RealtimeRecognition()
    app.run(camera_id=0)
