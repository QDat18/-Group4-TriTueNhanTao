# code thu thập dữ liệu nhân viên (In-housse Dataset)

import os
import cv2
import time
import csv
import argparse
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from src.config import DET_SIZE
from src.attendance.align_face import align_face


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_instruction(count):
    instructions = [
        "Nhin thang",
        "Quay mat sang trai",
        "Quay mat sang phai",
        "Cui nhe",
        "Ngua mat nhe",
        "Cuoi nhe",
        "Deo kinh/khau trang neu co"
    ]
    idx = min(count // 7, len(instructions) - 1)
    return instructions[idx]


def write_metadata(csv_path, row):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "employee_id",
                "filename",
                "timestamp",
                "instruction",
                "blur_score",
                "brightness_score",
                "face_width",
                "face_height"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--employee_id", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--save_root", type=str, default="dataset_inhouse")
    parser.add_argument("--ctx_id", type=int, default=-1, help="0 = GPU, -1 = CPU")
    args = parser.parse_args()

    employee_id = args.employee_id
    save_dir = os.path.join(args.save_root, employee_id)
    ensure_dir(save_dir)

    metadata_path = os.path.join(args.save_root, "metadata.csv")

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=args.ctx_id, det_size=DET_SIZE)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Khong mo duoc webcam.")
        return

    count = 0
    last_capture_time = 0

    min_blur = 80
    min_brightness = 50
    max_brightness = 220
    min_face_size = 90

    print("Nhan Q de thoat.")
    print("Nhan S de capture thu cong.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Khong doc duoc frame tu webcam.")
            break

        display = frame.copy()
        faces = app.get(frame)

        instruction = get_instruction(count)

        status_text = "Dang cho khuon mat hop le..."
        can_capture = False
        selected_face = None
        face_crop = None
        current_blur = 0
        current_brightness = 0
        face_width = 0
        face_height = 0

        if len(faces) == 1:
            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)

            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_width = x2 - x1
            face_height = y2 - y1

            if face_width > 0 and face_height > 0:
                face_crop = align_face(frame, face.kps)
                current_blur = blur_score(face_crop)
                current_brightness = brightness_score(face_crop)

                good_blur = current_blur >= min_blur
                good_light = min_brightness <= current_brightness <= max_brightness
                good_size = face_width >= min_face_size and face_height >= min_face_size

                can_capture = good_blur and good_light and good_size

                color = (0, 255, 0) if can_capture else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                if not good_size:
                    status_text = "Mat qua nho - hay tien gan camera"
                elif not good_blur:
                    status_text = "Anh bi mo - hay giu yen"
                elif not good_light:
                    status_text = "Anh qua toi/sang - dieu chinh anh sang"
                else:
                    status_text = "Hop le - dang capture tu dong"

                selected_face = face

        elif len(faces) > 1:
            status_text = "Chi de 1 khuon mat trong khung hinh"
        else:
            status_text = "Khong phat hien khuon mat"

        cv2.putText(display, f"Employee: {employee_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.putText(display, f"Instruction: {instruction}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.putText(display, f"Captured: {count}/{args.max_images}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(display, f"Status: {status_text}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(display, f"Blur: {current_blur:.1f} | Brightness: {current_brightness:.1f}",
                    (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Auto Capture Face Dataset", display)

        key = cv2.waitKey(1) & 0xFF

        now = time.time()
        auto_capture = can_capture and (now - last_capture_time >= args.interval)
        manual_capture = key == ord("s") and can_capture

        if auto_capture or manual_capture:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{employee_id}_{count:03d}_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            cv2.imwrite(filepath, face_crop)

            write_metadata(metadata_path, {
                "employee_id": employee_id,
                "filename": filepath,
                "timestamp": datetime.now().isoformat(),
"instruction": instruction,
                "blur_score": round(current_blur, 2),
                "brightness_score": round(current_brightness, 2),
                "face_width": face_width,
                "face_height": face_height
            })

            print(f"Saved: {filepath}")
            count += 1
            last_capture_time = now

        if key == ord("q"):
            break

        if count >= args.max_images:
            print("Da chup du so anh.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()