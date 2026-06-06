import os
import cv2
import time
import argparse
import numpy as np

from datetime import datetime

from insightface.app import FaceAnalysis

from src.config import DET_SIZE

from src.database.supabase_client import supabase
from src.attendance.align_face import align_face
from src.attendance.build_embeddings import (
    EmbeddingBuilder
)

def create_employee(
    employee_id,
    full_name,
    department,
    position
):

    payload = {
        "employee_id": employee_id,
        "full_name": full_name,
        "department": department,
        "position": position
    }

    (
        supabase
        .table("employees")
        .upsert(payload)
        .execute()
    )

    print(
        f"[OK] Employee created: "
        f"{employee_id}"
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def blur_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def brightness_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return np.mean(gray)


def capture_dataset(
    employee_id,
    max_images=100,
    interval=1.0
):

    save_dir = os.path.join(
        "dataset",
        "in-house",
        employee_id
    )

    ensure_dir(save_dir)

    app = FaceAnalysis(
        name="buffalo_l"
    )

    app.prepare(
        ctx_id=0,
        det_size=DET_SIZE
    )

    cap = cv2.VideoCapture(0)

    count = 0
    last_capture = 0

    print()
    print("=" * 60)
    print("AUTO CAPTURE STARTED")
    print("=" * 60)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        faces = app.get(frame)

        can_capture = False
        face_crop = None

        if len(faces) == 1:

            face = faces[0]

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            face_crop = align_face(
                frame,
                face.kps
            )

            blur = blur_score(
                face_crop
            )

            brightness = brightness_score(
                face_crop
            )

            good_blur = blur >= 80

            good_light = (
                50
                <= brightness
                <= 220
            )

            can_capture = (
                good_blur
                and
                good_light
            )

            color = (
                (0,255,0)
                if can_capture
                else (0,0,255)
            )

            cv2.rectangle(
                display,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

        cv2.putText(
            display,
            f"{employee_id}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            display,
            f"{count}/{max_images}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        cv2.imshow(
            "Register Employee",
            display
        )

        now = time.time()

        if (
            can_capture
            and
            now - last_capture >= interval
        ):

            filename = (
                f"{employee_id}_"
                f"{count:03d}.jpg"
            )

            filepath = os.path.join(
                save_dir,
                filename
            )

            cv2.imwrite(
                filepath,
                face_crop
            )

            print(
                f"Saved: {filename}"
            )

            count += 1

            last_capture = now

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        if count >= max_images:
            break

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("CAPTURE COMPLETED")
    print("=" * 60)

    return save_dir


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--employee_id",
        required=True
    )

    parser.add_argument(
        "--full_name",
        required=True
    )

    parser.add_argument(
        "--department",
        default="IT"
    )

    parser.add_argument(
        "--position",
        default="Employee"
    )

    parser.add_argument(
        "--max_images",
        default=100,
        type=int
    )

    args = parser.parse_args()

    create_employee(
        args.employee_id,
        args.full_name,
        args.department,
        args.position
    )

    capture_dataset(
        employee_id=args.employee_id,
        max_images=args.max_images
    )

    print()
    print(
        "[NEXT STEP]"
    )

    print()
    print("=" * 60)
    print("BUILDING EMBEDDING...")
    print("=" * 60)

    builder = EmbeddingBuilder()

    builder.run()

    print()
    print("=" * 60)
    print("EMPLOYEE REGISTERED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()



# chay: python -m src.attendance.register_employee 
# --employee_id NV001 
# --full_name "Nguyen Van A"