import numpy as np
from datetime import datetime, timedelta

from src.database.supabase_client import supabase
from src.config import (
    RECOGNITION_THRESHOLD,
    COOLDOWN_SECONDS
)


class AttendanceService:

    def __init__(self):

        self.employee_embeddings = {}

        self.load_embeddings()

    def load_embeddings(self):

        response = (
            supabase
            .table("face_embeddings")
            .select("*")
            .execute()
        )

        self.employee_embeddings = {}

        for row in response.data:

            self.employee_embeddings[
                row["employee_id"]
            ] = {

                "full_name":
                    row["full_name"],

                "embedding":
                    np.array(
                        row["embedding_vector"],
                        dtype=np.float32
                    )
            }

        print(
            f"Loaded embeddings: "
            f"{len(self.employee_embeddings)}"
        )

    def cosine_similarity(
        self,
        emb1,
        emb2
    ):

        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        return float(
            np.dot(
                emb1,
                emb2
            )
        )

    def find_best_match(
        self,
        embedding
    ):

        best_employee = None
        best_score = -1

        for employee_id, data in (
            self.employee_embeddings.items()
        ):

            score = self.cosine_similarity(
                embedding,
                data["embedding"]
            )

            if score > best_score:

                best_score = score
                best_employee = employee_id

        if best_score < RECOGNITION_THRESHOLD:

            return None

        return {

            "employee_id":
                best_employee,

            "full_name":
                self.employee_embeddings[
                    best_employee
                ]["full_name"],

            "similarity":
                best_score
        }

    def check_cooldown(
        self,
        employee_id
    ):

        response = (
            supabase
            .table("attendance_logs")
            .select("*")
            .eq(
                "employee_id",
                employee_id
            )
            .order(
                "check_time",
                desc=True
            )
            .limit(1)
            .execute()
        )

        if len(response.data) == 0:

            return True

        last_time = datetime.fromisoformat(
            response.data[0]["check_time"]
            .replace(
                "Z",
                "+00:00"
            )
        )

        delta = (
            datetime.utcnow()
            -
            last_time.replace(
                tzinfo=None
            )
        )

        return (
            delta.total_seconds()
            >
            COOLDOWN_SECONDS
        )

    def save_attendance(
        self,
        employee_id,
        similarity
    ):

        if not self.check_cooldown(
            employee_id
        ):
            return False

        payload = {

            "employee_id":
                employee_id,

            "similarity":
                similarity,

            "camera_id":
                "CAM001",

            "status":
                "SUCCESS"
        }

        (
            supabase
            .table("attendance_logs")
            .insert(payload)
            .execute()
        )

        return True