import os
import cv2
import numpy as np

from src.models.insightface_model import InsightFaceModel
from src.database.supabase_client import supabase
from src.config import INHOUSE_ROOT


class EmbeddingBuilder:

    def __init__(self):
        self.model = InsightFaceModel()

    def process_employee(
        self,
        employee_id,
        employee_dir
    ):

        embeddings = []

        for file_name in os.listdir(employee_dir):

            if not file_name.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                )
            ):
                continue

            image_path = os.path.join(
                employee_dir,
                file_name
            )

            try:
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Cannot read image: {image_path}")
                    continue

                embedding = self.model.get_embedding_from_aligned(
                    image
                )

                if embedding is None:
                    print(f"No embedding: {image_path}")
                    continue

                embeddings.append(embedding)

            except Exception as e:
                print(f"Error: {image_path}")
                print(e)

        if len(embeddings) == 0:
            return None

        average_embedding = np.mean(
            embeddings,
            axis=0
        )

        average_embedding = average_embedding / np.linalg.norm(
            average_embedding
        )

        return (
            average_embedding.astype(np.float32),
            len(embeddings)
        )

    def save_to_supabase(
        self,
        employee_id,
        embedding,
        image_count
    ):

        embedding_list = (
            embedding
            .astype(float)
            .tolist()
        )

        payload = {
            "employee_id": employee_id,
            "embedding_vector": embedding_list,
            "image_count": image_count
        }

        try:
            (
                supabase
                .table("face_embeddings")
                .delete()
                .eq("employee_id", employee_id)
                .execute()
            )
        except Exception as e:
            print(f"Warning deleting old embedding: {e}")

        (
            supabase
            .table("face_embeddings")
            .upsert(payload)
            .execute()
        )

    def run(self):

        employee_dirs = sorted(
            os.listdir(INHOUSE_ROOT)
        )

        total = len(employee_dirs)

        print(f"Employees found: {total}")

        for employee_id in employee_dirs:

            employee_path = os.path.join(
                INHOUSE_ROOT,
                employee_id
            )

            if not os.path.isdir(employee_path):
                continue

            result = self.process_employee(
                employee_id,
                employee_path
            )

            if result is None:
                print(f"No valid images: {employee_id}")
                continue

            embedding, image_count = result

            self.save_to_supabase(
                employee_id,
                embedding,
                image_count
            )

            print(
                f"Saved: {employee_id}"
                f" | Images: {image_count}"
            )


if __name__ == "__main__":
    builder = EmbeddingBuilder()
    builder.run()