import os
import numpy as np
from PIL import Image

import torch

from src.models.face_recognition_model import (
    FaceRecognitionModel
)

from src.database.supabase_client import (
    supabase
)

from src.config import (
    INHOUSE_ROOT
)

from src.utils.transforms import (
    get_train_transform
)


class EmbeddingBuilder:

    def __init__(self):

        self.model = FaceRecognitionModel()

        self.transform = get_train_transform()

    def process_employee(
        self,
        employee_id,
        employee_dir
    ):

        embeddings = []

        for file_name in os.listdir(
            employee_dir
        ):

            if not file_name.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):
                continue

            image_path = os.path.join(
                employee_dir,
                file_name
            )

            try:

                image = Image.open(
                    image_path
                ).convert(
                    "RGB"
                )

                image = self.transform(
                    image
                )

                image = image.unsqueeze(
                    0
                )

                embedding = (
                    self.model
                    .get_embedding(image)
                )

                embedding = (
                    embedding
                    .squeeze(0)
                    .numpy()
                )

                embeddings.append(
                    embedding
                )

            except Exception as e:

                print(
                    f"Error: {image_path}"
                )

                print(e)

        if len(embeddings) == 0:

            return None

        average_embedding = np.mean(
            embeddings,
            axis=0
        )

        return (
            average_embedding,
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

            "employee_id":
                employee_id,

            "full_name":
                employee_id,

            "embedding_vector":
                embedding_list,

            "image_count":
                image_count
        }

        (
            supabase
            .table("face_embeddings")
            .upsert(payload)
            .execute()
        )

    def run(self):

        employee_dirs = sorted(
            os.listdir(
                INHOUSE_ROOT
            )
        )

        total = len(
            employee_dirs
        )

        print(
            f"Employees found: {total}"
        )

        for employee_id in employee_dirs:

            employee_path = os.path.join(
                INHOUSE_ROOT,
                employee_id
            )

            if not os.path.isdir(
                employee_path
            ):
                continue

            result = self.process_employee(
                employee_id,
                employee_path
            )

            if result is None:

                print(
                    f"No images: {employee_id}"
                )

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