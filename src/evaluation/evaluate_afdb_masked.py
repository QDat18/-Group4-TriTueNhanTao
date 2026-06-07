import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models.face_recognition_model import FaceRecognitionModel
from src.utils.transforms import get_val_transform


AFDB_FACE_ROOT = "dataset/RWFRD/AFDB_face_dataset/AFDB_face_dataset"
AFDB_MASKED_ROOT = "dataset/RWFRD/AFDB_masked_face_dataset/AFDB_masked_face_dataset"

MAX_IDENTITIES = None
MAX_GALLERY_IMAGES = 5
MAX_QUERY_IMAGES = None

RECOGNITION_THRESHOLD = 0.45


class AFDBMaskedEvaluator:
    def __init__(self):
        self.model = FaceRecognitionModel()
        self.transform = get_val_transform()

        self.gallery_embeddings = {}

        self.total = 0
        self.correct = 0
        self.false_reject = 0
        self.wrong_match = 0
        self.similarities = []

    def get_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)

        embedding = self.model.get_embedding(image)
        embedding = embedding.squeeze(0).numpy()

        return embedding

    @staticmethod
    def cosine_similarity(emb1, emb2):
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1, emb2))

    def list_images(self, folder):
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    def build_gallery(self):
        identities = sorted(os.listdir(AFDB_FACE_ROOT))

        if MAX_IDENTITIES is not None:
            identities = identities[:MAX_IDENTITIES]

        print("=" * 70)
        print("BUILDING GALLERY FROM AFDB FACE DATASET")
        print("=" * 70)

        for identity in tqdm(identities, desc="Gallery"):
            identity_dir = os.path.join(AFDB_FACE_ROOT, identity)

            if not os.path.isdir(identity_dir):
                continue

            image_paths = self.list_images(identity_dir)

            if len(image_paths) == 0:
                continue

            image_paths = image_paths[:MAX_GALLERY_IMAGES]

            embeddings = []

            for image_path in image_paths:
                try:
                    emb = self.get_embedding(image_path)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"[WARNING] Skip image: {image_path} | {e}")

            if len(embeddings) == 0:
                continue

            avg_embedding = np.mean(embeddings, axis=0)
            self.gallery_embeddings[identity] = avg_embedding

        print(f"Gallery identities: {len(self.gallery_embeddings)}")

    def find_best_match(self, query_embedding):
        best_identity = None
        best_score = -1.0

        for identity, gallery_embedding in self.gallery_embeddings.items():
            score = self.cosine_similarity(query_embedding, gallery_embedding)

            if score > best_score:
                best_score = score
                best_identity = identity

        if best_score < RECOGNITION_THRESHOLD:
            return None, best_score

        return best_identity, best_score

    def evaluate(self):
        self.build_gallery()

        print("=" * 70)
        print("EVALUATING MASKED FACE RECOGNITION")
        print("=" * 70)

        identities = sorted(os.listdir(AFDB_MASKED_ROOT))

        if MAX_IDENTITIES is not None:
            identities = identities[:MAX_IDENTITIES]

        for identity in tqdm(identities, desc="Query"):
            query_dir = os.path.join(AFDB_MASKED_ROOT, identity)

            if not os.path.isdir(query_dir):
                continue

            if identity not in self.gallery_embeddings:
                continue

            image_paths = self.list_images(query_dir)

            if MAX_QUERY_IMAGES is not None:
                image_paths = image_paths[:MAX_QUERY_IMAGES]

            for image_path in image_paths:
                try:
                    query_embedding = self.get_embedding(image_path)
                except Exception as e:
                    print(f"[WARNING] Skip image: {image_path} | {e}")
                    continue

                predicted_identity, similarity = self.find_best_match(query_embedding)

                self.total += 1
                self.similarities.append(similarity)

                if predicted_identity is None:
                    self.false_reject += 1
                elif predicted_identity == identity:
                    self.correct += 1
                else:
                    self.wrong_match += 1

        self.report()

    def report(self):
        accuracy = self.correct / max(1, self.total)
        frr = self.false_reject / max(1, self.total)
        wrong_rate = self.wrong_match / max(1, self.total)
        mean_similarity = float(np.mean(self.similarities)) if self.similarities else 0.0

        print()
        print("=" * 70)
        print("AFDB MASKED FACE EVALUATION RESULT")
        print("=" * 70)
        print(f"Total query images : {self.total}")
        print(f"Correct matches    : {self.correct}")
        print(f"Wrong matches      : {self.wrong_match}")
        print(f"False rejects      : {self.false_reject}")
        print(f"Accuracy           : {accuracy * 100:.2f}%")
        print(f"Wrong Match Rate   : {wrong_rate * 100:.2f}%")
        print(f"FRR                : {frr * 100:.2f}%")
        print(f"Mean Similarity    : {mean_similarity:.4f}")
        print(f"Threshold          : {RECOGNITION_THRESHOLD}")
        print("=" * 70)


if __name__ == "__main__":
    evaluator = AFDBMaskedEvaluator()
    evaluator.evaluate()
# python -m src.evaluation.evaluate_afdb_masked