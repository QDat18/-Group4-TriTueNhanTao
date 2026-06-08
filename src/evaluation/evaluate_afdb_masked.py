import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from src import config
# from src.models.face_recognition_model import FaceRecognitionModel
from src.models.insightface_model import InsightFaceModel
from src.utils.transforms import get_val_transform


AFDB_FACE_ROOT = "dataset/RWFRD/AFDB_face_dataset/AFDB_face_dataset"
AFDB_MASKED_ROOT = "dataset/RWFRD/AFDB_masked_face_dataset/AFDB_masked_face_dataset"

MAX_IDENTITIES = None
MAX_GALLERY_IMAGES = 5
MAX_QUERY_IMAGES = None

RECOGNITION_THRESHOLD = 0.05
DEBUG_SAMPLES = 30


class AFDBMaskedEvaluator:
    def __init__(self, checkpoint_path=None):
        self.model = FaceRecognitionModel(checkpoint_path=checkpoint_path)
        self.transform = get_val_transform()

        self.gallery_embeddings = {}

        self.total = 0
        self.correct = 0
        self.false_reject = 0
        self.wrong_match = 0

        self.genuine_scores = []
        self.impostor_scores = []
        self.rank1_correct = 0
        self.rank5_correct = 0

    def get_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)

        embedding = self.model.get_embedding(image)
        embedding = embedding.squeeze(0).detach().cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def cosine_similarity(emb1, emb2):
        emb1 = emb1 / max(np.linalg.norm(emb1), 1e-12)
        emb2 = emb2 / max(np.linalg.norm(emb2), 1e-12)
        return float(np.dot(emb1, emb2))

    def list_images(self, folder):
        return sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

    def get_valid_identities(self, split="test"):
        face_ids = {
            d for d in os.listdir(AFDB_FACE_ROOT)
            if os.path.isdir(os.path.join(AFDB_FACE_ROOT, d))
        }

        masked_ids = {
            d for d in os.listdir(AFDB_MASKED_ROOT)
            if os.path.isdir(os.path.join(AFDB_MASKED_ROOT, d))
        }

        common_ids = face_ids & masked_ids
        txt_path = os.path.join("dataset/RWFRD", f"{split}_identities.txt")
        split_file_loaded = False

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                selected_ids = [line.strip() for line in f if line.strip()]
            # Giữ lại các danh tính hợp lệ xuất hiện trong cả 2 thư mục
            common_ids = sorted([i for i in selected_ids if i in common_ids])
            split_file_loaded = True
            print(f"[INFO] Loaded split '{split}' identities from file: {txt_path}")
        else:
            # Fallback sang chia động dựa trên bảng chữ cái
            sorted_common = sorted(list(common_ids))
            split_ratio = getattr(config, "RMFRD_SPLIT_RATIO", 0.8)
            num_ids = len(sorted_common)
            split_idx = int(num_ids * split_ratio)

            if split == "train":
                common_ids = sorted_common[:split_idx]
            elif split == "test":
                common_ids = sorted_common[split_idx:]
            else:
                common_ids = sorted_common

        print("=" * 70)
        print(f"CHECK AFDB IDENTITY MAPPING (Split: {split.upper() if split else 'ALL'})")
        print("=" * 70)
        print(f"AFDB face identities   : {len(face_ids)}")
        print(f"AFDB masked identities : {len(masked_ids)}")
        print(f"Selected identities    : {len(common_ids)}")

        only_face = sorted(face_ids - masked_ids)
        only_masked = sorted(masked_ids - face_ids)

        if only_face[:5]:
            print(f"Only in face sample    : {only_face[:5]}")
        if only_masked[:5]:
            print(f"Only in masked sample  : {only_masked[:5]}")

        if MAX_IDENTITIES is not None:
            common_ids = common_ids[:MAX_IDENTITIES]

        return common_ids

    def build_gallery(self, identities):
        print("=" * 70)
        print("BUILDING GALLERY FROM AFDB FACE DATASET")
        print("=" * 70)

        for identity in tqdm(identities, desc="Gallery"):
            identity_dir = os.path.join(AFDB_FACE_ROOT, identity)

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
                    print(f"[WARNING] Skip gallery image: {image_path} | {e}")

            if len(embeddings) == 0:
                continue

            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / max(np.linalg.norm(avg_embedding), 1e-12)

            self.gallery_embeddings[identity] = avg_embedding

        print(f"Gallery identities: {len(self.gallery_embeddings)}")

    def evaluate_query(self, query_embedding, true_identity):
        scores = []

        for identity, gallery_embedding in self.gallery_embeddings.items():
            score = self.cosine_similarity(query_embedding, gallery_embedding)
            scores.append((identity, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        best_identity, best_score = scores[0]

        top5 = scores[:5]
        top5_identities = [identity for identity, _ in top5]

        is_rank1 = best_identity == true_identity
        is_rank5 = true_identity in top5_identities

        score_dict = dict(scores)
        genuine_score = score_dict.get(true_identity, -1.0)

        best_impostor_score = max(
            [score for identity, score in scores if identity != true_identity],
            default=-1.0
        )

        all_impostor_scores = [
            score for identity, score in scores if identity != true_identity
        ]

        if best_score < RECOGNITION_THRESHOLD:
            predicted_identity = None
        else:
            predicted_identity = best_identity

        return {
            "predicted_identity": predicted_identity,
            "best_identity": best_identity,
            "best_score": best_score,
            "genuine_score": genuine_score,
            "best_impostor_score": best_impostor_score,
            "all_impostor_scores": all_impostor_scores,
            "rank1": is_rank1,
            "rank5": is_rank5,
            "top5": top5,
        }

    def print_debug_sample(self, image_path, true_identity, result):
        print()
        print("=" * 70)
        print("DEBUG SAMPLE")
        print("=" * 70)
        print(f"Query path     : {image_path}")
        print(f"True ID        : {true_identity}")
        print(f"Predicted ID   : {result['predicted_identity']}")
        print(f"Best ID        : {result['best_identity']}")
        print(f"Best score     : {result['best_score']:.4f}")
        print(f"True score     : {result['genuine_score']:.4f}")
        print(f"Best impostor  : {result['best_impostor_score']:.4f}")
        print(f"Rank-1 correct : {result['rank1']}")
        print(f"Rank-5 correct : {result['rank5']}")
        print("Top-5:")
        for rank, (identity, score) in enumerate(result["top5"], start=1):
            mark = "<-- TRUE" if identity == true_identity else ""
            print(f"  {rank}. {identity} | {score:.4f} {mark}")
        print("=" * 70)

    def evaluate(self, split="test"):
        identities = self.get_valid_identities(split=split)
        self.build_gallery(identities)

        print("=" * 70)
        print("EVALUATING MASKED FACE RECOGNITION")
        print("=" * 70)

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
                    print(f"[WARNING] Skip query image: {image_path} | {e}")
                    continue

                result = self.evaluate_query(query_embedding, identity)

                if self.total < DEBUG_SAMPLES:
                    self.print_debug_sample(image_path, identity, result)

                self.total += 1

                gen_score = result["genuine_score"]
                all_imp_scores = result["all_impostor_scores"]

                if gen_score != -1.0:
                    self.genuine_scores.append(gen_score)
                self.impostor_scores.extend(all_imp_scores)

                if result["rank1"]:
                    self.rank1_correct += 1
                if result["rank5"]:
                    self.rank5_correct += 1

                predicted_identity = result["predicted_identity"]

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

        rank1_acc = self.rank1_correct / max(1, self.total)
        rank5_acc = self.rank5_correct / max(1, self.total)

        genuine_scores = np.array(self.genuine_scores)
        impostor_scores = np.array(self.impostor_scores)

        gen_mean = np.mean(genuine_scores) if len(genuine_scores) > 0 else 0.0
        gen_std = np.std(genuine_scores) if len(genuine_scores) > 0 else 0.0
        imp_mean = np.mean(impostor_scores) if len(impostor_scores) > 0 else 0.0
        imp_std = np.std(impostor_scores) if len(impostor_scores) > 0 else 0.0

        eer_value = 0.0
        eer_threshold = 0.0

        if len(genuine_scores) > 0 and len(impostor_scores) > 0:
            thresholds = np.arange(-1.0, 1.01, 0.01)
            best_diff = float("inf")

            for thr in thresholds:
                frr_t = np.sum(genuine_scores < thr) / len(genuine_scores)
                far_t = np.sum(impostor_scores >= thr) / len(impostor_scores)

                diff = abs(frr_t - far_t)

                if diff < best_diff:
                    best_diff = diff
                    eer_threshold = thr
                    eer_value = (frr_t + far_t) / 2.0

        print()
        print("=" * 70)
        print("AFDB MASKED FACE EVALUATION RESULT")
        print("=" * 70)
        print(f"Total query images  : {self.total}")
        print(f"Correct matches     : {self.correct}")
        print(f"Wrong matches       : {self.wrong_match}")
        print(f"False rejects       : {self.false_reject}")
        print("-" * 70)
        print(f"Accuracy (with Thr) : {accuracy * 100:.2f}%")
        print(f"Wrong Match Rate    : {wrong_rate * 100:.2f}%")
        print(f"FRR (with Thr)      : {frr * 100:.2f}%")
        print("-" * 70)
        print(f"Rank-1 Accuracy     : {rank1_acc * 100:.2f}%")
        print(f"Rank-5 Accuracy     : {rank5_acc * 100:.2f}%")
        print("-" * 70)
        print(f"Genuine similarity  : Mean={gen_mean:.4f}, Std={gen_std:.4f}")
        print(f"Impostor similarity : Mean={imp_mean:.4f}, Std={imp_std:.4f}")
        print(f"Equal Error Rate    : EER={eer_value * 100:.2f}% @ Threshold={eer_threshold:.2f}")
        print(f"Threshold (setting) : {RECOGNITION_THRESHOLD}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập AFDB Masked")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn file checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"], help="Phân hoạch tập dữ liệu")
    args = parser.parse_args()

    split_val = args.split if args.split != "all" else None
    evaluator = AFDBMaskedEvaluator(checkpoint_path=args.checkpoint)
    evaluator.evaluate(split=split_val)

# python -m src.evaluation.evaluate_afdb_masked