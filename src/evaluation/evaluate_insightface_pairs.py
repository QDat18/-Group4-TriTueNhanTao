import os
import csv
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from insightface.model_zoo import get_model


def cosine_similarity(emb1, emb2):
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))


def calculate_best_threshold(labels, scores):
    best_acc = 0.0
    best_threshold = 0.0

    thresholds = np.linspace(-1, 1, 2001)

    labels = np.array(labels)
    scores = np.array(scores)

    for threshold in thresholds:
        preds = scores >= threshold
        acc = np.mean(preds == labels)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc


def calculate_eer(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]

    return eer, eer_threshold


class InsightFacePairEvaluator:
    def __init__(self, root_dir, pairs_file, output_csv, ctx_id=-1):
        self.root_dir = root_dir
        self.pairs_file = pairs_file
        self.output_csv = output_csv

        model_path = r"C:\Users\Admin\.insightface\models\buffalo_l\w600k_r50.onnx"

        self.rec_model = get_model(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.rec_model.prepare(ctx_id=ctx_id)

        self.cache = {}

    def read_pairs(self):
        pairs = []

        with open(self.pairs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 3:
                    continue

                label = int(parts[0])
                img1 = parts[1]
                img2 = parts[2]

                img1_path = os.path.join(self.root_dir, img1)
                img2_path = os.path.join(self.root_dir, img2)

                pairs.append((img1_path, img2_path, label))

        return pairs

    def load_image_embedding(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path]

        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        if image.shape[:2] != (112, 112):
            image = cv2.resize(image, (112, 112))

        embedding = self.rec_model.get_feat(image)
        embedding = embedding.flatten().astype(np.float32)

        embedding = embedding / np.linalg.norm(embedding)

        self.cache[image_path] = embedding

        return embedding

    def evaluate(self):
        pairs = self.read_pairs()

        labels = []
        scores = []

        missing = 0
        failed = 0

        for img1, img2, label in tqdm(pairs, desc="Evaluating pairs"):
            if not os.path.exists(img1) or not os.path.exists(img2):
                missing += 1
                continue

            try:
                emb1 = self.load_image_embedding(img1)
                emb2 = self.load_image_embedding(img2)

                score = cosine_similarity(emb1, emb2)

                labels.append(label)
                scores.append(score)

            except Exception as e:
                failed += 1
                print(f"[WARNING] Skip pair: {img1} | {img2} | {e}")

        if len(labels) == 0:
            raise RuntimeError("Không có cặp ảnh hợp lệ để đánh giá.")

        labels_np = np.array(labels)
        scores_np = np.array(scores)

        threshold, accuracy = calculate_best_threshold(labels_np, scores_np)
        auc = roc_auc_score(labels_np, scores_np)
        eer, eer_threshold = calculate_eer(labels_np, scores_np)

        preds = scores_np >= threshold

        tp = np.sum((preds == 1) & (labels_np == 1))
        tn = np.sum((preds == 0) & (labels_np == 0))
        fp = np.sum((preds == 1) & (labels_np == 0))
        fn = np.sum((preds == 0) & (labels_np == 1))

        far = fp / max(1, fp + tn)
        frr = fn / max(1, fn + tp)

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        base_name = os.path.splitext(
            os.path.basename(self.output_csv)
        )[0]

        cm_path = os.path.join(
            os.path.dirname(self.output_csv),
            f"{base_name}_confusion_matrix.png"
        )

        cm = confusion_matrix(
            labels_np,
            preds,
            labels=[0, 1]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Different", "Same"]
        )

        disp.plot(cmap="Blues", values_format="d")
        plt.title("InsightFace Buffalo_L Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "total_pairs",
                "valid_pairs",
                "missing_pairs",
                "failed_pairs",
                "accuracy",
                "auc",
                "eer",
                "best_threshold",
                "eer_threshold",
                "far",
                "frr",
                "tp",
                "tn",
                "fp",
                "fn"
            ])

            writer.writerow([
                len(pairs),
                len(labels),
                missing,
                failed,
                accuracy,
                auc,
                eer,
                threshold,
                eer_threshold,
                far,
                frr,
                tp,
                tn,
                fp,
                fn
            ])

        print()
        print("=" * 70)
        print("INSIGHTFACE BUFFALO_L PAIR VERIFICATION RESULT")
        print("=" * 70)
        print(f"Total pairs      : {len(pairs)}")
        print(f"Valid pairs      : {len(labels)}")
        print(f"Missing pairs    : {missing}")
        print(f"Failed pairs     : {failed}")
        print(f"Accuracy         : {accuracy * 100:.2f}%")
        print(f"AUC              : {auc:.4f}")
        print(f"EER              : {eer * 100:.2f}%")
        print(f"Best Threshold   : {threshold:.4f}")
        print(f"EER Threshold    : {eer_threshold:.4f}")
        print(f"FAR              : {far * 100:.2f}%")
        print(f"FRR              : {frr * 100:.2f}%")
        print(f"TP/TN/FP/FN      : {tp}/{tn}/{fp}/{fn}")
        print(f"Confusion Matrix : {cm_path}")
        print(f"Saved CSV        : {self.output_csv}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--pairs_file", required=True)
    parser.add_argument("--output_csv", default="outputs/evaluation/insightface_pair_eval.csv")
    parser.add_argument("--ctx_id", type=int, default=-1)

    args = parser.parse_args()

    evaluator = InsightFacePairEvaluator(
        root_dir=args.root_dir,
        pairs_file=args.pairs_file,
        output_csv=args.output_csv,
        ctx_id=args.ctx_id
    )

    evaluator.evaluate()


if __name__ == "__main__":
    main()