"""
Script đánh giá hiệu năng mô hình nhận diện khuôn mặt trên các tập benchmark chuẩn (LFW, CALFW, CPLFW, AgeDB-30).
Hỗ trợ cả mô hình đề xuất (Custom ResNet-50) và mô hình cơ sở (InsightFace buffalo_l).

Sử dụng:
    # Đánh giá mô hình đề xuất (Custom) trên tập LFW
    python -m src.evaluation.evaluate_benchmark --model-type custom --checkpoint checkpoints/arcface_vggface2_warmup.pth --dataset lfw

    # Đánh giá mô hình đề xuất (Custom) trên tất cả các tập
    python -m src.evaluation.evaluate_benchmark --model-type custom --checkpoint checkpoints/arcface_vggface2_warmup.pth --dataset all

    # Đánh giá mô hình InsightFace (Baseline) trên tập LFW
    python -m src.evaluation.evaluate_benchmark --model-type baseline --dataset lfw
"""

import os
import csv
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from src import config
from src.models.face_recognition_model import FaceRecognitionModel
from src.utils.transforms import get_val_transform


def cosine_similarity(emb1, emb2):
    # Chuẩn hóa L2 trước khi tính dot product để có Cosine Similarity chuẩn
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


class BenchmarkEvaluator:
    def __init__(self, model_type="custom", checkpoint_path=None, ctx_id=-1):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.cache = {}

        if self.model_type == "custom":
            print(f"[INFO] Initializing Custom Model (ResNet-50)...")
            self.model = FaceRecognitionModel(checkpoint_path=checkpoint_path)
            self.transform = get_val_transform()
        else:
            print(f"[INFO] Initializing Baseline Model (InsightFace buffalo_l)...")
            # Tải model từ InsightFace ONNX
            from insightface.model_zoo import get_model
            onnx_path = r"C:\Users\Admin\.insightface\models\buffalo_l\w600k_r50.onnx"
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"InsightFace ONNX model not found at: {onnx_path}")
            self.rec_model = get_model(onnx_path, providers=["CPUExecutionProvider"])
            self.rec_model.prepare(ctx_id=ctx_id)

    def load_image_embedding(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path]

        if self.model_type == "custom":
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            # Trích xuất đặc trưng
            embedding = self.model.get_embedding(image)
            embedding = embedding.squeeze(0).numpy()
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"Cannot read image: {image_path}")
            if image.shape[:2] != (112, 112):
                image = cv2.resize(image, (112, 112))
            embedding = self.rec_model.get_feat(image)
            embedding = embedding.flatten().astype(np.float32)

        # L2 Normalize
        embedding = embedding / np.linalg.norm(embedding)
        self.cache[image_path] = embedding
        return embedding

    def read_pairs(self, root_dir, pairs_file):
        pairs = []
        with open(pairs_file, "r", encoding="utf-8") as f:
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

                img1_path = os.path.join(root_dir, img1)
                img2_path = os.path.join(root_dir, img2)
                pairs.append((img1_path, img2_path, label))
        return pairs

    def evaluate_dataset(self, root_dir, pairs_file, output_csv):
        pairs = self.read_pairs(root_dir, pairs_file)
        labels = []
        scores = []
        missing = 0
        failed = 0

        dataset_name = os.path.basename(pairs_file).split("_")[0].upper()
        desc = f"Evaluating {dataset_name} ({self.model_type})"

        for img1, img2, label in tqdm(pairs, desc=desc):
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
                print(f"[WARNING] Skip pair: {img1} | {img2} | Error: {e}")

        if len(labels) == 0:
            raise RuntimeError(f"No valid pairs for {dataset_name}.")

        labels_np = np.array(labels)
        scores_np = np.array(scores)

        # Tính toán các chỉ số
        threshold, accuracy = calculate_best_threshold(labels_np, scores_np)
        auc_score = roc_auc_score(labels_np, scores_np)
        eer, eer_threshold = calculate_eer(labels_np, scores_np)

        preds = scores_np >= threshold
        tp = np.sum((preds == 1) & (labels_np == 1))
        tn = np.sum((preds == 0) & (labels_np == 0))
        fp = np.sum((preds == 1) & (labels_np == 0))
        fn = np.sum((preds == 0) & (labels_np == 1))

        far = fp / max(1, fp + tn)
        frr = fn / max(1, fn + tp)

        # Lưu CSV kết quả
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "total_pairs", "valid_pairs", "missing_pairs", "failed_pairs",
                "accuracy", "auc", "eer", "best_threshold", "eer_threshold",
                "far", "frr", "tp", "tn", "fp", "fn"
            ])
            writer.writerow([
                len(pairs), len(labels), missing, failed,
                accuracy, auc_score, eer, threshold, eer_threshold,
                far, frr, tp, tn, fp, fn
            ])

        # Vẽ Ma trận nhầm lẫn và lưu biểu đồ
        cm = confusion_matrix(labels_np, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
        disp.plot(cmap="Blues", values_format="d")
        
        plt.title(f"{dataset_name} Confusion Matrix ({self.model_type.upper()})")
        plt.tight_layout()
        
        base_name = os.path.splitext(os.path.basename(output_csv))[0]
        cm_path = os.path.join(os.path.dirname(output_csv), f"{base_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        # Vẽ đường cong ROC và lưu biểu đồ
        fpr, tpr, _ = roc_curve(labels_np, scores_np)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset_name} ROC Curve ({self.model_type.upper()})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        roc_path = os.path.join(os.path.dirname(output_csv), f"{base_name}_roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()

        return {
            "dataset": dataset_name,
            "total": len(pairs),
            "valid": len(labels),
            "accuracy": accuracy,
            "auc": auc_score,
            "eer": eer,
            "best_threshold": threshold,
            "far": far,
            "frr": frr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "cm_path": cm_path,
            "roc_path": roc_path
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on LFW, CALFW, CPLFW, AgeDB-30")
    parser.add_argument(
        "--model-type",
        choices=["custom", "baseline"],
        default="custom",
        help="Model type: custom (ResNet-50) or baseline (InsightFace buffalo_l)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "arcface_vggface2_warmup.pth"),
        help="Path to custom model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        choices=["lfw", "calfw", "cplfw", "agedb30", "all"],
        default="all",
        help="Dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="GPU ID (-1 for CPU)"
    )

    args = parser.parse_args()

    # Định nghĩa cấu hình các tập dữ liệu
    val_root = "dataset/benchmark/val"
    datasets_config = {
        "lfw": {
            "root": val_root,
            "pairs": os.path.join(val_root, "lfw_ann.txt"),
            "csv": "lfw_eval.csv" if args.model_type == "custom" else "lfw_buffalo_l_eval.csv"
        },
        "calfw": {
            "root": val_root,
            "pairs": os.path.join(val_root, "calfw_ann.txt"),
            "csv": "calfw_eval.csv" if args.model_type == "custom" else "calfw_buffalo_l_eval.csv"
        },
        "cplfw": {
            "root": val_root,
            "pairs": os.path.join(val_root, "cplfw_ann.txt"),
            "csv": "cplfw_eval.csv" if args.model_type == "custom" else "cplfw_buffalo_l_eval.csv"
        },
        "agedb30": {
            "root": val_root,
            "pairs": os.path.join(val_root, "agedb_30_ann.txt"),
            "csv": "agedb30_eval.csv" if args.model_type == "custom" else "agedb30_buffalo_l_eval.csv"
        }
    }

    # Lọc danh sách tập dữ liệu chạy thực tế
    target_keys = [args.dataset] if args.dataset != "all" else ["lfw", "calfw", "cplfw", "agedb30"]

    # Khởi tạo Evaluator
    evaluator = BenchmarkEvaluator(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint if args.model_type == "custom" else None,
        ctx_id=args.ctx_id
    )

    results = []

    for key in target_keys:
        cfg = datasets_config[key]
        csv_path = os.path.join(args.output_dir, cfg["csv"])
        
        print("\n" + "="*80)
        print(f"Evaluating: {key.upper()}...")
        print(f"  Images root: {cfg['root']}")
        print(f"  Pairs file:  {cfg['pairs']}")
        print(f"  Output CSV:  {csv_path}")
        print("="*80)

        res = evaluator.evaluate_dataset(
            root_dir=cfg["root"],
            pairs_file=cfg["pairs"],
            output_csv=csv_path
        )
        results.append(res)

    # In ra bảng tổng kết kết quả đánh giá
    print("\n" + "="*90)
    print(f" EVALUATION SUMMARY TABLE ({args.model_type.upper()})")
    print("="*90)
    print(f"{'Dataset':<12} | {'Pairs':<6} | {'Accuracy':<10} | {'AUC':<8} | {'EER':<8} | {'Threshold':<10} | {'FAR':<8} | {'FRR':<8}")
    print("-"*90)
    for r in results:
        print(f"{r['dataset']:<12} | {r['valid']:<6} | {r['accuracy']*100:>8.2f}% | {r['auc']:>8.4f} | {r['eer']*100:>6.2f}% | {r['best_threshold']:>10.4f} | {r['far']*100:>6.2f}% | {r['frr']*100:>6.2f}%")
    print("="*90)
    print(f"All reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
