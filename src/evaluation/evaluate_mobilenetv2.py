"""
Script đánh giá hiệu năng mô hình MobileNetV2 + ArcFace trên tập dữ liệu chuẩn LFW.
Sinh báo cáo kết quả và các biểu đồ ROC / Confusion Matrix phục vụ khóa luận.
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
from src.models.face_recognition_model import MobileNetV2FaceEmbeddingNet
from src.utils.transforms import get_val_transform


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


class MobileNetV2Evaluator:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        print(f"[INFO] Initializing MobileNetV2 Backbone...")
        self.model = MobileNetV2FaceEmbeddingNet(embedding_size=512, pretrained=False)
        
        print(f"[INFO] Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        self.transform = get_val_transform()
        self.cache = {}

    def load_image_embedding(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image)
        
        embedding = embedding.squeeze(0).cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        self.cache[image_path] = embedding
        return embedding


def read_pairs(pairs_file, root_dir):
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
            img1_path = os.path.join(root_dir, parts[1])
            img2_path = os.path.join(root_dir, parts[2])
            pairs.append((img1_path, img2_path, label))
    return pairs


def evaluate_dataset(evaluator, root_dir, pairs_file, output_csv, dataset_name):
    pairs = read_pairs(pairs_file, root_dir)
    labels = []
    scores = []
    missing = 0
    failed = 0

    desc = f"Evaluating {dataset_name.upper()}"
    for img1, img2, label in tqdm(pairs, desc=desc):
        if not os.path.exists(img1) or not os.path.exists(img2):
            missing += 1
            continue
        try:
            emb1 = evaluator.load_image_embedding(img1)
            emb2 = evaluator.load_image_embedding(img2)
            score = cosine_similarity(emb1, emb2)
            labels.append(label)
            scores.append(score)
        except Exception as e:
            failed += 1
            print(f"[WARNING] Skip pair: {img1} | {img2} | Error: {e}")

    if len(labels) == 0:
        print(f"[ERROR] No valid pairs found for {dataset_name}.")
        return None

    labels_np = np.array(labels)
    scores_np = np.array(scores)

    # Calculate metrics
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

    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
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

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{dataset_name.upper()} Confusion Matrix (MobileNetV2)")
    plt.tight_layout()
    base_name = os.path.splitext(os.path.basename(output_csv))[0]
    cm_path = os.path.join(output_dir, f"{base_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    plt.figure()
    plt.plot(fpr, tpr, color="teal", lw=2, label=f"ROC curve (area = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name.upper()} ROC Curve (MobileNetV2)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"{base_name}_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    return {
        "dataset": dataset_name.upper(),
        "total": len(pairs),
        "valid": len(labels),
        "accuracy": accuracy,
        "auc": auc_score,
        "eer": eer,
        "best_threshold": threshold,
        "far": far,
        "frr": frr
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 + ArcFace on Benchmarks")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/arcface_mobilenetv2_lite.pth",
        help="Path to MobileNetV2 model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        choices=["lfw", "calfw", "cplfw", "agedb30", "all"],
        default="all",
        help="Dataset name"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.checkpoint):
        fallback = args.checkpoint.replace("_lite.pth", ".pth")
        if os.path.exists(fallback):
            args.checkpoint = fallback
        else:
            print(f"[ERROR] Checkpoint not found at: {args.checkpoint} or {fallback}")
            print("[INFO] Please run the training script first to generate checkpoints.")
            return

    evaluator = MobileNetV2Evaluator(args.checkpoint, device)
    
    val_root = "dataset/benchmark/val"
    output_dir = "outputs/evaluation"
    
    datasets_config = {
        "lfw": {"pairs": "lfw_ann.txt", "csv": "lfw_mobilenetv2_eval.csv"},
        "calfw": {"pairs": "calfw_ann.txt", "csv": "calfw_mobilenetv2_eval.csv"},
        "cplfw": {"pairs": "cplfw_ann.txt", "csv": "cplfw_mobilenetv2_eval.csv"},
        "agedb30": {"pairs": "agedb_30_ann.txt", "csv": "agedb30_mobilenetv2_eval.csv"}
    }
    
    target_keys = [args.dataset] if args.dataset != "all" else ["lfw", "calfw", "cplfw", "agedb30"]
    
    results = []
    for key in target_keys:
        cfg = datasets_config[key]
        pairs_file = os.path.join(val_root, cfg["pairs"])
        output_csv = os.path.join(output_dir, cfg["csv"])
        
        print("\n" + "="*50)
        print(f"Evaluating MobileNetV2 on {key.upper()}...")
        print("="*50)
        
        res = evaluate_dataset(evaluator, val_root, pairs_file, output_csv, key)
        if res:
            results.append(res)
            
    print("\n" + "="*90)
    print(" MOBILENETV2 EVALUATION SUMMARY TABLE")
    print("="*90)
    print(f"{'Dataset':<12} | {'Pairs':<6} | {'Accuracy':<10} | {'AUC':<8} | {'EER':<8} | {'Threshold':<10} | {'FAR':<8} | {'FRR':<8}")
    print("-"*90)
    for r in results:
        print(f"{r['dataset']:<12} | {r['valid']:<6} | {r['accuracy']*100:>8.2f}% | {r['auc']:>8.4f} | {r['eer']*100:>6.2f}% | {r['best_threshold']:>10.4f} | {r['far']*100:>6.2f}% | {r['frr']*100:>6.2f}%")
    print("="*90)
    print(f"All reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
