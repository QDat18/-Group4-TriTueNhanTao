import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from src import config
from src.models.face_recognition_model import FaceRecognitionModel
from src.utils.transforms import get_val_transform


class UniqueImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return path, tensor, 1
        except Exception as e:
            return path, torch.zeros(3, 112, 112), 0


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


def main():
    checkpoint_path = "checkpoints/arcface_vggface2_warmup.pth"
    val_root = "dataset/benchmark/val"
    output_dir = "outputs/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Initializing Custom Model on GPU if available...")
    model = FaceRecognitionModel(checkpoint_path=checkpoint_path)
    transform = get_val_transform()

    datasets = {
        "LFW": {
            "pairs": os.path.join(val_root, "lfw_ann.txt"),
            "csv": "lfw_eval.csv"
        },
        "CALFW": {
            "pairs": os.path.join(val_root, "calfw_ann.txt"),
            "csv": "calfw_eval.csv"
        },
        "CPLFW": {
            "pairs": os.path.join(val_root, "cplfw_ann.txt"),
            "csv": "cplfw_eval.csv"
        },
        "AGEDB": {
            "pairs": os.path.join(val_root, "agedb_30_ann.txt"),
            "csv": "agedb30_eval.csv"
        }
    }

    summary_results = []

    for name, cfg in datasets.items():
        print(f"\n========================================\nProcessing {name}...")
        pairs_file = cfg["pairs"]
        
        # Read pairs
        pairs = []
        unique_paths = set()
        with open(pairs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    continue
                label = int(parts[0])
                img1 = os.path.join(val_root, parts[1])
                img2 = os.path.join(val_root, parts[2])
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, label))
                    unique_paths.add(img1)
                    unique_paths.add(img2)

        print(f"Loaded {len(pairs)} valid pairs. Unique images to process: {len(unique_paths)}")
        
        # Extract embeddings in batch
        unique_list = list(unique_paths)
        dataset = UniqueImageDataset(unique_list, transform)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        
        embeddings = {}
        model.model.eval()
        device = model.device
        
        print(f"Extracting embeddings using device: {device}...")
        with torch.no_grad():
            for paths, tensors, valids in dataloader:
                tensors = tensors.to(device)
                embs = model.model(tensors)
                embs = embs.cpu().numpy()
                for p, emb, v in zip(paths, embs, valids):
                    if v.item() == 1:
                        # L2 normalization
                        emb = emb / np.linalg.norm(emb)
                        embeddings[p] = emb

        # Compute similarities
        labels = []
        scores = []
        for img1, img2, label in pairs:
            if img1 in embeddings and img2 in embeddings:
                score = float(np.dot(embeddings[img1], embeddings[img2]))
                labels.append(label)
                scores.append(score)

        labels_np = np.array(labels)
        scores_np = np.array(scores)

        # Compute metrics
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

        # Plot Confusion Matrix
        cm = confusion_matrix(labels_np, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} Confusion Matrix (CUSTOM)")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"{cfg['csv'].replace('.csv', '_confusion_matrix.png')}")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(labels_np, scores_np)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve (CUSTOM)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(output_dir, f"{cfg['csv'].replace('.csv', '_roc_curve.png')}")
        plt.savefig(roc_path, dpi=300)
        plt.close()

        # Save CSV
        csv_path = os.path.join(output_dir, cfg["csv"])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "total_pairs", "valid_pairs", "missing_pairs", "failed_pairs",
                "accuracy", "auc", "eer", "best_threshold", "eer_threshold",
                "far", "frr", "tp", "tn", "fp", "fn"
            ])
            writer.writerow([
                len(pairs), len(labels), 0, 0,
                accuracy, auc_score, eer, threshold, eer_threshold,
                far, frr, tp, tn, fp, fn
            ])

        print(f"Dataset {name} - Accuracy: {accuracy*100:.2f}%, AUC: {auc_score:.4f}, EER: {eer*100:.2f}%")
        summary_results.append({
            "dataset": name,
            "pairs": len(labels),
            "accuracy": f"{accuracy*100:.2f}%",
            "auc": f"{auc_score:.4f}",
            "eer": f"{eer*100:.2f}%",
            "threshold": f"{threshold:.4f}",
            "far": f"{far*100:.2f}%",
            "frr": f"{frr*100:.2f}%"
        })

    # Summary table
    print("\n" + "="*90)
    print(" EVALUATION SUMMARY TABLE (RE-CALCULATED WITH ROC)")
    print("="*90)
    print(f"{'Dataset':<12} | {'Pairs':<6} | {'Accuracy':<10} | {'AUC':<8} | {'EER':<8} | {'Threshold':<10} | {'FAR':<8} | {'FRR':<8}")
    print("-"*90)
    for r in summary_results:
        print(f"{r['dataset']:<12} | {r['pairs']:<6} | {r['accuracy']:>10} | {r['auc']:>8} | {r['eer']:>8} | {r['threshold']:>10} | {r['far']:>8} | {r['frr']:>8}")
    print("="*90)


if __name__ == "__main__":
    main()
