"""
Đánh giá model ArcFace trên tập TEST SPLIT của VGGFace2.

Metrics:
  - Top-1 / Top-5 Classification Accuracy (qua ArcFace head)
  - Embedding Retrieval Accuracy (cosine similarity, không cần head)
  - Per-class accuracy breakdown
  - Loss trên test set
  - Confusion analysis (worst classes)

Sử dụng:
    python -m src.evaluation.evaluate_vggface2
"""

import os
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.models.face_recognition_model import FaceEmbeddingNet
from src.models.arcface_head import ArcMarginProduct
from src.datasets.dataset_vggface2 import VGGFace2Dataset
from src.utils.transforms import get_val_transform


REPORT_DIR = "evaluation_reports"


def load_full_checkpoint(checkpoint_path: str, device: str):
    """
    Load full checkpoint (backbone + ArcFace head) để đánh giá classification.
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Cần biết num_classes từ ArcFace head weights
    arcface_weight = checkpoint["arcface_head_state_dict"]["weight"]
    num_classes = arcface_weight.shape[0]
    embedding_size = arcface_weight.shape[1]

    print(f"  Num classes    : {num_classes}")
    print(f"  Embedding size : {embedding_size}")
    print(f"  Epoch          : {checkpoint.get('epoch', '?')}")
    print(f"  Val loss       : {checkpoint.get('val_loss', '?'):.4f}")

    # Build model
    model = FaceEmbeddingNet(embedding_size=embedding_size, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build ArcFace head
    arcface_head = ArcMarginProduct(
        embedding_size=embedding_size,
        num_classes=num_classes,
        scale=64.0,
        margin=0.5,
    ).to(device)
    arcface_head.load_state_dict(checkpoint["arcface_head_state_dict"])
    arcface_head.eval()

    return model, arcface_head, num_classes


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    arcface_head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
):
    """
    Đánh giá Top-1 / Top-5 accuracy + loss trên test split.
    """
    model.eval()
    arcface_head.eval()

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    all_preds = []
    all_labels = []
    all_similarities = []

    progress = tqdm(dataloader, desc="Evaluating (classification)")

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        logits = arcface_head(embeddings, labels)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        # Top-1
        _, pred_top1 = logits.max(dim=1)
        top1_correct += (pred_top1 == labels).sum().item()

        # Top-5
        _, pred_top5 = logits.topk(5, dim=1)
        top5_match = pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5))
        top5_correct += top5_match.any(dim=1).sum().item()

        total += labels.size(0)

        # Per-class stats
        for i in range(labels.size(0)):
            lbl = labels[i].item()
            per_class_total[lbl] += 1
            if pred_top1[i].item() == lbl:
                per_class_correct[lbl] += 1

        all_preds.extend(pred_top1.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Similarity score (cosine of correct class)
        cosine_scores = logits / 64.0  # undo scale to approximate cosine
        for i in range(labels.size(0)):
            all_similarities.append(cosine_scores[i, labels[i]].item())

        progress.set_postfix({
            "top1": f"{top1_correct / total * 100:.1f}%",
            "top5": f"{top5_correct / total * 100:.1f}%",
        })

    num_batches = max(1, len(dataloader))

    return {
        "total_samples": total,
        "top1_accuracy": top1_correct / max(1, total),
        "top5_accuracy": top5_correct / max(1, total),
        "avg_loss": total_loss / num_batches,
        "per_class_correct": dict(per_class_correct),
        "per_class_total": dict(per_class_total),
        "all_preds": np.array(all_preds),
        "all_labels": np.array(all_labels),
        "all_similarities": np.array(all_similarities),
    }


@torch.no_grad()
def evaluate_retrieval(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
):
    """
    Đánh giá embedding retrieval:
    - Trích xuất tất cả embeddings
    - Tính centroid mỗi class
    - Kiểm tra nearest centroid accuracy (giống thực tế inference)
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    progress = tqdm(dataloader, desc="Extracting embeddings")

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    # Tính centroid cho mỗi class
    centroids = {}
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        centroid = all_embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[cls] = centroid

    centroid_matrix = np.array([centroids[cls] for cls in sorted(centroids.keys())])
    centroid_labels = np.array(sorted(centroids.keys()))

    # Cosine similarity = dot product (vì embeddings đã L2 normalized)
    similarities = all_embeddings @ centroid_matrix.T  # (N, num_centroids)

    # Nearest centroid
    pred_indices = similarities.argmax(axis=1)
    pred_labels = centroid_labels[pred_indices]

    retrieval_acc = (pred_labels == all_labels).mean()

    # Phân bố similarity scores
    correct_mask = pred_labels == all_labels
    correct_sims = similarities[np.arange(len(all_labels)), pred_indices][correct_mask]
    wrong_sims = similarities[np.arange(len(all_labels)), pred_indices][~correct_mask]

    return {
        "retrieval_accuracy": retrieval_acc,
        "correct_similarities": correct_sims,
        "wrong_similarities": wrong_sims,
        "all_embeddings": all_embeddings,
        "all_labels": all_labels,
    }


def plot_per_class_accuracy(per_class_correct, per_class_total, report_dir, timestamp):
    """Vẽ histogram phân bố accuracy theo class."""
    accuracies = []
    for cls in per_class_total:
        total = per_class_total[cls]
        correct = per_class_correct.get(cls, 0)
        if total > 0:
            accuracies.append(correct / total * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(accuracies, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(accuracies), color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(accuracies):.1f}%")
    ax.axvline(np.median(accuracies), color="#2ecc71", linestyle="--", linewidth=2,
               label=f"Median: {np.median(accuracies):.1f}%")

    ax.set_xlabel("Accuracy per Class (%)", fontsize=12)
    ax.set_ylabel("Number of Classes", fontsize=12)
    ax.set_title("Distribution of Per-Class Accuracy (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(report_dir, f"per_class_accuracy_{timestamp}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Per-class accuracy: {path}")


def plot_retrieval_similarity(correct_sims, wrong_sims, report_dir, timestamp):
    """Vẽ phân bố similarity cho retrieval (correct vs wrong)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(correct_sims) > 0:
        ax.hist(correct_sims, bins=50, alpha=0.6, color="#2ecc71",
                label=f"Correct ({len(correct_sims)})", density=True)

    if len(wrong_sims) > 0:
        ax.hist(wrong_sims, bins=50, alpha=0.6, color="#e74c3c",
                label=f"Wrong ({len(wrong_sims)})", density=True)

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Retrieval Similarity Distribution (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(report_dir, f"retrieval_similarity_{timestamp}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Retrieval similarity: {path}")


def plot_worst_classes(per_class_correct, per_class_total, report_dir, timestamp, top_n=20):
    """Vẽ bar chart cho top-N class kém nhất."""
    class_accs = []
    for cls in per_class_total:
        total = per_class_total[cls]
        correct = per_class_correct.get(cls, 0)
        if total > 0:
            class_accs.append((cls, correct / total * 100, total))

    class_accs.sort(key=lambda x: x[1])
    worst = class_accs[:top_n]

    if not worst:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"Class {c[0]} (n={c[2]})" for c in worst]
    values = [c[1] for c in worst]
    colors = ["#e74c3c" if v < 50 else "#f39c12" if v < 75 else "#3498db" for v in values]

    bars = ax.barh(range(len(worst)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(worst)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Top-{top_n} Worst Performing Classes", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(report_dir, f"worst_classes_{timestamp}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Worst classes: {path}")


def main():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  ĐÁNH GIÁ MODEL ARCFACE TRÊN VGGFACE2 TEST SPLIT")
    print("=" * 70)

    device = config.DEVICE
    print(f"Device: {device}")

    # ─── Load checkpoint ───
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)

    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Không tìm thấy checkpoint: {checkpoint_path}")
        return

    model, arcface_head, num_classes = load_full_checkpoint(checkpoint_path, device)

    # ─── Load test dataset ───
    split_ratio = getattr(config, "SPLIT_RATIO", (0.8, 0.1, 0.1))
    max_classes = config.MAX_CLASSES if config.USE_SUBSET else None
    max_images = config.MAX_IMAGES_PER_CLASS if config.USE_SUBSET else None

    test_dataset = VGGFace2Dataset(
        root_dir=config.VGGFACE2_ROOT,
        transform=get_val_transform(),
        max_classes=max_classes,
        max_images_per_class=max_images,
        split="test",
        split_ratio=split_ratio,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\nTest images      : {len(test_dataset)}")
    print(f"Test identities  : {test_dataset.num_classes}")
    print(f"Test batches     : {len(test_loader)}")
    print(f"Batch size       : {config.BATCH_SIZE}")

    os.makedirs(REPORT_DIR, exist_ok=True)

    # ═══════════════════════════════════════
    # 1. Classification Evaluation
    # ═══════════════════════════════════════

    print("\n" + "=" * 70)
    print("  1. CLASSIFICATION EVALUATION (ArcFace Head)")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    cls_results = evaluate_classification(
        model=model,
        arcface_head=arcface_head,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    cls_time = time.time() - start_time

    print(f"\n  Total samples  : {cls_results['total_samples']}")
    print(f"  Test Loss      : {cls_results['avg_loss']:.4f}")
    print(f"  Top-1 Accuracy : {cls_results['top1_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy : {cls_results['top5_accuracy'] * 100:.2f}%")
    print(f"  Eval time      : {cls_time:.1f}s")

    # ═══════════════════════════════════════
    # 2. Embedding Retrieval Evaluation
    # ═══════════════════════════════════════

    print("\n" + "=" * 70)
    print("  2. EMBEDDING RETRIEVAL EVALUATION (Cosine Similarity)")
    print("=" * 70)

    start_time = time.time()

    ret_results = evaluate_retrieval(
        model=model,
        dataloader=test_loader,
        device=device,
        num_classes=num_classes,
    )

    ret_time = time.time() - start_time

    print(f"\n  Retrieval Accuracy  : {ret_results['retrieval_accuracy'] * 100:.2f}%")

    if len(ret_results["correct_similarities"]) > 0:
        print(f"  Correct sim (mean)  : {ret_results['correct_similarities'].mean():.4f}")
        print(f"  Correct sim (std)   : {ret_results['correct_similarities'].std():.4f}")

    if len(ret_results["wrong_similarities"]) > 0:
        print(f"  Wrong sim (mean)    : {ret_results['wrong_similarities'].mean():.4f}")
        print(f"  Wrong sim (std)     : {ret_results['wrong_similarities'].std():.4f}")

    print(f"  Eval time           : {ret_time:.1f}s")

    # ═══════════════════════════════════════
    # 3. Per-Class Analysis
    # ═══════════════════════════════════════

    print("\n" + "=" * 70)
    print("  3. PER-CLASS ANALYSIS")
    print("=" * 70)

    per_class_accs = []
    for cls in cls_results["per_class_total"]:
        total = cls_results["per_class_total"][cls]
        correct = cls_results["per_class_correct"].get(cls, 0)
        if total > 0:
            per_class_accs.append(correct / total * 100)

    if per_class_accs:
        print(f"\n  Classes evaluated : {len(per_class_accs)}")
        print(f"  Mean accuracy     : {np.mean(per_class_accs):.2f}%")
        print(f"  Median accuracy   : {np.median(per_class_accs):.2f}%")
        print(f"  Std accuracy      : {np.std(per_class_accs):.2f}%")
        print(f"  Min accuracy      : {np.min(per_class_accs):.2f}%")
        print(f"  Max accuracy      : {np.max(per_class_accs):.2f}%")

        # Count classes by accuracy range
        bins = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 100)]
        print(f"\n  {'Range':<15} {'Count':>8} {'Percent':>10}")
        print("  " + "-" * 35)
        for low, high in bins:
            count = sum(1 for a in per_class_accs if low <= a < high)
            pct = count / len(per_class_accs) * 100
            print(f"  {low:>3}% - {high:<3}%     {count:>8} {pct:>9.1f}%")
        count_100 = sum(1 for a in per_class_accs if a == 100)
        pct_100 = count_100 / len(per_class_accs) * 100
        print(f"  {'100%':<15} {count_100:>8} {pct_100:>9.1f}%")

    # ═══════════════════════════════════════
    # 4. So sánh với Training Metrics
    # ═══════════════════════════════════════

    print("\n" + "=" * 70)
    print("  4. SO SÁNH TRAIN / VAL / TEST")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_loss = checkpoint.get("train_loss", "?")
    val_loss = checkpoint.get("val_loss", "?")

    print(f"\n  {'':>18} {'Train':>12} {'Val':>12} {'Test':>12}")
    print("  " + "-" * 50)

    if isinstance(train_loss, float) and isinstance(val_loss, float):
        print(f"  {'Loss':>18} {train_loss:>12.4f} {val_loss:>12.4f} {cls_results['avg_loss']:>12.4f}")
    else:
        print(f"  {'Loss':>18} {'?':>12} {'?':>12} {cls_results['avg_loss']:>12.4f}")

    print(f"  {'Top-1 Accuracy':>18} {'—':>12} {'—':>12} {cls_results['top1_accuracy'] * 100:>11.2f}%")
    print(f"  {'Top-5 Accuracy':>18} {'—':>12} {'—':>12} {cls_results['top5_accuracy'] * 100:>11.2f}%")
    print(f"  {'Retrieval Acc':>18} {'—':>12} {'—':>12} {ret_results['retrieval_accuracy'] * 100:>11.2f}%")

    # ═══════════════════════════════════════
    # 5. Xuất Charts
    # ═══════════════════════════════════════

    print("\n" + "=" * 70)
    print("  5. XUẤT BIỂU ĐỒ")
    print("=" * 70 + "\n")

    plot_per_class_accuracy(
        cls_results["per_class_correct"],
        cls_results["per_class_total"],
        REPORT_DIR, timestamp,
    )

    plot_retrieval_similarity(
        ret_results["correct_similarities"],
        ret_results["wrong_similarities"],
        REPORT_DIR, timestamp,
    )

    plot_worst_classes(
        cls_results["per_class_correct"],
        cls_results["per_class_total"],
        REPORT_DIR, timestamp,
    )

    # ═══════════════════════════════════════
    # 6. Lưu summary CSV
    # ═══════════════════════════════════════

    import csv
    summary_path = os.path.join(REPORT_DIR, f"test_summary_{timestamp}.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["checkpoint", checkpoint_path])
        writer.writerow(["test_samples", cls_results["total_samples"]])
        writer.writerow(["num_classes", num_classes])
        writer.writerow(["test_loss", f"{cls_results['avg_loss']:.6f}"])
        writer.writerow(["top1_accuracy", f"{cls_results['top1_accuracy']:.6f}"])
        writer.writerow(["top5_accuracy", f"{cls_results['top5_accuracy']:.6f}"])
        writer.writerow(["retrieval_accuracy", f"{ret_results['retrieval_accuracy']:.6f}"])
        writer.writerow(["mean_class_accuracy", f"{np.mean(per_class_accs):.6f}" if per_class_accs else "N/A"])
        writer.writerow(["classification_time_sec", f"{cls_time:.2f}"])
        writer.writerow(["retrieval_time_sec", f"{ret_time:.2f}"])

    print(f"\n  📄 Summary CSV: {summary_path}")

    print("\n" + "=" * 70)
    print("  ✅ ĐÁNH GIÁ HOÀN TẤT")
    print(f"  📁 Báo cáo: {REPORT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
