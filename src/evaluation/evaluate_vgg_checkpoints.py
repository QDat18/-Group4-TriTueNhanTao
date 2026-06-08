"""
Đánh giá checkpoint mô hình ArcFace VGG trên tập kiểm thử chưa huấn luyện (VGGFace2 test split).

Chạy:
    venv/Scripts/python -m src.evaluation.evaluate_vgg_checkpoints --checkpoint checkpoints/arcface_vggface2_warmup.pth --max-images 200000
"""

import os
import csv
import time
import argparse
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Chạy headless không cần giao diện đồ họa
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from src import config
from src.models.face_recognition_model import FaceEmbeddingNet
from src.models.arcface_head import ArcMarginProduct
from src.datasets.dataset_vggface2 import VGGFace2Dataset
from src.utils.transforms import get_val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint VGG trên tập dữ liệu chưa train")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "arcface_vggface2_warmup.pth"),
        help="Đường dẫn đến file checkpoint (.pth)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size để trích xuất đặc trưng"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200000,
        help="Số lượng ảnh tối đa để đánh giá (-1 để đánh giá toàn bộ tập test)"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=50000,
        help="Số lượng cặp ảnh (50%% genuine, 50%% impostor) để đánh giá verification"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config.NUM_WORKERS,
        help="Số lượng worker cho DataLoader"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_reports",
        help="Thư mục lưu báo cáo kết quả"
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(fpr, tpr, thresholds):
    """Tính Equal Error Rate (EER) và ngưỡng tương ứng."""
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return eer, eer_threshold


def generate_verification_pairs(labels, num_pairs=50000, seed=42):
    """
    Tạo các cặp ảnh Genuine (cùng identity) và Impostor (khác identity) từ tập nhãn.
    Đảm bảo nhanh và không bị trùng lặp.
    """
    rng = np.random.default_rng(seed)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # Loại bỏ các identity có ít hơn 2 ảnh đối với cặp genuine
    valid_genuine_labels = [lbl for lbl, idxs in label_to_indices.items() if len(idxs) >= 2]
    all_labels = list(label_to_indices.keys())

    if len(valid_genuine_labels) == 0:
        raise ValueError("Không có danh tính nào có từ 2 ảnh trở lên để tạo cặp Genuine.")

    pairs_idx1 = []
    pairs_idx2 = []
    pair_labels = []

    half_pairs = num_pairs // 2

    # 1. Tạo cặp Genuine (cùng người)
    print(f"Đang sinh {half_pairs} cặp Genuine...")
    for _ in range(half_pairs):
        lbl = rng.choice(valid_genuine_labels)
        idxs = label_to_indices[lbl]
        idx1, idx2 = rng.choice(idxs, size=2, replace=False)
        pairs_idx1.append(idx1)
        pairs_idx2.append(idx2)
        pair_labels.append(1)  # 1 = Same Identity

    # 2. Tạo cặp Impostor (khác người)
    print(f"Đang sinh {half_pairs} cặp Impostor...")
    for _ in range(half_pairs):
        lbl1, lbl2 = rng.choice(all_labels, size=2, replace=False)
        idx1 = rng.choice(label_to_indices[lbl1])
        idx2 = rng.choice(label_to_indices[lbl2])
        pairs_idx1.append(idx1)
        pairs_idx2.append(idx2)
        pair_labels.append(0)  # 0 = Different Identity

    return np.array(pairs_idx1), np.array(pairs_idx2), np.array(pair_labels)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("  ĐÁNH GIÁ CHECKPOINT VGG TRÊN TẬP ẢNH CHƯA TRAIN")
    print("=" * 70)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Device       : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Name     : {torch.cuda.get_device_name(0)}")
    print(f"  Batch Size   : {args.batch_size}")
    print("=" * 70)

    # 1. Tải checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {args.checkpoint}")
        
    print("Đang đọc file checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Khởi tạo mô hình trích xuất đặc trưng
    model = FaceEmbeddingNet(embedding_size=512, pretrained=False).to(device)
    
    # Load model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Tải thành công model_state_dict của backbone.")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        print("Tải thành công model_state_dict trực tiếp từ file.")
    else:
        raise ValueError("Cấu trúc file checkpoint không hợp lệ.")
        
    model.eval()

    # Kiểm tra xem có ArcFace head trong checkpoint không để chạy Classification
    arcface_head = None
    has_head = False
    
    if isinstance(checkpoint, dict) and "arcface_head_state_dict" in checkpoint:
        head_state = checkpoint["arcface_head_state_dict"]
        num_classes = head_state["weight"].shape[0]
        print(f"Phát hiện ArcFace classification head với {num_classes} classes.")
        
        arcface_head = ArcMarginProduct(
            embedding_size=512,
            num_classes=num_classes,
            scale=64.0,
            margin=0.5
        ).to(device)
        arcface_head.load_state_dict(head_state)
        arcface_head.eval()
        has_head = True
    else:
        print("[INFO] Không tìm thấy ArcFace head trong checkpoint. Sẽ chỉ đánh giá Verification.")

    # 2. Tải dataset test split
    print("Đang khởi tạo Dataset (VGGFace2 test split)...")
    split_ratio = getattr(config, "SPLIT_RATIO", (0.8, 0.1, 0.1))
    max_classes = config.MAX_CLASSES if config.USE_SUBSET else None
    max_images_per_class = config.MAX_IMAGES_PER_CLASS if config.USE_SUBSET else None

    # Tải dataset test
    test_dataset = VGGFace2Dataset(
        root_dir=config.VGGFACE2_ROOT,
        transform=get_val_transform(),
        max_classes=max_classes,
        max_images_per_class=max_images_per_class,
        split="test",
        split_ratio=split_ratio,
        seed=args.seed
    )

    # Giới hạn số lượng ảnh đánh giá nếu cần
    if args.max_images > 0 and len(test_dataset) > args.max_images:
        print(f"Giới hạn dữ liệu đánh giá từ {len(test_dataset):,} xuống {args.max_images:,} ảnh.")
        test_dataset.samples = test_dataset.samples[:args.max_images]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    num_samples = len(test_dataset)
    print(f"Tổng số ảnh đánh giá thực tế: {num_samples:,}")

    # Khởi tạo các mảng chứa kết quả
    all_embeddings = np.zeros((num_samples, 512), dtype=np.float32)
    all_labels = np.zeros(num_samples, dtype=np.int64)

    # Các biến thống kê classification
    class_loss = 0.0
    class_correct_top1 = 0
    class_correct_top5 = 0
    criterion = nn.CrossEntropyLoss()

    print("\nBắt đầu chạy Forward qua mô hình...")
    start_time = time.time()
    
    idx_start = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Extracting Embeddings"):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            # Trích xuất embeddings
            embeddings = model(images)
            all_embeddings[idx_start:idx_start + batch_size] = embeddings.cpu().numpy()
            all_labels[idx_start:idx_start + batch_size] = labels.cpu().numpy()

            # Đánh giá phân loại nếu có head
            if has_head:
                logits = arcface_head(embeddings, labels)
                loss = criterion(logits, labels)
                class_loss += loss.item() * batch_size

                # Top-1
                preds_top1 = torch.argmax(logits, dim=1)
                class_correct_top1 += (preds_top1 == labels).sum().item()

                # Top-5
                _, preds_top5 = torch.topk(logits, k=5, dim=1)
                class_correct_top5 += (preds_top5 == labels.view(-1, 1)).sum().item()

            idx_start += batch_size

    elapsed_forward = time.time() - start_time
    print(f"Hoàn thành forward trong {elapsed_forward:.1f} giây ({num_samples / elapsed_forward:.1f} ảnh/giây).")

    # 3. Tính toán kết quả Classification
    classification_results = {}
    if has_head:
        avg_loss = class_loss / num_samples
        top1_acc = class_correct_top1 / num_samples
        top5_acc = class_correct_top5 / num_samples
        
        classification_results = {
            "loss": avg_loss,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc
        }
        
        print("\n" + "=" * 50)
        print("  KẾT QUẢ PHÂN LOẠI CHƯA TRAIN (CLOSED-SET)")
        print("=" * 50)
        print(f"  Cross Entropy Loss : {avg_loss:.4f}")
        print(f"  Top-1 Accuracy     : {top1_acc * 100:.2f}%")
        print(f"  Top-5 Accuracy     : {top5_acc * 100:.2f}%")
        print("=" * 50)

    # 4. Tính toán kết quả Verification (Open-set)
    print("\nBắt đầu sinh cặp ảnh để đánh giá Verification...")
    pair_idx1, pair_idx2, pair_labels = generate_verification_pairs(
        all_labels, 
        num_pairs=args.num_pairs, 
        seed=args.seed
    )

    print("Đang tính Cosine Similarity cho các cặp...")
    # Vì embeddings đã được chuẩn hóa L2, cosine similarity chính là dot product
    emb1 = all_embeddings[pair_idx1]
    emb2 = all_embeddings[pair_idx2]
    similarities = np.sum(emb1 * emb2, axis=1)

    # Tách similarities theo nhãn cặp
    genuine_scores = similarities[pair_labels == 1]
    impostor_scores = similarities[pair_labels == 0]

    # Tính ROC, AUC, EER
    fpr, tpr, thresholds = roc_curve(pair_labels, similarities)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)

    # Tính FAR / FRR tại các ngưỡng quan trọng
    # FAR (False Accept Rate): impostor vượt qua ngưỡng
    # FRR (False Reject Rate): genuine dưới ngưỡng
    print("\n" + "=" * 50)
    print("  KẾT QUẢ XÁC THỰC KHOẢNG CÁCH (OPEN-SET)")
    print("=" * 50)
    print(f"  ROC AUC            : {roc_auc * 100:.2f}%")
    print(f"  Equal Error Rate   : {eer * 100:.2f}%")
    print(f"  EER Threshold      : {eer_threshold:.4f}")
    print(f"  Genuine Sim Mean   : {np.mean(genuine_scores):.4f} (std: {np.std(genuine_scores):.4f})")
    print(f"  Impostor Sim Mean  : {np.mean(impostor_scores):.4f} (std: {np.std(impostor_scores):.4f})")
    print("=" * 50)

    # 5. Xuất báo cáo CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_csv_path = os.path.join(args.output_dir, f"vgg_eval_{timestamp}.csv")
    
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Evaluation Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["Checkpoint", args.checkpoint])
        writer.writerow(["Device Used", device])
        writer.writerow(["Total Evaluated Images", num_samples])
        writer.writerow(["Verification Pairs Count", args.num_pairs])
        writer.writerow(["ROC AUC", f"{roc_auc:.6f}"])
        writer.writerow(["Equal Error Rate (EER)", f"{eer:.6f}"])
        writer.writerow(["EER Threshold", f"{eer_threshold:.6f}"])
        writer.writerow(["Genuine Similarity Mean", f"{np.mean(genuine_scores):.6f}"])
        writer.writerow(["Genuine Similarity Std", f"{np.std(genuine_scores):.6f}"])
        writer.writerow(["Impostor Similarity Mean", f"{np.mean(impostor_scores):.6f}"])
        writer.writerow(["Impostor Similarity Std", f"{np.std(impostor_scores):.6f}"])
        
        if has_head:
            writer.writerow(["Classification Loss", f"{classification_results['loss']:.6f}"])
            writer.writerow(["Top-1 Accuracy", f"{classification_results['top1_acc']:.6f}"])
            writer.writerow(["Top-5 Accuracy", f"{classification_results['top5_acc']:.6f}"])
            
    print(f"\nĐã xuất báo cáo CSV tại: {report_csv_path}")

    # 6. Vẽ biểu đồ kết quả
    report_img_path = os.path.join(args.output_dir, f"vgg_eval_plots_{timestamp}.png")
    
    plt.figure(figsize=(18, 5))
    
    # Subplot 1: ROC Curve
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc * 100:.2f}%)")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER = {eer * 100:.2f}%")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Subplot 2: Similarity Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(genuine_scores, color="green", kde=True, label="Genuine Pairs", stat="density", bins=50, alpha=0.5)
    sns.histplot(impostor_scores, color="red", kde=True, label="Impostor Pairs", stat="density", bins=50, alpha=0.5)
    plt.axvline(eer_threshold, color="black", linestyle="--", lw=1.5, label=f"EER Threshold ({eer_threshold:.3f})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Cosine Similarity Score Distribution")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: FAR/FRR vs Threshold
    plt.subplot(1, 3, 3)
    # Lọc lấy một lượng threshold vừa phải để vẽ biểu đồ mượt mà
    threshold_sweep = np.linspace(-1.0, 1.0, 500)
    far_sweep = [np.mean(impostor_scores >= th) for th in threshold_sweep]
    frr_sweep = [np.mean(genuine_scores < th) for th in threshold_sweep]
    
    plt.plot(threshold_sweep, far_sweep, color="red", lw=2, label="FAR")
    plt.plot(threshold_sweep, frr_sweep, color="blue", lw=2, label="FRR")
    plt.axvline(eer_threshold, color="black", linestyle="--", lw=1.5, label=f"EER Threshold ({eer_threshold:.3f})")
    plt.xlim([0.0, 1.0])  # Cosine similarity thường dương đối với ảnh khuôn mặt
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR / FRR vs Threshold")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(report_img_path, dpi=150)
    plt.close()
    
    print(f"Đã xuất các biểu đồ chẩn đoán tại: {report_img_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
