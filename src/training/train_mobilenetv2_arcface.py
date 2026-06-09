"""
Script huấn luyện mô hình nhận diện khuôn mặt sử dụng MobileNetV2 + ArcFace.
Tự động lưu checkpoint riêng biệt và cập nhật logs đầy đủ.

Cấu hình huấn luyện:
    - Backbone: MobileNetV2
    - Embedding: 512
    - Loss: ArcFace
    - Input: 112x112
    - Epoch: 8 (Warm-up 2 epochs)
    - Batch size: 64
    - LR: 1e-4
    - Optimizer: AdamW
"""

import os
import time
import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from src import config
from src.models.face_recognition_model import MobileNetV2FaceEmbeddingNet
from src.models.arcface_head import ArcMarginProduct
from src.datasets.dataset_vggface2 import VGGFace2Dataset
from src.utils.transforms import get_train_transform, get_val_transform
from src.utils.warmup_scheduler import get_warmup_cosine_scheduler
from src.utils.logger import CSVLogger

# Kích hoạt benchmark cudnn giúp tăng tốc độ
torch.backends.cudnn.benchmark = True


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_gradient_norm(model: nn.Module, head: nn.Module) -> float:
    total_norm = 0.0
    for module in [model, head]:
        for param in module.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def save_checkpoint(
    model: nn.Module,
    arcface_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    save_path: str,
) -> None:
    """Lưu checkpoint đầy đủ (có thể resume) + bản lite (chỉ chứa backbone)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- Full checkpoint ---
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "arcface_head_state_dict": arcface_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }, save_path)

    # --- Lite checkpoint (chỉ lưu backbone phục vụ deploy/inference, nhẹ hơn) ---
    lite_path = save_path.replace(".pth", "_lite.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, lite_path)
    print(f"[INFO] Checkpoint saved to: {save_path} (Lite version: {lite_path})")


def train_one_epoch(
    model: nn.Module,
    arcface_head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: str,
) -> Tuple[float, float, float, float, int]:
    model.train()
    arcface_head.train()

    total_loss = 0.0
    total_acc = 0.0
    total_embedding_norm = 0.0
    total_gradient_norm = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Sử dụng chế độ chính xác hỗn hợp AMP trên GPU
        with autocast('cuda', enabled=torch.cuda.is_available()):
            embeddings = model(images)
            logits = arcface_head(embeddings, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Tránh bùng nổ gradient
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface_head.parameters()),
            max_norm=5.0,
        )

        grad_norm = calculate_gradient_norm(model, arcface_head)

        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(logits.detach(), labels)
        emb_norm = embeddings.detach().norm(dim=1).mean().item()

        total_loss += loss.item()
        total_acc += acc
        total_embedding_norm += emb_norm
        total_gradient_norm += grad_norm
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc * 100:.2f}%",
        })

    avg_loss = total_loss / max(1, num_batches)
    avg_accuracy = total_acc / max(1, num_batches)
    avg_embedding_norm = total_embedding_norm / max(1, num_batches)
    avg_gradient_norm = total_gradient_norm / max(1, num_batches)

    return avg_loss, avg_accuracy, avg_embedding_norm, avg_gradient_norm, num_batches


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    arcface_head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, int]:
    model.eval()
    arcface_head.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        logits = arcface_head(embeddings, labels)
        loss = criterion(logits, labels)

        acc = calculate_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

        progress_bar.set_postfix({
            "val_loss": f"{loss.item():.4f}",
            "val_acc": f"{acc * 100:.2f}%",
        })

    avg_loss = total_loss / max(1, num_batches)
    avg_acc = total_acc / max(1, num_batches)

    return avg_loss, avg_acc, num_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 + ArcFace Model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Number of warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--embedding-size", type=int, default=512, help="Feature embedding dimension")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (0 for Windows stability)")
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING MOBILENETV2 + ARCFACE ON VGGFACE2")
    print(f"Total Epochs   : {args.epochs}")
    print(f"Warm-up Epochs : {args.warmup_epochs}")
    print(f"Batch Size     : {args.batch_size}")
    print(f"Learning Rate  : {args.lr}")
    print(f"Embedding Size : {args.embedding_size}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU Model: {torch.cuda.get_device_name(0)}")

    # Setup paths riêng biệt cho MobileNetV2 để tránh đè checkpoint ResNet-50
    checkpoint_dir = config.CHECKPOINT_DIR
    best_ckpt_path = os.path.join(checkpoint_dir, "arcface_mobilenetv2.pth")
    latest_ckpt_path = os.path.join(checkpoint_dir, "arcface_mobilenetv2_latest.pth")
    log_path = "outputs/logs/train_mobilenetv2.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 1. Load Datasets
    split_ratio = getattr(config, "SPLIT_RATIO", (0.8, 0.1, 0.1))
    max_classes = 1000  # Limit to 1000 classes for faster MobileNetV2 training
    max_images = 100    # Limit to 100 images per class for faster MobileNetV2 training

    train_dataset = VGGFace2Dataset(
        root_dir=config.VGGFACE2_ROOT,
        transform=get_train_transform(),
        max_classes=max_classes,
        max_images_per_class=max_images,
        split="train",
        split_ratio=split_ratio,
    )

    val_dataset = VGGFace2Dataset(
        root_dir=config.VGGFACE2_ROOT,
        transform=get_val_transform(),
        max_classes=max_classes,
        max_images_per_class=max_images,
        split="val",
        split_ratio=split_ratio,
    )

    num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[INFO] Train samples     : {len(train_dataset)}")
    print(f"[INFO] Val samples       : {len(val_dataset)}")
    print(f"[INFO] Target identities : {num_classes}")

    # 2. Build Model
    model = MobileNetV2FaceEmbeddingNet(
        embedding_size=args.embedding_size,
        pretrained=True
    ).to(device)

    arcface_head = ArcMarginProduct(
        embedding_size=args.embedding_size,
        num_classes=num_classes,
        scale=64.0,
        margin=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=1e-6,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())
    logger = CSVLogger(log_path)

    best_val_loss = float("inf")
    start_epoch = 0

    # Auto Resume from latest checkpoint if exists
    if os.path.exists(latest_ckpt_path):
        print(f"\n[INFO] Found latest checkpoint. Resuming from {latest_ckpt_path}...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        arcface_head.load_state_dict(checkpoint["arcface_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[INFO] Resumed at epoch: {start_epoch + 1}")

    # 3. Training Loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch + 1}/{args.epochs}] LR: {current_lr:.6f}")

        # Train one epoch
        train_loss, train_acc, emb_norm, grad_norm, train_batches = train_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        # Validate one epoch
        val_loss, val_acc, val_batches = validate_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - epoch_start_time
        scheduler.step()

        print(f"Result - Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Result - Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc * 100:.2f}%")
        print(f"Result - Grad Norm : {grad_norm:.4f} | Time     : {epoch_time:.1f}s")

        # Save latest checkpoint
        save_checkpoint(
            model=model,
            arcface_head=arcface_head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            save_path=latest_ckpt_path,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"[INFO] New best val loss: {best_val_loss:.4f}. Saving best checkpoint...")
            save_checkpoint(
                model=model,
                arcface_head=arcface_head,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                save_path=best_ckpt_path,
            )

        # Log to CSV
        logger.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": current_lr,
            "grad_norm": grad_norm,
            "epoch_duration_seconds": epoch_time,
            "train_batches": train_batches,
            "val_batches": val_batches
        })

    print(f"\n[SUCCESS] Training finished! Best checkpoint saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
