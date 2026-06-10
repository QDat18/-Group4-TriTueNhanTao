"""
Train MobileNetV2 + ArcFace on VGGFace2.
"""

import os
import time
import argparse
from typing import Tuple

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


torch.backends.cudnn.benchmark = True


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / max(1, total)


def save_checkpoint(
    model: nn.Module,
    arcface_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "arcface_head_state_dict": arcface_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        save_path,
    )

    lite_path = save_path.replace(".pth", "_lite.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        lite_path,
    )

    print(f"[INFO] Checkpoint saved to: {save_path}")
    print(f"[INFO] Lite checkpoint saved to: {lite_path}")


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

        with autocast("cuda", enabled=torch.cuda.is_available()):
            embeddings = model(images)
            logits = arcface_head(embeddings, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface_head.parameters()),
            max_norm=5.0,
        )

        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(logits.detach(), labels)
        emb_norm = embeddings.detach().norm(dim=1).mean().item()

        total_loss += loss.item()
        total_acc += acc
        total_embedding_norm += emb_norm
        total_gradient_norm += float(grad_norm)
        num_batches += 1

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc * 100:.2f}%",
                "grad": f"{float(grad_norm):.2f}",
            }
        )

    return (
        total_loss / max(1, num_batches),
        total_acc / max(1, num_batches),
        total_embedding_norm / max(1, num_batches),
        total_gradient_norm / max(1, num_batches),
        num_batches,
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    arcface_head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    desc: str = "Validating",
) -> Tuple[float, float, int]:
    model.eval()
    arcface_head.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=desc, leave=False)

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

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc * 100:.2f}%",
            }
        )

    return (
        total_loss / max(1, num_batches),
        total_acc / max(1, num_batches),
        num_batches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 + ArcFace on VGGFace2"
    )

    parser.add_argument("--epochs", type=int, default=config.TOTAL_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=config.WARMUP_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.BASE_LR)
    parser.add_argument("--min-lr", type=float, default=config.MIN_LR)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--embedding-size", type=int, default=config.EMBEDDING_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)

    parser.add_argument("--max-classes", type=int, default=config.MAX_CLASSES)
    parser.add_argument("--max-images", type=int, default=config.MAX_IMAGES_PER_CLASS)

    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING MOBILENETV2 + ARCFACE ON VGGFACE2")
    print(f"Total Epochs   : {args.epochs}")
    print(f"Warm-up Epochs : {args.warmup_epochs}")
    print(f"Batch Size     : {args.batch_size}")
    print(f"Learning Rate  : {args.lr}")
    print(f"Min LR         : {args.min_lr}")
    print(f"Weight Decay   : {args.weight_decay}")
    print(f"Embedding Size : {args.embedding_size}")
    print(f"Max Classes    : {args.max_classes}")
    print(f"Max Images/ID  : {args.max_images}")
    print(f"Resume         : {args.resume}")
    print("=" * 70)

    device = config.DEVICE if hasattr(config, "DEVICE") else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Training device: {device}")

    if torch.cuda.is_available():
        print(f"[INFO] GPU Model: {torch.cuda.get_device_name(0)}")

    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_ckpt_path = os.path.join(checkpoint_dir, "arcface_mobilenetv2.pth")
    latest_ckpt_path = os.path.join(checkpoint_dir, "arcface_mobilenetv2_latest.pth")

    log_path = "outputs/logs/train_mobilenetv2.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    split_ratio = getattr(config, "SPLIT_RATIO", (0.8, 0.1, 0.1))

    train_dataset = VGGFace2Dataset(
        root_dir=config.VGGFACE2_ROOT,
        transform=get_train_transform(),
        max_classes=args.max_classes,
        max_images_per_class=args.max_images,
        split="train",
        split_ratio=split_ratio,
    )

    val_dataset = VGGFace2Dataset(
        root_dir=config.VGGFACE2_ROOT,
        transform=get_val_transform(),
        max_classes=args.max_classes,
        max_images_per_class=args.max_images,
        split="val",
        split_ratio=split_ratio,
    )

    test_dataset = VGGFace2Dataset(
        root_dir=config.VGGFACE2_ROOT,
        transform=get_val_transform(),
        max_classes=args.max_classes,
        max_images_per_class=args.max_images,
        split="test",
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[INFO] Split ratio       : {split_ratio}")
    print(f"[INFO] Train samples     : {len(train_dataset)}")
    print(f"[INFO] Val samples       : {len(val_dataset)}")
    print(f"[INFO] Test samples      : {len(test_dataset)}")
    print(f"[INFO] Target identities : {num_classes}")

    model = MobileNetV2FaceEmbeddingNet(
        embedding_size=args.embedding_size,
        pretrained=True,
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
        weight_decay=args.weight_decay,
    )

    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())
    logger = CSVLogger(log_path)

    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"\n[INFO] Resuming from checkpoint: {latest_ckpt_path}")

        checkpoint = torch.load(
            latest_ckpt_path,
            map_location=device,
            weights_only=False,
        )

        ckpt_classes = checkpoint.get("num_classes", None)
        if ckpt_classes is not None and ckpt_classes != num_classes:
            raise ValueError(
                f"Checkpoint num_classes={ckpt_classes}, "
                f"but current dataset num_classes={num_classes}."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        arcface_head.load_state_dict(checkpoint["arcface_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"[INFO] Resumed at epoch: {start_epoch + 1}")

    else:
        print("[INFO] Training from scratch.")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n[Epoch {epoch + 1}/{args.epochs}] LR: {current_lr:.6f}")

        train_loss, train_acc, emb_norm, grad_norm, train_batches = train_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        val_loss, val_acc, val_batches = evaluate_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            desc="Validating",
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start_time

        is_best = val_loss < best_val_loss

        print(f"Result - Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Result - Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc * 100:.2f}%")
        print(f"Result - Emb Norm  : {emb_norm:.4f}")
        print(f"Result - Grad Norm : {grad_norm:.4f} | Time: {epoch_time:.1f}s")

        save_checkpoint(
            model=model,
            arcface_head=arcface_head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=min(best_val_loss, val_loss),
            save_path=latest_ckpt_path,
        )

        if is_best:
            best_val_loss = val_loss
            print(f"[INFO] New best val loss: {best_val_loss:.4f}")

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

        logger.log(
            {
                "epoch": epoch + 1,
                "phase": "TRAIN_VAL",
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "embedding_norm": emb_norm,
                "gradient_norm": grad_norm,
                "train_batches": train_batches,
                "val_batches": val_batches,
                "epoch_time_sec": epoch_time,
                "device": str(device),
                "gpu_memory_mb": (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "checkpoint_saved": "yes" if is_best else "no",
            }
        )

    print("\n[INFO] Loading best checkpoint for final test evaluation...")

    best_checkpoint = torch.load(
        best_ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])
    arcface_head.load_state_dict(best_checkpoint["arcface_head_state_dict"])

    test_loss, test_acc, test_batches = evaluate_one_epoch(
        model=model,
        arcface_head=arcface_head,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        desc="Testing",
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULT - MOBILENETV2 + ARCFACE")
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Acc     : {test_acc * 100:.2f}%")
    print(f"Test Batches : {test_batches}")
    print("=" * 70)

    logger.log(
        {
            "epoch": args.epochs,
            "phase": "TEST",
            "learning_rate": 0.0,
            "train_loss": "",
            "train_accuracy": "",
            "val_loss": "",
            "val_accuracy": "",
            "embedding_norm": "",
            "gradient_norm": "",
            "train_batches": "",
            "val_batches": test_batches,
            "epoch_time_sec": "",
            "device": str(device),
            "gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else 0.0
            ),
            "checkpoint_saved": "final_test",
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
    )

    print(f"\n[SUCCESS] Training finished!")
    print(f"[SUCCESS] Best checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()