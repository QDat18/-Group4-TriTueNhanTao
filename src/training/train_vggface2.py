import os
import time
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.amp import autocast, GradScaler

from src import config
from src.models.face_recognition_model import FaceEmbeddingNet
from src.models.arcface_head import ArcMarginProduct
from src.datasets.dataset_vggface2 import VGGFace2Dataset
from src.utils.transforms import get_train_transform, get_val_transform
from src.utils.warmup_scheduler import get_warmup_cosine_scheduler
from src.utils.logger import CSVLogger


torch.backends.cudnn.benchmark = True

# =====================================================================
# Đường dẫn checkpoint
# =====================================================================
# best  → chỉ lưu khi val_loss tốt nhất
# latest → lưu SAU MỖI EPOCH để có thể resume bất cứ lúc nào
# =====================================================================


def _best_ckpt_path() -> str:
    return os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)


def _latest_ckpt_path() -> str:
    name = config.CHECKPOINT_NAME.replace(".pth", "_latest.pth")
    return os.path.join(config.CHECKPOINT_DIR, name)


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


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024

    return 0.0


# =====================================================================
# Checkpoint: Save & Load
# =====================================================================

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
    """Lưu checkpoint đầy đủ (có thể resume) + bản lite (chỉ backbone)."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- Full checkpoint (có thể resume training) ---
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

    # --- Lite checkpoint (chỉ backbone, nhẹ ~3x) ---
    lite_path = save_path.replace(".pth", "_lite.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, lite_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    arcface_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: str,
) -> Tuple[int, float]:
    """
    Load checkpoint và khôi phục toàn bộ trạng thái training.

    Returns:
        resume_epoch: epoch tiếp theo cần chạy (0-indexed)
        best_val_loss: giá trị best_val_loss đã lưu
    """

    print(f"\n{'=' * 70}")
    print(f"RESUMING FROM CHECKPOINT: {path}")
    print(f"{'=' * 70}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    arcface_head.load_state_dict(checkpoint["arcface_head_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    resume_epoch = checkpoint["epoch"]  # epoch đã hoàn thành → tiếp từ epoch này
    best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))

    print(f"  Đã train xong epoch  : {resume_epoch}")
    print(f"  Sẽ tiếp tục từ epoch : {resume_epoch + 1}")
    print(f"  Best val_loss        : {best_val_loss:.4f}")
    print(f"  Train loss (cuối)    : {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val loss (cuối)      : {checkpoint.get('val_loss', 'N/A')}")
    print(f"{'=' * 70}\n")

    return resume_epoch, best_val_loss


def find_latest_checkpoint() -> Optional[str]:
    """
    Tìm checkpoint mới nhất để resume.
    Ưu tiên: latest > best (latest luôn là epoch gần nhất).
    """
    latest_path = _latest_ckpt_path()
    best_path = _best_ckpt_path()

    if os.path.exists(latest_path):
        return latest_path
    elif os.path.exists(best_path):
        return best_path

    return None


# =====================================================================
# Train / Validate one epoch
# =====================================================================

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

        with autocast('cuda', enabled=torch.cuda.is_available()):
            embeddings = model(images)
            logits = arcface_head(embeddings, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Unscale trước khi tính gradient norm để giá trị grad_norm là thật
        scaler.unscale_(optimizer)
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
    """
    Đánh giá model trên tập validation.
    Không dùng AMP để đảm bảo kết quả chính xác.
    """
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


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    print("=" * 70)
    print("TRAIN VGGFACE2 WITH WARM-UP + ARC FACE + AMP")
    print("Split: Train / Val / Test = 80% / 10% / 10%")
    print("Hỗ trợ RESUME: tự động tiếp tục nếu đã có checkpoint")
    print("=" * 70)

    device = config.DEVICE
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("AMP: enabled")
    else:
        print("AMP: disabled")

    # =========================
    # CONFIG
    # =========================

    split_ratio = getattr(config, "SPLIT_RATIO", (0.8, 0.1, 0.1))

    max_classes = config.MAX_CLASSES if config.USE_SUBSET else None
    max_images = config.MAX_IMAGES_PER_CLASS if config.USE_SUBSET else None

    # =========================
    # 1. Load datasets (Train + Val)
    # =========================

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

    # num_classes giống nhau giữa các split
    num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
    )

    print(f"\nTrain images     : {len(train_dataset)}")
    print(f"Val images       : {len(val_dataset)}")
    print(f"Total identities : {num_classes}")
    print(f"Split ratio      : {split_ratio[0]:.0%} / {split_ratio[1]:.0%} / {split_ratio[2]:.0%}")
    print(f"Batch size       : {config.BATCH_SIZE}")
    print(f"Total epochs     : {config.TOTAL_EPOCHS}")
    print(f"Warm-up epochs   : {config.WARMUP_EPOCHS}")

    # =========================
    # 2. Build model
    # =========================

    # pretrained=True vì đây là lần training đầu tiên từ ImageNet weights
    model = FaceEmbeddingNet(
        embedding_size=config.EMBEDDING_SIZE,
        pretrained=True
    ).to(device)

    arcface_head = ArcMarginProduct(
        embedding_size=config.EMBEDDING_SIZE,
        num_classes=num_classes,
        scale=64.0,
        margin=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=config.BASE_LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.TOTAL_EPOCHS,
        base_lr=config.BASE_LR,
        min_lr=config.MIN_LR,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())
    logger = CSVLogger(config.TRAIN_LOG_PATH)

    best_val_loss = float("inf")
    start_epoch = 0

    # =========================
    # 2.5. TỰ ĐỘNG RESUME NẾU CÓ CHECKPOINT
    # =========================

    resume_ckpt = find_latest_checkpoint()

    if resume_ckpt is not None:
        start_epoch, best_val_loss = load_checkpoint(
            path=resume_ckpt,
            model=model,
            arcface_head=arcface_head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
    else:
        print("\nKhông tìm thấy checkpoint → bắt đầu training từ đầu.\n")

    # Kiểm tra đã train xong chưa
    if start_epoch >= config.TOTAL_EPOCHS:
        print(f"\nĐã hoàn thành {config.TOTAL_EPOCHS} epochs. Không cần train thêm.")
        print(f"Best checkpoint: {_best_ckpt_path()}")
        return

    remaining = config.TOTAL_EPOCHS - start_epoch
    print(f"\nSẽ train {remaining} epoch(s) còn lại "
          f"(epoch {start_epoch + 1} → {config.TOTAL_EPOCHS})\n")

    # =========================
    # 3. Training loop (có thể resume)
    # =========================

    checkpoint_path = _best_ckpt_path()
    latest_path = _latest_ckpt_path()

    for epoch in range(start_epoch, config.TOTAL_EPOCHS):
        epoch_start_time = time.time()

        phase = "WARM-UP" if epoch < config.WARMUP_EPOCHS else "COSINE-DECAY"
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train ---
        train_loss, train_acc, emb_norm, grad_norm, train_batches = train_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        # --- Validate ---
        val_loss, val_acc, val_batches = validate_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        gpu_memory = get_gpu_memory_mb()

        # -------------------------------------------------------
        # Lưu checkpoint BEST (khi val_loss cải thiện)
        # -------------------------------------------------------
        checkpoint_saved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss

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
                save_path=checkpoint_path,
            )

            checkpoint_saved = True

            # In kích thước file
            full_size = os.path.getsize(checkpoint_path) / 1024 / 1024
            lite_path = checkpoint_path.replace(".pth", "_lite.pth")
            lite_size = os.path.getsize(lite_path) / 1024 / 1024
            print(f"  >> Best: {full_size:.1f} MB | Lite: {lite_size:.1f} MB")

        # -------------------------------------------------------
        # Lưu checkpoint LATEST (mỗi epoch, để resume khi thoát nhầm)
        # -------------------------------------------------------
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
            save_path=latest_path,
        )

        logger.log({
            "epoch": epoch + 1,
            "phase": phase,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "embedding_norm": emb_norm,
            "gradient_norm": grad_norm,
            "train_batches": train_batches,
            "val_batches": val_batches,
            "epoch_time_sec": round(epoch_time, 2),
            "device": device,
            "gpu_memory_mb": round(gpu_memory, 2),
            "checkpoint_saved": checkpoint_saved,
        })

        print("=" * 70)
        print(f"Epoch          : {epoch + 1}/{config.TOTAL_EPOCHS}")
        print(f"Phase          : {phase}")
        print(f"Learning Rate  : {current_lr:.8f}")
        print(f"Train Loss     : {train_loss:.4f}   |  Val Loss     : {val_loss:.4f}")
        print(f"Train Accuracy : {train_acc * 100:.2f}%  |  Val Accuracy : {val_acc * 100:.2f}%")
        print(f"Embedding Norm : {emb_norm:.4f}")
        print(f"Gradient Norm  : {grad_norm:.4f}")
        print(f"Batches        : {train_batches} train / {val_batches} val")
        print(f"Epoch Time     : {epoch_time:.2f} sec")
        print(f"GPU Memory     : {gpu_memory:.2f} MB")
        print(f"Checkpoint     : {'✅ Saved (best val_loss)' if checkpoint_saved else '— Not saved (best)'}")
        print(f"Latest saved   : ✅ {latest_path}")
        print("=" * 70)

    # =========================
    # 4. Xóa latest checkpoint khi train xong hoàn toàn
    # =========================
    if os.path.exists(latest_path):
        os.remove(latest_path)
        lite_latest = latest_path.replace(".pth", "_lite.pth")
        if os.path.exists(lite_latest):
            os.remove(lite_latest)
        print(f"\nĐã xóa latest checkpoint (training hoàn tất).")

    print("\nTraining finished.")
    print(f"Best val_loss  : {best_val_loss:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log   : {config.TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()