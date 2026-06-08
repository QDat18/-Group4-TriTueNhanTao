import os
import time
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

from src import config
from src.models.face_recognition_model import FaceEmbeddingNet
from src.models.arcface_head import ArcMarginProduct
from src.datasets.dataset_rmfrd import RMFRDDataset
from PIL import Image
import numpy as np
from src.utils.transforms import get_train_transform, get_val_transform
from src.utils.warmup_scheduler import get_warmup_cosine_scheduler
from src.utils.logger import CSVLogger


torch.backends.cudnn.benchmark = True


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / max(1, total)


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def save_checkpoint(
    model: nn.Module,
    arcface_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    val_acc: float,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "arcface_head_state_dict": arcface_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "val_acc": val_acc,
    }, save_path)

    lite_path = save_path.replace(".pth", "_lite.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "val_acc": val_acc,
    }, lite_path)


def load_vggface2_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint VGGFace2: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(f"Loaded VGGFace2 checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint train_loss: {checkpoint.get('train_loss', checkpoint.get('loss', 'unknown'))}")
    print(f"Checkpoint val_loss: {checkpoint.get('val_loss', 'N/A')}")


def train_one_epoch(
    model: nn.Module,
    arcface_head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: str,
) -> Tuple[float, float, int, float]:
    model.train()
    arcface_head.train()

    # Freeze BN running statistics to prevent drift during fine-tuning
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training RMFRD", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            embeddings = model(images)
            logits = arcface_head(embeddings, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface_head.parameters()),
            max_norm=5.0
        )

        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(logits.detach(), labels)

        total_loss += loss.item()
        total_acc += acc

        if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
            total_grad_norm += grad_norm.item()

        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc * 100:.2f}%"
        })

    return (
        total_loss / max(1, num_batches),
        total_acc / max(1, num_batches),
        num_batches,
        total_grad_norm / max(1, num_batches),
    )


@torch.no_grad()
def validate_model(model: nn.Module, transform, device: str) -> float:
    model.eval()

    txt_path = os.path.join("dataset/RWFRD", "test_identities.txt")
    if not os.path.exists(txt_path):
        print(f"[WARNING] Không tìm thấy file validation split: {txt_path}. Bỏ qua validate.")
        return 0.0

    with open(txt_path, "r", encoding="utf-8") as f:
        selected_ids = [line.strip() for line in f if line.strip()]

    from src.evaluation.evaluate_afdb_masked import AFDB_FACE_ROOT, AFDB_MASKED_ROOT

    if not os.path.exists(AFDB_FACE_ROOT) or not os.path.exists(AFDB_MASKED_ROOT):
        print("[WARNING] Thư mục AFDB không tồn tại. Bỏ qua validate.")
        return 0.0

    face_ids = {
        d for d in os.listdir(AFDB_FACE_ROOT)
        if os.path.isdir(os.path.join(AFDB_FACE_ROOT, d))
    }

    masked_ids = {
        d for d in os.listdir(AFDB_MASKED_ROOT)
        if os.path.isdir(os.path.join(AFDB_MASKED_ROOT, d))
    }

    common_ids = sorted([i for i in selected_ids if i in (face_ids & masked_ids)])

    if len(common_ids) == 0:
        print("[WARNING] Không tìm thấy danh tính validation chung. Bỏ qua validate.")
        return 0.0

    val_ids = common_ids[:30]

    def get_embedding(img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        emb = model(tensor).squeeze(0).detach().cpu().numpy()
        emb = emb / max(np.linalg.norm(emb), 1e-12)

        return emb

    gallery_embeddings = {}

    for identity in val_ids:
        identity_dir = os.path.join(AFDB_FACE_ROOT, identity)

        img_files = [
            os.path.join(identity_dir, f)
            for f in sorted(os.listdir(identity_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ][:3]

        embs = []

        for img_path in img_files:
            try:
                embs.append(get_embedding(img_path))
            except Exception:
                continue

        if len(embs) > 0:
            avg_emb = np.mean(embs, axis=0)
            avg_emb = avg_emb / max(np.linalg.norm(avg_emb), 1e-12)
            gallery_embeddings[identity] = avg_emb

    total = 0
    correct = 0

    for identity in val_ids:
        if identity not in gallery_embeddings:
            continue

        query_dir = os.path.join(AFDB_MASKED_ROOT, identity)

        img_files = [
            os.path.join(query_dir, f)
            for f in sorted(os.listdir(query_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ][:2]

        for img_path in img_files:
            try:
                q_emb = get_embedding(img_path)
            except Exception:
                continue

            best_id = None
            best_score = -1.0

            for g_id, g_emb in gallery_embeddings.items():
                score = float(np.dot(q_emb, g_emb))

                if score > best_score:
                    best_score = score
                    best_id = g_id

            if best_id == identity:
                correct += 1

            total += 1

    model.train()

    return correct / max(1, total)


def main() -> None:
    print("=" * 70)
    print("FINE-TUNE RMFRD FROM VGGFACE2 CHECKPOINT")
    print("=" * 70)

    device = config.DEVICE
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("AMP: enabled")
    else:
        print("AMP: disabled")

    rmfrd_roots: List[str] = getattr(config, "RMFRD_ROOTS", [
        "dataset/RWFRD/AFDB_masked_face_dataset/AFDB_masked_face_dataset"
    ])

    rmfrd_use_subset = getattr(config, "RMFRD_USE_SUBSET", False)
    rmfrd_max_classes = getattr(config, "RMFRD_MAX_CLASSES", None)
    rmfrd_max_images_per_class = getattr(config, "RMFRD_MAX_IMAGES_PER_CLASS", None)

    batch_size = getattr(config, "RMFRD_BATCH_SIZE", 64)
    num_workers = getattr(config, "RMFRD_NUM_WORKERS", 4)

    total_epochs = getattr(config, "RMFRD_TOTAL_EPOCHS", 15)
    warmup_epochs = getattr(config, "RMFRD_WARMUP_EPOCHS", 5)

    base_lr = getattr(config, "RMFRD_BASE_LR", 1e-5)
    min_lr = getattr(config, "RMFRD_MIN_LR", 1e-6)
    weight_decay = getattr(config, "RMFRD_WEIGHT_DECAY", 5e-4)

    arcface_scale = getattr(config, "RMFRD_ARCFACE_SCALE", 30.0)
    arcface_margin = getattr(config, "RMFRD_ARCFACE_MARGIN", 0.3)

    vgg_checkpoint_path = getattr(
        config,
        "VGG_CHECKPOINT_PATH",
        "checkpoints/arcface_vggface2_warmup.pth"
    )

    rmfrd_checkpoint_dir = getattr(config, "RMFRD_CHECKPOINT_DIR", "checkpoints")
    rmfrd_checkpoint_name = getattr(
        config,
        "RMFRD_CHECKPOINT_NAME",
        "arcface_rmfrd_finetuned_s30_m03.pth"
    )

    rmfrd_log_path = getattr(
        config,
        "RMFRD_LOG_PATH",
        "logs/train_rmfrd_s30_m03_log.csv"
    )

    rmfrd_split_ratio = getattr(config, "RMFRD_SPLIT_RATIO", 0.8)

    dataset = RMFRDDataset(
        root_dir=rmfrd_roots,
        transform=get_train_transform(),
        max_classes=rmfrd_max_classes if rmfrd_use_subset else None,
        max_images_per_class=rmfrd_max_images_per_class if rmfrd_use_subset else None,
        split="train",
        split_ratio=rmfrd_split_ratio,
    )

    num_classes = len(dataset.class_to_idx)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    print(f"Total RMFRD images     : {len(dataset)}")
    print(f"Total RMFRD identities : {num_classes}")
    print(f"Batch size             : {batch_size}")
    print(f"Total epochs           : {total_epochs}")
    print(f"Warm-up epochs         : {warmup_epochs}")
    print(f"ArcFace scale          : {arcface_scale}")
    print(f"ArcFace margin         : {arcface_margin}")
    print(f"Base LR                : {base_lr}")

    model = FaceEmbeddingNet(
        embedding_size=config.EMBEDDING_SIZE,
        pretrained=False
    ).to(device)

    load_vggface2_checkpoint(
        model=model,
        checkpoint_path=vgg_checkpoint_path,
        device=device
    )

    arcface_head = ArcMarginProduct(
        embedding_size=config.EMBEDDING_SIZE,
        num_classes=num_classes,
        scale=arcface_scale,
        margin=arcface_margin,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": base_lr * 0.1},
        {"params": arcface_head.parameters(), "lr": base_lr},
    ], weight_decay=weight_decay)

    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())
    logger = CSVLogger(rmfrd_log_path)

    best_val_acc = -1.0
    val_transform = get_val_transform()

    checkpoint_path = os.path.join(
        rmfrd_checkpoint_dir,
        rmfrd_checkpoint_name
    )

    for param in model.parameters():
        param.requires_grad = False

    for epoch in range(total_epochs):
        epoch_start = time.time()

        if epoch == warmup_epochs:
            print("Unfreezing backbone for joint fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True

        if epoch >= warmup_epochs:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[1]["lr"] * 0.1
        else:
            optimizer.param_groups[0]["lr"] = 0.0

        phase = "WARM-UP" if epoch < warmup_epochs else "COSINE-DECAY"
        current_lr = optimizer.param_groups[1]["lr"]

        train_loss, train_acc, num_batches, train_grad_norm = train_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        scheduler.step()

        val_acc = validate_model(
            model=model,
            transform=val_transform,
            device=device,
        )

        epoch_time = time.time() - epoch_start
        gpu_memory = get_gpu_memory_mb()

        checkpoint_saved = False

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                model=model,
                arcface_head=arcface_head,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                loss=train_loss,
                val_acc=val_acc,
                save_path=checkpoint_path,
            )

            checkpoint_saved = True

        # Early stopping check: Stop if best validation accuracy is below 18.52% starting from epoch 6
        if epoch + 1 >= 6 and best_val_acc < 0.1852:
            print(f"\n[Early Stopping] Best validation accuracy {best_val_acc * 100:.2f}% did not exceed 18.52% by epoch {epoch + 1}. Stopping training.")
            break

        logger.log({
            "epoch": epoch + 1,
            "phase": phase,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": 0.0,
            "val_accuracy": val_acc,
            "embedding_norm": 1.0,
            "gradient_norm": train_grad_norm,
            "train_batches": num_batches,
            "val_batches": 30,
            "epoch_time_sec": round(epoch_time, 2),
            "device": device,
            "gpu_memory_mb": round(gpu_memory, 2),
            "checkpoint_saved": checkpoint_saved,
        })

        print("=" * 70)
        print(f"Epoch          : {epoch + 1}/{total_epochs}")
        print(f"Phase          : {phase}")
        print(f"Learning Rate  : {current_lr:.8f}")
        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_acc * 100:.2f}%")
        print(f"Val Accuracy   : {val_acc * 100:.2f}% (Best: {best_val_acc * 100:.2f}%)")
        print(f"Gradient Norm  : {train_grad_norm:.4f}")
        print(f"Num Batches    : {num_batches}")
        print(f"Epoch Time     : {epoch_time:.2f} sec")
        print(f"GPU Memory     : {gpu_memory:.2f} MB")
        print(f"Checkpoint     : {'Saved (best val_acc)' if checkpoint_saved else 'Not saved'}")
        print("=" * 70)

    print("RMFRD fine-tuning finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log   : {rmfrd_log_path}")


if __name__ == "__main__":
    main()