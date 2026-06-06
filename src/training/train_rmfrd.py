import os
import time
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

from src import config
from src.datasets.dataset_rmfrd import RMFRDDataset
from src.utils.transforms import get_train_transform
from src.utils.warmup_scheduler import get_warmup_cosine_scheduler
from src.utils.logger import CSVLogger


torch.backends.cudnn.benchmark = True


class FaceEmbeddingNet(nn.Module):
    """
    Backbone trích xuất đặc trưng khuôn mặt.
    Dùng cùng kiến trúc với giai đoạn VGGFace2 để load checkpoint.
    """

    def __init__(self, embedding_size: int = 512) -> None:
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_layer = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)

        embeddings = self.embedding_layer(features)
        embeddings = self.bn(embeddings)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ArcMarginProduct(nn.Module):
    """
    ArcFace Head cho RMFRD.
    Head này được khởi tạo lại vì số class RMFRD khác VGGFace2.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight)
        )

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        return logits


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)

    return correct / total


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024

    return 0.0


def save_checkpoint(
    model: nn.Module,
    arcface_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    loss: float,
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
    }, save_path)


def load_vggface2_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> None:
    """
    Load backbone đã fine-tune từ VGGFace2.
    Chỉ load model_state_dict, không load ArcFace Head vì số class khác.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint VGGFace2: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(f"Loaded VGGFace2 checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")


def train_one_epoch(
    model: nn.Module,
    arcface_head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: str,
) -> Tuple[float, float, int]:
    model.train()
    arcface_head.train()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training RMFRD", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            embeddings = model(images)
            logits = arcface_head(embeddings, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(logits.detach(), labels)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc * 100:.2f}%"
        })

    avg_loss = total_loss / max(1, num_batches)
    avg_acc = total_acc / max(1, num_batches)

    return avg_loss, avg_acc, num_batches


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

    # =========================
    # RMFRD CONFIG
    # =========================

    rmfrd_roots: List[str] = getattr(config, "RMFRD_ROOTS", [
        "dataset/RMFRDvaSMFRD/Real-World-Masked-Face-Dataset/RWMFD_part_1"
    ])

    rmfrd_use_subset = getattr(config, "RMFRD_USE_SUBSET", True)
    rmfrd_max_classes = getattr(config, "RMFRD_MAX_CLASSES", 1000)
    rmfrd_max_images_per_class = getattr(config, "RMFRD_MAX_IMAGES_PER_CLASS", 50)

    batch_size = getattr(config, "RMFRD_BATCH_SIZE", 64)
    num_workers = getattr(config, "RMFRD_NUM_WORKERS", 4)

    total_epochs = getattr(config, "RMFRD_TOTAL_EPOCHS", 10)
    warmup_epochs = getattr(config, "RMFRD_WARMUP_EPOCHS", 2)

    base_lr = getattr(config, "RMFRD_BASE_LR", 5e-5)
    min_lr = getattr(config, "RMFRD_MIN_LR", 1e-6)
    weight_decay = getattr(config, "RMFRD_WEIGHT_DECAY", 5e-4)

    vgg_checkpoint_path = getattr(
        config,
        "VGG_CHECKPOINT_PATH",
        "checkpoints/arcface_vggface2_warmup.pth"
    )

    rmfrd_checkpoint_dir = getattr(config, "RMFRD_CHECKPOINT_DIR", "checkpoints")
    rmfrd_checkpoint_name = getattr(
        config,
        "RMFRD_CHECKPOINT_NAME",
        "arcface_rmfrd_finetuned.pth"
    )

    rmfrd_log_path = getattr(
        config,
        "RMFRD_LOG_PATH",
        "logs/train_rmfrd_log.csv"
    )

    # =========================
    # 1. Load dataset
    # =========================

    dataset = RMFRDDataset(
        root_dir=rmfrd_roots,
        transform=get_train_transform(),
        max_classes=rmfrd_max_classes if rmfrd_use_subset else None,
        max_images_per_class=rmfrd_max_images_per_class if rmfrd_use_subset else None,
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

    # =========================
    # 2. Build model
    # =========================

    model = FaceEmbeddingNet(embedding_size=config.EMBEDDING_SIZE).to(device)

    load_vggface2_checkpoint(
        model=model,
        checkpoint_path=vgg_checkpoint_path,
        device=device
    )

    arcface_head = ArcMarginProduct(
        embedding_size=config.EMBEDDING_SIZE,
        num_classes=num_classes,
        scale=64.0,
        margin=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    logger = CSVLogger(rmfrd_log_path)

    best_loss = float("inf")
    checkpoint_path = os.path.join(rmfrd_checkpoint_dir, rmfrd_checkpoint_name)

    # =========================
    # 3. Training loop
    # =========================

    for epoch in range(total_epochs):
        epoch_start = time.time()

        phase = "WARM-UP" if epoch < warmup_epochs else "COSINE-DECAY"
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, num_batches = train_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        gpu_memory = get_gpu_memory_mb()

        checkpoint_saved = False

        if train_loss < best_loss:
            best_loss = train_loss

            save_checkpoint(
                model=model,
                arcface_head=arcface_head,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                loss=train_loss,
                save_path=checkpoint_path,
            )

            checkpoint_saved = True

        logger.log({
            "epoch": epoch + 1,
            "phase": phase,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "embedding_norm": 1.0,
            "gradient_norm": 0.0,
            "num_batches": num_batches,
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
        print(f"Num Batches    : {num_batches}")
        print(f"Epoch Time     : {epoch_time:.2f} sec")
        print(f"GPU Memory     : {gpu_memory:.2f} MB")
        print(f"Checkpoint     : {'Saved' if checkpoint_saved else 'Not saved'}")
        print("=" * 70)

    print("RMFRD fine-tuning finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log   : {rmfrd_log_path}")


if __name__ == "__main__":
    main()