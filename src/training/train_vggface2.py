import os
import time
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.config import (
    VGGFACE2_ROOT,
    USE_SUBSET,
    MAX_CLASSES,
    MAX_IMAGES_PER_CLASS,
    BATCH_SIZE,
    NUM_WORKERS,
    TOTAL_EPOCHS,
    WARMUP_EPOCHS,
    BASE_LR,
    MIN_LR,
    WEIGHT_DECAY,
    DEVICE,
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
    TRAIN_LOG_PATH
)

from datasets.dataset_vggface2 import VGGFace2Dataset
from utils.transforms import get_train_transform
from utils.warmup_scheduler import get_warmup_cosine_lr, set_optimizer_lr
from utils.logger import CSVLogger
from config import *

class FaceEmbeddingNet(nn.Module):
    """
    Backbone trích xuất đặc trưng khuôn mặt.

    Phiên bản này dùng ResNet50 để test pipeline huấn luyện trước.
    Khi tích hợp iResNet-100, chỉ cần thay phần backbone tại đây.
    """

    def __init__(self, embedding_size: int = 512) -> None:
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Bỏ lớp phân loại cuối của ResNet
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Đưa đặc trưng về vector 512 chiều
        self.embedding_layer = nn.Linear(2048, embedding_size)

        # BatchNorm giúp embedding ổn định hơn
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)

        embeddings = self.embedding_layer(features)
        embeddings = self.bn(embeddings)

        # Chuẩn hóa embedding để phù hợp với cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ArcMarginProduct(nn.Module):
    """
    Lớp ArcFace Head.

    Mục tiêu:
    - Làm embedding của cùng một người gần nhau hơn.
    - Làm embedding của những người khác nhau xa nhau hơn.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
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
    """Tính accuracy theo batch."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_gradient_norm(model: nn.Module) -> float:
    """
    Tính gradient norm để theo dõi độ ổn định huấn luyện.
    Nếu giá trị quá lớn, mô hình có thể bị gradient exploding.
    """
    total_norm = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


def get_gpu_memory_mb() -> float:
    """Lấy dung lượng GPU đang dùng, nếu có CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def save_checkpoint(
    model: nn.Module,
    arcface_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str
) -> None:
    """Lưu checkpoint mô hình."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "arcface_head_state_dict": arcface_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, save_path)


def train_one_epoch(
    model: nn.Module,
    arcface_head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, float, float, int]:
    """
    Huấn luyện một epoch.

    Returns:
        avg_loss
        avg_accuracy
        avg_embedding_norm
        avg_gradient_norm
        num_batches
    """
    model.train()
    arcface_head.train()

    total_loss = 0.0
    total_acc = 0.0
    total_embedding_norm = 0.0
    total_gradient_norm = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        logits = arcface_head(embeddings, labels)

        loss = criterion(logits, labels)

        loss.backward()

        grad_norm = calculate_gradient_norm(model)

        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        emb_norm = embeddings.norm(dim=1).mean().item()

        total_loss += loss.item()
        total_acc += acc
        total_embedding_norm += emb_norm
        total_gradient_norm += grad_norm
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc * 100:.2f}%"
        })

    avg_loss = total_loss / num_batches
    avg_accuracy = total_acc / num_batches
    avg_embedding_norm = total_embedding_norm / num_batches
    avg_gradient_norm = total_gradient_norm / num_batches

    return avg_loss, avg_accuracy, avg_embedding_norm, avg_gradient_norm, num_batches


def main() -> None:
    print("=" * 70)
    print("TRAIN VGGFACE2 WITH WARM-UP + ARC FACE")
    print("=" * 70)

    device = DEVICE
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # =========================
    # 1. Load dataset
    # =========================

    dataset = VGGFace2Dataset(
        root_dir=VGGFACE2_ROOT,
        transform=get_train_transform(),
        max_classes=MAX_CLASSES if USE_SUBSET else None,
        max_images_per_class=MAX_IMAGES_PER_CLASS if USE_SUBSET else None
    )

    num_classes = len(dataset.class_to_idx)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Total images: {len(dataset)}")
    print(f"Total identities: {num_classes}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total epochs: {TOTAL_EPOCHS}")
    print(f"Warm-up epochs: {WARMUP_EPOCHS}")

    # =========================
    # 2. Build model
    # =========================

    model = FaceEmbeddingNet(embedding_size=512).to(device)

    arcface_head = ArcMarginProduct(
        embedding_size=512,
        num_classes=num_classes,
        scale=64.0,
        margin=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY
    )

    logger = CSVLogger(TRAIN_LOG_PATH)

    best_loss = float("inf")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    # =========================
    # 3. Training loop
    # =========================

    for epoch in range(TOTAL_EPOCHS):
        epoch_start_time = time.time()

        phase = "WARM-UP" if epoch < WARMUP_EPOCHS else "COSINE-DECAY"

        current_lr = get_warmup_cosine_lr(
            current_epoch=epoch,
            total_epochs=TOTAL_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            base_lr=BASE_LR,
            min_lr=MIN_LR
        )

        set_optimizer_lr(optimizer, current_lr)

        train_loss, train_acc, emb_norm, grad_norm, num_batches = train_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        epoch_time = time.time() - epoch_start_time
        gpu_memory = get_gpu_memory_mb()

        checkpoint_saved = False

        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(
                model=model,
                arcface_head=arcface_head,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=train_loss,
                save_path=checkpoint_path
            )
            checkpoint_saved = True

        logger.log({
            "epoch": epoch + 1,
            "phase": phase,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "embedding_norm": emb_norm,
            "gradient_norm": grad_norm,
            "num_batches": num_batches,
            "epoch_time_sec": round(epoch_time, 2),
            "device": device,
            "gpu_memory_mb": round(gpu_memory, 2),
            "checkpoint_saved": checkpoint_saved
        })

        print("=" * 70)
        print(f"Epoch          : {epoch + 1}/{TOTAL_EPOCHS}")
        print(f"Phase          : {phase}")
        print(f"Learning Rate  : {current_lr:.8f}")
        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_acc * 100:.2f}%")
        print(f"Embedding Norm : {emb_norm:.4f}")
        print(f"Gradient Norm  : {grad_norm:.4f}")
        print(f"Num Batches    : {num_batches}")
        print(f"Epoch Time     : {epoch_time:.2f} sec")
        print(f"GPU Memory     : {gpu_memory:.2f} MB")
        print(f"Checkpoint     : {'Saved' if checkpoint_saved else 'Not saved'}")
        print("=" * 70)

    print("Training finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log: {TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()