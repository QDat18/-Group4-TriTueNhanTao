"""
Unit tests cho hệ thống nhận diện khuôn mặt.
Chạy: python -m pytest tests/ -v
"""

import numpy as np
import pytest
from datetime import datetime, timedelta


# ═══════════════════════════════════════════
# TEST: Anti-Spoofing Module
# ═══════════════════════════════════════════

class TestAntiSpoofing:
    """Test module chống giả mạo khuôn mặt."""

    def setup_method(self):
        from src.anti_spoofing.anti_spoofing import LivenessDetector
        self.detector = LivenessDetector()

    def test_none_image_returns_not_live(self):
        result = self.detector.check_liveness(None)
        assert result["is_live"] is False
        assert result["score"] == 0.0

    def test_empty_image_returns_not_live(self):
        empty = np.array([], dtype=np.uint8)
        result = self.detector.check_liveness(empty)
        assert result["is_live"] is False

    def test_valid_face_returns_dict_keys(self):
        # Tạo ảnh giả lập khuôn mặt (noise RGB 112x112)
        fake_face = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        result = self.detector.check_liveness(fake_face)
        assert "is_live" in result
        assert "score" in result
        assert "reason" in result
        assert "blur" in result
        assert "peak_ratio" in result

    def test_low_quality_image_detected(self):
        # Ảnh mờ (toàn màu đồng nhất) → blur_score thấp
        blurry = np.full((112, 112, 3), 128, dtype=np.uint8)
        result = self.detector.check_liveness(blurry)
        assert result["is_live"] is False


# ═══════════════════════════════════════════
# TEST: Face Alignment
# ═══════════════════════════════════════════

class TestFaceAlignment:
    """Test face alignment module."""

    def test_align_face_with_valid_keypoints(self):
        from src.attendance.align_face import align_face
        # Tạo ảnh giả 640x480
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # 5 keypoints giả lập
        kps = np.array([
            [200, 200], [300, 200], [250, 250], [210, 300], [290, 300]
        ], dtype=np.float32)
        aligned = align_face(image, kps)
        assert aligned.shape == (112, 112, 3)

    def test_align_face_without_keypoints(self):
        from src.attendance.align_face import align_face
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        aligned = align_face(image, None)
        assert aligned.shape == (112, 112, 3)

    def test_align_face_wrong_keypoint_count(self):
        from src.attendance.align_face import align_face
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        kps = np.array([[200, 200], [300, 200]], dtype=np.float32)  # Chỉ 2 điểm
        aligned = align_face(image, kps)
        assert aligned.shape == (112, 112, 3)  # Fallback resize


# ═══════════════════════════════════════════
# TEST: Image Transforms
# ═══════════════════════════════════════════

class TestTransforms:
    """Test image preprocessing transforms."""

    def test_train_transform_output_shape(self):
        from src.utils.transforms import get_train_transform
        from PIL import Image
        transform = get_train_transform()
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 112, 112)

    def test_val_transform_output_shape(self):
        from src.utils.transforms import get_val_transform
        from PIL import Image
        transform = get_val_transform()
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 112, 112)

    def test_val_transform_deterministic(self):
        from src.utils.transforms import get_val_transform
        from PIL import Image
        import torch
        transform = get_val_transform()
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        t1 = transform(img)
        t2 = transform(img)
        assert torch.equal(t1, t2)  # Val transform should be deterministic


# ═══════════════════════════════════════════
# TEST: Warmup Scheduler
# ═══════════════════════════════════════════

class TestWarmupScheduler:
    """Test learning rate scheduler."""

    def test_warmup_increases_lr(self):
        import torch
        from src.utils.warmup_scheduler import get_warmup_cosine_scheduler

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs=5, total_epochs=20,
            base_lr=1e-3, min_lr=1e-6
        )

        lrs = []
        for _ in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should increase during warmup (first 5 epochs)
        assert lrs[4] > lrs[0], "LR should increase during warmup"

    def test_cosine_decay_decreases_lr(self):
        import torch
        from src.utils.warmup_scheduler import get_warmup_cosine_scheduler

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs=2, total_epochs=10,
            base_lr=1e-3, min_lr=1e-6
        )

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should decrease after warmup
        assert lrs[-1] < lrs[3], "LR should decrease during cosine decay"


# ═══════════════════════════════════════════
# TEST: CSV Logger
# ═══════════════════════════════════════════

class TestCSVLogger:
    """Test training logger."""

    def test_logger_creates_file(self, tmp_path):
        from src.utils.logger import CSVLogger
        log_path = str(tmp_path / "test_log.csv")
        logger = CSVLogger(log_path)
        assert (tmp_path / "test_log.csv").exists()

    def test_logger_writes_row(self, tmp_path):
        from src.utils.logger import CSVLogger
        import csv
        log_path = str(tmp_path / "test_log.csv")
        logger = CSVLogger(log_path)
        logger.log({
            "epoch": 1, "phase": "WARM-UP", "learning_rate": 0.001,
            "train_loss": 0.5, "train_accuracy": 0.8,
            "embedding_norm": 1.0, "gradient_norm": 0.1,
            "num_batches": 100, "epoch_time_sec": 60.0,
            "device": "cpu", "gpu_memory_mb": 0.0,
            "checkpoint_saved": True
        })
        with open(log_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2  # header + 1 data row


# ═══════════════════════════════════════════
# TEST: ArcFace Head
# ═══════════════════════════════════════════

class TestArcFaceHead:
    """Test ArcFace loss head."""

    def test_output_shape(self):
        import torch
        from src.models.arcface_head import ArcMarginProduct
        head = ArcMarginProduct(embedding_size=512, num_classes=100)
        embeddings = torch.randn(8, 512)
        labels = torch.randint(0, 100, (8,))
        logits = head(embeddings, labels)
        assert logits.shape == (8, 100)

    def test_scale_applied(self):
        import torch
        from src.models.arcface_head import ArcMarginProduct
        head = ArcMarginProduct(embedding_size=512, num_classes=10, scale=64.0)
        embeddings = torch.randn(4, 512)
        labels = torch.randint(0, 10, (4,))
        logits = head(embeddings, labels)
        # With scale=64, logits should have large absolute values
        assert logits.abs().max().item() > 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
