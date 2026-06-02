import torch

# =========================
# DATASET CONFIG
# =========================

VGGFACE2_ROOT = "dataset/VGGFace2/train"

# Train thử trước, không dùng full dataset ngay
USE_SUBSET = True
MAX_CLASSES = 500
MAX_IMAGES_PER_CLASS = 50

# =========================
# TRAINING CONFIG
# =========================
IMAGE_SIZE = 112
EMBEDDING_SIZE = 512

BATCH_SIZE = 64
NUM_WORKERS = 2

TOTAL_EPOCHS = 10
WARMUP_EPOCHS = 3

BASE_LR = 1e-4
MIN_LR = 1e-6

WEIGHT_DECAY = 5e-4

# =========================
# DEVICE CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# OUTPUT CONFIG
# =========================

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "arcface_vggface2_warmup.pth"