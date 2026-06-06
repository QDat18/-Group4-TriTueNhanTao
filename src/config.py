import os
import torch

# =========================
# DATASET CONFIG - VGGFACE2
# =========================

VGGFACE2_ROOT = "dataset/VGGFace2/train"

USE_SUBSET = True
MAX_CLASSES = 4000
MAX_IMAGES_PER_CLASS = 200

# =========================
# TRAINING CONFIG - VGGFACE2
# =========================

IMAGE_SIZE = 112
EMBEDDING_SIZE = 512

BATCH_SIZE = 128
NUM_WORKERS = 4

TOTAL_EPOCHS = 25
WARMUP_EPOCHS = 5

BASE_LR = 1e-4
MIN_LR = 1e-6

WEIGHT_DECAY = 5e-4

# =========================
# DEVICE CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# OUTPUT CONFIG - VGGFACE2
# =========================

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "arcface_vggface2_warmup.pth"

LOG_DIR = "logs"
TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train_log.csv")

# =========================
# DATASET CONFIG - RMFRD / RWMFD
# =========================

RMFRD_ROOTS = [
    # Real-World Masked Face Dataset - Part 1
    "dataset/RMFRDvaSMFRD/Real-World-Masked-Face-Dataset/RWMFD_part_1",

    # Real-World Masked Face Dataset - Part 2
    "dataset/RMFRDvaSMFRD/Real-World-Masked-Face-Dataset/RMFD/RWMFD_part_2_pro",

    # Folder thứ 3 nếu có cấu trúc theo danh tính
    # Nếu folder chưa tồn tại hoặc chưa đúng cấu trúc, loader sẽ bỏ qua.
    "dataset/RMFRDvaSMFRD/Real-World-Masked-Face-Dataset/RMFD/RWMFD_part_3_pro",
]

RMFRD_USE_SUBSET = True

# Có thể tăng sau khi test loader ổn
RMFRD_MAX_CLASSES = 2000
RMFRD_MAX_IMAGES_PER_CLASS = 30

# =========================
# TRAINING CONFIG - RMFRD
# =========================

RMFRD_BATCH_SIZE = 64
RMFRD_NUM_WORKERS = 4

RMFRD_TOTAL_EPOCHS = 8
RMFRD_WARMUP_EPOCHS = 1

# LR nhỏ hơn VGGFace2 vì đây là fine-tuning tiếp
RMFRD_BASE_LR = 5e-5
RMFRD_MIN_LR = 1e-6
RMFRD_WEIGHT_DECAY = 5e-4

# =========================
# CHECKPOINT CONFIG - RMFRD
# =========================

VGG_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "arcface_vggface2_warmup.pth"
)

RMFRD_CHECKPOINT_DIR = "checkpoints"
RMFRD_CHECKPOINT_NAME = "arcface_rmfrd_finetuned.pth"

RMFRD_LOG_PATH = os.path.join(LOG_DIR, "train_rmfrd_log.csv")

# =========================
# ATTENDANCE CONFIG
# =========================

INHOUSE_ROOT = "dataset/in-house"

FINAL_CHECKPOINT_PATH = "checkpoints/arcface_vggface2_warmup.pth"

EMPLOYEE_EMBEDDINGS_PATH = "models/embeddings/employee_embeddings.pkl"

ATTENDANCE_DB_PATH = "attendance.db"

RECOGNITION_THRESHOLD = 0.45
COOLDOWN_SECONDS = 60

# Face detection size for InsightFace - e.g., (128, 128) for high speed, (640, 640) for standard accuracy
DET_SIZE = (640, 640)
