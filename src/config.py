import os
import torch

# =========================
# DATASET CONFIG - VGGFACE2
# =========================

VGGFACE2_ROOT = "dataset/vggface2/train"

USE_SUBSET = True
MAX_CLASSES = 4000
MAX_IMAGES_PER_CLASS = 250

# Tỷ lệ chia Train / Val / Test
SPLIT_RATIO = (0.8, 0.1, 0.1)


# =========================
# TRAINING CONFIG - VGGFACE2
# =========================

IMAGE_SIZE = 112
EMBEDDING_SIZE = 512

BATCH_SIZE = 128
NUM_WORKERS = 6

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
    # AFDB unmasked face dataset
    "dataset/RWFRD/AFDB_face_dataset/AFDB_face_dataset",
    # AFDB masked face dataset
    "dataset/RWFRD/AFDB_masked_face_dataset/AFDB_masked_face_dataset",
]

RMFRD_USE_SUBSET = True
RMFRD_SPLIT_RATIO = 0.8

# Có thể tăng sau khi test loader ổn
RMFRD_MAX_CLASSES = 2000
RMFRD_MAX_IMAGES_PER_CLASS = 30

# =========================
# TRAINING CONFIG - RMFRD
# =========================

RMFRD_BATCH_SIZE = 64
RMFRD_NUM_WORKERS = 4

RMFRD_TOTAL_EPOCHS = 8
RMFRD_WARMUP_EPOCHS = 2

# LR nhỏ hơn VGGFace2 vì đây là fine-tuning tiếp
RMFRD_BASE_LR = 1e-4
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

FINAL_CHECKPOINT_PATH = "checkpoints/arcface_rmfrd_finetuned_lite.pth"

EMPLOYEE_EMBEDDINGS_PATH = "models/embeddings/employee_embeddings.pkl"

ATTENDANCE_DB_PATH = "attendance.db"

import json

# Default settings
RECOGNITION_THRESHOLD = 0.45
COOLDOWN_SECONDS = 43200
WORK_START_TIME = "08:00"
ALLOW_LATE_MINUTES = 30
CAMERA_SOURCE_TYPE = "webcam"  # "webcam" or "ip_camera"
CAMERA_WEBCAM_INDEX = 0
CAMERA_IP_URL = ""

# Load from dynamic config.json if present
CONFIG_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
if os.path.exists(CONFIG_JSON_PATH):
    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            RECOGNITION_THRESHOLD = cfg.get("recognition_threshold", RECOGNITION_THRESHOLD)
            COOLDOWN_SECONDS = cfg.get("cooldown_seconds", COOLDOWN_SECONDS)
            WORK_START_TIME = cfg.get("work_start_time", WORK_START_TIME)
            ALLOW_LATE_MINUTES = cfg.get("allow_late_minutes", ALLOW_LATE_MINUTES)
            CAMERA_SOURCE_TYPE = cfg.get("camera_source_type", CAMERA_SOURCE_TYPE)
            CAMERA_WEBCAM_INDEX = cfg.get("camera_webcam_index", CAMERA_WEBCAM_INDEX)
            CAMERA_IP_URL = cfg.get("camera_ip_url", CAMERA_IP_URL)
    except Exception as e:
        print("Error loading config.json:", e)

# Face detection size for InsightFace - e.g., (128, 128) for high speed, (640, 640) for standard accuracy
DET_SIZE = (320, 320)
