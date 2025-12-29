# utils/config.py
"""
Central configuration for the Bird Detection project.

All important paths and hyperparameters live here so that
other scripts can just import config instead of hard-coding paths.
"""

from pathlib import Path

# -------------------------
# Project paths
# -------------------------

# <project_root>/utils/config.py -> go two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Roboflow / YOLO dataset
DATA_DIR = PROJECT_ROOT / "data" / "yolo_dataset-2"
YOLO_DATA_CONFIG = DATA_DIR / "data.yaml"

# YOLO training runs
YOLO_RUNS_DIR = PROJECT_ROOT / "runs_yolo"

# Path to best YOLO weights (after training finishes)
# NOTE: adjust 'yolov8m_bird_no_bird' if you change the run name.
YOLO_WEIGHTS = (
    YOLO_RUNS_DIR
    / "yolov8m_bird_no_bird"
    / "weights"
    / "best.pt"
)

# -------------------------
# Class / label config
# -------------------------

# From data.yaml: names: ['bird', 'no-bird']
CLASS_NAMES = ["bird", "no-bird"]
NUM_CLASSES = len(CLASS_NAMES)

BIRD_CLASS_ID = 0      # YOLO class index for "bird"
NO_BIRD_CLASS_ID = 1   # YOLO class index for "no-bird"

# -------------------------
# Crop dataset for classifier
# -------------------------

CROP_ROOT = PROJECT_ROOT / "classifier" / "crop_dataset-2"

CROP_TRAIN_DIR = CROP_ROOT / "train"
CROP_VAL_DIR = CROP_ROOT / "val"
CROP_TEST_DIR = CROP_ROOT / "test"

# Where to store classifier weights
CLASSIFIER_WEIGHTS_DIR = PROJECT_ROOT / "classifier" / "weights"
CLASSIFIER_WEIGHTS = CLASSIFIER_WEIGHTS_DIR / "resnet50_birdrefiner.pt"

# -------------------------
# YOLO inference thresholds
# -------------------------

YOLO_CONF_THRESHOLD = 0.25  # minimum YOLO confidence
YOLO_IOU_THRESHOLD = 0.45   # for NMS

# -------------------------
# Classifier (ResNet-50) params
# -------------------------

CLF_IMAGE_SIZE = 224       # crops will be resized to 224x224
CLF_BATCH_SIZE = 64
CLF_MAX_EPOCHS = 30
CLF_LEARNING_RATE = 1e-4
CLF_WEIGHT_DECAY = 1e-4
CLF_NUM_WORKERS = 4
CLF_EARLY_STOPPING_PATIENCE = 5

# At inference: minimum probability from classifier to keep a detection as "bird"
REFINER_BIRD_THRESHOLD = 0.6


def ensure_dirs():
    """
    Create all necessary directories if they don't exist.
    Call this at the start of scripts that need paths.
    """
    CROP_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CROP_VAL_DIR.mkdir(parents=True, exist_ok=True)
    CROP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick sanity check
    print("Project root:", PROJECT_ROOT)
    print("YOLO data:", YOLO_DATA_CONFIG)
    print("YOLO weights (expected):", YOLO_WEIGHTS)
    print("Crop root:", CROP_ROOT)
    ensure_dirs()
    print("All directories ensured.")
