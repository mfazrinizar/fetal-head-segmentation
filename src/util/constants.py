"""
Constants and configuration for the fetal head segmentation project.
"""
from pathlib import Path
import os

# Path Configuration
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    BASE_DIR = Path("/kaggle/working/fetal-head-segmentation")
    DATA_DIR = Path("/kaggle/input/fetal-head-segmentation-yolo-splitted")
    RAW_DATA_DIR = DATA_DIR
else:
    BASE_DIR = Path("/mnt/mfn/Projects/fetal-head-segmentation")
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "fetal-head"

RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Dataset Configuration
CLASSES = {0: "Brain", 1: "CSP", 2: "LV"}
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = list(CLASSES.values())

PLANES = [
    "Trans-cerebellum",
    "Trans-thalamic",
    "Trans-ventricular",
    "Diverse Fetal Head Images"
]

ORIGINAL_WIDTH = 959
ORIGINAL_HEIGHT = 661

# Training Configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMG_SIZE = 640
DEFAULT_GPUS = "0,1"

# Model Configuration (YOLO26)
MODEL_SIZES = ["n", "s", "m", "l", "x"]
DEFAULT_MODEL_SIZE = "n"

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    "horizontal_flip_p": 0.3,
    "vertical_flip_p": 0.3,
    "rotation_limit": 30,
    "rotation_p": 0.3,
    "random_crop_size": (512, 512),
    "rrc_scale": (1.2, 1.4),
    "brightness_contrast_p": 0.5,
    "elastic_transform_p": 0.5,
    "elastic_alpha": 1,
    "elastic_sigma": 50,
    "gaussian_noise_p": 0.5,
    "gaussian_noise_var_limit": (10.0, 50.0),
    "gaussian_blur_p": 0.5,
    "gaussian_blur_limit": (3, 7),
}

# Experiment Configurations
EXPERIMENTS = {
    "baseline": {
        "name": "Baseline YOLO26n-seg",
        "description": "Standard YOLO26n with YOLO default loss (BCE+Dice)",
        "model_size": "n",
        "augmentation": "basic",
        "loss_config": "default",
    },
    "fetsam_aug": {
        "name": "FetSAM Augmentation",
        "description": "YOLO26n with FetSAM paper augmentations",
        "model_size": "n",
        "augmentation": "fetsam_full",
        "loss_config": "default",
    },
    "fetsam_loss": {
        "name": "FetSAM Combined Loss",
        "description": "Weighted Dice + Weighted Lovasz (α=0.5, β=0.5)",
        "model_size": "n",
        "augmentation": "fetsam_full",
        "loss_config": "fetsam",
    },
    "fetsam_full": {
        "name": "Full FetSAM Pipeline",
        "description": "FetSAM augmentations + FetSAM loss",
        "model_size": "n",
        "augmentation": "fetsam_full",
        "loss_config": "fetsam",
    },
}

# FetSAM Loss Implementation Notes:
#
# YOLO26 Default Loss (v8SegmentationLoss):
#   - BCEDiceLoss: 0.5 * BCE + 0.5 * Dice
#   - No class weighting
#   - No Lovasz loss
#
# FetSAM Combined Loss (from paper):
#   - L_combined = α × L_dice + β × L_lovasz (α=0.5, β=0.5)
#   - Class weights: [0.1, 0.9, 0.7] for [Brain, CSP, LV]
#   - Lovasz loss directly optimizes IoU metric
#   - Key difference: Lovasz provides better gradients for IoU optimization
#
# Why FetSAM Loss Matters:
#   1. Lovasz Loss (Berman et al., 2018) is a convex surrogate for IoU
#   2. Dice loss optimizes soft overlap, Lovasz optimizes hard IoU
#   3. Combination captures both soft and hard region metrics
#   4. Class weights [0.1, 0.9, 0.7] prioritize small structures (CSP > LV > Brain)

# Training Hyperparameters
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "patience": 20,
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "cos_lr": True,
    "workers": 8,
    "seed": 42,
}

# Evaluation Metrics
METRICS = {
    "classification": ["accuracy", "precision", "recall", "specificity", "f1_score"],
    "detection": ["mAP50", "mAP50_95", "iou", "miou", "dice", "mdice"],
    "segmentation": ["mAP50", "mAP50_95", "iou", "miou", "dice", "mdice"],
}


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    dirs = [
        RESULTS_DIR,
        TRAIN_DIR / "images",
        TRAIN_DIR / "labels",
        VAL_DIR / "images",
        VAL_DIR / "labels",
        TEST_DIR / "images",
        TEST_DIR / "labels",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_experiment_dir(experiment_name: str) -> Path:
    """Get the results directory for a specific experiment."""
    exp_dir = RESULTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
