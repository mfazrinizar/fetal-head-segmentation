"""
Fetal Head Segmentation - Complete Pipeline

Complete pipeline for data loading, model training with YOLO26, evaluation,
and visualization. Compatible with both local and Kaggle environments.
"""

import os
import sys
import json
from pathlib import Path

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    print("Running in Kaggle environment")
    os.system("pip install -q git+https://github.com/mfazrinizar/ultralytics.git")
    BASE_DIR = Path("/kaggle/working/fetal-head-segmentation")
    DATA_DIR = Path("/kaggle/input/fetal-head-segmentation-yolo-splitted")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE_DIR / "src"))
else:
    print("Running in local environment")
    BASE_DIR = Path("/mnt/mfn/Projects/fetal-head-segmentation")
    DATA_DIR = BASE_DIR / "data"
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import matplotlib.pyplot as plt

# Configuration flags
RUN_EDA = False
RUN_PREPROCESSING = False
RUN_TRAINING = True
RUN_EVALUATION = True

# Training configuration
EXPERIMENT = "baseline"
MODEL_SIZE = "n"
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = "0"
RESUME_FROM = None
CUSTOM_NAME = None
AUGMENTATION = "fetsam_full"


def verify_data():
    """Verify dataset exists and is properly formatted."""
    print("\n" + "="*60)
    print("DATA VERIFICATION")
    print("="*60)

    dataset_yaml = DATA_DIR / "dataset.yaml"

    if dataset_yaml.exists():
        print(f"✓ Dataset config found: {dataset_yaml}")

        for split in ["train", "val", "test"]:
            split_dir = DATA_DIR / split
            if split_dir.exists():
                n_images = len(list((split_dir / "images").glob("*.png")))
                n_labels = len(list((split_dir / "labels").glob("*.txt")))
                print(f"  {split}: {n_images} images, {n_labels} labels")
    else:
        print("✗ Dataset not found!")
        if not IS_KAGGLE and RUN_PREPROCESSING:
            print("Running preprocessing...")
            from src.preprocess.data_preprocessing import run_preprocessing
            run_preprocessing(seed=42)
        else:
            raise FileNotFoundError("Dataset not found. Please run preprocessing first.")

    return dataset_yaml


def run_eda():
    """Run exploratory data analysis."""
    if RUN_EDA and not IS_KAGGLE:
        print("\n" + "="*60)
        print("RUNNING EDA")
        print("="*60)

        from src.eda.comprehensive_eda import run_full_eda
        run_full_eda()


def train_model(dataset_yaml):
    """Train YOLO26 model."""
    if not RUN_TRAINING:
        return

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    from ultralytics import YOLO

    # YOLO26 model
    model_path = f"yolo26{MODEL_SIZE}-seg.yaml"

    if AUGMENTATION == "fetsam_full":
        aug_config = {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.4,
            "degrees": 30.0,
            "translate": 0.1,
            "scale": 0.3,
            "flipud": 0.3,
            "fliplr": 0.3,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }
    else:
        aug_config = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }

    experiment_name = CUSTOM_NAME if CUSTOM_NAME else EXPERIMENT

    print(f"Experiment: {experiment_name}")
    print(f"Model: YOLO26{MODEL_SIZE}-seg ({model_path})")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    if RESUME_FROM:
        print(f"Resuming from: {RESUME_FROM}")
        model = YOLO(RESUME_FROM)
    else:
        model = YOLO(model_path)

    results = model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=640,
        device=DEVICE,
        workers=8,
        project=str(BASE_DIR / "results") if not IS_KAGGLE else "/kaggle/working/results",
        name=experiment_name,
        exist_ok=True,
        pretrained=False,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        weight_decay=1e-4,
        warmup_epochs=3,
        cos_lr=True,
        patience=20,
        seed=42,
        verbose=True,
        save=True,
        save_period=10,
        plots=True,
        **aug_config,
    )

    print("\n✓ Training complete!")
    return experiment_name


def evaluate_model(dataset_yaml, experiment_name):
    """Evaluate trained model."""
    if not RUN_EVALUATION:
        return

    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    if experiment_name is None:
        experiment_name = CUSTOM_NAME if CUSTOM_NAME else EXPERIMENT

    results_path = (BASE_DIR / "results" / experiment_name) if not IS_KAGGLE else Path(f"/kaggle/working/results/{experiment_name}")

    if not results_path.exists():
        print(f"✗ Results directory not found: {results_path}")
        return

    best_model_path = results_path / "weights" / "best.pt"

    if not best_model_path.exists():
        print(f"✗ Best model not found at: {best_model_path}")
        return

    print(f"Loading best model: {best_model_path}")

    from ultralytics import YOLO
    model = YOLO(str(best_model_path))

    print("\nRunning validation on test set...")
    val_results = model.val(
        data=str(dataset_yaml),
        split="test",
        batch=BATCH_SIZE,
        imgsz=640,
        device=DEVICE,
        plots=True,
        save_json=True,
    )

    print("\n--- Test Set Results ---")
    print(f"mAP@50 (Box): {val_results.box.map50:.4f}")
    print(f"mAP@50-95 (Box): {val_results.box.map:.4f}")
    print(f"mAP@50 (Mask): {val_results.seg.map50:.4f}")
    print(f"mAP@50-95 (Mask): {val_results.seg.map:.4f}")

    print("\n--- Per-Class Results ---")
    class_names = ["Brain", "CSP", "LV"]
    for i, name in enumerate(class_names):
        if i < len(val_results.box.maps):
            print(f"{name}:")
            print(f"  Box mAP@50: {val_results.box.maps[i]:.4f}")
            if hasattr(val_results.seg, 'maps') and i < len(val_results.seg.maps):
                print(f"  Mask mAP@50: {val_results.seg.maps[i]:.4f}")

    eval_results = {
        "experiment": experiment_name,
        "model_size": MODEL_SIZE,
        "epochs": EPOCHS,
        "test_results": {
            "box_map50": float(val_results.box.map50),
            "box_map": float(val_results.box.map),
            "mask_map50": float(val_results.seg.map50),
            "mask_map": float(val_results.seg.map),
        }
    }

    eval_path = results_path / "test_evaluation.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n✓ Evaluation results saved to: {eval_path}")


def generate_predictions(dataset_yaml, experiment_name):
    """Generate sample predictions."""
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)

    if experiment_name is None:
        experiment_name = CUSTOM_NAME if CUSTOM_NAME else EXPERIMENT

    try:
        from ultralytics import YOLO

        results_path = (BASE_DIR / "results" / experiment_name) if not IS_KAGGLE else Path(f"/kaggle/working/results/{experiment_name}")
        best_model_path = results_path / "weights" / "best.pt"

        if not best_model_path.exists():
            print("Model not found for predictions")
            return

        model = YOLO(str(best_model_path))
        test_images = list((DATA_DIR / "test" / "images").glob("*.png"))[:6]

        if test_images:
            results = model.predict(
                source=[str(img) for img in test_images],
                save=True,
                save_txt=False,
                conf=0.25,
                iou=0.7,
                project=str(results_path / "predictions"),
                name="test_samples",
                exist_ok=True,
            )
            print(f"✓ Predictions saved to: {results_path / 'predictions' / 'test_samples'}")
        else:
            print("No test images found")
    except Exception as e:
        print(f"Could not generate predictions: {e}")


def main():
    """Run complete pipeline."""
    dataset_yaml = verify_data()
    run_eda()
    experiment_name = train_model(dataset_yaml) if RUN_TRAINING else None
    evaluate_model(dataset_yaml, experiment_name)

    if RUN_TRAINING or RUN_EVALUATION:
        generate_predictions(dataset_yaml, experiment_name)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
