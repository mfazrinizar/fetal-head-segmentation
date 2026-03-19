"""
Training script for fetal head segmentation using YOLO26-seg.

Provides configurable experiment settings, DDP multi-GPU training support,
comprehensive metric logging, and Kaggle compatibility.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.util.constants import (
    DATA_DIR, RESULTS_DIR, IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
    TRAINING_CONFIG, EXPERIMENTS, FETSAM_AUGMENTATION, BASIC_AUGMENTATION,
    IS_KAGGLE, ensure_dirs
)


def get_augmentation_config(experiment_type: str = "basic") -> Dict:
    """
    Get YOLO-compatible augmentation configuration.
    Based on FetSAM paper augmentation strategy.
    """
    if experiment_type == "basic":
        return BASIC_AUGMENTATION.copy()
    elif experiment_type == "fetsam_full":
        return FETSAM_AUGMENTATION.copy()
    return {}


def create_training_config(
    experiment_name: str,
    model_size: str = "n",
    augmentation: str = "basic",
    custom_modules: bool = False,
    epochs: int = None,
    batch_size: int = None,
    device: str = "0,1",
    resume: Optional[str] = None,
) -> Dict:
    """Create training configuration for an experiment."""
    config = TRAINING_CONFIG.copy()

    if epochs is not None:
        config['epochs'] = epochs
    if batch_size is not None:
        config['batch_size'] = batch_size

    aug_config = get_augmentation_config(augmentation)

    # YOLO26 model path
    model_path = f"yolo26{model_size}-seg.yaml"

    output_dir = RESULTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "experiment_name": experiment_name,
        "model": model_path,
        "model_size": model_size,
        "data": str(DATA_DIR / "dataset.yaml"),
        "epochs": config['epochs'],
        "batch": config['batch_size'],
        "imgsz": IMG_SIZE,
        "device": device,
        "workers": config['workers'],
        "project": str(RESULTS_DIR),
        "name": experiment_name,
        "exist_ok": True,
        "pretrained": False,  # YOLO26 doesn't have pretrained weights yet
        "optimizer": config['optimizer'],
        "lr0": config['lr0'],
        "lrf": config['lrf'],
        "weight_decay": config['weight_decay'],
        "warmup_epochs": config['warmup_epochs'],
        "cos_lr": config['cos_lr'],
        "patience": config['patience'],
        "seed": config['seed'],
        "verbose": True,
        "save": True,
        "save_period": 10,
        "plots": True,
        "resume": resume,
        **aug_config,
    }

    return full_config


def train_yolo(config: Dict) -> None:
    """Run YOLO26 training with the given configuration."""
    from ultralytics import YOLO

    model_path = config.pop('model')
    experiment_name = config.pop('experiment_name')
    model_size = config.pop('model_size')

    print(f"\n{'='*60}")
    print(f"TRAINING: {experiment_name}")
    print(f"{'='*60}")
    print(f"Model: YOLO26{model_size}-seg ({model_path})")
    print(f"Data: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch']}")
    print(f"Image size: {config['imgsz']}")
    print(f"Device: {config['device']}")
    print(f"{'='*60}\n")

    if config.get('resume'):
        print(f"Resuming from: {config['resume']}")
        model = YOLO(config['resume'])
    else:
        model = YOLO(model_path)

    results = model.train(**config)

    output_dir = Path(config['project']) / config['name']

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump({**config, 'model': model_path, 'experiment_name': experiment_name}, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


def run_experiment(
    experiment_key: str,
    epochs: int = None,
    batch_size: int = None,
    device: str = "0,1",
    resume: Optional[str] = None,
) -> None:
    """Run a predefined experiment."""
    if experiment_key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_key}. Available: {list(EXPERIMENTS.keys())}")

    exp_config = EXPERIMENTS[experiment_key]

    config = create_training_config(
        experiment_name=experiment_key,
        model_size=exp_config.get('model_size', 'n'),
        augmentation=exp_config.get('augmentation', 'basic'),
        custom_modules=exp_config.get('custom_modules', False),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        resume=resume,
    )

    train_yolo(config)


def main():
    parser = argparse.ArgumentParser(description="Train fetal head segmentation model with YOLO26")

    parser.add_argument("--experiment", type=str, default="baseline",
                        choices=list(EXPERIMENTS.keys()),
                        help="Experiment to run")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: from config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: from config)")
    parser.add_argument("--device", type=str, default="0,1",
                        help="GPU devices (e.g., '0,1' for DDP)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--model-size", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="Model size for custom training")
    parser.add_argument("--augmentation", type=str, default="basic",
                        choices=["basic", "fetsam_full"],
                        help="Augmentation strategy")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom experiment name")

    args = parser.parse_args()

    ensure_dirs()

    if args.name:
        config = create_training_config(
            experiment_name=args.name,
            model_size=args.model_size,
            augmentation=args.augmentation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            resume=args.resume,
        )
        train_yolo(config)
    else:
        run_experiment(
            experiment_key=args.experiment,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
