"""
Quick validation test for all experiment configurations.

Tests each experiment with 1 epoch on mini dataset (80 train, 20 val images)
to verify training pipeline works correctly.
"""

import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "ultralytics-yolo"))
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "validation_tests"

EXPERIMENTS = {
    "baseline": {
        "description": "Standard YOLO26n with YOLO default loss",
        "model_size": "n",
        "aug_config": {
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "degrees": 0.0, "flipud": 0.0, "fliplr": 0.5,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        }
    },
    "fetsam_aug": {
        "description": "YOLO26n with FetSAM paper augmentations",
        "model_size": "n",
        "aug_config": {
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.4,
            "degrees": 30.0, "flipud": 0.3, "fliplr": 0.3,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        }
    },
    "fetsam_loss": {
        "description": "FetSAM Combined Loss (Weighted Dice + Lovasz)",
        "model_size": "n",
        "aug_config": {
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.4,
            "degrees": 30.0, "flipud": 0.3, "fliplr": 0.3,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        }
    },
    "fetsam_full": {
        "description": "Full FetSAM Pipeline (aug + loss)",
        "model_size": "n",
        "aug_config": {
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.4,
            "degrees": 30.0, "flipud": 0.3, "fliplr": 0.3,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        }
    },
}


def create_mini_dataset():
    """Create a mini dataset with 100 images for quick testing."""
    mini_dir = DATA_DIR / "mini_test"
    
    if mini_dir.exists():
        shutil.rmtree(mini_dir)
    
    for split in ["train", "val"]:
        src_images = DATA_DIR / split / "images"
        src_labels = DATA_DIR / split / "labels"
        
        dst_images = mini_dir / split / "images"
        dst_labels = mini_dir / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        n_images = 80 if split == "train" else 20
        images = sorted(src_images.glob("*.png"))[:n_images]
        
        for img_path in images:
            shutil.copy2(img_path, dst_images / img_path.name)
            label_path = src_labels / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, dst_labels / label_path.name)
    
    yaml_content = f"""path: {mini_dir}
train: train/images
val: val/images

names:
  0: Brain
  1: CSP
  2: LV

nc: 3
"""
    with open(mini_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"Created mini dataset at {mini_dir}")
    return mini_dir / "dataset.yaml"


def test_loss_functions():
    """Test custom loss functions work correctly."""
    print("\n" + "="*60)
    print("Testing Custom Loss Functions")
    print("="*60)
    
    try:
        import torch
        from src.model.losses import get_loss_function, FetSAMCombinedLoss
        
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 2, (2, 3, 64, 64)).float()
        
        # Test all configurations including fetsam
        for config in ['default', 'fetsam', 'boundary_aware', 'class_balanced']:
            loss_fn = get_loss_function(config)
            losses = loss_fn(pred, target)
            print(f"  ✓ {config}: total={losses['total'].item():.4f}")
        
        # Specific FetSAM test
        fetsam = FetSAMCombinedLoss()
        fetsam_losses = fetsam(pred, target)
        print(f"\n  FetSAM Components:")
        print(f"    Weighted Dice: {fetsam_losses['dice'].item():.4f}")
        print(f"    Weighted Lovasz: {fetsam_losses['lovasz'].item():.4f}")
        print(f"    Combined (0.5*D + 0.5*L): {fetsam_losses['total'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Loss functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment(name, config, dataset_yaml):
    """Test a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Description: {config['description']}")
    print(f"Model: YOLO26{config['model_size']}-seg")
    print("="*60)
    
    model_path = f"yolo26{config['model_size']}-seg.yaml"
    model = YOLO(model_path)
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=1,
            batch=4,
            imgsz=320,
            device="0",
            workers=2,
            project=str(RESULTS_DIR),
            name=name,
            exist_ok=True,
            pretrained=False,
            optimizer="AdamW",
            lr0=1e-3,
            patience=1,
            seed=42,
            verbose=False,
            save=False,
            plots=False,
            **config["aug_config"],
        )
        
        print(f"✓ {name}: Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ {name}: FAILED - {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("YOLO26 EXPERIMENT VALIDATION TESTS")
    print("RTX 2060 Mobile Compatibility Check")
    print("="*60)
    
    loss_ok = test_loss_functions()
    
    print("\nCreating mini dataset...")
    dataset_yaml = create_mini_dataset()
    
    results = {"loss_functions": loss_ok}
    for name, config in EXPERIMENTS.items():
        results[name] = test_experiment(name, config, dataset_yaml)
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    
    print("\nCleaning up...")
    mini_dir = DATA_DIR / "mini_test"
    if mini_dir.exists():
        shutil.rmtree(mini_dir)
    
    test_results = RESULTS_DIR
    if test_results.exists():
        shutil.rmtree(test_results)
    
    print("✓ Cleanup complete")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
