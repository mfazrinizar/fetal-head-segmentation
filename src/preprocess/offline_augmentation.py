"""
Offline augmentation for fetal head segmentation (FetSAM-style).
Generates augmented samples to increase dataset diversity.
Transforms ENTIRE images, preserving anatomical relationships.

Key differences from flawed approach:
- Augments whole images (all classes transform together)
- Properly transforms polygon coordinates for geometric operations
- Goal: increase diversity, not artificially balance counts
"""
import json
import shutil
from pathlib import Path
from collections import Counter
import random
import cv2
import numpy as np


def count_class_instances(labels_dir: Path) -> dict:
    """Count instances per class in label files."""
    counts = Counter()
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counts[cls_id] += 1
    return dict(counts)


def transform_yolo_polygon(polygon_points: list, transform_type: str, img_w: int, img_h: int) -> list:
    """
    Transform YOLO polygon coordinates based on augmentation type.
    YOLO format: normalized [x1, y1, x2, y2, ...] coordinates (0-1 range)
    """
    points = np.array(polygon_points, dtype=np.float32).reshape(-1, 2)
    
    if transform_type == "hflip":
        # Horizontal flip: x' = 1 - x
        points[:, 0] = 1.0 - points[:, 0]
    elif transform_type == "vflip":
        # Vertical flip: y' = 1 - y
        points[:, 1] = 1.0 - points[:, 1]
    elif transform_type == "hvflip":
        # Both flips
        points[:, 0] = 1.0 - points[:, 0]
        points[:, 1] = 1.0 - points[:, 1]
    elif transform_type == "rot90":
        # 90 degree clockwise: (x, y) -> (1-y, x)
        new_points = np.zeros_like(points)
        new_points[:, 0] = 1.0 - points[:, 1]
        new_points[:, 1] = points[:, 0]
        points = new_points
    elif transform_type == "rot180":
        # 180 degree: (x, y) -> (1-x, 1-y)
        points[:, 0] = 1.0 - points[:, 0]
        points[:, 1] = 1.0 - points[:, 1]
    elif transform_type == "rot270":
        # 270 degree clockwise: (x, y) -> (y, 1-x)
        new_points = np.zeros_like(points)
        new_points[:, 0] = points[:, 1]
        new_points[:, 1] = 1.0 - points[:, 0]
        points = new_points
    
    # Clip to valid range
    points = np.clip(points, 0.0, 1.0)
    
    return points.flatten().tolist()


def transform_image(img: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply geometric transformation to image."""
    if transform_type == "hflip":
        return cv2.flip(img, 1)
    elif transform_type == "vflip":
        return cv2.flip(img, 0)
    elif transform_type == "hvflip":
        return cv2.flip(img, -1)
    elif transform_type == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif transform_type == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    elif transform_type == "rot270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transform_type == "brightness_up":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    elif transform_type == "brightness_down":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=-30)
    elif transform_type == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    elif transform_type == "blur":
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img


def augment_single_image(
    img_path: Path,
    label_path: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    transform_type: str,
    suffix: str
) -> bool:
    """Augment a single image with proper label transformation."""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # Transform image
    aug_img = transform_image(img, transform_type)
    
    # Read and transform labels
    new_labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls_id = parts[0]
            polygon_points = [float(p) for p in parts[1:]]
            
            # Only transform coordinates for geometric operations
            if transform_type in ["hflip", "vflip", "hvflip", "rot90", "rot180", "rot270"]:
                # Handle rotation size change
                if transform_type in ["rot90", "rot270"]:
                    new_w, new_h = h, w  # Dimensions swap
                else:
                    new_w, new_h = w, h
                
                transformed_points = transform_yolo_polygon(polygon_points, transform_type, new_w, new_h)
                new_labels.append(f"{cls_id} " + " ".join(f"{p:.6f}" for p in transformed_points))
            else:
                # Non-geometric transforms: keep original labels
                new_labels.append(line.strip())
    
    # Save augmented image
    new_img_name = f"{img_path.stem}_{suffix}{img_path.suffix}"
    cv2.imwrite(str(output_img_dir / new_img_name), aug_img)
    
    # Save transformed labels
    new_label_name = f"{label_path.stem}_{suffix}.txt"
    with open(output_label_dir / new_label_name, 'w') as f:
        f.write("\n".join(new_labels))
    
    return True


def offline_augment_dataset(
    data_dir: Path,
    target_multiplier: float = 2.0,
    prioritize_minority: bool = True,
    verbose: bool = True
) -> dict:
    """
    Offline augmentation to increase dataset size (FetSAM-style).
    
    Args:
        data_dir: Path to data directory with train/images and train/labels
        target_multiplier: Target dataset size multiplier (2.0 = double the dataset)
        prioritize_minority: If True, augment images with CSP/LV more
        verbose: Print progress
        
    Returns:
        dict with augmentation stats
    """
    train_images = data_dir / "train" / "images"
    train_labels = data_dir / "train" / "labels"
    
    # Check marker file
    marker_file = data_dir / ".offline_augmented"
    if marker_file.exists():
        with open(marker_file) as f:
            info = json.load(f)
        if verbose:
            print(f"[SKIP] Dataset already augmented: {info.get('total_after', 'unknown')} images")
        return {"status": "skipped", "info": info}
    
    # Get original counts
    original_counts = count_class_instances(train_labels)
    original_images = list(train_images.glob("*.png")) + list(train_images.glob("*.jpg"))
    original_count = len(original_images)
    
    if verbose:
        print(f"Original dataset: {original_count} images")
        print(f"Class distribution: {original_counts}")
        print(f"Target: ~{int(original_count * target_multiplier)} images")
    
    # Categorize images by minority class presence
    images_with_csp_lv = []  # Images containing CSP or LV
    images_brain_only = []   # Images with only Brain
    
    for img_path in original_images:
        label_path = train_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        
        classes_in_file = set()
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))
        
        if 1 in classes_in_file or 2 in classes_in_file:
            images_with_csp_lv.append(img_path)
        else:
            images_brain_only.append(img_path)
    
    if verbose:
        print(f"Images with CSP/LV: {len(images_with_csp_lv)}")
        print(f"Images with Brain only: {len(images_brain_only)}")
    
    # Define augmentation transforms (anatomy-preserving)
    transforms = [
        ("hflip", "hf"),
        ("vflip", "vf"),
        ("brightness_up", "bu"),
        ("brightness_down", "bd"),
        ("contrast", "ct"),
        ("blur", "bl"),
    ]
    
    generated = 0
    target_new = int(original_count * (target_multiplier - 1))
    
    if prioritize_minority:
        # Augment CSP/LV images more (2x augments per image)
        # Augment Brain-only images less (1x augment per image)
        
        if verbose:
            print("\nPrioritizing images with CSP/LV...")
        
        # First pass: augment all CSP/LV images with multiple transforms
        for img_path in images_with_csp_lv:
            if generated >= target_new:
                break
            
            label_path = train_labels / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Apply 2-3 random transforms per minority image
            num_augs = min(3, target_new - generated)
            selected_transforms = random.sample(transforms, min(num_augs, len(transforms)))
            
            for transform_type, suffix in selected_transforms:
                if generated >= target_new:
                    break
                success = augment_single_image(
                    img_path, label_path,
                    train_images, train_labels,
                    transform_type, suffix
                )
                if success:
                    generated += 1
        
        # Second pass: augment Brain-only images with 1 transform each
        if generated < target_new:
            for img_path in images_brain_only:
                if generated >= target_new:
                    break
                
                label_path = train_labels / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                
                transform_type, suffix = random.choice(transforms)
                success = augment_single_image(
                    img_path, label_path,
                    train_images, train_labels,
                    transform_type, suffix
                )
                if success:
                    generated += 1
    else:
        # Uniform augmentation
        for img_path in original_images:
            if generated >= target_new:
                break
            
            label_path = train_labels / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            transform_type, suffix = random.choice(transforms)
            success = augment_single_image(
                img_path, label_path,
                train_images, train_labels,
                transform_type, suffix
            )
            if success:
                generated += 1
    
    # Final counts
    final_counts = count_class_instances(train_labels)
    final_images = len(list(train_images.glob("*.png"))) + len(list(train_images.glob("*.jpg")))
    
    result = {
        "status": "augmented",
        "original_images": original_count,
        "generated": generated,
        "total_after": final_images,
        "original_class_counts": original_counts,
        "final_class_counts": final_counts,
        "prioritize_minority": prioritize_minority
    }
    
    # Save marker
    with open(marker_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    if verbose:
        print(f"\nGenerated {generated} augmented samples")
        print(f"Total dataset: {final_images} images")
        print(f"Final class distribution: {final_counts}")
    
    return result


def check_augmentation_status(data_dir: Path) -> dict:
    """Check if dataset has been augmented successfully."""
    marker_file = data_dir / ".offline_augmented"
    train_labels = data_dir / "train" / "labels"
    
    counts = count_class_instances(train_labels)
    
    if marker_file.exists():
        with open(marker_file) as f:
            info = json.load(f)
        
        # Validate marker - if it shows 0 generated or 0 total, it's invalid
        if info.get("total_after", 0) == 0 or info.get("generated", 0) == 0:
            # Invalid marker from failed run - delete it
            marker_file.unlink()
            return {"augmented": False, "current_counts": counts, "note": "Invalid marker removed"}
        
        return {"augmented": True, "info": info, "current_counts": counts}
    
    return {"augmented": False, "current_counts": counts}


if __name__ == "__main__":
    from src.util.constants import DATA_DIR
    
    print("Checking augmentation status...")
    status = check_augmentation_status(DATA_DIR)
    print(f"Augmented: {status['augmented']}")
    print(f"Current counts: {status['current_counts']}")
