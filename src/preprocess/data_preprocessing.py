"""
Data preprocessing module for fetal head segmentation.

This module handles:
1. Loading COCO format annotations
2. Converting RLE masks to YOLO polygon format
3. Inter-patient train/val/test splitting
4. Creating YOLO-compatible dataset structure
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.util.constants import (
    RAW_DATA_DIR, DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASSES, CLASS_NAMES, PLANES, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    ORIGINAL_WIDTH, ORIGINAL_HEIGHT, IMG_SIZE,
    ensure_dirs
)


def decode_rle_to_mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """Decode RLE (Run-Length Encoding) to binary mask."""
    counts = rle['counts']
    size = rle['size']
    
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Odd indices are foreground
            mask[pos:pos + count] = 1
        pos += count
    
    mask = mask.reshape(size, order='F')  # Fortran order (column-major)
    return mask


def mask_to_polygon(mask: np.ndarray, simplify: bool = True, epsilon: float = 2.0) -> List[List[float]]:
    """
    Convert binary mask to polygon coordinates.
    
    Args:
        mask: Binary mask (H, W)
        simplify: Whether to simplify the polygon
        epsilon: Simplification tolerance
        
    Returns:
        List of polygons, each polygon is a flat list of normalized coordinates [x1, y1, x2, y2, ...]
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    polygons = []
    h, w = mask.shape
    
    for contour in contours:
        # Simplify contour if requested
        if simplify:
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Need at least 3 points for a valid polygon
        if len(contour) < 3:
            continue
        
        # Flatten and normalize coordinates
        polygon = []
        for point in contour:
            x, y = point[0]
            # Normalize to [0, 1]
            polygon.extend([x / w, y / h])
        
        polygons.append(polygon)
    
    return polygons


def get_image_directory(plane: str) -> Path:
    """Get the correct image directory for a given plane."""
    plane_path = RAW_DATA_DIR / plane
    
    if plane == "Trans-cerebellum":
        return plane_path / "Trans-cerebellum"
    elif plane == "Trans-thalamic":
        return plane_path / "Trans-thalamic"
    elif plane == "Trans-ventricular":
        return plane_path / "Trans-ventricular"
    else:  # Diverse Fetal Head Images
        return plane_path / "Orginal_train_images_to_959_661"


def get_segmentation_directory(plane: str) -> Path:
    """Get the segmentation mask directory for a given plane."""
    plane_path = RAW_DATA_DIR / plane
    
    if plane == "Trans-cerebellum":
        return plane_path / "Trans-cerebellum-Segmentation" / "SegmentationClass"
    elif plane == "Trans-thalamic":
        return plane_path / "Trans-thalamic-Segmentation" / "SegmentationClass"
    elif plane == "Trans-ventricular":
        return plane_path / "Trans-ventricular-segmentation-mask" / "SegmentationClass"
    else:  # Diverse Fetal Head Images
        return plane_path / "Test-Dataset-Segmentation" / "SegmentationClass"


# Color mapping for segmentation masks (RGB format as stored in BGR by OpenCV)
# labelmap: Brain:255,0,0 -> BGR (0,0,255), CSP:0,255,0 -> BGR (0,255,0), LV:0,0,255 -> BGR (255,0,0)
MASK_COLORS = {
    'Brain': (0, 0, 255),    # Red in BGR
    'CSP': (0, 255, 0),      # Green in BGR
    'LV': (255, 0, 0),       # Blue in BGR
}

CLASS_TO_ID = {'Brain': 0, 'CSP': 1, 'LV': 2}


def extract_class_mask(segmentation_mask: np.ndarray, class_name: str) -> np.ndarray:
    """Extract binary mask for a specific class from RGB segmentation mask."""
    color = MASK_COLORS[class_name]
    # Create binary mask where all channels match the class color
    mask = np.all(segmentation_mask == color, axis=2).astype(np.uint8)
    return mask


def load_all_annotations() -> Dict:
    """Load all COCO annotations from all planes."""
    all_data = {
        'images': [],
        'annotations': [],
        'categories': None,
        'image_to_plane': {},
        'patient_to_images': defaultdict(list),
    }
    
    image_id_offset = 0
    ann_id_offset = 0
    
    for plane in PLANES:
        plane_path = RAW_DATA_DIR / plane
        ann_path = plane_path / "annotations" / "instances_default.json"
        
        if not ann_path.exists():
            print(f"WARNING: Annotations not found for {plane}")
            continue
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Set categories from first plane (should be same for all)
        if all_data['categories'] is None:
            all_data['categories'] = data['categories']
        
        # Process images
        old_to_new_img_id = {}
        for img in data['images']:
            old_id = img['id']
            new_id = old_id + image_id_offset
            old_to_new_img_id[old_id] = new_id
            
            # Extract patient ID
            patient_id = img['file_name'].split('_Plane')[0]
            
            img_info = {
                'id': new_id,
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'plane': plane,
                'patient_id': patient_id,
            }
            
            all_data['images'].append(img_info)
            all_data['image_to_plane'][new_id] = plane
            all_data['patient_to_images'][patient_id].append(new_id)
        
        # Process annotations
        for ann in data['annotations']:
            new_ann = {
                'id': ann['id'] + ann_id_offset,
                'image_id': old_to_new_img_id[ann['image_id']],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'segmentation': ann.get('segmentation'),
                'area': ann.get('area', 0),
            }
            all_data['annotations'].append(new_ann)
        
        # Update offsets
        if data['images']:
            image_id_offset = max(img['id'] for img in all_data['images']) + 1
        if data['annotations']:
            ann_id_offset = max(ann['id'] for ann in all_data['annotations']) + 1
    
    return all_data


def inter_patient_split(
    patient_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Split patients into train/val/test sets.
    
    Args:
        patient_ids: List of patient IDs
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_patients, val_patients, test_patients)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    patients = list(patient_ids)
    random.shuffle(patients)
    
    n = len(patients)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_patients = set(patients[:train_end])
    val_patients = set(patients[train_end:val_end])
    test_patients = set(patients[val_end:])
    
    return train_patients, val_patients, test_patients


def convert_to_yolo_format(
    all_data: Dict,
    output_dir: Path,
    split: str,
    image_ids: Set[int],
    resize_to: Optional[int] = None
) -> Dict:
    """
    Convert annotations to YOLO format using segmentation masks.
    
    Args:
        all_data: All loaded annotations
        output_dir: Output directory (train/val/test)
        split: Split name
        image_ids: Set of image IDs to include
        resize_to: Optional resize dimension
        
    Returns:
        Statistics about conversion
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'images_with_annotations': 0,
        'total_annotations': 0,
        'class_counts': {name: 0 for name in CLASS_NAMES},
        'failed_conversions': 0,
        'missing_masks': 0,
    }
    
    # Process each image
    for img_info in tqdm(all_data['images'], desc=f"Processing {split}"):
        if img_info['id'] not in image_ids:
            continue
        
        stats['total_images'] += 1
        
        # Get image path
        plane = img_info['plane']
        img_dir = get_image_directory(plane)
        img_path = img_dir / img_info['file_name']
        
        if not img_path.exists():
            print(f"WARNING: Image not found: {img_path}")
            continue
        
        # Get segmentation mask path
        seg_dir = get_segmentation_directory(plane)
        mask_path = seg_dir / img_info['file_name']
        
        # Copy image
        dest_img_path = images_dir / img_info['file_name']
        shutil.copy2(img_path, dest_img_path)
        
        # Process segmentation mask
        yolo_lines = []
        
        if mask_path.exists():
            seg_mask = cv2.imread(str(mask_path))
            
            if seg_mask is not None:
                h, w = seg_mask.shape[:2]
                
                # Extract polygons for each class
                for class_name in CLASS_NAMES:
                    class_mask = extract_class_mask(seg_mask, class_name)
                    
                    # Check if class exists in this image
                    if np.sum(class_mask) == 0:
                        continue
                    
                    # Convert to polygons
                    polygons = mask_to_polygon(class_mask, simplify=True, epsilon=2.0)
                    
                    for polygon in polygons:
                        if len(polygon) >= 6:  # At least 3 points
                            class_id = CLASS_TO_ID[class_name]
                            line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in polygon)
                            yolo_lines.append(line)
                            stats['class_counts'][class_name] += 1
                            stats['total_annotations'] += 1
            else:
                stats['failed_conversions'] += 1
        else:
            stats['missing_masks'] += 1
        
        # Write label file
        label_path = labels_dir / (img_info['file_name'].rsplit('.', 1)[0] + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        if yolo_lines:
            stats['images_with_annotations'] += 1
    
    return stats


def create_dataset_yaml(output_dir: Path):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# Fetal Head Segmentation Dataset
# Auto-generated configuration file

path: {output_dir}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (optional)

# Classes
names:
  0: Brain
  1: CSP
  2: LV

# Number of classes
nc: 3
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset config: {yaml_path}")
    return yaml_path


def run_preprocessing(seed: int = 42):
    """Run the complete preprocessing pipeline."""
    print("=" * 60)
    print("FETAL HEAD SEGMENTATION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_dirs()
    
    # Step 1: Load all annotations
    print("\n[1/5] Loading annotations from all planes...")
    all_data = load_all_annotations()
    
    print(f"  Total images: {len(all_data['images'])}")
    print(f"  Total annotations: {len(all_data['annotations'])}")
    print(f"  Total patients: {len(all_data['patient_to_images'])}")
    
    # Step 2: Perform inter-patient split
    print("\n[2/5] Performing inter-patient split (70/15/15)...")
    patient_ids = list(all_data['patient_to_images'].keys())
    
    train_patients, val_patients, test_patients = inter_patient_split(
        patient_ids,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=seed
    )
    
    print(f"  Train patients: {len(train_patients)}")
    print(f"  Val patients: {len(val_patients)}")
    print(f"  Test patients: {len(test_patients)}")
    
    # Verify no overlap
    assert len(train_patients & val_patients) == 0, "Train/Val overlap!"
    assert len(train_patients & test_patients) == 0, "Train/Test overlap!"
    assert len(val_patients & test_patients) == 0, "Val/Test overlap!"
    print("  ✓ No patient overlap between splits")
    
    # Get image IDs for each split
    train_images = set()
    val_images = set()
    test_images = set()
    
    for patient_id, img_ids in all_data['patient_to_images'].items():
        if patient_id in train_patients:
            train_images.update(img_ids)
        elif patient_id in val_patients:
            val_images.update(img_ids)
        else:
            test_images.update(img_ids)
    
    print(f"\n  Train images: {len(train_images)}")
    print(f"  Val images: {len(val_images)}")
    print(f"  Test images: {len(test_images)}")
    
    # Step 3: Convert to YOLO format
    print("\n[3/5] Converting to YOLO format...")
    
    train_stats = convert_to_yolo_format(all_data, TRAIN_DIR, "train", train_images)
    val_stats = convert_to_yolo_format(all_data, VAL_DIR, "val", val_images)
    test_stats = convert_to_yolo_format(all_data, TEST_DIR, "test", test_images)
    
    # Step 4: Create dataset.yaml
    print("\n[4/5] Creating dataset configuration...")
    create_dataset_yaml(DATA_DIR)
    
    # Step 5: Print summary
    print("\n[5/5] Preprocessing Summary")
    print("=" * 60)
    
    for split_name, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        print(f"\n{split_name}:")
        print(f"  Images: {stats['total_images']}")
        print(f"  Images with annotations: {stats['images_with_annotations']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Class distribution:")
        for cls_name, count in stats['class_counts'].items():
            print(f"    {cls_name}: {count}")
        if stats['failed_conversions'] > 0:
            print(f"  Failed conversions: {stats['failed_conversions']}")
    
    # Save split info
    split_info = {
        'seed': seed,
        'train_patients': list(train_patients),
        'val_patients': list(val_patients),
        'test_patients': list(test_patients),
        'train_images': len(train_images),
        'val_images': len(val_images),
        'test_images': len(test_images),
        'train_stats': train_stats,
        'val_stats': val_stats,
        'test_stats': test_stats,
    }
    
    with open(DATA_DIR / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print(f"Dataset saved to: {DATA_DIR}")
    print("=" * 60)
    
    return split_info


if __name__ == "__main__":
    run_preprocessing(seed=42)
