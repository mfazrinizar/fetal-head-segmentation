"""
Domain-guided data augmentation for fetal head segmentation.
Based on: https://pmc.ncbi.nlm.nih.gov/articles/PMC10035842/

Key concept: Context-preserving cut-paste that maintains anatomical relationships.
- CSP and LV are always within Brain region
- We extract CSP/LV from donor images and paste into acceptor Brain regions
- Preserves spatial relationships unlike naive copy-paste
"""
import json
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from typing import Optional, Tuple, List


def parse_yolo_label(label_path: Path) -> List[dict]:
    """Parse YOLO segmentation label file."""
    objects = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            polygon = [float(p) for p in parts[1:]]
            objects.append({"class": cls_id, "polygon": polygon})
    return objects


def polygon_to_mask(polygon: List[float], img_w: int, img_h: int) -> np.ndarray:
    """Convert normalized YOLO polygon to binary mask."""
    points = np.array(polygon).reshape(-1, 2)
    points[:, 0] *= img_w
    points[:, 1] *= img_h
    points = points.astype(np.int32)
    
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def mask_to_polygon(mask: np.ndarray) -> Optional[List[float]]:
    """Convert binary mask to normalized YOLO polygon."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    
    h, w = mask.shape
    points = contour.reshape(-1, 2).astype(float)
    points[:, 0] /= w
    points[:, 1] /= h
    
    return points.flatten().tolist()


def get_mask_centroid(mask: np.ndarray) -> Tuple[int, int]:
    """Get centroid of binary mask."""
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        h, w = mask.shape
        return w // 2, h // 2
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box of mask (x, y, w, h)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0
    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def domain_guided_paste(
    acceptor_img: np.ndarray,
    acceptor_objects: List[dict],
    donor_img: np.ndarray,
    donor_objects: List[dict],
    target_class: int,  # 1=CSP or 2=LV
    blend_alpha: float = 0.95
) -> Tuple[Optional[np.ndarray], Optional[List[dict]]]:
    """
    Domain-guided copy-paste: paste CSP/LV from donor into acceptor's Brain region.
    
    Args:
        acceptor_img: Image to paste into (must have Brain)
        acceptor_objects: YOLO objects in acceptor
        donor_img: Image to copy from (must have target_class)
        donor_objects: YOLO objects in donor
        target_class: Class to copy (1=CSP, 2=LV)
        blend_alpha: Blending factor for seamless paste
        
    Returns:
        (augmented_image, new_objects) or (None, None) if invalid
    """
    h, w = acceptor_img.shape[:2]
    
    # Find Brain in acceptor (class 0)
    acceptor_brain = None
    for obj in acceptor_objects:
        if obj["class"] == 0:
            acceptor_brain = obj
            break
    
    if acceptor_brain is None:
        return None, None
    
    # Find target object in donor
    donor_target = None
    donor_brain = None
    for obj in donor_objects:
        if obj["class"] == target_class:
            donor_target = obj
        if obj["class"] == 0:
            donor_brain = obj
    
    if donor_target is None or donor_brain is None:
        return None, None
    
    # Create masks
    acceptor_brain_mask = polygon_to_mask(acceptor_brain["polygon"], w, h)
    donor_brain_mask = polygon_to_mask(donor_brain["polygon"], w, h)
    donor_target_mask = polygon_to_mask(donor_target["polygon"], w, h)
    
    # Get relative position of target within donor brain
    donor_brain_cx, donor_brain_cy = get_mask_centroid(donor_brain_mask)
    donor_target_cx, donor_target_cy = get_mask_centroid(donor_target_mask)
    
    # Relative offset
    rel_x = donor_target_cx - donor_brain_cx
    rel_y = donor_target_cy - donor_brain_cy
    
    # Get acceptor brain centroid
    acceptor_brain_cx, acceptor_brain_cy = get_mask_centroid(acceptor_brain_mask)
    
    # Calculate paste position (maintaining relative position)
    paste_x = acceptor_brain_cx + rel_x
    paste_y = acceptor_brain_cy + rel_y
    
    # Get donor target bounding box
    tx, ty, tw, th = get_mask_bbox(donor_target_mask)
    if tw == 0 or th == 0:
        return None, None
    
    # Extract donor target region
    donor_crop = donor_img[ty:ty+th, tx:tx+tw].copy()
    target_crop_mask = donor_target_mask[ty:ty+th, tx:tx+tw]
    
    # Calculate paste region in acceptor
    paste_x1 = paste_x - tw // 2
    paste_y1 = paste_y - th // 2
    paste_x2 = paste_x1 + tw
    paste_y2 = paste_y1 + th
    
    # Clip to image bounds
    src_x1 = max(0, -paste_x1)
    src_y1 = max(0, -paste_y1)
    src_x2 = tw - max(0, paste_x2 - w)
    src_y2 = th - max(0, paste_y2 - h)
    
    paste_x1 = max(0, paste_x1)
    paste_y1 = max(0, paste_y1)
    paste_x2 = min(w, paste_x2)
    paste_y2 = min(h, paste_y2)
    
    if paste_x2 <= paste_x1 or paste_y2 <= paste_y1:
        return None, None
    
    # Ensure paste is within brain region
    paste_region_mask = np.zeros((h, w), dtype=np.uint8)
    paste_region_mask[paste_y1:paste_y2, paste_x1:paste_x2] = 255
    overlap = cv2.bitwise_and(paste_region_mask, acceptor_brain_mask)
    overlap_ratio = np.sum(overlap > 0) / max(1, np.sum(paste_region_mask > 0))
    
    if overlap_ratio < 0.7:  # At least 70% must be within brain
        return None, None
    
    # Create augmented image
    aug_img = acceptor_img.copy()
    
    # Crop source region to match destination
    donor_crop = donor_crop[src_y1:src_y2, src_x1:src_x2]
    target_crop_mask = target_crop_mask[src_y1:src_y2, src_x1:src_x2]
    
    # Blend paste
    mask_3ch = np.stack([target_crop_mask] * 3, axis=-1) / 255.0
    acceptor_region = aug_img[paste_y1:paste_y2, paste_x1:paste_x2]
    
    if acceptor_region.shape != donor_crop.shape:
        return None, None
    
    blended = (blend_alpha * donor_crop * mask_3ch + 
               (1 - blend_alpha * mask_3ch) * acceptor_region).astype(np.uint8)
    aug_img[paste_y1:paste_y2, paste_x1:paste_x2] = blended
    
    # Create new polygon for pasted object
    new_mask = np.zeros((h, w), dtype=np.uint8)
    new_mask[paste_y1:paste_y2, paste_x1:paste_x2] = target_crop_mask
    new_polygon = mask_to_polygon(new_mask)
    
    if new_polygon is None:
        return None, None
    
    # Combine objects: acceptor objects + new pasted object
    new_objects = acceptor_objects.copy()
    new_objects.append({"class": target_class, "polygon": new_polygon})
    
    return aug_img, new_objects


def save_yolo_labels(objects: List[dict], label_path: Path):
    """Save objects to YOLO label format."""
    with open(label_path, 'w') as f:
        for obj in objects:
            polygon_str = " ".join(f"{p:.6f}" for p in obj["polygon"])
            f.write(f"{obj['class']} {polygon_str}\n")


def run_domain_guided_augmentation(
    data_dir: Path,
    target_multiplier: float = 1.5,
    verbose: bool = True
) -> dict:
    """
    Run domain-guided augmentation on dataset.
    
    Pastes CSP/LV from donor images into acceptor Brain regions.
    """
    train_images = data_dir / "train" / "images"
    train_labels = data_dir / "train" / "labels"
    
    marker_file = data_dir / ".domain_guided_augmented"
    if marker_file.exists():
        with open(marker_file) as f:
            info = json.load(f)
        if verbose:
            print(f"[SKIP] Domain-guided augmentation already done")
        return {"status": "skipped", "info": info}
    
    # Categorize images
    images_with_brain_only = []  # Have Brain, no CSP/LV (acceptors)
    images_with_csp = []  # Have CSP (donors for CSP)
    images_with_lv = []   # Have LV (donors for LV)
    
    all_images = list(train_images.glob("*.png")) + list(train_images.glob("*.jpg"))
    
    for img_path in all_images:
        label_path = train_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        
        objects = parse_yolo_label(label_path)
        classes = {obj["class"] for obj in objects}
        
        if 0 in classes and 1 not in classes and 2 not in classes:
            images_with_brain_only.append(img_path)
        if 1 in classes:
            images_with_csp.append(img_path)
        if 2 in classes:
            images_with_lv.append(img_path)
    
    if verbose:
        print(f"Acceptors (Brain only): {len(images_with_brain_only)}")
        print(f"CSP donors: {len(images_with_csp)}")
        print(f"LV donors: {len(images_with_lv)}")
    
    generated = 0
    target_new = int(len(all_images) * (target_multiplier - 1))
    
    if verbose:
        print(f"Target: generate {target_new} new images")
    
    # Generate augmented images
    random.shuffle(images_with_brain_only)
    
    for acceptor_path in images_with_brain_only:
        if generated >= target_new:
            break
        
        acceptor_label_path = train_labels / f"{acceptor_path.stem}.txt"
        acceptor_img = cv2.imread(str(acceptor_path))
        if acceptor_img is None:
            continue
        acceptor_objects = parse_yolo_label(acceptor_label_path)
        
        # Try to paste CSP
        if images_with_csp and generated < target_new:
            donor_path = random.choice(images_with_csp)
            donor_label_path = train_labels / f"{donor_path.stem}.txt"
            donor_img = cv2.imread(str(donor_path))
            if donor_img is not None:
                donor_objects = parse_yolo_label(donor_label_path)
                
                aug_img, new_objects = domain_guided_paste(
                    acceptor_img, acceptor_objects,
                    donor_img, donor_objects,
                    target_class=1  # CSP
                )
                
                if aug_img is not None:
                    # Save
                    suffix = f"dg_csp_{generated}"
                    new_img_path = train_images / f"{acceptor_path.stem}_{suffix}{acceptor_path.suffix}"
                    new_label_path = train_labels / f"{acceptor_path.stem}_{suffix}.txt"
                    
                    cv2.imwrite(str(new_img_path), aug_img)
                    save_yolo_labels(new_objects, new_label_path)
                    generated += 1
        
        # Try to paste LV
        if images_with_lv and generated < target_new:
            donor_path = random.choice(images_with_lv)
            donor_label_path = train_labels / f"{donor_path.stem}.txt"
            donor_img = cv2.imread(str(donor_path))
            if donor_img is not None:
                donor_objects = parse_yolo_label(donor_label_path)
                
                aug_img, new_objects = domain_guided_paste(
                    acceptor_img, acceptor_objects,
                    donor_img, donor_objects,
                    target_class=2  # LV
                )
                
                if aug_img is not None:
                    suffix = f"dg_lv_{generated}"
                    new_img_path = train_images / f"{acceptor_path.stem}_{suffix}{acceptor_path.suffix}"
                    new_label_path = train_labels / f"{acceptor_path.stem}_{suffix}.txt"
                    
                    cv2.imwrite(str(new_img_path), aug_img)
                    save_yolo_labels(new_objects, new_label_path)
                    generated += 1
    
    result = {
        "status": "augmented",
        "generated": generated,
        "acceptors_used": len(images_with_brain_only),
        "csp_donors": len(images_with_csp),
        "lv_donors": len(images_with_lv)
    }
    
    with open(marker_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    if verbose:
        print(f"Generated {generated} domain-guided augmented images")
    
    return result


def check_domain_guided_status(data_dir: Path) -> dict:
    """Check if domain-guided augmentation has been applied."""
    marker_file = data_dir / ".domain_guided_augmented"
    if marker_file.exists():
        with open(marker_file) as f:
            return {"done": True, "info": json.load(f)}
    return {"done": False}


if __name__ == "__main__":
    from src.util.constants import DATA_DIR
    print("Domain-guided augmentation module")
    status = check_domain_guided_status(DATA_DIR)
    print(f"Status: {status}")
