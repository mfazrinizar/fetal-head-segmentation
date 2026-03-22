"""
Test-Time Augmentation (TTA) for YOLO Segmentation Models.

YOLO26 does not natively support `augment=True` for TTA. This module
implements TTA by running predictions on geometric transforms and
merging results via NMS + confidence-weighted mask averaging.

Transforms applied:
- Original (identity)
- Horizontal flip
- Vertical flip

Usage with ComprehensiveEvaluator:
    from src.postprocess.tta import SegmentationTTA
    evaluator = ComprehensiveEvaluator(...)
    evaluator.model = SegmentationTTA(evaluator.model)
    results = evaluator.evaluate_split(split='test', ...)
"""

import cv2
import numpy as np
import torch
from types import SimpleNamespace
from typing import List, Optional


class SegmentationTTA:
    """Test-Time Augmentation wrapper for YOLO segmentation models.

    Drop-in replacement for a YOLO model's predict interface.
    Runs prediction on multiple geometric transforms, un-transforms
    the results, and merges via per-class NMS + weighted mask averaging.
    """

    def __init__(
        self,
        model,
        transforms: Optional[List[str]] = None,
        nms_iou_threshold: float = 0.5,
    ):
        """
        Args:
            model: Loaded YOLO model instance.
            transforms: List of transforms to apply.
                Supported: 'identity', 'hflip', 'vflip', 'hvflip'.
                Default: ['identity', 'hflip', 'vflip'].
            nms_iou_threshold: IoU threshold for NMS deduplication.
        """
        self.model = model
        self.transforms = transforms or ["identity", "hflip", "vflip"]
        self.nms_iou_threshold = nms_iou_threshold

    def predict(self, source, **kwargs):
        """Run TTA prediction.

        Accepts the same arguments as YOLO model.predict().
        Returns a list with one result matching the ultralytics Results
        interface (result.masks.data, result.boxes.xyxy/cls/conf).
        """
        # Load image
        if isinstance(source, (str,)):
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
        else:
            img = source
        h, w = img.shape[:2]

        # Remove augment kwarg to avoid YOLO warning
        kwargs.pop("augment", None)

        # Collect predictions from all transforms
        all_masks = []
        all_boxes = []
        all_classes = []
        all_confs = []

        for transform in self.transforms:
            aug_img = _apply_transform(img, transform)

            results = self.model.predict(source=aug_img, **kwargs)
            result = results[0]

            if result.masks is None or result.boxes is None:
                continue

            for mask, box, cls, conf in zip(
                result.masks.data,
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf,
            ):
                mask_np = mask.cpu().numpy()
                box_np = box.cpu().numpy()
                mh, mw = mask_np.shape[:2]

                # Un-transform mask (at model resolution) and box (at image resolution)
                mask_orig = _undo_transform_mask(mask_np, transform, mh, mw)
                box_orig = _undo_transform_box(box_np, transform, h, w)

                all_masks.append(mask_orig)
                all_boxes.append(box_orig)
                all_classes.append(int(cls.item()))
                all_confs.append(conf.item())

        # No detections from any transform
        if not all_masks:
            return [_empty_result()]

        all_masks = np.array(all_masks)
        all_boxes = np.array(all_boxes)
        all_classes = np.array(all_classes)
        all_confs = np.array(all_confs)

        # Per-class NMS + merge
        final_masks, final_boxes, final_classes, final_confs = [], [], [], []
        used = set()

        for cls_id in sorted(set(all_classes)):
            cls_idx = np.where(all_classes == cls_id)[0]
            cls_boxes = all_boxes[cls_idx]
            cls_confs = all_confs[cls_idx]
            cls_masks = all_masks[cls_idx]

            keep = _nms(cls_boxes, cls_confs, self.nms_iou_threshold)

            for k in keep:
                global_k = cls_idx[k]
                if global_k in used:
                    continue

                # Find overlapping detections to merge masks
                overlapping_local = [k]
                for j in range(len(cls_boxes)):
                    if j == k:
                        continue
                    global_j = cls_idx[j]
                    if global_j in used:
                        continue
                    if _box_iou(cls_boxes[k], cls_boxes[j]) > self.nms_iou_threshold * 0.7:
                        overlapping_local.append(j)
                        used.add(global_j)

                used.add(global_k)

                # Weighted average of overlapping masks
                if len(overlapping_local) > 1:
                    weights = cls_confs[overlapping_local]
                    weights = weights / weights.sum()
                    merged = np.zeros_like(cls_masks[0], dtype=np.float32)
                    for idx, weight in zip(overlapping_local, weights):
                        merged += weight * cls_masks[idx].astype(np.float32)
                    merged_mask = merged  # keep as float for downstream thresholding
                else:
                    merged_mask = cls_masks[k].astype(np.float32)

                avg_conf = cls_confs[overlapping_local].mean()

                final_masks.append(merged_mask)
                final_boxes.append(cls_boxes[k])
                final_classes.append(cls_id)
                final_confs.append(float(avg_conf))

        if not final_masks:
            return [_empty_result()]

        result = SimpleNamespace(
            masks=SimpleNamespace(data=torch.tensor(np.array(final_masks))),
            boxes=SimpleNamespace(
                xyxy=torch.tensor(np.array(final_boxes), dtype=torch.float32),
                cls=torch.tensor(final_classes, dtype=torch.float32),
                conf=torch.tensor(final_confs, dtype=torch.float32),
            ),
        )
        return [result]

# Helper functions (module-level for pickling compatibility)

def _empty_result():
    return SimpleNamespace(
        masks=None,
        boxes=SimpleNamespace(
            xyxy=torch.zeros((0, 4)),
            cls=torch.zeros(0),
            conf=torch.zeros(0),
        ),
    )


def _apply_transform(img, transform):
    if transform == "identity":
        return img.copy()
    elif transform == "hflip":
        return cv2.flip(img, 1)
    elif transform == "vflip":
        return cv2.flip(img, 0)
    elif transform == "hvflip":
        return cv2.flip(img, -1)
    raise ValueError(f"Unknown transform: {transform}")


def _undo_transform_mask(mask, transform, h, w):
    # Flips are self-inverse
    if transform == "identity":
        return mask
    elif transform == "hflip":
        return cv2.flip(mask, 1)
    elif transform == "vflip":
        return cv2.flip(mask, 0)
    elif transform == "hvflip":
        return cv2.flip(mask, -1)
    raise ValueError(f"Unknown transform: {transform}")


def _undo_transform_box(box, transform, h, w):
    """Undo transform on an xyxy box."""
    x1, y1, x2, y2 = box
    if transform == "identity":
        return box
    elif transform == "hflip":
        return np.array([w - x2, y1, w - x1, y2])
    elif transform == "vflip":
        return np.array([x1, h - y2, x2, h - y1])
    elif transform == "hvflip":
        return np.array([w - x2, h - y2, w - x1, h - y1])
    raise ValueError(f"Unknown transform: {transform}")


def _box_iou(box1, box2):
    """IoU between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, scores, threshold):
    """Greedy NMS. Returns list of kept indices."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        remaining = order[1:]
        ious = np.array([_box_iou(boxes[i], boxes[j]) for j in remaining])
        order = remaining[ious < threshold]
    return keep
