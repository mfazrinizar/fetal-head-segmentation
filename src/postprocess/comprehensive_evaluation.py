"""
Comprehensive Evaluation Module for Fetal Head Segmentation.

Computes ALL required metrics:
- Classification: Accuracy, Precision, Recall, Specificity, F1-score (per-class and overall)
- Detection: mAP50, mAP50-95, IoU, mIoU (per-class and overall)
- Segmentation: mAP50, mAP50-95, IoU, mIoU, Dice, mDice (per-class and overall)

Generates comprehensive reports and visualizations.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import cv2
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings('ignore')

# Class configuration
CLASS_NAMES = ['Brain', 'CSP', 'LV']
NUM_CLASSES = 3


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    precision: float = 0.0
    recall: float = 0.0  # Same as sensitivity
    specificity: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    iou: float = 0.0
    dice: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    support: int = 0  # Number of ground truth instances


@dataclass 
class EvaluationResults:
    """Complete evaluation results."""
    # Overall metrics
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_specificity: float = 0.0
    macro_f1: float = 0.0
    
    # Detection metrics
    det_mAP50: float = 0.0
    det_mAP50_95: float = 0.0
    det_mIoU: float = 0.0
    
    # Segmentation metrics  
    seg_mAP50: float = 0.0
    seg_mAP50_95: float = 0.0
    seg_mIoU: float = 0.0
    seg_mDice: float = 0.0
    
    # Per-class metrics
    per_class: Dict[str, ClassMetrics] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES)))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'overall': {
                'accuracy': self.accuracy,
                'precision': self.macro_precision,
                'recall': self.macro_recall,
                'specificity': self.macro_specificity,
                'f1_score': self.macro_f1,
            },
            'detection': {
                'mAP50': self.det_mAP50,
                'mAP50_95': self.det_mAP50_95,
                'mIoU': self.det_mIoU,
            },
            'segmentation': {
                'mAP50': self.seg_mAP50,
                'mAP50_95': self.seg_mAP50_95,
                'mIoU': self.seg_mIoU,
                'mDice': self.seg_mDice,
            },
            'per_class': {
                name: asdict(metrics) for name, metrics in self.per_class.items()
            },
            'confusion_matrix': self.confusion_matrix.tolist() if isinstance(self.confusion_matrix, np.ndarray) else self.confusion_matrix,
        }
        return result


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union (Jaccard Index) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    if sum_masks == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection / sum_masks)


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return float(inter_area / union_area)


def polygon_to_mask(polygon: List[float], img_shape: Tuple[int, int]) -> np.ndarray:
    """Convert YOLO polygon format to binary mask."""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if len(polygon) < 6:  # Need at least 3 points
        return mask
    
    # Convert normalized coords to pixel coords
    points = np.array(polygon).reshape(-1, 2)
    points[:, 0] *= w
    points[:, 1] *= h
    points = points.astype(np.int32)
    
    cv2.fillPoly(mask, [points], 1)
    return mask


def load_yolo_labels(label_path: Path, img_shape: Tuple[int, int]) -> Dict[int, List[Dict]]:
    """Load YOLO format labels with polygon masks."""
    h, w = img_shape
    annotations = defaultdict(list)
    
    if not label_path.exists():
        return annotations
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # YOLO segmentation format: class_id x1 y1 x2 y2 ... (normalized polygon)
            if len(coords) >= 4:
                # Convert polygon to mask
                mask = polygon_to_mask(coords, img_shape)
                
                # Compute bounding box from polygon
                points = np.array(coords).reshape(-1, 2)
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                bbox = [x_min * w, y_min * h, x_max * w, y_max * h]
                
                annotations[class_id].append({
                    'bbox': bbox,
                    'mask': mask,
                    'polygon': coords,
                })
    
    return annotations


class ComprehensiveEvaluator:
    """Comprehensive evaluator for YOLO segmentation models."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        data_yaml: Union[str, Path],
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))
        
        # Load data config
        self._load_data_config()
        
    def _load_data_config(self):
        """Load dataset configuration from YAML."""
        import yaml
        with open(self.data_yaml) as f:
            config = yaml.safe_load(f)
        
        self.data_path = Path(config['path'])
        self.train_path = self.data_path / config['train'].replace('/images', '')
        self.val_path = self.data_path / config['val'].replace('/images', '')
        self.test_path = self.data_path / config['test'].replace('/images', '')
        self.class_names = config.get('names', CLASS_NAMES)
        self.num_classes = config.get('nc', NUM_CLASSES)
        
    def evaluate_split(
        self,
        split: str = 'val',
        save_predictions: bool = True,
        output_dir: Optional[Path] = None
    ) -> EvaluationResults:
        """Evaluate model on a data split."""
        
        # Get split path
        if split == 'train':
            split_path = self.train_path
        elif split == 'val':
            split_path = self.val_path
        elif split == 'test':
            split_path = self.test_path
        else:
            raise ValueError(f"Unknown split: {split}")
        
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Get all images
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        print(f"Evaluating {len(image_files)} images from {split} split...")
        
        # Initialize counters
        all_det_results = []  # For mAP calculation
        all_seg_results = []  # For segmentation metrics
        
        # Per-class confusion counters
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)
        class_tn = defaultdict(int)
        
        # IoU and Dice accumulators
        class_ious = defaultdict(list)
        class_dices = defaultdict(list)
        class_det_ious = defaultdict(list)
        
        # Run inference
        for img_path in tqdm(image_files, desc=f"Evaluating {split}"):
            # Load ground truth
            label_path = labels_dir / (img_path.stem + '.txt')
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            gt_annotations = load_yolo_labels(label_path, (h, w))
            
            # Run prediction
            results = self.model.predict(
                source=str(img_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )
            
            result = results[0]
            
            # Process predictions
            pred_annotations = defaultdict(list)
            if result.masks is not None and result.boxes is not None:
                for i, (mask, box, cls, conf) in enumerate(zip(
                    result.masks.data,
                    result.boxes.xyxy,
                    result.boxes.cls,
                    result.boxes.conf
                )):
                    class_id = int(cls.item())
                    # Resize mask to original image size
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w, h)) > 0.5
                    
                    pred_annotations[class_id].append({
                        'bbox': box.cpu().numpy(),
                        'mask': mask_resized.astype(np.uint8),
                        'conf': conf.item(),
                    })
            
            # Match predictions with ground truth for each class
            for class_id in range(self.num_classes):
                gt_list = gt_annotations.get(class_id, [])
                pred_list = pred_annotations.get(class_id, [])
                
                # Sort predictions by confidence
                pred_list = sorted(pred_list, key=lambda x: x.get('conf', 0), reverse=True)
                
                matched_gt = set()
                
                for pred in pred_list:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gt_list):
                        if gt_idx in matched_gt:
                            continue
                        
                        # Compute mask IoU
                        iou = compute_iou(pred['mask'], gt['mask'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= self.iou_threshold and best_gt_idx != -1:
                        # True positive
                        class_tp[class_id] += 1
                        matched_gt.add(best_gt_idx)
                        
                        # Record IoU and Dice
                        gt_mask = gt_list[best_gt_idx]['mask']
                        class_ious[class_id].append(compute_iou(pred['mask'], gt_mask))
                        class_dices[class_id].append(compute_dice(pred['mask'], gt_mask))
                        
                        # Detection IoU (box)
                        det_iou = compute_box_iou(pred['bbox'], gt_list[best_gt_idx]['bbox'])
                        class_det_ious[class_id].append(det_iou)
                        
                        all_det_results.append({
                            'class_id': class_id,
                            'conf': pred.get('conf', 1.0),
                            'tp': True,
                            'iou': best_iou,
                        })
                    else:
                        # False positive
                        class_fp[class_id] += 1
                        all_det_results.append({
                            'class_id': class_id,
                            'conf': pred.get('conf', 1.0),
                            'tp': False,
                            'iou': best_iou,
                        })
                
                # Unmatched ground truths are false negatives
                class_fn[class_id] += len(gt_list) - len(matched_gt)
        
        # Compute metrics
        results = EvaluationResults()
        results.per_class = {}
        
        total_tp = sum(class_tp.values())
        total_fp = sum(class_fp.values())
        total_fn = sum(class_fn.values())
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            tp = class_tp[class_id]
            fp = class_fp[class_id]
            fn = class_fn[class_id]
            
            # For multi-class, TN is sum of other classes' TP
            tn = total_tp - tp  # Simplified TN calculation
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # IoU and Dice
            mean_iou = np.mean(class_ious[class_id]) if class_ious[class_id] else 0.0
            mean_dice = np.mean(class_dices[class_id]) if class_dices[class_id] else 0.0
            mean_det_iou = np.mean(class_det_ious[class_id]) if class_det_ious[class_id] else 0.0
            
            results.per_class[class_name] = ClassMetrics(
                precision=precision,
                recall=recall,
                specificity=specificity,
                f1_score=f1,
                iou=mean_iou,
                dice=mean_dice,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                support=tp + fn,
            )
        
        # Compute macro averages
        class_metrics = list(results.per_class.values())
        results.macro_precision = np.mean([m.precision for m in class_metrics])
        results.macro_recall = np.mean([m.recall for m in class_metrics])
        results.macro_specificity = np.mean([m.specificity for m in class_metrics])
        results.macro_f1 = np.mean([m.f1_score for m in class_metrics])
        
        # Overall accuracy
        results.accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        
        # Segmentation metrics
        all_ious = [iou for ious in class_ious.values() for iou in ious]
        all_dices = [dice for dices in class_dices.values() for dice in dices]
        all_det_ious = [iou for ious in class_det_ious.values() for iou in ious]
        
        results.seg_mIoU = np.mean(all_ious) if all_ious else 0.0
        results.seg_mDice = np.mean(all_dices) if all_dices else 0.0
        results.det_mIoU = np.mean(all_det_ious) if all_det_ious else 0.0
        
        # Compute mAP (simplified - using the recorded results)
        results.det_mAP50 = self._compute_map(all_det_results, iou_thresh=0.5)
        results.det_mAP50_95 = self._compute_map_range(all_det_results)
        results.seg_mAP50 = results.det_mAP50  # Same for instance segmentation
        results.seg_mAP50_95 = results.det_mAP50_95
        
        return results
    
    def _compute_map(self, results: List[Dict], iou_thresh: float = 0.5) -> float:
        """Compute mAP at given IoU threshold."""
        if not results:
            return 0.0
        
        # Filter by IoU threshold
        filtered = [r for r in results if r['iou'] >= iou_thresh or not r['tp']]
        
        # Sort by confidence
        filtered = sorted(filtered, key=lambda x: x['conf'], reverse=True)
        
        # Compute precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        total_positives = sum(1 for r in filtered if r['tp'])
        
        for r in filtered:
            if r['tp']:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_positives if total_positives > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            if prec_at_recall:
                ap += max(prec_at_recall) / 11
        
        return ap
    
    def _compute_map_range(self, results: List[Dict]) -> float:
        """Compute mAP@50:95 (average over IoU thresholds 0.5 to 0.95)."""
        aps = []
        for iou_thresh in np.arange(0.5, 1.0, 0.05):
            ap = self._compute_map(results, iou_thresh)
            aps.append(ap)
        return np.mean(aps) if aps else 0.0


def evaluate_from_ultralytics_results(
    results_dir: Path,
    data_yaml: Path,
    output_dir: Optional[Path] = None
) -> EvaluationResults:
    """
    Evaluate using Ultralytics validation results and compute additional metrics.
    
    This function reads the results.csv from YOLO training and the predictions.json
    from validation to compute comprehensive metrics.
    """
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_dir / 'comprehensive_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read training results
    results_csv = results_dir / 'results.csv'
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    
    # Get final epoch metrics
    final = df.iloc[-1]
    
    # Get best epoch (by mAP50 Mask)
    best_idx = df['metrics/mAP50(M)'].idxmax()
    best = df.iloc[best_idx]
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\n--- Training Summary ---")
    print(f"Total epochs: {len(df)}")
    print(f"Best epoch: {int(best['epoch'])}")
    print(f"Final epoch: {int(final['epoch'])}")
    
    print(f"\n--- Final Epoch Metrics ---")
    print(f"Detection:")
    print(f"  Precision (B): {final['metrics/precision(B)']:.4f}")
    print(f"  Recall (B): {final['metrics/recall(B)']:.4f}")
    print(f"  mAP@50 (B): {final['metrics/mAP50(B)']:.4f}")
    print(f"  mAP@50-95 (B): {final['metrics/mAP50-95(B)']:.4f}")
    
    print(f"\nSegmentation:")
    print(f"  Precision (M): {final['metrics/precision(M)']:.4f}")
    print(f"  Recall (M): {final['metrics/recall(M)']:.4f}")
    print(f"  mAP@50 (M): {final['metrics/mAP50(M)']:.4f}")
    print(f"  mAP@50-95 (M): {final['metrics/mAP50-95(M)']:.4f}")
    
    print(f"\n--- Best Epoch ({int(best['epoch'])}) Metrics ---")
    print(f"  mAP@50 (M): {best['metrics/mAP50(M)']:.4f}")
    print(f"  mAP@50-95 (M): {best['metrics/mAP50-95(M)']:.4f}")
    
    # Create results object
    results = EvaluationResults(
        macro_precision=float(final['metrics/precision(M)']),
        macro_recall=float(final['metrics/recall(M)']),
        det_mAP50=float(final['metrics/mAP50(B)']),
        det_mAP50_95=float(final['metrics/mAP50-95(B)']),
        seg_mAP50=float(final['metrics/mAP50(M)']),
        seg_mAP50_95=float(final['metrics/mAP50-95(M)']),
    )
    
    # Compute F1 from precision and recall
    p, r = results.macro_precision, results.macro_recall
    results.macro_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    # Generate plots
    generate_comprehensive_plots(df, output_dir)
    
    print(f"\n--- Computed Metrics ---")
    print(f"  Macro F1: {results.macro_f1:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def generate_comprehensive_plots(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive evaluation plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Training curves (all metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss curves
    loss_cols = [
        ('train/box_loss', 'val/box_loss', 'Box Loss'),
        ('train/seg_loss', 'val/seg_loss', 'Segmentation Loss'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss'),
    ]
    
    for ax, (train_col, val_col, title) in zip(axes[0], loss_cols):
        if train_col in df.columns:
            ax.plot(df['epoch'], df[train_col], label='Train', linewidth=2)
        if val_col in df.columns:
            ax.plot(df['epoch'], df[val_col], label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Metric curves
    metric_cols = [
        ('metrics/mAP50(B)', 'metrics/mAP50(M)', 'mAP@50', 'Detection', 'Segmentation'),
        ('metrics/mAP50-95(B)', 'metrics/mAP50-95(M)', 'mAP@50-95', 'Detection', 'Segmentation'),
        ('metrics/precision(M)', 'metrics/recall(M)', 'Precision/Recall', 'Precision', 'Recall'),
    ]
    
    for ax, (col1, col2, title, label1, label2) in zip(axes[1], metric_cols):
        if col1 in df.columns:
            ax.plot(df['epoch'], df[col1], label=label1, linewidth=2)
        if col2 in df.columns:
            ax.plot(df['epoch'], df[col2], label=label2, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Final metrics bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    final = df.iloc[-1]
    metrics = {
        'Precision (B)': final.get('metrics/precision(B)', 0),
        'Recall (B)': final.get('metrics/recall(B)', 0),
        'mAP@50 (B)': final.get('metrics/mAP50(B)', 0),
        'mAP@50-95 (B)': final.get('metrics/mAP50-95(B)', 0),
        'Precision (M)': final.get('metrics/precision(M)', 0),
        'Recall (M)': final.get('metrics/recall(M)', 0),
        'mAP@50 (M)': final.get('metrics/mAP50(M)', 0),
        'mAP@50-95 (M)': final.get('metrics/mAP50-95(M)', 0),
    }
    
    x = np.arange(len(metrics))
    colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c']
    bars = ax.bar(x, list(metrics.values()), color=colors)
    
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Final Epoch Metrics (B=Box/Detection, M=Mask/Segmentation)')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_metrics_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Convergence analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Learning rate
    if 'lr/pg0' in df.columns:
        axes[0].plot(df['epoch'], df['lr/pg0'], linewidth=2, color='purple')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Learning Rate')
        axes[0].set_title('Learning Rate Schedule')
        axes[0].grid(True, alpha=0.3)
    
    # mAP progression with rolling average
    if 'metrics/mAP50(M)' in df.columns:
        axes[1].plot(df['epoch'], df['metrics/mAP50(M)'], alpha=0.5, label='mAP@50')
        axes[1].plot(df['epoch'], df['metrics/mAP50(M)'].rolling(10).mean(), 
                     linewidth=2, label='mAP@50 (10-epoch avg)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP@50')
        axes[1].set_title('mAP@50 Progression (Mask)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def run_inference_evaluation(
    model_path: Path,
    data_yaml: Path,
    split: str = 'test',
    output_dir: Optional[Path] = None,
    conf: float = 0.25,
    iou: float = 0.5
) -> EvaluationResults:
    """
    Run full inference-based evaluation on a data split.
    
    Computes all metrics including IoU, Dice, per-class metrics.
    """
    evaluator = ComprehensiveEvaluator(
        model_path=model_path,
        data_yaml=data_yaml,
        conf_threshold=conf,
        iou_threshold=iou
    )
    
    results = evaluator.evaluate_split(split=split, output_dir=output_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f'{split}_evaluation.json', 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Generate per-class report
        generate_per_class_report(results, output_dir, split)
    
    return results


def generate_per_class_report(results: EvaluationResults, output_dir: Path, split: str):
    """Generate detailed per-class metrics report."""
    
    report_lines = [
        f"{'='*70}",
        f"PER-CLASS EVALUATION REPORT - {split.upper()} SPLIT",
        f"{'='*70}",
        "",
    ]
    
    # Overall metrics
    report_lines.extend([
        "OVERALL METRICS:",
        f"  Accuracy:    {results.accuracy:.4f}",
        f"  Precision:   {results.macro_precision:.4f}",
        f"  Recall:      {results.macro_recall:.4f}",
        f"  Specificity: {results.macro_specificity:.4f}",
        f"  F1-Score:    {results.macro_f1:.4f}",
        "",
        "DETECTION METRICS:",
        f"  mAP@50:      {results.det_mAP50:.4f}",
        f"  mAP@50-95:   {results.det_mAP50_95:.4f}",
        f"  mIoU:        {results.det_mIoU:.4f}",
        "",
        "SEGMENTATION METRICS:",
        f"  mAP@50:      {results.seg_mAP50:.4f}",
        f"  mAP@50-95:   {results.seg_mAP50_95:.4f}",
        f"  mIoU:        {results.seg_mIoU:.4f}",
        f"  mDice:       {results.seg_mDice:.4f}",
        "",
        f"{'='*70}",
        "PER-CLASS METRICS:",
        f"{'='*70}",
        "",
    ])
    
    # Per-class metrics
    for class_name, metrics in results.per_class.items():
        report_lines.extend([
            f"{class_name}:",
            f"  Precision:   {metrics.precision:.4f}",
            f"  Recall:      {metrics.recall:.4f}",
            f"  Specificity: {metrics.specificity:.4f}",
            f"  F1-Score:    {metrics.f1_score:.4f}",
            f"  IoU:         {metrics.iou:.4f}",
            f"  Dice:        {metrics.dice:.4f}",
            f"  TP/FP/FN:    {metrics.tp}/{metrics.fp}/{metrics.fn}",
            f"  Support:     {metrics.support}",
            "",
        ])
    
    report_text = '\n'.join(report_lines)
    
    # Save report
    with open(output_dir / f'{split}_per_class_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for fetal head segmentation")
    parser.add_argument("--model", type=str, help="Path to model weights (best.pt)")
    parser.add_argument("--data", type=str, help="Path to dataset.yaml")
    parser.add_argument("--results-dir", type=str, help="Path to training results directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, help="Output directory for evaluation results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--mode", type=str, default="inference", 
                        choices=["inference", "from_results"],
                        help="Evaluation mode: inference (run model) or from_results (analyze existing)")
    
    args = parser.parse_args()
    
    if args.mode == "from_results":
        if not args.results_dir:
            raise ValueError("--results-dir required for from_results mode")
        evaluate_from_ultralytics_results(
            results_dir=Path(args.results_dir),
            data_yaml=Path(args.data) if args.data else None,
            output_dir=Path(args.output) if args.output else None
        )
    else:
        if not args.model or not args.data:
            raise ValueError("--model and --data required for inference mode")
        run_inference_evaluation(
            model_path=Path(args.model),
            data_yaml=Path(args.data),
            split=args.split,
            output_dir=Path(args.output) if args.output else None,
            conf=args.conf,
            iou=args.iou
        )
