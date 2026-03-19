"""
Comprehensive evaluation module for fetal head segmentation.

Provides all required metrics (classification, detection, segmentation),
per-class and overall metrics, visualization generation, and report generation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.util.constants import DATA_DIR, RESULTS_DIR, CLASS_NAMES, NUM_CLASSES, IMG_SIZE


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    if sum_masks == 0:
        return 0.0
    return 2 * intersection / sum_masks


def compute_hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Hausdorff Distance between two binary masks."""
    from scipy.ndimage import distance_transform_edt

    if mask1.sum() == 0 or mask2.sum() == 0:
        return float('inf')

    boundary1 = mask1 ^ cv2.erode(mask1.astype(np.uint8), np.ones((3,3))).astype(bool)
    boundary2 = mask2 ^ cv2.erode(mask2.astype(np.uint8), np.ones((3,3))).astype(bool)

    if boundary1.sum() == 0 or boundary2.sum() == 0:
        return float('inf')

    dist1 = distance_transform_edt(~boundary1)
    dist2 = distance_transform_edt(~boundary2)

    hd1 = dist1[boundary2].max() if boundary2.any() else 0
    hd2 = dist2[boundary1].max() if boundary1.any() else 0

    return max(hd1, hd2)


def compute_average_surface_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Average Surface Distance between two binary masks."""
    from scipy.ndimage import distance_transform_edt

    if mask1.sum() == 0 or mask2.sum() == 0:
        return float('inf')

    boundary1 = mask1 ^ cv2.erode(mask1.astype(np.uint8), np.ones((3,3))).astype(bool)
    boundary2 = mask2 ^ cv2.erode(mask2.astype(np.uint8), np.ones((3,3))).astype(bool)

    if boundary1.sum() == 0 or boundary2.sum() == 0:
        return float('inf')

    dist1 = distance_transform_edt(~boundary1)
    dist2 = distance_transform_edt(~boundary2)

    asd1 = dist1[boundary2].mean() if boundary2.any() else 0
    asd2 = dist2[boundary1].mean() if boundary1.any() else 0

    return (asd1 + asd2) / 2


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    specificity = np.array(specificity)

    metrics = {
        'accuracy': accuracy,
        'precision': {
            'per_class': {CLASS_NAMES[i]: float(precision[i]) for i in range(num_classes)},
            'macro': precision.mean(),
        },
        'recall': {
            'per_class': {CLASS_NAMES[i]: float(recall[i]) for i in range(num_classes)},
            'macro': recall.mean(),
        },
        'specificity': {
            'per_class': {CLASS_NAMES[i]: float(specificity[i]) for i in range(num_classes)},
            'macro': specificity.mean(),
        },
        'f1_score': {
            'per_class': {CLASS_NAMES[i]: float(f1[i]) for i in range(num_classes)},
            'macro': f1.mean(),
        },
        'confusion_matrix': cm.tolist(),
    }

    return metrics


def compute_segmentation_metrics(pred_masks: Dict[int, np.ndarray], gt_masks: Dict[int, np.ndarray]) -> Dict:
    """Compute segmentation metrics for each class."""
    metrics = {
        'iou': {'per_class': {}, 'mean': 0},
        'dice': {'per_class': {}, 'mean': 0},
        'hausdorff_distance': {'per_class': {}, 'mean': 0},
        'average_surface_distance': {'per_class': {}, 'mean': 0},
    }

    valid_classes = 0
    total_iou, total_dice, total_hd, total_asd = 0, 0, 0, 0

    for class_id in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_id]

        pred = pred_masks.get(class_id, np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool))
        gt = gt_masks.get(class_id, np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool))

        if pred.sum() == 0 and gt.sum() == 0:
            continue

        valid_classes += 1

        iou = compute_iou(pred, gt)
        dice = compute_dice(pred, gt)
        hd = compute_hausdorff_distance(pred, gt)
        asd = compute_average_surface_distance(pred, gt)

        metrics['iou']['per_class'][class_name] = iou
        metrics['dice']['per_class'][class_name] = dice
        metrics['hausdorff_distance']['per_class'][class_name] = hd if hd != float('inf') else -1
        metrics['average_surface_distance']['per_class'][class_name] = asd if asd != float('inf') else -1

        total_iou += iou
        total_dice += dice
        if hd != float('inf'):
            total_hd += hd
        if asd != float('inf'):
            total_asd += asd

    if valid_classes > 0:
        metrics['iou']['mean'] = total_iou / valid_classes
        metrics['dice']['mean'] = total_dice / valid_classes
        metrics['hausdorff_distance']['mean'] = total_hd / valid_classes
        metrics['average_surface_distance']['mean'] = total_asd / valid_classes

    return metrics


def evaluate_yolo_results(results_path: Path, output_dir: Path) -> Dict:
    """Evaluate YOLO training results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_path / "results.csv"
    if not metrics_path.exists():
        print(f"WARNING: Results not found at {metrics_path}")
        return {}

    import pandas as pd
    df = pd.read_csv(metrics_path)
    final_metrics = df.iloc[-1].to_dict()
    final_metrics = {k.strip(): v for k, v in final_metrics.items()}

    evaluation_results = {
        'final_epoch': int(df['epoch'].iloc[-1]) if 'epoch' in df.columns else len(df),
        'detection_metrics': {},
        'segmentation_metrics': {},
        'training_history': df.to_dict('list'),
    }

    det_metrics_map = {
        'metrics/precision(B)': 'precision_box',
        'metrics/recall(B)': 'recall_box',
        'metrics/mAP50(B)': 'mAP50_box',
        'metrics/mAP50-95(B)': 'mAP50_95_box',
    }

    for yolo_key, metric_name in det_metrics_map.items():
        if yolo_key in final_metrics:
            evaluation_results['detection_metrics'][metric_name] = final_metrics[yolo_key]

    seg_metrics_map = {
        'metrics/precision(M)': 'precision_mask',
        'metrics/recall(M)': 'recall_mask',
        'metrics/mAP50(M)': 'mAP50_mask',
        'metrics/mAP50-95(M)': 'mAP50_95_mask',
    }

    for yolo_key, metric_name in seg_metrics_map.items():
        if yolo_key in final_metrics:
            evaluation_results['segmentation_metrics'][metric_name] = final_metrics[yolo_key]

    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    return evaluation_results


def generate_evaluation_plots(results_path: Path, output_dir: Path) -> None:
    """Generate evaluation visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    metrics_path = results_path / "results.csv"
    if not metrics_path.exists():
        print(f"WARNING: Results not found at {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    df.columns = [c.strip() for c in df.columns]

    plt.style.use('seaborn-v0_8-whitegrid')

    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    loss_cols = ['train/box_loss', 'train/seg_loss', 'train/cls_loss']
    titles = ['Box Loss', 'Segmentation Loss', 'Classification Loss']

    for ax, col, title in zip(axes, loss_cols, titles):
        if col in df.columns:
            ax.plot(df['epoch'], df[col], label='Train', color='blue')
            if col.replace('train/', 'val/') in df.columns:
                ax.plot(df['epoch'], df[col.replace('train/', 'val/')], label='Val', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # mAP curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if 'metrics/mAP50(B)' in df.columns:
        axes[0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='blue')
    if 'metrics/mAP50-95(B)' in df.columns:
        axes[0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('mAP')
    axes[0].set_title('Detection mAP')
    axes[0].legend()

    if 'metrics/mAP50(M)' in df.columns:
        axes[1].plot(df['epoch'], df['metrics/mAP50(M)'], label='mAP@50', color='blue')
    if 'metrics/mAP50-95(M)' in df.columns:
        axes[1].plot(df['epoch'], df['metrics/mAP50-95(M)'], label='mAP@50-95', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].set_title('Segmentation mAP')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'mAP_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Precision-Recall curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        axes[0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
        axes[0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Detection Precision/Recall')
        axes[0].legend()

    if 'metrics/precision(M)' in df.columns and 'metrics/recall(M)' in df.columns:
        axes[1].plot(df['epoch'], df['metrics/precision(M)'], label='Precision', color='blue')
        axes[1].plot(df['epoch'], df['metrics/recall(M)'], label='Recall', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Segmentation Precision/Recall')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Evaluation plots saved to: {output_dir}")


def run_full_evaluation(experiment_name: str) -> Dict:
    """Run full evaluation for an experiment."""
    results_path = RESULTS_DIR / experiment_name
    output_dir = results_path / "evaluation"

    if not results_path.exists():
        raise ValueError(f"Experiment results not found: {results_path}")

    print(f"\n{'='*60}")
    print(f"EVALUATING: {experiment_name}")
    print(f"{'='*60}")

    results = evaluate_yolo_results(results_path, output_dir)
    generate_evaluation_plots(results_path, output_dir)

    print("\n--- Detection Metrics ---")
    for metric, value in results.get('detection_metrics', {}).items():
        print(f"  {metric}: {value:.4f}")

    print("\n--- Segmentation Metrics ---")
    for metric, value in results.get('segmentation_metrics', {}).items():
        print(f"  {metric}: {value:.4f}")

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fetal head segmentation model")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name to evaluate")
    args = parser.parse_args()

    run_full_evaluation(args.experiment)
