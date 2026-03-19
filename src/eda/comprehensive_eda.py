"""
Comprehensive Exploratory Data Analysis for Fetal Head Ultrasound Dataset.

Performs detailed analysis including dataset statistics, class distribution,
patient distribution, image characteristics, annotation quality, and visualization.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.util.constants import (
    RAW_DATA_DIR, RESULTS_DIR, CLASSES, CLASS_NAMES, PLANES,
    ORIGINAL_WIDTH, ORIGINAL_HEIGHT, ensure_dirs
)


def load_coco_annotations(plane: str) -> Dict:
    """Load COCO format annotations for a given plane."""
    plane_path = RAW_DATA_DIR / plane
    ann_path = plane_path / "annotations" / "instances_default.json"

    if ann_path.exists():
        with open(ann_path, 'r') as f:
            return json.load(f)
    return None


def decode_rle_to_mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """Decode RLE (Run-Length Encoding) to binary mask."""
    counts = rle['counts']
    size = rle['size']

    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos + count] = 1
        pos += count

    mask = mask.reshape(size, order='F')
    return mask


def get_image_directory(plane: str) -> Path:
    """Get the correct image directory for a given plane."""
    plane_path = RAW_DATA_DIR / plane

    if plane == "Trans-cerebellum":
        return plane_path / "Trans-cerebellum"
    elif plane == "Trans-thalamic":
        return plane_path / "Trans-thalamic"
    elif plane == "Trans-ventricular":
        return plane_path / "Trans-ventricular"
    else:
        return plane_path / "Orginal_train_images_to_959_661"


def analyze_dataset_statistics() -> Dict:
    """Analyze comprehensive dataset statistics."""
    print("\n" + "="*60)
    print("ANALYZING DATASET STATISTICS")
    print("="*60)

    stats = {
        'planes': {},
        'total': {
            'images': 0,
            'patients': set(),
            'annotations': {cat: 0 for cat in CLASS_NAMES},
            'images_with_brain': 0,
            'images_with_csp': 0,
            'images_with_lv': 0,
            'images_with_all_3': 0,
        }
    }

    all_patients = set()

    for plane in PLANES:
        print(f"\nProcessing: {plane}")
        data = load_coco_annotations(plane)

        if data is None:
            print(f"  WARNING: Could not load annotations for {plane}")
            continue

        images = data['images']
        annotations = data['annotations']
        categories = {cat['id']: cat['name'] for cat in data['categories']}

        patients = set()
        for img in images:
            patient_id = img['file_name'].split('_Plane')[0]
            patients.add(patient_id)
            all_patients.add(patient_id)

        cat_counts = defaultdict(int)
        img_categories = defaultdict(set)

        for ann in annotations:
            cat_name = categories[ann['category_id']]
            cat_counts[cat_name] += 1
            img_categories[ann['image_id']].add(cat_name)

        images_with_brain = sum(1 for cats in img_categories.values() if 'Brain' in cats)
        images_with_csp = sum(1 for cats in img_categories.values() if 'CSP' in cats)
        images_with_lv = sum(1 for cats in img_categories.values() if 'LV' in cats)
        images_with_all_3 = sum(1 for cats in img_categories.values()
                                 if {'Brain', 'CSP', 'LV'} == cats)

        stats['planes'][plane] = {
            'images': len(images),
            'patients': len(patients),
            'patient_ids': patients,
            'annotations': dict(cat_counts),
            'images_with_brain': images_with_brain,
            'images_with_csp': images_with_csp,
            'images_with_lv': images_with_lv,
            'images_with_all_3': images_with_all_3,
        }

        stats['total']['images'] += len(images)
        stats['total']['patients'].update(patients)
        for cat, count in cat_counts.items():
            stats['total']['annotations'][cat] += count
        stats['total']['images_with_brain'] += images_with_brain
        stats['total']['images_with_csp'] += images_with_csp
        stats['total']['images_with_lv'] += images_with_lv
        stats['total']['images_with_all_3'] += images_with_all_3

        print(f"  Images: {len(images)}, Patients: {len(patients)}")
        print(f"  Brain: {cat_counts.get('Brain', 0)}, CSP: {cat_counts.get('CSP', 0)}, LV: {cat_counts.get('LV', 0)}")
        print(f"  Images with all 3 classes: {images_with_all_3}")

    stats['total']['patients'] = len(all_patients)

    return stats


def analyze_patient_overlap(stats: Dict) -> Dict:
    """Analyze patient overlap between planes."""
    print("\n" + "="*60)
    print("ANALYZING PATIENT OVERLAP BETWEEN PLANES")
    print("="*60)

    overlap_stats = {}
    plane_names = list(stats['planes'].keys())

    for i, p1 in enumerate(plane_names):
        for p2 in plane_names[i+1:]:
            patients1 = stats['planes'][p1]['patient_ids']
            patients2 = stats['planes'][p2]['patient_ids']
            overlap = patients1 & patients2

            key = f"{p1} <-> {p2}"
            overlap_stats[key] = {'count': len(overlap), 'patients': overlap}
            print(f"  {key}: {len(overlap)} overlapping patients")

    return overlap_stats


def analyze_image_characteristics() -> Dict:
    """Analyze image characteristics (dimensions, intensity distribution)."""
    print("\n" + "="*60)
    print("ANALYZING IMAGE CHARACTERISTICS")
    print("="*60)

    characteristics = {'dimensions': [], 'intensity_stats': [], 'sample_images': []}

    for plane in PLANES:
        data = load_coco_annotations(plane)
        if data is None:
            continue

        img_dir = get_image_directory(plane)

        if img_dir.exists():
            sample_images = list(img_dir.glob("*.png"))[:5]

            for img_path in sample_images:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    characteristics['dimensions'].append({
                        'plane': plane, 'file': img_path.name,
                        'height': img.shape[0], 'width': img.shape[1]
                    })
                    characteristics['intensity_stats'].append({
                        'plane': plane, 'file': img_path.name,
                        'mean': float(np.mean(img)), 'std': float(np.std(img)),
                        'min': int(np.min(img)), 'max': int(np.max(img))
                    })

            if sample_images:
                characteristics['sample_images'].append({'plane': plane, 'path': str(sample_images[0])})

    if characteristics['dimensions']:
        dims = characteristics['dimensions']
        unique_dims = set((d['height'], d['width']) for d in dims)
        print(f"  Unique dimensions found: {unique_dims}")

    if characteristics['intensity_stats']:
        stats = characteristics['intensity_stats']
        mean_intensity = np.mean([s['mean'] for s in stats])
        std_intensity = np.mean([s['std'] for s in stats])
        print(f"  Average intensity: {mean_intensity:.2f} ± {std_intensity:.2f}")

    return characteristics


def analyze_annotation_quality() -> Dict:
    """Analyze annotation quality (mask sizes, bounding boxes, etc.)."""
    print("\n" + "="*60)
    print("ANALYZING ANNOTATION QUALITY")
    print("="*60)

    quality_stats = {
        'bbox_sizes': {cat: [] for cat in CLASS_NAMES},
        'mask_areas': {cat: [] for cat in CLASS_NAMES},
        'aspect_ratios': {cat: [] for cat in CLASS_NAMES},
    }

    for plane in PLANES:
        data = load_coco_annotations(plane)
        if data is None:
            continue

        categories = {cat['id']: cat['name'] for cat in data['categories']}

        for ann in data['annotations']:
            cat_name = categories[ann['category_id']]
            bbox = ann['bbox']

            width, height = bbox[2], bbox[3]
            area = width * height
            aspect_ratio = width / max(height, 1)

            quality_stats['bbox_sizes'][cat_name].append({'width': width, 'height': height, 'area': area})
            quality_stats['aspect_ratios'][cat_name].append(aspect_ratio)

            if 'segmentation' in ann and isinstance(ann['segmentation'], dict):
                rle = ann['segmentation']
                if 'counts' in rle:
                    mask = decode_rle_to_mask(rle, ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
                    mask_area = np.sum(mask)
                    quality_stats['mask_areas'][cat_name].append(mask_area)

    for cat in CLASS_NAMES:
        if quality_stats['bbox_sizes'][cat]:
            areas = [b['area'] for b in quality_stats['bbox_sizes'][cat]]
            print(f"\n  {cat}:")
            print(f"    BBox area - Mean: {np.mean(areas):.2f}, Std: {np.std(areas):.2f}")
            print(f"    BBox area - Min: {np.min(areas):.2f}, Max: {np.max(areas):.2f}")

            if quality_stats['mask_areas'][cat]:
                mask_areas = quality_stats['mask_areas'][cat]
                print(f"    Mask area - Mean: {np.mean(mask_areas):.2f}, Std: {np.std(mask_areas):.2f}")

    return quality_stats


def generate_visualizations(stats: Dict, quality_stats: Dict, output_dir: Path):
    """Generate all EDA visualizations."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Dataset overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plane_names = [p.replace("Fetal Head Images", "FHI") for p in stats['planes'].keys()]
    images_per_plane = [stats['planes'][p]['images'] for p in stats['planes'].keys()]

    ax = axes[0, 0]
    bars = ax.bar(plane_names, images_per_plane, color=sns.color_palette("husl", 4))
    ax.set_title('Images per Ultrasound Plane', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, images_per_plane):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val), ha='center', va='bottom', fontweight='bold')

    patients_per_plane = [stats['planes'][p]['patients'] for p in stats['planes'].keys()]
    ax = axes[0, 1]
    bars = ax.bar(plane_names, patients_per_plane, color=sns.color_palette("husl", 4))
    ax.set_title('Unique Patients per Ultrasound Plane', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, patients_per_plane):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', va='bottom', fontweight='bold')

    ax = axes[1, 0]
    x = np.arange(len(plane_names))
    width = 0.25
    for i, cat in enumerate(CLASS_NAMES):
        counts = [stats['planes'][p]['annotations'].get(cat, 0) for p in stats['planes'].keys()]
        ax.bar(x + i*width, counts, width, label=cat)
    ax.set_title('Class Distribution per Plane', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Annotations')
    ax.set_xticks(x + width)
    ax.set_xticklabels(plane_names, rotation=15)
    ax.legend()

    all_3_counts = [stats['planes'][p]['images_with_all_3'] for p in stats['planes'].keys()]
    total_images = [stats['planes'][p]['images'] for p in stats['planes'].keys()]
    percentages = [a/t*100 if t > 0 else 0 for a, t in zip(all_3_counts, total_images)]

    ax = axes[1, 1]
    bars = ax.bar(plane_names, all_3_counts, color=sns.color_palette("husl", 4))
    ax.set_title('Images with All 3 Classes (Brain, CSP, LV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images')
    ax.tick_params(axis='x', rotation=15)
    for bar, val, pct in zip(bars, all_3_counts, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: dataset_overview.png")

    # Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    total_counts = [stats['total']['annotations'][cat] for cat in CLASS_NAMES]
    colors = sns.color_palette("husl", 3)

    ax = axes[0]
    bars = ax.bar(CLASS_NAMES, total_counts, color=colors)
    ax.set_title('Total Class Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Annotations')
    for bar, val in zip(bars, total_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, str(val), ha='center', va='bottom', fontweight='bold')

    ax = axes[1]
    ax.pie(total_counts, labels=CLASS_NAMES, autopct='%1.1f%%', colors=colors, explode=(0.02, 0.02, 0.02))
    ax.set_title('Class Distribution (Percentage)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: class_distribution.png")

    # BBox size distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, cat in enumerate(CLASS_NAMES):
        ax = axes[idx]
        if quality_stats['bbox_sizes'][cat]:
            areas = [b['area'] for b in quality_stats['bbox_sizes'][cat]]
            ax.hist(areas, bins=50, color=colors[idx], edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(areas), color='red', linestyle='--', label=f'Mean: {np.mean(areas):.0f}')
            ax.set_title(f'{cat} - Bounding Box Area Distribution', fontsize=11, fontweight='bold')
            ax.set_xlabel('Area (pixels²)')
            ax.set_ylabel('Frequency')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: bbox_size_distribution.png")

    # Summary statistics table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    table_data = [
        ['Metric', 'Value'],
        ['Total Images', str(stats['total']['images'])],
        ['Total Unique Patients', str(stats['total']['patients'])],
        ['Total Brain Annotations', str(stats['total']['annotations']['Brain'])],
        ['Total CSP Annotations', str(stats['total']['annotations']['CSP'])],
        ['Total LV Annotations', str(stats['total']['annotations']['LV'])],
        ['Images with All 3 Classes', str(stats['total']['images_with_all_3'])],
        ['Class Imbalance Ratio (Brain:LV)', f"{stats['total']['annotations']['Brain']/max(stats['total']['annotations']['LV'],1):.2f}:1"],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center', colColours=['#4CAF50', '#4CAF50'])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Dataset Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_dir / 'summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: summary_statistics.png")


def generate_sample_annotations(output_dir: Path, num_samples: int = 6):
    """Generate sample images with annotations overlay."""
    print("\n" + "="*60)
    print("GENERATING SAMPLE ANNOTATION VISUALIZATIONS")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {'Brain': (255, 0, 0), 'CSP': (0, 255, 0), 'LV': (0, 0, 255)}

    samples_collected = 0
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for plane in PLANES:
        if samples_collected >= num_samples:
            break

        data = load_coco_annotations(plane)
        if data is None:
            continue

        categories = {cat['id']: cat['name'] for cat in data['categories']}
        img_dir = get_image_directory(plane)

        if not img_dir.exists():
            print(f"  WARNING: Image directory not found: {img_dir}")
            continue

        img_annotations = defaultdict(list)
        for ann in data['annotations']:
            img_annotations[ann['image_id']].append(ann)

        for img_info in data['images']:
            if samples_collected >= num_samples:
                break

            img_id = img_info['id']
            anns = img_annotations[img_id]
            ann_classes = set(categories[ann['category_id']] for ann in anns)

            if len(ann_classes) < 2:
                continue

            img_path = img_dir / img_info['file_name']
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlay = img_rgb.copy()

            for ann in anns:
                cat_name = categories[ann['category_id']]
                color = colors[cat_name]

                if 'segmentation' in ann and isinstance(ann['segmentation'], dict):
                    rle = ann['segmentation']
                    mask = decode_rle_to_mask(rle, img.shape[0], img.shape[1])
                    mask_overlay = np.zeros_like(img_rgb)
                    mask_overlay[mask == 1] = color
                    overlay = cv2.addWeighted(overlay, 1, mask_overlay, 0.3, 0)

                bbox = ann['bbox']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                cv2.putText(overlay, cat_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ax = axes[samples_collected]
            ax.imshow(overlay)
            ax.set_title(f'{plane}\n{img_info["file_name"]}', fontsize=9)
            ax.axis('off')
            samples_collected += 1

    legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor=np.array(colors[cat])/255,
                                   markersize=15, label=cat) for cat in CLASS_NAMES]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_dir / 'sample_annotations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sample_annotations.png ({samples_collected} samples)")


def save_eda_report(stats: Dict, quality_stats: Dict, output_dir: Path):
    """Save EDA findings to a markdown report."""
    print("\n" + "="*60)
    print("SAVING EDA REPORT")
    print("="*60)

    report_path = output_dir / "EDA_REPORT.md"

    with open(report_path, 'w') as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        f.write("## Fetal Head Ultrasound Dataset\n\n")

        f.write("### Dataset Overview\n\n")
        f.write(f"- **Total Images**: {stats['total']['images']}\n")
        f.write(f"- **Total Unique Patients**: {stats['total']['patients']}\n")
        f.write(f"- **Number of Classes**: 3 (Brain, CSP, LV)\n")
        f.write(f"- **Image Dimensions**: {ORIGINAL_WIDTH}×{ORIGINAL_HEIGHT} pixels\n\n")

        f.write("### Class Distribution\n\n")
        f.write("| Class | Total Annotations | Percentage |\n")
        f.write("|-------|-------------------|------------|\n")
        total_ann = sum(stats['total']['annotations'].values())
        for cat in CLASS_NAMES:
            count = stats['total']['annotations'][cat]
            pct = count / total_ann * 100 if total_ann > 0 else 0
            f.write(f"| {cat} | {count} | {pct:.1f}% |\n")

        f.write("\n### Images per Plane\n\n")
        f.write("| Plane | Images | Patients | Brain | CSP | LV | All 3 Classes |\n")
        f.write("|-------|--------|----------|-------|-----|-----|---------------|\n")
        for plane in stats['planes']:
            p = stats['planes'][plane]
            f.write(f"| {plane} | {p['images']} | {p['patients']} | ")
            f.write(f"{p['annotations'].get('Brain', 0)} | ")
            f.write(f"{p['annotations'].get('CSP', 0)} | ")
            f.write(f"{p['annotations'].get('LV', 0)} | ")
            f.write(f"{p['images_with_all_3']} |\n")

        f.write("\n### Key Findings\n\n")
        f.write("1. **Class Imbalance**: Brain class dominates the dataset, while LV is underrepresented\n")
        f.write("2. **Trans-cerebellum**: Has very few LV annotations (only 2), making it less suitable for LV segmentation\n")
        f.write("3. **Diverse dataset**: Has the most balanced distribution of all 3 classes\n")
        f.write("4. **Patient overlap**: Significant overlap between planes - same patients appear in multiple planes\n")
        f.write("5. **Inter-patient splitting**: Essential to prevent data leakage during train/val/test split\n\n")

        f.write("### Recommendations\n\n")
        f.write("1. Use all planes for robust training (Option C from planning)\n")
        f.write("2. Apply class weighting during training to handle imbalance\n")
        f.write("3. Implement strict inter-patient splitting (70/15/15)\n")
        f.write("4. Use augmentations from FetSAM paper to improve generalization\n")
        f.write("5. Consider focal loss or Dice loss to handle class imbalance\n")

    print(f"  Saved: {report_path}")


def run_full_eda():
    """Run complete EDA pipeline."""
    print("\n" + "="*60)
    print("FETAL HEAD ULTRASOUND DATASET - COMPREHENSIVE EDA")
    print("="*60)

    ensure_dirs()
    eda_output_dir = RESULTS_DIR / "eda"
    eda_output_dir.mkdir(parents=True, exist_ok=True)

    stats = analyze_dataset_statistics()
    overlap_stats = analyze_patient_overlap(stats)
    img_characteristics = analyze_image_characteristics()
    quality_stats = analyze_annotation_quality()

    generate_visualizations(stats, quality_stats, eda_output_dir)
    generate_sample_annotations(eda_output_dir)
    save_eda_report(stats, quality_stats, eda_output_dir)

    stats_to_save = {
        'planes': {k: {kk: vv for kk, vv in v.items() if kk != 'patient_ids'}
                   for k, v in stats['planes'].items()},
        'total': {k: v if not isinstance(v, set) else len(v)
                  for k, v in stats['total'].items()}
    }

    with open(eda_output_dir / "statistics.json", 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    print(f"\n  Saved: statistics.json")

    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print(f"Results saved to: {eda_output_dir}")
    print("="*60)

    return stats, quality_stats


if __name__ == "__main__":
    run_full_eda()
