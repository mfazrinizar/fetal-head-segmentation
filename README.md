# Fetal Head Ultrasound Segmentation with YOLO26

Multi-class instance segmentation of fetal head structures (Brain, CSP, LV) using YOLO26-seg with FetSAM augmentations and loss functions.

## Overview

| Component | Description |
|-----------|-------------|
| **Model** | YOLO26-seg from [mfazrinizar/ultralytics](https://github.com/mfazrinizar/ultralytics) |
| **Classes** | Brain, CSP (Cavum Septum Pellucidum), LV (Lateral Ventricles) |
| **Dataset** | 3,832 images, 2,037 patients |
| **Split** | 70/15/15 inter-patient (no data leakage) |

## Dataset Statistics

| Split | Patients | Images | Brain | CSP | LV |
|-------|----------|--------|-------|-----|-----|
| Train | 1,425 | 2,654 | 2,626 | 890 | 1,026 |
| Val | 305 | 603 | 598 | 221 | 238 |
| Test | 307 | 575 | 568 | 172 | 212 |

## FetSAM Implementation

Based on Alzubaidi et al. (2024): "FetSAM: Advanced Segmentation Techniques for Fetal Head Biometrics"

**Augmentations:**
- Rotation: ±30°
- Horizontal/Vertical Flip: 0.3
- Brightness: 0.4
- No mosaic/mixup (medical imaging)

**Loss Function:**
```
L_combined = 0.5 × WeightedDice + 0.5 × WeightedLovasz
Class weights: [Brain=0.1, CSP=0.9, LV=0.7]
```

## Project Structure

```
├── src/
│   ├── eda/                 # Exploratory data analysis
│   ├── preprocess/          # Data preprocessing & splitting
│   ├── model/               # Custom models & losses
│   │   ├── losses.py        # FetSAM loss functions
│   │   └── fetsam_loss_integration.py
│   ├── train/               # Training scripts
│   ├── postprocess/         # Evaluation
│   └── util/                # Constants & utilities
├── notebooks/
│   └── fetal_head_segmentation_yolo26.ipynb  # Kaggle notebook
├── data/                    # Dataset (not tracked)
└── results/                 # Training outputs (not tracked)
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Or use conda environment
conda activate sign-yolo26

# Run training
python src/train/train.py --experiment fetsam_full --epochs 100

# With FetSAM loss integration
python src/demo_with_fetsam_loss.py
```

## Experiments

| Experiment | Description |
|------------|-------------|
| `baseline` | YOLO26 default (BCE + Dice) |
| `fetsam_aug` | FetSAM augmentations only |
| `fetsam_loss` | FetSAM loss (Dice + Lovasz) |
| `fetsam_full` | Full FetSAM pipeline |

## Key Features

- **Inter-patient split**: Strict separation ensures no data leakage
- **FetSAM loss**: Weighted Dice + Lovasz for better IoU optimization
- **Class weighting**: Prioritizes small structures (CSP, LV)
- **Kaggle compatible**: Notebook ready for cloud training

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- [mfazrinizar/ultralytics](https://github.com/mfazrinizar/ultralytics) (YOLO26)

## References

1. Alzubaidi et al. (2024). "FetSAM: Advanced Segmentation Techniques for Fetal Head Biometrics". IEEE OJEMB.
2. Berman et al. (2018). "The Lovász-Softmax Loss". CVPR.
3. Dataset: [Zenodo 8265464](https://zenodo.org/records/8265464)

## License

Research use only. Dataset under CC BY 4.0.
