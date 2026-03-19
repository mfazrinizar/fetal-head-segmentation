"""
Custom loss functions for fetal head segmentation.

Implements FetSAM combined loss (Weighted Dice + Weighted Lovasz) and other losses:
- Alzubaidi et al. (2024): "FetSAM: Advanced Segmentation Techniques for Fetal Head Biometrics"
- Berman et al. (2018): "The Lovász-Softmax Loss: A Tractable Surrogate for IoU Optimization"
- Kervadec et al. (2019): "Boundary Loss for Highly Unbalanced Segmentation"

FetSAM Loss Formula:
    L_combined = α × L_dice + β × L_lovasz (where α=0.5, β=0.5)
    
Class weights from FetSAM: [0.1, 0.1, 0.9, 0.7] for [background, brain, CSP, LV]
For our 3-class setup (no background): normalize to [0.1, 0.9, 0.7] -> [0.059, 0.529, 0.412]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """
    Boundary loss for precise anatomical edge detection.
    
    Based on Kervadec et al. (2019): Uses distance transform to weight
    errors near boundaries more heavily than interior regions.
    
    For fetal head segmentation, precise boundaries are critical for:
    - Head Circumference (HC) measurement
    - Biparietal Diameter (BPD) measurement
    - Accurate CSP/LV delineation
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
    
    def compute_distance_map(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute distance transform for boundary weighting."""
        mask_np = mask.cpu().numpy()
        batch_size = mask_np.shape[0]
        
        distance_maps = []
        for b in range(batch_size):
            if mask_np[b].sum() == 0:
                dist = np.zeros_like(mask_np[b])
            else:
                pos_dist = distance_transform_edt(mask_np[b])
                neg_dist = distance_transform_edt(1 - mask_np[b])
                dist = pos_dist - neg_dist
            distance_maps.append(dist)
        
        return torch.from_numpy(np.stack(distance_maps)).to(mask.device).float()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss.
        
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        """
        pred_soft = torch.sigmoid(pred)
        
        total_loss = 0
        for c in range(self.num_classes):
            dist_map = self.compute_distance_map(target[:, c])
            boundary_loss = (pred_soft[:, c] * dist_map).mean()
            total_loss += boundary_loss
        
        return total_loss / self.num_classes


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Based on Lin et al. (2017): Down-weights easy examples to focus
    training on hard negatives.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary Lovasz hinge loss.
    
    Args:
        logits: Predicted logits (N,)
        labels: Ground truth binary labels (N,)
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax Loss for IoU optimization.
    
    Based on Berman et al. (2018): "The Lovász-Softmax Loss: A Tractable 
    Surrogate for the Optimization of the Intersection-Over-Union Measure"
    
    Directly optimizes IoU metric through convex surrogate.
    """
    
    def __init__(self, per_image: bool = False, num_classes: int = 3):
        super().__init__()
        self.per_image = per_image
        self.num_classes = num_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Lovasz loss.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        """
        total_loss = 0
        
        for c in range(self.num_classes):
            if self.per_image:
                loss = 0
                for b in range(pred.shape[0]):
                    loss += lovasz_hinge_flat(
                        pred[b, c].view(-1),
                        target[b, c].view(-1)
                    )
                total_loss += loss / pred.shape[0]
            else:
                total_loss += lovasz_hinge_flat(
                    pred[:, c].reshape(-1),
                    target[:, c].reshape(-1)
                )
        
        return total_loss / self.num_classes


class WeightedDiceLoss(nn.Module):
    """
    Weighted Dice Loss as used in FetSAM.
    
    Formula: L_dice = 1 - (2 * sum(y * p * w)) / (sum((y + p) * w))
    
    Where w are class weights emphasizing CSP and LV.
    """
    
    def __init__(self, class_weights: list = None, smooth: float = 1.0, num_classes: int = 3):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        
        if class_weights is None:
            # FetSAM weights: [0.1, 0.9, 0.7] for [Brain, CSP, LV]
            # Normalized: sum = 1.7, so [0.059, 0.529, 0.412]
            class_weights = [0.1, 0.9, 0.7]
        
        total = sum(class_weights)
        self.class_weights = torch.FloatTensor([w / total for w in class_weights])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted Dice loss."""
        pred_soft = torch.sigmoid(pred)
        weights = self.class_weights.to(pred.device)
        
        total_loss = 0
        for c in range(self.num_classes):
            intersection = (pred_soft[:, c] * target[:, c]).sum()
            union = pred_soft[:, c].sum() + target[:, c].sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            total_loss += weights[c] * (1 - dice)
        
        return total_loss


class WeightedLovaszLoss(nn.Module):
    """
    Weighted Lovasz Loss as used in FetSAM.
    
    Applies class weights to Lovasz hinge loss for each class.
    """
    
    def __init__(self, class_weights: list = None, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        if class_weights is None:
            class_weights = [0.1, 0.9, 0.7]
        
        total = sum(class_weights)
        self.class_weights = torch.FloatTensor([w / total for w in class_weights])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted Lovasz loss."""
        weights = self.class_weights.to(pred.device)
        
        total_loss = 0
        for c in range(self.num_classes):
            lovasz = lovasz_hinge_flat(
                pred[:, c].reshape(-1),
                target[:, c].reshape(-1)
            )
            total_loss += weights[c] * lovasz
        
        return total_loss


class FetSAMCombinedLoss(nn.Module):
    """
    FetSAM Combined Loss: Weighted Dice + Weighted Lovasz.
    
    From Alzubaidi et al. (2024):
        L_combined = α × L_dice + β × L_lovasz
        
    Default: α = 0.5, β = 0.5 (equal contribution)
    Class weights: [0.1, 0.9, 0.7] for [Brain, CSP, LV]
    
    This loss is NOT implemented in YOLO26 by default.
    YOLO26 uses BCEDiceLoss (BCE + Dice with equal weights).
    FetSAM adds Lovasz for direct IoU optimization.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        class_weights: list = None,
        num_classes: int = 3,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        if class_weights is None:
            class_weights = [0.1, 0.9, 0.7]
        
        self.dice_loss = WeightedDiceLoss(class_weights, num_classes=num_classes)
        self.lovasz_loss = WeightedLovaszLoss(class_weights, num_classes=num_classes)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute FetSAM combined loss.
        
        Returns:
            Dictionary with total and component losses
        """
        dice = self.dice_loss(pred, target)
        lovasz = self.lovasz_loss(pred, target)
        
        total = self.alpha * dice + self.beta * lovasz
        
        return {
            'dice': dice,
            'lovasz': lovasz,
            'total': total,
        }


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Based on Cui et al. (2019): Weights classes by inverse effective
    frequency to handle long-tail distributions.
    
    Class weights for fetal head dataset:
    - Brain: 2626 -> weight ~0.65
    - CSP: 890 -> weight ~1.92
    - LV: 1026 -> weight ~1.67
    """
    
    def __init__(self, class_counts: list, beta: float = 0.9999, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        
        self.class_weights = torch.FloatTensor(weights)
        print(f"Class-balanced weights: {dict(zip(['Brain', 'CSP', 'LV'], weights))}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced BCE loss."""
        weights = self.class_weights.to(pred.device)
        
        total_loss = 0
        for c in range(self.num_classes):
            class_loss = F.binary_cross_entropy_with_logits(
                pred[:, c], target[:, c], reduction='mean'
            )
            total_loss += weights[c] * class_loss
        
        return total_loss / self.num_classes


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for fetal head segmentation.
    
    Combines:
    - BCE/Focal Loss: Pixel-wise classification
    - Dice Loss: Region overlap (handles imbalance)
    - Boundary Loss: Precise edge delineation
    """
    
    def __init__(
        self,
        use_focal: bool = False,
        use_boundary: bool = False,
        class_counts: list = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_weight: float = 0.1,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
    ):
        super().__init__()
        
        self.use_focal = use_focal
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        if use_focal and class_counts is not None:
            self.bce_loss = ClassBalancedLoss(class_counts)
        elif use_focal:
            self.bce_loss = FocalLoss(focal_alpha, focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        
        if use_boundary:
            self.boundary_loss = BoundaryLoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Compute Dice loss."""
        pred_soft = torch.sigmoid(pred)
        
        intersection = (pred_soft * target).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss.
        
        Returns:
            Dictionary with total loss and component losses
        """
        losses = {}
        
        losses['bce'] = self.bce_loss(pred, target)
        losses['dice'] = self.dice_loss(pred, target)
        
        total = self.bce_weight * losses['bce'] + self.dice_weight * losses['dice']
        
        if self.use_boundary:
            losses['boundary'] = self.boundary_loss(pred, target)
            total += self.boundary_weight * losses['boundary']
        
        losses['total'] = total
        return losses


def get_loss_function(loss_config: str, class_counts: list = None, class_weights: list = None):
    """
    Get loss function based on configuration.
    
    Args:
        loss_config: One of 'default', 'fetsam', 'boundary_aware', 'class_balanced'
        class_counts: List of instance counts per class [Brain, CSP, LV]
        class_weights: List of class weights [Brain, CSP, LV] (default: FetSAM weights)
    """
    if class_counts is None:
        class_counts = [2626, 890, 1026]
    
    if class_weights is None:
        # FetSAM paper weights for [Brain, CSP, LV]
        class_weights = [0.1, 0.9, 0.7]
    
    if loss_config == 'default':
        return CombinedSegmentationLoss(
            use_focal=False,
            use_boundary=False,
        )
    elif loss_config == 'fetsam':
        # FetSAM Combined Loss: Weighted Dice + Weighted Lovasz
        return FetSAMCombinedLoss(
            alpha=0.5,
            beta=0.5,
            class_weights=class_weights,
        )
    elif loss_config == 'boundary_aware':
        return CombinedSegmentationLoss(
            use_focal=False,
            use_boundary=True,
            boundary_weight=0.1,
        )
    elif loss_config == 'class_balanced':
        return CombinedSegmentationLoss(
            use_focal=True,
            use_boundary=False,
            class_counts=class_counts,
        )
    else:
        raise ValueError(f"Unknown loss config: {loss_config}")


if __name__ == "__main__":
    print("Testing custom loss functions...")
    print("="*60)
    
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 2, (2, 3, 64, 64)).float()
    
    # Test all loss configurations
    configs = ['default', 'fetsam', 'boundary_aware', 'class_balanced']
    
    for config in configs:
        loss_fn = get_loss_function(config)
        losses = loss_fn(pred, target)
        print(f"\n{config}:")
        for name, value in losses.items():
            print(f"  {name}: {value.item():.4f}")
    
    # Test FetSAM components individually
    print("\n" + "="*60)
    print("FetSAM Component Tests:")
    
    dice_loss = WeightedDiceLoss()
    lovasz_loss = WeightedLovaszLoss()
    
    print(f"  WeightedDiceLoss: {dice_loss(pred, target).item():.4f}")
    print(f"  WeightedLovaszLoss: {lovasz_loss(pred, target).item():.4f}")
    
    print("\n✓ All loss functions working!")
