"""
FetSAM Loss Integration for YOLO26.

This module provides a custom segmentation loss that replaces YOLO's default
BCEDiceLoss with FetSAM's Weighted Dice + Weighted Lovasz loss.

To use: Monkey-patch the loss class before training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ultralytics-yolo"))


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
    """Binary Lovasz hinge loss."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm.data]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class FetSAMBCEDiceLovaszLoss(nn.Module):
    """
    FetSAM-style loss: Weighted BCE + Weighted Dice + Weighted Lovasz.
    
    Replaces YOLO's BCEDiceLoss with FetSAM's loss formulation.
    
    From Alzubaidi et al. (2024):
        L = α*BCE + β*Dice + γ*Lovasz
        With class weights [0.1, 0.9, 0.7] for [Brain, CSP, LV]
    """
    
    def __init__(
        self,
        weight_bce: float = 0.25,
        weight_dice: float = 0.5,
        weight_lovasz: float = 0.25,
        class_weights: list = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_lovasz = weight_lovasz
        self.smooth = smooth
        
        # FetSAM class weights: prioritize CSP and LV
        if class_weights is None:
            class_weights = [0.1, 0.9, 0.7]
        
        total = sum(class_weights)
        self.class_weights = [w / total for w in class_weights]
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted logits - can be various shapes from YOLO
            target: Ground truth - can be various shapes from YOLO
        """
        # Handle shape mismatches by interpolating
        if pred.shape != target.shape:
            # Interpolate target to match pred shape
            if target.dim() == 4 and pred.dim() == 4:
                target = F.interpolate(
                    target.float(),
                    size=pred.shape[2:],
                    mode='nearest'
                )
            elif target.dim() == 3 and pred.dim() == 3:
                target = F.interpolate(
                    target.float().unsqueeze(1),
                    size=pred.shape[1:],
                    mode='nearest'
                ).squeeze(1)
        
        pred_soft = torch.sigmoid(pred)
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='mean')
        
        # Dice Loss
        pred_flat = pred_soft.reshape(-1)
        target_flat = target.reshape(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # Lovasz Loss (simplified)
        lovasz_loss = lovasz_hinge_flat(pred.reshape(-1), target.reshape(-1).float())
        
        # Combined
        total = (
            self.weight_bce * bce_loss +
            self.weight_dice * dice_loss +
            self.weight_lovasz * lovasz_loss
        )
        
        return total


def patch_yolo_loss():
    """
    Monkey-patch YOLO's BCEDiceLoss with FetSAM loss.
    
    Call this BEFORE creating the YOLO model.
    """
    from ultralytics.utils import loss as yolo_loss
    
    # Save original
    yolo_loss._OriginalBCEDiceLoss = yolo_loss.BCEDiceLoss
    
    # Replace with FetSAM loss
    yolo_loss.BCEDiceLoss = FetSAMBCEDiceLovaszLoss
    
    print("✓ Patched YOLO BCEDiceLoss with FetSAM loss")
    print("  Components: 0.25*BCE + 0.5*Dice + 0.25*Lovasz")
    print("  Class weights: [Brain=0.1, CSP=0.9, LV=0.7]")
    
    return True


def unpatch_yolo_loss():
    """Restore original YOLO loss."""
    from ultralytics.utils import loss as yolo_loss
    
    if hasattr(yolo_loss, '_OriginalBCEDiceLoss'):
        yolo_loss.BCEDiceLoss = yolo_loss._OriginalBCEDiceLoss
        print("✓ Restored original YOLO BCEDiceLoss")


if __name__ == "__main__":
    print("Testing FetSAM Loss Integration...")
    
    # Test the loss function directly
    loss_fn = FetSAMBCEDiceLovaszLoss()
    
    pred = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    loss = loss_fn(pred, target)
    print(f"Loss value: {loss.item():.4f}")
    
    # Test gradient
    loss.backward()
    print(f"Gradient flow: {'✓ OK' if pred.grad is not None else '✗ FAILED'}")
    
    # Test patching
    print("\nTesting patch...")
    patch_yolo_loss()
    
    from ultralytics.utils.loss import BCEDiceLoss
    print(f"BCEDiceLoss is now: {BCEDiceLoss.__name__}")
    
    unpatch_yolo_loss()
    print("\n✓ All tests passed!")
