
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        N, C, H, W = probs.shape
        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            probs = probs * mask
            target_onehot = target_onehot * mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice.mean()

class ComboLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=None, dice_weight=0.5, ce_weight_factor=1.0, ignore_index=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight_factor = ce_weight_factor

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        return self.ce_weight_factor * ce + self.dice_weight * dice
