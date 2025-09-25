from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(labels: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None) -> torch.Tensor:
    # labels: (B,H,W)
    if ignore_index is not None:
        valid = labels != ignore_index
        labels = labels.clone()
        labels[~valid] = 0
    y = F.one_hot(labels.long(), num_classes=num_classes)  # (B,H,W,C)
    y = y.permute(0, 3, 1, 2).float()                      # (B,C,H,W)
    if ignore_index is not None:
        y *= (labels != ignore_index).unsqueeze(1)
    return y

def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
    exclude_background: bool = False,
    present_only: bool = False,   # <- new: average only over classes present in batch
) -> torch.Tensor:
    """
    Multi-class soft Dice loss. If present_only=True, averages Dice only over classes
    with at least one positive pixel in the batch (after ignore_index and bg slicing).
    """
    probs = F.softmax(logits, dim=1)                      # (B,C,H,W)
    tgt_1h = one_hot(target, num_classes, ignore_index)   # (B,C,H,W)

    c0 = 1 if exclude_background else 0
    probs   = probs[:, c0:, ...]
    tgt_1h  = tgt_1h[:, c0:, ...]

    dims = (0, 2, 3)
    inter = (probs * tgt_1h).sum(dims)                    # (C')
    denom = probs.sum(dims) + tgt_1h.sum(dims)            # (C')

    dice_per_class = (2.0 * inter + eps) / (denom + eps)  # (C')

    if present_only:
        present = (tgt_1h.sum(dims) > 0)                  # (C')
        if present.any():
            dice = dice_per_class[present].mean()
        else:
            dice = dice_per_class.mean()  # fallback: no positives at all
    else:
        dice = dice_per_class.mean()

    return 1.0 - dice

class ComboLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ce_weight: Optional[torch.Tensor] = None,
        dice_weight: float = 1.0,
        ce_weight_factor: float = 1.0,
        ignore_index: Optional[int] = None,
        exclude_background_in_dice: bool = False,
        present_only_dice: bool = False,  # <- new passthrough
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index: int = -100 if ignore_index is None else int(ignore_index)
        self.exclude_bg = exclude_background_in_dice
        self.present_only = present_only_dice
        self.dice_weight = float(dice_weight)
        self.ce_weight_factor = float(ce_weight_factor)
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=self.ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice = soft_dice_loss(
            logits, target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            exclude_background=self.exclude_bg,
            present_only=self.present_only,
        )
        return self.ce_weight_factor * ce + self.dice_weight * dice
