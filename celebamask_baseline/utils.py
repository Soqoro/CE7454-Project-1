import numpy as np
import torch
from typing import Optional


def count_trainable_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def fscore_mean_per_class(
    gt: torch.Tensor,
    pred: torch.Tensor,
    num_classes: Optional[int] = None,
    beta: float = 1.0,
    ignore_index: Optional[int] = None,
    exclude_background: bool = False,
    present_only: bool = False,
) -> float:
    """
    Macro F-score across classes for a single (or stacked) prediction.

    Args:
        gt:   (B,H,W) or (H,W) ground-truth class indices.
        pred: (B,H,W) or (H,W) predicted class indices.
        num_classes: If None, use unique classes in gt; else use range(num_classes).
        beta: F-beta (default F1).
        ignore_index: label value to ignore in gt (masked out from both gt & pred).
        exclude_background: if True, drop class 0 from averaging.
        present_only: if True, average only over classes that appear at least
                      once in gt after masking (useful for small batch instability).

    Returns:
        Macro F-score (float).
    """
    # To numpy
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    # Broadcast to (N, H, W)
    if gt_np.ndim == 2:
        gt_np = gt_np[None, ...]
        pred_np = pred_np[None, ...]
    assert gt_np.shape == pred_np.shape, "gt and pred must have the same shape"

    # Mask out ignore_index from both
    if ignore_index is not None:
        valid = gt_np != ignore_index
        # If ALL invalid (shouldn't happen), early return 0
        if not valid.any():
            return 0.0
        # Flatten valid pixels
        gt_flat = gt_np[valid]
        pred_flat = pred_np[valid]
    else:
        gt_flat = gt_np.reshape(-1)
        pred_flat = pred_np.reshape(-1)

    # Determine class set
    if num_classes is None:
        classes = np.unique(gt_flat)
    else:
        classes = np.arange(int(num_classes), dtype=int)

    if exclude_background:
        classes = classes[classes != 0]

    if present_only:
        # Keep classes that actually appear in GT after masking
        present_mask = np.array([(gt_flat == c).any() for c in classes], dtype=bool)
        classes = classes[present_mask]

    if classes.size == 0:
        return 0.0

    eps = 1e-7
    scores = []
    for c in classes:
        tp = np.sum((gt_flat == c) & (pred_flat == c))
        fp = np.sum((gt_flat != c) & (pred_flat == c))
        fn = np.sum((gt_flat == c) & (pred_flat != c))
        # precision/recall and F-beta
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall + eps)
        scores.append(f)

    return float(np.mean(scores))


@torch.no_grad()
def fscore_micro(
    gt: torch.Tensor,
    pred: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    exclude_background: bool = False,
) -> float:
    """
    Micro F1 across all classes (computed from global TP/FP/FN).
    """
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    if gt_np.ndim == 2:
        gt_np = gt_np[None, ...]
        pred_np = pred_np[None, ...]

    if ignore_index is not None:
        valid = gt_np != ignore_index
        if not valid.any():
            return 0.0
        gt_flat = gt_np[valid]
        pred_flat = pred_np[valid]
    else:
        gt_flat = gt_np.reshape(-1)
        pred_flat = pred_np.reshape(-1)

    classes = range(num_classes)
    if exclude_background:
        classes = range(1, num_classes)

    T = 0
    Fp = 0
    Fn = 0
    for c in classes:
        p = pred_flat == c
        m = gt_flat == c
        T  += int(np.sum(p & m))
        Fp += int(np.sum(p & (~m)))
        Fn += int(np.sum((~p) & m))

    eps = 1e-8
    return float((2.0 * T) / (2.0 * T + Fp + Fn + eps))
