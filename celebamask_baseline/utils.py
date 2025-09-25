
import torch
import numpy as np

def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def fscore_mean_per_class(gt, pred, num_classes=None, beta=1.0):
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    classes = np.unique(gt) if num_classes is None else np.arange(num_classes)
    scores = []
    for c in classes:
        tp = np.sum((gt == c) & (pred == c))
        fp = np.sum((gt != c) & (pred == c))
        fn = np.sum((gt == c) & (pred != c))
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-7)
        scores.append(f)
    return float(np.mean(scores))
