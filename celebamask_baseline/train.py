import os
import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CelebAMaskMini
from model import DWUNet
from losses import ComboLoss


# ----------------------------
# Reproducibility
# ----------------------------
def set_seeds(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Albumentations CoarseDropout (strict: no Cutout fallback)
# ----------------------------
def make_coarse_dropout(p: float = 0.3):
    """
    Always use albumentations.CoarseDropout.

    Different albumentations versions expose different kwarg names.
    We introspect the signature and pass only supported kwargs.
    """
    import inspect

    if not hasattr(A, "CoarseDropout"):
        raise ImportError("albumentations.CoarseDropout not found. Please upgrade albumentations.")

    CD = A.CoarseDropout  # type: ignore[attr-defined]
    sig = inspect.signature(CD.__init__)
    params = sig.parameters

    kwargs = {"p": p}

    # Holes count
    if "max_holes" in params:
        kwargs.update({"max_holes": 4})
        if "min_holes" in params:
            kwargs.update({"min_holes": 1})
    elif "num_holes" in params:
        kwargs.update({"num_holes": 2})

    # Size parameters
    if "max_height" in params and "max_width" in params:
        kwargs.update({"max_height": 32, "max_width": 32})
        if "min_height" in params and "min_width" in params:
            kwargs.update({"min_height": 8, "min_width": 8})
    elif "max_h_size" in params and "max_w_size" in params:
        kwargs.update({"max_h_size": 32, "max_w_size": 32})

    # Fill values
    if "fill_value" in params:
        kwargs["fill_value"] = 0
    if "mask_fill_value" in params:
        kwargs["mask_fill_value"] = 0

    return CD(**kwargs)  # type: ignore[call-arg]


# ----------------------------
# Transforms
# ----------------------------
def get_train_tfms(mean, std):
    transforms_list = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, border_mode=0, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        make_coarse_dropout(p=0.3),
        A.Normalize(mean=mean, std=std),
    ]
    return A.Compose(transforms_list)

def get_val_tfms(mean, std):
    return A.Compose([A.Normalize(mean=mean, std=std)])


# ----------------------------
# Stats & Class Weights
# ----------------------------
def compute_mean_std(image_dir: str, sample_max: int = 200):
    """
    Returns mean/std computed on images scaled to [0,1].
    """
    files = sorted(os.listdir(image_dir))[:sample_max]
    means = np.zeros(3, dtype=np.float64)
    sq_means = np.zeros(3, dtype=np.float64)
    n = 0
    for f in files:
        img_bgr = cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_COLOR)  # np.ndarray | None
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        flat = img_rgb.reshape(-1, 3)
        means    += flat.mean(axis=0)
        sq_means += (flat ** 2).mean(axis=0)
        n += 1
    if n == 0:
        return [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]
    means /= n
    sq_means /= n
    stds = np.sqrt(np.maximum(sq_means - means**2, 1e-8))
    return means.tolist(), stds.tolist()

def compute_class_weights(mask_dir: str, files: list[str], num_classes: int = 19,
                          clip_min=0.5, clip_max=6.0, mask_ext: str = ".png"):
    """
    Compute inverse-sqrt frequency weights from masks. `files` should be *image* filenames;
    we convert to mask filenames by replacing the extension with `mask_ext` (default .png).
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for f in files:
        base, _ = os.path.splitext(f)
        mask_name = base + (mask_ext if mask_ext.startswith(".") else f".{mask_ext}")
        m = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)  # np.ndarray | None
        if m is None:
            continue
        counts += np.bincount(m.ravel(), minlength=num_classes)
    counts = counts.astype(np.float64)
    counts[counts == 0] = 1.0
    freq = counts / counts.sum()
    inv_sqrt = 1.0 / np.sqrt(freq)
    inv_sqrt /= inv_sqrt.mean()
    inv_sqrt = np.clip(inv_sqrt, clip_min, clip_max)
    return torch.tensor(inv_sqrt, dtype=torch.float32)


# ----------------------------
# Dataset-level F1
# ----------------------------
@torch.no_grad()
def evaluate_dataset_f1(model, loader, device, num_classes=19, ignore_index: int = -100,
                        exclude_background: bool = True):
    model.eval()

    ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
    total_loss = 0.0
    total_pixels = 0

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        total_loss += ce(logits, masks).item()
        total_pixels += (masks != ignore_index).sum().item() if ignore_index is not None else masks.numel()

        preds = torch.argmax(logits, dim=1)

        if ignore_index is not None:
            valid = masks != ignore_index
            preds = preds[valid]
            masks = masks[valid]

        for c in range(num_classes):
            if exclude_background and c == 0:
                continue
            p = preds == c
            m = masks == c
            tp[c] += (p & m).sum().item()
            fp[c] += (p & (~m)).sum().item()
            fn[c] += ((~p) & m).sum().item()

    f_per_class = []
    for c in range(num_classes):
        if exclude_background and c == 0:
            continue
        denom = (2 * tp[c] + fp[c] + fn[c])
        if denom == 0:
            continue
        f1c = (2.0 * tp[c]) / (denom + 1e-8)
        f_per_class.append(float(f1c))

    macro_f = float(np.mean(f_per_class)) if len(f_per_class) else 0.0
    T = tp.sum().item()
    Fp = fp.sum().item()
    Fn = fn.sum().item()
    micro_f = float((2.0 * T) / (2.0 * T + Fp + Fn + 1e-8))

    val_loss = total_loss / max(1, total_pixels)
    return val_loss, macro_f, micro_f


# ----------------------------
# Training
# ----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None, grad_accum: int = 1):
    model.train()
    total = 0.0
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss = loss_fn(logits, masks) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.item()) * imgs.size(0) * grad_accum
        step += 1

    return total / len(loader.dataset)


class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.warm = int(warmup_epochs)
        self.total = int(total_epochs)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warm and self.warm > 0:
            lr = self.base_lr * self.epoch / self.warm
        else:
            t = 0.0 if self.total <= self.warm else (self.epoch - self.warm) / max(1, self.total - self.warm)
            t = min(max(t, 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr


def main():
    set_seeds(123)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="path containing train/images and train/masks")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps (effective batch = bs*accum)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ckpt_out", type=str, default="solution/ckpt.pth")
    ap.add_argument("--base", type=int, default=56)  # ≈ 1.59M params (< 1,821,085 cap)
    ap.add_argument("--ignore_index", type=int, default=-100)
    ap.add_argument("--exclude_bg_in_dice", action="store_true", default=False)
    ap.add_argument("--exclude_bg_in_macroF", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19

    train_images = os.path.join(args.data_root, "train/images")
    train_masks  = os.path.join(args.data_root, "train/masks")

    # Split: last 100 → val (same convention as before)
    all_files = sorted(os.listdir(train_images))
    val_ids = set(all_files[-100:])
    train_ids = [f for f in all_files if f not in val_ids]

    # Compute / cache dataset stats
    stats_path = Path(args.data_root) / "train_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        mean, std = stats["mean"], stats["std"]
        print(f"[stats] loaded mean/std from {stats_path}")
    else:
        mean, std = compute_mean_std(train_images, sample_max=200)
        stats_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
        print(f"[stats] computed & saved mean/std to {stats_path}: mean={mean}, std={std}")

    # Transforms
    t_train = get_train_tfms(mean, std)
    t_val   = get_val_tfms(mean, std)

    # Datasets
    ds_train = CelebAMaskMini(train_images, train_masks, transform=t_train)
    ds_val   = CelebAMaskMini(train_images, train_masks, transform=t_val)
    ds_train.files = [f for f in ds_train.files if f in train_ids]
    ds_val.files   = [f for f in ds_val.files   if f in val_ids]

    # Class weights from TRAIN set only (use .png mask names)
    ce_weights = compute_class_weights(train_masks, ds_train.files, num_classes=num_classes, mask_ext=".png").to(device)

    # Loaders
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=2*args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = DWUNet(in_ch=3, num_classes=num_classes, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 1_821_085, f"Model has {n_params} params, exceeds cap."
    print(f"Params: {n_params}")

    # Loss
    loss_fn = ComboLoss(
        num_classes=num_classes,
        ce_weight=ce_weights,
        dice_weight=1.0,
        ce_weight_factor=0.5,
        ignore_index=args.ignore_index,
        exclude_background_in_dice=args.exclude_bg_in_dice
    )

    # Optim & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = WarmupCosine(optimizer, warmup_epochs=args.warmup, total_epochs=args.epochs, base_lr=args.lr, min_lr=1e-6)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f = -1.0
    Path(os.path.dirname(args.ckpt_out) or ".").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, loader_train, optimizer, loss_fn, device, scaler=scaler, grad_accum=args.accum)
        lr_now = sched.step()
        val_loss, macroF, microF = evaluate_dataset_f1(
            model, loader_val, device, num_classes=num_classes,
            ignore_index=args.ignore_index, exclude_background=args.exclude_bg_in_macroF
        )
        val_F = macroF  # choose the variant matching the official script

        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} | val_loss {val_loss:.6f} | "
              f"val_F(macro) {macroF:.4f} | val_F(micro) {microF:.4f} | lr {lr_now:.2e}")

        if val_F > best_f:
            best_f = val_F
            torch.save({
                "model": model.state_dict(),
                "base": args.base,
                "mean": mean,   # <- save stats for run.py
                "std": std,
            }, args.ckpt_out)
            print(f"  Saved checkpoint -> {args.ckpt_out}")


if __name__ == "__main__":
    main()
