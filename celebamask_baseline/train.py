
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from albumentations import Compose, HorizontalFlip, RandomRotate90, ShiftScaleRotate, ColorJitter

from dataset import CelebAMaskMini
from model import DWUNet
from losses import ComboLoss
from utils import count_trainable_params, fscore_mean_per_class

def get_train_tfms():
    return Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.2),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
    ], additional_targets={'mask':'mask'})

def get_val_tfms():
    return Compose([])

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * imgs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, num_classes=19):
    model.eval()
    losses = []
    f_scores = []
    ce = torch.nn.CrossEntropyLoss()
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss = ce(logits, masks)
        preds = torch.argmax(logits, dim=1)
        f = fscore_mean_per_class(masks, preds, num_classes=num_classes)
        losses.append(float(loss.item()) * imgs.size(0))
        f_scores.append(f * imgs.size(0))
    return sum(losses)/len(loader.dataset), sum(f_scores)/len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="path containing train/images and train/masks")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt_out", type=str, default="solution/ckpt.pth")
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images = os.path.join(args.data_root, "train/images")
    train_masks = os.path.join(args.data_root, "train/masks")

    all_files = sorted(os.listdir(train_images))
    val_ids = set(all_files[-100:])
    train_ids = [f for f in all_files if f not in val_ids]

    t_train = get_train_tfms()
    t_val = get_val_tfms()

    ds_train = CelebAMaskMini(train_images, train_masks, transform=t_train)
    ds_val   = CelebAMaskMini(train_images, train_masks, transform=t_val)

    ds_train.files = [f for f in ds_train.files if f in train_ids]
    ds_val.files   = [f for f in ds_val.files if f in val_ids]

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=2*args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DWUNet(in_ch=3, num_classes=19, base=args.base).to(device)
    n_params = count_trainable_params(model)
    assert n_params < 1821085, f"Model has {n_params} params, which exceeds the limit."
    print(f"Params: {n_params}")

    loss_fn = ComboLoss(num_classes=19, ce_weight=None, dice_weight=0.5, ce_weight_factor=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f = -1.0
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, loader_train, optimizer, loss_fn, device)
        val_loss, val_f = evaluate(model, loader_val, device, num_classes=19)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_F {val_f:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
        if val_f > best_f:
            best_f = val_f
            torch.save({"model": model.state_dict(), "base": args.base}, args.ckpt_out)
            print(f"  Saved checkpoint -> {args.ckpt_out}")

if __name__ == "__main__":
    main()
