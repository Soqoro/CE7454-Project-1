import os
import time
import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast           # <- updated import
from torch.amp.grad_scaler import GradScaler           # <- updated import
from torchvision.utils import save_image

from unet import unet
from utils import (
    denorm,
    cross_entropy2d,
    soft_dice_loss,
    compute_class_weights,
    fast_hist_np,
    f1_from_confusion,
    generate_label,
)

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/training')


class Trainer(object):
    def __init__(self, data_loader, config):
        # ----------------- Data & config -----------------
        self.data_loader = data_loader                   # train loader
        self.val_loader = getattr(config, "val_loader", None)  # may be None

        # exact model and loss (kept for compat)
        self.model = config.model

        # Hyper-parameters / paths
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay  # kept; we use poly_power below
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.use_tensorboard = config.use_tensorboard
        self.img_path = config.img_path
        self.label_path = config.label_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # NEW (safe defaults if not in config)
        self.ignore_index = int(getattr(config, "ignore_index", 250))
        self.dice_weight = float(getattr(config, "dice_weight", 0.5))
        self.use_amp = bool(getattr(config, "use_amp", True))
        self.weight_decay = float(getattr(config, "weight_decay", 1e-4))
        self.poly_power = float(getattr(config, "poly_power", 0.9))
        self.validate_every = int(getattr(config, "validate_every", 1000))
        self.num_classes = int(getattr(config, "num_classes", 19))
        self.use_class_weights = bool(getattr(config, "use_class_weights", False))
        self.class_weights_path: Optional[str] = getattr(config, "class_weights_path", None)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        self.log_path = os.path.join(self.log_path, self.version)
        self.sample_path = os.path.join(self.sample_path, self.version)
        self.model_save_path = os.path.join(self.model_save_path, self.version)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        # Build model/opt
        self.build_model()

        # TensorBoard
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model (resume)
        if self.pretrained_model:
            self.load_pretrained_model()

        # Class weights (optional)
        self.class_weights = None
        if self.use_class_weights:
            if self.class_weights_path and os.path.isfile(self.class_weights_path):
                self.class_weights = torch.load(self.class_weights_path, map_location="cpu")
                print(f"Loaded class weights from {self.class_weights_path}")
            else:
                print("Computing class weights from training masks...")
                self.class_weights = compute_class_weights(
                    self.label_path, num_classes=self.num_classes, ignore_index=self.ignore_index
                )
                torch.save(self.class_weights, os.path.join(self.model_save_path, "class_weights.pt"))
                print("Saved class weights to checkpoints directory.")
            self.class_weights = self.class_weights.to(self.device)

        # AMP scaler (new API / import path)
        self.scaler = GradScaler('cuda', enabled=(self.use_amp and torch.cuda.is_available()))

        # Best val metric
        self.best_mf1 = -1.0

        # Optionally build a val loader if config provides paths (only if not already passed in)
        if self.val_loader is None and hasattr(config, "val_img_path") and hasattr(config, "val_label_path"):
            try:
                from data_loader import Data_Loader  # local import to avoid circulars
                val_dl = Data_Loader(
                    img_path=config.val_img_path,
                    label_path=config.val_label_path,
                    image_size=self.imsize,
                    batch_size=self.batch_size,
                    mode=False,
                )
                self.val_loader = val_dl.loader()
                print("Built internal validation DataLoader from provided paths.")
            except Exception as e:
                print(f"Warning: could not build internal val loader ({e}). Validation will be skipped.")

    def _state_dict(self):
        """Handle DataParallel transparently."""
        return self.G.module.state_dict() if isinstance(self.G, nn.DataParallel) else self.G.state_dict()

    def train(self):
        assert self.data_loader is not None, "Trainer.data_loader is None"
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        start = self.pretrained_model + 1 if self.pretrained_model else 0
        start_time = time.time()

        for step in range(start, self.total_step):
            self.G.train()

            # Poly LR
            lr = self.g_lr * (1.0 - float(step) / float(max(1, self.total_step))) ** self.poly_power
            for pg in self.g_optimizer.param_groups:
                pg["lr"] = lr

            # Next batch
            try:
                imgs, labels = next(data_iter)
            except Exception:
                data_iter = iter(self.data_loader)
                imgs, labels = next(data_iter)

            # labels: [B,1,H,W] float in [0,1] -> restore ids
            labels = labels.clone()
            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0   # 0..18 / 250 (/possibly 255)

            # targets (sanitize)
            labels_real_plain = labels[:, 0, :, :].to(self.device)  # [B,H,W] float
            labels_real_plain = labels_real_plain.clamp_(0, 255)
            # Any id >= num_classes (e.g., 250, 255) -> ignore_index
            oob = labels_real_plain >= self.num_classes
            labels_real_plain[oob] = float(self.ignore_index)

            # For visualization, build safe one-hot without touching training labels
            labels_index_long = labels[:, 0, :, :].long()           # CPU float->long ok for viz path
            # Replace ignore with 0, clamp all values into [0, C-1]
            labels_vis = torch.where(labels_index_long == self.ignore_index,
                                     torch.zeros_like(labels_index_long),
                                     labels_index_long)
            labels_vis = labels_vis.clamp_(0, self.num_classes - 1)
            labels_real = F.one_hot(labels_vis, num_classes=self.num_classes).permute(0, 3, 1, 2).float().to(self.device)

            imgs = imgs.to(self.device, non_blocking=True)

            # ================== Train G =================== #
            self.reset_grad()
            dice_value = torch.tensor(0.0, device=self.device)  # pre-init for logging

            with autocast('cuda', enabled=(self.use_amp and torch.cuda.is_available())):
                logits = self.G(imgs)  # [B,C,H,W]
                ce = cross_entropy2d(
                    logits,
                    labels_real_plain.long(),
                    weight=self.class_weights,
                    ignore_index=self.ignore_index,
                )
                if self.dice_weight > 0.0:
                    dice_value = soft_dice_loss(logits, labels_real_plain.long(), ignore_index=self.ignore_index)
                    loss = ce + self.dice_weight * dice_value
                else:
                    loss = ce

            # Backward, clip, step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.g_optimizer)
            nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
            self.scaler.step(self.g_optimizer)
            self.scaler.update()

            # --------- Logging (DETACHED SCALARS) ---------
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=int(elapsed)))

                # detach + item() to avoid UserWarning
                ce_val   = ce.detach().item()
                loss_val = loss.detach().item()
                dice_val = dice_value.detach().item() if self.dice_weight > 0.0 else 0.0

                msg = (f"Elapsed [{elapsed}], G_step [{step + 1}/{self.total_step}], "
                       f"lr={lr:.6f}, CE: {ce_val:.4f}")
                if self.dice_weight > 0.0:
                    msg += f", Dice: {dice_val:.4f}"
                msg += f", Loss: {loss_val:.4f}"
                print(msg)

                writer.add_scalar('Opt/lr', lr, step)
                writer.add_scalar('Loss/CE', ce_val, step)
                if self.dice_weight > 0.0:
                    writer.add_scalar('Loss/Dice', dice_val, step)
                writer.add_scalar('Loss/Total', loss_val, step)

            # ---- Visualize labels (real vs predict) ----
            with torch.no_grad():
                label_batch_predict = generate_label(logits, self.imsize)     # Tensor [B,3,H,W]
                label_batch_real = generate_label(labels_real, self.imsize)   # Tensor [B,3,H,W]

            # TBX images
            B = imgs.shape[0]
            img_combine = imgs[0]
            real_combine = label_batch_real[0]
            predict_combine = label_batch_predict[0]
            for i in range(1, B):
                img_combine = torch.cat([img_combine, imgs[i]], 2)
                real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                predict_combine = torch.cat([predict_combine, label_batch_predict[i]], 2)

            writer.add_image('imresult/img', (img_combine.detach() + 1) / 2.0, step)
            writer.add_image('imresult/real', real_combine.detach(), step)
            writer.add_image('imresult/predict', predict_combine.detach(), step)

            # ---- Sample grid save ----
            if (step + 1) % self.sample_step == 0:
                with torch.no_grad():
                    labels_sample = self.G(imgs)
                    labels_sample = generate_label(labels_sample, self.imsize)  # Tensor
                    labels_sample = labels_sample.detach().cpu().float()
                save_image(labels_sample, os.path.join(self.sample_path, f'{step + 1}_predict.png'))

            # ---- CHECKPOINTING ----
            if (step + 1) % model_save_step == 0:
                torch.save(self._state_dict(), os.path.join(self.model_save_path, f'{step + 1}_G.pth'))
                torch.save(self._state_dict(), os.path.join(self.model_save_path, 'latest_G.pth'))

            # ---- Periodic validation ----
            if (self.val_loader is not None) and ((step + 1) % self.validate_every == 0 or (step + 1) == self.total_step):
                mf1 = self.validate(self.val_loader)
                print(f"Validation mF1 @ step {step + 1}: {mf1:.4f}")
                writer.add_scalar('Val/mF1', mf1, step)
                if mf1 > self.best_mf1:
                    self.best_mf1 = mf1
                    best_path = os.path.join(self.model_save_path, "best_G.pth")
                    torch.save(self._state_dict(), best_path)
                    print(f"Saved new best to {best_path} (mF1={mf1:.4f})")

        # Final rolling checkpoint
        torch.save(self._state_dict(), os.path.join(self.model_save_path, 'latest_G.pth'))
        print(f'Final checkpoint saved to {os.path.join(self.model_save_path, "latest_G.pth")}')

    def build_model(self):
        self.G = unet().to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        num_params = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
        assert num_params < 1_821_085, f"Model too large: {num_params} parameters (cap = 1,821,085)"
        print(f"Trainable params: {num_params:,}")

        # Optimizer: AdamW
        self.g_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.G.parameters()),
            lr=self.g_lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )
        print(self.G)

    def build_tensorboard(self):
        try:
            from logger import Logger  # type: ignore
            self.logger = Logger(self.log_path)
        except Exception:
            self.logger = None

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model)), map_location=self.device))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def validate(self, loader) -> float:
        """Compute mean F1 on the given validation loader (ignore label respected)."""
        if loader is None:
            return 0.0
        self.G.eval()
        import numpy as np

        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for imgs, labels in loader:
            labels = labels.clone()
            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
            labs = labels[:, 0, :, :].long().cpu().numpy()  # [B,H,W]

            imgs = imgs.to(self.device, non_blocking=True)
            logits = self.G(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # [B,H,W]

            for p, t in zip(preds, labs):
                hist = fast_hist_np(t.flatten(), p.flatten(), self.num_classes, ignore_index=self.ignore_index)
                confusion += hist

        _, mf1 = f1_from_confusion(confusion)
        self.G.train()
        return float(mf1)

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
