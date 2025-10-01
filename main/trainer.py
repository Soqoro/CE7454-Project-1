import os
import time
import torch
import datetime

import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F

from unet import unet
from utils import *
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/training')


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
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

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def _state_dict(self):
        """Handle DataParallel transparently."""
        return self.G.module.state_dict() if isinstance(self.G, nn.DataParallel) else self.G.state_dict()

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Start with trained model
        start = self.pretrained_model + 1 if self.pretrained_model else 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            self.G.train()
            try:
                imgs, labels = next(data_iter)
            except Exception:
                data_iter = iter(self.data_loader)
                imgs, labels = next(data_iter)

            # labels expected as [B,1,H,W] float in [0,1] from the patched dataloader
            size = labels.size()
            # restore class ids 0..18
            labels = labels.clone()
            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0

            # targets
            labels_real_plain = labels[:, 0, :, :].to(self.device)  # [B,H,W] long later
            labels_index = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])

            # one-hot for visualization only
            oneHot_size = (size[0], 19, size[2], size[3])
            labels_real = torch.zeros(oneHot_size, dtype=torch.float32, device=self.device)
            labels_real = labels_real.scatter_(1, labels_index.long().to(self.device), 1.0)

            imgs = imgs.to(self.device, non_blocking=True)

            # ================== Train G =================== #
            labels_predict = self.G(imgs)  # [B,19,H,W]

            # Cross entropy expects target Long [B,H,W]
            c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())

            self.reset_grad()
            c_loss.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=int(elapsed)))
                print("Elapsed [{}], G_step [{}/{}], Cross_entropy_loss: {:.4f}".format(
                    elapsed, step + 1, self.total_step, c_loss.item()))

            # Visualize labels (both real one-hot and predicted logits)
            label_batch_predict = generate_label(labels_predict, self.imsize)  # Tensor
            label_batch_real = generate_label(labels_real, self.imsize)        # Tensor

            # scalar info on tensorboardX
            writer.add_scalar('Loss/Cross_entropy_loss', c_loss.item(), step)

            # images on tensorboardX
            B = imgs.shape[0]
            img_combine = imgs[0]
            real_combine = label_batch_real[0]
            predict_combine = label_batch_predict[0]
            for i in range(1, B):
                img_combine = torch.cat([img_combine, imgs[i]], 2)
                real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                predict_combine = torch.cat([predict_combine, label_batch_predict[i]], 2)

            # inputs likely normalized to [-1,1]; map back to [0,1]
            writer.add_image('imresult/img', (img_combine.detach() + 1) / 2.0, step)
            writer.add_image('imresult/real', real_combine.detach(), step)
            writer.add_image('imresult/predict', predict_combine.detach(), step)

            # Sample images (save predicted labels grid)
            if (step + 1) % self.sample_step == 0:
                labels_sample = self.G(imgs)
                labels_sample = generate_label(labels_sample, self.imsize)  # returns Tensor
                if isinstance(labels_sample, torch.Tensor):
                    labels_sample = labels_sample.detach().cpu().float()
                else:
                    import numpy as np
                    if isinstance(labels_sample, np.ndarray):
                        labels_sample = torch.from_numpy(labels_sample).float()
                    else:
                        raise TypeError(f"generate_label returned unsupported type: {type(labels_sample)}")

                save_image(
                    labels_sample,
                    os.path.join(self.sample_path, f'{step + 1}_predict.png')
                )

            # ---------------------- CHECKPOINTING ----------------------
            if (step + 1) % model_save_step == 0:
                # numbered checkpoint
                torch.save(self._state_dict(),
                           os.path.join(self.model_save_path, f'{step + 1}_G.pth'))
                # rolling/stable checkpoint for tester & sweeps
                torch.save(self._state_dict(),
                           os.path.join(self.model_save_path, 'latest_G.pth'))
            # ----------------------------------------------------------

        # Always leave a final rolling checkpoint even if loop ended off-cycle
        torch.save(self._state_dict(),
                   os.path.join(self.model_save_path, 'latest_G.pth'))
        print(f'Final checkpoint saved to {os.path.join(self.model_save_path, "latest_G.pth")}')

    def build_model(self):

        self.G = unet().to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
        
        num_params = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
        assert num_params < 1_821_085, f"Model too large: {num_params} parameters (cap = 1,821,085)"
        print(f"Trainable params: {num_params:,}")

        # Loss and optimizer (use tuple for betas; remove duplicate assignment)
        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.G.parameters()),
            lr=self.g_lr,
            betas=(self.beta1, self.beta2)
        )

        # print networks
        print(self.G)

    def build_tensorboard(self):
        # Optional legacy logger; ignore if not present
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
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
