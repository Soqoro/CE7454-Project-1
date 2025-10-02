import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from PIL import Image

from unet import unet
from utils import generate_label_plain, generate_label


def transformer(resize: bool, totensor: bool, normalize: bool, centercrop: bool, imsize: int):
    """
    Test-time image transform:
    - Center crop (optional)
    - Resize with BILINEAR (RGB only)
    - ToTensor()
    - Normalize to match training
    """
    ops = []
    if centercrop:
        ops.append(transforms.CenterCrop(160))
    if resize:
        ops.append(transforms.Resize((imsize, imsize), interpolation=InterpolationMode.BILINEAR))
    if totensor:
        ops.append(transforms.ToTensor())  # float [0,1]
    if normalize:
        ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # match training
    return transforms.Compose(ops)


def list_images(dir_path: str):
    """Return a sorted list of image paths from dir (supports jpg/jpeg/png)."""
    p = Path(dir_path)
    if not p.is_dir():
        raise AssertionError(f"{dir_path} is not a valid directory")
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted([str(x) for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])
    print(dir_path, len(files))
    return files


class Tester(object):
    def __init__(self, config):
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

        self.img_path = config.img_path
        self.label_path = config.label_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Output / input paths for testing
        self.test_label_path = config.test_label_path               # plain id masks
        self.test_color_label_path = config.test_color_label_path   # color viz
        self.test_image_path = config.test_image_path               # images folder

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        # New (optional) toggles
        self.use_amp = bool(getattr(config, "use_amp", True))
        self.tta = bool(getattr(config, "tta", True))  # enable flip-TTA by default

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths (versioned)
        self.log_path = os.path.join(self.log_path, self.version)
        self.sample_path = os.path.join(self.sample_path, self.version)
        self.model_save_path = os.path.join(self.model_save_path, self.version)

        # Ensure output dirs exist
        os.makedirs(self.test_label_path, exist_ok=True)
        os.makedirs(self.test_color_label_path, exist_ok=True)

        self.build_model()

    def build_model(self):
        self.G = unet().to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
        print(self.G)

    @torch.no_grad()
    def test(self):
        transform = transformer(True, True, True, False, self.imsize)

        test_paths = list_images(self.test_image_path)
        if len(test_paths) == 0:
            raise RuntimeError(f"No test images found in {self.test_image_path}")

        # load weights (try configured file, fallback to latest_G.pth)
        cand = [
            os.path.join(self.model_save_path, self.model_name),
            os.path.join(self.model_save_path, "best_G.pth"),
            os.path.join(self.model_save_path, "latest_G.pth"),
        ]
        state_path = next((p for p in cand if os.path.isfile(p)), None)
        if state_path is None:
            raise FileNotFoundError(f"No checkpoint found in {self.model_save_path} among: {cand}")

        print(f"Loading checkpoint: {state_path}")
        self.G.load_state_dict(torch.load(state_path, map_location=self.device))
        self.G.eval()

        start_time = time.time()
        for start in range(0, len(test_paths), self.batch_size):
            end = min(start + self.batch_size, len(test_paths))
            curr_paths = test_paths[start:end]

            imgs = []
            for path in curr_paths:
                img = transform(Image.open(path).convert("RGB"))
                imgs.append(img)
            imgs = torch.stack(imgs).to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.G(imgs)  # [B,19,H,W] logits
                if self.tta:
                    # horizontal flip TTA: same model, average logits
                    logits_flip = self.G(torch.flip(imgs, dims=[3]))
                    logits_flip = torch.flip(logits_flip, dims=[3])
                    logits = 0.5 * (logits + logits_flip)

            # Convert predictions to plain mask and color visualization
            labels_predict_plain = generate_label_plain(logits, self.imsize)  # np [B,H,W] uint8
            labels_predict_color = generate_label(logits, self.imsize)        # Tensor [B,3,H,W] in [0,1]

            # Ensure tensor for color save
            if isinstance(labels_predict_color, np.ndarray):
                labels_predict_color = torch.from_numpy(labels_predict_color)
            labels_predict_color = labels_predict_color.detach().cpu().float()

            # Save each item with its original filename stem
            for k, path in enumerate(curr_paths):
                stem = Path(path).stem

                # plain mask (np array)
                plain_k = labels_predict_plain[k]
                if isinstance(plain_k, torch.Tensor):
                    plain_k = plain_k.detach().cpu().numpy()
                if plain_k.dtype != np.uint8:
                    plain_k = plain_k.astype(np.uint8)
                cv2.imwrite(os.path.join(self.test_label_path, f"{stem}.png"), plain_k)

                # color viz (tensor [3,H,W] in [0,1])
                color_k = labels_predict_color[k]
                save_image(color_k, os.path.join(self.test_color_label_path, f"{stem}.png"))

        elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print(f"Finished inference on {len(test_paths)} images in {elapsed}. "
              f"Masks -> {self.test_label_path}, Colors -> {self.test_color_label_path}")
