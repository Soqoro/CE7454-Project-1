import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import cv2
from pathlib import Path
from unet import unet
from utils import *
from PIL import Image


def transformer(resize: bool, totensor: bool, normalize: bool, centercrop: bool, imsize: int):
    ops = []
    if centercrop:
        ops.append(transforms.CenterCrop(160))
    if resize:
        ops.append(transforms.Resize((imsize, imsize), interpolation=InterpolationMode.NEAREST))
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

        # Paths
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.test_label_path = config.test_label_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure output dirs exist
        os.makedirs(self.test_label_path, exist_ok=True)
        os.makedirs(self.test_color_label_path, exist_ok=True)

        self.build_model()

    def test(self):
        transform = transformer(True, True, True, False, self.imsize)

        test_paths = list_images(self.test_image_path)
        if len(test_paths) == 0:
            raise RuntimeError(f"No test images found in {self.test_image_path}")

        # load weights
        state_path = os.path.join(self.model_save_path, self.model_name)
        self.G.load_state_dict(torch.load(state_path, map_location=self.device))
        self.G.eval()

        # We iterate over actual number of images (not the config.test_size)
        with torch.no_grad():
            for start in range(0, len(test_paths), self.batch_size):
                end = min(start + self.batch_size, len(test_paths))
                curr_paths = test_paths[start:end]

                imgs = []
                for path in curr_paths:
                    img = transform(Image.open(path).convert("RGB"))
                    imgs.append(img)
                imgs = torch.stack(imgs).to(self.device, non_blocking=True)

                labels_predict = self.G(imgs)  # [B,19,H,W] logits

                # Convert predictions to plain mask and color visualization
                # IMPORTANT: pass imsize (fixes your "Argument missing for parameter 'imsize'" error)
                labels_predict_plain = generate_label_plain(labels_predict, self.imsize)  # expected np arrays
                labels_predict_color = generate_label(labels_predict, self.imsize)        # expected torch.Tensor

                # Ensure types
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
                    # Ensure uint8 for cv2.imwrite
                    if plain_k.dtype != np.uint8:
                        plain_k = plain_k.astype(np.uint8)
                    cv2.imwrite(os.path.join(self.test_label_path, f"{stem}.png"), plain_k)

                    # color viz (tensor [3,H,W] or [1,H,W])
                    color_k = labels_predict_color[k]
                    save_image(color_k, os.path.join(self.test_color_label_path, f"{stem}.png"))

    def build_model(self):
        self.G = unet().to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
        print(self.G)
