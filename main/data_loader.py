import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


IGNORE_INDEX = 250  # keep consistent with training/eval


# ---------------------------
# Paired (image+mask) augmentation for train mode
# ---------------------------
class PairedAugment:
    """
    Apply the same geometric transform to image & mask.
    - Geometry is done on tensors to satisfy torchvision type hints.
    - Image: bilinear interpolation, RGB-only photometric jitter, optional cutout.
    - Mask: nearest interpolation; out-of-bounds fill -> IGNORE_INDEX.
    """

    def __init__(
        self,
        hflip_p: float = 0.5,
        degrees: float = 15.0,
        translate: float = 0.05,            # fraction of width/height
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shear: float = 5.0,
        color_jitter: Tuple[float, float, float, float] = (0.2, 0.2, 0.15, 0.05),  # brightness, contrast, saturation, hue
        cutout_p: float = 0.3,
        cutout_size: int = 64,
    ) -> None:
        self.hflip_p = hflip_p
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.shear = shear
        self.color_jitter = color_jitter
        self.cutout_p = cutout_p
        self.cutout_size = cutout_size

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Use tensor ops for geometry (matches TF type hints), then convert back to PIL.
        - img_t: float32 [C,H,W] in [0,1]
        - mask_t: uint8   [H,W]   (class indices)
        """
        # --- PIL -> Tensor
        img_t = TF.pil_to_tensor(img).float() / 255.0          # [C,H,W], float in [0,1]
        mask_t = TF.pil_to_tensor(mask).squeeze(0)             # [H,W], uint8

        # --- Horizontal flip (tensor)
        if random.random() < self.hflip_p:
            img_t = torch.flip(img_t, dims=[-1])               # flip width
            mask_t = torch.flip(mask_t, dims=[-1])

        # --- Shared affine params
        C = int(img_t.shape[0])
        H = int(img_t.shape[-2])
        W = int(img_t.shape[-1])

        angle = float(random.uniform(-self.degrees, self.degrees))

        max_dx = self.translate * W
        max_dy = self.translate * H
        # -> list[int] for type hints
        translations = [int(round(random.uniform(-max_dx, max_dx))),
                        int(round(random.uniform(-max_dy, max_dy)))]

        scale = float(random.uniform(self.scale_range[0], self.scale_range[1]))
        # -> list[float] (shear_x, shear_y)
        shear = [float(random.uniform(-self.shear, self.shear)), 0.0]

        # --- Apply affine
        # fill: list[float] with length C for tensor images
        img_fill = [0.0] * C
        img_t = TF.affine(
            img_t, angle=angle, translate=translations, scale=scale, shear=shear,
            interpolation=InterpolationMode.BILINEAR, fill=img_fill
        )  # -> float [C,H,W]

        # mask_t is [H,W]; use channel dim for fill handling (length=1 list)
        mask_fill = [float(IGNORE_INDEX)]
        mask_t = TF.affine(
            mask_t.unsqueeze(0), angle=angle, translate=translations, scale=scale, shear=shear,
            interpolation=InterpolationMode.NEAREST, fill=mask_fill
        ).squeeze(0).to(torch.uint8)  # -> uint8 [H,W]

        # --- Photometric jitter (image only) on tensor
        if self.color_jitter is not None:
            b, c, s, h = self.color_jitter
            img_t = TF.adjust_brightness(img_t, 1.0 + float(random.uniform(-b, b)))
            img_t = TF.adjust_contrast(img_t,   1.0 + float(random.uniform(-c, c)))
            img_t = TF.adjust_saturation(img_t, 1.0 + float(random.uniform(-s, s)))
            img_t = TF.adjust_hue(img_t,                 float(random.uniform(-h, h)))

        # --- Cutout (image only) on tensor
        if random.random() < self.cutout_p:
            sz = int(self.cutout_size)
            x0 = int(random.randint(0, max(0, W - sz)))
            y0 = int(random.randint(0, max(0, H - sz)))
            img_t[:, y0:y0 + sz, x0:x0 + sz] = 0.0

        # --- Tensor -> PIL (preserve downstream pipeline expectations)
        img_out = TF.to_pil_image(img_t.clamp(0.0, 1.0))       # back to PIL RGB
        mask_out = TF.to_pil_image(mask_t)                     # back to PIL L
        return img_out, mask_out


class CelebAMaskHQ(Dataset):
    """
    - Discovers files from directories (no fixed numbering).
    - Pairs image<->mask by filename stem.
    - Skips items with missing masks.
    - Returns labels as float [1,H,W] in [0,1] so trainer's '*255' restores 0..18 (and 250 for ignore).
    """

    def __init__(
        self,
        img_path: str,
        label_path: str,
        transform_img: transforms.Compose,
        transform_label: transforms.Compose,
        mode: bool,
    ) -> None:
        self.img_path = Path(img_path)
        self.label_path = Path(label_path)
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.mode = mode  # True = train, False = val/test
        self.augment = PairedAugment() if self.mode else None

        self.samples: List[Tuple[str, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        img_dir, label_dir = self.img_path, self.label_path
        exts = {".jpg", ".jpeg", ".png"}
        img_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

        found, missing = 0, 0
        for img in img_files:
            stem = img.stem
            # prefer .png for masks but accept common variants
            cand = [label_dir / f"{stem}.png", label_dir / f"{stem}.jpg", label_dir / f"{stem}.jpeg"]
            mask = next((p for p in cand if p.exists()), None)
            if mask is None:
                print(f"WARNING: missing mask for {img.name}")
                missing += 1
                continue
            self.samples.append((str(img), str(mask)))
            found += 1

        split = "train" if self.mode else "val/test"
        print(f"[{split}] Finished preprocessing. Pairs found: {found}, missing masks: {missing}")
        if found == 0:
            raise RuntimeError(f"No (image, mask) pairs found in {img_dir} / {label_dir}")

    def __getitem__(self, index: int):
        img_path, label_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)  # expect 'L' or 'P' (single-channel / palette)

        if label.mode not in ("L", "P"):
            raise ValueError(
                f"Mask '{label_path}' has mode '{label.mode}'. "
                f"Expected a single-channel label map (mode 'L' or 'P'). "
                f"Do NOT use colorized masks."
            )

        # Paired augmentations first (original resolution), train only
        if self.augment is not None:
            image, label = self.augment(image, label)

        # Deterministic transforms (resize -> tensor / normalize)
        image = self.transform_img(image)
        label = self.transform_label(label)  # float [1,H,W] in [0,1]
        return image, label

    def __len__(self) -> int:
        return len(self.samples)


class Data_Loader:
    def __init__(self, img_path: str, label_path: str, image_size: int, batch_size: int, mode: bool) -> None:
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize: bool, totensor: bool, normalize: bool, centercrop: bool):
        ops = []
        if centercrop:
            ops.append(transforms.CenterCrop(160))
        if resize:
            # Explicitly use BILINEAR for RGB images
            ops.append(transforms.Resize((self.imsize, self.imsize), interpolation=InterpolationMode.BILINEAR))
        if totensor:
            ops.append(transforms.ToTensor())  # float [0,1]
        if normalize:
            ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return transforms.Compose(ops)

    def transform_label(self, resize: bool, totensor: bool, normalize: bool, centercrop: bool):
        """
        Keep trainer-compatible output:
        - NEAREST resize (no interpolation artifacts).
        - ToTensor() -> float in [0,1] with a channel dim [1,H,W].
        - No normalization.
        """
        ops = []
        if centercrop:
            ops.append(transforms.CenterCrop(160))
        if resize:
            ops.append(transforms.Resize((self.imsize, self.imsize), interpolation=InterpolationMode.NEAREST))
        if totensor:
            ops.append(transforms.ToTensor())  # keeps shape [1,H,W], scales 0..255 -> 0..1
        # ignore `normalize` for labels on purpose
        return transforms.Compose(ops)

    def loader(self) -> torch.utils.data.DataLoader:
        transform_img = self.transform_img(True, True, True, False)
        transform_label = self.transform_label(True, True, False, False)

        dataset: Dataset = CelebAMaskHQ(
            self.img_path, self.label_path, transform_img, transform_label, self.mode
        )

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=self.mode,     # shuffle only for train
            num_workers=2,         # adjust as needed
            pin_memory=True,
            drop_last=False,
        )
        return loader
