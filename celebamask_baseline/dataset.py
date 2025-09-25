import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Help avoid OpenCV thread contention across DataLoader workers
cv2.setNumThreads(0)


class CelebAMaskMini(Dataset):
    """
    Expects:
      images_dir/  # RGB images (*.png|*.jpg|*.jpeg)
      masks_dir/   # indexed masks (*.png) with the SAME basename as the image but .png extension

    Example:
      images_dir/image_000123.jpg
      masks_dir/image_000123.png
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform=None,
        num_classes: int = 19,
        validate_labels: bool = False,
        mask_ext: str = ".png",
    ) -> None:
        """
        Args:
            images_dir: directory with RGB images
            masks_dir: directory with indexed PNG masks
            transform: Albumentations transform (expects keys 'image' and 'mask')
            num_classes: number of classes (for optional validation)
            validate_labels: if True, assert mask values are in [0, num_classes-1]
            mask_ext: mask filename extension (default '.png')
        """
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.num_classes = int(num_classes)
        self.validate_labels = bool(validate_labels)
        self.mask_ext = mask_ext if mask_ext.startswith(".") else f".{mask_ext}"

        # Build (image -> mask) pairs where mask uses fixed .png extension
        all_imgs = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(self.IMG_EXTS)]
        self.files: List[str] = []
        missing = 0
        for f in all_imgs:
            base, _ = os.path.splitext(f)
            mask_fname = base + self.mask_ext  # enforce .png masks
            if os.path.isfile(os.path.join(masks_dir, mask_fname)):
                self.files.append(f)  # store the image filename; derive mask on the fly
            else:
                missing += 1

        if len(self.files) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs found.\n"
                f"Checked {len(all_imgs)} images under {images_dir} with masks under {masks_dir} "
                f"(expected mask names to be <image_basename>{self.mask_ext})."
            )
        if missing > 0:
            print(f"[dataset] Skipped {missing} images with missing masks (expected extension {self.mask_ext}).")
        print(f"[dataset] Using {len(self.files)} pairs from {images_dir} and {masks_dir}.")

    def __len__(self) -> int:
        return len(self.files)

    def _read_image(self, path: str) -> np.ndarray:
        """
        Read an RGB image as HxWx3 uint8 (0..255).
        """
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # uint8
        return img_rgb

    def _read_mask(self, path: str) -> np.ndarray:
        """
        Read a mask as HxW uint8 class indices.
        """
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # single channel 0..255
        if m is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)

        if self.validate_labels:
            maxv = int(m.max())
            minv = int(m.min())
            assert 0 <= minv <= maxv <= (self.num_classes - 1), (
                f"Mask values out of range: min={minv}, max={maxv}, expected [0,{self.num_classes-1}] "
                f"at {path}"
            )
        return m  # HxW uint8

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_fname = self.files[idx]
        img_path = os.path.join(self.images_dir, img_fname)
        base, _ = os.path.splitext(img_fname)
        mask_path = os.path.join(self.masks_dir, base + self.mask_ext)

        img = self._read_image(img_path)   # HxWx3 uint8 [0..255]
        mask = self._read_mask(mask_path)  # HxW   uint8 class ids

        if self.transform is not None:
            # Albumentations handles float conversion and normalization if A.Normalize is in the pipeline.
            transformed = self.transform(image=img, mask=mask)
            img_np = transformed["image"]  # HxWx3, typically float32 normalized already
            mask_np = transformed["mask"]  # HxW uint8
            # Convert to tensors WITHOUT extra /255
            img_t = torch.from_numpy(np.ascontiguousarray(img_np.transpose(2, 0, 1))).float()
            mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).long()
        else:
            # No transform: scale image to [0,1], keep mask as indices
            img_np = img.astype(np.float32) / 255.0
            img_t = torch.from_numpy(np.ascontiguousarray(img_np.transpose(2, 0, 1))).float()
            mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t, img_fname
