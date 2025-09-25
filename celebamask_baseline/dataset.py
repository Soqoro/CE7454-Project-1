
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CelebAMaskMini(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert('RGB')

        if self.masks_dir is not None:
            mask_path = os.path.join(self.masks_dir, os.path.splitext(fname)[0] + '.png')
            mask = Image.open(mask_path).convert('P')
            img_np = np.array(img)
            mask_np = np.array(mask).astype(np.int64)
            if self.transform:
                augmented = self.transform(image=img_np, mask=mask_np)
                img_np = augmented['image']
                mask_np = augmented['mask']
            img_t = torch.from_numpy(img_np.transpose(2,0,1)).float() / 255.0
            mask_t = torch.from_numpy(mask_np).long()
            return img_t, mask_t, fname
        else:
            img_np = np.array(img)
            if self.transform:
                augmented = self.transform(image=img_np)
                img_np = augmented['image']
            img_t = torch.from_numpy(img_np.transpose(2,0,1)).float() / 255.0
            return img_t, fname
