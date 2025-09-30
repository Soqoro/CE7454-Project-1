import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path
from typing import List, Tuple


class CelebAMaskHQ(Dataset):
    """
    - Discovers files from directories (no fixed numbering).
    - Pairs image<->mask by filename stem.
    - Skips items with missing masks.
    - Returns labels as float [1,H,W] in [0,1] so trainer's '*255' restores 0..18.
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
            ops.append(transforms.Resize((self.imsize, self.imsize)))
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
            shuffle=True,
            num_workers=2,  # set to 0 temporarily for clearer tracebacks
            drop_last=False,
        )
        return loader
