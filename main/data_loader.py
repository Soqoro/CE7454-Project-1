import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path

class CelebAMaskHQ():
    """
    - Discovers files from directories (no fixed numbering).
    - Pairs image<->mask by filename stem.
    - Skips items with missing masks.
    - Returns labels as float [1,H,W] in [0,1] so trainer's "* 255" restores 0..18.
    """
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = Path(img_path)
        self.label_path = Path(label_path)
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode  # True = train

        self.preprocess()
        self.num_images = len(self.train_dataset) if self.mode else len(self.test_dataset)

    def preprocess(self):
        img_dir, label_dir = self.img_path, self.label_path
        exts = {".jpg", ".jpeg", ".png"}
        img_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

        found, missing = 0, 0
        for img in img_files:
            stem = img.stem
            cand = [label_dir / f"{stem}.png", label_dir / f"{stem}.jpg", label_dir / f"{stem}.jpeg"]
            mask = next((p for p in cand if p.exists()), None)
            if mask is None:
                print(f"WARNING: missing mask for {img.name}")
                missing += 1
                continue
            pair = [str(img), str(mask)]
            (self.train_dataset if self.mode else self.test_dataset).append(pair)
            found += 1

        split = "train" if self.mode else "val/test"
        print(f"[{split}] Finished preprocessing. Pairs found: {found}, missing masks: {missing}")
        if found == 0:
            raise RuntimeError(f"No (image, mask) pairs found in {img_dir} / {label_dir}")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode else self.test_dataset
        img_path, label_path = dataset[index]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)  # expect 'L' or 'P' (single-channel / palette)

        # Sanity: avoid accidental colorized masks
        if label.mode not in ("L", "P"):
            raise ValueError(
                f"Mask '{label_path}' has mode '{label.mode}'. "
                f"Expected a single-channel label map (mode 'L' or 'P'). "
                f"Do NOT use colorized masks."
            )

        image = self.transform_img(image)
        label = self.transform_label(label)  # -> float [1,H,W] in [0,1]
        return image, label

    def __len__(self):
        return self.num_images


class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize, totensor, normalize, centercrop):
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

    def transform_label(self, resize, totensor, normalize, centercrop):
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

    def loader(self):
        transform_img = self.transform_img(True, True, True, False)
        transform_label = self.transform_label(True, True, False, False)

        dataset = CelebAMaskHQ(self.img_path, self.label_path, transform_img, transform_label, self.mode)

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=True,
            num_workers=2,   # set to 0 temporarily for clearer tracebacks
            drop_last=False
        )
        return loader
