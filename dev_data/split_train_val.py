# split_train_val.py
import argparse
import random
from pathlib import Path
import shutil

def main(root=".", val_ratio=0.2, seed=42):
    root = Path(root)
    train_images = root / "train" / "images"
    train_masks  = root / "train" / "masks"
    val_images   = root / "val" / "images"
    val_masks    = root / "val" / "masks"

    # Make output dirs
    val_images.mkdir(parents=True, exist_ok=True)
    val_masks.mkdir(parents=True, exist_ok=True)

    # Match images to masks by filename stem
    image_paths = sorted(p for p in train_images.glob("*") if p.is_file())
    pairs = []
    missing = []

    for img in image_paths:
        stem = img.stem
        # accept any mask extension
        candidates = list(train_masks.glob(stem + ".*"))
        if candidates:
            pairs.append((img, candidates[0]))
        else:
            missing.append(img.name)

    if missing:
        print(f"WARNING: {len(missing)} image(s) have no matching mask and will be ignored.")
        print("Examples:", missing[:10])

    total = len(pairs)
    if total == 0:
        raise SystemExit("No (image, mask) pairs found. Check your paths.")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(round(total * val_ratio))
    val_pairs = pairs[:n_val]

    # Move validation pairs out of train into val
    for img_path, mask_path in val_pairs:
        shutil.move(str(img_path), str(val_images / img_path.name))
        shutil.move(str(mask_path), str(val_masks / mask_path.name))

    print("Done.")
    print(f"Total pairs originally in train: {total}")
    print(f"Moved to val: {len(val_pairs)}")
    print(f"Remaining in train: {total - len(val_pairs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="dataset root dir")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="fraction for validation set")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args.root, args.val_ratio, args.seed)
