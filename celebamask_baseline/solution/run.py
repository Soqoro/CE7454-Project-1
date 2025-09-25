# solution/run.py
import os
import json
import argparse
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image

from model import DWUNet


def load_stats(ckpt_path: str, mean_arg: Optional[str], std_arg: Optional[str]) -> Tuple[List[float], List[float]]:
    """
    Load mean/std with this precedence:
      1) From checkpoint keys "mean"/"std"
      2) From 'train_stats.json' in the checkpoint directory
      3) From CLI --mean/--std (comma-separated 'r,g,b')
      4) Fallback to [0.5, 0.5, 0.5] / [0.25, 0.25, 0.25]
    Always returns a (mean, std) tuple of 3 floats each.
    """
    # 1) Try checkpoint
    try:
        ckpt_cpu = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt_cpu, dict) and "mean" in ckpt_cpu and "std" in ckpt_cpu:
            mean = ckpt_cpu.get("mean")
            std = ckpt_cpu.get("std")
            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)) and len(mean) == 3 and len(std) == 3:
                m = [float(x) for x in mean]
                s = [float(x) for x in std]
                print(f"[stats] Using mean/std from checkpoint: mean={m}, std={s}")
                return m, s
    except Exception:
        pass

    # 2) Try JSON next to checkpoint
    stats_json = os.path.join(os.path.dirname(ckpt_path) or ".", "train_stats.json")
    if os.path.isfile(stats_json):
        try:
            with open(stats_json, "r") as f:
                js = json.load(f)
            mean = js.get("mean")
            std = js.get("std")
            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)) and len(mean) == 3 and len(std) == 3:
                m = [float(x) for x in mean]
                s = [float(x) for x in std]
                print(f"[stats] Using mean/std from {stats_json}: mean={m}, std={s}")
                return m, s
        except Exception:
            pass

    # 3) CLI
    def parse_vec3(s: Optional[str]) -> Optional[List[float]]:
        if not s:
            return None
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            return None
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None

    mean_cli = parse_vec3(mean_arg)
    std_cli = parse_vec3(std_arg)
    if mean_cli and std_cli:
        print(f"[stats] Using mean/std from CLI: mean={mean_cli}, std={std_cli}")
        return mean_cli, std_cli

    # 4) Fallback
    m = [0.5, 0.5, 0.5]
    s = [0.25, 0.25, 0.25]
    print(f"[stats] Using fallback mean/std: mean={m}, std={s}")
    return m, s


def normalize_np(img_np: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """
    img_np: HxWx3 in [0,255] or [0,1]. Returns normalized float32 HxWx3.
    """
    if img_np.max() > 1.0:
        img_np = img_np.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (img_np - mean_arr) / (std_arr + 1e-8)


def ensure_pil_palette(palette: Optional[List[int]], num_classes: int = 19) -> List[int]:
    """
    Always return a full 256*3 palette (list of ints).
    If palette is shorter or None, pad with zeros.
    """
    PAL_SIZE = 256 * 3
    out: List[int] = [0] * PAL_SIZE
    if palette:
        ncopy = min(len(palette), PAL_SIZE)
        out[:ncopy] = list(palette[:ncopy])
    return out


def save_indexed_png(mask_np: np.ndarray, out_path: str, palette256: Optional[List[int]] = None) -> None:
    """
    Save an indexed PNG (mode 'P'). Provide a 256*3 palette for predictable colors.
    """
    mask_img = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    if palette256 is not None:
        mask_img.putpalette(palette256)  # list[int] of length 768
    mask_img.save(out_path, format="PNG", optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="directory of input images")
    ap.add_argument("--output_dir", type=str, required=True, help="directory to write output PNG masks")
    ap.add_argument("--ckpt", type=str, default="ckpt.pth")
    ap.add_argument("--base", type=int, default=56, help="fallback base channels if ckpt doesn't include it")
    ap.add_argument("--mean", type=str, default=None, help="comma-separated mean, e.g. 0.51,0.41,0.36")
    ap.add_argument("--std", type=str, default=None, help="comma-separated std,  e.g. 0.31,0.27,0.27")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    # Read base from ckpt if present
    ckpt_cpu = torch.load(args.ckpt, map_location="cpu")
    base = int(ckpt_cpu.get("base", args.base)) if isinstance(ckpt_cpu, dict) else args.base
    print(f"[model] Using base={base}")

    # Load normalization stats
    mean, std = load_stats(args.ckpt, args.mean, args.std)

    # Build and load model
    model = DWUNet(in_ch=3, num_classes=19, base=base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # Palette (sanitize to full 256*3 if present)
    raw_palette = ckpt.get("palette", None) if isinstance(ckpt, dict) else None
    palette256 = ensure_pil_palette(raw_palette, num_classes=19) if raw_palette is not None else None

    # Gather input files
    files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    num_written = 0
    with torch.no_grad():
        for fname in files:
            in_path = os.path.join(args.input_dir, fname)

            # Load & normalize
            img = Image.open(in_path).convert("RGB")
            img_np = np.array(img)  # HxWx3 uint8
            img_norm = normalize_np(img_np, mean, std)  # HxWx3 float32
            x = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)  # 1x3xHxW

            # Forward
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Save indexed PNG
            out_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(args.output_dir, out_name)
            save_indexed_png(pred, out_path, palette256=palette256)
            num_written += 1

    print(f"[done] Wrote {num_written} masks to {args.output_dir}")


if __name__ == "__main__":
    main()
