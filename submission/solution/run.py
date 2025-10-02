# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2025-02-18 19:09:59
# @Last Modified by: ChatGPT
# @Last Modified at: 2025-10-02
# @Email:  root@haozhexie.com

import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image

# Match training/tester: images are resized to (imsize, imsize) with **bilinear**
TARGET_SIZE = 512
USE_TTA = True  # horizontal flip TTA (single model, averaged logits)


def load_model(weights_path: str, device: torch.device):
    """Construct UNet and load weights; handle both raw state_dict and {'state_dict': ...}."""
    from unet import unet
    model = unet().to(device)
    model.eval()

    ckpt = torch.load(weights_path, map_location=device)
    # Accept common formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # assume it's already a state_dict
        state = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {weights_path}")

    # strip 'module.' (DataParallel)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def preprocess(input_path: str, device: torch.device):
    """Read image (BGR), convert to RGB, resize (bilinear), normalize to [-1,1], to NCHW float32."""
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")
    h0, w0 = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Bilinear resize to model's training size
    if (h0, w0) != (TARGET_SIZE, TARGET_SIZE):
        img_rgb = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

    # To float32 and normalize like transforms.Normalize((0.5,)*3, (0.5,)*3)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # -> [-1, 1]

    # HWC -> NCHW -> tensor
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]
    x = x.to(device, dtype=torch.float32, non_blocking=True)
    return x, (h0, w0)


def postprocess_and_save(mask_pred: np.ndarray, orig_hw: tuple[int, int], output_path: str):
    """mask_pred: [H,W] uint8 in 0..18 at TARGET_SIZE; resize back to original with NEAREST and save."""
    h0, w0 = orig_hw
    if (TARGET_SIZE, TARGET_SIZE) != (w0, h0):
        mask_img = Image.fromarray(mask_pred, mode="L").resize((w0, h0), Image.Resampling.NEAREST)
    else:
        mask_img = Image.fromarray(mask_pred, mode="L")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mask_img.save(output_path)


def infer(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Run model (with optional flip-TTA) and return uint8 mask [H,W] in 0..18."""
    with torch.no_grad():
        logits = model(x)  # [1,C,H,W]
        if USE_TTA:
            logits_flip = model(torch.flip(x, dims=[3]))
            logits_flip = torch.flip(logits_flip, dims=[3])
            logits = 0.5 * (logits + logits_flip)

        pred = logits.argmax(1)  # [1,H,W]
        mask = pred.squeeze(0).contiguous().detach().cpu().numpy().astype(np.uint8)
    return mask


def main(input_path: str, output_path: str, weights_path: str):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = load_model(weights_path, device)

    # Preprocess
    x, orig_hw = preprocess(input_path, device)

    # Inference
    mask = infer(model, x)

    # Save
    postprocess_and_save(mask, orig_hw, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
