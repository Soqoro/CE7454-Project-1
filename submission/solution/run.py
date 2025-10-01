# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2025-02-18 19:09:59
# @Last Modified by: ChatGPT
# @Last Modified at: 2025-10-01
# @Email:  root@haozhexie.com

import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image

# Match training: images were resized to (imsize, imsize) then normalized
TARGET_SIZE = 512   # your training resolution

def main(input, output, weights):
    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Read & preprocess image (match tester.py) ----------
    # Read BGR -> RGB
    img_bgr = cv2.imread(input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {input}")
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to training size using NEAREST (your tester does NEAREST)
    if (h0, w0) != (TARGET_SIZE, TARGET_SIZE):
        img_rgb = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

    # Normalize to [-1, 1] like transforms.Normalize((0.5,)*3, (0.5,)*3)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # == 2*img - 1

    # HWC -> NCHW torch tensor
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W], float32

    # ---------- Initialize model ----------
    # unet() must be available in unet.py and match your training model
    from unet import unet
    model = unet().to(device)  # same default signature you used in training
    model.eval()

    # ---------- Load checkpoint (CondaBench format) ----------
    ckpt = torch.load(weights, map_location=device)
    # Per CondaBench spec: weights must be under "state_dict"
    state = ckpt["state_dict"]
    # Strip 'module.' if saved with DataParallel
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    # ---------- Inference ----------
    with torch.no_grad():
        logits = model(x)          # [1, C, H, W] with C=19
        pred = logits.argmax(1)    # [1, H, W]
        mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)  # 0..18

    # ---------- Resize mask back to original size (NEAREST) ----------
    if (TARGET_SIZE, TARGET_SIZE) != (w0, h0):
        mask_img = Image.fromarray(mask, mode="L").resize((w0, h0), Image.Resampling.NEAREST)
    else:
        mask_img = Image.fromarray(mask, mode="L")

    # ---------- Save output (8-bit index map) ----------
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mask_img.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
