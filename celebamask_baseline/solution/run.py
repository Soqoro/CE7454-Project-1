
import os
import argparse
import torch
import numpy as np
from PIL import Image
from model import DWUNet

def save_indexed_png(mask_np, out_path, palette=None):
    mask_img = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    if palette is not None and len(palette) >= 57:
        mask_img.putpalette(palette[:57])
    mask_img.save(out_path, format="PNG", optimize=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="ckpt.pth")
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DWUNet(in_ch=3, num_classes=19, base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for fname in files:
        path = os.path.join(args.input_dir, fname)
        img = Image.open(path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(args.output_dir, out_name)
        save_indexed_png(pred, out_path)

if __name__ == "__main__":
    main()
