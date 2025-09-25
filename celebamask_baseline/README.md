
# CelebAMask Mini-Challenge — Baseline (DW‑UNet < 1.82M params)

From‑scratch, single‑model baseline compliant with the rules.

## Train
```bash
pip install -r solution/requirements.txt
python train.py --data_root /path/to/dev-public
```
This uses the last 100 train images as a validation split and saves `solution/ckpt.pth` on best val F.

## Predict (for test/images)
```bash
python solution/run.py --input_dir /path/to/dev-public/test/images --output_dir ./masks --ckpt solution/ckpt.pth
```
Writes indexed PNG masks (values 0..18).

Param count is asserted to be < 1,821,085 before training.
