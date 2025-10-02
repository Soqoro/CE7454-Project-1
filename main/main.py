from parameter import *
from trainer import Trainer
from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder

# --- NEW: imports for sweep & scoring ---
import os, copy, csv, itertools, glob
import numpy as np
from PIL import Image


# ===== helpers for sweep =====
def _parse_csv_list(s, cast=float):
    s = (s or "").strip()
    if not s:
        return None
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def set_seed(seed: int):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fast_hist(pred, gt, n=19, ignore=250):
    mask = gt != ignore
    pred = pred[mask]
    gt = gt[mask]
    k = (gt * n + pred).astype(np.int64)
    hist = np.bincount(k, minlength=n * n).reshape(n, n).astype(np.float64)
    return hist


def eval_dir_mean_f1(pred_dir, gt_dir, num_classes=19, ignore=250):
    files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No predictions found in {pred_dir}")
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    for pf in files:
        name = os.path.basename(pf)
        gtf = os.path.join(gt_dir, name)
        if not os.path.exists(gtf):
            # allow missing GT (handy if you ran on subset)
            continue
        p = np.array(Image.open(pf))
        g = np.array(Image.open(gtf))
        hist += fast_hist(p, g, n=num_classes, ignore=ignore)
    tp = np.diag(hist)
    fp = hist.sum(0) - tp
    fn = hist.sum(1) - tp
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    mF1 = float(np.nanmean(f1))
    pixacc = float(tp.sum() / (hist.sum() + 1e-7))
    return mF1, pixacc


def _maybe_build_val_loader(config):
    """
    Build and return a validation DataLoader if config has val paths; else return None.
    Accepts either 'val_image_path' or 'val_img_path' naming.
    """
    val_img_path = getattr(config, "val_image_path", None) or getattr(config, "val_img_path", None)
    val_lbl_path = getattr(config, "val_label_path", None)
    if val_img_path and val_lbl_path:
        val_dl = Data_Loader(
            img_path=val_img_path,
            label_path=val_lbl_path,
            image_size=config.imsize,
            batch_size=config.batch_size,
            mode=False,
        )
        return val_dl.loader()
    return None


def train_once(config):
    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)

    # Optionally attach a validation loader to the config for Trainer
    val_loader = _maybe_build_val_loader(config)
    if val_loader is not None:
        setattr(config, "val_loader", val_loader)

    # Data loader & trainer
    data_loader = Data_Loader(config.img_path, config.label_path, config.imsize,
                              config.batch_size, config.train)
    trainer = Trainer(data_loader.loader(), config)
    trainer.train()


def test_once(config):
    tester = Tester(config)
    tester.test()


def main(config):
    # For fast training
    cudnn.benchmark = True

    # ---------------------- SWEEP MODE ----------------------
    if config.sweep:
        # Build search space (fallback to single current value if list is empty)
        lrs = _parse_csv_list(config.sweep_g_lr, float) or [config.g_lr]
        bss = _parse_csv_list(config.sweep_batch_size, int) or [config.batch_size]
        b1s = _parse_csv_list(config.sweep_beta1, float) or [config.beta1]
        decs = _parse_csv_list(config.sweep_lr_decay, float) or [config.lr_decay]
        seeds = _parse_csv_list(config.sweep_seed, int) or [config.seed]

        combos = list(itertools.product(lrs, bss, b1s, decs, seeds))
        os.makedirs("sweep_preds", exist_ok=True)
        results_csv = "sweep_results.csv"
        write_header = not os.path.exists(results_csv)

        best = (-1.0, None)  # (mF1, version)

        with open(results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["trial", "g_lr", "batch_size", "beta1", "lr_decay", "seed",
                            "mF1", "pixel_acc", "model_dir", "pred_dir"])

            for i, (lr, bs, b1, dec, sd) in enumerate(combos):
                trial_version = f"{config.version}_t{i:03d}_lr{lr}_bs{bs}_b1{b1}_dec{dec}_s{sd}"

                # ---- TRAIN ----
                c_train = copy.deepcopy(config)
                c_train.train = True
                c_train.g_lr = float(lr)
                c_train.batch_size = int(bs)
                c_train.beta1 = float(b1)
                c_train.lr_decay = float(dec)
                c_train.seed = int(sd)
                c_train.version = trial_version

                # alias both names (helps any downstream that expects either)
                if hasattr(config, "val_image_path"):
                    c_train.val_img_path = config.val_image_path  # alias for Trainer's internal checks
                # attach a val loader so Trainer can do periodic validation/best checkpoint
                val_loader = _maybe_build_val_loader(c_train)
                if val_loader is not None:
                    setattr(c_train, "val_loader", val_loader)

                set_seed(c_train.seed)
                train_once(c_train)

                # ---- TEST on VALIDATION ----
                # Reuse Tester but point it at val images; save plain preds under sweep_preds
                plain_dir = os.path.join("sweep_preds", trial_version, "plain")
                color_dir = os.path.join("sweep_preds", trial_version, "color")
                os.makedirs(plain_dir, exist_ok=True)
                os.makedirs(color_dir, exist_ok=True)

                c_test = copy.deepcopy(c_train)
                c_test.train = False
                c_test.test_image_path = getattr(config, "val_image_path", None) or getattr(config, "val_img_path", None)
                c_test.test_label_path = plain_dir
                c_test.test_color_label_path = color_dir
                # Ensure it loads the rolling/best checkpoint we saved
                c_test.model_name = "best_G.pth"  # prefer best if Trainer saved it
                test_once(c_test)

                # ---- EVAL mean F1 on validation ----
                mF1, pix = eval_dir_mean_f1(plain_dir, config.val_label_path)
                print(f"[{trial_version}] mF1={mF1:.4f}  pixel_acc={pix:.4f}")

                w.writerow([trial_version, lr, bs, b1, dec, sd, f"{mF1:.6f}", f"{pix:.6f}",
                            os.path.join(config.model_save_path, trial_version), plain_dir])
                f.flush()

                if mF1 > best[0]:
                    best = (mF1, trial_version)

        print(f"\nBest trial: {best[1]}  (mF1={best[0]:.4f})")
        return
    # -------------------- END SWEEP MODE --------------------

    # -------- normal single run --------
    # Set seed for reproducibility if present
    if hasattr(config, "seed"):
        set_seed(config.seed)

    # Provide alias so Trainer can detect either name
    if hasattr(config, "val_image_path"):
        setattr(config, "val_img_path", config.val_image_path)

    if config.train:
        # Attach a val loader (if paths available) so Trainer can validate & save best_G.pth
        val_loader = _maybe_build_val_loader(config)
        if val_loader is not None:
            setattr(config, "val_loader", val_loader)
        train_once(config)
    else:
        test_once(config)


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
