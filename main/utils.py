import os
import torch
import numpy as np
import torch.nn.functional as F

# ---------- filesystem ----------
def make_folder(path, version: str | None = None):
    """Create path (and optional subfolder 'version') if missing."""
    target = os.path.join(path, version) if (version not in (None, "")) else path
    os.makedirs(target, exist_ok=True)


# ---------- tensor / image helpers ----------
def tensor2im(x, imtype=np.uint8):
    """
    Convert a torch Tensor or numpy array to an image (H,W,3) or (H,W) uint8.
    - If tensor is 4D, takes the first item.
    - If 3xHxW, permutes to HxWx3.
    - If 1xHxW or HxW, squeezes to HxW.
    - Clips floats to [0,1] before scaling to 0..255.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dim() == 4:
            x = x[0]
        if x.dim() == 3 and x.size(0) in (1, 3):
            # CxHxW -> HxWxC or HxW
            if x.size(0) == 3:
                x = x.permute(1, 2, 0)
            else:
                x = x.squeeze(0)
        elif x.dim() == 2:
            pass
        else:
            # Fallback: best effort to last two dims as HxW
            x = x.squeeze()
        x = x.numpy()

    # now numpy
    x = np.asarray(x)
    if x.dtype.kind == "f":
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).round().astype(imtype)
    elif x.dtype != imtype:
        x = x.astype(imtype)
    return x


def denorm(x):
    """Map images normalized to [-1,1] back to [0,1] (tensor op)."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


# ---------- color maps ----------
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 19:  # CelebAMask-HQ
        cmap = np.array(
            [
                (0, 0, 0), (204, 0, 0), (76, 153, 0), (204, 204, 0),
                (51, 51, 255), (204, 0, 204), (0, 255, 255), (51, 255, 255),
                (102, 51, 0), (255, 0, 0), (102, 204, 0), (255, 255, 0),
                (0, 0, 153), (0, 0, 204), (255, 51, 153), (0, 204, 204),
                (0, 51, 0), (255, 153, 51), (0, 204, 0),
            ],
            dtype=np.uint8,
        )
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            _id = i
            for j in range(7):
                str_id = uint82bin(_id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                _id = _id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=19):
        # cmap: [n, 3] uint8
        self.cmap = torch.from_numpy(labelcolormap(n)[:n])

    def __call__(self, gray_image: torch.Tensor) -> torch.Tensor:
        """
        gray_image: Tensor [1,H,W] of class ids (already on CPU in our pipeline)
        returns: ByteTensor [3,H,W] with RGB colors (uint8)
        """
        c, h, w = gray_image.size()
        # create output on CPU, uint8
        color_image = torch.zeros(3, h, w, dtype=torch.uint8)

        # explicit tensor comparison avoids Pylance 'bool.cpu' warning
        for lbl in range(len(self.cmap)):
            mask = torch.eq(gray_image[0], int(lbl))   # BoolTensor on CPU
            # assign per channel
            color_image[0][mask] = self.cmap[lbl][0]
            color_image[1][mask] = self.cmap[lbl][1]
            color_image[2][mask] = self.cmap[lbl][2]

        return color_image



def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        # Fall back to plain tensor->image conversion
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.detach().cpu().float()
    if label_tensor.size()[0] > 1:  # if one-hot or logits, take argmax
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)  # [3,H,W], uint8
    label_numpy = label_tensor.numpy() / 255.0      # to [0,1] for saving
    return label_numpy


# ---------- label renderers ----------
def generate_label(inputs, imsize):
    """
    inputs: Tensor [B,C,H,W] (logits or one-hot)
    returns: Tensor [B,3,H,W] float in [0,1] with colorized predictions
    """
    preds = []
    with torch.no_grad():
        for x in inputs:
            x = x.view(1, 19, imsize, imsize)
            pred = torch.argmax(x, dim=1).squeeze(0).cpu().numpy()  # HxW
            preds.append(pred)

    preds = np.stack(preds, axis=0)  # [B,H,W]
    # colorize each
    colored = []
    for p in preds:
        p_t = torch.from_numpy(p).view(1, imsize, imsize)  # [1,H,W]
        colored.append(tensor2label(p_t, 19))              # np [3,H,W] in [0,1]
    colored = np.stack(colored, axis=0)                    # [B,3,H,W]
    return torch.from_numpy(colored)                       # Tensor float [0,1]


def generate_label_plain(inputs, imsize):
    """
    inputs: Tensor [B,C,H,W] (logits or one-hot)
    returns: np.ndarray [B,H,W] uint8 of class indices
    """
    preds = []
    with torch.no_grad():
        for x in inputs:
            x = x.view(1, 19, imsize, imsize)
            pred = torch.argmax(x, dim=1).squeeze(0).cpu().numpy()  # HxW
            preds.append(pred.astype(np.uint8))
    return np.stack(preds, axis=0)  # [B,H,W], uint8


# ---------- loss ----------
def cross_entropy2d(input, target, weight=None, size_average=True):
    """
    input:  [B,C,H,W] logits
    target: [B,H,W]   long
    """
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.permute(0, 2, 3, 1).contiguous().view(-1, c)  # [B*H*W, C]
    target = target.view(-1)                                     # [B*H*W]

    reduction = "mean" if size_average else "sum"
    loss = F.cross_entropy(input, target, weight=weight, reduction=reduction, ignore_index=250)
    return loss
