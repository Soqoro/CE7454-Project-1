# unet.py
from __future__ import annotations
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weight_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        bias = getattr(m, "bias", None)
        if isinstance(bias, torch.Tensor):
            nn.init.zeros_(bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def u3pblock(in_ch: int, out_ch: int, num_block: int = 2, *, down_sample: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    if down_sample:
        layers.append(nn.MaxPool2d(kernel_size=2))
    ch = in_ch
    for _ in range(num_block):
        layers.append(conv_bn_relu(ch, out_ch, 3, 1))
        ch = out_ch
    return nn.Sequential(*layers)


def en2dec_layer(in_ch: int, out_ch: int, scale: int) -> nn.Sequential:
    ops: List[nn.Module] = [nn.Identity()] if scale == 1 else [nn.MaxPool2d(scale, scale, ceil_mode=True)]
    ops.append(u3pblock(in_ch, out_ch, num_block=1))
    return nn.Sequential(*ops)


def dec2dec_layer(in_ch: int, out_ch: int, scale: int, *, fast_up: bool = True) -> nn.Sequential:
    up = [nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)] if scale != 1 else [nn.Identity()]
    core = [u3pblock(in_ch, out_ch, num_block=1)]
    return nn.Sequential(*(core + up if fast_up else up + core))


class FullScaleSkipConnect(nn.Module):
    def __init__(
        self,
        en_channels: Sequence[int],
        en_scales: Sequence[int],
        num_dec: int,
        *,
        skip_ch: int = 12,
        dec_scales: Optional[Sequence[int]] = None,
        bottom_dec_ch: int = 128,
        dropout: float = 0.0,
        fast_up: bool = True,
    ) -> None:
        super().__init__()
        en_channels = list(en_channels)
        en_scales = list(en_scales)

        concat_ch = skip_ch * (len(en_channels) + num_dec)
        self.en2dec_layers = nn.ModuleList([
            en2dec_layer(ch, skip_ch, scale) for ch, scale in zip(en_channels, en_scales)
        ])

        if dec_scales is None:
            dec_scales = [2 ** (i + 1) for i in reversed(range(num_dec))]
        else:
            dec_scales = list(dec_scales)

        self.dec2dec_layers = nn.ModuleList()
        for i, scale in enumerate(dec_scales):
            dec_in = bottom_dec_ch if i == 0 else concat_ch
            self.dec2dec_layers.append(dec2dec_layer(dec_in, skip_ch, scale, fast_up=fast_up))

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.fuse = u3pblock(concat_ch, concat_ch, num_block=1)

    def forward(self, en_maps: List[torch.Tensor], dec_maps: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for em, layer in zip(en_maps, self.en2dec_layers):
            outs.append(layer(em))
        if dec_maps is not None and len(dec_maps) > 0:
            for dm, layer in zip(dec_maps, self.dec2dec_layers):
                outs.append(layer(dm))
        x = torch.cat(outs, dim=1)
        return self.fuse(self.dropout(x))


class U3PEncoderDefault(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        chs = list(channels)
        assert len(chs) >= 2
        blocks: List[nn.Module] = []
        for i, (cin, cout) in enumerate(zip(chs[:-1], chs[1:])):
            blocks.append(u3pblock(cin, cout, num_block=2, down_sample=(i > 0)))
        self.blocks = nn.ModuleList(blocks)
        self.channels = chs
        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs  # [e1, e2, e3, e4, e5]


class U3PDecoder(nn.Module):
    def __init__(self, en_channels: Sequence[int], *, skip_ch: int = 12, dropout: float = 0.0, fast_up: bool = True) -> None:
        super().__init__()
        enc = list(reversed(list(en_channels)))  # high->low
        num_en = len(enc)

        decoders = nn.ModuleDict()
        for i in range(num_en):
            if i == 0:
                decoders["decoder1"] = nn.Identity()
                continue
            decoders[f"decoder{i+1}"] = FullScaleSkipConnect(
                en_channels=enc[i:],
                en_scales=[2 ** k for k in range(0, num_en - i)],
                num_dec=i,
                skip_ch=skip_ch,
                bottom_dec_ch=enc[0],
                dropout=dropout,
                fast_up=fast_up,
            )
        self.decoders = decoders
        self.apply(_weight_init)

    def forward(self, enc_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        enc_rev = list(reversed(enc_maps))
        dec_maps: List[torch.Tensor] = []
        for idx, key in enumerate(self.decoders):
            layer = self.decoders[key]
            if idx == 0:
                dec_maps.append(layer(enc_rev[0]))
            else:
                dec_maps.append(layer(enc_rev[idx:], dec_maps))
        return dec_maps


class UNet3Plus(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 19,
        channels: Sequence[int] = (3, 16, 32, 64, 96, 128),
        skip_ch: int = 12,
        dropout: float = 0.0,
        fast_up: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = U3PEncoderDefault(channels)
        self.decoder = U3PDecoder(self.encoder.channels[1:], skip_ch=skip_ch, dropout=dropout, fast_up=fast_up)

        num_decoders = len(self.encoder.channels) - 1
        decoder_ch = skip_ch * num_decoders
        self.head = nn.Conv2d(decoder_ch, num_classes, kernel_size=3, padding=1)

        self.apply(_weight_init)

    def _resize(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if x.shape[-2] != h or x.shape[-1] != w:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        dec_maps = self.decoder(self.encoder(x))
        logits = self.head(dec_maps[-1])
        return self._resize(logits, h, w)


def unet(num_classes: int = 19) -> nn.Module:
    return UNet3Plus(num_classes=num_classes)
