
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=max(1, out_ch // 8), num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        return self.act(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = DSConv(in_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DWUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=19, base=16):
        super().__init__()
        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base*2)
        self.e3 = EncoderBlock(base*2, base*4)
        self.e4 = EncoderBlock(base*4, base*8)

        self.b1 = DSConv(base*8, base*16)
        self.b2 = DSConv(base*16, base*8)

        self.d1 = DecoderBlock(base*8 + base*8, base*4)
        self.d2 = DecoderBlock(base*4 + base*4, base*2)
        self.d3 = DecoderBlock(base*2 + base*2, base)
        self.d4 = DecoderBlock(base + base, base)

        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.d1(x, s4)
        x = self.d2(x, s3)
        x = self.d3(x, s2)
        x = self.d4(x, s1)
        return self.head(x)

if __name__ == "__main__":
    model = DWUNet(in_ch=3, num_classes=19, base=16)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)
    x = torch.randn(1,3,512,512)
    y = model(x)
    print("Output shape:", y.shape)
