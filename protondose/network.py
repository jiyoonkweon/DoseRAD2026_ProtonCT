"""3D residual U-Net with a depth-sequential ConvLSTM bottleneck (6,453,261 parameters).

    input  (B, 5, 361, 33, 33)   channels on the BEV box
    stem   (B, 24, 361, 33, 33)
    enc 1  (B, 48, 180, 16, 16)  stride (2, 2, 2)
    enc 2  (B, 96, 90, 8, 8)     stride (2, 2, 2)
    enc 3  (B, 192, 45, 8, 8)    stride (2, 1, 1), depth only
    ConvLSTM over the 45 depth slices, upstream to downstream, + dropout
    3 decoder levels: 1x1x1 conv, trilinear resize, concat skip, residual block
    head   (B, 1, 361, 33, 33)   linear; dose / NORM_SCALE

Layer names define the state-dict keys of the released weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEPTH, DROPOUT, IN_CHANNELS, STRIDES, WIDTH


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c), c)


class SE3d(nn.Module):
    """Squeeze-and-excitation channel gating."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        h = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(inplace=True),
                                nn.Linear(h, ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(dim=(2, 3, 4)))[:, :, None, None, None]


class ResBlock3d(nn.Module):
    """Two 3x3x3 convolutions with GroupNorm, SE gating, residual add."""
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.SiLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), _gn(cout),
        )
        self.se = SE3d(cout)
        self.skip = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.se(self.body(x)) + self.skip(x))


class ConvLSTM3dCore(nn.Module):
    """Unidirectional ConvLSTM over the depth axis, applied residually.

    The volume is read as a sequence of lateral slices, so the features at any
    depth are conditioned on everything upstream of it.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.hidden = ch
        self.conv = nn.Conv2d(ch + ch, 4 * ch, 3, padding=1)    # i, f, o, g gates
        self.proj = nn.Conv2d(ch, ch, 1)
        self.gn = _gn(ch)

    def forward(self, x):
        b, _, d, hgt, wid = x.shape
        h = x.new_zeros(b, self.hidden, hgt, wid)
        c = torch.zeros_like(h)
        outs = []
        for t in range(d):
            i, f, o, g = torch.chunk(self.conv(torch.cat([x[:, :, t], h], dim=1)), 4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs.append(self.proj(h))
        return self.gn(x + torch.stack(outs, dim=2))


class Dose3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        ws = [WIDTH * 2 ** i for i in range(DEPTH + 1)]         # 24, 48, 96, 192
        self.stem = ResBlock3d(IN_CHANNELS, ws[0])
        self.downs, self.enc = nn.ModuleList(), nn.ModuleList()
        for i, s in enumerate(STRIDES):
            self.downs.append(nn.Conv3d(ws[i], ws[i + 1], kernel_size=s, stride=s, bias=False))
            self.enc.append(ResBlock3d(ws[i + 1], ws[i + 1]))
        self.bottleneck = ConvLSTM3dCore(ws[-1])
        self.drop = nn.Dropout3d(DROPOUT)
        self.reduce, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(DEPTH)):
            self.reduce.append(nn.Conv3d(ws[i + 1], ws[i], 1, bias=False))
            self.dec.append(ResBlock3d(ws[i] * 2, ws[i]))
        self.head = nn.Conv3d(ws[0], 1, 1)

    def forward(self, x):
        h = self.stem(x)
        skips = [h]
        for down, enc in zip(self.downs, self.enc):
            h = enc(down(h))
            skips.append(h)
        h = self.drop(self.bottleneck(h))
        for i, (red, dec) in enumerate(zip(self.reduce, self.dec)):
            skip = skips[-(i + 2)]
            # Resize to the skip tensor: the depth axis is odd, so no integer
            # upsampling factor is exact.
            h = F.interpolate(red(h), size=skip.shape[2:], mode="trilinear",
                              align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))
        return self.head(h)
