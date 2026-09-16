"""3D residual U-Net with a depth-sequential ConvLSTM bottleneck.

Maps the 5-channel beam's-eye-view volume to a single-channel dose field on the
same grid.

Layer names define the state-dict keys of the released checkpoint; renaming one
breaks loading.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 5
WIDTH = 24
DEPTH = 3
STRIDES = ((2, 2, 2), (2, 2, 2), (2, 1, 1))
DROPOUT = 0.1


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c), c)


class SE3d(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        h = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(inplace=True),
                                nn.Linear(h, ch), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.mean(dim=(2, 3, 4)))[:, :, None, None, None]
        return x * s


class ResBlock3d(nn.Module):
    """Two 3x3x3 convolutions with GroupNorm, SE recalibration, residual add."""
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


def _lstm_step(conv, x_t, h, c):
    i, f, o, g = torch.chunk(conv(torch.cat([x_t, h], dim=1)), 4, dim=1)
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTM3dCore(nn.Module):
    """Unidirectional ConvLSTM over the depth axis, applied residually.

    The volume is read as a sequence of D lateral slices, so the dose at any
    depth is conditioned on all upstream depths.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.hidden = ch
        self.conv = nn.Conv2d(ch + ch, 4 * ch, 3, padding=1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.gn = _gn(ch)

    def forward(self, x):
        b, _, d, hgt, wid = x.shape
        h = x.new_zeros(b, self.hidden, hgt, wid)
        c = torch.zeros_like(h)
        outs = []
        for t in range(d):
            h, c = _lstm_step(self.conv, x[:, :, t], h, c)
            outs.append(self.proj(h))
        return self.gn(x + torch.stack(outs, dim=2))


class Dose3DNet(nn.Module):
    def __init__(self, in_ch: int = IN_CHANNELS, width: int = WIDTH,
                 depth: int = DEPTH, strides=STRIDES, dropout: float = DROPOUT):
        super().__init__()
        ws = [width * (2 ** i) for i in range(depth + 1)]
        self.depth = depth
        self.strides = tuple(tuple(int(v) for v in s) for s in strides)
        assert len(self.strides) == depth

        self.stem = ResBlock3d(in_ch, ws[0])
        self.downs, self.enc = nn.ModuleList(), nn.ModuleList()
        for i in range(depth):
            s = self.strides[i]
            self.downs.append(nn.Conv3d(ws[i], ws[i + 1], kernel_size=s, stride=s, bias=False))
            self.enc.append(ResBlock3d(ws[i + 1], ws[i + 1]))

        self.bottleneck = ConvLSTM3dCore(ws[-1])
        self.drop = nn.Dropout3d(dropout)

        self.reduce, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(depth)):
            self.reduce.append(nn.Conv3d(ws[i + 1], ws[i], 1, bias=False))
            self.dec.append(ResBlock3d(ws[i] * 2, ws[i]))
        self.head = nn.Conv3d(ws[0], 1, 1)

    def forward(self, x):
        skips = []
        h = self.stem(x)
        skips.append(h)
        for down, enc in zip(self.downs, self.enc):
            h = enc(down(h))
            skips.append(h)
        h = self.drop(self.bottleneck(h))
        for i, (red, dec) in enumerate(zip(self.reduce, self.dec)):
            skip = skips[-(i + 2)]
            h = red(h)
            # Resize to the skip tensor: the depth axis is odd, so no integer
            # upsampling factor is exact.
            h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))
        return self.head(h)
