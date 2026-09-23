"""Patient-space supervision points.

The loss is computed on the patient grid, where the challenge scores, rather
than on the BEV box: regressing a BEV-resampled reference would make the
BEV -> patient round-trip error a floor under the achievable score.

Per beamlet, reference-dose voxels are split into dose bands (config.BANDS) and
up to a pool size of each is stored, with its normalized BEV coordinate. Each
training step draws a fresh subset from the pool. In the pooled "low" band,
where voxel values are dominated by Monte Carlo noise, whole 2x2x2 blocks are
drawn and compared as block means.
"""
from __future__ import annotations

import torch

from .config import BANDS, NORM_SCALE, POOL_BLOCK_VOXELS

PAD_BAND = -1          # band of padding rows; matches no band


def build_pools(gt, grid):
    """Supervision pools of one beamlet.

    gt   : (z, y, x) reference dose on the GPU, Gy
    grid : (1, nx, ny, nz, 3) normalized BEV coordinates of every patient voxel
    returns dict of coords (P, 3), values (P,) = dose / NORM_SCALE, band (P,)
            and block (P,), the block id in pooled bands and -1 elsewhere
    """
    g = grid[0].permute(2, 1, 0, 3)                     # -> (z, y, x, 3)
    rel = gt / gt.max()
    coords, values, bands, blocks = [], [], [], []
    block_offset = 0
    for bi, (_, lo, hi, pooled, _, pool_size, _) in enumerate(BANDS):
        idx = ((rel >= lo) & (rel < hi)).nonzero(as_tuple=False)      # (N, 3) as (z, y, x)
        if idx.shape[0] == 0:
            continue
        if idx.shape[0] > pool_size:
            idx = idx[torch.randperm(idx.shape[0], device=idx.device)[:pool_size]]
        z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
        coords.append(g[z, y, x].float())
        values.append((gt[z, y, x] / NORM_SCALE).float())
        bands.append(torch.full((idx.shape[0],), bi, dtype=torch.int8, device=gt.device))
        if pooled:
            q = torch.div(idx, POOL_BLOCK_VOXELS, rounding_mode="floor")
            key = (q[:, 0] * 100003 + q[:, 1]) * 100003 + q[:, 2]
            _, inv = torch.unique(key, return_inverse=True)
            blocks.append(inv + block_offset)
            block_offset += int(inv.max()) + 1
        else:
            blocks.append(torch.full((idx.shape[0],), -1, dtype=torch.int64, device=gt.device))
    return dict(coords=torch.cat(coords), values=torch.cat(values),
                band=torch.cat(bands), block=torch.cat(blocks))


def _draw_one(coords, values, band, block):
    parts = []
    for bi, (_, _, _, pooled, _, _, n) in enumerate(BANDS):
        sel = (band == bi).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        if pooled:
            # Draw whole blocks, about n points' worth, so block means stay means.
            blk = block[sel]
            uniq = torch.unique(blk)
            per_block = max(int(blk.numel() / max(uniq.numel(), 1)), 1)
            want = max(n // per_block, 1)
            if uniq.numel() > want:
                keep = uniq[torch.randperm(uniq.numel(), device=uniq.device)[:want]]
                sel = sel[torch.isin(blk, keep)]
        elif sel.numel() > n:
            sel = sel[torch.randperm(sel.numel(), device=sel.device)[:n]]
        parts.append(sel)
    sel = torch.cat(parts)
    return coords[sel], values[sel], band[sel], block[sel]


def draw_points(coords, values, band, block):
    """Per-step subset of batched, padded pools (B, P, ...) -> (B, N, ...).

    N is the sum of the per-band draw sizes; rows that cannot be filled are
    padding with band PAD_BAND.
    """
    b = coords.shape[0]
    n = sum(bd[6] for bd in BANDS)
    oc = coords.new_zeros((b, n, 3))
    ov = values.new_zeros((b, n))
    ob = torch.full((b, n), PAD_BAND, dtype=torch.int8, device=coords.device)
    ok = torch.full((b, n), -1, dtype=torch.int64, device=coords.device)
    for i in range(b):
        c, v, bd, bk = _draw_one(coords[i], values[i], band[i], block[i])
        k = min(int(c.shape[0]), n)
        oc[i, :k], ov[i, :k], ob[i, :k], ok[i, :k] = c[:k], v[:k], bd[:k], bk[:k]
    return oc, ov, ob, ok
