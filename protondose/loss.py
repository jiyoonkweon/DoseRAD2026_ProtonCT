"""Training loss.

    L =   sum_b  w_b * mean((pred - ref)^2 over band b) / s_b      patient-space points
        + w_bev(epoch) * mean((pred - ref)^2 over the BEV box) / s_bev
        + COLD_WEIGHT * mean((pred - ref)^2 where ref < COLD_THRESHOLD * max) / s_bev
        + w_idd(epoch) * mean(((P_pred - P_ref) / max P_ref)^2)

pred is the network output on the BEV box. The band terms sample it at the
patient-space points of pools.py; w_b and the fixed normalizers s_b are in
config.BANDS / BAND_SCALE, and s_bev is the mean squared BEV reference of the
batch. The cold term anchors the region below 0.1 % of the maximum, which no
band covers. P is the integrated depth-dose profile, the sum over each depth
slice of the box; the box is a parallel grid, so every voxel has the same area
and no divergence correction applies. The last two weights change per epoch
(see train.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import BAND_SCALE, BANDS, COLD_THRESHOLD, COLD_WEIGHT


def sample_bev(bev, coords):
    """bev (B, 1, Nw, Nv, Nu) at normalized coords (B, N, 3) -> (B, N)."""
    b, n, _ = coords.shape
    out = F.grid_sample(bev, coords.view(b, 1, 1, n, 3).to(bev.dtype),
                        mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[:, 0, 0, 0, :]


def block_means(pred, target, block):
    """Means of pred and target within each block id."""
    _, block = torch.unique(block, return_inverse=True)
    n = int(block.max()) + 1
    cnt = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add_(
        0, block, torch.ones_like(pred))
    ps = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add_(0, block, pred)
    ts = torch.zeros(n, device=target.device, dtype=target.dtype).index_add_(0, block, target)
    keep = cnt > 0
    return ps[keep] / cnt[keep], ts[keep] / cnt[keep]


class PatientSpaceLoss:
    def __init__(self):
        self.bev_weight = 0.0        # set per epoch
        self.idd_weight = 0.0        # set per epoch

    def __call__(self, pred, coords, values, band, block, target):
        """pred, target (B, 1, Nw, Nv, Nu); coords (B, N, 3); values, band, block (B, N)."""
        parts = {}
        total = pred.new_zeros(())

        # Patient-space points, one mean per band across the whole batch.
        b, n = values.shape
        p = sample_bev(pred, coords).reshape(-1).float()
        v = values.reshape(-1).float()
        bd = band.reshape(-1).to(torch.int64)
        # Block ids are per sample; offset them so blocks of different
        # beamlets are never averaged together.
        off = torch.arange(b, device=block.device).view(b, 1) * (n + 1)
        bk = torch.where(block >= 0, block + off, block).reshape(-1)
        for bi, (name, _, _, pooled, weight, _, _) in enumerate(BANDS):
            m = bd == bi
            if not bool(m.any()):
                continue
            pp, tt = p[m], v[m]
            if pooled:
                pp, tt = block_means(pp, tt, bk[m])
            e = ((pp - tt) ** 2).mean() / BAND_SCALE[bi]
            total = total + weight * e
            parts[name] = float(e.detach())

        pb, tb = pred[:, 0].float(), target[:, 0].float()
        s_bev = tb.detach().pow(2).mean().clamp_min(1e-8)

        # BEV stabilizer: dense gradient early in training, where points are sparse.
        if self.bev_weight > 0:
            e = ((pb - tb) ** 2).mean() / s_bev
            total = total + self.bev_weight * e
            parts["bev"] = float(e.detach())

        # Cold anchor.
        bmax = tb.detach().flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        cold = tb.detach() < COLD_THRESHOLD * bmax
        if bool(cold.any()):
            e = ((pb[cold] - tb[cold]) ** 2).mean() / s_bev
            total = total + COLD_WEIGHT * e
            parts["cold"] = float(e.detach())

        # Integrated depth-dose alignment.
        if self.idd_weight > 0:
            prof_p = pb.sum(dim=(2, 3))
            prof_t = tb.sum(dim=(2, 3))
            den = prof_t.detach().abs().amax(1, keepdim=True).clamp_min(1e-12)
            e = (((prof_p - prof_t) / den) ** 2).mean()
            total = total + self.idd_weight * e
            parts["idd"] = float(e.detach())

        parts["total"] = float(total.detach())
        return total, parts
