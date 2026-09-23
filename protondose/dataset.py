"""Cached beamlets -> training batches.

prepare_data.py writes one file per beamlet, <cache>/<patient>/<B..._R.._L.>.npz:

    density_bev, dose_bev   (361, 33, 33) float16   CT density and reference dose on the box
    coords, values          (P, 3), (P,)  float32   supervision pools (pools.py)
    band, block             (P,)          int8, int32
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .channels import ChannelBuilder
from .config import NORM_SCALE, POOL_PAD
from .pools import PAD_BAND


def cache_file(cache_dir, beamlet) -> Path:
    return Path(cache_dir) / beamlet.patient / f"{beamlet.name}.npz"


class BeamletDataset(torch.utils.data.Dataset):
    """input (5, Nw, Nv, Nu), target (1, Nw, Nv, Nu) = dose / NORM_SCALE, and with
    pools=True the supervision pools padded to POOL_PAD rows."""

    def __init__(self, cache_dir, beamlets, builder: ChannelBuilder, pools: bool):
        self.cache_dir = cache_dir
        self.builder = builder
        self.pools = pools
        self.items = [b for b in beamlets if cache_file(cache_dir, b).is_file()]
        if len(self.items) < len(beamlets):
            print(f"[dataset] {len(beamlets) - len(self.items)} of {len(beamlets)} "
                  f"beamlets have no cache file and are skipped", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        b = self.items[i]
        z = np.load(cache_file(self.cache_dir, b))
        x = self.builder.input(z["density_bev"].astype(np.float32), b.energy)
        target = z["dose_bev"].astype(np.float32) / NORM_SCALE
        out = {"input": torch.from_numpy(x), "target": torch.from_numpy(target)[None]}
        if self.pools:
            n = z["values"].shape[0]
            if n > POOL_PAD:
                raise RuntimeError(f"{b.name}: pool has {n} rows > POOL_PAD {POOL_PAD}")
            pad = POOL_PAD - n
            out["coords"] = torch.from_numpy(np.pad(z["coords"], ((0, pad), (0, 0))))
            out["values"] = torch.from_numpy(np.pad(z["values"], (0, pad)))
            out["band"] = torch.from_numpy(np.pad(z["band"], (0, pad),
                                                  constant_values=PAD_BAND))
            out["block"] = torch.from_numpy(np.pad(z["block"].astype(np.int64), (0, pad),
                                                   constant_values=-1))
        return out
