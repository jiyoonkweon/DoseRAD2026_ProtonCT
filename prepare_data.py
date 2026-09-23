#!/usr/bin/env python3
"""Build the training cache: per beamlet, the BEV density and reference dose,
and the patient-space supervision pools.

    python prepare_data.py --data data/DoseRAD2026/proton/training --cache cache

Runs on one GPU; to use several, start one process per GPU with --shard i
--num-shards n. Beamlets already in the cache are skipped, so an interrupted
run can simply be restarted.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from protondose import data
from protondose.channels import ChannelBuilder
from protondose.dataset import cache_file
from protondose.geometry import (beamlet_frame, make_bev_grid, make_patient_grid,
                                 patient_phys_coords, resample_to_bev)
from protondose.pools import build_pools


def save(path: Path, arrays: dict):
    """Write through a temporary file, so an interrupted run leaves no partial file."""
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory holding the patient folders")
    p.add_argument("--cache", required=True, help="output directory")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    dev = "cuda"
    builder = ChannelBuilder(data.beam_parameters_path(args.data))
    patients = data.list_patients(args.data)[args.shard::args.num_shards]
    read_pool = ThreadPoolExecutor(max_workers=8)
    save_pool = ProcessPoolExecutor(max_workers=8)
    pending = []

    def read_dose(b):
        return b, sitk.GetArrayFromImage(sitk.ReadImage(str(data.dose_path(args.data, b))))

    for pid in tqdm(patients, desc="patients"):
        todo = [b for b in data.patient_beamlets(args.data, pid)
                if not cache_file(args.cache, b).is_file()]
        if not todo:
            continue
        (Path(args.cache) / pid).mkdir(parents=True, exist_ok=True)
        ct = sitk.ReadImage(str(data.ct_path(args.data, pid)))
        hu = sitk.GetArrayFromImage(ct).astype(np.float32)
        density = torch.as_tensor(builder.density(hu), device=dev)
        phys = patient_phys_coords(ct, dev)

        ray_key = None
        for b, gt in read_pool.map(read_dose, todo):
            if (b.ray_source, b.ray_target) != ray_key:     # both beamlets of a ray share these
                ray_key = (b.ray_source, b.ray_target)
                frame = beamlet_frame(b.ray_source, b.ray_target)
                bev_grid = make_bev_grid(frame, ct, dev)
                density_bev = resample_to_bev(density, bev_grid).cpu().numpy()
                patient_grid = None                         # free before rebuilding
                patient_grid = make_patient_grid(frame, phys)
            gt = torch.as_tensor(gt.astype(np.float32), device=dev)
            if float(gt.max()) <= 0:
                print(f"[skip] {pid} {b.name}: empty reference dose", flush=True)
                continue
            dose_bev = resample_to_bev(gt, bev_grid).cpu().numpy()
            pools = {k: v.cpu().numpy() for k, v in build_pools(gt, patient_grid).items()}
            arrays = dict(density_bev=density_bev.astype(np.float16),
                          dose_bev=dose_bev.astype(np.float16),
                          coords=pools["coords"].astype(np.float32),
                          values=pools["values"].astype(np.float32),
                          band=pools["band"].astype(np.int8),
                          block=pools["block"].astype(np.int32))
            pending.append(save_pool.submit(save, cache_file(args.cache, b), arrays))
            if len(pending) > 64:
                keep = []
                for f in pending:
                    if f.done():
                        f.result()                          # re-raises a worker error
                    else:
                        keep.append(f)
                pending = keep
        del density, phys, patient_grid
        torch.cuda.empty_cache()

    for f in pending:
        f.result()
    save_pool.shutdown()
    read_pool.shutdown()


if __name__ == "__main__":
    main()
