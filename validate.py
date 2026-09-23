#!/usr/bin/env python3
"""Score the model on the validation patients against the reference Monte Carlo dose.

    python validate.py --data data/DoseRAD2026/proton/training

Runs the full inference path, CT -> dose on the patient grid, so no cache is
needed. From each validation patient it takes 40 beamlets spread over the
energy range (440 over the 11 validation patients) and reports the challenge's
two beam-level metrics, masked beam MAE and IDD distance, after the reference
cutoff (config.GT_CUTOFF_REL) is applied to both volumes.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import SimpleITK as sitk
import torch

from protondose import data
from protondose.channels import ChannelBuilder
from protondose.geometry import beamlet_frame
from protondose.inference import load_model, predict, to_full
from protondose.metrics import score


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory holding the patient folders")
    p.add_argument("--weights", default="weights/model.pt")
    p.add_argument("--split", default="splits/train_val.json")
    p.add_argument("--per-patient", type=int, default=40,
                   help="beamlets per validation patient; 0 = all of them")
    p.add_argument("--csv", help="write per-beamlet results here")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    _, val_pids = data.load_split(args.split, args.data)
    if not val_pids:
        raise SystemExit(f"{args.split}: no validation patient in {args.data}")
    net = load_model(args.weights)
    builder = ChannelBuilder(data.beam_parameters_path(args.data))

    rows = []
    for pid in val_pids:
        beamlets = data.patient_beamlets(args.data, pid)
        if args.per_patient:
            beamlets = data.energy_spread(beamlets, args.per_patient)
        ct = sitk.ReadImage(str(data.ct_path(args.data, pid)))
        spacing = np.asarray(ct.GetSpacing(), np.float64)
        n0 = len(rows)
        for b, dose, bbox in predict(net, builder, ct, beamlets):
            pred = to_full(dose, bbox, ct).astype(np.float64)
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(data.dose_path(args.data, b))))
            mae, idd = score(pred, gt.astype(np.float64),
                             beamlet_frame(b.ray_source, b.ray_target).w_hat, spacing)
            rows.append((pid, b.dose_file, b.energy, mae, idd))
        pm = np.array([r[3] for r in rows[n0:]])
        pi = np.array([r[4] for r in rows[n0:]])
        print(f"{pid}  n={len(pm):4d}  masked MAE {np.nanmean(pm):.5f}  "
              f"IDD {np.nanmean(pi):.5f}", flush=True)

    mae = np.array([r[3] for r in rows])
    idd = np.array([r[4] for r in rows])
    print(f"\n{len(rows)} beamlets, {len(val_pids)} patients")
    print(f"  masked beam MAE      {np.nanmean(mae):.5f} +/- {np.nanstd(mae):.5f}")
    print(f"  IDD curve distance   {np.nanmean(idd):.5f} +/- {np.nanstd(idd):.5f}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["patient", "dose_file", "energy_mev", "masked_mae", "idd_distance"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
