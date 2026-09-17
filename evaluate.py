#!/usr/bin/env python3
"""Score predicted beamlet dose against the reference Monte Carlo dose.

    python evaluate.py --case /path/to/DoseRAD2026/proton/training/1THB016

Reports the two beam-level metrics the challenge scores, masked beam MAE and
IDD curve distance, over beamlets taken from the case's own plan file. The
organizers' implementation is authoritative:
https://github.com/DoseRAD2026/evaluation-setup

--runtime adds the other half of the challenge score, measured as the
leaderboard computes it. It runs a second pass over the plan because timing
wants a different sample of beamlets than scoring does; protondose/timing.py
says why.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from protondose import inference, timing
from protondose.geometry import patient_phys_coords


def masked_beam_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """MAE over voxels above 10 % of the beamlet maximum, relative to it."""
    gt_max = float(gt.max())
    if gt_max <= 0:
        return float("nan")
    mask = gt >= 0.1 * gt_max
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - gt[mask])) / gt_max)


def idd_curve(dose: np.ndarray, direction: np.ndarray,
              spacing: tuple[float, ...]) -> np.ndarray:
    """Dose integrated over planes perpendicular to the beam, against depth."""
    if abs(direction[2]) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")

    plane = dose.sum(axis=0, dtype=np.float64)
    ny, nx = plane.shape
    sx, sy = float(spacing[0]), float(spacing[1])

    # square grid, one sample per voxel, wide enough to hold the diagonal so
    # that the curve length does not depend on the beam angle
    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1

    source = sitk.GetImageFromArray(plane)
    source.SetSpacing((sx, sy))

    # Resample maps output points back to the input, so rotating by the beam
    # angle about the plane centre lays the beam along the first output axis.
    to_beam = sitk.Euler2DTransform()
    to_beam.SetCenter((0.0, 0.0))
    to_beam.SetAngle(math.atan2(direction[1], direction[0]))
    to_beam.SetTranslation(((nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0))
    aligned = sitk.Resample(source, (n, n), to_beam, sitk.sitkLinear,
                            (-(n - 1) * step / 2.0,) * 2, (step, step),
                            (1.0, 0.0, 0.0, 1.0), 0.0, sitk.sitkFloat64)
    return sitk.GetArrayFromImage(aligned).sum(axis=0)


def idd_distance(pred: np.ndarray, gt: np.ndarray, direction: np.ndarray,
                 spacing: tuple[float, ...]) -> float:
    """RMS difference of the two IDD curves, relative to the reference peak."""
    reference = idd_curve(gt, direction, spacing)
    peak = float(reference.max())
    if peak <= 0:
        return float("nan")
    predicted = idd_curve(pred, direction, spacing)
    return float(np.sqrt(np.mean(((predicted - reference) / peak) ** 2)))


def plan_beamlets(case: Path) -> list[dict]:
    plan_file = case / f"{case.name}.json"
    if not plan_file.is_file():
        raise SystemExit(f"no plan file at {plan_file}; --case wants a patient "
                         "directory from the training set")
    plan = json.loads(plan_file.read_text())
    items = []
    for beam in plan["beams"]:
        for ray in beam["rays"]:
            source = np.asarray(ray["ray_source"], float)
            target = np.asarray(ray["ray_target"], float)
            direction = (target - source) / np.linalg.norm(target - source)
            for bl in ray["beamlets"]:
                items.append({
                    "dose_file": (f"Dose_B{beam['beam_idx']}"
                                  f"_R{ray['ray_idx']}_L{bl['beamlet_idx']}.mha"),
                    "energy": bl["energy"],
                    "direction": direction,
                    "beamlet": inference.Beamlet(ray["ray_source"],
                                                 ray["ray_target"], bl["energy"]),
                })
    return items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, type=Path,
                   help="patient directory holding image/, dose/ and the plan json")
    p.add_argument("--limit", type=int, default=8,
                   help="beamlets to score, spread evenly through the plan (0 = all)")
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    p.add_argument("--runtime", action="store_true",
                   help="also measure inference runtime the way the leaderboard "
                        "computes it, in a second pass over the plan")
    p.add_argument("--runtime-beamlets", type=int, default=0,
                   help="beamlets for that pass, as whole rays drawn evenly across "
                        "the plan; 0, the default, runs the whole plan")
    p.add_argument("--csv", type=Path, help="write the per-beamlet table here")
    args = p.parse_args()

    ct_file = args.case / "image" / "ct.mha"
    if not ct_file.is_file():
        raise SystemExit(f"no CT at {ct_file}")

    items = plan_beamlets(args.case)
    if args.limit and args.limit < len(items):
        picks = np.linspace(0, len(items) - 1, args.limit).round().astype(int)
        items = [items[i] for i in dict.fromkeys(picks.tolist())]

    def _load():
        m = inference.build_model(Path(args.model_dir), device="cuda")
        inference.warmup(m)
        return m
    model, t_startup = timing._timed(_load)

    image = sitk.ReadImage(str(ct_file))
    spacing = image.GetSpacing()
    density = inference.make_density(image, model["hlut"], "cuda")
    phys = patient_phys_coords(image, device="cuda")

    nx, ny, nz = image.GetSize()
    rows = []
    gen = inference.predict_beamlets(model, [it["beamlet"] for it in items],
                                     image, density, phys)
    print(f"{'beamlet':<26}{'E (MeV)':>9}{'MAE':>10}{'IDD':>10}")
    for item, (sub, bbox) in zip(items, gen):
        pred = np.zeros((nz, ny, nx), np.float64)
        x0, x1, y0, y1, z0, z1 = bbox
        pred[z0:z1, y0:y1, x0:x1] = np.array(sub, dtype=np.float32) * inference.NORM_SCALE

        gt_path = args.case / "dose" / item["dose_file"]
        if not gt_path.is_file():
            raise SystemExit(f"no reference dose at {gt_path}")
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.float64)
        if gt.shape != pred.shape:
            raise SystemExit(f"{gt_path.name}: shape {gt.shape} does not match the CT")

        mae = masked_beam_mae(pred, gt)
        idd = idd_distance(pred, gt, item["direction"], spacing)
        rows.append((item["dose_file"], item["energy"], mae, idd))
        print(f"{item['dose_file']:<26}{item['energy']:>9.2f}{mae:>10.4f}{idd:>10.4f}",
              flush=True)

    mae = np.array([r[2] for r in rows])
    idd = np.array([r[3] for r in rows])
    print(f"\n{len(rows)} beamlets of {args.case.name}")
    print(f"  masked beam MAE      {np.nanmean(mae):.4f} +/- {np.nanstd(mae):.4f}")
    print(f"  IDD curve distance   {np.nanmean(idd):.4f} +/- {np.nanstd(idd):.4f}")

    if args.runtime:
        rt = timing.measure_runtime(args.case, Path(args.model_dir),
                                    n=args.runtime_beamlets, model=model,
                                    t_startup=t_startup)
        print()
        print(rt.report())

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dose_file", "energy_mev", "beam_mae", "idd_distance"])
            w.writerows(rows)
        print(f"  wrote {args.csv}")


if __name__ == "__main__":
    main()
