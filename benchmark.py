#!/usr/bin/env python3
"""Measure inference runtime the way the DoseRAD2026 leaderboard computes it.

The challenge fits a non-negative linear model to the wall time of every job it
runs,

    T = t_fix + N_images * t_img + N_beamlets * t_dose,

and reports it at one image and 500 beamlets. This script measures the two terms
that scale -- t_img and t_dose -- directly on one patient, so the number can be
reproduced on any machine without the submission container or the platform.

Loading the weights and warming up cuDNN are not part of it. The container does
both while answering /health, before the platform starts timing /invoke, so they
belong to no term of the model above. They are reported separately. The fitted
t_fix is what remains of the per-invoke overhead, a few tenths of a second,
which this script does not try to reproduce.

    python benchmark.py --case /path/to/DoseRAD2026/proton/training/1THB016

What is timed
    t_img    reading the CT, converting it to density, building the coordinate
             grid, once per image
    t_dose   steady state per beamlet: channel construction, the network, the
             back-projection to the patient grid, the copy back to the host, and
             the post-processing the challenge requires -- rescaling to absolute
             Gy and zeroing everything at or below the plan's per-beamlet cutoff

Writing the predictions out is not timed. On the platform that cost is
overlapped with prediction and is small (about 0.1 s per output slot), but it is
set by the storage behind the output directory, so including it would measure
the disk rather than the model. Keep the output of any container-based
measurement off a network filesystem for the same reason.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from protondose import inference
from protondose.geometry import patient_phys_coords

LEADERBOARD_BEAMLETS = 500
LEADERBOARD_IMAGES = 1


def plan_rays(case: Path) -> list[list[inference.Beamlet]]:
    """The patient's rays in plan order, each holding its own beamlets."""
    plan_file = case / f"{case.name}.json"
    if not plan_file.is_file():
        raise SystemExit(f"no plan file at {plan_file}; --case wants a patient "
                         "directory from the training set")
    plan = json.loads(plan_file.read_text())
    return [[inference.Beamlet(ray["ray_source"], ray["ray_target"], bl["energy"])
             for bl in ray["beamlets"]]
            for beam in plan["beams"] for ray in beam["rays"]]


def select(rays: list[list[inference.Beamlet]], n: int,
           contiguous: bool) -> list[inference.Beamlet]:
    """n beamlets, keeping every chosen ray whole.

    Beamlets that share a ray reuse its resampling grids, so the second of a
    pair is cheaper than the first. The platform sees plans in beam/ray order
    and gets that reuse; a subset that breaks rays apart reads several percent
    slower per beamlet and is not what the leaderboard measures.

    Whole rays are therefore always kept together. They are drawn evenly across
    the plan so the sample spans its gantry angles and energies -- taking the
    first n beamlets instead would cover only the first few beams, whose ranges
    need not be typical.
    """
    flat = [b for r in rays for b in r]
    if n >= len(flat):
        return flat
    if contiguous:
        return flat[:n]
    per_ray = max(1, round(len(flat) / len(rays)))
    picks = np.linspace(0, len(rays) - 1, max(1, n // per_ray)).round().astype(int)
    out: list[inference.Beamlet] = []
    for i in dict.fromkeys(picks.tolist()):
        out.extend(rays[i])
    return out


def timed(fn):
    """Wall time of fn(), with the GPU queue drained on both sides."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, type=Path,
                   help="patient directory holding image/ and the plan json")
    p.add_argument("--n", type=int, default=200,
                   help="beamlets to run, as whole rays drawn evenly across the plan")
    p.add_argument("--contiguous", action="store_true",
                   help="take the first --n beamlets in plan order instead; faster to "
                        "read from disk but covers only the first few gantry angles")
    p.add_argument("--warmup-beamlets", type=int, default=16,
                   help="leading beamlets excluded from t_dose (default: two batches)")
    p.add_argument("--cutoff", type=float, default=1e-6,
                   help="per-beamlet minimum cutoff in Gy, applied as the platform "
                        "applies the one its metadata carries")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat the beamlet loop; t_dose is the fastest pass")
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    p.add_argument("--csv", type=Path, help="write the measured terms here")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmark.py needs a CUDA device")
    if args.n <= args.warmup_beamlets:
        raise SystemExit(f"--n must exceed --warmup-beamlets ({args.warmup_beamlets})")

    ct_file = args.case / "image" / "ct.mha"
    if not ct_file.is_file():
        raise SystemExit(f"no CT at {ct_file}")

    items = select(plan_rays(args.case), args.n, args.contiguous)
    n_timed = len(items) - args.warmup_beamlets

    # ── startup (not part of the platform's timed window) ────────────────────
    def _load():
        m = inference.build_model(Path(args.model_dir), device="cuda")
        inference.warmup(m)
        return m
    model, t_start = timed(_load)

    # ── t_img ────────────────────────────────────────────────────────────────
    def _image():
        img = sitk.ReadImage(str(ct_file))
        dens = inference.make_density(img, model["hlut"], "cuda")
        ph = patient_phys_coords(img, device="cuda")
        return img, dens, ph
    (image, density, phys), t_img = timed(_image)

    # ── t_dose ───────────────────────────────────────────────────────────────
    # The leading beamlets are discarded: the first batch of a fresh ray pays
    # for grid construction that later beamlets on the same ray reuse, and the
    # allocator is still growing its pools.
    per_pass = []
    for _ in range(args.repeats):
        gen = inference.predict_beamlets(model, items, image, density, phys)
        t0 = None
        for i, (sub, _bbox) in enumerate(gen):
            # In place, on the pinned buffer the generator hands over, which is
            # what the container does before passing it to the compressor. An
            # extra copy here would time the copy, not the model.
            sub *= inference.NORM_SCALE
            inference.apply_cutoff(sub, args.cutoff)
            if i == args.warmup_beamlets - 1:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
        torch.cuda.synchronize()
        per_pass.append((time.perf_counter() - t0) / n_timed)
    t_dose = min(per_pass)

    total = LEADERBOARD_IMAGES * t_img + LEADERBOARD_BEAMLETS * t_dose

    name = torch.cuda.get_device_name(0)
    nx, ny, nz = image.GetSize()
    print(f"\ndevice            {name}")
    print(f"case              {args.case.name}  CT {nx}x{ny}x{nz}")
    print(f"beamlets timed    {n_timed}  ({len(items)} run, first "
          f"{args.warmup_beamlets} discarded)"
          + (f", best of {args.repeats} passes" if args.repeats > 1 else ""))
    print(f"\n  startup weights + warm-up          {t_start:8.2f} s   "
          f"(/health; the platform times /invoke)")
    print(f"  t_img   CT -> density, coordinates {t_img:8.2f} s")
    print(f"  t_dose  per beamlet                {t_dose * 1000:8.1f} ms")
    print(f"\n  runtime for {LEADERBOARD_IMAGES} image and "
          f"{LEADERBOARD_BEAMLETS} beamlets")
    print(f"    t_img + {LEADERBOARD_BEAMLETS} * t_dose = {total:.1f} s")
    print("    the platform adds its own per-invoke t_fix, a few tenths of a second\n")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["device", "case", "n_timed", "startup_s", "t_img_s",
                        "t_dose_ms", "runtime_s"])
            w.writerow([name, args.case.name, n_timed, f"{t_start:.4f}",
                        f"{t_img:.4f}", f"{t_dose * 1000:.3f}", f"{total:.3f}"])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
