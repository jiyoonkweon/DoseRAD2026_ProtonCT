#!/usr/bin/env python3
"""Measure inference runtime on its own, without scoring anything.

    python benchmark.py --case /path/to/DoseRAD2026/proton/training/1THB016

`evaluate.py --runtime` reports the same number next to the accuracy metrics and
is the usual way in. This entry point exists because it needs only the CT and
the plan file, not the reference Monte Carlo dose, so runtime can be checked on
a machine that does not hold the reference volumes.

protondose/timing.py holds the measurement and explains what it does and does
not include.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from protondose import timing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, type=Path,
                   help="patient directory holding image/ and the plan json")
    p.add_argument("--n", type=int, default=0,
                   help="beamlets to run, as whole rays drawn evenly across the plan; "
                        "0, the default, runs the whole plan")
    p.add_argument("--contiguous", action="store_true",
                   help="take the first --n beamlets in plan order instead; covers "
                        "only the first few gantry angles")
    p.add_argument("--warmup-beamlets", type=int, default=16,
                   help="leading beamlets excluded from t_dose (default: two batches)")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat the beamlet loop and keep the fastest pass; that is "
                        "warmer than a freshly started container, so 1 is the honest "
                        "default")
    p.add_argument("--cutoff", type=float, default=timing.DEFAULT_CUTOFF_GY,
                   help="per-beamlet minimum cutoff in Gy, applied as the platform "
                        "applies the one its metadata carries")
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    p.add_argument("--csv", type=Path, help="write the measured terms here")
    args = p.parse_args()

    rt = timing.measure_runtime(args.case, Path(args.model_dir), n=args.n,
                                warmup_beamlets=args.warmup_beamlets,
                                repeats=args.repeats, cutoff=args.cutoff,
                                contiguous=args.contiguous)
    print(f"\ndevice            {rt.device}")
    print(f"case              {rt.case}")
    print()
    print(rt.report())
    print()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["device", "case", "n_timed", "startup_s", "t_img_s",
                        "t_dose_ms", "runtime_s"])
            w.writerow([rt.device, rt.case, rt.n_timed, f"{rt.t_startup:.4f}",
                        f"{rt.t_img:.4f}", f"{rt.t_dose * 1000:.3f}",
                        f"{rt.total:.3f}"])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
