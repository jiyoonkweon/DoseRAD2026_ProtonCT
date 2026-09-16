#!/usr/bin/env python3
"""Predict beamlet dose on a patient CT.

    python predict.py --ct CT.mha --beamlets plan.json --out dose/

The beamlet file is either a patient plan from the DoseRAD2026 training set or
the challenge's beam-level metadata; both carry the ray endpoints and nominal
energies. Same computation as the container, without the request handling and
the 4-D stacked output format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from protondose import inference
from protondose.geometry import patient_phys_coords


def read_beamlets(path: Path) -> list[tuple[inference.Beamlet, float | None]]:
    """Beamlets and their dose cutoffs, from either accepted schema."""
    doc = json.loads(path.read_text())
    images = doc if isinstance(doc, list) else [doc]     # plan file: a single image
    out = []
    for image in images:
        for beam in image.get("beams", []):
            for ray in beam.get("rays", []):
                for bl in ray.get("beamlets", []):
                    cutoff = bl.get("output_info", {}).get("minimum_cutoff")
                    out.append((inference.Beamlet(ray["ray_source"], ray["ray_target"],
                                                  bl["energy"]), cutoff))
    if not out:
        raise SystemExit(f"no beamlets in {path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ct", required=True, help="patient CT, .mha")
    p.add_argument("--beamlets", required=True, type=Path,
                   help="patient plan json, or challenge beam-level metadata")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    args = p.parse_args()

    items = read_beamlets(args.beamlets)
    model = inference.build_model(Path(args.model_dir), device="cuda")
    # Also for reproducibility: cuDNN picks its convolution algorithm on the
    # first forward pass, and a different choice shifts results by ~1e-4.
    inference.warmup(model)

    image = sitk.ReadImage(args.ct)
    density = inference.make_density(image, model["hlut"], "cuda")
    phys = patient_phys_coords(image, device="cuda")

    nx, ny, nz = image.GetSize()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = inference.predict_beamlets(model, [b for b, _ in items], image, density, phys)
    for k, ((beamlet, cutoff), (sub, bbox)) in enumerate(zip(items, gen)):
        sub = np.array(sub, dtype=np.float32) * inference.NORM_SCALE
        sub = inference.apply_cutoff(sub, cutoff)
        full = np.zeros((nz, ny, nx), np.float32)
        x0, x1, y0, y1, z0, z1 = bbox
        full[z0:z1, y0:y1, x0:x1] = sub
        dose = sitk.GetImageFromArray(full)
        dose.CopyInformation(image)
        path = out_dir / f"dose_{k:03d}.mha"
        sitk.WriteImage(dose, str(path), useCompression=True)
        print(f"{path}  E={beamlet.energy:.2f} MeV  max={full.max():.6g} Gy", flush=True)


if __name__ == "__main__":
    main()
