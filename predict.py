#!/usr/bin/env python3
"""Predict the dose of every beamlet in a patient's plan.

    python predict.py --data data/DoseRAD2026/proton/training --patient 1THB063 --out pred/

Writes one MetaImage per beamlet in absolute Gy on the CT grid, named like the
reference dose: <out>/Dose_B<beam>_R<ray>_L<beamlet>.mha.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from protondose import data
from protondose.channels import ChannelBuilder
from protondose.inference import load_model, predict, to_full


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory holding the patient folders")
    p.add_argument("--patient", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--weights", default="weights/model.pt")
    p.add_argument("--limit", type=int, default=0,
                   help="predict only this many beamlets, spread evenly over the plan")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    beamlets = data.patient_beamlets(args.data, args.patient)
    if args.limit:
        pick = np.unique(np.linspace(0, len(beamlets) - 1, args.limit).round().astype(int))
        beamlets = [beamlets[i] for i in pick]
    net = load_model(args.weights)
    builder = ChannelBuilder(data.beam_parameters_path(args.data))
    ct = sitk.ReadImage(str(data.ct_path(args.data, args.patient)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for b, dose, bbox in predict(net, builder, ct, beamlets):
        img = sitk.GetImageFromArray(to_full(dose, bbox, ct))
        img.CopyInformation(ct)
        sitk.WriteImage(img, str(out / b.dose_file), useCompression=True)
        print(f"{b.dose_file:<22} E={b.energy:7.2f} MeV  max={dose.max():.4g} Gy", flush=True)


if __name__ == "__main__":
    main()
