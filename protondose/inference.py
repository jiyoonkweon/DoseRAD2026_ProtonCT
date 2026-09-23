"""CT + beamlets -> dose on the patient grid.

    CT -> density -> BEV box per ray -> 5 channels -> network -> clamp at 0
       -> back-projection to the patient grid -> x NORM_SCALE (Gy)
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch

from .channels import ChannelBuilder
from .config import NORM_SCALE
from .geometry import (beamlet_frame, make_bev_grid, make_patient_grid, patient_bbox,
                       patient_phys_coords, resample_bev_to_patient, resample_to_bev)
from .network import Dose3DNet

BATCH_SIZE = 8


def load_model(weights, device="cuda") -> Dose3DNet:
    ck = torch.load(str(weights), map_location="cpu", weights_only=True)
    net = Dose3DNet()
    net.load_state_dict(ck["model"])
    return net.to(device).eval()


@torch.no_grad()
def predict(net, builder: ChannelBuilder, ct: sitk.Image, beamlets, device="cuda"):
    """Yield (beamlet, dose, bbox) in input order.

    dose is the patient-grid sub-volume (z, y, x) in Gy over bbox =
    (x0, x1, y0, y1, z0, z1); everything outside it is exactly zero.
    """
    hu = sitk.GetArrayFromImage(ct).astype(np.float32)
    density = torch.as_tensor(builder.density(hu), device=device)
    phys = patient_phys_coords(ct, device)

    ray_key = None
    for start in range(0, len(beamlets), BATCH_SIZE):
        chunk = beamlets[start:start + BATCH_SIZE]
        inputs, grids, bboxes = [], [], []
        for b in chunk:
            if (b.ray_source, b.ray_target) != ray_key:     # both beamlets of a ray share this
                ray_key = (b.ray_source, b.ray_target)
                frame = beamlet_frame(b.ray_source, b.ray_target)
                dens_bev = resample_to_bev(density, make_bev_grid(frame, ct, device))
                dens_bev = dens_bev.cpu().numpy()
                bbox = patient_bbox(ct, frame)
                grid = make_patient_grid(frame, phys, bbox)
            inputs.append(builder.input(dens_bev, b.energy))
            grids.append(grid)
            bboxes.append(bbox)
        x = torch.from_numpy(np.stack(inputs)).to(device)
        dose_bev = net(x)[:, 0].clamp(min=0)
        for k, b in enumerate(chunk):
            dose = resample_bev_to_patient(dose_bev[k], grids[k]) * NORM_SCALE
            yield b, dose.cpu().numpy(), bboxes[k]


def to_full(sub: np.ndarray, bbox, ct: sitk.Image) -> np.ndarray:
    """Place a bbox sub-volume into a zero volume of the CT's size."""
    nx, ny, nz = ct.GetSize()
    full = np.zeros((nz, ny, nx), np.float32)
    x0, x1, y0, y1, z0, z1 = bbox
    full[z0:z1, y0:y1, x0:x1] = sub
    return full
