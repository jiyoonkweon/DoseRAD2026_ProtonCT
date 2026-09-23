"""Beam's-eye-view geometry: one box per beamlet, anchored at the ray target
with the depth axis along the beam.

The beams in this dataset are parallel pencil beams, so the box is a plain
rotated grid: no divergence, no inverse-square scaling. BEV arrays are ordered
(w, v, u), depth first, so depth integration is a cumulative sum along axis 0.

Sampling uses grid_sample with align_corners=True, with coordinates normalized
as index / (n - 1) * 2 - 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from .config import BOX_LAT_MM, BOX_W_HI_MM, BOX_W_LO_MM, SPACING_MM

_Z_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass
class Frame:
    """Orthonormal beamlet frame, in patient millimetres."""
    target: np.ndarray   # ray target; the BEV origin
    u_hat: np.ndarray    # lateral
    v_hat: np.ndarray    # superior-inferior, orthogonalized against w
    w_hat: np.ndarray    # beam direction


def beamlet_frame(ray_source, ray_target) -> Frame:
    # The direction comes from the ray itself, not from the plan's gantry angle,
    # which is rounded.
    s = np.asarray(ray_source, float)
    t = np.asarray(ray_target, float)
    w = t - s
    n = np.linalg.norm(w)
    if n <= 0:
        raise ValueError(f"degenerate ray: source == target {s}")
    w = w / n
    vz = _Z_AXIS - np.dot(_Z_AXIS, w) * w
    nv = np.linalg.norm(vz)
    if nv < 1e-6:
        raise ValueError(f"ray is parallel to the SI axis: w={w}")
    v = vz / nv
    return Frame(t, np.cross(v, w), v, w)


def bev_axes():
    """The (u, v, w) sample positions of the box, in millimetres.

    Take the point counts from len(): rounding the extent instead can drop the
    endpoint and change the tensor shape.
    """
    half = BOX_LAT_MM / 2
    u = np.arange(-half, half + 1e-6, SPACING_MM)
    v = np.arange(-half, half + 1e-6, SPACING_MM)
    w = np.arange(BOX_W_LO_MM, BOX_W_HI_MM + 1e-6, SPACING_MM)
    return u, v, w


def bev_shape():
    u, v, w = bev_axes()
    return len(w), len(v), len(u)            # (361, 33, 33)


def _image_affine(image: sitk.Image, device):
    origin = torch.as_tensor(np.asarray(image.GetOrigin(), float), dtype=torch.float32,
                             device=device)
    spacing = torch.as_tensor(np.asarray(image.GetSpacing(), float), dtype=torch.float32,
                              device=device)
    D = np.asarray(image.GetDirection(), float).reshape(3, 3)
    return origin, spacing, D


def make_bev_grid(frame: Frame, image: sitk.Image, device="cuda"):
    """grid_sample grid that maps a patient volume onto the BEV box.

    Both beamlets of a ray share it, as do the density and dose volumes.
    """
    ua, va, wa = bev_axes()
    dev = torch.device(device)
    f32 = dict(dtype=torch.float32, device=dev)
    W_, V_, U_ = torch.meshgrid(torch.as_tensor(wa, **f32), torch.as_tensor(va, **f32),
                                torch.as_tensor(ua, **f32), indexing="ij")
    pts = (torch.as_tensor(frame.target, **f32)
           + U_[..., None] * torch.as_tensor(frame.u_hat, **f32)
           + V_[..., None] * torch.as_tensor(frame.v_hat, **f32)
           + W_[..., None] * torch.as_tensor(frame.w_hat, **f32))
    origin, spacing, D = _image_affine(image, dev)
    Dinv = torch.as_tensor(np.linalg.inv(D), **f32)
    idx = ((pts - origin) @ Dinv.T) / spacing                   # (..., 3) = (i, j, k)
    nx, ny, nz = image.GetSize()
    g = torch.stack([idx[..., 0] / (nx - 1), idx[..., 1] / (ny - 1),
                     idx[..., 2] / (nz - 1)], dim=-1) * 2 - 1
    return g[None]                                              # (1, Nw, Nv, Nu, 3)


def resample_to_bev(volume, grid):
    """Patient volume (z, y, x) on the GPU -> BEV volume (Nw, Nv, Nu)."""
    out = F.grid_sample(volume[None, None].float(), grid,
                        mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, 0]


def patient_phys_coords(image: sitk.Image, device="cuda"):
    """Physical coordinates of every patient voxel, (nx, ny, nz, 3).

    Computed once per image and reused by every beamlet of it.
    """
    dev = torch.device(device)
    nx, ny, nz = image.GetSize()
    I, J, K = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev),
                             torch.arange(nz, device=dev), indexing="ij")
    idx = torch.stack([I, J, K], dim=-1).float()
    origin, spacing, D = _image_affine(image, dev)
    return origin + (idx * spacing) @ torch.as_tensor(D, dtype=torch.float32, device=dev).T


def patient_bbox(image: sitk.Image, frame: Frame, margin=2):
    """Patient-grid bounding box of the BEV box, (x0, x1, y0, y1, z0, z1), half-open.

    Outside it the back-projection is exactly zero, so only this region has to
    be evaluated. `margin` covers the bilinear interpolation stencil.
    """
    ua, va, wa = bev_axes()
    corners = np.array([frame.target + du * frame.u_hat + dv * frame.v_hat + dw * frame.w_hat
                        for du in (ua[0], ua[-1])
                        for dv in (va[0], va[-1])
                        for dw in (wa[0], wa[-1])])
    origin = np.asarray(image.GetOrigin(), float)
    spacing = np.asarray(image.GetSpacing(), float)
    D = np.asarray(image.GetDirection(), float).reshape(3, 3)
    idx = ((corners - origin) @ np.linalg.inv(D).T) / spacing
    lo = np.floor(idx.min(axis=0)).astype(int) - margin
    hi = np.ceil(idx.max(axis=0)).astype(int) + margin + 1
    n = np.array(image.GetSize(), int)
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    return (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))


def make_patient_grid(frame: Frame, phys, bbox=None):
    """Normalized BEV coordinates of patient voxels, (1, nx, ny, nz, 3).

    Used both to back-project a BEV prediction and to locate supervision
    points. With `bbox`, only that sub-volume.
    """
    ua, va, wa = bev_axes()
    if bbox is not None:
        x0, x1, y0, y1, z0, z1 = bbox
        phys = phys[x0:x1, y0:y1, z0:z1]
    f32 = dict(dtype=torch.float32, device=phys.device)
    rel = phys - torch.as_tensor(frame.target, **f32)
    uu = (rel * torch.as_tensor(frame.u_hat, **f32)).sum(-1)
    vv = (rel * torch.as_tensor(frame.v_hat, **f32)).sum(-1)
    ww = (rel * torch.as_tensor(frame.w_hat, **f32)).sum(-1)
    gu = ((uu - ua[0]) / SPACING_MM) / (len(ua) - 1) * 2 - 1
    gv = ((vv - va[0]) / SPACING_MM) / (len(va) - 1) * 2 - 1
    gw = ((ww - wa[0]) / SPACING_MM) / (len(wa) - 1) * 2 - 1
    return torch.stack([gu, gv, gw], dim=-1)[None]


def resample_bev_to_patient(bev, grid):
    """BEV volume (Nw, Nv, Nu) -> patient (sub-)volume (z, y, x) over the grid."""
    out = F.grid_sample(bev[None, None].float(), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out[0, 0].permute(2, 1, 0).contiguous()
