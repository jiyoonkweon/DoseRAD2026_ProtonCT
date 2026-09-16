"""Beam's-eye-view geometry: one box per beamlet, anchored at the ray target
with the depth axis along the beam.

BEV arrays are ordered (Nw, Nv, Nu), depth first, so that the depth integration
in channels.py is a cumulative sum along axis 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

# 64 mm across, 720 mm deep, 2 mm isotropic.
BOX_LAT_MM = 64.0
BOX_W_LO_MM = -380.0
BOX_W_HI_MM = 340.0
SPACING_MM = 2.0

_Z_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass
class ProtonFrame:
    """Orthonormal beamlet frame, in patient millimetres."""
    target: np.ndarray   # ray target; the BEV origin
    u_hat: np.ndarray    # lateral
    v_hat: np.ndarray    # superior-inferior, orthogonalized against w
    w_hat: np.ndarray    # beam direction


def beamlet_frame(ray_source, ray_target) -> ProtonFrame:
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
    return ProtonFrame(t, np.cross(v, w), v, w)


def _sp3(sp):
    if np.isscalar(sp):
        return (float(sp),) * 3
    su, sv, sw = sp
    return (float(su), float(sv), float(sw))


def bev_axes(lat_mm=BOX_LAT_MM, w_lo=BOX_W_LO_MM, w_hi=BOX_W_HI_MM, sp=SPACING_MM):
    """The (u, v, w) axes of the BEV box, in millimetres.

    Read the extent back with len(): rounding to get the point count instead
    drops the endpoint and changes the tensor shape.
    """
    su, sv, sw = _sp3(sp)
    u = np.arange(-lat_mm / 2, lat_mm / 2 + 1e-6, su)
    v = np.arange(-lat_mm / 2, lat_mm / 2 + 1e-6, sv)
    w = np.arange(w_lo, w_hi + 1e-6, sw)
    return u, v, w


def make_bev_grid(frame: ProtonFrame, ct_image: sitk.Image,
                  lat_mm=BOX_LAT_MM, w_lo=BOX_W_LO_MM, w_hi=BOX_W_HI_MM,
                  sp=SPACING_MM, device="cuda"):
    """Sampling grid mapping the patient volume onto the BEV box.

    Both beamlets of a ray share it, as do the density and dose volumes.
    """
    import torch
    ua, va, wa = bev_axes(lat_mm, w_lo, w_hi, sp)
    dev = torch.device(device)
    u = torch.as_tensor(ua, dtype=torch.float32, device=dev)
    v = torch.as_tensor(va, dtype=torch.float32, device=dev)
    w = torch.as_tensor(wa, dtype=torch.float32, device=dev)
    uh = torch.as_tensor(frame.u_hat, dtype=torch.float32, device=dev)
    vh = torch.as_tensor(frame.v_hat, dtype=torch.float32, device=dev)
    wh = torch.as_tensor(frame.w_hat, dtype=torch.float32, device=dev)
    tg = torch.as_tensor(frame.target, dtype=torch.float32, device=dev)
    W_, V_, U_ = torch.meshgrid(w, v, u, indexing="ij")
    pts = tg + U_[..., None] * uh + V_[..., None] * vh + W_[..., None] * wh
    origin = np.asarray(ct_image.GetOrigin(), float)
    spacing = np.asarray(ct_image.GetSpacing(), float)
    D = np.asarray(ct_image.GetDirection(), float).reshape(3, 3)
    Dinv = torch.as_tensor(np.linalg.inv(D), dtype=torch.float32, device=dev)
    ori = torch.as_tensor(origin, dtype=torch.float32, device=dev)
    spc = torch.as_tensor(spacing, dtype=torch.float32, device=dev)
    idx = ((pts - ori) @ Dinv.T) / spc
    nx, ny, nz = ct_image.GetSize()
    gx = idx[..., 0] / (nx - 1) * 2 - 1
    gy = idx[..., 1] / (ny - 1) * 2 - 1
    gz = idx[..., 2] / (nz - 1) * 2 - 1
    return torch.stack([gx, gy, gz], dim=-1)[None]


def resample_to_bev(volume_gpu, grid):
    """Patient volume (z, y, x) on the GPU -> BEV volume (Nw, Nv, Nu)."""
    import torch.nn.functional as F
    out = F.grid_sample(volume_gpu[None, None].float(), grid,
                        mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, 0]


def patient_phys_coords(ct_image: sitk.Image, device="cuda"):
    """Physical coordinates of every patient voxel, (nx, ny, nz, 3).

    Computed once per image and reused by every beamlet of it.
    """
    import torch
    dev = torch.device(device)
    nx, ny, nz = ct_image.GetSize()
    origin = np.asarray(ct_image.GetOrigin(), float)
    spacing = np.asarray(ct_image.GetSpacing(), float)
    D = np.asarray(ct_image.GetDirection(), float).reshape(3, 3)
    ii = torch.arange(nx, device=dev)
    jj = torch.arange(ny, device=dev)
    kk = torch.arange(nz, device=dev)
    I, J, K = torch.meshgrid(ii, jj, kk, indexing="ij")
    idx = torch.stack([I, J, K], dim=-1).float()
    Dt = torch.as_tensor(D, dtype=torch.float32, device=dev)
    return (torch.as_tensor(origin, dtype=torch.float32, device=dev)
            + (idx * torch.as_tensor(spacing, dtype=torch.float32, device=dev)) @ Dt.T)


def patient_bbox_for_frame(ct_image: sitk.Image, frame: ProtonFrame,
                           lat_mm=BOX_LAT_MM, w_lo=BOX_W_LO_MM, w_hi=BOX_W_HI_MM,
                           margin=2):
    """Patient-grid bounding box of the BEV box, half-open indices.

    Outside it the back-projection is exactly zero, so only this region has to
    be evaluated. `margin` covers the bilinear interpolation stencil.
    """
    ua, va, wa = bev_axes(lat_mm, w_lo, w_hi, SPACING_MM)
    corners = np.array([frame.target + du * frame.u_hat + dv * frame.v_hat + dw * frame.w_hat
                        for du in (ua[0], ua[-1])
                        for dv in (va[0], va[-1])
                        for dw in (wa[0], wa[-1])])
    origin = np.asarray(ct_image.GetOrigin(), float)
    spacing = np.asarray(ct_image.GetSpacing(), float)
    D = np.asarray(ct_image.GetDirection(), float).reshape(3, 3)
    idx = ((corners - origin) @ np.linalg.inv(D).T) / spacing
    lo = np.floor(idx.min(axis=0)).astype(int) - margin
    hi = np.ceil(idx.max(axis=0)).astype(int) + margin + 1
    n = np.array(ct_image.GetSize(), int)
    lo = np.clip(lo, 0, n)
    hi = np.clip(hi, 0, n)
    return (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))


def make_patient_grid(ct_image: sitk.Image, frame: ProtonFrame,
                      lat_mm=BOX_LAT_MM, w_lo=BOX_W_LO_MM, w_hi=BOX_W_HI_MM,
                      sp=SPACING_MM, device="cuda", phys=None, bbox=None):
    """Maps the patient bounding box back onto normalized BEV coordinates."""
    import torch
    ua, va, wa = bev_axes(lat_mm, w_lo, w_hi, sp)
    Nu, Nv, Nw = len(ua), len(va), len(wa)
    dev = torch.device(device)
    if phys is None:
        phys = patient_phys_coords(ct_image, device)
    if bbox is not None:
        x0, x1, y0, y1, z0, z1 = bbox
        phys = phys[x0:x1, y0:y1, z0:z1]
    su, sv, sw = _sp3(sp)
    rel = phys - torch.as_tensor(frame.target, dtype=torch.float32, device=dev)
    uu = (rel * torch.as_tensor(frame.u_hat, dtype=torch.float32, device=dev)).sum(-1)
    vv = (rel * torch.as_tensor(frame.v_hat, dtype=torch.float32, device=dev)).sum(-1)
    ww = (rel * torch.as_tensor(frame.w_hat, dtype=torch.float32, device=dev)).sum(-1)
    gu = ((uu - ua[0]) / su) / (Nu - 1) * 2 - 1
    gv = ((vv - va[0]) / sv) / (Nv - 1) * 2 - 1
    gw = ((ww - wa[0]) / sw) / (Nw - 1) * 2 - 1
    return torch.stack([gu, gv, gw], dim=-1)[None]


def resample_bev_to_patient(bev_volume, grid):
    """BEV dose -> patient sub-volume over the bounding box, (nz, ny, nx)."""
    import torch.nn.functional as F
    vol = bev_volume[None, None] if bev_volume.dim() == 3 else bev_volume
    out = F.grid_sample(vol.float(), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out[0, 0].permute(2, 1, 0).contiguous()
