"""Beamlet-wise proton dose prediction.

    patient CT -> BEV resample -> 5 physics channels -> 3D U-Net -> BEV dose
    -> clamp -> back-projection to the patient grid -> absolute Gy -> cutoff

Independent of the challenge request format; see challenge_io.py for that.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch

from .bragg import BraggTable
from .channels import (ProtonEnergyTable, energy_channel, hu_to_density,
                       load_hu_to_density, radiological_depth, spot_fluence)
from .geometry import (beamlet_frame, bev_axes, make_bev_grid, make_patient_grid,
                       patient_bbox_for_frame, resample_bev_to_patient,
                       resample_to_bev)
from .network import Dose3DNet

CKPT_NAME = "model.pt"
BEAM_PARAMS_NAME = "beam_parameters.json"

# Shared with challenge_io: the device-to-host ring below must not hand a buffer
# back before the compression worker holding it has finished, so its capacity is
# tied to how many frames the writer keeps in flight.
N_COMPRESS_WORKERS = max(2, (os.cpu_count() or 8) - 2)
MAX_INFLIGHT = N_COMPRESS_WORKERS + 2

# Fixed configuration of the released checkpoint, asserted in build_model().
BATCH = 8
IN_CHANNELS = 5
NORM_SCALE = 1.1187e-3                   # prediction -> absolute Gy
BOX_LAT_MM = 64.0
BOX_W_LO_MM = -380.0
BOX_W_HI_MM = 340.0
SPACING_MM = (2.0, 2.0, 2.0)

EXPECTED_ARGS = {
    "model": "dose3d", "predict": "dose", "core3d": "convlstm",
    "width": 24, "depth3d": 3, "strides": "222,222,211",
    "se3d": True, "attn3d": False, "twostream3d": False, "bidir3d": False,
    "refine": 0, "bragg_prior": True, "bragg_split": False, "mcs": False,
    "norm": "global", "norm_scale": NORM_SCALE,
    "wed_rsp": False, "wed_trapezoid": True,
    "spot_divergence": False, "bragg_closed": True,
    "w_lo_mm": BOX_W_LO_MM, "w_hi_mm": BOX_W_HI_MM,
}

# Skip the cutoff if it exceeds this fraction of the predicted maximum, which
# would zero the whole beamlet. Never triggered on challenge data.
CUTOFF_GUARD_RATIO = 0.5


class Beamlet:
    """One beamlet request; the frame is determined by the ray vector alone."""
    __slots__ = ("ray_source", "ray_target", "energy", "ray_key")

    def __init__(self, ray_source, ray_target, energy):
        self.ray_source = np.asarray(ray_source, dtype=float)
        self.ray_target = np.asarray(ray_target, dtype=float)
        self.energy = float(energy)
        # Plan JSON repeats ray coordinates bit-for-bit, so exact equality is a
        # valid test for "same ray".
        self.ray_key = (tuple(self.ray_source), tuple(self.ray_target))


def build_model(model_dir: Path, device: str = "cuda") -> Dict:
    if device != "cuda":
        raise RuntimeError("this model requires a CUDA device")

    # Part of the released configuration, not a tuning knob: changing either
    # shifts the output by a few times 1e-4 relative to the beamlet maximum.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ckpt = Path(model_dir) / CKPT_NAME
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    args = ck.get("args") or {}

    # A checkpoint that disagrees would produce silently wrong channels.
    for key, want in EXPECTED_ARGS.items():
        got = args.get(key)
        if isinstance(want, bool):
            got = bool(got)
        elif isinstance(want, float) and got is not None:
            got = float(got)
        if got != want:
            raise RuntimeError(f"checkpoint arg {key}={got!r} but this code "
                               f"implements {want!r}")
    box = args.get("box")
    if tuple(box or ()) != (BOX_LAT_MM, BOX_LAT_MM, BOX_W_HI_MM - BOX_W_LO_MM):
        raise RuntimeError(f"checkpoint box={box!r} does not match this code")
    if tuple(float(x) for x in args.get("spacing") or ()) != SPACING_MM:
        raise RuntimeError(f"checkpoint spacing={args.get('spacing')!r} does not match")

    net = Dose3DNet()
    missing, unexpected = net.load_state_dict(ck["model"], strict=True)
    assert not missing and not unexpected
    net = net.to(device).eval()

    bp = Path(model_dir) / BEAM_PARAMS_NAME
    if not bp.exists():
        raise FileNotFoundError(f"beam parameters not found: {bp}")

    print(f"[build_model] {ckpt.name} epoch={ck.get('epoch')} in_ch={IN_CHANNELS} "
          f"norm_scale={NORM_SCALE:g} lat={BOX_LAT_MM} "
          f"w=[{BOX_W_LO_MM},{BOX_W_HI_MM}] spacing={SPACING_MM} batch={BATCH}",
          flush=True)
    return {"net": net, "device": device,
            "table": ProtonEnergyTable(str(bp)), "bragg": BraggTable(str(bp)),
            "hlut": load_hu_to_density(str(bp))}


# C2 and C4 depend only on the beam energy, and there are 85 of those, so they
# can be built before any request arrives.
_FLU_CACHE: Dict[float, np.ndarray] = {}
_ECH_CACHE: Dict[float, np.ndarray] = {}


def prewarm_energy_channels(model: Dict) -> None:
    u, v, w = bev_axes(BOX_LAT_MM, BOX_W_LO_MM, BOX_W_HI_MM, SPACING_MM)
    shape = (len(w), len(v), len(u))
    dev, table = model["device"], model["table"]
    t0 = time.time()
    for _e in table.E:
        e = float(_e)
        _, sigma_spot = table.lookup(e)
        flu = spot_fluence(sigma_spot, BOX_LAT_MM, BOX_W_LO_MM, BOX_W_HI_MM, SPACING_MM)
        ech = energy_channel(e, shape)
        _FLU_CACHE[e], _ECH_CACHE[e] = flu, ech
        _gpu_const(("flu", e), flu, dev)
        _gpu_const(("ech", e), ech, dev)
    mb = sum(a.nbytes for a in _FLU_CACHE.values()) \
        + sum(a.nbytes for a in _ECH_CACHE.values())
    print(f"[prewarm] {len(_FLU_CACHE)} energy channels "
          f"({mb / 2 ** 20:.0f} MiB host, same on GPU) in {time.time() - t0:.2f}s",
          flush=True)


def warmup(model: Dict) -> None:
    """Run the real batch shape once so cuDNN autotuning happens for free.

    The shape comes from bev_axes(), the same way the request path computes it;
    a different shape would discard the autotuned plans on the first beamlet.
    """
    net, device = model["net"], model["device"]
    u, v, w = bev_axes(BOX_LAT_MM, BOX_W_LO_MM, BOX_W_HI_MM, SPACING_MM)
    nu, nv, nw = len(u), len(v), len(w)
    t0 = time.time()
    x = torch.zeros((BATCH, IN_CHANNELS, nw, nv, nu), device=device)
    with torch.no_grad():
        for _ in range(2):                  # first pass searches, second is real
            net(x)
    del x
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    # Not empty_cache(): releasing the warmed allocator blocks would force
    # reallocation inside the timed window.
    print(f"[warmup] shape=({nw},{nv},{nu}) batch={BATCH} "
          f"took {time.time() - t0:.2f}s peak {peak:.2f} GiB", flush=True)
    prewarm_energy_channels(model)


def make_density(image: sitk.Image, hlut, device: str) -> torch.Tensor:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    return torch.as_tensor(hu_to_density(arr, hlut), dtype=torch.float32, device=device)




CUTOFF_SKIPPED = {"n": 0, "max_seen": 0.0, "cutoffs": set()}


def apply_cutoff(dose: np.ndarray, cutoff) -> np.ndarray:
    """Zero 0 < dose <= cutoff, in place.

    The evaluator counts (pred > 0) & (pred < cutoff), so <= is conservative.
    Applying this to the sub-volume matches applying it to the full frame:
    everything outside the box is exactly zero.
    """
    if cutoff is None:
        return dose
    c = float(cutoff)
    if c <= 0:
        return dose
    dmax = float(dose.max()) if dose.size else 0.0
    if dmax > 0 and c > CUTOFF_GUARD_RATIO * dmax:
        CUTOFF_SKIPPED["n"] += 1
        CUTOFF_SKIPPED["max_seen"] = max(CUTOFF_SKIPPED["max_seen"], dmax)
        CUTOFF_SKIPPED["cutoffs"].add(c)
        return dose
    dose[dose <= c] = 0.0
    return dose


# Host-to-device traffic, not the forward pass, dominates, so the input tensor is
# built on the GPU: C1-C4 are already resident and filled by device-to-device
# copies, and only C5 goes through a pinned staging buffer. Results come back
# into a ring of pinned buffers so the compression workers can read them without
# a copy. All of this moves bytes; none of it changes a value.

_GPU_IN = None                  # persistent network input buffer
_GPU_CH: Dict = {}              # per-energy constant channels, on the GPU
_H2D_BUF = None                 # pinned staging buffer for C5
_H2D_EVT = None


def _gpu_in(shape, dev):
    global _GPU_IN
    if _GPU_IN is None or tuple(_GPU_IN.shape) != tuple(shape):
        _GPU_IN = torch.empty(tuple(shape), dtype=torch.float32, device=dev)
    return _GPU_IN


def _gpu_const(key, arr, dev):
    t = _GPU_CH.get(key)
    if t is None:
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(dev)
        _GPU_CH[key] = t
    return t


def _h2d_staging(shape):
    # Wait for the previous transfer before overwriting: otherwise the host
    # rewrites bytes that are still in flight.
    global _H2D_BUF
    if _H2D_EVT is not None:
        _H2D_EVT.synchronize()
    if _H2D_BUF is None or tuple(_H2D_BUF.shape) != tuple(shape):
        _H2D_BUF = torch.empty(tuple(shape), dtype=torch.float32, pin_memory=True)
    return _H2D_BUF, _H2D_BUF.numpy()


_D2H_RING: List = []
_D2H_POS = 0


def _d2h_next(shape, dtype):
    """Next pinned buffer from the ring.

    The writer waits for the oldest compression once MAX_INFLIGHT frames are
    queued, so a slot's previous contents are consumed before it comes round
    again. The slack covers the slot boundary, where the in-flight queue restarts
    and one slot's tail can overlap the next.
    """
    global _D2H_POS
    if not _D2H_RING:
        _D2H_RING.extend([None] * (2 * MAX_INFLIGHT + BATCH + 2))
    i = _D2H_POS % len(_D2H_RING)
    _D2H_POS += 1
    n = int(np.prod(shape))
    buf = _D2H_RING[i]
    if buf is None or buf.numel() < n or buf.dtype != dtype:
        _D2H_RING[i] = torch.empty(int(n * 1.2) + 1, dtype=dtype, pin_memory=True)
        buf = _D2H_RING[i]
    return buf[:n].view(*shape)


def predict_beamlets(model: Dict, items: List[Beamlet], image: sitk.Image,
                     density_gpu: torch.Tensor, phys: torch.Tensor):
    """Yield (sub_volume, bbox) per beamlet, in the order given.

    Only the patient-grid bounding box of the BEV box is produced; everything
    outside it back-projects to exactly zero.

    Order is preserved rather than sorted by ray: the platform already emits
    beamlets in beam/ray order, so the ray cache hits anyway, and the output has
    to be written in the requested order.
    """
    global _H2D_EVT

    net, dev = model["net"], model["device"]
    table, bragg = model["table"], model["bragg"]

    cur_key = None
    cur_dens_np = cur_wed = cur_pgrid = cur_bbox = None
    cur_dens_gpu = cur_wed_gpu = None

    for start in range(0, len(items), BATCH):
        chunk = range(start, min(start + BATCH, len(items)))
        pgrids, pboxes = [], []
        c5_gpu = c5_np = None
        for i in chunk:
            it = items[i]
            if it.ray_key != cur_key:
                cur_pgrid = None        # free before allocating the replacement
                frame = beamlet_frame(it.ray_source, it.ray_target)
                bevgrid = make_bev_grid(frame, image, BOX_LAT_MM, BOX_W_LO_MM,
                                        BOX_W_HI_MM, SPACING_MM, dev)
                dens = resample_to_bev(density_gpu, bevgrid)
                cur_dens_np = dens.cpu().numpy()
                cur_dens_gpu = dens
                cur_wed = radiological_depth(cur_dens_np, SPACING_MM[2])
                cur_wed_gpu = torch.from_numpy(np.ascontiguousarray(cur_wed)).to(dev)
                cur_bbox = patient_bbox_for_frame(image, frame, BOX_LAT_MM,
                                                  BOX_W_LO_MM, BOX_W_HI_MM)
                cur_pgrid = make_patient_grid(image, frame, BOX_LAT_MM, BOX_W_LO_MM,
                                              BOX_W_HI_MM, SPACING_MM, dev,
                                              phys=phys, bbox=cur_bbox)
                cur_key = it.ray_key
                del bevgrid

            flu = _FLU_CACHE.get(it.energy)
            if flu is None:                       # outside the prewarmed table
                _, sigma_spot = table.lookup(it.energy)
                flu = spot_fluence(sigma_spot, BOX_LAT_MM, BOX_W_LO_MM,
                                   BOX_W_HI_MM, SPACING_MM)
                _FLU_CACHE[it.energy] = flu
            ech = _ECH_CACHE.get(it.energy)
            if ech is None:
                ech = energy_channel(it.energy, cur_dens_np.shape)
                _ECH_CACHE[it.energy] = ech
            c5 = bragg.channel(cur_wed, it.energy)

            k = len(pgrids)
            gin = _gpu_in((BATCH, IN_CHANNELS) + cur_dens_np.shape, dev)
            gin[k, 0].copy_(cur_dens_gpu, non_blocking=True)
            gin[k, 1].copy_(_gpu_const(("flu", it.energy), flu, dev), non_blocking=True)
            gin[k, 2].copy_(cur_wed_gpu, non_blocking=True)
            gin[k, 3].copy_(_gpu_const(("ech", it.energy), ech, dev), non_blocking=True)
            if c5_np is None:
                c5_gpu, c5_np = _h2d_staging((BATCH, 1) + cur_dens_np.shape)
            np.copyto(c5_np[k, 0], c5, casting="same_kind")
            pgrids.append(cur_pgrid)
            pboxes.append(cur_bbox)

        n_real = len(pgrids)
        gin = _GPU_IN
        # Pad to a fixed batch shape so the warmed cuDNN plans keep applying;
        # the padding is discarded after the forward pass.
        for j in range(n_real, BATCH):
            c5_np[j] = c5_np[n_real - 1]
        gin[:, 4:].copy_(c5_gpu, non_blocking=True)
        _H2D_EVT = torch.cuda.Event()
        _H2D_EVT.record()
        for j in range(n_real, BATCH):
            gin[j, :4].copy_(gin[n_real - 1, :4], non_blocking=True)

        with torch.no_grad():
            dose_bev = net(gin)[:, 0].clamp(min=0)[:n_real]

        pending = []
        for k in range(n_real):
            sub = resample_bev_to_patient(dose_bev[k], pgrids[k])
            buf = _d2h_next(sub.shape, sub.dtype)
            buf.copy_(sub.detach(), non_blocking=True)
            pending.append((buf, pboxes[k]))
            del sub
        torch.cuda.synchronize()
        for buf, bbox in pending:
            yield buf.numpy(), bbox     # the ring keeps this valid; see _d2h_next
        del dose_bev

