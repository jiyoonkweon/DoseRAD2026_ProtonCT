"""Input channels C1-C4 and the HU-to-density table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import bev_axes, BOX_LAT_MM, BOX_W_LO_MM, BOX_W_HI_MM, SPACING_MM

E_MAX_MEV = 200.7966
N_ENERGIES = 85


class ProtonEnergyTable:
    """Spot size and energy spread per nominal energy, from beam_parameters.json."""

    def __init__(self, beam_parameters_path: str):
        with open(beam_parameters_path) as f:
            p = json.load(f)["proton"]
        assert p.get("source_model") == "point_source", p.get("source_model")
        assert p.get("spot_profile") == "Gaussian", p.get("spot_profile")
        tab = p["energy_table"]
        self.E = np.array([r["energy_mev"] for r in tab], float)
        self.sE = np.array([r["sigma_energy_mev"] for r in tab], float)
        self.sS = np.array([r["sigma_spot_mm"] for r in tab], float)
        assert len(self.E) == N_ENERGIES and np.all(np.diff(self.E) > 0)

    def lookup(self, energy_mev: float, tol: float = 1e-3):
        # Plan energies come from this table, so a miss means the input does not
        # match the beam model. Interpolating would silently change C2.
        i = int(np.argmin(np.abs(self.E - energy_mev)))
        if abs(self.E[i] - energy_mev) > tol:
            raise KeyError(f"energy {energy_mev} is not in the table "
                           f"(nearest {self.E[i]})")
        return float(self.sE[i]), float(self.sS[i])


def spot_fluence(sigma_spot_mm: float, lat_mm=BOX_LAT_MM, w_lo=BOX_W_LO_MM,
                 w_hi=BOX_W_HI_MM, sp=SPACING_MM) -> np.ndarray:
    """C2: Gaussian spot profile, (Nw, Nv, Nu).

    The beams are parallel, so the width does not vary with depth and the
    amplitude is exactly 1: every depth slice carries the same lateral Gaussian.
    """
    u, v, w = bev_axes(lat_mm, w_lo, w_hi, sp)
    sig = max(float(sigma_spot_mm), 1e-3)
    r2 = u[None, :] ** 2 + v[:, None] ** 2
    plane = np.exp(-r2 / (2.0 * sig ** 2))
    return np.broadcast_to(plane, (len(w),) + plane.shape).astype(np.float32)


def radiological_depth(density_bev: np.ndarray, sp=SPACING_MM) -> np.ndarray:
    """C3: water-equivalent path length along the depth axis, in cm.

    `sp` is the spacing of the depth axis, since the integration runs along
    axis 0. The half-cell correction makes the sum voxel-centred; a plain
    cumulative sum would place the Bragg prior, which indexes into C3, half a
    cell too shallow.
    """
    c = np.cumsum(density_bev, axis=0) - 0.5 * density_bev
    return (c * (sp / 10.0)).astype(np.float32)


def energy_channel(energy_mev: float, shape) -> np.ndarray:
    """C4: nominal energy normalized by the table maximum."""
    return np.full(shape, energy_mev / E_MAX_MEV, np.float32)


def load_hu_to_density(beam_parameters_path: str | Path) -> np.ndarray:
    """Official HU -> density table as (N, 2) = [HU, density_g_cm3].

    This is mass density, not relative electron density: the same calibration
    Geant4 used to build the reference doses.
    """
    with open(beam_parameters_path) as f:
        entries = json.load(f)["hu_to_density"]["entries"]
    hu = np.array([e["hu"] for e in entries], dtype=float)
    rho = np.array([e["density_g_cm3"] for e in entries], dtype=float)
    return np.column_stack([hu, rho])


def hu_to_density(hu: np.ndarray, hlut: np.ndarray) -> np.ndarray:
    return np.interp(hu, hlut[:, 0], hlut[:, 1]).astype(np.float32)
