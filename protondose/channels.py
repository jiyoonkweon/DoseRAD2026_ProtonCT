"""The five input channels, built on the BEV box.

    C1 density   mass density from CT via the official HU-to-density table
    C2 fluence   Gaussian spot of the energy's sigma, identical at every depth
    C3 WED       water-equivalent depth along the beam, sum(rho * dw), in cm
    C4 energy    nominal energy / E_MAX_MEV, constant
    C5 Bragg     analytic Bragg curve looked up at each voxel's WED (bragg.py)

Training builds them from the cached density (prepare_data.py), inference from
the CT directly; both go through ChannelBuilder.input().
"""
from __future__ import annotations

import json

import numpy as np

from .bragg import BraggTable
from .config import E_MAX_MEV, SPACING_MM
from .geometry import bev_axes


class EnergyTable:
    """Spot size and energy spread per nominal energy (85 levels)."""

    def __init__(self, beam_parameters_path):
        with open(beam_parameters_path) as f:
            tab = json.load(f)["proton"]["energy_table"]
        self.E = np.array([r["energy_mev"] for r in tab], float)
        self.sigma_energy = np.array([r["sigma_energy_mev"] for r in tab], float)
        self.sigma_spot = np.array([r["sigma_spot_mm"] for r in tab], float)

    def index(self, energy_mev: float, tol: float = 1e-3) -> int:
        # Plan energies come from this table. Anything else means the input does
        # not match the beam model, and interpolating would hide that.
        i = int(np.argmin(np.abs(self.E - energy_mev)))
        if abs(self.E[i] - energy_mev) > tol:
            raise KeyError(f"energy {energy_mev} is not in the table (nearest {self.E[i]})")
        return i


def load_hu_to_density(beam_parameters_path) -> np.ndarray:
    """Official HU -> mass density table as (N, 2) = [HU, g/cm^3]."""
    with open(beam_parameters_path) as f:
        entries = json.load(f)["hu_to_density"]["entries"]
    return np.array([[e["hu"], e["density_g_cm3"]] for e in entries], dtype=float)


def spot_fluence(sigma_spot_mm: float) -> np.ndarray:
    """C2: lateral Gaussian with peak 1, the same on every depth slice."""
    u, v, w = bev_axes()
    sig = max(float(sigma_spot_mm), 1e-3)
    r2 = u[None, :] ** 2 + v[:, None] ** 2
    plane = np.exp(-r2 / (2.0 * sig ** 2))
    return np.broadcast_to(plane, (len(w),) + plane.shape).astype(np.float32)


def radiological_depth(density_bev: np.ndarray) -> np.ndarray:
    """C3: water-equivalent depth at voxel centres, in cm (= g/cm^2).

    The half-cell term makes the sum voxel-centred. A plain cumulative sum
    overestimates every voxel by rho * dw / 2, about 1 mm in soft tissue, and C5
    would inherit that as a shallow Bragg peak.
    """
    c = np.cumsum(density_bev, axis=0) - 0.5 * density_bev
    return (c * (SPACING_MM / 10.0)).astype(np.float32)


class ChannelBuilder:
    def __init__(self, beam_parameters_path):
        self.table = EnergyTable(beam_parameters_path)
        self.bragg = BraggTable(self.table)
        self.hlut = load_hu_to_density(beam_parameters_path)
        self._fluence = {}                 # energy -> C2, rendered once per energy

    def density(self, hu: np.ndarray) -> np.ndarray:
        """C1 source: HU -> g/cm^3, piecewise linear, no clipping or masking."""
        return np.interp(hu, self.hlut[:, 0], self.hlut[:, 1]).astype(np.float32)

    def fluence(self, energy: float) -> np.ndarray:
        f = self._fluence.get(energy)
        if f is None:
            f = spot_fluence(self.table.sigma_spot[self.table.index(energy)])
            self._fluence[energy] = f
        return f

    def input(self, density_bev: np.ndarray, energy: float) -> np.ndarray:
        """(5, Nw, Nv, Nu) network input for one beamlet."""
        c1 = density_bev.astype(np.float32)
        c3 = radiological_depth(c1)
        c4 = np.full(c1.shape, energy / E_MAX_MEV, np.float32)
        c5 = self.bragg.channel(c3, energy)
        return np.stack([c1, self.fluence(energy), c3, c4, c5], 0)
