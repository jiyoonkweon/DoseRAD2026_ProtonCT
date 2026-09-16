"""C5: analytic Bragg-curve prior (Bortfeld 1997) in water.

Per nominal energy the depth-dose curve is evaluated once, convolved with the
range spread, normalized to its peak and cached; the channel is a lookup of that
curve at the radiological depth of every voxel.
"""
from __future__ import annotations

import json

import numpy as np

ALPHA_CM = 0.0022          # R0 = ALPHA * E**P_EXP, R0 in cm, E in MeV
P_EXP = 1.77
Z_STEP_CM = 0.02           # output table resolution
N_PER_SIGMA = 40           # internal grid samples per straggling sigma
K_SIGMA = 6.0              # convolution kernel half-width, in sigma
N_ENERGIES = 85


def r0_cm(e_mev: float) -> float:
    return ALPHA_CM * e_mev ** P_EXP


def _sigma_r(e_mev: float, sigma_e_mev: float):
    """(R0, sigma_R) in cm: monoenergetic straggling plus energy spread."""
    R0 = r0_cm(e_mev)
    s_mono = 0.012 * R0 ** 0.935
    s_e = sigma_e_mev * ALPHA_CM * P_EXP * e_mev ** (P_EXP - 1.0)
    return R0, float(np.hypot(s_mono, s_e))


def _cell_avg_pow(R0, z, h, a):
    """Cell average of (R0 - t)**a over [z - h/2, z + h/2] clipped to [0, R0].

    Integrating over the cell rather than point-sampling keeps the singular
    primary term finite at the peak, so no clipping is needed and the peak
    normalization does not depend on the grid spacing. Dividing by the full cell
    width h, not the overlap width, is what makes sum(value) * h the exact
    integral of the function extended by zero.
    """
    lo = np.clip(z - 0.5 * h, 0.0, R0)
    hi = np.clip(z + 0.5 * h, 0.0, R0)
    A = np.clip(R0 - lo, 0.0, None) ** (a + 1.0)
    B = np.clip(R0 - hi, 0.0, None) ** (a + 1.0)
    return (A - B) / ((a + 1.0) * h)


def _conv_same(x, ker):
    # numpy's mode="same" returns max(len(x), len(ker)), which is wrong at low
    # energies where the kernel is longer than the curve.
    f = np.convolve(x, ker, "full")
    st = (len(ker) - 1) // 2
    return f[st:st + len(x)]


def _dd_curve(e_mev: float, sigma_e_mev: float):
    """(z_cm, depth_dose), normalized so the peak of the total is 1."""
    R0, s_r = _sigma_r(e_mev, sigma_e_mev)
    h = min(Z_STEP_CM, s_r / N_PER_SIGMA)
    zi = np.arange(0.0, R0 + (K_SIGMA + 1.0) * s_r + 0.5, h)
    prim = 17.93 * _cell_avg_pow(R0, zi, h, -0.435)      # primary stopping
    nucl = 0.444 * _cell_avg_pow(R0, zi, h, 0.565)       # nuclear secondaries
    k = int(np.ceil(K_SIGMA * s_r / h))
    kz = np.arange(-k, k + 1) * h
    kern = np.exp(-kz ** 2 / (2 * s_r ** 2))
    kern /= kern.sum()
    dd = _conv_same(prim, kern) + _conv_same(nucl, kern)
    peak = float(dd.max())
    z = np.arange(0.0, R0 + 6 * s_r + 0.5, Z_STEP_CM)
    return z, np.interp(z, zi, dd / peak).astype(np.float32)


class BraggTable:
    """Per-energy depth-dose curves, evaluated lazily and cached."""
    def __init__(self, beam_parameters_path: str):
        with open(beam_parameters_path) as f:
            tab = json.load(f)["proton"]["energy_table"]
        self.E = np.array([r["energy_mev"] for r in tab], float)
        self.sE = np.array([r["sigma_energy_mev"] for r in tab], float)
        assert len(self.E) == N_ENERGIES
        self._curves = {}

    def _get(self, e_mev: float, tol: float = 1e-3):
        key = round(float(e_mev), 4)
        c = self._curves.get(key)
        if c is None:
            i = int(np.argmin(np.abs(self.E - e_mev)))
            if abs(self.E[i] - e_mev) > tol:
                raise KeyError(f"energy {e_mev} is not in the table "
                               f"(nearest {self.E[i]})")
            c = _dd_curve(float(self.E[i]), float(self.sE[i]))
            self._curves[key] = c
        return c

    def channel(self, wed_bev: np.ndarray, e_mev: float) -> np.ndarray:
        """C5: the curve sampled at each voxel's radiological depth.

        Beyond the range the prior is zero; shallower than the first tabulated
        depth it is held at the entrance value.
        """
        z, dd = self._get(e_mev)
        return np.interp(wed_bev, z, dd, left=dd[0], right=0.0).astype(np.float32)
