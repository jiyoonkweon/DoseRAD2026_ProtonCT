"""C5: analytic Bragg-curve prior in water (Bortfeld 1997).

Per nominal energy the depth-dose curve is evaluated once, convolved with the
range spread, normalized to its peak and cached; the channel is that curve
looked up at every voxel's water-equivalent depth (C3).

    R0      = 0.0022 * E^1.77                       range, cm
    sigma_R = sqrt(sigma_mono^2 + sigma_E->R^2)     sigma_mono = 0.012 * R0^0.935
    D(z)   ~ 17.93 (R0 - z)^-0.435 + 0.444 (R0 - z)^0.565, convolved with sigma_R
"""
from __future__ import annotations

import numpy as np

ALPHA_CM = 0.0022
P_EXP = 1.77
Z_STEP_CM = 0.02           # output table resolution
N_PER_SIGMA = 40           # internal grid samples per straggling sigma
K_SIGMA = 6.0              # convolution kernel half-width, in sigma


def _range_and_spread(e_mev: float, sigma_e_mev: float):
    """(R0, sigma_R) in cm: monoenergetic straggling plus energy spread."""
    R0 = ALPHA_CM * e_mev ** P_EXP
    s_mono = 0.012 * R0 ** 0.935
    s_e = sigma_e_mev * ALPHA_CM * P_EXP * e_mev ** (P_EXP - 1.0)
    return R0, float(np.hypot(s_mono, s_e))


def _cell_avg_pow(R0, z, h, a):
    """Cell average of (R0 - t)^a over [z - h/2, z + h/2] clipped to [0, R0].

    Integrating over the cell rather than point-sampling keeps the singular
    primary term finite at the peak, so no clipping is needed and the peak
    normalization does not depend on the grid spacing. Dividing by the full cell
    width h, not the overlap, makes sum(value) * h the exact integral.
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


def depth_dose(e_mev: float, sigma_e_mev: float):
    """(z_cm, depth_dose) with the peak normalized to 1."""
    R0, s_r = _range_and_spread(e_mev, sigma_e_mev)
    h = min(Z_STEP_CM, s_r / N_PER_SIGMA)          # internal grid, finer than the table
    zi = np.arange(0.0, R0 + (K_SIGMA + 1.0) * s_r + 0.5, h)
    prim = 17.93 * _cell_avg_pow(R0, zi, h, -0.435)      # primary protons
    nucl = 0.444 * _cell_avg_pow(R0, zi, h, 0.565)       # nuclear secondaries
    k = int(np.ceil(K_SIGMA * s_r / h))
    kz = np.arange(-k, k + 1) * h
    kern = np.exp(-kz ** 2 / (2 * s_r ** 2))
    kern /= kern.sum()
    dd = _conv_same(prim, kern) + _conv_same(nucl, kern)
    z = np.arange(0.0, R0 + 6 * s_r + 0.5, Z_STEP_CM)
    return z, np.interp(z, zi, dd / float(dd.max())).astype(np.float32)


class BraggTable:
    def __init__(self, energy_table):
        self.table = energy_table
        self._curves = {}

    def channel(self, wed_cm: np.ndarray, energy: float) -> np.ndarray:
        """C5. Zero beyond the range; the entrance value above the first sample."""
        c = self._curves.get(energy)
        if c is None:
            i = self.table.index(energy)
            c = depth_dose(float(self.table.E[i]), float(self.table.sigma_energy[i]))
            self._curves[energy] = c
        z, dd = c
        return np.interp(wed_cm, z, dd, left=dd[0], right=0.0).astype(np.float32)
