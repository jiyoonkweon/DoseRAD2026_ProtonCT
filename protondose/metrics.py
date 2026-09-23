"""The challenge's beam-level metrics, as in the organizers' evaluator
(https://github.com/DoseRAD2026/evaluation-setup), plus the reference cutoff.
"""
from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk

from .config import GT_CUTOFF_REL


def masked_beam_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """MAE over voxels at or above 10 % of the reference maximum, relative to it."""
    gt_max = float(np.max(gt))
    if gt_max <= 0:
        return float("nan")
    mask = gt >= 0.1 * gt_max
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - gt[mask])) / gt_max)


def idd_curve(dose: np.ndarray, direction: np.ndarray, spacing) -> np.ndarray:
    """Dose integrated over planes perpendicular to the beam, against depth."""
    if abs(direction[2]) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")
    plane = dose.sum(axis=0, dtype=np.float64)
    ny, nx = plane.shape
    sx, sy = float(spacing[0]), float(spacing[1])
    # Square grid, one sample per voxel, wide enough for the diagonal, so the
    # curve length does not depend on the beam angle.
    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    source = sitk.GetImageFromArray(plane)
    source.SetSpacing((sx, sy))
    # Rotating by the beam angle about the plane centre lays the beam along the
    # first output axis.
    to_beam = sitk.Euler2DTransform()
    to_beam.SetCenter((0.0, 0.0))
    to_beam.SetAngle(math.atan2(direction[1], direction[0]))
    to_beam.SetTranslation(((nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0))
    aligned = sitk.Resample(source, (n, n), to_beam, sitk.sitkLinear,
                            (-(n - 1) * step / 2.0,) * 2, (step, step),
                            (1.0, 0.0, 0.0, 1.0), 0.0, sitk.sitkFloat64)
    return sitk.GetArrayFromImage(aligned).sum(axis=0)


def idd_curve_distance(pred: np.ndarray, gt: np.ndarray, direction: np.ndarray,
                       spacing) -> float:
    """RMS difference of the two IDD curves, relative to the reference peak."""
    idd_pred = idd_curve(pred, direction, spacing)
    idd_gt = idd_curve(gt, direction, spacing)
    peak = float(np.max(idd_gt))
    if peak <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((idd_pred / peak - idd_gt / peak) ** 2)))


def score(pred: np.ndarray, gt: np.ndarray, direction: np.ndarray, spacing):
    """(masked beam MAE, IDD distance) after zeroing both volumes at or below
    GT_CUTOFF_REL of the reference maximum."""
    c = GT_CUTOFF_REL * float(gt.max())
    gt = np.where((gt > 0) & (gt <= c), 0.0, gt)
    pred = np.where((pred > 0) & (pred <= c), 0.0, pred)
    return masked_beam_mae(pred, gt), idd_curve_distance(pred, gt, direction, spacing)
