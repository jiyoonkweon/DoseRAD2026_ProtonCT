"""Dataset layout and beamlet records.

    <data>/                          e.g. DoseRAD2026/proton/training
        beam_parameters.json         energy table and HU-to-density table
        <patient>/
            <patient>.json           plan: beams -> rays -> beamlets
            image/ct.mha
            dose/Dose_B<beam>_R<ray>_L<beamlet>.mha
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Beamlet:
    patient: str
    beam: int
    ray: int
    index: int                  # beamlet index within the ray
    energy: float               # MeV
    ray_source: tuple           # mm, patient frame
    ray_target: tuple           # mm, patient frame; origin of the BEV box

    @property
    def name(self) -> str:
        """Cache file stem."""
        return f"B{self.beam:02d}_R{self.ray:02d}_L{self.index}"

    @property
    def dose_file(self) -> str:
        """Reference dose file name in the dataset."""
        return f"Dose_B{self.beam}_R{self.ray}_L{self.index}.mha"


def beam_parameters_path(data_dir) -> Path:
    return Path(data_dir) / "beam_parameters.json"


def ct_path(data_dir, patient: str) -> Path:
    return Path(data_dir) / patient / "image" / "ct.mha"


def dose_path(data_dir, beamlet: Beamlet) -> Path:
    return Path(data_dir) / beamlet.patient / "dose" / beamlet.dose_file


def read_plan(plan_file, patient: str = "") -> list[Beamlet]:
    """All beamlets of a plan, in (beam, ray, beamlet) order."""
    plan = json.loads(Path(plan_file).read_text())
    out = []
    for beam in plan["beams"]:
        for ray in beam["rays"]:
            for bl in ray["beamlets"]:
                out.append(Beamlet(patient, int(beam["beam_idx"]), int(ray["ray_idx"]),
                                   int(bl["beamlet_idx"]), float(bl["energy"]),
                                   tuple(map(float, ray["ray_source"])),
                                   tuple(map(float, ray["ray_target"]))))
    return sorted(out, key=lambda b: (b.beam, b.ray, b.index))


def list_patients(data_dir) -> list[str]:
    """Patient directories that hold a plan file."""
    d = Path(data_dir)
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / f"{p.name}.json").is_file())


def patient_beamlets(data_dir, patient: str) -> list[Beamlet]:
    return read_plan(Path(data_dir) / patient / f"{patient}.json", patient)


def energy_quantiles(beamlets: list[Beamlet], per_patient: int) -> list[Beamlet]:
    """`per_patient` beamlets of each patient at the midpoint energy quantiles.

    Used for per-epoch validation. Picking by plan order instead would skew the
    sample, since plan order correlates with energy.
    """
    by_pid = {}
    for b in beamlets:
        by_pid.setdefault(b.patient, []).append(b)
    out = []
    for g in by_pid.values():
        g = sorted(g, key=lambda b: (b.energy, b.beam, b.ray, b.index))
        pos = ((np.arange(per_patient) + 0.5) / per_patient * len(g)).astype(int)
        out += [g[j] for j in np.unique(np.clip(pos, 0, len(g) - 1))]
    return sorted(out, key=lambda b: (b.patient, b.beam, b.ray, b.index))


def energy_spread(beamlets: list[Beamlet], per_patient: int) -> list[Beamlet]:
    """`per_patient` beamlets of each patient evenly spaced over its energy-sorted
    list, endpoints included. 40 per patient is the 440-beamlet validation set."""
    by_pid = {}
    for b in beamlets:
        by_pid.setdefault(b.patient, []).append(b)
    out = []
    for g in by_pid.values():
        g = sorted(g, key=lambda b: (b.energy, b.dose_file))
        pos = np.unique(np.linspace(0, len(g) - 1, per_patient).round().astype(int))
        out += [g[j] for j in pos]
    return out


def load_split(split_file, data_dir) -> tuple[list[str], list[str]]:
    """(train, val) patient ids from a split file, restricted to those present."""
    split = json.loads(Path(split_file).read_text())
    have = set(list_patients(data_dir))
    train = [p for p in split["train"] if p in have]
    val = [p for p in split["val"] if p in have]
    missing = len(split["train"]) + len(split["val"]) - len(train) - len(val)
    if missing:
        print(f"[split] {missing} patients of {split_file} are not in {data_dir}; "
              f"using {len(train)} train / {len(val)} val", flush=True)
    return train, val
