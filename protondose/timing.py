"""Measure inference runtime the way the DoseRAD2026 leaderboard computes it.

The challenge fits a non-negative linear model to the wall time of every job it
runs,

    T = t_fix + N_images * t_img + N_beamlets * t_dose,

and reports T at one image and 500 beamlets. :func:`measure_runtime` measures
the two terms that scale, on one patient, so the number can be reproduced
without the submission container or the platform.

Loading the weights and warming up cuDNN are not part of either term: the
container does both while answering /health, before the platform starts timing
/invoke. They are reported separately. The fitted t_fix is what remains of the
per-invoke overhead, a few tenths of a second, which is not reproduced here.

Timing needs a different sample of beamlets than scoring does, which is why it
gets its own pass over the plan rather than sharing the scoring loop:

* Two beamlets that share a ray reuse its resampling grids, so the second is
  cheaper. Plans reach the platform in beam and ray order and get that reuse,
  so :func:`select_rays` keeps every chosen ray whole. Scoring instead spreads
  single beamlets across the plan to cover its energies, which breaks the reuse
  and reads several percent slower per beamlet.
* The scoring loop reads a reference dose volume per beamlet. Timing the
  prediction inside it would measure the disk.

Writing predictions out is not timed either. On the platform that cost overlaps
with prediction and is small, about 0.1 s per output slot, but it is set by the
storage behind the output directory: measured with that directory on a network
filesystem, t_dose came out 2.7x too high. Keep the output of any
container-based measurement on local storage for the same reason.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from . import inference
from .geometry import patient_phys_coords

LEADERBOARD_BEAMLETS = 500
LEADERBOARD_IMAGES = 1
DEFAULT_CUTOFF_GY = 1e-6


@dataclass
class Runtime:
    """What one measurement produced. Times in seconds, t_dose in seconds too."""
    device: str
    case: str
    n_timed: int
    t_startup: float
    t_img: float
    t_dose: float

    @property
    def total(self) -> float:
        """Runtime at the leaderboard's load: one image and 500 beamlets."""
        return LEADERBOARD_IMAGES * self.t_img + LEADERBOARD_BEAMLETS * self.t_dose

    def report(self) -> str:
        return (
            f"runtime ({self.n_timed} beamlets, whole rays across the plan)\n"
            f"  startup weights + warm-up          {self.t_startup:8.2f} s   "
            f"(/health; the platform times /invoke)\n"
            f"  t_img   CT -> density, coordinates {self.t_img:8.2f} s\n"
            f"  t_dose  per beamlet                {self.t_dose * 1000:8.1f} ms\n"
            f"  {LEADERBOARD_IMAGES} image + {LEADERBOARD_BEAMLETS} beamlets"
            f"  =  {self.total:.1f} s\n"
            f"  the platform adds its own per-invoke t_fix, a few tenths of a second"
        )


def plan_rays(case: Path) -> list[list[inference.Beamlet]]:
    """The patient's rays in plan order, each holding its own beamlets."""
    plan_file = case / f"{case.name}.json"
    if not plan_file.is_file():
        raise SystemExit(f"no plan file at {plan_file}; --case wants a patient "
                         "directory from the training set")
    plan = json.loads(plan_file.read_text())
    return [[inference.Beamlet(ray["ray_source"], ray["ray_target"], bl["energy"])
             for bl in ray["beamlets"]]
            for beam in plan["beams"] for ray in beam["rays"]]


def select_rays(rays: list[list[inference.Beamlet]], n: int = 0,
                contiguous: bool = False) -> list[inference.Beamlet]:
    """n beamlets, keeping every chosen ray whole. n <= 0 takes the whole plan.

    Rays are drawn evenly across the plan so the sample spans its gantry angles
    and energies. Taking the first n beamlets instead covers only the first few
    beams, whose ranges need not be typical; ``contiguous`` does that.
    """
    flat = [b for r in rays for b in r]
    if n <= 0 or n >= len(flat):
        return flat
    if contiguous:
        return flat[:n]
    per_ray = max(1, round(len(flat) / len(rays)))
    picks = np.linspace(0, len(rays) - 1, max(1, n // per_ray)).round().astype(int)
    out: list[inference.Beamlet] = []
    for i in dict.fromkeys(picks.tolist()):
        out.extend(rays[i])
    return out


def _timed(fn):
    """Wall time of fn(), with the GPU queue drained on both sides."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def measure_runtime(case: Path, model_dir: Path, n: int = 0,
                    warmup_beamlets: int = 16, repeats: int = 1,
                    cutoff: float = DEFAULT_CUTOFF_GY,
                    contiguous: bool = False,
                    model: dict | None = None,
                    t_startup: float = float("nan")) -> Runtime:
    """Measure t_img and t_dose on one patient.

    ``n`` defaults to 0, the patient's whole plan. t_dose is a marginal cost --
    the challenge's fit puts one-off costs in t_fix -- so it is best read over as
    many beamlets as there are. A short pass instead spreads the first
    allocations, which size the resampling grids and host buffers per ray, over
    too few beamlets: one patient reads 33.8 ms at n=200, 29.3 ms at n=500 and
    29.1 ms across all 1,080, so the estimate has converged well before the plan
    runs out.

    ``repeats`` above 1 reports the fastest pass, which is warmer than anything
    a freshly started container sees; 1 is the honest default.

    Pass ``model`` to reuse one already built, as evaluate.py does; ``t_startup``
    then carries however long that took, for the report. Left out, the model is
    built here and the startup cost measured with it.
    """
    if not torch.cuda.is_available():
        raise SystemExit("runtime measurement needs a CUDA device")
    if 0 < n <= warmup_beamlets:
        raise SystemExit(f"n must exceed the {warmup_beamlets} discarded beamlets")

    ct_file = case / "image" / "ct.mha"
    if not ct_file.is_file():
        raise SystemExit(f"no CT at {ct_file}")

    items = select_rays(plan_rays(case), n, contiguous)
    n_timed = len(items) - warmup_beamlets

    if model is None:
        def _load():
            m = inference.build_model(Path(model_dir), device="cuda")
            inference.warmup(m)
            return m
        model, t_startup = _timed(_load)

    def _image():
        img = sitk.ReadImage(str(ct_file))
        dens = inference.make_density(img, model["hlut"], "cuda")
        return img, dens, patient_phys_coords(img, device="cuda")
    (image, density, phys), t_img = _timed(_image)

    # The leading beamlets are discarded: the first batch of a fresh ray pays for
    # grid construction that later beamlets on the same ray reuse, and the
    # allocator is still growing its pools.
    per_pass = []
    for _ in range(repeats):
        gen = inference.predict_beamlets(model, items, image, density, phys)
        t0 = None
        for i, (sub, _bbox) in enumerate(gen):
            # In place, on the pinned buffer the generator hands over, which is
            # what the container does before passing it to the compressor. An
            # extra copy here would time the copy, not the model.
            sub *= inference.NORM_SCALE
            inference.apply_cutoff(sub, cutoff)
            if i == warmup_beamlets - 1:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
        torch.cuda.synchronize()
        per_pass.append((time.perf_counter() - t0) / n_timed)

    return Runtime(device=torch.cuda.get_device_name(0), case=case.name,
                   n_timed=n_timed, t_startup=t_startup, t_img=t_img,
                   t_dose=min(per_pass))
