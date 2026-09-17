# DoseRAD2026 — Proton Dose on CT

Beamlet-wise proton dose prediction from a patient CT, in place of Monte Carlo
transport. This is the submission to
[DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/),
**Task 3 (proton, CT)**, which placed **2nd overall**.

Each beamlet is predicted independently, so a full plan dose is a linear
combination of beamlets under arbitrary monitor-unit weights — no retraining or
re-inference during plan optimization. Inference is fast enough to matter: the
challenge's runtime metric — the time for one CT and 500 beamlets — came out at
17.2 s, the fastest of all submissions.

The repository contains the submitted algorithm and its trained weights, with
every value fixed to the one that was submitted.

## Results

Final test leaderboard, per-metric position in parentheses:

| Metric | Value | Position |
| --- | --- | ---: |
| Beam MAE | 0.0061 ± 0.0014 | 3 |
| IDD curve distance | 0.0047 ± 0.0012 | 5 |
| Stratified plan-level MAE | 0.0055 ± 0.0015 | 5 |
| Local gamma index (1 % / 1 mm) | 96.7438 ± 2.7703 % | 4 |
| DVH-based clinical score | 0.3095 ± 0.2100 | 2 |
| Runtime | 17.2345 s | 1 |
| **Mean position** | **3.00** | **2nd overall** |

Runtime carries double weight, so the mean is over seven terms:
(3 + 5 + 5 + 4 + 2 + 1 + 1) / 7.

## Method

The CT is resampled into a canonical beam's-eye-view box aligned with the ray —
361 × 33 × 33 voxels at 2 mm isotropic spacing — and five physics channels are
built on it: mass density, spot fluence, radiological depth, normalized nominal
energy, and an analytic Bragg-curve prior. A 3D residual U-Net with a
depth-sequential ConvLSTM bottleneck (6,453,261 parameters) predicts the dose,
which is then clamped, resampled back to the patient grid, rescaled to absolute
Gy, and cut off below the plan's per-beamlet threshold. One model serves both
anatomical sites; no ensembling and no test-time augmentation.

## Installation

Requires a CUDA device and about 4.2 GiB of
GPU memory.

```bash
git clone https://github.com/jiyoonkweon/DoseRAD2026_ProtonCT.git
cd DoseRAD2026_ProtonCT

# PyTorch first, with the build matching your CUDA version:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

That installs PyTorch, NumPy and SimpleITK, which is everything the model needs.
`requirements-container.txt` adds the three packages the submission container
uses on top; Docker is not needed otherwise.

Verified on Python 3.10.13 with PyTorch 2.1.0 (CUDA 12.1, cuDNN 8.9.2), NumPy
1.26.0 and SimpleITK 2.5.6, and inside the submission image
(`pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime`), both on an NVIDIA
A100-PCIE-40GB with driver 580.126.09. The two combinations produce the same
output, so the code is not tied to one PyTorch release.

## Usage

### Command line

Beamlets are read from a JSON file, never typed in: either a patient plan from
the training set (`1THB016/1THB016.json`) or the challenge's beam-level
metadata. Both carry the ray endpoints and the nominal energy, which is all a
beamlet needs. `example/beamlets.json` holds four beamlets of patient 1THB016
in the second schema. The CT is not redistributed here, so point `--ct` at your
own copy of the dataset
([Zenodo](https://doi.org/10.5281/zenodo.19347848)):

```bash
python predict.py \
    --ct /path/to/DoseRAD2026/proton/training/1THB016/image/ct.mha \
    --beamlets example/beamlets.json \
    --out dose/
```

This writes one 3-D MetaImage per beamlet, in absolute Gy, on the patient grid,
named `dose_000.mha`, `dose_001.mha`, ... in the order the beamlets appear in
the file. Give it a whole plan file and it predicts every beamlet in that plan.

### As a library

```python
from pathlib import Path
import SimpleITK as sitk
from protondose import (Beamlet, NORM_SCALE, build_model, make_density,
                        predict_beamlets, warmup)
from protondose.geometry import patient_phys_coords

model = build_model(Path("model"))
warmup(model)                                   # cuDNN autotuning, done once

ct = sitk.ReadImage("CT.mha")
density = make_density(ct, model["hlut"], "cuda")
phys = patient_phys_coords(ct, device="cuda")

beamlet = Beamlet(ray_source, ray_target, energy_mev)
for sub, bbox in predict_beamlets(model, [beamlet], ct, density, phys):
    dose_gy = sub * NORM_SCALE
```

`predict_beamlets` yields the patient-grid sub-volume covered by the beam's-eye
view, together with its bounding box `(x0, x1, y0, y1, z0, z1)`; everything
outside is exactly zero. Pass several beamlets at once to batch them —
beamlets that share a ray reuse the same resampling grids. `predict.py` shows
how to expand a sub-volume to a full volume and write it out.

### The submission container

`Dockerfile` builds the image that was submitted, in case you want to reproduce
the leaderboard runtime or re-submit. It wraps the same computation in the
challenge's `/health` and `/invoke` API and writes the 4-D stacked MetaImage the
platform expects, one file per output slot. Running the model does not require
it.

## Evaluation

`evaluate.py` scores predictions against the reference Monte Carlo dose that
ships with the training set, using the challenge's two beam-level metrics:
masked beam MAE and IDD curve distance. Point it at a patient directory and it
reads the beamlets from that patient's own plan file, predicts them, and
compares each one against `dose/Dose_B<beam>_R<ray>_L<beamlet>.mha`:

```bash
python evaluate.py --case /path/to/DoseRAD2026/proton/training/1THB016 --limit 8
```

```
beamlet                     E (MeV)       MAE       IDD
Dose_B0_R0_L0.mha            120.43    0.0076    0.0123
Dose_B5_R2_L0.mha             83.13    0.0095    0.0045
Dose_B10_R4_L0.mha            60.13    0.0085    0.0048
...
8 beamlets of 1THB016
  masked beam MAE      0.0083 +/- 0.0011
  IDD curve distance   0.0061 +/- 0.0024
```

`--limit` takes that many beamlets spread evenly through the plan, so a quick
check covers a range of energies and gantry angles; `--limit 0` scores all of
them, which for the example case means 1,080 beamlets and as many reference
volumes to read. `--csv FILE` writes the per-beamlet table. Both metrics are
reimplemented here to give identical results to the organizers' evaluator; that
one, which also covers the plan-level metrics, is at
[DoseRAD2026/evaluation-setup](https://github.com/DoseRAD2026/evaluation-setup).

Note that repeated runs are usually but not always bit-identical, because cuDNN
picks its convolution algorithm by measured speed. The variation is around
10⁻⁴ of the beamlet maximum and does not move these metrics.

## Runtime

`benchmark.py` measures inference speed the way the challenge scores it. The
leaderboard fits

```
T = t_fix + N_images * t_img + N_beamlets * t_dose
```

to the wall time of every job it runs, and reports `T` at one image and 500
beamlets. The script measures the two terms that scale, on a single patient,
without the container:

```bash
python benchmark.py --case /path/to/DoseRAD2026/proton/training/1THB016
```

```
device            NVIDIA A100-PCIE-40GB
case              1THB143  CT 590x518x146
beamlets timed    184  (200 run, first 16 discarded), best of 2 passes

  startup weights + warm-up              2.76 s   (/health; the platform times /invoke)
  t_img   CT -> density, coordinates     1.03 s
  t_dose  per beamlet                    26.6 ms

  runtime for 1 image and 500 beamlets
    t_img + 500 * t_dose = 14.3 s
```

Loading the weights and warming up cuDNN are reported but not added in: the
container does both while answering `/health`, before the platform starts timing
`/invoke`. What is timed per beamlet is the whole path the platform pays for —
channel construction, the network, back-projection to the patient grid, the copy
back to the host, the rescaling to absolute Gy, and the cutoff — everything
except writing the result out.

On one A100, three patients read `t_dose` between 22.7 and 26.6 ms and a runtime
between 12.0 and 14.3 s, against the 17.2345 s the platform returned. The
evaluation hardware is an A10G, which accounts for most of what is left.

Two things move this number more than the model does.

**Where the output goes.** The container writes a compressed 4-D MetaImage per
output slot. Measured with that directory on a network filesystem, the write ran
at 28.5 MiB/s, the queue backed up, and the reported cost per beamlet climbed
from 34 to 98 ms over the slots of one job while prediction itself stayed flat at
1.1 to 2.2 s per slot — `t_dose` came out 2.7x too high. On local storage the
write drops to about 0.2 s per slot and the two agree again. `benchmark.py` does
not write at all, so it is not exposed to this; a container-based measurement is,
and its output directory should be kept off a network filesystem.

**Which beamlets.** Two beamlets that share a ray reuse its resampling grids, so
the second is cheaper than the first. Plans arrive in beam and ray order, so the
platform gets that reuse. The script keeps every chosen ray whole and draws rays
evenly across the plan, which preserves the reuse while still spanning the plan's
gantry angles and energies; sampling individual beamlets breaks it and reads
several percent slower per beamlet.

## Repository layout

```
predict.py               command line: CT + beamlets -> dose
evaluate.py              score predictions against the reference Monte Carlo dose
benchmark.py             measure runtime the way the leaderboard computes it
Dockerfile               submission image, as submitted
protondose/
  geometry.py            beam's-eye-view box, beamlet frame, resampling
  channels.py            input channels C1-C4, and the HU-to-density table
  bragg.py               input channel C5, the analytic Bragg-curve prior
  network.py             the Dose3DNet architecture
  inference.py           weight loading, warm-up, batched beamlet prediction
  challenge_io.py        challenge metadata in, 4-D compressed MetaImage out
  server.py              /health and /invoke endpoints; the container entry point
model/
  model.pt               trained weights
  beam_parameters.json   energy table and HU-to-density calibration
example/
  beamlets.json          four beamlets, in the challenge metadata schema
```

`inference.py` is the method. `challenge_io.py` and `server.py` are the
packaging the challenge required, and nothing in `inference.py` depends on them.

## Training

The weights were trained from scratch on
the DoseRAD2026 training split — no external data, no pretrained weights — in
two stages totalling about 38 GPU-hours on one NVIDIA A100-PCIE-40GB. The loss
is evaluated on the patient grid, where scoring happens, rather than on the
network's own grid.

## Data

The model was trained on the DoseRAD2026 public training set, CC BY-NC 4.0, DOI
[10.5281/zenodo.19347848](https://doi.org/10.5281/zenodo.19347848). None of it is
redistributed here. Thanks to the organizers for providing it.

## Citation

```bibtex
@software{kweon2026protonct,
  author = {Kweon, Jiyoon and Park, Hyunjin},
  title  = {Fast Proton Dose Calculation via Beam's-Eye-View ConvLSTM},
  year   = {2026},
  url    = {https://github.com/jiyoonkweon/DoseRAD2026_ProtonCT}
}
```

Please also cite the dataset:

```bibtex
@article{xiao2026doserad,
  author  = {Xiao, F. and Delopoulos, N. and Wahl, N. and others},
  title   = {{DoseRAD2026} Challenge dataset: {AI} accelerated photon and
             proton dose calculation for radiotherapy},
  journal = {arXiv preprint arXiv:2604.12778},
  year    = {2026}
}
```

## Contact

Questions and bug reports are welcome as
[GitHub issues](https://github.com/jiyoonkweon/DoseRAD2026_ProtonCT/issues), or
by email to jiyoonkweon@skku.edu.

## License

MIT, see [LICENSE](LICENSE).
