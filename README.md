# DoseRAD2026 — Proton Dose on CT

Beamlet-wise proton dose prediction from a patient CT, in place of Monte Carlo
transport. This is the model submitted to the
[DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/),
**Task 3 (proton, CT)**, where it placed **2nd overall**.

The repository contains the trained weights and everything needed to rebuild
them: data preparation, the two-stage training, validation and prediction.
Every setting is fixed to the submitted model, in
[`protondose/config.py`](protondose/config.py); the scripts take paths only.

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

Runtime carries double weight, so the mean is over seven terms. The 11
validation patients also served for model selection, so scores on them are not
an unbiased estimate; `validate.py` reports them as a check of the installation.

## Method

**Input.** Each beamlet gets its own box aligned with its ray: 64 mm × 64 mm
laterally, from 380 mm upstream to 340 mm downstream of the ray target, at 2 mm
— 361 × 33 × 33 voxels. The CT is resampled into it and five channels are built
on it:

| | Channel | |
| --- | --- | --- |
| C1 | mass density | official HU-to-density table |
| C2 | spot fluence | Gaussian of the energy's spot sigma; the beams are parallel, so the same at every depth |
| C3 | water-equivalent depth | ∑ρ·dw along the beam, voxel-centred |
| C4 | energy | E / 200.7966 MeV |
| C5 | Bragg prior | analytic Bortfeld curve in water, looked up at C3 |

**Network.** 3D residual U-Net, width 24, three levels with strides
(2,2,2), (2,2,2), (2,1,1), squeeze-and-excitation in every block, and a
unidirectional ConvLSTM that scans the 45 bottleneck slices from upstream to
downstream. 6,453,261 parameters. The output is the dose on the box divided by
a global constant (1.1187 × 10⁻³ Gy); inference clamps it at zero,
back-projects it to the patient grid and multiplies the constant back.

**Loss.** Computed on the patient grid, where the challenge scores, not on the
box. For every beamlet, reference-dose voxels are grouped into three dose bands
(≥10 %, 1–10 %, 0.1–1 % of the maximum), up to 48k / 24k / 16k of them are
stored with their box coordinates, and each step draws 32k / 16k / 12k. The
loss is a band-weighted MSE at those points (weights 1 / 0.25 / 0.15; the
lowest band is compared as 2×2×2 block means), plus an MSE on the box that fades
over 120 epochs, an anchor on voxels below 0.1 % of the maximum (weight 2.5),
and an integrated-depth-dose term (weight ramped to 0.5). Details are in
[`protondose/loss.py`](protondose/loss.py) and
[`protondose/pools.py`](protondose/pools.py).

**Training.** AdamW (weight decay 1e-4), cosine learning rate, gradient
clipping 1.0, batch 32, fp32, EMA of the weights (0.9995), no augmentation.
Every epoch is validated on 8 beamlets per validation patient, with the EMA
weights, on the patient grid with the official metrics; the score is
masked beam MAE + IDD distance.

| | Stage 1 | Stage 2 (released) |
| --- | --- | --- |
| Start | random initialization | stage 1 `best.pt` |
| Epochs / learning rate | 60 / 3e-4 | 16 / 5e-5 |
| IDD term ramp starts at | epoch 10 | epoch 0 |
| Early stopping | patience 10 | none |
| Best epoch | 52 | **14** |
| Validation score | 0.013150 | **0.012895** |
| Time on one L40S 46 GB | ≈ 31 h | ≈ 8.3 h |

Split: 64 training and 11 validation patients, stratified by site,
[`splits/train_val.json`](splits/train_val.json). No external data and no
pretrained weights were used.

## Installation

Needs a CUDA GPU. Training at batch 32 in fp32 uses about 38 GB of GPU memory;
inference fits in a few GB.

```bash
git clone https://github.com/jiyoonkweon/DoseRAD2026_ProtonCT.git
cd DoseRAD2026_ProtonCT

# PyTorch first, with the build matching your CUDA driver:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The released model was trained with Python 3.11.15, PyTorch 2.13.0 (CUDA 13.0),
NumPy 1.26.4 and SimpleITK 2.5.5 on an NVIDIA L40S (46 GB). The figures quoted
in this README for memory, cache building, validation and runtime were measured
on an NVIDIA A100-PCIE-40GB, which the code also runs on.

## Data

Download the DoseRAD2026 proton training set from
[Zenodo](https://doi.org/10.5281/zenodo.19347848) and place it as

```
data/DoseRAD2026/proton/training/
├── beam_parameters.json          energy table, HU-to-density table
├── 1ABB006/
│   ├── 1ABB006.json              plan: beams → rays → beamlets
│   ├── image/ct.mha
│   └── dose/Dose_B<beam>_R<ray>_L<beamlet>.mha
└── ...                           75 patients, 1,080 beamlets each
```

Any other location works: pass it as `--data`. The MR images are not used.

## Usage

### 1. Prepare the training cache

```bash
python prepare_data.py --data data/DoseRAD2026/proton/training --cache cache
```

Writes one file per beamlet to `cache/<patient>/`: the density and reference
dose resampled onto the beamlet's box, and its supervision points. About 5
minutes and 1.1 GB per patient on one A100 — 6 hours and 80 GB for all 75.
For several GPUs, run one process per GPU with `--shard i --num-shards n`.
Finished beamlets are skipped, so an interrupted run can be restarted.

### 2. Train

```bash
python train.py --stage 1 --data data/DoseRAD2026/proton/training --cache cache
python train.py --stage 2 --data data/DoseRAD2026/proton/training --cache cache
```

Stage 1 writes `runs/stage1/`, and stage 2 starts from `runs/stage1/best.pt`
and writes `runs/stage2/`. Each directory gets `best.pt` (EMA weights of the
best epoch, the file to use), `last.pt` (full state; continue with `--resume`)
and `log.csv`. `--split splits/example.json` trains and validates on a single
patient, to check the pipeline end to end.

### 3. Validate

```bash
python validate.py --data data/DoseRAD2026/proton/training
```

Runs the full CT-to-dose inference on 40 beamlets per validation patient,
spread over the energy range, and reports masked beam MAE and IDD curve
distance against the reference dose. `--weights runs/stage2/best.pt` scores
your own training; `--per-patient 0` takes every beamlet; `--csv FILE` writes
the per-beamlet table. With the released weights it prints

```
1ABB030  n=  40  masked MAE 0.00608  IDD 0.00495
1ABB031  n=  40  masked MAE 0.00786  IDD 0.00850
...
1THB143  n=  40  masked MAE 0.00851  IDD 0.00627

440 beamlets, 11 patients
  masked beam MAE      0.00720 +/- 0.00289
  IDD curve distance   0.00579 +/- 0.00336
```

up to cuDNN algorithm choice, which moves the last digit. These numbers check
that the installation reproduces the released model; the performance of the
method is the leaderboard result above.

### 4. Predict

```bash
python predict.py --data data/DoseRAD2026/proton/training --patient 1THB063 --out pred/
```

Writes the dose of every beamlet in the patient's plan to
`pred/Dose_B<beam>_R<ray>_L<beamlet>.mha`, in Gy on the CT grid. `--limit N`
predicts only N beamlets spread over the plan.

The output is the raw prediction, clamped at zero. The challenge additionally
zeroes every voxel below the per-beamlet `minimum_cutoff` given in its test
metadata; the training plans carry no such field, so `predict.py` leaves that
step to the caller.

## Repository layout

```
prepare_data.py          dataset -> training cache
train.py                 two-stage training, per-epoch validation
validate.py              official beam-level metrics on the validation patients
predict.py               patient plan -> dose .mha per beamlet
protondose/
  config.py              every fixed setting of the released model
  data.py                dataset layout, plan reading, beamlet selection
  geometry.py            beamlet frame, BEV box, resampling both ways
  channels.py            input channels C1-C4 and the channel builder
  bragg.py               input channel C5, the Bragg-curve prior
  network.py             Dose3DNet
  pools.py               patient-space supervision points
  loss.py                training loss
  dataset.py             cache files -> training batches
  metrics.py             masked beam MAE and IDD distance
  inference.py           CT + beamlets -> dose on the patient grid
splits/
  train_val.json         64 training / 11 validation patients
  example.json           one patient, for a pipeline check
weights/
  model.pt               released weights (stage 2, epoch 14)
```

## Reproducibility

cuDNN picks convolution algorithms by measured speed, so repeated runs can
differ slightly. Scoring the released weights with `train.py`'s per-epoch
validation gives 0.0128935, against 0.0128949 logged when the checkpoint was
saved; the difference is in the sixth decimal. The draw of supervision points
in `prepare_data.py` and during training is random and not seeded.

The metrics follow the organizers' evaluator,
[DoseRAD2026/evaluation-setup](https://github.com/DoseRAD2026/evaluation-setup),
which also covers the plan-level metrics.

## Data license

The model was trained on the DoseRAD2026 public training set, CC BY-NC 4.0,
DOI [10.5281/zenodo.19347848](https://doi.org/10.5281/zenodo.19347848). None
of it is redistributed here. Thanks to the organizers for providing it.

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
