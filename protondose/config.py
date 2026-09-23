"""Every fixed setting of the released model, in one place.

Nothing here is meant to be tuned: these are the values the submitted weights
were trained with. Scripts take only paths on the command line.
"""

# ---------------------------------------------------------------------------
# Beam's-eye-view (BEV) box, one per beamlet
# ---------------------------------------------------------------------------
# Lateral u, v in [-32, +32] mm; depth w in [-380, +340] mm from the ray target;
# 2 mm isotropic. Arrays are ordered (w, v, u) -> shape (361, 33, 33).
BOX_LAT_MM = 64.0
BOX_W_LO_MM = -380.0
BOX_W_HI_MM = 340.0
SPACING_MM = 2.0

# Reference dose is divided by this constant for training, and predictions are
# multiplied by it. It is the mean per-beamlet maximum over the training set.
NORM_SCALE = 1.1187e-3            # Gy

# ---------------------------------------------------------------------------
# Input channels
# ---------------------------------------------------------------------------
# C1 density, C2 spot fluence, C3 water-equivalent depth, C4 energy, C5 Bragg prior
IN_CHANNELS = 5
E_MAX_MEV = 200.7966              # C4 = E / E_MAX_MEV (highest energy in the table)

# ---------------------------------------------------------------------------
# Network: 3D residual U-Net with a depth-sequential ConvLSTM bottleneck
# ---------------------------------------------------------------------------
WIDTH = 24                        # channels at full resolution, doubled per level
DEPTH = 3                         # downsampling levels
STRIDES = ((2, 2, 2), (2, 2, 2), (2, 1, 1))   # (w, v, u); the last one is depth-only
DROPOUT = 0.1                     # channel dropout after the bottleneck

# ---------------------------------------------------------------------------
# Patient-space supervision points (built once by prepare_data.py)
# ---------------------------------------------------------------------------
# Reference-dose voxels are grouped by dose relative to the beamlet maximum.
# Per band: (name, low, high, pooled, loss weight, pool size, drawn per step).
# "pooled" bands are compared as means over 2x2x2 patient-voxel blocks.
BANDS = (
    ("hot", 0.10, 1.01, False, 1.00, 48000, 32000),    # = official masked-MAE region
    ("mid", 0.01, 0.10, False, 0.25, 24000, 16000),
    ("low", 0.001, 0.01, True, 0.15, 16000, 12000),
)
POOL_BLOCK_VOXELS = 2
POOL_PAD = 120000                 # pools are zero-padded to this length for batching
# Per-band loss normalizers: mean squared target of each band over the training set.
BAND_SCALE = (0.09781, 0.0020783, 1.608e-05)

# ---------------------------------------------------------------------------
# Loss (see protondose/loss.py)
# ---------------------------------------------------------------------------
BEV_WEIGHT = 0.1                  # w_bev(epoch) = BEV_WEIGHT * (1 - epoch / BEV_ANNEAL_EPOCHS)
BEV_ANNEAL_EPOCHS = 120
COLD_WEIGHT = 2.5                 # BEV voxels below COLD_THRESHOLD * max are anchored
COLD_THRESHOLD = 0.001
IDD_WEIGHT = 0.5                  # ramped linearly from the stage's idd_ramp_start
# Reference dose at or below this fraction of its maximum is set to zero, in
# the training targets and in validation. The organizers' reference dose has
# the same cutoff applied.
GT_CUTOFF_REL = 0.00088

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4               # AdamW
GRAD_CLIP = 1.0
EMA_DECAY = 0.9995                # validation and best.pt use the EMA weights
MIN_IMPROVEMENT = 1e-5            # a validation score must drop by more than this

# Two stages. Stage 2 starts from the best stage-1 checkpoint; the released
# weights are its epoch 14.
STAGES = {
    1: dict(epochs=60, lr=3e-4, idd_ramp_start=10, patience=10),
    2: dict(epochs=16, lr=5e-5, idd_ramp_start=0, patience=0),   # 0 = no early stop
}

# Per-epoch validation: this many beamlets per validation patient, spread over
# that patient's energies. Score = masked beam MAE + IDD distance.
VAL_BEAMLETS_PER_PATIENT = 8

# Data loading
NUM_WORKERS = 12
PREFETCH_FACTOR = 6
