#!/usr/bin/env python3
"""Train the model in two stages, exactly as the released weights were.

    python train.py --stage 1 --data data/DoseRAD2026/proton/training --cache cache
    python train.py --stage 2 --data data/DoseRAD2026/proton/training --cache cache

Stage 1 trains from scratch; stage 2 fine-tunes from stage 1's best.pt. Every
hyperparameter is fixed in protondose/config.py; the command line takes paths
only. Each epoch is validated on the validation patients with the EMA weights,
scored as masked beam MAE + IDD distance on the patient grid; best.pt holds the
EMA weights of the best epoch, last.pt the full state for --resume.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protondose import config as C
from protondose import data
from protondose.channels import ChannelBuilder
from protondose.dataset import BeamletDataset
from protondose.geometry import (beamlet_frame, make_patient_grid, patient_bbox,
                                 patient_phys_coords, resample_bev_to_patient)
from protondose.inference import to_full
from protondose.loss import PatientSpaceLoss
from protondose.metrics import score
from protondose.network import Dose3DNet
from protondose.pools import draw_points


class EMA:
    """Exponential moving average of the weights."""

    def __init__(self, net, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in net.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, net):
        for k, v in net.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def apply_to(self, net):
        sd = net.state_dict()
        self.backup = {k: sd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))

    def restore(self, net):
        sd = net.state_dict()
        for k, v in self.backup.items():
            sd[k].copy_(v)
        self.backup = {}


def loss_weights(epoch, stage):
    """(w_bev, w_idd) for this epoch."""
    s = C.STAGES[stage]
    w_bev = C.BEV_WEIGHT * max(0.0, 1.0 - epoch / C.BEV_ANNEAL_EPOCHS)
    r = (epoch - s["idd_ramp_start"]) / max(s["epochs"] - 1 - s["idd_ramp_start"], 1)
    return w_bev, C.IDD_WEIGHT * min(max(r, 0.0), 1.0)


@torch.no_grad()
def validate(net, val_ds, data_dir, dev):
    """Mean (masked beam MAE, IDD distance) on the patient grid, as scored."""
    net.eval()
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    maes, idds = [], []
    pid = ct = phys = None
    for b, batch in zip(val_ds.items, tqdm(loader, desc="validate", leave=False)):
        pred_bev = net(batch["input"].to(dev, non_blocking=True)).clamp(min=0)[0, 0]
        if b.patient != pid:
            pid, phys = b.patient, None
            torch.cuda.empty_cache()
            ct = sitk.ReadImage(str(data.ct_path(data_dir, pid)))
            phys = patient_phys_coords(ct, dev)
        frame = beamlet_frame(b.ray_source, b.ray_target)
        bbox = patient_bbox(ct, frame)
        sub = resample_bev_to_patient(pred_bev, make_patient_grid(frame, phys, bbox)) * C.NORM_SCALE
        pred = to_full(sub.cpu().numpy(), bbox, ct)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(data.dose_path(data_dir, b))))
        mae, idd = score(pred, gt.astype(np.float32), frame.w_hat,
                         np.asarray(ct.GetSpacing(), np.float64))
        maes.append(mae)
        idds.append(idd)
    net.train()
    return float(np.nanmean(maes)), float(np.nanmean(idds))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--data", required=True, help="directory holding the patient folders")
    p.add_argument("--cache", required=True, help="output directory of prepare_data.py")
    p.add_argument("--split", default="splits/train_val.json")
    p.add_argument("--out", default=None, help="checkpoint directory (default runs/stage<N>)")
    p.add_argument("--init", default="runs/stage1/best.pt",
                   help="stage 2 only: weights to start from")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    args = p.parse_args()

    stage, cfg = args.stage, C.STAGES[args.stage]
    out = Path(args.out or f"runs/stage{stage}")
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    builder = ChannelBuilder(data.beam_parameters_path(args.data))
    train_pids, val_pids = data.load_split(args.split, args.data)
    if not train_pids or not val_pids:
        raise SystemExit(f"{args.split}: no training or no validation patient in {args.data}")
    train_bl = [b for pid in train_pids for b in data.patient_beamlets(args.data, pid)]
    val_bl = data.energy_quantiles([b for pid in val_pids
                                    for b in data.patient_beamlets(args.data, pid)],
                                   C.VAL_BEAMLETS_PER_PATIENT)
    train_ds = BeamletDataset(args.cache, train_bl, builder, pools=True)
    val_ds = BeamletDataset(args.cache, val_bl, builder, pools=False)
    loader_kw = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=C.PREFETCH_FACTOR)
    loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True,
                        **loader_kw)
    print(f"[stage {stage}] train {len(train_pids)} patients / {len(train_ds)} beamlets, "
          f"val {len(val_pids)} patients / {len(val_ds)} beamlets, "
          f"{cfg['epochs']} epochs at lr {cfg['lr']}", flush=True)

    net = Dose3DNet().to(dev)
    if stage == 2 and not args.resume:
        net.load_state_dict(torch.load(args.init, map_location=dev, weights_only=True)["model"])
        print(f"[stage 2] initialized from {args.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    ema = EMA(net, C.EMA_DECAY)
    loss_fn = PatientSpaceLoss()

    start, best, bad_epochs = 0, float("inf"), 0
    if args.resume:
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=True)
        net.load_state_dict(ck["model"])
        ema.shadow = ck["ema"]
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best, bad_epochs = ck["epoch"] + 1, ck["best"], ck["bad_epochs"]
        print(f"[resume] from epoch {start}, best {best:.6f}", flush=True)

    log_file = out / "log.csv"
    new_log = not log_file.exists() or not args.resume
    log = open(log_file, "w" if new_log else "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["epoch", "lr", "w_bev", "w_idd", "loss", "val_mae", "val_idd",
                         "val_score", "best"])

    for epoch in range(start, cfg["epochs"]):
        loss_fn.bev_weight, loss_fn.idd_weight = loss_weights(epoch, stage)
        lr = sched.get_last_lr()[0]
        net.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        bar = tqdm(loader, desc=f"epoch {epoch}/{cfg['epochs'] - 1}")
        for batch in bar:
            x = batch["input"].to(dev, non_blocking=True)
            target = batch["target"].to(dev, non_blocking=True)
            pred = net(x)

            # The pool holds every hot voxel, so its maximum is the reference maximum.
            cut = C.GT_CUTOFF_REL * batch["values"].amax(dim=1).to(dev, non_blocking=True)
            coords, values, band, block = draw_points(
                *(batch[k].to(dev, non_blocking=True)
                  for k in ("coords", "values", "band", "block")))
            target = torch.where((target > 0) & (target <= cut.view(-1, 1, 1, 1, 1)),
                                 torch.zeros_like(target), target)
            values = torch.where((values > 0) & (values <= cut.view(-1, 1)),
                                 torch.zeros_like(values), values)

            loss, parts = loss_fn(pred, coords, values, band, block, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema.update(net)
            losses.append(parts["total"])
            bar.set_postfix(loss=f"{parts['total']:.4f}", lr=f"{lr:.2e}")
        sched.step()

        ema.apply_to(net)
        mae, idd = validate(net, val_ds, args.data, dev)
        val_score = mae + idd
        improved = val_score < best - C.MIN_IMPROVEMENT
        if improved:
            best, bad_epochs = val_score, 0
            torch.save({"model": net.state_dict(), "stage": stage, "epoch": epoch,
                        "val_score": val_score}, out / "best.pt")
        else:
            bad_epochs += 1
        ema.restore(net)
        torch.save({"model": net.state_dict(), "ema": ema.shadow,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad_epochs": bad_epochs}, out / "last.pt")

        print(f"[val] epoch {epoch}  masked_mae={mae:.5f}  idd={idd:.5f}  "
              f"score={val_score:.6f}" + ("  -> best" if improved else ""), flush=True)
        writer.writerow([epoch, lr, loss_fn.bev_weight, loss_fn.idd_weight,
                         float(np.mean(losses)), mae, idd, val_score, int(improved)])
        log.flush()
        if cfg["patience"] and bad_epochs >= cfg["patience"]:
            print(f"[early stop] no improvement for {cfg['patience']} epochs", flush=True)
            break
    log.close()
    print(f"done: best score {best:.6f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
