"""Train CL-TransCPD (HTFP denoising pretraining, then contrastive change-point pretraining).

  python scripts/pretrain_clt.py --variant full --seed 0
  python scripts/pretrain_clt.py --task-index 7          # PBS array: index into tasks_pretrain()
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _common import config_from, parser, pick_task
from clcpd.data.triplets import CCPPDataset, DenoiseDataset
from clcpd.models.clt import CLTransCPD, count_parameters
from clcpd.models.losses import contrastive_loss, masked_mse
from clcpd.pipeline import MAIN_ABLATIONS, PTBXL, load_pickle, run_dir, variant_config
from clcpd.utils import get_device, log, pbs_array_index, save_json, set_seed


def tasks_pretrain(cfg):
    """Full model first, then ablations seed by seed (so a partial run still covers every variant)."""
    tasks = [("full", s) for s in cfg.eval.seeds]
    tasks += [(v, s) for s in cfg.eval.ablation_seeds for v in MAIN_ABLATIONS if v != "full"]
    return tasks


LOCK_STALE_S = 30 * 60


def acquire_lock(rd: Path) -> bool:
    """One training per run directory at a time (several PBS jobs may sweep the same task list).
    A lock is stale when its process is gone (same host) or it has not been refreshed for 30 min."""
    import os
    import socket

    lk = rd / "RUNNING"
    me = f"{socket.gethostname()} {os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                host, pid = lk.read_text().split()
                alive = True
                if host == socket.gethostname() and os.name != "nt":   # os.kill(pid, 0) only probes on POSIX
                    try:
                        os.kill(int(pid), 0)
                    except ProcessLookupError:
                        alive = False
                    except PermissionError:
                        pass
                stale = (not alive) or time.time() - lk.stat().st_mtime > LOCK_STALE_S
            except (OSError, ValueError):
                stale = True
            if not stale:
                return False
            lk.unlink(missing_ok=True)
    return False


def save_checkpoint(path: Path, state: dict):
    import os

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _loader(ds, cfg, shuffle=False):
    nw = int(cfg.ccpp.num_workers)
    return DataLoader(ds, batch_size=int(cfg.ccpp.batch_size), shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=False)


def pretrain_htfp(model, P, cfg, device, seed):
    noise = {}
    npath = Path(cfg.paths.work_dir) / "cache" / "nstdb" / "noise.pkl"
    if npath.exists():
        noise = {k: v["train"] for k, v in load_pickle(npath).items()}
    clean = P.train[P.meta["clean"].values[P.train] == 1]
    if len(clean) < 10:
        clean = P.train
    ds = DenoiseDataset(P.signals, P.rpeaks, clean, cfg, (P.mean, P.std), epoch_size=min(20000, 40 * len(clean)),
                        seed=seed, noise_sources=noise)
    opt = torch.optim.AdamW(model.htfp.parameters(), lr=float(cfg.htfp.lr))
    for ep in range(int(cfg.htfp.pretrain_epochs)):
        ds.set_epoch(ep)
        tot, n = 0.0, 0
        for b in _loader(ds, cfg, shuffle=False):
            x, y, m = b["x"].to(device), b["y"].to(device), b["m"].to(device)
            out, _ = model.htfp(x, m)
            loss = masked_mse(out, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        log(f"  HTFP denoising epoch {ep + 1}: mse={tot / max(1, n):.4f} (noise: {'NSTDB' if noise else 'synthetic'})")


def ccpp_step(model, batch, cfg, device, amp):
    B = batch["xa"].shape[0]
    names = ("a", "p", "n", "u", "b") if "xb" in batch else ("a", "p", "n", "u")
    x = torch.cat([batch["x" + k] for k in names]).to(device, non_blocking=True)
    m = torch.cat([batch["m" + k] for k in names]).to(device, non_blocking=True)
    y = batch["y"].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(x, m)
    z = out["z"].float()
    za, zp, zn, zu = z[:B], z[B:2 * B], z[2 * B:3 * B], z[3 * B:4 * B]
    c = cfg.ccpp
    l_con = contrastive_loss(c.loss, za, zp, zn, float(c.margin), float(c.temperature))
    if c.nuisance_positive and c.loss != "none":
        l_con = l_con + contrastive_loss(c.loss, za, zu, zn, float(c.margin), float(c.temperature))
    if "xb" in batch and c.loss != "none":
        zb = z[4 * B:5 * B]
        l_con = l_con + contrastive_loss(c.loss, za, zp, zb, float(c.margin), float(c.temperature))
        if c.nuisance_positive:
            l_con = l_con + contrastive_loss(c.loss, za, zu, zb, float(c.margin), float(c.temperature))
    l_rec = masked_mse(out["recon"][:2 * B].float(), out["target"][:2 * B].float(), m[:2 * B])
    l_cls = F.binary_cross_entropy_with_logits(out["logits"][:B].float(), y)
    loss = l_con + float(c.lambda_rec) * l_rec + float(c.lambda_cls) * l_cls
    return loss, {"con": l_con.item(), "rec": l_rec.item(), "cls": l_cls.item()}


def main():
    ap = parser(__doc__)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    idx = args.task_index if args.task_index is not None else pbs_array_index(None)
    variant, seed = (args.variant, args.seed) if idx is None else pick_task(tasks_pretrain(cfg), idx)
    vcfg = variant_config(cfg, variant)
    set_seed(seed)
    device = get_device()
    rd = run_dir(cfg, variant, seed)
    if (rd / "summary.json").exists():
        log(f"{variant} seed {seed} already trained - skipping")
        return
    if not acquire_lock(rd):
        log(f"{variant} seed {seed} is being trained by another process - try again later")
        raise SystemExit(75)          # EX_TEMPFAIL: run_stage.py re-queues the task
    try:
        train(cfg, vcfg, variant, seed, rd, device)
    finally:
        (rd / "RUNNING").unlink(missing_ok=True)


def train(cfg, vcfg, variant, seed, rd, device):
    amp = bool(vcfg.ccpp.amp) and device.type == "cuda"
    save_json(vcfg.to_dict(), rd / "config.json")
    log(f"variant={variant} seed={seed} device={device}")
    P = PTBXL(vcfg)
    model = CLTransCPD(vcfg).to(device)
    n_params = {"total": count_parameters(model), "clt": count_parameters(model.clt),
                "htfp": count_parameters(model.htfp) if model.htfp is not None else 0,
                "decoder": count_parameters(model.clt.decoder),
                "encoder": count_parameters(model.clt) - count_parameters(model.clt.decoder)
                - count_parameters(model.clt.cls_mlp) - count_parameters(model.clt.cls_out)}
    log(f"parameters: {n_params}")
    t0 = time.time()
    c = vcfg.ccpp
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=float(c.weight_decay))
    best, bad, hist, start, prior_min = np.inf, 0, [], 0, 0.0
    ck_path = rd / "ckpt_last.pt"
    if ck_path.exists():
        # resume after the job was killed: per-epoch data are seeded by the epoch number and the RNG states are
        # restored, so the remaining epochs see the same anchors as an uninterrupted run. Load on the CPU: the RNG
        # states must stay CPU ByteTensors; load_state_dict moves model and optimiser state to the device.
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best, bad, hist, start, prior_min = ck["best"], ck["bad"], ck["hist"], ck["epoch"], ck["minutes"]
        torch.set_rng_state(ck["rng_cpu"])
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"], device)
        np.random.set_state(ck["rng_np"])
        log(f"  resumed after epoch {start} (best val_loss={best:.4f})")
    elif vcfg.htfp.enabled and int(vcfg.htfp.pretrain_epochs) > 0:
        pretrain_htfp(model, P, vcfg, device, seed)
    ds_tr = CCPPDataset(P.signals, P.rpeaks, P.meta, P.train, vcfg, (P.mean, P.std), int(c.anchors_per_epoch), seed)
    ds_va = CCPPDataset(P.signals, P.rpeaks, P.meta, P.val, vcfg, (P.mean, P.std),
                        min(8192, 4 * len(P.val)), seed + 10_000, augment=False)
    epochs, warm = int(c.epochs), int(c.warmup_epochs)
    lr_at = lambda e: float(c.lr) * (e + 1) / max(1, warm) if e < warm else \
        float(c.lr_min) + 0.5 * (float(c.lr) - float(c.lr_min)) * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm)))
    stopped = bad >= int(c.patience)
    for ep in range(start, epochs):
        if stopped:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        ds_tr.set_epoch(ep)
        model.train()
        parts, n = {"con": 0.0, "rec": 0.0, "cls": 0.0}, 0
        for batch in _loader(ds_tr, vcfg):
            loss, d = ccpp_step(model, batch, vcfg, device, amp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in parts:
                parts[k] += d[k]
            n += 1
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in _loader(ds_va, vcfg):
                l, _ = ccpp_step(model, batch, vcfg, device, amp)
                vloss, vn = vloss + l.item(), vn + 1
        vloss /= max(1, vn)
        rec = {"epoch": ep + 1, "lr": lr_at(ep), "val_loss": vloss, **{k: v / max(1, n) for k, v in parts.items()}}
        hist.append(rec)
        log(f"  epoch {ep + 1}/{epochs} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"))
        if vloss < best:
            best, bad = vloss, 0
            torch.save(model.state_dict(), rd / "model.pt")
        else:
            bad += 1
            if bad >= int(c.patience):
                log("  early stopping")
                stopped = True
        save_checkpoint(ck_path, {"epoch": ep + 1, "model": model.state_dict(), "opt": opt.state_dict(), "best": best,
                                  "bad": bad, "hist": hist, "minutes": prior_min + (time.time() - t0) / 60.0,
                                  "rng_cpu": torch.get_rng_state(),
                                  "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                                  "rng_np": np.random.get_state()})
        (rd / "RUNNING").touch()
    minutes = prior_min + (time.time() - t0) / 60.0
    save_json({"variant": variant, "seed": seed, "best_val_loss": best, "epochs_run": len(hist),
               "minutes": minutes, "resumed_at_epoch": start or None, "params": n_params, "history": hist},
              rd / "summary.json")
    ck_path.unlink(missing_ok=True)
    log(f"done in {minutes:.1f} min")


if __name__ == "__main__":
    main()
