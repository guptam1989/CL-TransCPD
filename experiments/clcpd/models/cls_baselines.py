"""End-to-end classification baselines on raw 10-s, 12-lead, 100-Hz records.

resnet1d_wang / inception1d / bilstm follow the PTB-XL benchmark of Strodthoff et al. (2021);
transformer is a convolution-stem Transformer encoder with lead-concatenated tokens;
patchtst follows Nie et al. (2023) (channel-independent patches, shared encoder);
clocs is CMSC contrastive pretraining (Kiyasseh et al., 2021) of the resnet1d_wang encoder on
temporally adjacent 5-s halves of the same record, followed by supervised fine-tuning.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..eval.cls_metrics import multiclass_metrics, multilabel_metrics


class _ResBlockWang(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(cin, cout, 8, padding="same"), nn.BatchNorm1d(cout)
        self.c2, self.b2 = nn.Conv1d(cout, cout, 5, padding="same"), nn.BatchNorm1d(cout)
        self.c3, self.b3 = nn.Conv1d(cout, cout, 3, padding="same"), nn.BatchNorm1d(cout)
        self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout)) if cin != cout else nn.BatchNorm1d(cout)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = F.relu(self.b2(self.c2(y)))
        return F.relu(self.b3(self.c3(y)) + self.sc(x))


class ResNet1dWang(nn.Module):
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()
        self.body = nn.Sequential(_ResBlockWang(n_leads, 64), _ResBlockWang(64, 128), _ResBlockWang(128, 128))
        self.dim = 128
        self.head = nn.Linear(128, n_classes)

    def features(self, x):
        return self.body(x).mean(-1)

    def forward(self, x):
        return self.head(self.features(x))


class _Inception(nn.Module):
    def __init__(self, cin, nf=32, ks=(39, 19, 9), bottleneck=32):
        super().__init__()
        self.bott = nn.Conv1d(cin, bottleneck, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(bottleneck, nf, k, padding=k // 2, bias=False) for k in ks])
        self.mp = nn.Sequential(nn.MaxPool1d(3, 1, padding=1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))

    def forward(self, x):
        b = self.bott(x)
        return F.relu(self.bn(torch.cat([c(b) for c in self.convs] + [self.mp(x)], 1)))


class Inception1d(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, depth=6, nf=32):
        super().__init__()
        self.mods = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        cin, res_in = n_leads, n_leads
        for d in range(depth):
            self.mods.append(_Inception(cin, nf))
            cin = nf * 4
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_in, cin, 1, bias=False), nn.BatchNorm1d(cin)))
                res_in = cin
        self.dim = cin
        self.head = nn.Linear(cin, n_classes)

    def features(self, x):
        res, s = x, 0
        for d, m in enumerate(self.mods):
            x = m(x)
            if d % 3 == 2:
                x = F.relu(x + self.shortcuts[s](res))
                res, s = x, s + 1
        return x.mean(-1)

    def forward(self, x):
        return self.head(self.features(x))


class BiLSTM(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_leads, hidden, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(4 * hidden, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(1), h.max(1).values], 1))


class ECGTransformer(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, d=128, layers=4, heads=4, d_ff=256, max_len=256):
        super().__init__()
        self.stem = nn.Conv1d(n_leads, d, 10, stride=5)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d_ff, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h = self.enc(h + self.pos[:, :h.shape[1]])
        return self.head(h.mean(1))


class PatchTST(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, patch=25, d=128, layers=3, heads=4, d_ff=256, seq_len=1000):
        super().__init__()
        self.patch, self.L = patch, n_leads
        n_p = seq_len // patch
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, n_p, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d_ff, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(n_leads * d, n_classes)

    def forward(self, x):
        B, L, T = x.shape
        n_p = T // self.patch
        tok = self.embed(x[:, :, :n_p * self.patch].reshape(B * L, n_p, self.patch))
        h = self.enc(tok + self.pos[:, :n_p]).mean(1).reshape(B, L * tok.shape[-1])
        return self.head(self.drop(h))


def build_classifier(name, n_leads=12, n_classes=5, seq_len=1000):
    if name in ("resnet1d_wang", "clocs"):
        return ResNet1dWang(n_leads, n_classes)
    if name == "inception1d":
        return Inception1d(n_leads, n_classes)
    if name == "bilstm":
        return BiLSTM(n_leads, n_classes)
    if name == "transformer":
        return ECGTransformer(n_leads, n_classes)
    if name == "patchtst":
        return PatchTST(n_leads, n_classes, seq_len=seq_len)
    raise ValueError(name)


def _batches(n, bs, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict_proba(model, X, task, device, bs=512):
    model.eval()
    out = []
    for b in _batches(len(X), bs):
        logits = model(torch.as_tensor(X[b], device=device)).float()
        out.append((torch.sigmoid(logits) if task == "ml" else torch.softmax(logits, -1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 5))


def _val_score(task, y, p):
    if task == "ml":
        return multilabel_metrics(y, p)["macro_auc"]
    return multiclass_metrics(y, p, p.shape[1])["auc"]


def fit_classifier(model, Xtr, ytr, Xva, yva, task, device, epochs=50, bs=128, lr=1e-3, wd=1e-2,
                   seed=0, patience=10, log=None):
    """task 'ml' (multi-label BCE) or 'sl' (single-label CE). Keeps the best validation state."""
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps), pct_start=0.1)
    best, best_state, bad = -np.inf, None, 0
    use_amp = device.type == "cuda" and not isinstance(model, BiLSTM)   # keep cuDNN LSTM in fp32
    for ep in range(epochs):
        model.train()
        for b in _batches(len(Xtr), bs, rng):
            x = torch.as_tensor(Xtr[b], device=device)
            y = torch.as_tensor(ytr[b], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x).float()
            loss = F.binary_cross_entropy_with_logits(logits, y.float()) if task == "ml" \
                else F.cross_entropy(logits, y.long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        score = _val_score(task, yva, predict_proba(model, Xva, task, device))
        if log:
            log(f"  epoch {ep + 1}/{epochs} val={score:.4f}")
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def nt_xent(a, b, tau=0.1):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.t() / tau
    target = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def pretrain_clocs(model: ResNet1dWang, Xtr, device, epochs=50, bs=256, lr=1e-3, seed=0, log=None):
    """CMSC: the two non-overlapping 5-s halves of a record form a positive pair."""
    rng = np.random.default_rng(seed)
    proj = nn.Sequential(nn.Linear(model.dim, model.dim), nn.ReLU(), nn.Linear(model.dim, 64)).to(device)
    model.to(device)
    params = list(model.body.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    half = Xtr.shape[-1] // 2
    use_amp = device.type == "cuda"
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for b in _batches(len(Xtr), bs, rng):
            if len(b) < 2:
                continue
            x = torch.as_tensor(Xtr[b], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                za = proj(model.features(x[..., :half])).float()
                zb = proj(model.features(x[..., half:2 * half])).float()
            loss = nt_xent(za, zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        if log:
            log(f"  CMSC epoch {ep + 1}/{epochs} loss={tot / max(1, nb):.4f}")
    return model
