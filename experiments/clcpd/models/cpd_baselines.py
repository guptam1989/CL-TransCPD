"""Online change-point detection baselines, all exposing ``reset()`` and ``step(x) -> score``.

Classical detectors operate on a compact 12-lead morphology descriptor per 2-s window
(per-lead mean, SD, max, min and mean absolute first difference), standardised and projected
with PCA fitted on training-fold windows:
  CUSUM  : multivariate CUSUM on the Mahalanobis distance to the stream's warm-up reference
           (within-record covariance, allowance kappa = 75th percentile of training distances);
  BOCPD  : Adams & MacKay (2007), Normal-Gamma model per dimension, constant hazard 1/lambda,
           run length truncated at r_max; score = P(r_t <= 2);
  RuLSIF : Liu et al. (2013), symmetric alpha-relative Pearson divergence between the previous
           and the current n windows.
Deep detectors are trained on consecutive 2-s windows of training-fold records:
  TIRE   : De Ryck et al. (2021), autoencoder with time-invariant (TI) latent features
           (time-domain variant); score = distance between consecutive TI features;
  LSTM-AE: sequence autoencoder (adapted from Hou et al., 2020); score = reconstruction error
           relative to the warm-up level;
  TS-CP2 : Deldari et al. (2021), TCN encoder trained with InfoNCE on adjacent windows;
           score = 1 - cosine similarity of consecutive embeddings;
  OCAE   : re-implementation of the per-lead overcomplete autoencoder of the authors' earlier
           work (hidden = 2 x window, L2 activity penalty); score = relative reconstruction error.
Supervised reference (uses the diagnostic labels, like CL-TransCPD):
  ResNet-CPD: ResNet1d-Wang trained on random 2-s crops of training-fold records with the record's
           superclass labels; score = deviation of its output (superclass probabilities or penultimate
           features, chosen on the validation streams) from an EMA reference, i.e. the change score of
           CL-TransCPD with mu = 1.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gammaln, logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .cls_baselines import ResNet1dWang, nt_xent, predict_proba


# ----------------------------------------------------------------------------- windows / features
def fixed_bounds(T, fs, window_s, hop_s):
    w, h = int(round(window_s * fs)), int(round(hop_s * fs))
    return [(i, w) for i in range(0, T - w + 1, h)]


def windows_of(sig_std, bounds):
    """sig_std: (T, L) standardised -> (K, L, w)."""
    return np.stack([sig_std[s:s + w].T for s, w in bounds]).astype(np.float32) if bounds else \
        np.zeros((0, sig_std.shape[1], 1), np.float32)


def morph_features(X):
    d = np.abs(np.diff(X, axis=-1)).mean(-1)
    return np.concatenate([X.mean(-1), X.std(-1), X.max(-1), X.min(-1), d], axis=1)


class FeatureProjector:
    def __init__(self, dims=10):
        self.dims = dims

    def fit(self, Fm):
        self.sc = StandardScaler().fit(Fm)
        self.pca = PCA(min(self.dims, Fm.shape[1]), whiten=True, random_state=0).fit(self.sc.transform(Fm))
        return self

    def transform(self, Fm):
        return self.pca.transform(self.sc.transform(Fm))


# ----------------------------------------------------------------------------- classical detectors
class CUSUMDetector:
    def __init__(self, inv_cov, kappa, warm=3):
        self.inv_cov, self.kappa, self.warm = inv_cov, float(kappa), warm
        self.reset()

    def reset(self):
        self.buf, self.mu, self.C = [], None, 0.0

    def step(self, f):
        if self.mu is None:
            self.buf.append(f)
            if len(self.buf) >= self.warm:
                self.mu = np.mean(self.buf, 0)
            return np.nan
        d = f - self.mu
        m = float(np.sqrt(max(d @ self.inv_cov @ d, 0.0)))
        self.C = max(0.0, self.C + m - self.kappa)
        return self.C

    @staticmethod
    def fit(feats_by_record, warm=3):
        cen = np.concatenate([Fr - Fr.mean(0) for Fr in feats_by_record if len(Fr) > 1])
        cov = np.cov(cen.T) + 1e-3 * np.eye(cen.shape[1])
        inv = np.linalg.inv(cov)
        dists = []
        for Fr in feats_by_record:
            if len(Fr) <= warm:
                continue
            mu = Fr[:warm].mean(0)
            for f in Fr[warm:]:
                d = f - mu
                dists.append(np.sqrt(max(d @ inv @ d, 0.0)))
        return inv, float(np.percentile(dists, 75)) if dists else 1.0


class BOCPDDetector:
    def __init__(self, dim, hazard=20.0, rmax=200, r0=2, warm=4):
        self.D, self.hazard, self.rmax, self.r0, self.warm = dim, float(hazard), rmax, r0, warm
        self.mu0, self.k0, self.a0, self.b0 = 0.0, 1.0, 1.0, 1.0
        self.reset()

    def reset(self):
        self.t = 0
        self.logR = np.zeros(1)
        self.mu = np.full((1, self.D), self.mu0)
        self.kappa = np.array([self.k0])
        self.alpha = np.array([self.a0])
        self.beta = np.full((1, self.D), self.b0)

    def step(self, x):
        x = np.asarray(x, dtype=float)
        df = 2.0 * self.alpha
        scale2 = self.beta * ((self.kappa + 1.0) / (self.alpha * self.kappa))[:, None]
        z2 = (x[None, :] - self.mu) ** 2 / scale2
        logp = (gammaln((df + 1) / 2) - gammaln(df / 2))[:, None] - 0.5 * np.log(np.pi * df[:, None] * scale2) \
            - ((df + 1) / 2)[:, None] * np.log1p(z2 / df[:, None])
        logpred = logp.sum(1)
        H = 1.0 / self.hazard
        growth = self.logR + logpred + np.log(1.0 - H)
        cp = logsumexp(self.logR + logpred + np.log(H))
        logR = np.concatenate([[cp], growth])
        logR -= logsumexp(logR)
        k1 = (self.kappa + 1.0)[:, None]
        mu_new = (self.kappa[:, None] * self.mu + x[None, :]) / k1
        beta_new = self.beta + self.kappa[:, None] * (x[None, :] - self.mu) ** 2 / (2.0 * k1)
        self.mu = np.vstack([np.full((1, self.D), self.mu0), mu_new])
        self.kappa = np.concatenate([[self.k0], self.kappa + 1.0])
        self.alpha = np.concatenate([[self.a0], self.alpha + 0.5])
        self.beta = np.vstack([np.full((1, self.D), self.b0), beta_new])
        if len(logR) > self.rmax:
            logR = logR[:self.rmax] - logsumexp(logR[:self.rmax])
            self.mu, self.kappa = self.mu[:self.rmax], self.kappa[:self.rmax]
            self.alpha, self.beta = self.alpha[:self.rmax], self.beta[:self.rmax]
        self.logR = logR
        self.t += 1
        if self.t <= self.warm:
            return np.nan
        return float(np.exp(logR[:self.r0 + 1]).sum())


class RuLSIFDetector:
    def __init__(self, n=4, alpha=0.1, lam=0.1):
        self.n, self.alpha, self.lam = n, alpha, lam
        self.reset()

    def reset(self):
        self.buf = []

    def _pe(self, X, Xp):
        allx = np.vstack([X, Xp])
        d2 = ((allx[:, None, :] - allx[None, :, :]) ** 2).sum(-1)
        sigma2 = max(np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0, 1e-6)
        kern = lambda A: np.exp(-((A[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * sigma2))
        Kx, Kp = kern(X), kern(Xp)
        n, npr = len(X), len(Xp)
        Hm = self.alpha / n * Kx.T @ Kx + (1 - self.alpha) / npr * Kp.T @ Kp
        h = Kx.mean(0)
        theta = np.linalg.solve(Hm + self.lam * np.eye(len(h)), h)
        gx, gp = Kx @ theta, Kp @ theta
        pe = -self.alpha / (2 * n) * (gx ** 2).sum() - (1 - self.alpha) / (2 * npr) * (gp ** 2).sum() + gx.mean() - 0.5
        return max(float(pe), 0.0)

    def step(self, f):
        self.buf.append(np.asarray(f, dtype=float))
        if len(self.buf) > 2 * self.n:
            self.buf.pop(0)
        if len(self.buf) < 2 * self.n:
            return np.nan
        Xr, Xt = np.array(self.buf[:self.n]), np.array(self.buf[self.n:])
        return self._pe(Xr, Xt) + self._pe(Xt, Xr)


class EmbeddingDistanceDetector:
    def __init__(self, metric="euclidean"):
        self.metric = metric
        self.reset()

    def reset(self):
        self.prev = None

    def step(self, e):
        e = np.asarray(e, dtype=float)
        if self.prev is None:
            self.prev = e
            return np.nan
        if self.metric == "cosine":
            s = 1.0 - float(e @ self.prev / (np.linalg.norm(e) * np.linalg.norm(self.prev) + 1e-12))
        else:
            s = float(np.linalg.norm(e - self.prev))
        self.prev = e
        return s


class ErrorDetector:
    def __init__(self, warm=3):
        self.warm = warm
        self.reset()

    def reset(self):
        self.buf, self.ref = [], None

    def step(self, err):
        if self.ref is None:
            self.buf.append(float(err))
            if len(self.buf) >= self.warm:
                self.ref = float(np.mean(self.buf)) + 1e-8
            return np.nan
        return float(err) / self.ref - 1.0


# ----------------------------------------------------------------------------- deep models
class TIRE(nn.Module):
    def __init__(self, in_dim, ti=8, tv=2, hidden=64):
        super().__init__()
        self.ti = ti
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, ti + tv))
        self.dec = nn.Sequential(nn.Linear(ti + tv, hidden), nn.GELU(), nn.Linear(hidden, in_dim))

    def forward(self, x):
        h = self.enc(x)
        return h, self.dec(h)


class LSTMAE(nn.Module):
    def __init__(self, n_leads=12, hidden=64):
        super().__init__()
        self.enc = nn.LSTM(n_leads, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_leads)

    def forward(self, x):                      # x: (B, T, L)
        _, (h, _) = self.enc(x)
        rep = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.dec(rep)
        return self.out(y)


class _Chomp(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[..., :-self.n] if self.n else x


class TCNEncoder(nn.Module):
    def __init__(self, n_leads=12, ch=64, out=32, dilations=(1, 2, 4, 8), k=3):
        super().__init__()
        layers, cin = [], n_leads
        for d in dilations:
            layers += [nn.Conv1d(cin, ch, k, dilation=d, padding=(k - 1) * d), _Chomp((k - 1) * d), nn.ReLU()]
            cin = ch
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(ch, out)

    def forward(self, x):
        return self.proj(self.net(x).mean(-1))


class OCAE(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(w, 2 * w), nn.ReLU())
        self.dec = nn.Linear(2 * w, w)

    def forward(self, x):                      # x: (B, L, w)
        h = self.enc(x)
        return h, self.dec(h)


def _down(x, f=4):
    """(B, L, w) -> (B, L, w // f) by average pooling (torch)."""
    return F.avg_pool1d(x, f, f) if f > 1 else x


class PairSampler:
    """Consecutive-window pairs (record, start_k, start_k+1) drawn from training records."""

    def __init__(self, signals_std, rows, fs, window_s, hop_s):
        self.sig = signals_std
        T = signals_std.shape[1]
        self.bounds = fixed_bounds(T, fs, window_s, hop_s)
        self.rows = np.asarray(rows)
        self.w = self.bounds[0][1]

    def batch(self, rng, bs):
        r = self.rows[rng.integers(len(self.rows), size=bs)]
        k = rng.integers(len(self.bounds) - 1, size=bs)
        s1 = np.array([self.bounds[j][0] for j in k])
        s2 = np.array([self.bounds[j + 1][0] for j in k])
        idx = np.arange(self.w)
        x1 = self.sig[r[:, None], s1[:, None] + idx[None, :]].transpose(0, 2, 1)
        x2 = self.sig[r[:, None], s2[:, None] + idx[None, :]].transpose(0, 2, 1)
        return torch.from_numpy(np.ascontiguousarray(x1)), torch.from_numpy(np.ascontiguousarray(x2))


def build_deep(name, n_leads, w):
    if name == "tire":
        return TIRE(n_leads * (w // 4))
    if name == "lstm_ae":
        return LSTMAE(n_leads)
    if name == "tscp2":
        return TCNEncoder(n_leads)
    if name == "ocae":
        return OCAE(w)
    raise ValueError(name)


def train_deep_baseline(name, sampler, n_leads, device, epochs=30, steps_per_epoch=200, bs=256, lr=1e-3,
                        seed=0, log=None, transform=None):
    """transform: optional preprocessing applied to every (B, L, w) batch (shared RSSA denoising)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_deep(name, n_leads, sampler.w).to(device)
    tf = transform or (lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            x1, x2 = sampler.batch(rng, bs)
            with torch.no_grad():
                x1, x2 = tf(x1.to(device)), tf(x2.to(device))
            if name == "tire":
                a, b = _down(x1).flatten(1), _down(x2).flatten(1)
                h1, r1 = model(a)
                h2, r2 = model(b)
                loss = F.mse_loss(r1, a) + F.mse_loss(r2, b) + ((h1[:, :model.ti] - h2[:, :model.ti]) ** 2).sum(1).mean()
            elif name == "lstm_ae":
                a = _down(x1).transpose(1, 2)
                loss = F.mse_loss(model(a), a)
            elif name == "tscp2":
                loss = nt_xent(model(x1), model(x2))
            else:
                h, r = model(x1)
                loss = F.mse_loss(r, x1) + 1e-4 * (h ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        if log:
            log(f"  {name} epoch {ep + 1}/{epochs} loss={tot / steps_per_epoch:.4f}")
    return model


@torch.no_grad()
def deep_outputs(name, model, X, device, bs=1024):
    """X: (K, L, w) standardised windows -> per-window score inputs (embedding or error)."""
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], device=device)
        if name == "tire":
            h, _ = model(_down(x).flatten(1))
            outs.append(h[:, :model.ti].cpu().numpy())
        elif name == "lstm_ae":
            a = _down(x).transpose(1, 2)
            outs.append(((model(a) - a) ** 2).mean((1, 2)).cpu().numpy())
        elif name == "tscp2":
            outs.append(model(x).cpu().numpy())
        else:
            _, r = model(x)
            outs.append(((r - x) ** 2).mean((1, 2)).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0)


class CropSampler:
    """Random fixed-length crops of training records with the superclass label set of the record."""

    def __init__(self, signals_std, Y, fs, window_s):
        self.sig, self.Y = signals_std, np.asarray(Y, dtype=np.float32)
        self.w = int(round(window_s * fs))

    def batch(self, rng, bs):
        r = rng.integers(len(self.sig), size=bs)
        s = rng.integers(self.sig.shape[1] - self.w + 1, size=bs)
        idx = np.arange(self.w)
        x = self.sig[r[:, None], s[:, None] + idx[None, :]].transpose(0, 2, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.Y[r])


def train_supervised_baseline(sampler, Xva, yva, n_leads, device, epochs=30, steps_per_epoch=200, bs=256, lr=1e-3,
                              wd=1e-2, patience=10, seed=0, log=None, transform=None):
    """ResNet1d-Wang on random crops (multi-label BCE, AdamW, one-cycle schedule); keeps the state with the best
    macro-AUC on the validation crops (fold 9). Returns (model, training summary)."""
    from ..eval.cls_metrics import multilabel_metrics

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = ResNet1dWang(n_leads, yva.shape[1]).to(device)
    tf = transform or (lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch),
                                                pct_start=0.1)
    best, best_state, best_ep, bad, ep = -np.inf, None, 0, 0, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            x, y = sampler.batch(rng, bs)
            with torch.no_grad():
                x = tf(x.to(device))
            loss = F.binary_cross_entropy_with_logits(model(x).float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        score = multilabel_metrics(yva, predict_proba(model, Xva, "ml", device))["macro_auc"]
        if log:
            log(f"  supervised epoch {ep + 1}/{epochs} loss={tot / steps_per_epoch:.4f} val macro-AUC={score:.4f}")
        if np.isfinite(score) and score > best:
            best, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_macro_auc_crops": float(best), "best_epoch": best_ep, "epochs_run": ep + 1}


@torch.no_grad()
def supervised_outputs(model, X, device, bs=1024):
    """X: (K, L, w) -> superclass probabilities (K, C) and l2-normalised penultimate features (K, d)."""
    model.eval()
    prob, feat = [], []
    for i in range(0, len(X), bs):
        h = model.features(torch.as_tensor(X[i:i + bs], device=device)).float()
        prob.append(torch.sigmoid(model.head(h)).cpu().numpy())
        feat.append(F.normalize(h, dim=-1).cpu().numpy())
    return {"prob": np.concatenate(prob), "feat": np.concatenate(feat)}


@torch.no_grad()
def ocae_latent(model, X, device, bs=1024):
    """Mean hidden activation over leads (prior-work latent feature F_au), X: (K, L, w)."""
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        h, _ = model(torch.as_tensor(X[i:i + bs], device=device))
        outs.append(h.mean(1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1))


def make_detector(name, fitted, cfg):
    """fitted: dict produced in scripts/run_cpd.py (projector, cusum params, ...)."""
    if name == "cusum":
        return CUSUMDetector(fitted["inv_cov"], fitted["kappa"], warm=cfg.score.warmup_segments)
    if name == "bocpd":
        return BOCPDDetector(fitted["dim"], hazard=cfg.cpd_baselines.bocpd_hazard)
    if name == "rulsif":
        return RuLSIFDetector(n=cfg.cpd_baselines.rulsif_n, alpha=cfg.cpd_baselines.rulsif_alpha)
    if name == "tire":
        return EmbeddingDistanceDetector("euclidean")
    if name == "tscp2":
        return EmbeddingDistanceDetector("cosine")
    if name in ("lstm_ae", "ocae"):
        return ErrorDetector(warm=cfg.score.warmup_segments)
    raise ValueError(name)
