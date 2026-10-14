"""SkEnsem: stacked ensemble (gradient boosting, AdaBoost, random forest, 1-D CNN) with an
XGBoost meta-learner (Wolpert, 1992), trained with patient-grouped out-of-fold predictions.

Pipeline (fitted on the training folds only):
  median imputation -> feature selection (keep features whose extra-trees importance is at least
  the mean importance) -> 5-fold StratifiedGroupKFold (groups = patients) out-of-fold base
  predictions -> meta-learner. Meta-learner hyper-parameters are chosen on the validation fold.
Supports single-label ('sl', multi-class) and multi-label ('ml', one-vs-rest meta-learners).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..utils import threads


class CNN1DClassifier:
    """Small 1-D CNN over the standardised feature vector (sklearn-like API)."""

    def __init__(self, n_out, multilabel=False, epochs=60, lr=1e-3, bs=256, seed=0):
        self.n_out, self.multilabel, self.epochs, self.lr, self.bs, self.seed = n_out, multilabel, epochs, lr, bs, seed

    def _net(self):
        return nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, self.n_out))

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sc = StandardScaler().fit(X)
        Xs = torch.as_tensor(np.nan_to_num(self.sc.transform(X)), dtype=torch.float32)
        Y = torch.as_tensor(y, dtype=torch.float32 if self.multilabel else torch.long)
        self.net = self._net().to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(Xs))
            for i in range(0, len(Xs), self.bs):
                b = perm[i:i + self.bs]
                out = self.net(Xs[b].unsqueeze(1).to(self.dev))
                yb = Y[b].to(self.dev)
                loss = F.binary_cross_entropy_with_logits(out, yb) if self.multilabel else F.cross_entropy(out, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        if "net" in state:
            state["net"] = self.net.to("cpu")
        state.pop("dev", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "net" in state:
            self.net.to(self.dev)

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        Xs = torch.as_tensor(np.nan_to_num(self.sc.transform(X)), dtype=torch.float32)
        outs = []
        for i in range(0, len(Xs), 4096):
            o = self.net(Xs[i:i + 4096].unsqueeze(1).to(self.dev)).float()
            outs.append((torch.sigmoid(o) if self.multilabel else torch.softmax(o, -1)).cpu().numpy())
        return np.concatenate(outs)


def _proba(est, X, multilabel, n_classes):
    p = est.predict_proba(X)
    if isinstance(p, list):                       # native multi-output (random forest)
        return np.stack([pi[:, 1] if pi.shape[1] > 1 else np.zeros(len(X)) for pi in p], 1)
    if not multilabel and p.shape[1] != n_classes:
        full = np.zeros((len(X), n_classes))
        full[:, est.classes_.astype(int)] = p
        return full
    return p


class SkEnsem:
    BASE = ("gb", "ada", "rf", "cnn")

    def __init__(self, task="sl", n_classes=5, n_folds=5, seed=0, cfg=None, bases=None, select=True):
        self.task, self.C, self.n_folds, self.seed = task, n_classes, n_folds, seed
        self.multilabel = task == "ml"
        c = cfg or {}
        self.n_est = int(c.get("n_estimators", 200))
        self.depth = int(c.get("max_depth", 4))
        self.lr = float(c.get("learning_rate", 0.05))
        self.cnn_epochs = int(c.get("cnn_epochs", 60))
        self.bases = tuple(bases or self.BASE)
        self.select = select

    def _make(self, name):
        s = self.seed
        if name == "gb":
            est = HistGradientBoostingClassifier(max_iter=self.n_est, learning_rate=self.lr, max_depth=self.depth, random_state=s)
        elif name == "ada":
            est = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=self.n_est // 2,
                                     learning_rate=0.5, random_state=s)
        elif name == "rf":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=threads(), random_state=s)
        elif name == "cnn":
            return CNN1DClassifier(self.C, self.multilabel, epochs=self.cnn_epochs, seed=s)
        else:
            raise ValueError(name)
        return OneVsRestClassifier(est, n_jobs=1) if self.multilabel else est

    def _strata(self, Y):
        if self.multilabel:
            return np.array([int("".join(map(str, row.astype(int))), 2) for row in Y])
        return Y

    def _meta_model(self, n_est, depth):
        # disable_default_eval_metric: the training metric is only logged (no eval set / early stopping), and
        # computing it fails in CUDA builds of xgboost 3.2 on CPU-only nodes for multi-class objectives
        kw = dict(n_estimators=n_est, max_depth=depth, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                  random_state=self.seed, n_jobs=min(4, threads()), tree_method="hist", verbosity=0,
                  disable_default_eval_metric=True)
        if self.multilabel:
            return [XGBClassifier(objective="binary:logistic", **kw) for _ in range(self.C)]
        return XGBClassifier(objective="multi:softprob", num_class=self.C, **kw)

    def _fit_meta(self, meta, Y, n_est, depth):
        m = self._meta_model(n_est, depth)
        if self.multilabel:
            for c, mc in enumerate(m):
                mc.fit(meta, Y[:, c])
        else:
            m.fit(meta, Y)
        return m

    def _meta_proba(self, m, meta):
        if self.multilabel:
            return np.stack([mc.predict_proba(meta)[:, 1] for mc in m], 1)
        return m.predict_proba(meta)

    def _transform(self, X):
        X = self.imp.transform(X)
        return X[:, self.keep] if self.keep is not None else X

    def fit(self, X, Y, groups, Xval=None, Yval=None, score_fn=None):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xi = self.imp.transform(X)
        self.keep = None
        if self.select and Xi.shape[1] > 20:
            et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=threads(), random_state=self.seed)
            et.fit(Xi, Y)
            imp = et.feature_importances_
            self.keep = np.where(imp >= imp.mean())[0]
        Xs = Xi[:, self.keep] if self.keep is not None else Xi
        strata = self._strata(Y)
        oof = np.zeros((len(Xs), len(self.bases) * self.C))
        skf = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(Xs, strata, groups):
            for b, name in enumerate(self.bases):
                est = self._make(name).fit(Xs[tr], Y[tr])
                oof[va, b * self.C:(b + 1) * self.C] = _proba(est, Xs[va], self.multilabel, self.C)
        self.base_models = [self._make(name).fit(Xs, Y) for name in self.bases]
        grid = [(100, 2), (200, 3), (300, 4)]
        self.meta_choice = (200, 3)
        if Xval is not None and score_fn is not None:
            mv = self._base_meta(Xval)
            best = -np.inf
            for n_est, depth in grid:
                sc = score_fn(Yval, self._meta_proba(self._fit_meta(oof, Y, n_est, depth), mv))
                if sc > best:
                    best, self.meta_choice = sc, (n_est, depth)
        self.meta = self._fit_meta(oof, Y, *self.meta_choice)
        return self

    def _base_meta(self, X):
        Xs = self._transform(X)
        return np.concatenate([_proba(m, Xs, self.multilabel, self.C) for m in self.base_models], 1)

    def predict_proba(self, X):
        return self._meta_proba(self.meta, self._base_meta(X))

    def selected_fraction(self):
        return None if self.keep is None else float(len(self.keep)) / float(len(self.imp.statistics_))
