"""Cross-fitted LightGBM models and rival (competition) features.

GroupKFold by S1 entity (config.N_FOLDS folds): model j is trained on every fold
except fold j, and a pair whose S1 is in fold j is scored by model j (out-of-fold);
pairs of S1 outside the folds (test) get the average of all models. Stage 1.5,
Stage 2 and Stage 3 all use this, so the scores that feed later stages are computed
the same way on training pairs as on test pairs.

(Older two-fold files stored "model k trained ON half k" instead; they carry no
`excluded` flag and are read with that convention.)"""
import json
import os

import lightgbm as lgb
import numpy as np

import config


def make_folds(part, train_code=0, seed=config.SEED + 1):
    """fold[l] in {0, 1} for training S1 rows, -1 elsewhere."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(part)).astype(np.int8)
    fold[part != train_code] = -1
    return fold


class CrossModel:
    # prediction early stopping with a margin of 20 (|raw score| > 20, i.e. p < 2e-9) was
    # measured lossless for top-M ranking and 1.7x faster (prerank_speed.py)
    EARLY = dict(pred_early_stop=True, pred_early_stop_freq=5, pred_early_stop_margin=20.0)

    def __init__(self, models, cols, fold=None, excluded=True):
        self.models, self.cols, self.fold, self.excluded = models, cols, fold, excluded

    def _pred(self, m, X):
        return m.predict(X, num_threads=config.N_JOBS, **self.EARLY)

    def model_for_fold(self, k):
        return self.models[k] if self.excluded else self.models[1 - k]

    def predict_rows(self, X, l=None):
        """X: 2-D float32 array (or DataFrame) with self.cols; l: S1 rows (for OOF)."""
        X = X[self.cols].to_numpy(np.float32) if hasattr(X, "columns") else X
        p = np.zeros(len(X), np.float64)
        f = np.full(len(X), -1, np.int8) if (l is None or self.fold is None) else self.fold[l]
        avg = f < 0
        if avg.any():
            p[avg] = np.mean([self._pred(m, X[avg]) for m in self.models], axis=0)
        for k in range(len(self.models)):
            sel = f == k
            if sel.any():
                p[sel] = self._pred(self.model_for_fold(k), X[sel])
        return p.astype(np.float32)

    def __call__(self, F):
        return self.predict_rows(F, F["l"].values)

    def save(self, prefix):
        for k, m in enumerate(self.models):
            m.save_model(f"{prefix}_m{k}.txt")
        json.dump({"cols": self.cols, "n": len(self.models), "excluded": self.excluded},
                  open(prefix + ".json", "w"))

    @classmethod
    def load(cls, prefix, fold=None):
        meta = json.load(open(prefix + ".json"))
        models = [lgb.Booster(model_file=f"{prefix}_m{k}.txt") for k in range(meta["n"])]
        return cls(models, meta["cols"], fold, meta.get("excluded", False))


def fit_two_fold(X, y, fold_of_row, params, rounds, early=50, weight=None, log=print):
    """Train model k on fold-k rows, early-stopping on the other fold's rows.
    X: float32 2-D array; fold_of_row: 0/1 per row. Returns (models, oof)."""
    models, oof = [], np.zeros(len(y), np.float32)
    for k in (0, 1):
        tr, va = np.flatnonzero(fold_of_row == k), np.flatnonzero(fold_of_row == 1 - k)
        dtr = lgb.Dataset(X[tr], y[tr], weight=None if weight is None else weight[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        models.append(m)
        log(f"  fold model {k}: best_iter {m.best_iteration}, "
            f"valid logloss {m.best_score['valid_0']['binary_logloss']:.5f}")
    return models, oof


def competition(l, r, p):
    """Rival features in numpy (memory-light): for each pair, how does p compare with
    the other candidates of its S1 (l) and of its S2/S3 record (r)."""
    n = len(p)
    out = {}
    for key, tag in ((l, "l"), (r, "r")):
        order = np.lexsort((-p, key))
        ks, ps = key[order], p[order]
        start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        size = np.diff(np.r_[start, n])
        first = np.repeat(start, size)
        rank = np.arange(n) - first
        best = ps[first]
        second = np.where(size > 1, ps[np.minimum(start + 1, n - 1)], 0.0)
        second = np.repeat(second, size)
        cnt = np.repeat(size, size).astype(np.float32)
        sums = np.repeat(np.add.reduceat(ps, start), size)
        inv = np.empty(n, np.int64)
        inv[order] = np.arange(n)
        out[f"rank_{tag}"] = (rank + 1).astype(np.float32)[inv]
        out[f"gap_{tag}"] = (best - ps).astype(np.float32)[inv]
        out[f"n_{tag}"] = cnt[inv]
        if tag == "l":
            # margin of the best over the runner-up; for the best pair it is its lead,
            # for the others it is how far the leader is ahead of the second
            out["lead_l"] = (best - second).astype(np.float32)[inv]
            out["sum_l"] = sums.astype(np.float32)[inv]
            out["second_gap_l"] = np.where(rank == 0, ps - second, ps - best).astype(np.float32)[inv]
    return out
