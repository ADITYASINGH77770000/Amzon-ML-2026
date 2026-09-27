"""Scorer (macro F0.5), blocking metrics, guard, holdout and drift helpers.

Truth is held as true_s1[r] = index of the S1 row that right record r belongs to
(-1 if none). This works because every S2/S3 record matches at most one S1 (H1)."""
import numpy as np
import pandas as pd


def f05_entity(pred, true):
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def f05_arrays(n_pred, tp, n_true):
    """Vectorised per-entity F0.5."""
    n_pred, tp, n_true = (np.asarray(x, np.float64) for x in (n_pred, tp, n_true))
    f = np.zeros(len(n_true))
    single = n_true == 0
    f[single] = (n_pred[single] == 0).astype(float)
    ok = (~single) & (tp > 0)
    p = tp[ok] / n_pred[ok]
    r = tp[ok] / n_true[ok]
    f[ok] = 1.25 * p * r / (0.25 * p + r)
    return f


def macro_from_pairs(l, y, n_left, n_true, entities=None):
    """Macro F0.5 when the predicted pairs are (l, y) and n_true[l] true matches per S1.
    entities: optional index array of S1 rows to average over (e.g. a fold)."""
    n_pred = np.bincount(l, minlength=n_left)
    tp = np.bincount(l, weights=y.astype(float), minlength=n_left)
    f = f05_arrays(n_pred, tp, n_true)
    return float(f.mean() if entities is None else f[entities].mean())


def blocking_report(C, true_s1, n_true, entities=None):
    """C: DataFrame with l, r. Oracle F0.5 = perfect matcher inside the candidates."""
    n_left = len(n_true)
    y = true_s1[C["r"].values] == C["l"].values
    sizes = np.bincount(C["l"].values, minlength=n_left)
    hit = np.bincount(C["l"].values[y], minlength=n_left)
    ent = np.arange(n_left) if entities is None else entities
    oracle = f05_arrays(hit, hit, n_true)[ent].mean()
    return {"pairs": int(len(C) if entities is None else sizes[ent].sum()),
            "pair_recall": round(float(hit[ent].sum() / max(n_true[ent].sum(), 1)), 5),
            "avg_cands": round(float(sizes[ent].mean()), 2),
            "p95_cands": float(np.percentile(sizes[ent], 95)),
            "zero_cand_share": round(float((sizes[ent] == 0).mean()), 5),
            "oracle_f05": round(float(oracle), 5)}


def guard(before, after, allowed=0.003):
    lost = before["oracle_f05"] - after["oracle_f05"]
    if lost > allowed:
        raise ValueError(f"pruning lost {lost:.4f} oracle F0.5 (allowed {allowed})")


def frozen_holdout(n_left, frac=0.1, seed=7):
    return np.random.default_rng(seed).random(n_left) < frac


def drift_report(p_oof, p_test, bins=(0, .1, .3, .5, .7, .9, 1.0)):
    h1 = np.histogram(p_oof, bins)[0] / max(len(p_oof), 1)
    h2 = np.histogram(p_test, bins)[0] / max(len(p_test), 1)
    return pd.DataFrame({"bucket": [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])],
                         "oof": h1.round(3), "test": h2.round(3)})
