"""Stage 4: decision layer. Only removes pairs, so matches are a subset of candidates.

Rules, applied in order:
  1. assignment (optional, tuned): each S2/S3 record keeps only its best-scoring S1 —
     every record belongs to at most one S1 in this data;
  2. an S1 opens only if its best remaining pair has p >= t_open;
  3. further pairs are added if p >= t_add and p >= rel * best.
Thresholds are tuned coarse (0.05) then fine (0.01) on macro F0.5."""
import itertools

import numpy as np

from evaluate import f05_arrays


def group_max(key, v):
    """Per-row maximum of v over the rows sharing the same key (numpy, no pandas)."""
    key = np.asarray(key)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    size = np.diff(np.r_[start, len(ks)])
    out = np.empty(len(v), np.float64)
    out[order] = np.repeat(np.maximum.reduceat(np.asarray(v, np.float64)[order], start), size)
    return out


def assign_mask(r, p):
    """Each S2/S3 record keeps only its best S1 (ties keep all)."""
    return p >= group_max(r, p)


def _prepared(l, p, pre):
    pp = np.where(pre, p, -1.0) if pre is not None else np.asarray(p, np.float64)
    return pp, group_max(l, pp)


def _mask(pp, best, t_open, t_add, rel):
    return (best >= t_open) & ((pp >= t_add) | (pp == best)) & (pp >= rel * best) & (pp >= 0)


def decide_mask(l, p, t_open, t_add, rel=0.0, pre=None):
    """pre: optional boolean mask applied first (e.g. the assignment mask)."""
    pp, best = _prepared(l, p, pre)
    return _mask(pp, best, t_open, t_add, rel)


def score(l, y, mask, n_true, entities):
    """Macro F0.5 over `entities` (S1 rows) for the pairs selected by mask."""
    n = len(n_true)
    n_pred = np.bincount(l[mask], minlength=n)
    tp = np.bincount(l[mask], weights=y[mask].astype(float), minlength=n)
    return float(f05_arrays(n_pred, tp, n_true)[entities].mean())


def _grid_search(l, y, n_true, entities, pp, best, opens, adds, rels):
    top = (-1.0, None)
    for t_open, t_add, rel in itertools.product(opens, adds, rels):
        s = score(l, y, _mask(pp, best, t_open, t_add, rel), n_true, entities)
        if s > top[0]:
            top = (s, dict(t_open=round(float(t_open), 3), t_add=round(float(t_add), 3), rel=float(rel)))
    return top


def tune_rows(l, p, y, n_true, entities, assign_pre, log=print, name=""):
    """Coarse (0.05) then fine (0.01) search, with and without assignment.

    l, p, y: the pairs of the `entities` being tuned on; assign_pre: the assignment
    mask on those same rows, computed by the caller over ALL S1 so that competitors
    from other S1 count exactly as they will on test."""
    results = {}
    for use_assign in (False, True):
        pp, best = _prepared(l, p, assign_pre if use_assign else None)   # once per option
        coarse = np.round(np.arange(0.10, 0.96, 0.05), 3)
        s, cfg = _grid_search(l, y, n_true, entities, pp, best, coarse, coarse, (0.0, 0.3, 0.5))
        fo = np.round(np.arange(max(cfg["t_open"] - 0.05, 0.01), min(cfg["t_open"] + 0.051, 0.99), 0.01), 3)
        fa = np.round(np.arange(max(cfg["t_add"] - 0.05, 0.01), min(cfg["t_add"] + 0.051, 0.99), 0.01), 3)
        fr = sorted({max(cfg["rel"] - 0.1, 0.0), cfg["rel"], min(cfg["rel"] + 0.1, 0.9)})
        s, cfg = _grid_search(l, y, n_true, entities, pp, best, fo, fa, fr)
        cfg["assign"] = use_assign
        results[use_assign] = (s, cfg)
        log(f"  {name} assign={use_assign}: macro F0.5 {s:.5f} with {cfg}")
    return max(results.values(), key=lambda x: x[0])


def apply(l, r, p, cfg, assign_all=None):
    """Final mask for a decision config; assign_all = assignment mask (if cfg asks)."""
    pre = assign_all if cfg.get("assign") else None
    return decide_mask(l, p, cfg["t_open"], cfg["t_add"], cfg.get("rel", 0.0), pre)
