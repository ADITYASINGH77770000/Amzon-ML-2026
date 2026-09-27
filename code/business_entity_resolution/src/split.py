"""Stratified train / validation / holdout split of the labelled Source 1 entities.

Strata: country x number of true matches (0..6, 7+) x "has a matched record with
an empty address" x "has a website-like matched record". Inside each stratum the
entities are shuffled with a fixed seed and dealt 80 / 10 / 10, so every
partition keeps the same mix of hard cases. Only S1 entities are split; the
S2/S3 pool stays whole, exactly as in test."""
import os

import numpy as np
import pandas as pd

import config

SPLIT_PATH = os.path.join(config.MODELS, "split.npy")          # per mode (subsets differ)
TRAIN, VALID, HOLDOUT = 0, 1, 2


def _stratum_ranks(left, right, true_s1, n_true, seed):
    """Random rank of every S1 inside its stratum (country x match count x has an
    empty-address match x has a website-like match), and the stratum size."""
    country, _ = left.codes("country")
    matched = np.flatnonzero(true_s1 >= 0)
    owner = true_s1[matched]
    empty = np.zeros(left.n, bool)
    web = np.zeros(left.n, bool)
    empty[owner[right.np("addr_empty", matched).astype(bool)]] = True
    web[owner[right.np("is_web", matched).astype(bool)]] = True
    strata = pd.DataFrame({"c": country, "k": np.minimum(n_true, 7), "e": empty, "w": web})
    key = strata.groupby(["c", "k", "e", "w"]).ngroup().values
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(left.n), key))            # shuffled within stratum
    starts = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1]
    sizes = np.diff(np.r_[starts, left.n])
    pos, size = np.empty(left.n, np.int64), np.empty(left.n, np.int64)
    pos[order] = np.arange(left.n) - np.repeat(starts, sizes)   # rank inside stratum
    size[order] = np.repeat(sizes, sizes)
    return pos, size


def make_group_kfold(left, right, true_s1, n_true, k, seed=config.SEED + 5):
    """GroupKFold by S1 entity, stratified: fold[l] in 0..k-1 (whole entities per fold)."""
    pos, _ = _stratum_ranks(left, right, true_s1, n_true, seed)
    return (pos % k).astype(np.int8)


def load_folds(left, right, true_s1, n_true, k=None):
    """Fold of every labelled S1 for cross-fitting / out-of-fold evaluation.
    k == 2 keeps the historical halves (the FINAL-mode filter models were fitted on them)."""
    k = k or config.N_FOLDS
    if k == 2:
        from crossfit import make_folds
        part = load_or_make(left, right, true_s1, n_true)
        f = make_folds(part)
        return np.where((f == 0) | (part == VALID), 0, 1).astype(np.int8)
    path = os.path.join(config.MODELS, f"folds_k{k}.npy")
    if os.path.exists(path):
        fold = np.load(path)
        if len(fold) == left.n:
            return fold
    fold = make_group_kfold(left, right, true_s1, n_true, k)
    os.makedirs(config.MODELS, exist_ok=True)
    np.save(path, fold)
    return fold


def make_split(left, right, true_s1, n_true, fracs=(0.8, 0.1, 0.1), seed=config.SEED):
    pos, size = _stratum_ranks(left, right, true_s1, n_true, seed)
    frac = (pos + 0.5) / size
    part = np.full(left.n, TRAIN, np.int8)
    part[frac >= fracs[0]] = VALID
    part[frac >= fracs[0] + fracs[1]] = HOLDOUT
    return part


def load_or_make(left, right, true_s1, n_true):
    if os.path.exists(SPLIT_PATH):
        part = np.load(SPLIT_PATH)
        if len(part) == left.n:
            return part
    part = make_split(left, right, true_s1, n_true)
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    np.save(SPLIT_PATH, part)
    return part
