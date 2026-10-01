"""Stage 2 of the key matcher: rival features on the Stage-1 scores (rank / gap / score mass
per S1 and per right record) + a few Stage-1 features -> LightGBM -> the same decision.
Reuses key_matcher's saved feature matrices and scores (cache/key_matcher).

    python key_stage2.py train    # 2-fold OOF (same S1 folds as Stage 1) + thresholds
    python key_stage2.py test     # -> output_km2/*.tsv
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
import key_matcher as km
from data import Side, truth_arrays
from io_utils import write_tsv

DIR = km.DIR
OUT = os.path.join(config.ROOT, "output_km2")
KEEP = ["n_keys", "amb", "n_cand_l", "n_cand_r", "nc_tset", "nj_ratio", "ad_tset", "house_cf", "house_eq",
        "addr_empty_a", "addr_empty_b", "is_s3", "city_cf", "postal_cf"]
RIV = ["p", "pmax_r", "gap_r", "rank_r", "psum_r", "pmax_l", "gap_l", "rank_l", "psum_l", "p2_l", "gap2_l",
       "nhi_l", "own", "own_sum_l", "own_nhi_l"]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.7, bagging_freq=1, num_threads=4, verbose=-1, bagging_seed=7,
              feature_fraction_seed=7)
ROUNDS = 150
log = km.log


def build(L, R, p, X, names):
    cols = RIV + KEEP
    F = np.empty((len(L), len(cols)), np.float32, order="F")
    c = {k: i for i, k in enumerate(cols)}
    s = pd.Series(p)
    sR, sL = s.groupby(R), s.groupby(L)
    F[:, c["p"]] = p
    F[:, c["pmax_r"]] = sR.transform("max").values
    F[:, c["gap_r"]] = p - F[:, c["pmax_r"]]
    F[:, c["rank_r"]] = sR.rank(ascending=False, method="first").values
    F[:, c["psum_r"]] = sR.transform("sum").values
    F[:, c["pmax_l"]] = sL.transform("max").values
    F[:, c["gap_l"]] = p - F[:, c["pmax_l"]]
    rl = sL.rank(ascending=False, method="first").values
    F[:, c["rank_l"]] = rl
    F[:, c["psum_l"]] = sL.transform("sum").values
    p2 = pd.Series(np.where(rl == 2, p, 0.0)).groupby(L).transform("max").values
    F[:, c["p2_l"]] = p2
    F[:, c["gap2_l"]] = F[:, c["pmax_l"]] - p2
    F[:, c["nhi_l"]] = pd.Series((p >= 0.5).astype(np.int8)).groupby(L).transform("sum").values
    own = km.owner_mask(L, R, p)
    F[:, c["own"]] = own
    F[:, c["own_sum_l"]] = pd.Series(np.where(own, p, 0.0)).groupby(L).transform("sum").values
    F[:, c["own_nhi_l"]] = pd.Series((own & (p >= 0.5)).astype(np.int8)).groupby(L).transform("sum").values
    for k in KEEP:
        F[:, c[k]] = X[:, names.index(k)]
    log(f"  stage-2 features {F.shape}")
    return F, cols


def predict(m, F, rows=None, step=1_000_000):
    n = len(F) if rows is None else len(rows)
    out = []
    for s in range(0, n, step):
        part = F[s:s + step] if rows is None else F[rows[s:s + step]]
        out.append(m.predict(np.ascontiguousarray(part), num_threads=4))
    return np.concatenate(out)


def train():
    left, right = Side(["train_source1"]), Side(["train_source2", "train_source3"])
    true_s1, n_true = truth_arrays(left, right)
    L, R, bits, amb = km.candidates(left, right)
    y = true_s1[R] == L
    p = np.load(os.path.join(DIR, "p_oof.npy"))
    assert len(p) == len(L), (len(p), len(L))
    ev1 = json.load(open(os.path.join(DIR, "eval.json")))
    X = np.load(os.path.join(DIR, "X_train.npy"), mmap_mode="r")
    F, cols = build(L, R, p, X, ev1["names"])
    del X
    fold = np.random.default_rng(2026).integers(0, 2, left.n).astype(np.int8)   # Stage-1 folds
    fl = fold[L]
    ds = lgb.Dataset(F, y.astype(np.float32), feature_name=cols, params={"max_bin": 63, "verbose": -1},
                     free_raw_data=False).construct()
    p2 = np.zeros(len(L))
    for j in range(2):
        tr, te = np.flatnonzero(fl != j), np.flatnonzero(fl == j)
        m = lgb.train(PARAMS, ds.subset(tr.astype(np.int32)), ROUNDS)
        p2[te] = predict(m, F, te)
        m.save_model(os.path.join(DIR, f"s2_fold{j}.txt"))
        log(f"  stage-2 fold {j} done")
    c1, lab = left.codes("country")
    best = km.tune(L, R, p2, y, n_true, c1, list(lab))
    log("STAGE-2 OOF macro F0.5:", json.dumps(best))
    log("stage-1 was:", json.dumps(ev1["thresholds"]["global"]))
    json.dump({"thresholds": best, "cols": cols}, open(os.path.join(DIR, "s2_eval.json"), "w"), indent=1)


def test():
    th = json.load(open(os.path.join(DIR, "s2_eval.json")))["thresholds"]
    names = json.load(open(os.path.join(DIR, "eval.json")))["names"]
    left, right = Side(["test_source1"]), Side(["test_source2", "test_source3"])
    L, R, bits, amb = km.candidates(left, right)
    p = np.load(os.path.join(DIR, "p_test.npy"))
    assert len(p) == len(L), (len(p), len(L))
    X = np.load(os.path.join(DIR, "X_test.npy"), mmap_mode="r")
    F, cols = build(L, R, p, X, names)
    del X
    p2 = np.mean([predict(lgb.Booster(model_file=os.path.join(DIR, f"s2_fold{j}.txt")), F) for j in range(2)], axis=0)
    del F
    c1, lab = left.codes("country")
    own, q, bestL = km.prep(L, R, p2, left.n)
    keep = np.zeros(len(L), bool)
    for code, name in enumerate(lab):
        t = th.get(name, th["global"])
        keep |= km.decide(p2, own, q, bestL, t["t_open"], t["t_add"]) & (c1[L] == code)
        log(f"  {name}: t_open {t['t_open']} t_add {t['t_add']}")
    s1_ids = left.list("entity_id")
    os.makedirs(OUT, exist_ok=True)
    # candidates are identical to key_matcher's (same keys), so output_km/candidate_pairs.tsv is reused
    for fname, colname, m in (("matching_results.tsv", "matched_entity_ids", keep),):
        idx = np.flatnonzero(m)
        r_ids = right.list("entity_id", R[idx])
        out = {}
        for a, b in zip(L[idx].tolist(), r_ids):
            out.setdefault(s1_ids[a], []).append(b)
        write_tsv(s1_ids, out, os.path.join(OUT, fname), colname)
        log(f"  wrote {fname}: {len(idx):,} pairs for {len(out):,} S1")


if __name__ == "__main__":
    {"train": train, "test": test}[sys.argv[1]]()
