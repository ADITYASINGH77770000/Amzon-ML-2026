"""Key-candidate matcher: exact-key candidate generation on the normalised fields (a key may
be shared by up to `cap` S1, unlike fast_fallback's unique keys) -> cheap similarity features ->
LightGBM -> one-owner decision tuned for macro F0.5 (singletons included).

    python key_matcher.py train   # 2-fold GroupKFold by S1 over ALL train S1: fold models, OOF
                                  # macro F0.5 and the decision thresholds
    python key_matcher.py test    # candidates + scores on test -> output/matching_results.tsv
                                  #                                + output/candidate_pairs.tsv
Test pairs are scored with the average of the two fold models (each trained on half of all
labelled S1); this is exactly how the submitted files were produced.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process

import config
from data import Side, truth_arrays
from evaluate import f05_arrays
from io_utils import write_tsv

T0 = time.time()
DIR = os.path.join(config.CACHE, "key_matcher")
OUT = os.path.join(config.ROOT, "output")

# (name, parts, cap): a key links a right record to every S1 carrying it, if at most `cap` do.
# "pre6:" / "suf6:" = first / last 6 characters (typos at the other end of the name).
KEYS = [("nj_street", ["name_join", "street_keys"], 3),
        ("nj_house", ["name_join", "house_no"], 3),
        ("nj_city", ["name_join", "city"], 5),
        ("nj_state", ["name_join", "state"], 5),
        ("nj", ["name_join"], 5),
        ("alt", ["name_core"], 5, ["name_alt"]),         # right record's alternate name = S1 name
        ("street_city", ["street_keys", "city"], 3),
        ("addr", ["addr_core"], 3),
        ("house_city", ["house_no", "city"], 3),
        ("pre6_house", ["pre6:name_join", "house_no"], 3),
        ("suf6_house", ["suf6:name_join", "house_no"], 3),
        ("pre6_city", ["pre6:name_join", "city"], 3),
        ("pre6_postal", ["pre6:name_join", "postal"], 3)]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=4, verbose=-1, bagging_seed=7, feature_fraction_seed=7)
ROUNDS = 300


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *a, flush=True)


def _col(t, p):
    if p.startswith("pre6:"):
        return pc.utf8_slice_codeunits(t[p[5:]], 0, 6)
    if p.startswith("suf6:"):
        return pc.utf8_slice_codeunits(pc.utf8_reverse(t[p[5:]]), 0, 6)
    return t[p]


def _key(t, parts):
    k = pc.binary_join_element_wise(t["country"], *[_col(t, p) for p in parts], "|")
    ok = None
    for p in parts:
        c = pc.not_equal(t[p.split(":")[-1]], "")
        ok = c if ok is None else pc.and_(ok, c)
    return pc.if_else(ok, k, pa.scalar(None, pa.string()))


def _codes(kl, kr, nl):
    both = pa.chunked_array(kl.chunks + kr.chunks)
    idx = pc.fill_null(pc.dictionary_encode(both).combine_chunks().indices, -1)
    idx = idx.to_numpy(zero_copy_only=False).astype(np.int64)
    return idx[:nl], idx[nl:]


def key_pairs(left, right, parts, cap, rparts=None):
    cl, cr = _codes(_key(left.t, parts), _key(right.t, rparts or parts), left.n)
    ncode = int(max(cl.max(), cr.max())) + 1
    cnt = np.bincount(cl[cl >= 0], minlength=ncode)
    r = np.flatnonzero(cr >= 0)
    m = cnt[cr[r]]
    keep = (m >= 1) & (m <= cap)
    r, m = r[keep], m[keep]
    order = np.argsort(cl, kind="stable")
    st = np.searchsorted(cl[order], cr[r])
    rr = np.repeat(r, m)
    off = np.arange(len(rr)) - np.repeat(np.cumsum(m) - m, m)
    ll = order[np.repeat(st, m) + off]
    return ll, rr, np.repeat(m, m)


def candidates(left, right):
    P, K, M = [], [], []
    for j, (name, parts, cap, *rp) in enumerate(KEYS):
        ll, rr, mm = key_pairs(left, right, parts, cap, rp[0] if rp else None)
        P.append(ll.astype(np.int64) * right.n + rr)
        K.append(np.full(len(ll), j, np.int8))
        M.append(np.minimum(mm, 100).astype(np.int8))
        log(f"  key {name}: {len(ll):,} pairs")
    P, K, M = np.concatenate(P), np.concatenate(K), np.concatenate(M)
    o = np.argsort(P, kind="stable")
    P, K, M = P[o], K[o], M[o]
    del o
    starts = np.flatnonzero(np.r_[True, P[1:] != P[:-1]])
    bits = np.bitwise_or.reduceat(np.left_shift(1, K.astype(np.int32)), starts)
    amb = np.minimum.reduceat(M, starts)
    pid = P[starts]
    return (pid // right.n).astype(np.int64), (pid % right.n).astype(np.int64), bits, amb


def _eq(left, right, col, L, R):
    cl, cr = _codes(pc.if_else(pc.not_equal(left.t[col], ""), left.t[col], pa.scalar(None, pa.string())),
                    pc.if_else(pc.not_equal(right.t[col], ""), right.t[col], pa.scalar(None, pa.string())), left.n)
    a, b = cl[L], cr[R]
    both = (a >= 0) & (b >= 0)
    return both & (a == b), both & (a != b), (a < 0) | (b < 0)


def _flag(side, col, idx):
    c = side.t[col]
    if pa.types.is_string(c.type) or pa.types.is_large_string(c.type):
        c = pc.not_equal(c, "")
    return np.asarray(c.to_numpy(zero_copy_only=False))[idx].astype(np.float32)


def _grp_gap(g, v):
    return v - pd.Series(v).groupby(g).transform("max").values


def features(left, right, L, R, bits, amb, path):
    """Pair features into a column-major float32 memmap at `path` (columns are written whole)."""
    n = len(L)
    names = [f"k_{k[0]}" for k in KEYS] + [
        "n_keys", "amb", "n_cand_l", "n_cand_r", "nc_tset", "nj_ratio", "nj_partial", "nn_tsort",
        "ad_tset", "ad_ratio", "nc_tset_gap_r", "ad_tset_gap_r", "nc_tset_gap_l", "ad_tset_gap_l",
        "house_eq", "house_cf", "city_eq", "city_cf", "state_eq", "state_cf", "postal_eq", "postal_cf",
        "postal_miss", "addr_empty_a", "addr_empty_b", "is_web_b", "native_b", "is_s3", "len_a", "len_b"]
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, len(names)), fortran_order=True)
    col = {c: i for i, c in enumerate(names)}
    for j in range(len(KEYS)):
        X[:, j] = (bits >> j) & 1
    X[:, col["n_keys"]] = np.bitwise_count(bits.astype(np.uint32)) if hasattr(np, "bitwise_count") else \
        sum(((bits >> j) & 1) for j in range(len(KEYS)))
    X[:, col["amb"]] = amb
    X[:, col["n_cand_l"]] = np.bincount(L, minlength=left.n)[L]
    X[:, col["n_cand_r"]] = np.bincount(R, minlength=right.n)[R]
    sims = [("nc_tset", "name_core", fuzz.token_set_ratio), ("nj_ratio", "name_join", fuzz.ratio),
            ("nj_partial", "name_join", fuzz.partial_ratio), ("nn_tsort", "name_norm", fuzz.token_sort_ratio),
            ("ad_tset", "addr_core", fuzz.token_set_ratio), ("ad_ratio", "addr_core", fuzz.ratio)]
    step = 1_000_000
    for s in range(0, n, step):
        li, ri = pa.array(L[s:s + step]), pa.array(R[s:s + step])
        cache = {}
        for fname, c, scorer in sims:
            if c not in cache:
                cache[c] = (left.t[c].take(li).to_pylist(), right.t[c].take(ri).to_pylist())
            a, b = cache[c]
            X[s:s + step, col[fname]] = process.cpdist(a, b, scorer=scorer, workers=-1)
    log(f"  string similarities for {n:,} pairs")
    for fname, src in (("nc_tset_gap_r", "nc_tset"), ("ad_tset_gap_r", "ad_tset")):
        X[:, col[fname]] = _grp_gap(R, X[:, col[src]])
    for fname, src in (("nc_tset_gap_l", "nc_tset"), ("ad_tset_gap_l", "ad_tset")):
        X[:, col[fname]] = _grp_gap(L, X[:, col[src]])
    for c in ("house_no", "city", "state", "postal"):
        eq, cf, miss = _eq(left, right, c, L, R)
        short = c.split("_")[0]
        X[:, col[f"{short}_eq"]], X[:, col[f"{short}_cf"]] = eq, cf
        if c == "postal":
            X[:, col["postal_miss"]] = miss
    X[:, col["addr_empty_a"]] = _flag(left, "addr_empty", L)
    X[:, col["addr_empty_b"]] = _flag(right, "addr_empty", R)
    X[:, col["is_web_b"]] = _flag(right, "is_web", R)
    X[:, col["native_b"]] = _flag(right, "name_native", R)
    X[:, col["is_s3"]] = R >= right.sizes[0]
    X[:, col["len_a"]] = pc.utf8_length(left.t["name_join"]).to_numpy(zero_copy_only=False)[L]
    X[:, col["len_b"]] = pc.utf8_length(right.t["name_join"]).to_numpy(zero_copy_only=False)[R]
    X.flush()
    log(f"  features done: {X.shape}")
    return X, names


def predict(model, X, step=1_000_000):
    return np.concatenate([model.predict(np.ascontiguousarray(X[s:s + step]), num_threads=4)
                           for s in range(0, len(X), step)])


def owner_mask(L, R, p):
    """Each right record keeps only its best-scoring S1."""
    o = np.lexsort((-p, R))
    first = np.r_[True, R[o][1:] != R[o][:-1]]
    own = np.zeros(len(p), bool)
    own[o[first]] = True
    return own


def prep(L, R, p, nL):
    """Owner mask, owned score q (-1 if not owned) and each S1's best owned score, per pair."""
    own = owner_mask(L, R, p)
    q = np.where(own, p, -1.0)
    best = np.full(nL, -1.0)
    o = np.lexsort((-q, L))
    first = o[np.r_[True, L[o][1:] != L[o][:-1]]]
    best[L[first]] = q[first]
    return own, q, best[L]


def decide(p, own, q, bestL, t_open, t_add):
    return own & (bestL >= t_open) & ((p >= t_add) | (q == bestL))


def tune(L, R, p, y, n_true, c1, lab):
    nL = len(n_true)
    own, q, bestL = prep(L, R, p, nL)
    res = []
    for to in np.arange(0.2, 0.86, 0.05):
        for ta in np.arange(0.2, 0.96, 0.05):
            if ta < to - 1e-9:
                continue
            m = decide(p, own, q, bestL, to, ta)
            n_pred = np.bincount(L[m], minlength=nL)
            tp = np.bincount(L[m], weights=y[m].astype(float), minlength=nL)
            f = f05_arrays(n_pred, tp, n_true)
            row = dict(t_open=round(float(to), 2), t_add=round(float(ta), 2), all=float(f.mean()))
            for code, name in enumerate(lab):
                row[name] = float(f[c1 == code].mean())
            res.append(row)
    best = {"global": max(res, key=lambda r: r["all"])}
    for name in lab:
        best[name] = max(res, key=lambda r: r[name])
    return best


def train():
    os.makedirs(DIR, exist_ok=True)
    left, right = Side(["train_source1"]), Side(["train_source2", "train_source3"])
    true_s1, n_true = truth_arrays(left, right)
    log(f"train: {left.n:,} S1, {right.n:,} S2+S3, {int(n_true.sum()):,} true pairs")
    L, R, bits, amb = candidates(left, right)
    y = true_s1[R] == L
    log(f"candidates: {len(L):,} pairs, {len(L) / left.n:.2f} per S1, pair recall {y.sum() / n_true.sum():.4f}")
    fold = np.random.default_rng(2026).integers(0, 2, left.n).astype(np.int8)   # GroupKFold by S1
    X, names = features(left, right, L, R, bits, amb, os.path.join(DIR, "X_train.npy"))
    c1, lab = left.codes("country")
    full = lgb.Dataset(X, y.astype(np.float32), feature_name=names, params={"max_bin": 63, "verbose": -1},
                       free_raw_data=False).construct()
    log("  binned dataset constructed")
    p = np.zeros(len(L))
    fl = fold[L]
    for j in range(2):
        tr, te = np.flatnonzero(fl != j), np.flatnonzero(fl == j)
        m = lgb.train(PARAMS, full.subset(tr.astype(np.int32)), ROUNDS)
        p[te] = np.concatenate([m.predict(np.asarray(X[te[s:s + 1_000_000]]), num_threads=4)
                                for s in range(0, len(te), 1_000_000)])
        m.save_model(os.path.join(DIR, f"fold{j}.txt"))
        log(f"  fold model {j}: trained on {len(tr):,} pairs, scored {len(te):,}")
    np.save(os.path.join(DIR, "p_oof.npy"), p)
    best = tune(L, R, p, y, n_true, c1, list(lab))
    log("OOF macro F0.5:", json.dumps(best))
    json.dump({"thresholds": best, "names": names, "n_pairs": int(len(L)),
               "pair_recall": float(y.sum() / n_true.sum())}, open(os.path.join(DIR, "eval.json"), "w"), indent=1)


def test():
    ev = json.load(open(os.path.join(DIR, "eval.json")))
    th = ev["thresholds"]
    left, right = Side(["test_source1"]), Side(["test_source2", "test_source3"])
    log(f"test: {left.n:,} S1, {right.n:,} S2+S3")
    L, R, bits, amb = candidates(left, right)
    log(f"candidates: {len(L):,} pairs")
    X, names = features(left, right, L, R, bits, amb, os.path.join(DIR, "X_test.npy"))
    files = [os.path.join(DIR, f"fold{j}.txt") for j in range(2)]   # each saw half of all labelled S1
    log("  scoring with", [os.path.basename(f) for f in files])
    p = np.mean([predict(lgb.Booster(model_file=f), X) for f in files], axis=0)
    del X
    np.save(os.path.join(DIR, "p_test.npy"), p)
    c1, lab = left.codes("country")
    own, q, bestL = prep(L, R, p, left.n)
    keep = np.zeros(len(L), bool)
    for code, name in enumerate(lab):
        t = th.get(name, th["global"])
        k = decide(p, own, q, bestL, t["t_open"], t["t_add"])
        keep |= k & (c1[L] == code)
        log(f"  {name}: t_open {t['t_open']} t_add {t['t_add']}")
    s1_ids = left.list("entity_id")
    os.makedirs(OUT, exist_ok=True)
    for fname, colname, m in (("matching_results.tsv", "matched_entity_ids", keep),
                              ("candidate_pairs.tsv", "candidate_entity_ids", np.ones(len(L), bool))):
        idx = np.flatnonzero(m)
        r_ids = right.list("entity_id", R[idx])
        out = {}
        for a, b in zip(L[idx].tolist(), r_ids):
            out.setdefault(s1_ids[a], []).append(b)
        write_tsv(s1_ids, out, os.path.join(OUT, fname), colname)
        log(f"  wrote {fname}: {len(idx):,} pairs for {len(out):,} S1")


def cands():
    """Candidate recall / oracle F0.5 on train (no model)."""
    left, right = Side(["train_source1"]), Side(["train_source2", "train_source3"])
    true_s1, n_true = truth_arrays(left, right)
    L, R, bits, amb = candidates(left, right)
    y = true_s1[R] == L
    tp = np.bincount(L[y], minlength=left.n)
    f = f05_arrays(tp, tp.astype(float), n_true)
    log(f"candidates {len(L):,} ({len(L) / left.n:.2f}/S1), pair recall {y.sum() / n_true.sum():.4f}, "
        f"oracle F0.5 {f.mean():.4f}")
    for j, (name, *_) in enumerate(KEYS):
        h = ((bits >> j) & 1).astype(bool)
        only = h & (bits == (1 << j))
        log(f"  {name:12s} pairs {h.sum():>11,}  prec {y[h].mean():.3f}  unique-to-key true {int(y[only].sum()):,}")


if __name__ == "__main__":
    {"train": train, "test": test, "cands": cands}[sys.argv[1]]()
