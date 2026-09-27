"""Recall lab: tune blocking passes on a validation dev sample before any full run.

Each pass is queried once with generous settings (k=100, no score floor) and
the hits are cached; tighter k / min_score settings and pass subsets are then
simulated offline. Reports pair recall, candidate-oracle macro F0.5 and
candidate counts per country, plus a categorised sample of the misses.

    python recall_lab.py query            # cache hits for every pass
    python recall_lab.py report           # compare settings, print misses
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

import config
import split as splitmod
from blocking import EXTRA_PASSES, PASSES, field_texts, index_tag
from candidates import PassIndex
from data import load_split, truth_arrays
from evaluate import f05_arrays

LAB = os.path.join(config.CACHE, f"lab_v{config.NORM_VERSION}")
K_LAB = 100
ALL = {**PASSES, **EXTRA_PASSES}


def setup(n_dev=20000):
    left, right = load_split("train")
    true_s1, n_true = truth_arrays(left, right)
    part = splitmod.load_or_make(left, right, true_s1, n_true)
    dev_path = os.path.join(LAB, "dev.npy")
    if os.path.exists(dev_path):
        dev = np.load(dev_path)
    else:
        valid = np.flatnonzero(part == splitmod.VALID)
        dev = np.sort(np.random.default_rng(config.SEED).choice(valid, n_dev, replace=False))
        os.makedirs(LAB, exist_ok=True)
        np.save(dev_path, dev)
    return left, right, true_s1, n_true, part, dev


def hits_path(p, country):
    return os.path.join(LAB, f"{country}_{index_tag(p, ALL[p])}.parquet")


def query(passes):
    left, right, true_s1, n_true, part, dev = setup()
    c1, lab1 = left.codes("country")
    c2, lab2 = right.codes("country")
    for code in np.unique(c1[dev]):
        country = lab1[code]
        Li = dev[c1[dev] == code]
        Ri = np.flatnonzero(c2 == lab2.index(country))
        for p in passes:
            out = hits_path(p, country)
            if os.path.exists(out):
                continue
            cfg = dict(ALL[p], k=K_LAB if not ALL[p].get("reverse") else 30, min_score=0.0)
            path = os.path.join(config.INDEX, "train", f"{country}_{index_tag(p, cfg)}")
            t = time.time()
            if cfg.get("reverse"):
                # index every S1 of the country; query the filtered right records
                Lall = np.flatnonzero(c1 == code)
                Rs = Ri[right.np(cfg["right_filter"], Ri).astype(bool)]
                idx = PassIndex(lambda: field_texts(left, cfg["field"], Lall), cfg, path)
                tb = time.time() - t
                t = time.time()
                i, j, v, rank = idx.query(field_texts(right, cfg["field"], Rs))   # rank inside each right record
                rank = rank.astype(np.int16)
                l, r = Lall[j], Rs[i]
                keep = np.isin(l, Li)
                l, r, v, rank = l[keep], r[keep], v[keep], rank[keep]
            else:
                idx = PassIndex(lambda: field_texts(right, cfg["field"], Ri), cfg, path)
                tb = time.time() - t
                t = time.time()
                i, j, v, rank = idx.query(field_texts(left, cfg["field"], Li))
                rank = rank.astype(np.int16)
                l, r = Li[i], Ri[j]
            pd.DataFrame({"l": l.astype(np.int32), "r": r.astype(np.int32), "score": v,
                          "rank": rank}).to_parquet(out)
            i = l
            print(f"{country} {p}: index {tb:.0f}s, query {time.time() - t:.0f}s for {len(Li):,} rows, "
                  f"{len(i) / len(Li):.1f} hits/row", flush=True)
            del idx


def load_hits(settings, country):
    frames = []
    for p, (k, ms) in settings.items():
        h = pd.read_parquet(hits_path(p, country))
        frames.append(h[(h["rank"] < k) & (h["score"] >= ms)][["l", "r"]])
    C = pd.concat(frames, ignore_index=True)
    key = (C["l"].values.astype(np.int64) << 32) | C["r"].values
    key = np.unique(key)
    return (key >> 32).astype(np.int64), (key & 0xFFFFFFFF).astype(np.int64)


def score(settings, left, true_s1, n_true, dev):
    c1, lab1 = left.codes("country")
    res = {}
    tot_hit = tot_true = 0
    f_all = []
    for code in np.unique(c1[dev]):
        country = lab1[code]
        Li = dev[c1[dev] == code]
        l, r = load_hits(settings, country)
        y = true_s1[r] == l
        hit = np.bincount(l[y], minlength=left.n)[Li]
        cnt = np.bincount(l, minlength=left.n)[Li]
        f = f05_arrays(hit, hit, n_true[Li])
        f_all.append(f)
        res[country] = dict(recall=round(hit.sum() / max(n_true[Li].sum(), 1), 4),
                            oracle=round(float(f.mean()), 5), cands=round(float(cnt.mean()), 1))
        tot_hit += hit.sum()
        tot_true += n_true[Li].sum()
    res["all"] = dict(recall=round(tot_hit / tot_true, 4), oracle=round(float(np.concatenate(f_all).mean()), 5))
    return res


def misses(settings, left, right, true_s1, n_true, dev, country, n_show=30):
    c1, lab1 = left.codes("country")
    Li = dev[c1[dev] == lab1.index(country)]
    l, r = load_hits(settings, country)
    got = set(zip(l.tolist(), r.tolist()))
    inL = np.zeros(left.n, bool)
    inL[Li] = True
    tr = np.flatnonzero((true_s1 >= 0) & inL[np.maximum(true_s1, 0)])
    miss = [(int(true_s1[x]), int(x)) for x in tr if (int(true_s1[x]), int(x)) not in got]
    if not miss:
        return
    a = np.array([m[0] for m in miss])
    b = np.array([m[1] for m in miss])
    na, nb = left.list("name_core", a), right.list("name_core", b)
    tset = process.cpdist(na, nb, scorer=fuzz.token_set_ratio, workers=-1)
    cat = pd.DataFrame({
        "right_addr_empty": right.np("addr_empty", b).astype(bool),
        "right_web": right.np("is_web", b).astype(bool),
        "right_native": right.np("name_native", b).astype(bool),
        "name_tset>=90": tset >= 90,
        "name_tset<50": tset < 50,
    })
    print(f"\n{country}: {len(miss):,} missed true pairs of {len(tr):,}")
    print("  share by category:", cat.mean().round(3).to_dict())
    print("  addr empty & name>=90:", round(float((cat['right_addr_empty'] & cat['name_tset>=90']).mean()), 3),
          " addr empty & name<90:", round(float((cat['right_addr_empty'] & ~cat['name_tset>=90']).mean()), 3))
    rng = np.random.default_rng(3)
    for k in rng.choice(len(miss), min(n_show, len(miss)), replace=False):
        print(f"  S1: {na[k]} | {left.list('addr_core', [a[k]])[0]}")
        print(f"   R: {nb[k]} | {right.list('addr_core', [b[k]])[0]}   (tset {tset[k]:.0f})")


def wide_hits(settings, country):
    """Union of pass hits with one score and one rank column per pass (for Stage 1.5)."""
    frames = []
    for p, (k, ms) in settings.items():
        h = pd.read_parquet(hits_path(p, country))
        h = h[(h["rank"] < k) & (h["score"] >= ms)]
        frames.append(h.assign(p=p))
    H = pd.concat(frames, ignore_index=True)
    W = H.pivot_table(index=["l", "r"], columns="p", values=["score", "rank"], aggfunc="min", observed=True)
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    for p in settings:
        if f"score_{p}" not in W:
            W[f"score_{p}"] = np.nan
            W[f"rank_{p}"] = np.nan
    W["n_passes"] = W[[f"score_{p}" for p in settings]].notna().sum(axis=1)
    return W


def prerank(settings, left, true_s1, n_true, dev, budgets=(20, 30, 50, 80, 120), right=None):
    """Stage 1.5: LightGBM on pass scores / ranks (+ record-signature agreement when
    `right` is given), 2-fold by S1, then candidate-oracle F0.5 at the top-M per S1."""
    import lightgbm as lgb
    c1, lab1 = left.codes("country")
    sig = None
    if right is not None:
        from signatures import pair_signature_features, side_signatures
        sig = side_signatures("train", left, right)
    rows = []
    for code in np.unique(c1[dev]):
        country = lab1[code]
        Li = dev[c1[dev] == code]
        W = wide_hits(settings, country)
        if sig is not None:
            for k, v in pair_signature_features(sig[0].take(W["l"].values), sig[1].take(W["r"].values)).items():
                W[k] = v
        y = (true_s1[W["r"].values] == W["l"].values).astype(np.int8)
        X = W.drop(columns=["l", "r"]).astype(np.float32)
        fold = W["l"].values.astype(np.int64) % 2           # dev S1 rows are a random sample
        p = np.zeros(len(W))
        from monotone import params_for
        prm = params_for(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200,
                              lambda_l2=10.0, verbose=-1, num_threads=config.N_JOBS, seed=config.SEED),
                         list(X.columns))
        for f in (0, 1):
            m = lgb.train(prm, lgb.Dataset(X[fold != f], y[fold != f]), 200)
            p[fold == f] = m.predict(X[fold == f])
        W["p"] = p
        W = W.sort_values(["l", "p"], ascending=[True, False])
        W["rk"] = W.groupby("l").cumcount()
        for M in budgets + (10 ** 6,):
            Q = W[W["rk"] < M]
            yy = true_s1[Q["r"].values] == Q["l"].values
            hit = np.bincount(Q["l"].values[yy], minlength=left.n)[Li]
            cnt = np.bincount(Q["l"].values, minlength=left.n)[Li]
            f05 = f05_arrays(hit, hit, n_true[Li])
            rows.append(dict(country=country, M=M if M < 10 ** 6 else "all", oracle=round(float(f05.mean()), 5),
                             recall=round(hit.sum() / n_true[Li].sum(), 4), cands=round(float(cnt.mean()), 1)))
    print(pd.DataFrame(rows).to_string(index=False))


def report():
    left, right, true_s1, n_true, part, dev = setup()
    base = {p: (25, ALL[p].get("min_score", 0.0)) for p in PASSES}
    wide = {p: (100, 0.0) if not ALL[p].get("reverse") else (ALL[p]["k"], 0.0) for p in ALL}
    grid = {
        "current (k25, floors)": base,
        "current k100 no floor": {p: wide[p] for p in PASSES},
    }
    for p in EXTRA_PASSES:
        grid[f"current k100 + {p}"] = {**{q: wide[q] for q in PASSES}, p: wide[p]}
    grid["all passes k100"] = wide
    grid["all passes k50"] = {p: (min(50, v[0]), 0.0) for p, v in wide.items()}
    grid["all, RA k30"] = {**wide, "RA": (30, 0.0)}
    for name, st in grid.items():
        try:
            print(f"{name:28s}", score(st, left, true_s1, n_true, dev), flush=True)
        except FileNotFoundError as e:
            print(name, "missing hits:", e)
    for p in ALL:
        st = {q: v for q, v in wide.items() if q != p}
        print(f"  all without {p}:", score(st, left, true_s1, n_true, dev)["all"], flush=True)
    print("\nStage 1.5 pre-ranker (pass scores/ranks only) on all passes k100:")
    prerank(wide, left, true_s1, n_true, dev)
    c1, lab1 = left.codes("country")
    for code in np.unique(c1[dev]):
        misses(wide, left, right, true_s1, n_true, dev, lab1[code])


def report_prerank(ra_k=30):
    """Only the Stage 1.5 budget table and the misses of the chosen pass set."""
    left, right, true_s1, n_true, part, dev = setup()
    wide = {p: (100, 0.0) if not ALL[p].get("reverse") else (ra_k, 0.0) for p in ALL}
    print("pass scores / ranks + record signatures:")
    prerank(wide, left, true_s1, n_true, dev, right=right)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "query"
    if cmd == "query":
        query(sys.argv[2:] or list(ALL))
    elif cmd == "prerank":
        report_prerank()
    else:
        report()
