"""Lab: Stage 1.5 quality vs inference cost (trees, leaves, prediction early stopping)."""
import time

import lightgbm as lgb
import numpy as np

import config
from evaluate import f05_arrays
from monotone import params_for
from recall_lab import ALL, setup, wide_hits
from signatures import pair_signature_features, side_signatures

left, right, true_s1, n_true, part, dev = setup()
sig = side_signatures("train", left, right)
wide = {p: (100, 0.0) if not ALL[p].get("reverse") else (30, 0.0) for p in ALL}
c1, lab1 = left.codes("country")
data = {}
for code in np.unique(c1[dev]):
    country = lab1[code]
    W = wide_hits(wide, country)
    for k, v in pair_signature_features(sig[0].take(W["l"].values), sig[1].take(W["r"].values)).items():
        W[k] = v
    y = (true_s1[W["r"].values] == W["l"].values).astype(np.int8)
    l, r = W["l"].values, W["r"].values
    X = W.drop(columns=["l", "r"]).astype(np.float32)
    data[country] = (X, y, l, dev[c1[dev] == code])
cols = list(data["US"][0].columns)
import sys
configs = {
    "lr0.1 L31 x200": (dict(learning_rate=0.1, num_leaves=31), 200),
    "lr0.3 L31 x60": (dict(learning_rate=0.3, num_leaves=31), 60),
    "lr0.3 L15 x60": (dict(learning_rate=0.3, num_leaves=15), 60),
    "lr0.5 L15 x30": (dict(learning_rate=0.5, num_leaves=15), 30),
}
margins = [float(x) for x in sys.argv[2:]] or [8.0]
if len(sys.argv) > 1:
    configs = {k: v for k, v in configs.items() if k == sys.argv[1]}
for name, (extra, rounds) in configs.items():
    prm = params_for(dict(objective="binary", min_data_in_leaf=200, lambda_l2=10.0, verbose=-1,
                          num_threads=config.N_JOBS, seed=config.SEED, **extra), cols)
    for early in [False] + margins:
        out = []
        pt = 0.0
        nrows = 0
        for country, (X, y, l, Li) in data.items():
            fold = l % 2
            p = np.zeros(len(y))
            for f in (0, 1):
                m = lgb.train(prm, lgb.Dataset(X[fold != f], y[fold != f]), rounds)
                t = time.time()
                p[fold == f] = m.predict(X[fold == f], pred_early_stop=bool(early), pred_early_stop_freq=5,
                                         pred_early_stop_margin=float(early or 8.0))
                pt += time.time() - t
                nrows += int((fold == f).sum())
            order = np.lexsort((-p, l))
            ls = l[order]
            start = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]
            rank = np.arange(len(l)) - np.repeat(start, np.diff(np.r_[start, len(l)]))
            for M in (30, 60, 120):
                sel = order[rank < M]
                yy = y[sel].astype(bool)
                hit = np.bincount(l[sel][yy], minlength=left.n)[Li]
                out.append(f"{country}@{M}={f05_arrays(hit, hit, n_true[Li]).mean():.5f}")
        print(f"{name:16s} early_stop={early!s:5s} predict {pt / nrows * 1e6:.2f} us/row  " + " ".join(out), flush=True)
