"""Matcher variants on the cached DEV features (same folds, same decision tuning).
    set ER_MODE=DEV_FAST & python matcher_lab.py"""
import json
import os

import numpy as np

import config
import run_pipeline as rp
from run_pipeline import context, cv_fit_predict, _decide_oof, transitivity_features

ctx = context("train")
asm = os.path.join(config.WORK, "train_fin_asm")
b = {int(k): v for k, v in json.load(open(os.path.join(asm, "meta.json")))["bounds"].items()}
X = np.load(os.path.join(asm, "X.npy"))
cols = json.load(open(os.path.join(asm, "X_cols.json")))
y = np.load(os.path.join(asm, "y.npy"))
L = np.load(os.path.join(asm, "l.npy")).astype(np.int64)
R = np.load(os.path.join(asm, "r.npy")).astype(np.int64)
p1 = np.load(os.path.join(asm, "p1.npy")).astype(np.float64)
T = transitivity_features(L, R, p1, ctx["right"])
Tm = np.column_stack([np.asarray(v, np.float32) for v in T.values()])
tcols = ["t1_" + k for k in T]
filt = [i for i, c in enumerate(cols) if c in ("p0", "p1") or c.startswith("c1_")]
keep_nf = [i for i in range(len(cols)) if i not in filt]
variants = {
    "E1 baseline": (X, None),
    "E2 +transitivity(ref by p1)": (np.hstack([X, Tm]), None),
    "E3 E2 without filter scores": (np.hstack([X[:, keep_nf], Tm]), None),
    "E4 E2, 255 leaves, lr 0.03": (np.hstack([X, Tm]), dict(rp.STAGE3_PARAMS, num_leaves=255, learning_rate=0.03,
                                                               min_data_in_leaf=50)),
}
res = {}
for name, (Xv, prm) in variants.items():
    p, ms = cv_fit_predict(np.ascontiguousarray(Xv, np.float32), y, b, 6000, name, prm)
    r, _ = _decide_oof(name, L, R, p, y, ctx)
    res[name] = {"oof_f05": r["oof_f05"], **{k: v["f05"] for k, v in r["oof_report"].items()},
                 "trees": [m.best_iteration for m in ms]}
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(config.RUN, "matcher_lab.json"), "w"), indent=1)
