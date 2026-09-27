"""Decision-layer lab on out-of-fold scores: threshold rule vs expected-F0.5 per S1,
with fold-cross-fitted calibration.   set ER_MODE=DEV_FAST & python decision_lab.py"""
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression

import config
import decide
import evaluate
from expected_f import expected_f_mask
from run_pipeline import context

ctx = context("train")
asm = os.path.join(config.WORK, "train_fin_asm")
L = np.load(os.path.join(asm, "l.npy")).astype(np.int64)
R = np.load(os.path.join(asm, "r.npy")).astype(np.int64)
y = np.load(os.path.join(asm, "y.npy")).astype(bool)
p = np.load(os.path.join(asm, "p_oof.npy")).astype(np.float64)
ev = json.load(open(os.path.join(config.MODELS, "stage3_eval.json")))
n_true, n, ent, c1, lab = ctx["n_true"], ctx["left"].n, ctx["labelled"], ctx["c1"], ctx["lab1"]
fold = ctx["fold"][L]


def macro(mask):
    n_pred = np.bincount(L[mask], minlength=n)
    tp = np.bincount(L[mask], weights=y[mask].astype(float), minlength=n)
    f = evaluate.f05_arrays(n_pred, tp, n_true)
    return f[ent].mean(), {name: round(float(f[ent[c1[ent] == c]].mean()), 5) for c, name in enumerate(lab)}


assign = decide.assign_mask(R, p)
print("threshold rule (current):", macro(decide.apply(L, R, p, ev["global_cfg"], assign)))
pa = np.where(assign, p, 0.0)                       # a record claimed by a better S1 counts as 0
# expected missing true matches per S1 (candidate recall on these folds, by country)
hit = np.bincount(L[y], minlength=n)
for tag, q in (("raw p", pa),):
    for miss_scale in (0.0, 1.0):
        miss = 0.0
        if miss_scale:
            miss = float((n_true[ent] - hit[ent]).sum() / max(len(ent), 1))
        print(f"expected-F0.5, {tag}, miss={miss:.4f}:", macro(expected_f_mask(L, q, miss=miss)))
# isotonic calibration cross-fitted by fold (fit on the other fold, apply to this one)
cal = np.empty_like(p)
for k in np.unique(fold):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[fold != k], y[fold != k])
    cal[fold == k] = iso.predict(p[fold == k])
ca = np.where(assign, cal, 0.0)
print("expected-F0.5, isotonic (cross-fitted):", macro(expected_f_mask(L, ca)))
for g in (0.8, 1.2, 1.5, 2.0):
    print(f"expected-F0.5, p^{g}:", macro(expected_f_mask(L, pa ** g)))
