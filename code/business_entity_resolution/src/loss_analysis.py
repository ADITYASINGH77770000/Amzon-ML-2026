"""Where does out-of-fold macro F0.5 go? Splits the gap to the candidate oracle into
singleton false merges, ranking errors and "how many to accept" errors.

    set ER_MODE=DEV_FAST & python loss_analysis.py
"""
import json
import os

import numpy as np

import config
import decide
import evaluate
from run_pipeline import context

ctx = context("train")
asm = os.path.join(config.WORK, "train_fin_asm")
L = np.load(os.path.join(asm, "l.npy")).astype(np.int64)
R = np.load(os.path.join(asm, "r.npy")).astype(np.int64)
y = np.load(os.path.join(asm, "y.npy")).astype(bool)
p = np.load(os.path.join(asm, "p_oof.npy")).astype(np.float64)
ev = json.load(open(os.path.join(config.MODELS, "stage3_eval.json")))
n_true, n = ctx["n_true"], ctx["left"].n
ent = ctx["labelled"]
c1, lab = ctx["c1"], ctx["lab1"]


def macro(mask):
    n_pred = np.bincount(L[mask], minlength=n)
    tp = np.bincount(L[mask], weights=y[mask].astype(float), minlength=n)
    return evaluate.f05_arrays(n_pred, tp, n_true)


assign = decide.assign_mask(R, p)
cfg = ev["global_cfg"]
f_cur = macro(decide.apply(L, R, p, cfg, assign))
hit = np.bincount(L[y], minlength=n)
f_orc = evaluate.f05_arrays(hit, hit, n_true)
print(f"current OOF macro F0.5 {f_cur[ent].mean():.5f} | candidate oracle {f_orc[ent].mean():.5f}")

# rank inside each S1 by p
order = np.lexsort((-p, L))
Ls = L[order]
start = np.r_[0, np.flatnonzero(np.diff(Ls)) + 1]
rank = np.empty(len(L), np.int64)
rank[order] = np.arange(len(L)) - np.repeat(start, np.diff(np.r_[start, len(L)]))
# "right count" oracle: accept exactly the top n_true(in candidates) by p
k_in = hit
f_topk = macro(rank < k_in[L])
print(f"accept exactly top-(#true in candidates) by p: {f_topk[ent].mean():.5f}  (ranking-only ceiling)")
# perfect count for singletons too (accept nothing when there is nothing)
sing = n_true == 0
loss = lambda f: float((1 - f[ent]).sum() / len(ent))
print(f"loss vs 1.0: current {loss(f_cur):.5f} = singletons {float((1 - f_cur[ent][sing[ent]]).sum() / len(ent)):.5f}"
      f" + matched {float((1 - f_cur[ent][~sing[ent]]).sum() / len(ent)):.5f}")
print(f"          oracle {loss(f_orc):.5f}; ranking-only {loss(f_topk):.5f}")
# matched entities: classify the current errors
m = ~sing
pred = decide.apply(L, R, p, cfg, assign)
n_pred = np.bincount(L[pred], minlength=n)
tp = np.bincount(L[pred], weights=y[pred].astype(float), minlength=n)
cats = {
    "predicted nothing": m & (n_pred == 0),
    "all predicted wrong": m & (n_pred > 0) & (tp == 0),
    "missing some (no FP)": m & (tp > 0) & (tp == n_pred) & (tp < n_true),
    "has a false pair": m & (tp > 0) & (tp < n_pred),
}
for k, v in cats.items():
    e = np.flatnonzero(v)
    print(f"  {k:24s}: {len(e):6,} S1 ({len(e) / len(ent):.3%}), F0.5 loss {float((1 - f_cur[e]).sum() / len(ent)):.5f}")
# how often the best candidate of an S1 is wrong while a true one exists
best_wrong = m & (np.bincount(L[(rank == 0) & ~y], minlength=n) > 0) & (hit > 0)
print(f"  S1 whose top-scored candidate is wrong (a true one is in the list): {best_wrong.sum():,}")
for code, name in enumerate(lab):
    e = ent[c1[ent] == code]
    print(f"  {name}: current {f_cur[e].mean():.5f}  ranking-only {f_topk[e].mean():.5f}  oracle {f_orc[e].mean():.5f}")
