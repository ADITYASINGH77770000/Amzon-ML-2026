"""Quick provisional out-of-fold macro F0.5 on the full-data 30% S1 sample: a small
threshold grid around the DEV_FAST setting (the pipeline's full grid runs separately)."""
import itertools
import json
import os

import numpy as np

import config
import decide
import evaluate
from data import load_split, truth_arrays

asm = os.path.join(config.WORK, "train_sub_asm")
L = np.load(os.path.join(asm, "l.npy")).astype(np.int64)
R = np.load(os.path.join(asm, "r.npy")).astype(np.int64)
y = np.load(os.path.join(asm, "y.npy")).astype(bool)
p = np.load(os.path.join(asm, "p_oof.npy")).astype(np.float64)
ent = np.load(os.path.join(asm, "entities.npy"))
left, right = load_split("train")
_, n_true = truth_arrays(left, right)
c1, lab = left.codes("country")
dev = json.load(open(os.path.join(config.CACHE, "runs", "DEV_FAST", "models", "stage3_eval.json")))["global_cfg"]
assign = decide.assign_mask(R, p)
n = left.n
pp, best = decide._prepared(L, p, assign)
pp0, best0 = decide._prepared(L, p, None)
top = (-1, None, None)
grid = itertools.product([dev["t_open"] + d for d in (-0.1, -0.05, 0, 0.05, 0.1)],
                         [dev["t_add"] + d for d in (-0.1, -0.05, 0, 0.05, 0.1)], [0.0, 0.3], [True, False])
for to, ta, rel, asg in grid:
    m = decide._mask(pp if asg else pp0, best if asg else best0, to, ta, rel)
    n_pred = np.bincount(L[m], minlength=n)
    tp = np.bincount(L[m], weights=y[m].astype(float), minlength=n)
    f = evaluate.f05_arrays(n_pred, tp, n_true)
    s = f[ent].mean()
    if s > top[0]:
        top = (s, dict(t_open=round(to, 3), t_add=round(ta, 3), rel=rel, assign=asg), f)
s, cfg, f = top
out = {"oof_macro_f05": round(float(s), 5), "cfg": cfg, "n_s1": int(len(ent))}
for code, name in enumerate(lab):
    e = ent[c1[ent] == code]
    out[name] = round(float(f[e].mean()), 5)
sing = ent[n_true[ent] == 0]
out["singleton_acc"] = round(float(f[sing].mean()), 5)
print(json.dumps(out))
json.dump(out, open(os.path.join(config.MODELS, "quick_f05.json"), "w"), indent=1)
