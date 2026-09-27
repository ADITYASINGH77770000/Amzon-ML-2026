"""Which true pairs does the Stage 1.5 pre-ranker push below the top-M? (lab tool)"""
import sys

import lightgbm as lgb
import numpy as np

import config
from recall_lab import ALL, setup, wide_hits
from signatures import pair_signature_features, side_signatures

country = sys.argv[1] if len(sys.argv) > 1 else "India"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 120
left, right, true_s1, n_true, part, dev = setup()
sig = side_signatures("train", left, right)
wide = {p: (100, 0.0) if not ALL[p].get("reverse") else (30, 0.0) for p in ALL}
W = wide_hits(wide, country)
for k, v in pair_signature_features(sig[0].take(W["l"].values), sig[1].take(W["r"].values)).items():
    W[k] = v
y = (true_s1[W["r"].values] == W["l"].values).astype(np.int8)
X = W.drop(columns=["l", "r"]).astype(np.float32)
fold = W["l"].values % 2
p = np.zeros(len(W))
for f in (0, 1):
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, verbose=-1, num_threads=4),
                  lgb.Dataset(X[fold != f], y[fold != f]), 200)
    p[fold == f] = m.predict(X[fold == f])
W["p0"], W["y"] = p, y
W = W.sort_values(["l", "p0"], ascending=[True, False])
W["rk"] = W.groupby("l").cumcount()
lost = W[(W["y"] == 1) & (W["rk"] >= M)]
kept = W[(W["y"] == 1) & (W["rk"] < M)]
print(f"{country}: {len(lost)} true pairs ranked >= {M} (of {int(y.sum())})")
cols = [c for c in W.columns if c.startswith("sig_") or c == "n_passes"]
print("mean features, lost vs kept true pairs:")
print(np.round(lost[cols].mean(), 3).to_frame("lost").join(np.round(kept[cols].mean(), 3).to_frame("kept")).to_string())
passes = [c[6:] for c in W.columns if c.startswith("score_")]
print("share found by each pass, lost:", {q: round(float(lost[f"score_{q}"].notna().mean()), 3) for q in passes})
print("lost p0 quantiles:", np.round(np.quantile(lost["p0"], [0.1, 0.5, 0.9]), 5),
      " typical rank:", np.quantile(lost["rk"], [0.1, 0.5, 0.9]))
for _, row in lost.sample(min(25, len(lost)), random_state=0).iterrows():
    a, b = int(row["l"]), int(row["r"])
    print(f"  S1: {left.list('name_core', [a])[0]} | {left.list('addr_core', [a])[0][:90]}")
    print(f"   R: {right.list('name_core', [b])[0]} | {right.list('addr_core', [b])[0][:90]}"
          f"   rank {int(row['rk'])}, p0 {row['p0']:.4f}, passes {int(row['n_passes'])}")
