"""Lab: a rule pre-filter before Stage 1.5 — keep a raw pair if it was found by >= 2
passes, or by one pass at rank < R. Reports kept-row share and candidate oracle."""
import numpy as np

from evaluate import f05_arrays
from recall_lab import ALL, setup, wide_hits

left, right, true_s1, n_true, part, dev = setup()
wide = {p: (100, 0.0) if not ALL[p].get("reverse") else (30, 0.0) for p in ALL}
c1, lab1 = left.codes("country")
for code in np.unique(c1[dev]):
    country = lab1[code]
    Li = dev[c1[dev] == code]
    W = wide_hits(wide, country)
    l, r = W["l"].values, W["r"].values
    y = true_s1[r] == l
    ranks = W[[c for c in W.columns if c.startswith("rank_")]].to_numpy()
    minrank = np.nanmin(np.where(np.isnan(ranks), 1e9, ranks), axis=1)
    npass = W["n_passes"].values
    hit = np.bincount(l[y], minlength=left.n)[Li]
    base = f05_arrays(hit, hit, n_true[Li]).mean()
    print(f"{country}: raw rows {len(W):,}, oracle {base:.5f}; single-pass share {np.mean(npass == 1):.3f}")
    for R in (5, 10, 20, 30, 50):
        for need in (2, 3):
            keep = (npass >= need) | (minrank < R)
            hk = np.bincount(l[keep & y], minlength=left.n)[Li]
            print(f"   keep if passes>={need} or rank<{R}: rows {keep.mean():.3f}, "
                  f"oracle {f05_arrays(hk, hk, n_true[Li]).mean():.5f}")
