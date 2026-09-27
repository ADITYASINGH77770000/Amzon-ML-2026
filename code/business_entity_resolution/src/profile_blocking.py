"""Profile blocking passes: time, raw product size and recall per setting (US train)."""
import os, time
import numpy as np
import config
from candidates import PassIndex
from blocking import PASSES, _weight, topk_rows
from data import load_split, truth_arrays

left, right = load_split("train")
true_s1, n_true = truth_arrays(left, right)
c1, lab1 = left.codes("country"); c2, lab2 = right.codes("country")
cu = lab1.index("US")
Li = np.flatnonzero(c1 == cu); Ri = np.flatnonzero(c2 == lab2.index("US"))
rng = np.random.default_rng(0)
Q = np.sort(rng.choice(Li, 4000, replace=False))
tot_true = n_true[Q].sum()
# global position of right rows -> is it a true match of the query row?
for p, cfg in PASSES.items():
    t = time.time()
    idx = PassIndex(lambda: right.list(cfg["field"], Ri), cfg, os.path.join(config.CACHE, "index", "train", f"US_{p}"))
    print(p, "index ready", round(time.time() - t), "s, nnz", idx.Rt.nnz, flush=True)
    texts = left.list(cfg["field"], Q)
    for qmax in [cfg["max_df"], cfg["max_df"] // 4, cfg["max_df"] // 10]:
        idx.qmask = (idx.df_kept <= qmax).astype(np.float32)
        t = time.time()
        L = _weight(idx.vec.transform(texts), idx.idf, idx.keep)
        L.data *= idx.qmask[L.indices]; L.eliminate_zeros()
        M = (L @ idx.Rt).tocsr()
        tm = time.time() - t
        for ms in [0.0, 0.2, 0.35, 0.5]:
            M2 = M.copy()
            if ms: M2.data[M2.data < ms] = 0
            t2 = time.time()
            i, j, v = topk_rows(M2, cfg["k"])
            hit = (true_s1[Ri[j]] == Q[i]).sum()
            print(f"  {p} qmax={qmax} min={ms}: matmul {tm:.1f}s raw_nnz {M.nnz:,} topk {time.time()-t2:.1f}s "
                  f"pairs/row {len(i)/len(Q):.1f} recall {hit/tot_true:.4f}", flush=True)
    del idx
