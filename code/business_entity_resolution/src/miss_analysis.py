import os, sys, numpy as np
import config
from candidates import PassIndex
from blocking import PASSES, field_texts, index_tag
from data import load_split, truth_arrays
country = sys.argv[1] if len(sys.argv) > 1 else "US"
left, right = load_split("train")
true_s1, n_true = truth_arrays(left, right)
c1, lab1 = left.codes("country"); c2, lab2 = right.codes("country")
Li = np.flatnonzero(c1 == lab1.index(country)); Ri = np.flatnonzero(c2 == lab2.index(country))
Q = np.sort(np.random.default_rng(0).choice(Li, 4000, replace=False))
got = set()
per = {}
for p in PASSES:
    cfg = dict(PASSES[p])
    idx = PassIndex(lambda: field_texts(right, cfg["field"], Ri), cfg, os.path.join(config.INDEX, "train", f"{country}_{index_tag(p, cfg)}"))
    import time; t = time.time()
    i, j, v, _ = idx.query(field_texts(left, cfg["field"], Q))
    per[p] = set(zip(Q[i].tolist(), Ri[j].tolist()))
    got |= per[p]
    print(p, "query", round(time.time() - t, 1), "s", len(i) / len(Q), "pairs/row", flush=True)
    del idx
inQ = np.zeros(left.n, bool); inQ[Q] = True
tr = np.flatnonzero((true_s1 >= 0) & inQ[np.maximum(true_s1, 0)])
pairs = list(zip(true_s1[tr].tolist(), tr.tolist()))
miss = [p for p in pairs if p not in got]
print("union recall", 1 - len(miss) / len(pairs), "cands/row", len(got) / len(Q))
for p in per:
    print(" ", p, "alone", round(sum(x in per[p] for x in pairs) / len(pairs), 4), "without",
          round(sum(any(x in per[q] for q in per if q != p) for x in pairs) / len(pairs), 4))
rng = np.random.default_rng(1)
for k in rng.choice(len(miss), 40, replace=False):
    a, b = miss[k]
    print("S1:", left.list("name_core", [a])[0], "|", left.list("addr_core", [a])[0], "|", left.list("street_keys", [a])[0])
    print("  R:", right.list("name_core", [b])[0], "|", right.list("addr_core", [b])[0], "|", right.list("street_keys", [b])[0])
