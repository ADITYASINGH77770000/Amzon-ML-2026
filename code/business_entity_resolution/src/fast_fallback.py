"""Fast exact-key fallback matcher (used when there is no time for the full pipeline).

Exact keys on the normalised fields, inside a country (see SPECS). A key links a right
record to an S1 only if exactly ONE S1 carries that key (unambiguous); keys are applied
in priority order and every S2/S3 record goes to at most one S1. All joins run on
dictionary-encoded Arrow keys (no Python loops). Needs `run_pipeline.py prepare` first.

    python fast_fallback.py eval     # macro F0.5 on the full labelled training data
    python fast_fallback.py test     # write output/matching_results.tsv + candidate_pairs.tsv
"""
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import config
from data import Side, truth_arrays
from evaluate import f05_arrays
from io_utils import write_tsv


def _key(t, parts, need):
    """country|part1|part2... as one string column; null where a needed part is empty."""
    cols = [t["country"]] + [t[p] for p in parts]
    k = pc.binary_join_element_wise(*cols, "|")
    ok = pc.and_(*[pc.not_equal(t[p], "") for p in need]) if len(need) > 1 else pc.not_equal(t[need[0]], "")
    return pc.if_else(ok, k, pa.scalar(None, pa.string()))


# keys in order of measured precision on the full training data (name+city 0.884 and
# name+state 0.886 were dropped): name+street 0.998, name+house 0.975, core+house 0.974,
# name for address-less records 0.969, street+city (garbled names) 0.953
SPECS = [("name+street", ["name_join", "street_keys"], None),
         ("name+house", ["name_join", "house_no"], None),
         ("core+house", ["name_core", "house_no"], None),
         ("name, no address", ["name_join"], "addr_empty"),
         ("street+city", ["street_keys", "city"], None)]


def link(left, right, log=print, specs=None):
    """-> s1 row for every right row (-1 = unmatched)."""
    specs = specs or SPECS
    owner = np.full(right.n, -1, np.int64)
    for name, parts, only_if in specs:
        t0 = time.time()
        kl = _key(left.t, parts, parts)
        kr = _key(right.t, parts, parts)
        if only_if:
            kr = pc.if_else(pc.equal(right.t[only_if], 1), kr, pa.scalar(None, pa.string()))
        both = pa.chunked_array(kl.chunks + kr.chunks) if isinstance(kl, pa.ChunkedArray) else pa.concat_arrays([kl, kr])
        codes = pc.dictionary_encode(both).combine_chunks()
        idx = pc.fill_null(codes.indices, -1).to_numpy(zero_copy_only=False).astype(np.int64)
        cl, cr = idx[:left.n], idx[left.n:]
        ncode = int(idx.max()) + 1
        cnt = np.bincount(cl[cl >= 0], minlength=ncode)
        s1_of = np.full(ncode, -1, np.int64)
        uniq = (cl >= 0) & (cnt[np.maximum(cl, 0)] == 1)
        s1_of[cl[uniq]] = np.flatnonzero(uniq)
        hit = (cr >= 0) & (owner < 0)
        cand = np.where(hit, s1_of[np.maximum(cr, 0)], -1)
        new = cand >= 0
        owner[new] = cand[new]
        log(f"  key {name}: +{int(new.sum()):,} links ({time.time() - t0:.0f}s)")
    return owner


def evaluate_train():
    left = Side(["train_source1"])
    right = Side(["train_source2", "train_source3"])
    true_s1, n_true = truth_arrays(left, right)
    c1, lab = left.codes("country")
    for tag, specs in (("all 5 keys", SPECS), ("without street+city", SPECS[:4])):
        owner = link(left, right, specs=specs)
        m = owner >= 0
        l, y = owner[m], (true_s1[m] == owner[m])
        n_pred = np.bincount(l, minlength=left.n)
        tp = np.bincount(l, weights=y.astype(float), minlength=left.n)
        f = f05_arrays(n_pred, tp, n_true)
        print(f"[{tag}] train macro F0.5 = {f.mean():.5f} | pair precision {y.mean():.4f} recall {y.sum() / n_true.sum():.4f}"
              + "".join(f" | {name} {f[c1 == code].mean():.5f}" for code, name in enumerate(lab))
              + f" | singletons {f[n_true == 0].mean():.4f}", flush=True)


def write_test():
    left = Side(["test_source1"])
    right = Side(["test_source2", "test_source3"])
    owner = link(left, right)
    m = np.flatnonzero(owner >= 0)
    s1_ids = left.list("entity_id")
    r_ids = right.list("entity_id", m)
    out = {}
    for a, b in zip(owner[m], r_ids):
        out.setdefault(s1_ids[a], []).append(b)
    os.makedirs(os.path.join(config.ROOT, "output"), exist_ok=True)
    for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        write_tsv(s1_ids, out, os.path.join(config.ROOT, "output", name), col)
    print(f"test: {len(m):,} matched pairs for {len(out):,} of {left.n:,} S1; files written to output/")


if __name__ == "__main__":
    {"eval": evaluate_train, "test": write_test}[sys.argv[1]]()
