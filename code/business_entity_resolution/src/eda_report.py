"""Checkpoint 1: answers the plan's hypotheses H1-H10 on the training data (memory-light)."""
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RAW = "cache/raw/"


def main(n_sample=30000, seed=0):
    gt = pq.read_table(RAW + "train_ground_truth.parquet")
    s1 = gt["source1_entity_id"].to_numpy(zero_copy_only=False)
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(lists).to_numpy(zero_copy_only=False)
    flat_ok = flat != ""
    lens = lens - np.add.reduceat(~flat_ok, np.r_[0, np.cumsum(lens)[:-1]]) * (lens > 0)
    print("S1 entities", len(s1))
    print("H3 singleton share", round(float((lens == 0).mean()), 4))
    q = np.percentile(lens, [50, 90, 95, 99, 99.9])
    print("H9 matches per S1: mean %.2f max %d p50/p90/p95/p99/p99.9 %s" % (lens.mean(), lens.max(), q))
    print("   count histogram", sorted(Counter(lens.tolist()).items())[:20])
    ids = flat[flat_ok]
    print("pairs", len(ids), "S2 share", round(float(np.char.startswith(ids.astype(str), "S2-").mean()), 4))
    u, c = np.unique(ids, return_counts=True)
    print("H1 records matched by >1 S1:", int((c > 1).sum()), "of", len(u))
    n2 = pq.read_metadata(RAW + "train_source2.parquet").num_rows
    n3 = pq.read_metadata(RAW + "train_source3.parquet").num_rows
    n2m = int(np.char.startswith(u.astype(str), "S2-").sum())
    print("H6 unmatched S2 share %.4f, S3 share %.4f" % (1 - n2m / n2, 1 - (len(u) - n2m) / n3))

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(s1), n_sample, replace=False)
    pick_ids = set(s1[pick])
    starts = np.r_[0, np.cumsum(pc.list_value_length(lists).to_numpy(zero_copy_only=False))]
    pairs = [(s1[i], x) for i in pick for x in flat[starts[i]:starts[i + 1]] if x]
    want = pick_ids | {b for _, b in pairs}
    recs = {}
    for f in ["train_source1", "train_source2", "train_source3"]:
        t = pq.read_table(RAW + f + ".parquet")
        t = t.filter(pc.is_in(t["entity_id"], pa.array(list(want))))
        for r in t.to_pylist():
            recs[r["entity_id"]] = r
    return lens, pairs, recs


if __name__ == "__main__":
    lens, pairs, recs = main()
    import pickle
    pickle.dump((pairs, recs), open("cache/eda_sample.pkl", "wb"))
