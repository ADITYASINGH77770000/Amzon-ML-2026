"""Build a miniature copy of the challenge data for an end-to-end smoke test.

Train: a few thousand S1 entities from the EDA sample with their S2/S3 records
plus unmatched distractors. Test: other entities, a third of them relabelled
to an unseen country ("Testland") to check the open-set country handling.

    python make_smoke.py <out_root>
    set ER_ROOT=<out_root> & python run_pipeline.py all
"""
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cache", "eda_sample.pkl")


def write(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def main(out, n_train=5000, n_test=2500, seed=0):
    pairs, recs = pickle.load(open(SRC, "rb"))
    groups = defaultdict(list)
    for a, b in pairs:
        groups[a].append(b)
    s1 = sorted(k for k in recs if k.startswith("S1-"))
    rng = np.random.default_rng(seed)
    rng.shuffle(s1)
    tr, te, dis = s1[:n_train], s1[n_train:n_train + n_test], s1[n_train + n_test:n_train + n_test + 1500]
    hdr = ["entity_id", "business_name", "business_address", "country"]
    for split, ids in (("train", tr), ("test", te)):
        d = os.path.join(out, "dataset", split)
        os.makedirs(d, exist_ok=True)
        relabel = set(ids[: len(ids) // 3]) if split == "test" else set()
        # distractors: right records of entities not in this split (their S1 is absent)
        extra = [b for a in dis for b in groups[a]] if split == "train" else []
        rows = {1: [], 2: [], 3: []}
        for a in ids:
            c = "Testland" if a in relabel else recs[a]["country"]
            rows[1].append([a, recs[a]["business_name"], recs[a]["business_address"], c])
            for b in groups[a]:
                rows[int(b[1])].append([b, recs[b]["business_name"], recs[b]["business_address"], c])
        for b in extra:
            rows[int(b[1])].append([b, recs[b]["business_name"], recs[b]["business_address"], recs[b]["country"]])
        for k in (1, 2, 3):
            write(os.path.join(d, f"{split}_source{k}.tsv"), rows[k], hdr)
        if split == "train":
            write(os.path.join(d, "train_ground_truth.tsv"),
                  [[a, ",".join(groups[a])] for a in ids], ["source1_entity_id", "matched_entity_ids"])
    print("smoke data written to", out)


if __name__ == "__main__":
    main(sys.argv[1])
