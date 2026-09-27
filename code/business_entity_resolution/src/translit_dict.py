"""Learn native-script token -> Latin token maps from training ground-truth pairs.

Names: when a native-script name and its Source 1 name have the same token count,
tokens are aligned by position. Addresses: every Latin token in the paired S1
address is a candidate; the winner combines co-occurrence share with similarity
to the generic transliteration. Only the provided training data is used."""
import json
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from normalize import _INDIC, fold, transliterate, _NONALNUM

INDIC_RE = "[ऀ-෿]"


def _latin_tokens(s):
    return _NONALNUM.sub(" ", fold(s)).split()


def learn(raw_dir, out_path, min_support=3, min_share=0.5):
    gt = pq.read_table(raw_dir + "train_ground_truth.parquet")
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    lens = pc.list_value_length(lists)
    s1_rep = pc.take(gt["source1_entity_id"], pc.list_parent_indices(lists))
    pair = pa.table({"s1": s1_rep, "r": pc.list_flatten(lists)})
    right = []
    for f in ("train_source2", "train_source3"):
        t = pq.read_table(raw_dir + f + ".parquet", columns=["entity_id", "business_name", "business_address"])
        m = pc.or_(pc.match_substring_regex(t["business_name"], INDIC_RE),
                   pc.match_substring_regex(t["business_address"], INDIC_RE))
        right.append(t.filter(m))
    right = pa.concat_tables(right)
    right = right.join(pair, keys="entity_id", right_keys="r", join_type="inner")
    s1 = pq.read_table(raw_dir + "train_source1.parquet", columns=["entity_id", "business_name", "business_address"])
    s1 = s1.filter(pc.is_in(s1["entity_id"], right["s1"].unique()))
    s1 = s1.rename_columns(["s1", "s1_name", "s1_addr"])
    J = right.join(s1, keys="s1").to_pydict()
    print("native-script pairs", len(J["s1"]))

    name_co = defaultdict(Counter)
    addr_co, addr_n = defaultdict(Counter), Counter()
    for rn, ra, sn, sa in zip(J["business_name"], J["business_address"], J["s1_name"], J["s1_addr"]):
        if _INDIC.search(rn):
            a, b = rn.split(), _latin_tokens(sn)
            if len(a) == len(b):
                for x, y in zip(a, b):
                    if _INDIC.search(x):
                        name_co[x][y] += 1
        if _INDIC.search(ra):
            lat = set(_latin_tokens(sa))
            for x in set(t for t in _NONALNUM.sub(" ", ra.replace(",", " ")).split() if t) | \
                    set(t for t in ra.replace(",", " ").split() if _INDIC.search(t)):
                if _INDIC.search(x):
                    addr_n[x] += 1
                    addr_co[x].update(lat)
    out = {}
    for x, c in name_co.items():
        tot = sum(c.values())
        y, k = c.most_common(1)[0]
        if tot >= min_support and k / tot >= min_share:
            out[x] = y
    n_name = len(out)
    for x, c in addr_co.items():
        tot = addr_n[x]
        if tot < min_support or x in out:
            continue
        tr = transliterate(x)
        best, bs = None, 0.0
        for y, k in c.most_common(10):
            share = k / tot
            if share < min_share:
                break
            s = share * (0.5 + 0.5 * fuzz.ratio(tr, y) / 100)
            if s > bs:
                best, bs = y, s
        if best:
            out[x] = best
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print("learned", n_name, "name tokens,", len(out) - n_name, "address tokens")
    return out


if __name__ == "__main__":
    learn("cache/raw/", "cache/translit_dict.json")
