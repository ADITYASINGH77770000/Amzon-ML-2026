"""Column-wise loaders over the normalised cache, training truth arrays, and the
entity-level subsets used by the DEV / VALIDATION modes (config.MODE)."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config

TEXT_COLS = ["name_norm", "name_core", "name_join", "name_alt", "addr_core", "landmark", "postal",
             "nums", "house_no", "street_keys", "state", "city"]
FLAG_COLS = ["is_web", "name_native", "addr_empty"]


class Side:
    """One side of the join as arrow columns (compact); S2 and S3 are concatenated."""

    def __init__(self, files=None, table=None, sizes=None):
        if files is not None:
            # uncompressed Arrow IPC, memory-mapped: random take() reads through the OS
            # page cache instead of holding ~2 GB of strings on the Python heap
            tabs = [pa.ipc.open_file(pa.memory_map(config.NORM + f + ".arrow")).read_all() for f in files]
            table, sizes = pa.concat_tables(tabs), [x.num_rows for x in tabs]
        self.t = table
        self.n = table.num_rows
        self.sizes = list(sizes)

    def subset(self, rows):
        """A Side holding only `rows` (sorted global rows); per-file sizes are kept, so S2
        rows still come before S3 rows."""
        rows = np.asarray(rows, np.int64)
        bounds = np.cumsum([0] + self.sizes)
        sizes = [int(((rows >= a) & (rows < b)).sum()) for a, b in zip(bounds[:-1], bounds[1:])]
        return Side(table=self.t.take(pa.array(rows)).combine_chunks(), sizes=sizes)

    def codes(self, name):
        """Dictionary-encode a string column -> (int32 codes, list of labels)."""
        d = pc.dictionary_encode(self.t[name]).combine_chunks()
        return d.indices.to_numpy(zero_copy_only=False).astype(np.int32), d.dictionary.to_pylist()

    def col(self, name, idx=None):
        c = self.t[name]
        if idx is not None:
            c = c.take(pa.array(np.asarray(idx, np.int64)))
        return c

    def np(self, name, idx=None):
        return self.col(name, idx).to_numpy(zero_copy_only=False)

    def list(self, name, idx=None):
        return self.col(name, idx).to_pylist()

    def iter_lists(self, name, chunk=1_000_000):
        for s in range(0, self.n, chunk):
            yield self.t[name].slice(s, chunk).to_pylist()


def truth_arrays(left, right):
    """true_s1[r] = S1 row of right record r (-1 none); n_true[l] = true matches of l."""
    gt = pq.read_table(config.RAW + "train_ground_truth.parquet")
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    s1_rep = pc.take(gt["source1_entity_id"], pc.list_parent_indices(lists))
    r_ids = pc.list_flatten(lists)
    ok = pc.not_equal(r_ids, "")
    s1_rep, r_ids = s1_rep.filter(ok), r_ids.filter(ok)
    # records outside these sides (subset modes) come back as nulls -> -1
    li = pc.fill_null(pc.index_in(s1_rep, value_set=left.t["entity_id"].combine_chunks()), -1)
    ri = pc.fill_null(pc.index_in(r_ids, value_set=right.t["entity_id"].combine_chunks()), -1)
    li = li.to_numpy(zero_copy_only=False).astype(np.int64)
    ri = ri.to_numpy(zero_copy_only=False).astype(np.int64)
    keep = (li >= 0) & (ri >= 0)                     # pairs whose two records are in these sides
    true_s1 = np.full(right.n, -1, np.int32)
    true_s1[ri[keep]] = li[keep].astype(np.int32)
    n_true = np.bincount(li[keep], minlength=left.n).astype(np.int32)
    return true_s1, n_true


def subset_train(left, right, fraction, seed):
    """Entity-level subset for DEV / VALIDATION: a `fraction` of S1 entities (stratified
    by country), ALL their true S2/S3 matches, and the same `fraction` of the remaining
    S2/S3 pool (records whose true S1 is not sampled, and records matching nothing)."""
    true_s1, _ = truth_arrays(left, right)
    rng = np.random.default_rng(seed)
    country, _ = left.codes("country")
    pick = np.zeros(left.n, bool)
    for c in np.unique(country):
        rows = np.flatnonzero(country == c)
        pick[rng.choice(rows, int(round(len(rows) * fraction)), replace=False)] = True
    s1_rows = np.flatnonzero(pick)
    owner = true_s1 >= 0
    matched = owner & pick[np.maximum(true_s1, 0)]            # true matches of sampled S1: all kept
    rest = np.flatnonzero(~matched)
    extra = rest[rng.random(len(rest)) < fraction]
    r_rows = np.sort(np.concatenate([np.flatnonzero(matched), extra]))
    return left.subset(s1_rows), right.subset(r_rows)


_CACHE = {}


def load_split(split):
    """(left, right) for a split under the current mode; test is only used in FINAL."""
    if split in _CACHE:
        return _CACHE[split]
    if split == "test" and config.MODE != "FINAL":
        raise RuntimeError(f"the test split is used only in FINAL mode (current MODE={config.MODE})")
    left = Side([f"{split}_source1"])
    right = Side([f"{split}_source2", f"{split}_source3"])
    if split == "train" and config.FRACTION < 1.0:
        left, right = subset_train(left, right, config.FRACTION, config.SUBSET_SEED)
    _CACHE[split] = (left, right)
    return left, right
