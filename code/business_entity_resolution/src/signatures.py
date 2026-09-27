"""Numeric record signatures for the Stage 1.5 pre-ranker.

Every record gets a small uint32 vector: MinHash values of its name tokens, of
the character 3-grams of its joined name (typo / website tolerant), of the
phonetic skeletons of its name tokens (transliteration tolerant), of its address
words and of its address numbers, plus hashes of house number, city, state and
the first letters of the joined name, the name length and flags. Comparing two
signatures is plain numpy (equal-slot counts approximate Jaccard similarity), so
~350 raw candidates per S1 can be scored without touching a string. Signatures
are computed once per table and cached as memory-mapped .npy files."""
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pyarrow as pa

import config

SIG_VERSION = 2
K = 6                                  # MinHash slots per group
GROUPS = ("name", "tri", "skel", "addr", "nums")
SCALARS = ("h_house", "h_city", "h_state", "h_prefix", "name_len", "flags")
SIG_DIM = len(GROUPS) * K + len(SCALARS)
_EMPTY = 0xFFFFFFFF
_rng = np.random.default_rng(12345)
# multiply-shift hashing of 32-bit crc values: odd multipliers, one per MinHash slot
_A64 = _rng.integers(1, 1 << 31, len(GROUPS) * K).astype(np.uint64) | np.uint64(1)
_B64 = _rng.integers(0, 1 << 31, len(GROUPS) * K).astype(np.uint64)

FIELDS = ["name_core", "name_join", "name_alt", "addr_core", "nums", "house_no", "city", "state",
          "addr_empty", "is_web", "name_native"]


def _work(cols):
    """Signatures of one chunk. Python only collects item hashes per record and group;
    every MinHash slot is then computed with numpy segment minima."""
    from blocking_keys import skeleton
    core, join, alt, addr, nums, house, city, state, empty, web, native = cols
    n = len(core)
    crc = zlib.crc32
    items = [[] for _ in GROUPS]
    counts = np.zeros((len(GROUPS), n), np.int64)
    for r in range(n):
        toks = set(core[r].split()) | set(alt[r].split())
        j = join[r]
        sets = (toks,
                {j[i:i + 3] for i in range(max(len(j) - 2, 1))} if j else (),
                {skeleton(t) for t in toks} - {""},
                {t for t in addr[r].split() if not t.isdigit()},
                set(nums[r].split()))
        for g, st in enumerate(sets):
            counts[g, r] = len(st)
            items[g].extend(crc(t.encode()) for t in st)
    out = np.empty((n, SIG_DIM), np.uint32)
    mask32 = np.uint64(0xFFFFFFFF)
    for g in range(len(GROUPS)):
        H = np.asarray(items[g], np.uint64)
        cnt = counts[g]
        has = cnt > 0
        starts = np.r_[0, np.cumsum(cnt)[:-1]][has]
        for i in range(K):
            col = np.full(n, _EMPTY, np.uint32)
            if len(H):
                v = ((H * _A64[g * K + i] + _B64[g * K + i]) & mask32) >> np.uint64(1)
                col[has] = np.minimum.reduceat(v, starts).astype(np.uint32)
            out[:, g * K + i] = col
    base = len(GROUPS) * K
    out[:, base] = [crc(h.encode()) if h else 0 for h in house]
    out[:, base + 1] = [crc(c.encode()) if c else 0 for c in city]
    out[:, base + 2] = [crc(s.encode()) if s else 0 for s in state]
    out[:, base + 3] = [crc(j[:6].encode()) if len(j) >= 4 else 0 for j in join]
    out[:, base + 4] = [len(j) for j in join]
    out[:, base + 5] = (np.asarray(empty, np.uint32) | (np.asarray(web, np.uint32) << 1) |
                        (np.asarray(native, np.uint32) << 2))
    return out


def build(table, path, chunk=100_000):
    """table: pyarrow Table with FIELDS; writes an (n, SIG_DIM) uint32 .npy."""
    n = table.num_rows
    out = np.lib.format.open_memmap(path + ".tmp.npy", mode="w+", dtype=np.uint32, shape=(n, SIG_DIM))
    step = chunk * config.N_JOBS
    with Pool(config.N_JOBS) as pool:
        for s in range(0, n, step):
            t = table.slice(s, step)
            cols = [t[c].to_pylist() if t[c].type == pa.string() else t[c].to_numpy().tolist() for c in FIELDS]
            parts = [[c[i:i + chunk] for c in cols] for i in range(0, t.num_rows, chunk)]
            out[s:s + t.num_rows] = np.concatenate(pool.map(_work, parts))
    out.flush()
    del out
    os.replace(path + ".tmp.npy", path)


def load(table_name, table):
    """Memory-mapped signatures of one normalised table (built on first use)."""
    d = os.path.join(config.SIG, f"sig_v{config.NORM_VERSION}_{SIG_VERSION}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, table_name + ".npy")
    if not os.path.exists(path):
        build(table, path)
    return np.load(path, mmap_mode="r")


def compact(block):
    """uint32 signatures -> uint16: low 16 bits of every hash (MinHash sentinel kept as
    0xFFFF, missing scalar hashes stay 0); name length capped at 65535. A false 16-bit
    collision happens 1 in 65,536 slots — negligible for similarity estimates."""
    out = (block & 0xFFFF).astype(np.uint16)
    nm = len(GROUPS) * K
    out[:, :nm][block[:, :nm] == _EMPTY] = 0xFFFF
    base = nm
    out[:, base + 4] = np.minimum(block[:, base + 4], 65535)
    out[:, base + 5] = block[:, base + 5]
    return out


class CountrySig:
    """Compact signatures of one side's rows in one country, loaded fully into RAM
    (random reads of the full-size memmap were the bottleneck of candidate building)."""

    def __init__(self, parent, rows, path, chunk=500_000):
        self.rows = np.asarray(rows, np.int64)            # sorted global row numbers
        if not os.path.exists(path):
            arr = np.lib.format.open_memmap(path + ".tmp.npy", mode="w+", dtype=np.uint16,
                                            shape=(len(self.rows), SIG_DIM))
            for s in range(0, len(self.rows), chunk):
                arr[s:s + chunk] = compact(parent.take(self.rows[s:s + chunk]))
            arr.flush()
            del arr
            os.replace(path + ".tmp.npy", path)
        self.arr = np.load(path)

    def take(self, idx):
        return self.arr[np.searchsorted(self.rows, np.asarray(idx, np.int64))]


class SideSig:
    """Signatures of one side (S1, or S2 + S3 concatenated like data.Side)."""

    def __init__(self, side, files, name=None):
        self.parts = []
        self.name = name or "_".join(files)
        pos = 0
        for f, n in zip(files, side.sizes):
            self.parts.append((pos, pos + n, load(f, side.t.slice(pos, n))))
            pos += n

    def country(self, country, rows):
        d = os.path.join(config.SIG, f"sig_v{config.NORM_VERSION}_{SIG_VERSION}")
        safe = "".join(ch if ch.isalnum() else "_" for ch in country)
        return CountrySig(self, rows, os.path.join(d, f"{self.name}_{safe}_{len(rows)}_c16.npy"))

    def take(self, idx):
        idx = np.asarray(idx, np.int64)
        out = np.empty((len(idx), SIG_DIM), np.uint32)
        for a, b, arr in self.parts:
            m = (idx >= a) & (idx < b)
            if m.any():
                sub = idx[m] - a
                order = np.argsort(sub)                  # sequential reads of the memmap
                tmp = np.empty((len(sub), SIG_DIM), np.uint32)
                tmp[order] = arr[sub[order]]
                out[m] = tmp
        return out


def side_signatures(split, left, right):
    return (SideSig(left, [f"{split}_source1"], f"{split}_S1"),
            SideSig(right, [f"{split}_source2", f"{split}_source3"], f"{split}_S23"))


def pair_signature_features(SL, SR):
    """SL, SR: (m, SIG_DIM) signature rows (uint32 or compact uint16) of the two sides
    of m pairs -> feature dict."""
    F = {}
    empty = np.iinfo(SL.dtype).max
    for g, name in enumerate(GROUPS):
        a, b = SL[:, g * K:(g + 1) * K], SR[:, g * K:(g + 1) * K]
        valid = (a[:, 0] != empty) & (b[:, 0] != empty)
        F[f"sig_{name}"] = np.where(valid, (a == b).sum(axis=1) / K, np.nan).astype(np.float32)
    base = len(GROUPS) * K
    for j, name in enumerate(("house", "city", "state", "prefix")):
        a, b = SL[:, base + j], SR[:, base + j]
        both = (a != 0) & (b != 0)
        F[f"sig_{name}_eq"] = np.where(both, (a == b).astype(np.float32), np.nan)
    la, lb = SL[:, base + 4].astype(np.float32), SR[:, base + 4].astype(np.float32)
    F["sig_len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    fr = SR[:, base + 5]
    F["sig_b_empty"] = (fr & 1).astype(np.int8)
    F["sig_b_web"] = ((fr >> 1) & 1).astype(np.int8)
    F["sig_b_native"] = ((fr >> 2) & 1).astype(np.int8)
    return F


SIG_FEATS = ([f"sig_{g}" for g in GROUPS] + [f"sig_{n}_eq" for n in ("house", "city", "state", "prefix")] +
             ["sig_len_ratio", "sig_b_empty", "sig_b_web", "sig_b_native"])
