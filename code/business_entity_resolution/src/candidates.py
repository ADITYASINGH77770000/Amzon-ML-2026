"""Stage 1 + 1.5 + 2 driver: blocking, pass-score pre-ranker, cheap features, pruning.

Memory plan (8 GB laptop): for each country and pass the index is built once,
written to disk as .npy and memory-mapped. S1 rows are streamed in super-chunks;
each super-chunk is joined against every forward pass (plus the reverse-pass
hits that point at it), unioned, pre-ranked on pass scores / ranks (Stage 1.5,
no string work), cut to the top M, scored with cheap string features and pruned
by the Stage-2 ranker before the next super-chunk starts, so peak memory is
bounded by the chunk size rather than by |S1| x |S2 U S3|."""
import gc
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config
from blocking import _vectorizer, _weight, field_texts, index_tag, topk_rows, union, N_FEATURES
from features import pair_features

CHEAP_INPUT = ["name_core", "name_join", "addr_core", "house_no", "addr_empty", "is_web", "name_native"]


class PassIndex:
    """Inverted index of one pass for one country, memory-mapped from disk.

    Built once per (split, country, pass) and reused. The document frequency of
    every kept feature is stored so the purge cap can be tightened at query time
    (a feature dropped on the query side contributes nothing to the product)."""

    ARRS = ("data", "indices", "indptr")

    def __init__(self, texts, cfg, path, chunk=400_000):
        self.cfg = cfg
        self.vec = _vectorizer(cfg["analyzer"], cfg["ngram"])
        meta = os.path.join(path, "meta.npz")
        if not os.path.exists(meta):
            self._build(texts, path, chunk)
        z = np.load(meta)
        self.idf, self.keep, self.df_kept = z["idf"], z["keep"], z["df_kept"]
        self.path, self.shape = path, tuple(z["shape"])
        self.Rt = self._matrix("r")
        qmax = cfg.get("qmax", cfg["max_df"])
        self.qmask = (self.df_kept <= qmax).astype(np.float32)

    def _matrix(self, mode):
        return sp.csr_matrix(tuple(np.load(os.path.join(self.path, k + ".npy"), mmap_mode=mode)
                                   for k in self.ARRS), shape=self.shape)

    def load(self):
        """Read the whole index into RAM with one sequential read (random page faults on a
        cold memmap made each 2,000-row query chunk ~20x slower on this 8 GB machine)."""
        self.Rt = self._matrix(None)

    def release(self):
        self.Rt = self._matrix("r")

    def _build(self, texts, path, chunk):
        texts = texts() if callable(texts) else texts
        os.makedirs(path, exist_ok=True)
        df = np.zeros(N_FEATURES, np.int64)
        for s in range(0, len(texts), chunk):
            M = self.vec.transform(texts[s:s + chunk])
            df += np.bincount(M.indices, minlength=N_FEATURES)
        n = max(len(texts), 1)
        idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        keep = np.flatnonzero((df > 0) & (df <= self.cfg["max_df"]))
        parts = [_weight(self.vec.transform(texts[s:s + chunk]), idf, keep)
                 for s in range(0, len(texts), chunk)]
        R = sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0]
        del parts
        Rt = R.T.tocsr()
        del R
        for k in self.ARRS:
            np.save(os.path.join(path, k + ".npy"), getattr(Rt, k))
        np.savez(os.path.join(path, "meta.npz"), idf=idf, keep=keep, df_kept=df[keep].astype(np.int32),
                 shape=np.array(Rt.shape))

    def query(self, texts, chunk=2000):
        """-> (query row, indexed row, score, rank of the hit inside its query row)."""
        k, min_score = self.cfg["k"], self.cfg.get("min_score", 0.0)
        out_i, out_j, out_v = [], [], []
        for s in range(0, len(texts), chunk):
            L = _weight(self.vec.transform(texts[s:s + chunk]), self.idf, self.keep)
            L.data *= self.qmask[L.indices]
            L.eliminate_zeros()
            M = (L @ self.Rt).tocsr()
            if min_score > 0:
                M.data[M.data < min_score] = 0
            i, j, v = topk_rows(M, k)
            out_i.append(i + s)
            out_j.append(j.astype(np.int32))
            out_v.append(v.astype(np.float32))
        i, j, v = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_v)
        rank = (np.arange(len(i)) - np.searchsorted(i, i)).astype(np.float32)   # rows sorted, score desc
        return i, j, v, rank


def gather(side, idx, cols):
    """Columns for rows idx. The take runs on sorted indices (sequential reads of the
    memory-mapped table) and is put back in request order inside Arrow."""
    import pyarrow as pa
    idx = np.asarray(idx, np.int64)
    order = np.argsort(idx, kind="stable")
    inv = np.empty(len(idx), np.int64)
    inv[order] = np.arange(len(idx))
    s_idx, inv_a = pa.array(idx[order]), pa.array(inv)
    out = {}
    for c in cols:
        col = side.t[c].take(s_idx).take(inv_a)
        out[c] = col.to_pylist() if side.t[c].type == "string" else col.to_numpy()
    return out


def cheap_frame(W, left, right, n2):
    li, ri = W["l"].values, W["r"].values
    L = gather(left, li, CHEAP_INPUT)
    R = gather(right, ri, CHEAP_INPUT)
    R["is_s3"] = (ri >= n2).astype(np.int8)
    F = pair_features(L, R, None, None, level="cheap")
    pass_cols = [c for c in W.columns if c.startswith(("score_", "rank_", "sig_")) or c in ("n_passes", "p0")]
    for c in pass_cols:
        F[c] = W[c].values
    return F


def topk_by(F, col, k):
    F = F.sort_values(["l", col], ascending=[True, False])
    return F[F.groupby("l").cumcount() < k]


def build(split, left, right, n2, passes, s1_rows=None, stage15=None, M=None, stage2=None,
          budget=None, tau=None, superchunk=20_000, log=print, keep_cols=None, raw=False, sink=None,
          sigs=None, block=30_000):
    """Candidate table for the S1 rows `s1_rows` (default all).

    passes: {name: cfg}; cfg["reverse"] marks an S1-side index queried by the right
    records selected by cfg["right_filter"]. stage15 / stage2: callables F -> score
    (F keeps column 'l' so each row can use its out-of-fold model). sigs: (left,
    right) signatures.SideSig; their agreement features are added to the union.
    raw=True returns the union (pass scores / ranks / signatures), without pruning.

    Querying is pass-outer over blocks of `block` S1: one pass index is read into the
    page cache once per block instead of every pass once per super-chunk (the ten
    indexes of a country do not fit in RAM together; cycling them was 8x slower)."""
    from signatures import pair_signature_features
    tmp = os.path.join(config.INDEX, split)
    c1, lab1 = left.codes("country")
    c2, lab2 = right.codes("country")
    rows = np.arange(left.n) if s1_rows is None else np.asarray(s1_rows)
    names = list(passes)
    out = []
    for code in np.unique(c1[rows]):
        c = lab1[code]
        Li = rows[c1[rows] == code]
        Ri = np.flatnonzero(c2 == lab2.index(c)) if c in lab2 else np.array([], np.int64)
        log(f"country {c}: {len(Li):,} S1 rows vs {len(Ri):,} right rows")
        if len(Ri) == 0:
            continue
        idx, rev = {}, []
        for p, cfg in passes.items():
            t = time.time()
            path = os.path.join(tmp, f"{c}_{index_tag(p, cfg)}")
            if cfg.get("reverse"):
                Lall = np.flatnonzero(c1 == code)           # every S1 of the country is indexed
                Rs = Ri[right.np(cfg["right_filter"], Ri).astype(bool)]
                ix = PassIndex(lambda: field_texts(left, cfg["field"], Lall), cfg, path)
                i, j, v, rk = ix.query(field_texts(right, cfg["field"], Rs))
                l_, r_ = Lall[j], Rs[i]
                keep = np.isin(l_, Li)
                rev.append(pd.DataFrame({"l": l_[keep].astype(np.int32), "r": r_[keep].astype(np.int32),
                                         "pass": p, "score": v[keep], "rank": rk[keep]}))
                del ix
                log(f"  reverse {p}: {len(Rs):,} right rows -> {keep.sum():,} hits, {time.time() - t:.0f}s")
            else:
                idx[p] = PassIndex(lambda: field_texts(right, cfg["field"], Ri), cfg, path)
                log(f"  index {p}: {idx[p].Rt.nnz:,} nnz, {time.time() - t:.0f}s")
        rev = pd.concat(rev, ignore_index=True) if rev else None
        if sigs is not None:                        # compact per-country signatures, held in RAM
            t = time.time()
            sl = sigs[0].country(c, np.flatnonzero(c1 == code))
            sr = sigs[1].country(c, Ri)
            log(f"  signatures for {c}: {sl.arr.nbytes / 1e6:.0f} + {sr.arr.nbytes / 1e6:.0f} MB, {time.time() - t:.0f}s")
        inchunk = np.zeros(left.n, bool)
        for b0 in range(0, len(Li), block):
            Lb = Li[b0:b0 + block]
            t = time.time()
            hits = {}
            for p, ix in idx.items():                    # pass-outer: each index read once per block
                ix.load()
                i, j, v, rk = ix.query(field_texts(left, passes[p]["field"], Lb))
                ix.release()
                hits[p] = (i.astype(np.int32), Ri[j].astype(np.int32), v, rk)
            log(f"  block {b0 + len(Lb):,}/{len(Li):,}: queries {time.time() - t:.0f}s")
            for s in range(0, len(Lb), superchunk):
                t = time.time()
                Lc = Lb[s:s + superchunk]
                frames = []
                for p, (i, r_, v, rk) in hits.items():
                    a, e = np.searchsorted(i, s), np.searchsorted(i, s + superchunk)
                    frames.append(pd.DataFrame({"l": Lb[i[a:e]].astype(np.int32), "r": r_[a:e],
                                                "pass": p, "score": v[a:e], "rank": rk[a:e]}))
                if rev is not None:
                    inchunk[Lc] = True
                    frames.append(rev[inchunk[rev["l"].values]])
                    inchunk[Lc] = False
                W = union(frames, names)
                del frames
                if sigs is not None:
                    for k, v in pair_signature_features(sl.take(W["l"].values), sr.take(W["r"].values)).items():
                        W[k] = v
                n_raw = len(W)
                self_log = f"  S1 {b0 + s + len(Lc):,}/{len(Li):,}"
                W = _finish(W, raw, stage15, M, c, stage2, tau, budget, left, right, n2, keep_cols)
                n_kept = len(W)
                if sink is not None:
                    sink(W.reset_index(drop=True))      # stream to disk, keep nothing in RAM
                else:
                    out.append(W.reset_index(drop=True))
                del W
                log(f"{self_log}: raw {n_raw:,} -> kept {n_kept:,} pairs, {time.time() - t:.0f}s")
            del hits
        del idx
        gc.collect()
    return pd.concat(out, ignore_index=True) if out else None


def _finish(W, raw, stage15, M, c, stage2, tau, budget, left, right, n2, keep_cols):
    """Optional in-builder pruning (Stage 1.5 top-M, cheap features, Stage 2)."""
    if not raw:
        if stage15 is not None:
            W["p0"] = stage15(W).astype(np.float32)
            m = (M.get(c, max(M.values())) if isinstance(M, dict) else M)   # unseen country: largest
            if m:
                W = topk_by(W, "p0", m)
        F = cheap_frame(W, left, right, n2)
        F.insert(0, "l", W["l"].values)
        F.insert(1, "r", W["r"].values)
        if stage2 is not None:
            F["p1"] = stage2(F).astype(np.float32)
            if tau is not None:
                F = F[F["p1"] >= tau]
            if budget:
                F = topk_by(F, "p1", budget)
        W = F
    if keep_cols:
        W = W[[k for k in keep_cols if k in W]]
    return W
