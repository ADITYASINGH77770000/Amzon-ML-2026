"""Stage 1: memory-safe blocking.

Each pass hashes a text field into sparse features (no vocabulary kept in RAM),
weights them by IDF computed on the right side (S2 U S3), purges features whose
right-side document frequency exceeds max_df, and joins S1 against the right
side with a chunked sparse product (an inverted index). Per S1 row only the top
k scores are kept, so cost and output are bounded by the caps. Passes run inside
each country label found in the data (open set; nothing is hard-coded).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 1 << 22

from blocking_keys import field_texts  # noqa: F401  (derived key fields)

# k / min_score are query-time settings (not part of the index hash); every pass
# retrieves generously and Stage 2 prunes, which is safer than losing a match here.
PASSES = {
    # name tokens + bigrams (+ joined name): abbreviations, reordering, websites
    "A": dict(field="name_core_join", analyzer="word", ngram=(1, 2), k=25, max_df=2000, min_score=0.2),
    # house-number + street-word keys: trade names / garbled names at one address
    "B": dict(field="street_keys", analyzer="word", ngram=(1, 1), k=25, max_df=300),
    # name-token (incl. joined / domain form) x address-word conjunctions
    "D": dict(field="name_addr_keys", analyzer="word", ngram=(1, 1), k=25, max_df=300, min_score=0.1, v=2),
    # address-word x address-word conjunctions: garbled names, missing house numbers
    "E": dict(field="addr_pair_keys", analyzer="word", ngram=(1, 1), k=25, max_df=300, min_score=0.1),
}
# Candidate passes evaluated in recall_lab.py before being promoted into PASSES.
EXTRA_PASSES = {
    # phonetic consonant skeletons: transliteration and vowel typos
    "P": dict(field="skeleton_tokens", analyzer="word", ngram=(1, 2), k=25, max_df=2000),
    # 3-letter prefix pairs of name tokens: typo'd names with no address
    "X": dict(field="prefix_pair_keys", analyzer="word", ngram=(1, 1), k=25, max_df=300),
    # whole name x address word
    "N": dict(field="fullname_anchor_keys", analyzer="word", ngram=(1, 1), k=25, max_df=100),
    # exact order-free address
    "G": dict(field="exact_addr_key", analyzer="word", ngram=(1, 1), k=25, max_df=50),
    # joined-name prefixes (+ address word): partial website names
    "W": dict(field="join_prefix_keys", analyzer="word", ngram=(1, 1), k=25, max_df=300),
    # REVERSE: an S1-side name index queried by right records that have no address.
    # Forward top-k cannot reach them for generic names (hundreds of ties), but each
    # right record belongs to at most one S1, so its own top-k S1 list is short.
    "RA": dict(field="name_core_join", analyzer="word", ngram=(1, 2), k=10, max_df=1000,
               reverse=True, right_filter="addr_empty"),
}
# Char 3-grams of the name were profiled and dropped: 44% recall at ~19 s per
# 4,000 S1 rows on 6.2M right records (common names share thousands of trigrams).


def index_tag(p, cfg):
    """Index directory name: pass + short hash of its build settings (stale-proof)."""
    import hashlib
    key = repr(sorted((k, v) for k, v in cfg.items() if k not in ("k", "min_score", "qmax")))
    return f"{p}_{hashlib.md5(key.encode()).hexdigest()[:6]}"


def _vectorizer(analyzer, ngram):
    extra = {"token_pattern": r"\S+"} if analyzer == "word" else {}
    return HashingVectorizer(n_features=N_FEATURES, analyzer=analyzer, ngram_range=ngram,
                             alternate_sign=False, norm=None, binary=True, lowercase=False,
                             dtype=np.float32, **extra)


def _transform(vec, texts, chunk=500_000):
    mats = [vec.transform(texts[s:s + chunk]) for s in range(0, len(texts), chunk)]
    return sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]


def _weight(M, idf, keep):
    """IDF-weight, L2-normalise over all features, then drop purged ones."""
    M = M.multiply(idf.reshape(1, -1)).tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    M = sp.diags(1.0 / np.maximum(norms, 1e-9)).astype(np.float32) @ M
    M = M.tocsc()[:, keep].tocsr() if keep is not None else M
    return M


def topk_rows(M, k):
    """Vectorised: keep the k largest entries of each CSR row (k=None keeps all)."""
    M = M.tocsr()
    M.eliminate_zeros()
    rows = np.repeat(np.arange(M.shape[0], dtype=np.int32), np.diff(M.indptr))
    if k is None:
        return rows, M.indices.copy(), M.data.copy()
    # rows are < 2^20 and cosine scores in [0, 1]: rows * 4 - score orders by row, then score desc
    order = np.argsort(rows.astype(np.float64) * 4.0 - M.data, kind="quicksort")
    rank = np.arange(len(order)) - M.indptr[rows[order]]
    keep = order[rank < k]
    return rows[keep], M.indices[keep], M.data[keep]


def union(frames, pass_names):
    """Wide table of unique (l, r) with one score and one rank column per pass
    (NaN = not found by that pass). Frames carry l, r, pass, score, rank."""
    frames = [f for f in frames if len(f)]
    l = np.concatenate([f["l"].values for f in frames]).astype(np.int64)
    r = np.concatenate([f["r"].values for f in frames]).astype(np.int64)
    key = (l << 32) | r
    uk, inv = np.unique(key, return_inverse=True)
    wide = pd.DataFrame({"l": (uk >> 32).astype(np.int32), "r": (uk & 0xFFFFFFFF).astype(np.int32)})
    score = {p: np.full(len(uk), np.nan, np.float32) for p in pass_names}
    rank = {p: np.full(len(uk), np.nan, np.float32) for p in pass_names}
    pos = 0
    for f in frames:
        n, p = len(f), f["pass"].iloc[0]
        score[p][inv[pos:pos + n]] = f["score"].values
        rank[p][inv[pos:pos + n]] = f["rank"].values
        pos += n
    for p in pass_names:
        wide[f"score_{p}"] = score[p]
        wide[f"rank_{p}"] = rank[p]
    wide["n_passes"] = sum(~np.isnan(score[p]) for p in pass_names).astype(np.int8)
    return wide
