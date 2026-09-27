"""Stage 2 (cheap) and Stage 3 (full) pair features.

All features are relative (how do the two records compare), never absolute:
no country value, no ids, no raw strings. Token IDF comes from hashed document
frequencies over the right-side pool, and every IDF-weighted overlap is computed
with sparse row operations instead of Python loops."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import HashingVectorizer

NF = 1 << 21
_HV = HashingVectorizer(n_features=NF, token_pattern=r"\S+", lowercase=False, alternate_sign=False,
                        norm=None, binary=True, dtype=np.float32)


def token_matrix(texts, chunk=500_000):
    mats = [_HV.transform(texts[s:s + chunk]) for s in range(0, len(texts), chunk)]
    return sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]


class IdfTable:
    """Hashed token IDF for one field, fitted on the right-side pool."""

    def __init__(self, chunks):
        """chunks: iterable of lists of strings (streamed to bound memory)."""
        df = np.zeros(NF, np.int64)
        n = 0
        for texts in chunks:
            M = token_matrix(texts)
            df += np.bincount(M.indices, minlength=NF)
            n += M.shape[0]
        self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        self.rare = float(np.log((1 + n) / (1 + 50)) + 1)

    def save(self, path):
        np.savez_compressed(path, idf=self.idf, rare=self.rare)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        o = cls.__new__(cls)
        o.idf, o.rare = z["idf"], float(z["rare"])
        return o


def _idf_overlap(a, b, tab, prefix):
    A, B = token_matrix(a), token_matrix(b)
    S = A.multiply(B).tocsr()
    idf = tab.idf
    ia, ib, sh = A @ idf, B @ idf, S @ idf
    out = {
        f"{prefix}_idf_min": np.minimum(ia, ib),
        f"{prefix}_idf_max": np.maximum(ia, ib),
        f"{prefix}_idf_shared": sh,
        f"{prefix}_idf_ovl": sh / np.maximum(np.maximum(ia, ib), 1e-6),
        f"{prefix}_idf_ovl_min": sh / np.maximum(np.minimum(ia, ib), 1e-6),
        f"{prefix}_idf_jacc": sh / np.maximum(ia + ib - sh, 1e-6),
        f"{prefix}_rare_shared": S @ (idf >= tab.rare).astype(np.float32),
        f"{prefix}_n_shared": np.diff(S.indptr).astype(np.float32),
        f"{prefix}_ntok_a": np.diff(A.indptr).astype(np.float32),
        f"{prefix}_ntok_b": np.diff(B.indptr).astype(np.float32),
    }
    Sm = S.multiply(idf.reshape(1, -1)).tocsr()
    out[f"{prefix}_max_shared_idf"] = np.asarray(Sm.max(axis=1).todense()).ravel()
    return out


def _sim(x, y, scorer, workers):
    return process.cpdist(x, y, scorer=scorer, workers=workers, dtype=np.float32)


def _set_feats(a, b):
    """nums / street keys as sets: jaccard, conflict (both present, none shared)."""
    jac = np.empty(len(a), np.float32)
    conf = np.empty(len(a), np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            k = len(sx & sy)
            jac[i] = k / len(sx | sy)
            conf[i] = k == 0
        else:
            jac[i] = np.nan
            conf[i] = 0
    return jac, conf


def pair_features(L, R, idf_name, idf_addr, level="full", workers=-1):
    """L, R: dicts of equal-length lists/arrays (one entry per pair) for each field."""
    F = {}
    nc_a, nc_b = L["name_core"], R["name_core"]
    F["name_core_tset"] = _sim(nc_a, nc_b, fuzz.token_set_ratio, workers)
    F["name_core_jw"] = _sim(nc_a, nc_b, JaroWinkler.normalized_similarity, workers)
    ac_a, ac_b = L["addr_core"], R["addr_core"]
    F["addr_tset"] = _sim(ac_a, ac_b, fuzz.token_set_ratio, workers)
    ha, hb = np.asarray(L["house_no"], object), np.asarray(R["house_no"], object)
    hboth = (ha != "") & (hb != "")
    F["house_match"] = (hboth & (ha == hb)).astype(np.int8)
    F["house_conflict"] = (hboth & (ha != hb)).astype(np.int8)
    F["addr_empty_a"] = np.asarray(L["addr_empty"], np.int8)
    F["addr_empty_b"] = np.asarray(R["addr_empty"], np.int8)
    F["is_web_b"] = np.asarray(R["is_web"], np.int8)
    F["name_native_b"] = np.asarray(R["name_native"], np.int8)
    F["is_s3"] = np.asarray(R["is_s3"], np.int8)
    # joined-name partial match is cheap and carries the website / truncated-name cases
    F["name_join_partial"] = _sim(L["name_join"], R["name_join"], fuzz.partial_ratio, workers)
    if level == "cheap":
        return pd.DataFrame(F)

    nj_a, nj_b = L["name_join"], R["name_join"]
    F["name_core_tsort"] = _sim(nc_a, nc_b, fuzz.token_sort_ratio, workers)
    F["name_core_ratio"] = _sim(nc_a, nc_b, fuzz.ratio, workers)
    F["name_norm_tset"] = _sim(L["name_norm"], R["name_norm"], fuzz.token_set_ratio, workers)
    F["name_join_ratio"] = _sim(nj_a, nj_b, fuzz.ratio, workers)
    F["name_join_prefix"] = _sim([x[:6] for x in nj_a], [y[:6] for y in nj_b], Levenshtein.distance, workers)
    F["name_len_a"] = np.fromiter((len(x) for x in nj_a), np.float32, len(nj_a))
    F["name_len_b"] = np.fromiter((len(x) for x in nj_b), np.float32, len(nj_b))
    F["addr_tsort"] = _sim(ac_a, ac_b, fuzz.token_sort_ratio, workers)
    F["addr_jw"] = _sim(ac_a, ac_b, JaroWinkler.normalized_similarity, workers)
    F["addr_partial"] = _sim(ac_a, ac_b, fuzz.partial_token_set_ratio, workers)
    F.update(_idf_overlap(nc_a, nc_b, idf_name, "name"))
    F.update(_idf_overlap(ac_a, ac_b, idf_addr, "addr"))
    # house number near-miss (typo'd digits: 1922 vs 192)
    F["house_lev"] = np.where(hboth, _sim(list(ha), list(hb), Levenshtein.distance, workers), np.nan).astype(np.float32)
    F["nums_jacc"], F["nums_conflict"] = _set_feats(L["nums"], R["nums"])
    F["skey_jacc"], F["skey_conflict"] = _set_feats(L["street_keys"], R["street_keys"])
    pa_, pb_ = np.asarray(L["postal"], object), np.asarray(R["postal"], object)
    pboth = (pa_ != "") & (pb_ != "")
    F["postal_match"] = (pboth & (pa_ == pb_)).astype(np.int8)
    F["postal_conflict"] = (pboth & (pa_ != pb_)).astype(np.int8)
    la, lb = np.asarray(L["landmark"], object), np.asarray(R["landmark"], object)
    F["landmark_any"] = ((la != "") | (lb != "")).astype(np.int8)
    lt = _sim(list(la), list(lb), fuzz.token_set_ratio, workers)
    F["landmark_tset"] = np.where((la != "") & (lb != ""), lt, np.nan).astype(np.float32)
    F["postal_prefix3"] = (pboth & (np.asarray([x[:3] for x in pa_], object) ==
                                    np.asarray([y[:3] for y in pb_], object))).astype(np.int8)
    F.update(_component_feats(L, R, workers))
    F.update(_name_variant_feats(L, R, workers))
    # small deterministic layer, as features only
    same_core = np.asarray(nc_a, object) == np.asarray(nc_b, object)
    F["same_core"] = same_core.astype(np.int8)
    F["exact_pair"] = (same_core & (F["house_match"] == 1)).astype(np.int8)
    return pd.DataFrame(F)


def _both(a, b):
    a, b = np.asarray(a, object), np.asarray(b, object)
    return a, b, (a != "") & (b != "")


def _component_feats(L, R, workers):
    """Parsed city / state agreement and conflict (missing is never a conflict)."""
    F = {}
    ca, cb, cboth = _both(L["city"], R["city"])
    ct = _sim(list(ca), list(cb), fuzz.token_set_ratio, workers)
    F["city_tset"] = np.where(cboth, ct, np.nan).astype(np.float32)
    F["city_exact"] = (cboth & (ca == cb)).astype(np.int8)
    F["city_missing"] = (~cboth).astype(np.int8)
    # the S1 city appearing anywhere in the other address (handles reordered segments)
    F["city_in_addr_b"] = np.fromiter((bool(c) and (" " + c + " ") in (" " + a + " ")
                                       for c, a in zip(ca, R["addr_core"])), np.int8, len(ca))
    sa, sb, sboth = _both(L["state"], R["state"])
    F["state_match"] = (sboth & (sa == sb)).astype(np.int8)
    F["state_conflict"] = (sboth & (sa != sb)).astype(np.int8)
    F["state_missing"] = (~sboth).astype(np.int8)
    return F


def _initials(core):
    t = core.split()
    return "".join(x[0] for x in t) if len(t) >= 2 else ""


def _common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _name_variant_feats(L, R, workers):
    """Alternate / domain names, initials, joined-word prefixes, phonetic skeletons."""
    from blocking_keys import skeleton
    F = {}
    nc_a, nc_b = L["name_core"], R["name_core"]
    al_a, al_b = L["name_alt"], R["name_alt"]
    has_a = np.asarray([bool(x) for x in al_a])
    has_b = np.asarray([bool(x) for x in al_b])
    best = np.full(len(nc_a), np.nan, np.float32)
    for x, y, m in ((nc_a, al_b, has_b), (al_a, nc_b, has_a), (al_a, al_b, has_a & has_b)):
        if m.any():
            s = _sim(x, y, fuzz.token_set_ratio, workers)
            best = np.where(m, np.fmax(best, s), best)
    F["alt_best_tset"] = best
    F["has_alt_b"] = has_b.astype(np.int8)
    nj_a, nj_b = L["name_join"], R["name_join"]
    ia = [_initials(x) for x in nc_a]
    ib = [_initials(x) for x in nc_b]
    F["initials_match"] = np.fromiter(((x != "" and x == jb) or (y != "" and y == ja)
                                       for x, y, ja, jb in zip(ia, ib, nj_a, nj_b)), np.int8, len(ia))
    cp = np.fromiter((_common_prefix(a, b) for a, b in zip(nj_a, nj_b)), np.float32, len(nj_a))
    mn = np.fromiter((max(min(len(a), len(b)), 1) for a, b in zip(nj_a, nj_b)), np.float32, len(nj_a))
    F["join_prefix_len"] = cp
    F["join_prefix_frac"] = cp / mn            # 1.0 = one joined name is a prefix of the other
    ska = [" ".join(skeleton(t) for t in x.split()) for x in nc_a]
    skb = [" ".join(skeleton(t) for t in x.split()) for x in nc_b]
    F["skel_tset"] = _sim(ska, skb, fuzz.token_set_ratio, workers)
    F["skel_ratio"] = _sim([x.replace(" ", "") for x in ska], [y.replace(" ", "") for y in skb],
                           fuzz.ratio, workers)
    return F


def competition(P, score, prefix):
    """Rival features: how does this pair compare with the other candidates of its
    S1 (l) and of its S2/S3 record (r)."""
    s = P[score].values
    g = P.groupby("l")[score]
    best = g.transform("max").values
    P[f"{prefix}_rank_l"] = g.rank(ascending=False, method="first").values.astype(np.float32)
    P[f"{prefix}_gap_l"] = (best - s).astype(np.float32)
    top2 = P.sort_values(["l", score], ascending=[True, False]).groupby("l")[score].nth(1)
    sec = pd.Series(top2.values, index=P.loc[top2.index, "l"].values)
    P[f"{prefix}_best_minus_2nd_l"] = (best - P["l"].map(sec).fillna(0).values).astype(np.float32)
    P[f"{prefix}_n_l"] = g.transform("size").values.astype(np.float32)
    P[f"{prefix}_sum_l"] = g.transform("sum").values.astype(np.float32)
    h = P.groupby("r")[score]
    P[f"{prefix}_rank_r"] = h.rank(ascending=False, method="first").values.astype(np.float32)
    P[f"{prefix}_gap_r"] = (h.transform("max").values - s).astype(np.float32)
    P[f"{prefix}_n_r"] = h.transform("size").values.astype(np.float32)
    return P
