"""Expected-F0.5 decision: for each S1, accept the k top-scored candidates (k = 0..m)
that maximise the expected F0.5 under independent Bernoulli truths.

With TP_in ~ PoissonBinomial(p_1..p_k) and FN_out ~ PoissonBinomial(p_{k+1}..p_m) (+ an
expected number of true matches the candidates missed), F0.5 = 1.25 TP / (1.25 TP +
0.25 FN + FP) with FP = k - TP; a singleton (no true match) scores 1 only when k = 0.
Everything is exact (small m), vectorised over S1 groups of equal candidate count."""
import numpy as np


def _pb_pmf(P):
    """Poisson-binomial PMFs of every prefix of each row of P (n, m) -> (n, m+1, m+1):
    out[:, k, t] = P(sum of the first k Bernoullis = t)."""
    n, m = P.shape
    out = np.zeros((n, m + 1, m + 1))
    out[:, 0, 0] = 1.0
    for k in range(1, m + 1):
        q = P[:, k - 1:k]
        out[:, k, 1:] = out[:, k - 1, :-1] * q + out[:, k - 1, 1:] * (1 - q)
        out[:, k, 0] = out[:, k - 1, 0] * (1 - q[:, 0])
    return out


def best_k(P, miss=0.0):
    """P: (n, m) probabilities sorted descending per row (pad with 0). miss: expected
    number of true matches outside the candidates (adds to FN). Returns k per row."""
    n, m = P.shape
    pre = _pb_pmf(P)                                   # TP among the first k
    suf = _pb_pmf(P[:, ::-1])                          # trues among the last (m - k)
    t = np.arange(m + 1)
    best, arg = np.full(n, -1.0), np.zeros(n, np.int64)
    for k in range(m + 1):
        a = pre[:, k, :k + 1]                          # P(TP = i), i = 0..k
        b = suf[:, m - k, :m - k + 1]                  # P(FN_in = j), j = 0..m-k
        i = t[:k + 1][None, :, None]
        jj = t[:m - k + 1][None, None, :]              # true matches left among the candidates
        fp = k - i
        num = 1.25 * i
        den = 1.25 * i + 0.25 * (jj + miss) + fp
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        if k == 0:
            # predict nothing: scores 1 only if the S1 has no true match at all (none left
            # among the candidates and none missed by them: P = exp(-miss))
            f = np.where(jj == 0, np.exp(-miss), 0.0) + 0.0 * i
        e = np.einsum("ni,nj,nij->n", a, b, np.broadcast_to(f, (n, k + 1, m - k + 1)))
        better = e > best
        best[better], arg[better] = e[better], k
    return arg, best


def expected_f_mask(l, p, max_m=15, miss=0.0):
    """Mask of accepted pairs: per S1 (l), the best-k by expected F0.5."""
    l = np.asarray(l)
    order = np.lexsort((-p, l))
    ls, ps = l[order], p[order]
    start = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]
    size = np.diff(np.r_[start, len(ls)])
    rank = np.arange(len(ls)) - np.repeat(start, size)
    keep_sorted = np.zeros(len(ls), bool)
    for m in np.unique(np.minimum(size, max_m)):
        g = np.flatnonzero(np.minimum(size, max_m) == m)          # S1 groups with m candidates
        idx = start[g][:, None] + np.arange(m)[None, :]
        P = ps[idx]
        k, _ = best_k(P, miss)
        keep_sorted[idx[np.arange(m)[None, :] < k[:, None]]] = True
    out = np.zeros(len(l), bool)
    out[order] = keep_sorted & (rank < max_m)
    return out
