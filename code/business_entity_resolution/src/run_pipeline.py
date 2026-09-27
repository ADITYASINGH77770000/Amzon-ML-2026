"""One command: data -> candidate_pairs.tsv + matching_results.tsv.

Every model is trained on ALL labelled Source 1 records and their S2 / S3 data;
nothing is reserved as a validation or holdout set.

  * Filters (Stage 1.5 pre-ranker, Stage 2 fast ranker) are a cross-fitted pair over
    100% of the labelled S1: one model per random stratified half (A / B). Their
    scores are matcher inputs, so a training pair must be scored by the model that
    did not see it (as a test pair is); test pairs get the average of both models.
  * The matcher (Stage 3) is EVALUATED by 2-fold cross-validation over all labelled
    S1 (out-of-fold macro F0.5 over 2.2M S1, thresholds tuned on those scores) and
    the PRODUCTION matcher is then trained once on 100% of the labelled pairs.
  * The filters cannot hold ~770M / ~145M candidate pairs in 8 GB, so one `collect`
    pass over ALL labelled S1 keeps every true pair plus a weighted sample of
    negatives (hard ones at a higher rate, weight 1 / rate).

    python run_pipeline.py all
    python run_pipeline.py prepare prov collect fit12 cands_train assemble_train feats_train stage3 \
                           stack final test
"""
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
import candidates
import decide
import evaluate
import split as splitmod
from blocking import PASSES
from crossfit import CrossModel, competition, fit_two_fold, make_folds
from data import load_split, truth_arrays
from features import IdfTable, pair_features
from io_utils import write_tsv
from monotone import params_for
from signatures import SIG_FEATS, side_signatures

MODELS = config.MODELS                      # all mode-specific (config.RUN)
WORK = config.WORK
LOG_PATH = config.LOG
FULL_COLS = ["name_norm", "name_core", "name_join", "name_alt", "addr_core", "landmark", "postal", "nums",
             "house_no", "street_keys", "state", "city", "is_web", "name_native", "addr_empty"]
G_FOLD0, G_VALID, G_FOLD1, G_HOLD, G_TEST = 0, 1, 2, 3, 9
M_PRIME = {"US": 60, "India": 160}          # generous provisional top-M' for the Stage 2 collection
N_EVAL = 15_000                             # S1 whose complete candidate lists are kept (M / budget)

# ranking stages are monotone (monotone.py) and regularised: an unconstrained pre-ranker
# pushed identical true pairs to p ~ 1e-9 because of a spurious "pass rank >= 4" split
# Their size is capped because they score every raw pair (~770M per train pass):
# Stage 1.5 at 31 leaves x <= 300 trees costs ~4 us / pair with early-stopped prediction.
STAGE15_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200,
                      lambda_l2=10.0, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                      verbose=-1, num_threads=config.N_JOBS, seed=config.SEED)
STAGE15_ROUNDS = 300
STAGE2_PARAMS = dict(STAGE15_PARAMS, num_leaves=63, min_data_in_leaf=100)
STAGE2_ROUNDS = 500
PROV_PARAMS = dict(STAGE15_PARAMS, learning_rate=0.3)          # fast selector: 60 trees, ~1.5 us / pair
PROV_ROUNDS = 60
STAGE3_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                     max_bin=63, verbose=-1, num_threads=config.N_JOBS, seed=config.SEED)


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def jdump(obj, name):
    json.dump(obj, open(os.path.join(MODELS, name), "w"), indent=2, default=float)


def jload(name):
    return json.load(open(os.path.join(MODELS, name)))


def active_passes():
    """Blocking passes and query settings chosen in recall_lab (passes.json), else defaults."""
    path = os.path.join(MODELS, "passes.json")
    if not os.path.exists(path):
        path = os.path.join(config.CACHE, "models", "passes.json")     # shared lab choice
    if not os.path.exists(path):
        return dict(PASSES)
    raw = json.load(open(path))
    return {p: {k: tuple(v) if k == "ngram" else v for k, v in cfg.items()} for p, cfg in raw.items()}


def stage15_cols(passes):
    """Stage 1.5 inputs: pass scores / ranks and record-signature agreement (no strings)."""
    return [f"{a}_{p}" for p in passes for a in ("score", "rank")] + ["n_passes"] + SIG_FEATS


def context(split):
    left, right = load_split(split)
    ctx = dict(left=left, right=right, n2=right.sizes[0], sigs=side_signatures(split, left, right))
    ctx["c1"], ctx["lab1"] = left.codes("country")
    if split == "train":
        true_s1, n_true = truth_arrays(left, right)
        # GroupKFold by S1 (stratified): every labelled S1 belongs to exactly one fold; no
        # data is reserved — folds only define which model scores which S1 out-of-fold
        fold = splitmod.load_folds(left, right, true_s1, n_true)
        rng = np.random.default_rng(config.SEED + 2)
        # S1 whose complete candidate lists are kept (to choose M / budget); also trained on
        E = np.sort(rng.choice(left.n, min(N_EVAL, left.n // 8), replace=False))
        ctx.update(true_s1=true_s1, n_true=n_true, fold=fold, E=E, labelled=np.arange(left.n))
    return ctx


def oracle_by_country(l, r, ctx, entities):
    """Candidate-oracle macro F0.5 (a perfect matcher inside the candidates), per country."""
    left, true_s1, n_true = ctx["left"], ctx["true_s1"], ctx["n_true"]
    c1, lab = ctx["c1"], ctx["lab1"]
    y = true_s1[r] == l
    hit = np.bincount(l[y], minlength=left.n)
    cnt = np.bincount(l, minlength=left.n)
    f = evaluate.f05_arrays(hit, hit, n_true)
    res = {}
    for code, name in list(enumerate(lab)) + [(-1, "all")]:
        e = entities if code < 0 else entities[c1[entities] == code]
        if len(e):
            res[name] = dict(oracle=round(float(f[e].mean()), 5), cands=round(float(cnt[e].mean()), 2),
                             p95_cands=float(np.percentile(cnt[e], 95)),
                             recall=round(float(hit[e].sum() / max(n_true[e].sum(), 1)), 5))
    return res


def topk_mask(l, score, limit):
    """Rows among the `limit` best `score` of their l (limit: scalar)."""
    order = np.lexsort((-score, l))
    ls = l[order]
    start = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]
    rank = np.arange(len(l)) - np.repeat(start, np.diff(np.r_[start, len(l)]))
    mask = np.zeros(len(l), bool)
    mask[order[rank < limit]] = True
    return mask


def country_limit(M, country):
    return M.get(country, max(M.values())) if isinstance(M, dict) else M   # unseen country: largest


class PartWriter:
    def __init__(self, path, keep_first=0):
        """keep_first > 0 resumes: parts 0..keep_first-1 are kept. Older parts beyond that
        are never deleted: they are moved to a time-stamped backup folder."""
        self.path, self.k = path, keep_first
        os.makedirs(path, exist_ok=True)
        stale = [f for f in os.listdir(path) if f.endswith(".parquet") and int(f[4:-8]) >= keep_first]
        if stale:
            bak = f"{path}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(bak)
            for f in stale:
                os.replace(os.path.join(path, f), os.path.join(bak, f))

    def __call__(self, df):
        if len(df):
            df.to_parquet(os.path.join(self.path, f"part{self.k:05d}.parquet"))
            self.k += 1


# ------------------------------------------------------------------ provisional Stage 1.5
def step_prov():
    """A small, fast Stage 1.5 fitted on a 12k-S1 train sample. It is used ONLY to choose
    which NEGATIVES the collect pass samples for the Stage 2 training set (generous
    top-M'); every positive pair is kept regardless."""
    ctx = context("train")
    passes = active_passes()
    rows = np.flatnonzero(ctx["fold"] == 0)
    SA = np.sort(np.random.default_rng(config.SEED).choice(rows, min(12_000, len(rows) // 3), replace=False))
    W = candidates.build("train", ctx["left"], ctx["right"], ctx["n2"], passes, s1_rows=SA, raw=True,
                         log=log, sigs=ctx["sigs"])
    cols15 = stage15_cols(passes)
    X = W[cols15].to_numpy(np.float32)
    y = (ctx["true_s1"][W["r"].values] == W["l"].values).astype(np.int8)
    half = (W["l"].values % 2).astype(np.int8)
    models, _ = fit_two_fold(X, y, half, params_for(PROV_PARAMS, cols15), PROV_ROUNDS, early=30, log=log)
    CrossModel(models, cols15).save(os.path.join(MODELS, "stage15_prov"))


# ------------------------------------------------------------------ collect (all labelled S1)
class Collector:
    """Per super-chunk of the raw union: write the Stage 1.5 and Stage 2 training samples."""

    def __init__(self, ctx, prov, cols15, seed=config.SEED, keep_first=0):
        self.ctx, self.prov, self.cols15 = ctx, prov, cols15
        self.rng = np.random.default_rng(seed + keep_first)
        self.inE = np.zeros(ctx["left"].n, bool)
        self.inE[ctx["E"]] = True
        self.w15 = PartWriter(os.path.join(WORK, "collect15"), keep_first)
        self.w2 = PartWriter(os.path.join(WORK, "collect2"), keep_first)

    def __call__(self, W):
        ctx = self.ctx
        if not len(W):
            return
        l, r = W["l"].values, W["r"].values
        y = ctx["true_s1"][r] == l
        ev = self.inE[l]
        # Stage 1.5 sample: every positive, hard negatives at 5%, the rest at 0.5%;
        # validation-eval S1 keep their complete lists (weight 1)
        hard = (W["n_passes"].values >= 3) | (np.nan_to_num(W["sig_name"].values) >= 0.5) | \
               (np.nan_to_num(W["sig_addr"].values) >= 0.5)
        rate = np.where(y, 1.0, np.where(hard, 0.05, 0.005))
        keep = ev | (self.rng.random(len(W)) < rate)
        S = W.loc[keep, ["l", "r"] + self.cols15].copy()
        S["y"], S["w"], S["ev"] = y[keep].astype(np.int8), np.where(ev[keep], 1.0, 1 / rate[keep]).astype(np.float32), ev[keep].astype(np.int8)
        self.w15(S)
        # Stage 2 sample: EVERY positive, plus negatives from the provisional top-M' of each
        # S1 (hard ones at 25%, the rest at 1%). Validation-eval lists for Stage 2 are built
        # later in fit12 from the dev Stage 1.5, so no eval rows are needed here.
        country = ctx["lab1"][ctx["c1"][l[0]]]
        p0 = self.prov.predict_rows(W)
        top = topk_mask(l, p0, country_limit(M_PRIME, country))
        rate2 = np.where(y, 1.0, np.where(p0 >= 0.01, 0.25, 0.01))
        keep2 = y | (top & (self.rng.random(len(W)) < rate2))
        Wk = W[keep2]
        F = candidates.cheap_frame(Wk, ctx["left"], ctx["right"], ctx["n2"])
        F.insert(0, "l", Wk["l"].values)
        F.insert(1, "r", Wk["r"].values)
        F["y"] = y[keep2].astype(np.int8)
        F["w"] = (1 / rate2[keep2]).astype(np.float32)
        F["ev"] = np.zeros(int(keep2.sum()), np.int8)
        self.w2(F)


COLLECT_CHUNK = 10_000


def count_parts(path):
    return len([f for f in os.listdir(path) if f.endswith(".parquet")]) if os.path.isdir(path) else 0


def remaining_rows(rows, c1, done, chunk=COLLECT_CHUNK):
    """S1 rows not yet covered after `done` finished super-chunks. The builder writes one
    part per super-chunk in a fixed order: countries by code, each country's rows
    ascending in chunks of `chunk` (a country's last chunk may be partial)."""
    chunks = []
    for code in np.unique(c1[rows]):
        Lc = rows[c1[rows] == code]
        chunks += [Lc[s:s + chunk] for s in range(0, len(Lc), chunk)]
    covered = sum(len(x) for x in chunks[:done])
    rest = np.sort(np.concatenate(chunks[done:])) if done < len(chunks) else rows[:0]
    return rest, covered


def step_collect(resume=False):
    ctx = context("train")
    passes = active_passes()
    prov = CrossModel.load(os.path.join(MODELS, "stage15_prov"))
    labelled = ctx["labelled"]
    done = 0
    if resume:
        # a chunk counts as done when BOTH samples have its part
        done = min(count_parts(os.path.join(WORK, d)) for d in ("collect15", "collect2"))
        labelled, covered = remaining_rows(labelled, ctx["c1"], done)
        log(f"collect resume: keeping {done} finished chunks ({covered:,} S1), {len(labelled):,} S1 to go")
    col = Collector(ctx, prov, stage15_cols(passes), keep_first=done)
    candidates.build("train", ctx["left"], ctx["right"], ctx["n2"], passes, s1_rows=labelled, raw=True,
                     log=log, sigs=ctx["sigs"], sink=col, superchunk=COLLECT_CHUNK)
    log(f"collect: {col.w15.k} Stage-1.5 parts, {col.w2.k} Stage-2 parts")


def step_collect_resume():
    step_collect(resume=True)


# ------------------------------------------------------------------ fit Stage 1.5 / 2 (dev + final)
def assemble_sample(name, cols, ctx):
    """Parquet parts -> memmaps ordered by key = 2*fold + eval: fold j occupies keys 2j
    (sampled rows) and 2j+1 (complete eval lists), one contiguous block per fold."""
    src = os.path.join(WORK, name)
    files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".parquet"))
    keys = []
    for f in files:
        t = pq.read_table(f, columns=["l", "ev"])
        keys.append((2 * ctx["fold"][t["l"].to_numpy()].astype(np.int16) + t["ev"].to_numpy()).astype(np.int16))
    n = sum(len(k) for k in keys)
    dst = os.path.join(WORK, name + "_asm")
    os.makedirs(dst, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(dst, "X.npy"), mode="w+", dtype=np.float32, shape=(n, len(cols)))
    out = {c: np.empty(n, dt) for c, dt in (("l", np.int32), ("r", np.int32), ("y", np.int8), ("w", np.float32))}
    b, pos = {}, 0
    for k in range(2 * config.N_FOLDS):         # sequential writes, one key group at a time
        start = pos
        for f, kf in zip(files, keys):
            sel = np.flatnonzero(kf == k)
            if not len(sel):
                continue
            t = pq.read_table(f, columns=cols + ["l", "r", "y", "w"]).take(pa.array(sel))
            X[pos:pos + len(sel)] = np.column_stack([t[c].to_numpy().astype(np.float32) for c in cols])
            for c in out:
                out[c][pos:pos + len(sel)] = t[c].to_numpy()
            pos += len(sel)
        b[k] = (start, pos)
    X.flush()
    del X
    out["X"] = np.load(os.path.join(dst, "X.npy"), mmap_mode="r")
    out["b"] = b
    return out


def other_folds(b, j, n):
    """Row slices of every fold except fold j (rows are ordered by fold)."""
    lo, hi = b[j]
    return [(a, e) for a, e in ((0, lo), (hi, n)) if e > a]


def _dataset(X, y, w, parts, reference=None):
    """LightGBM dataset from row slices of a memmap without concatenating the matrix."""
    import lightgbm as lgb
    data = [X[a:e] for a, e in parts]
    lab = np.concatenate([y[a:e] for a, e in parts])
    wt = None if w is None else np.concatenate([w[a:e] for a, e in parts])
    return lgb.Dataset(data if len(data) > 1 else data[0], lab, weight=wt, free_raw_data=True,
                       reference=reference)


def fit_folds(name, S, cols, params, rounds, log_name):
    """Cross-fitted filter models over ALL labelled S1 (GroupKFold, config.N_FOLDS): model j
    trains on every fold except j with a fixed tree count (no early stopping on the
    held-out fold). Every labelled pair is scored by the model that did not see it and
    test pairs by the average (filter scores feed the matcher, so their distribution
    must be the same on training and test pairs)."""
    import lightgbm as lgb
    K, n = config.N_FOLDS, len(S["y"])
    bf = {j: (S["b"][2 * j][0], S["b"][2 * j + 1][1]) for j in range(K)}
    prm = params_for(params, cols)
    t = time.time()
    ms = [lgb.train(prm, _dataset(S["X"], S["y"], S["w"], other_folds(bf, j, n)), rounds) for j in range(K)]
    log(f"  {log_name}: {K} fold models x {rounds} trees on {n:,} sampled pairs; {time.time() - t:.0f}s")
    CrossModel(ms, cols, excluded=True).save(os.path.join(MODELS, f"{name}_fin"))
    return ms


def step_fit12():
    ctx = context("train")
    passes = active_passes()
    cols15 = stage15_cols(passes)
    E = ctx["E"]                     # random labelled S1 whose complete candidate lists were kept
    side = ctx["fold"]
    # ---------------- Stage 1.5
    S = assemble_sample("collect15", cols15, ctx)
    log(f"Stage 1.5 sample: {len(S['y']):,} rows ({int(S['y'].sum()):,} positives) from all labelled S1")
    s15 = CrossModel(fit_folds("stage15", S, cols15, STAGE15_PARAMS, STAGE15_ROUNDS, "Stage 1.5"), cols15, side)
    ev_idx = np.concatenate([np.arange(*S["b"][2 * j + 1]) for j in range(config.N_FOLDS)])
    el, er = S["l"][ev_idx].astype(np.int64), S["r"][ev_idx].astype(np.int64)   # complete lists of E
    Xe = np.asarray(S["X"][ev_idx])
    p0 = s15.predict_rows(Xe, el)                        # out-of-fold
    raw = oracle_by_country(el, er, ctx, E)
    log("  eval-list raw union:", raw)
    Ms, grid, M = (20, 30, 40, 60, 80, 120, 160), [], {}
    for m in Ms:
        mk = topk_mask(el, p0, m)
        rep = oracle_by_country(el[mk], er[mk], ctx, E)
        grid.append(dict(M=m, **{k: v["oracle"] for k, v in rep.items()}))
        log(f"  top-{m} by Stage 1.5:", rep)
    for c in raw:
        if c != "all":
            ok = [gr["M"] for gr in grid if raw[c]["oracle"] - gr[c] <= 0.0005]
            M[c] = min(ok) if ok else Ms[-1]
    log(f"  chosen M per country = {M} (<= 0.0005 oracle loss; unseen countries use the max)")
    # the eval S1's top-M under Stage 1.5, with cheap string features, for Stage 2 tuning
    c_of = ctx["c1"][el]
    top = np.zeros(len(el), bool)
    for code, cname in enumerate(ctx["lab1"]):
        sel = np.flatnonzero(c_of == code)
        if len(sel):
            top[sel[topk_mask(el[sel], p0[sel], country_limit(M, cname))]] = True
    We = pd.DataFrame(Xe[top], columns=cols15)
    We.insert(0, "l", el[top].astype(np.int32))
    We.insert(1, "r", er[top].astype(np.int32))
    Fe = candidates.cheap_frame(We, ctx["left"], ctx["right"], ctx["n2"])
    top_rep = oracle_by_country(el[top], er[top], ctx, E)
    log("  eval-list top-M:", top_rep)
    del S, Xe
    gc.collect()
    # ---------------- Stage 2 (inputs: cheap string features + pass scores + signatures; no p0)
    parts = os.path.join(WORK, "collect2")
    schema = pq.read_schema(os.path.join(parts, sorted(os.listdir(parts))[0])).names
    cols2 = [c for c in schema if c not in ("l", "r", "y", "w", "ev", "p0")]
    S = assemble_sample("collect2", cols2, ctx)
    log(f"Stage 2 sample: {len(S['y']):,} rows ({int(S['y'].sum()):,} positives)")
    s2 = CrossModel(fit_folds("stage2", S, cols2, STAGE2_PARAMS, STAGE2_ROUNDS, "Stage 2"), cols2, side)
    p1 = s2.predict_rows(Fe, We["l"].values)            # out-of-fold
    P = pd.DataFrame({"l": We["l"].values, "r": We["r"].values, "p1": p1})
    rows = []
    for budget in (8, 10, 12, 15, 20, 30, None):
        for tau in (None, 0.0005, 0.001, 0.003, 0.01):
            Q = P if tau is None else P[P["p1"] >= tau]
            if budget:
                Q = Q[topk_mask(Q["l"].values, Q["p1"].values, budget)]
            rep = oracle_by_country(Q["l"].values, Q["r"].values, ctx, E)
            rows.append(dict(budget=budget, tau=tau, **{k: v for k, v in rep["all"].items()},
                             US=rep.get("US", {}).get("oracle"), India=rep.get("India", {}).get("oracle")))
    ok = [x for x in rows if top_rep["all"]["oracle"] - x["oracle"] <= 0.0005]
    choice = min(ok, key=lambda x: x["cands"]) if ok else rows[-1]
    for x in rows:
        log("  prune", x)
    log("  chosen Stage-2 pruning (<= 0.0005 oracle loss vs top-M):", choice)
    jdump({"M": M, "budget": choice["budget"], "tau": choice["tau"], "raw_eval": raw, "stage15_grid": grid,
           "topM_eval": top_rep, "stage2_grid": rows, "choice": choice, "passes": list(passes)}, "stage12.json")
    del S
    gc.collect()
    # collected samples are kept (caches are never deleted automatically)


def remove_dir(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except OSError as e:                             # e.g. a file still memory-mapped on Windows
        log(f"  could not remove {path}: {e}")


# ------------------------------------------------------------------ final candidate pass
class Pruner:
    """Per super-chunk: Stage 1.5 top-M -> cheap features -> Stage 2 -> prune, for one or
    more model variants at once (dev + final on train, final only on test)."""

    def __init__(self, ctx, variants, cfg, keep_first=0):
        self.ctx, self.variants, self.cfg = ctx, variants, cfg
        self.writers = {v: PartWriter(os.path.join(WORK, f"{v}_cands"), keep_first) for v in variants}

    def __call__(self, W):
        ctx, cfg = self.ctx, self.cfg
        if not len(W):
            return
        l = W["l"].values
        country = ctx["lab1"][ctx["c1"][l[0]]]
        m = country_limit(cfg["M"], country)
        sel, p0s = {}, {}
        for v, (s15, _) in self.variants.items():
            p0s[v] = s15.predict_rows(W, l)
            sel[v] = topk_mask(l, p0s[v], m)
        any_sel = np.logical_or.reduce(list(sel.values()))
        Wu = W[any_sel]
        F = candidates.cheap_frame(Wu, ctx["left"], ctx["right"], ctx["n2"])
        F.insert(0, "l", Wu["l"].values)
        F.insert(1, "r", Wu["r"].values)
        for v, (_, s2) in self.variants.items():
            mv = sel[v][any_sel]
            Fv = F[mv].copy()
            Fv["p0"] = p0s[v][any_sel][mv]
            Fv["p1"] = s2.predict_rows(Fv, Fv["l"].values)
            if cfg["tau"] is not None:
                Fv = Fv[Fv["p1"].values >= cfg["tau"]]
            if cfg["budget"]:
                Fv = Fv[topk_mask(Fv["l"].values, Fv["p1"].values, cfg["budget"])]
            self.writers[v](Fv[keep_columns(Fv.columns)])


def keep_columns(cols):
    return ["l", "r", "p0", "p1"] + [c for c in cols if c.startswith("score_")] + ["n_passes"] + SIG_FEATS[:10]


def step_cands(split, resume=False):
    ctx = context(split)
    passes = active_passes()
    cfg = jload("stage12.json")
    fin = (CrossModel.load(os.path.join(MODELS, "stage15_fin"), ctx.get("fold")),
           CrossModel.load(os.path.join(MODELS, "stage2_fin"), ctx.get("fold")))
    variant = f"{split}_fin"                         # train: out-of-half filter scores for every S1
    rows = ctx["labelled"] if split == "train" else np.arange(ctx["left"].n)
    done = 0
    if resume:
        done = count_parts(os.path.join(WORK, f"{variant}_cands"))
        rows, covered = remaining_rows(rows, ctx["c1"], done)
        log(f"{split} candidates resume: keeping {done} finished chunks ({covered:,} S1), {len(rows):,} to go")
    pr = Pruner(ctx, {variant: fin}, cfg, keep_first=done)
    candidates.build(split, ctx["left"], ctx["right"], ctx["n2"], passes, s1_rows=rows, raw=True, log=log,
                     sigs=ctx["sigs"], sink=pr, superchunk=COLLECT_CHUNK)
    log(f"{split}: candidate parts " + ", ".join(f"{v}={w.k}" for v, w in pr.writers.items()))


def step_cands_train():
    step_cands("train")


def step_cands_train_resume():
    step_cands("train", resume=True)


def step_cands_test():
    step_cands("test")


def step_cands_test_resume():
    step_cands("test", resume=True)


def step_assemble(variant):
    """Concatenate candidate parts, order rows by group, add rival features on p1."""
    split = variant.split("_")[0]
    ctx = context(split)
    src = os.path.join(WORK, f"{variant}_cands")
    files = sorted(f for f in os.listdir(src) if f.endswith(".parquet"))
    cols = pq.read_schema(os.path.join(src, files[0])).names
    l = np.concatenate([pq.read_table(os.path.join(src, f), columns=["l"])["l"].to_numpy() for f in files])
    g = ctx["fold"][l] if split == "train" else np.full(len(l), G_TEST, np.int8)
    order = np.lexsort((l, g))
    dst = os.path.join(WORK, f"{variant}_asm")
    os.makedirs(dst, exist_ok=True)
    for c in cols:
        v = np.concatenate([pq.read_table(os.path.join(src, f), columns=[c])[c].to_numpy() for f in files])
        np.save(os.path.join(dst, c + ".npy"), v[order])
        del v
    gs = g[order]
    np.save(os.path.join(dst, "group.npy"), gs)
    L, R, P1 = (np.load(os.path.join(dst, c + ".npy")) for c in ("l", "r", "p1"))
    comp = competition(L.astype(np.int64), R.astype(np.int64), P1.astype(np.float64))
    for k, v in comp.items():
        np.save(os.path.join(dst, "c1_" + k + ".npy"), v)
    meta = {"n": int(len(L)), "cols": cols + ["c1_" + k for k in comp],
            "bounds": {str(int(x)): [int(np.searchsorted(gs, x)), int(np.searchsorted(gs, x, "right"))]
                       for x in np.unique(gs)}}
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=2)
    if split == "train":
        y = ctx["true_s1"][R] == L
        np.save(os.path.join(dst, "y.npy"), y.astype(np.int8))
        ent = ctx["labelled"]
        rep = oracle_by_country(L.astype(np.int64), R.astype(np.int64), ctx, ent)
        log(f"  {variant} final candidates, all {len(ent):,} labelled S1 (candidate-oracle F0.5):", rep)
        jdump(rep, "candidates_oracle.json")
    log(f"{variant}: assembled {len(L):,} candidate pairs (candidate parts kept in {src})")


def step_assemble_train():
    step_assemble("train_fin")


# ------------------------------------------------------------------ Stage 3 features
def idf_tables(split, right):
    out = []
    for c in ("name_core", "addr_core"):
        p = os.path.join(MODELS, f"idf_v{config.NORM_VERSION}_{split}_{c}.npz")
        if os.path.exists(p):
            out.append(IdfTable.load(p))
        else:
            t = IdfTable(right.iter_lists(c))
            t.save(p)
            out.append(t)
    return out


def step_feats(variant, chunk=400_000, resume=False):
    """Stage 3 features into an on-disk float32 matrix; progress is saved after every
    chunk so that resume=True continues after a crash instead of starting over."""
    split = variant.split("_")[0]
    ctx = context(split)
    left, right, n2 = ctx["left"], ctx["right"], ctx["n2"]
    asm = os.path.join(WORK, f"{variant}_asm")
    meta = json.load(open(os.path.join(asm, "meta.json")))
    extra = [c for c in meta["cols"] if c not in ("l", "r")]
    L = np.load(os.path.join(asm, "l.npy"))
    R = np.load(os.path.join(asm, "r.npy"))
    ex = {c: np.load(os.path.join(asm, c + ".npy"), mmap_mode="r") for c in extra}
    idf_name, idf_addr = idf_tables(split, right)
    X, start = None, 0
    prog = os.path.join(asm, "X_progress.json")
    if resume and os.path.exists(prog) and os.path.exists(os.path.join(asm, "X.npy")):
        start = json.load(open(prog))["done"]
        cols = json.load(open(os.path.join(asm, "X_cols.json")))
        X = np.lib.format.open_memmap(os.path.join(asm, "X.npy"), mode="r+")
        log(f"  {variant} features resume at {start:,}/{len(L):,}")
    t0 = time.time()
    for s in range(start, len(L), chunk):
        li, ri = L[s:s + chunk], R[s:s + chunk]
        A = candidates.gather(left, li, FULL_COLS)
        B = candidates.gather(right, ri, FULL_COLS)
        B["is_s3"] = (ri >= n2).astype(np.int8)
        F = pair_features(A, B, idf_name, idf_addr, level="full")
        for c in extra:
            F[c] = np.asarray(ex[c][s:s + chunk], np.float32)
        if X is None:
            cols = list(F.columns)
            X = np.lib.format.open_memmap(os.path.join(asm, "X.npy"), mode="w+", dtype=np.float32,
                                          shape=(len(L), len(cols)))
            json.dump(cols, open(os.path.join(asm, "X_cols.json"), "w"))
        X[s:s + len(li)] = F[cols].to_numpy(np.float32)
        X.flush()
        del A, B, F
        gc.collect()
        done = s + len(li)
        json.dump({"done": done, "n": int(len(L))}, open(prog, "w"))
        log(f"  {variant} features {done:,}/{len(L):,}  ({(time.time() - t0) / max(done - start, 1) * 1e6:.0f} us/pair)")
    if X is not None:
        X.flush()


def step_feats_train():
    step_feats("train_fin")


def step_feats_train_resume():
    step_feats("train_fin", resume=True)


def step_feats_test_resume():
    step_feats("test_fin", resume=True)


# ------------------------------------------------------------------ Stage 3 model + decision
EARLY_STOP_ROWS = 3_000_000       # early-stopping slice of the other half (RAM: 8 GB machine)


def _train_parts(X, y, parts, es, rounds, log_name, params=None):
    """Train on row slices `parts`; `es` = a slice of TRAINING-side rows held back for
    early stopping (never the evaluated fold), or None for a fixed tree count."""
    import lightgbm as lgb
    params = params or STAGE3_PARAMS
    params = dict(params)
    dtr = _dataset(X, y, None, parts)
    dtr.params = {"max_bin": params.get("max_bin", 255)}
    callbacks, valid_sets = [], []
    if es is not None:
        valid_sets = [_dataset(X, y, None, [es], reference=dtr)]
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
    t = time.time()
    m = lgb.train(params, dtr, rounds, valid_sets=valid_sets, callbacks=callbacks)
    log(f"  {log_name}: {m.current_iteration()} trees (best {m.best_iteration}) in {time.time() - t:.0f}s")
    return m


def _train_slice(X, y, a, b, va, rounds, log_name, params=None):
    return _train_parts(X, y, [(a, b)], va, rounds, log_name, params)


def _cv_parts(b, j, n):
    """Training slices for fold model j (all other folds) minus an inner early-stopping
    slice carved from the end of the training rows (whole S1 entities: rows are
    ordered by fold, then S1)."""
    parts = other_folds(b, j, n)
    total = sum(e - a for a, e in parts)
    es_n = min(EARLY_STOP_ROWS, max(total // 10, 1))
    a, e = parts[-1]
    es_n = min(es_n, (e - a) // 2)
    parts[-1] = (a, e - es_n)
    return parts, (e - es_n, e)


def cv_fit_predict(X, y, b, rounds, tag, params=None):
    """GroupKFold: model j trains on the other folds and scores fold j (out-of-fold)."""
    n, K = len(y), config.N_FOLDS
    p = np.empty(n, np.float32)
    ms = []
    for j in range(K):
        parts, es = _cv_parts(b, j, n)
        m = _train_parts(X, y, parts, es, rounds, f"{tag} fold model {j} (scores fold {j})", params)
        p[b[j][0]:b[j][1]] = _predict(m, X, *b[j])
        ms.append(m)
    return p, ms


def _predict(m, X, a, b, step=500_000, col_idx=None):
    out = np.empty(b - a, np.float32)
    for s in range(a, b, step):
        e = min(s + step, b)
        Xc = X[s:e] if col_idx is None else X[s:e][:, col_idx]
        out[s - a:e - a] = m.predict(Xc, num_iteration=m.best_iteration or None, num_threads=config.N_JOBS)
    return out


def _report_mask(name, l, y, mask, ctx, entities):
    """Macro F0.5 per country + singleton accuracy, Fm, pair precision / recall."""
    left, n_true, c1, lab = ctx["left"], ctx["n_true"], ctx["c1"], ctx["lab1"]
    n_pred = np.bincount(l[mask], minlength=left.n)
    tp = np.bincount(l[mask], weights=y[mask].astype(float), minlength=left.n)
    f = evaluate.f05_arrays(n_pred, tp, n_true)
    res = {}
    for code, cname in list(enumerate(lab)) + [(-1, "all")]:
        e = entities if code < 0 else entities[c1[entities] == code]
        if not len(e):
            continue
        s = n_true[e] == 0
        res[cname] = dict(f05=round(float(f[e].mean()), 5), n_s1=int(len(e)),
                          singleton_acc=round(float(f[e][s].mean()), 5) if s.any() else None,
                          f_matched=round(float(f[e][~s].mean()), 5),
                          pair_precision=round(float(tp[e].sum() / max(n_pred[e].sum(), 1)), 5),
                          pair_recall=round(float(tp[e].sum() / max(n_true[e].sum(), 1)), 5))
    log(f"  {name}:", res)
    return res


def _decide_oof(tag, L, R, p, y, ctx):
    """Tune the decision on the out-of-fold scores of ALL labelled S1 (every S1 is scored
    by the model that did not train on it). Per-country thresholds are used only if they
    beat global ones on the same scores."""
    n_true, c1, lab = ctx["n_true"], ctx["c1"], ctx["lab1"]
    ent = ctx["labelled"]
    assign_all = decide.assign_mask(R, p)
    best, gcfg = decide.tune_rows(L, p, y, n_true, ent, assign_all, log=log, name=f"{tag} global")
    ccfg, pc_mask = {}, np.zeros(len(p), bool)
    for code, cname in enumerate(lab):
        e = ent[c1[ent] == code]
        if len(e):
            sel = np.flatnonzero(c1[L] == code)
            _, ccfg[cname] = decide.tune_rows(L[sel], p[sel], y[sel], n_true, e, assign_all[sel],
                                              log=log, name=f"{tag} {cname}")
            pc_mask[sel] = decide.apply(L[sel], R[sel], p[sel], ccfg[cname], assign_all[sel])
    g_mask = decide.apply(L, R, p, gcfg, assign_all)
    rep_g = _report_mask(f"{tag} out-of-fold, global thresholds", L, y, g_mask, ctx, ent)
    rep_c = _report_mask(f"{tag} out-of-fold, per-country thresholds", L, y, pc_mask, ctx, ent)
    use_pc = rep_c["all"]["f05"] > rep_g["all"]["f05"]
    res = {"global_cfg": gcfg, "country_cfg": ccfg, "use_country_thresholds": bool(use_pc),
           "oof_global": rep_g, "oof_per_country": rep_c,
           "oof_f05": (rep_c if use_pc else rep_g)["all"]["f05"],
           "oof_report": rep_c if use_pc else rep_g}
    log(f"  {tag}: per-country thresholds {'adopted' if use_pc else 'not adopted'}; "
        f"out-of-fold macro F0.5 = {res['oof_f05']:.5f}")
    return res, (pc_mask if use_pc else g_mask)


def _error_analysis(L, y, mask, X, cols, tag):
    """Share of false positives / false negatives (inside the candidates) by category."""
    fp = np.flatnonzero(mask & (y == 0))
    fn = np.flatnonzero(~mask & (y == 1))
    ci = {c: i for i, c in enumerate(cols)}

    def cats(ix):
        if not len(ix):
            return {}
        ix = np.sort(ix)[:: max(1, len(ix) // 200_000)]
        Z = X[ix]
        return {"right_addr_empty": float(Z[:, ci["addr_empty_b"]].mean()),
                "right_web": float(Z[:, ci["is_web_b"]].mean()),
                "right_native": float(Z[:, ci["name_native_b"]].mean()),
                "generic_name (name_idf_min<8)": float((Z[:, ci["name_idf_min"]] < 8).mean()),
                "same_core_name": float(Z[:, ci["same_core"]].mean()),
                "house_conflict": float(Z[:, ci["house_conflict"]].mean())}
    log(f"  {tag} errors: {len(fp):,} false positives, {len(fn):,} false negatives (inside candidates)")
    log("   FP categories:", {k: round(v, 3) for k, v in cats(fp).items()})
    log("   FN categories:", {k: round(v, 3) for k, v in cats(fn).items()})


SUB_FRACTION = float(os.environ.get("ER_SUB_FRACTION", "0.30"))


def step_subsample_asm():
    """A random SUB_FRACTION of S1 entities (whole entities with all their candidates)
    from the full-data candidate table: candidates, pool and rival features stay those
    of the complete data; only the matcher evaluation runs on fewer S1."""
    ctx = context("train")
    src, dst = os.path.join(WORK, "train_fin_asm"), os.path.join(WORK, "train_sub_asm")
    meta = json.load(open(os.path.join(src, "meta.json")))
    pick = np.random.default_rng(config.SEED + 7).random(ctx["left"].n) < SUB_FRACTION
    L = np.load(os.path.join(src, "l.npy"))
    sel = np.flatnonzero(pick[L])
    os.makedirs(dst, exist_ok=True)
    for c in meta["cols"] + ["group", "y"]:
        np.save(os.path.join(dst, c + ".npy"), np.load(os.path.join(src, c + ".npy"), mmap_mode="r")[sel])
    gs = np.load(os.path.join(dst, "group.npy"))
    meta = dict(meta, n=int(len(sel)), bounds={str(int(x)): [int(np.searchsorted(gs, x)), int(np.searchsorted(gs, x, "right"))]
                                             for x in np.unique(gs)})
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=2)
    ent = np.flatnonzero(pick)
    np.save(os.path.join(dst, "entities.npy"), ent)
    Ls, Rs = np.load(os.path.join(dst, "l.npy")).astype(np.int64), np.load(os.path.join(dst, "r.npy")).astype(np.int64)
    log(f"  sample: {len(ent):,} S1 ({SUB_FRACTION:.0%}), {len(sel):,} candidate pairs; candidate-oracle F0.5:",
        oracle_by_country(Ls, Rs, ctx, ent))


def step_feats_sub():
    step_feats("train_sub")


def step_stage3_sub():
    step_stage3("train_sub")


def step_stage3(variant="train_fin"):
    """Evaluation by 2-fold cross-validation over the labelled S1 (no reserved split):
    each fold model trains on the other fold and scores its own fold. The out-of-fold
    macro F0.5 is the reported score; thresholds are tuned on it."""
    ctx = context("train")
    asm = os.path.join(WORK, f"{variant}_asm")
    if os.path.exists(os.path.join(asm, "entities.npy")):         # S1 sample: score exactly those S1
        ctx["labelled"] = np.load(os.path.join(asm, "entities.npy"))
    b = {int(k): v for k, v in json.load(open(os.path.join(asm, "meta.json")))["bounds"].items()}
    X = np.load(os.path.join(asm, "X.npy"), mmap_mode="r")
    cols = json.load(open(os.path.join(asm, "X_cols.json")))
    y = np.load(os.path.join(asm, "y.npy"))
    L = np.load(os.path.join(asm, "l.npy")).astype(np.int64)
    R = np.load(os.path.join(asm, "r.npy")).astype(np.int64)
    log(f"Stage 3 cross-validation (GroupKFold by S1, {config.N_FOLDS} folds): {len(y):,} pairs, "
        f"{len(cols)} features; fold rows {b}")
    p, ms = cv_fit_predict(X, y, b, 4000, "Stage 3")
    np.save(os.path.join(asm, "p_oof.npy"), p)
    for k, m in enumerate(ms):
        m.save_model(os.path.join(MODELS, f"stage3_cv_m{k}.txt"), num_iteration=m.best_iteration)
    imp = pd.Series(ms[0].feature_importance("gain"), index=cols).sort_values(ascending=False)
    log("  top-25 features (gain):", imp.head(25).round(0).to_dict())
    res, mask = _decide_oof("STAGE 3", L, R, p, y, ctx)
    res.update(best_iters=[int(m.best_iteration) for m in ms], features=cols)
    _error_analysis(L, y, mask, X, cols, "STAGE 3 out-of-fold")
    jdump(res, "stage3_eval.json")


# ------------------------------------------------------------------ stacked re-scoring
STACK_PARAMS = dict(STAGE3_PARAMS, num_leaves=63, min_data_in_leaf=200, max_bin=255)


def transitivity_features(L, R, p, right, chunk=1_000_000):
    """S2 <-> S3 transitivity: compare each candidate with the best OTHER candidate of the
    same S1 (by Stage 3 score). True matches of one S1 are noisy copies of one business,
    so they resemble each other; a wrong candidate rarely resembles the S1's best match."""
    from rapidfuzz import fuzz, process
    order = np.lexsort((-p, L))
    Ls = L[order]
    start = np.r_[0, np.flatnonzero(np.diff(Ls)) + 1]
    size = np.diff(np.r_[start, len(L)])
    first = np.repeat(start, size)
    rank = np.arange(len(L)) - first
    top1 = order[first]                                           # best row of each pair's S1
    top2 = np.where(np.repeat(size, size) > 1, order[np.minimum(first + 1, len(L) - 1)], -1)
    ref_sorted = np.where(rank == 0, top2, top1)
    ref = np.empty(len(L), np.int64)
    ref[order] = ref_sorted
    has = ref >= 0
    F = {"tr_has_ref": has.astype(np.int8), "tr_p_ref": np.where(has, p[np.maximum(ref, 0)], np.nan).astype(np.float32)}
    cols = ("name_core", "name_join", "addr_core", "house_no")
    out = {k: np.full(len(L), np.nan, np.float32) for k in
           ("tr_name_tset", "tr_join_ratio", "tr_addr_tset", "tr_house_match", "tr_house_conflict", "tr_same_name")}
    idx = np.flatnonzero(has)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        A = candidates.gather(right, R[ii], cols)
        B = candidates.gather(right, R[ref[ii]], cols)
        out["tr_name_tset"][ii] = process.cpdist(A["name_core"], B["name_core"], scorer=fuzz.token_set_ratio,
                                                 workers=-1, dtype=np.float32)
        out["tr_join_ratio"][ii] = process.cpdist(A["name_join"], B["name_join"], scorer=fuzz.ratio,
                                                  workers=-1, dtype=np.float32)
        out["tr_addr_tset"][ii] = process.cpdist(A["addr_core"], B["addr_core"], scorer=fuzz.token_set_ratio,
                                                 workers=-1, dtype=np.float32)
        ha, hb = np.asarray(A["house_no"], object), np.asarray(B["house_no"], object)
        both = (ha != "") & (hb != "")
        out["tr_house_match"][ii] = np.where(both, (ha == hb).astype(np.float32), np.nan)
        out["tr_house_conflict"][ii] = np.where(both, (ha != hb).astype(np.float32), np.nan)
        out["tr_same_name"][ii] = (np.asarray(A["name_core"], object) == np.asarray(B["name_core"], object))
    F.update(out)
    return F


def _stack_matrix(asm, tag, p, L, R, right=None):
    """Stacker inputs: the Stage 3 score, its rival features within each S1 and each S2/S3
    record, transitivity features (similarity to the S1's best other candidate), and the
    stored per-pair columns (Stage 1.5 / 2 scores, pass scores, signature agreement,
    Stage 2 rival features)."""
    meta = json.load(open(os.path.join(asm, "meta.json")))
    base_cols = [c for c in meta["cols"] if c not in ("l", "r")]
    comp = competition(L, R, p.astype(np.float64))
    if right is not None:
        for k, v in transitivity_features(L, R, p.astype(np.float64), right).items():
            comp["T_" + k] = v
    zcols = ["p3", "p3_logit"] + ["c3_" + k for k in comp] + base_cols
    path = os.path.join(asm, f"Z_{tag}.npy")
    Z = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(p), len(zcols)))
    pc = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    Z[:, 0] = p
    Z[:, 1] = np.log(pc / (1 - pc))
    for j, k in enumerate(comp):
        Z[:, 2 + j] = comp[k]
    for j, c in enumerate(base_cols):
        Z[:, 2 + len(comp) + j] = np.load(os.path.join(asm, c + ".npy"), mmap_mode="r")
    Z.flush()
    del Z
    return np.load(path, mmap_mode="r"), zcols


def step_stack():
    """Stacked re-scoring, evaluated the same way (2-fold CV over all labelled S1 on the
    out-of-fold Stage 3 scores); adopted only if it raises the out-of-fold F0.5."""
    ctx = context("train")
    asm = os.path.join(WORK, "train_fin_asm")
    b = {int(k): v for k, v in json.load(open(os.path.join(asm, "meta.json")))["bounds"].items()}
    p = np.load(os.path.join(asm, "p_oof.npy"))
    y = np.load(os.path.join(asm, "y.npy"))
    L = np.load(os.path.join(asm, "l.npy")).astype(np.int64)
    R = np.load(os.path.join(asm, "r.npy")).astype(np.int64)
    Z, zcols = _stack_matrix(asm, "oof", p, L, R, ctx["right"])
    log(f"Stacker cross-validation ({config.N_FOLDS} folds): {len(y):,} pairs, {len(zcols)} inputs")
    q, ms = cv_fit_predict(Z, y, b, 3000, "stacker", STACK_PARAMS)
    np.save(os.path.join(asm, "q_oof.npy"), q)
    base = jload("stage3_eval.json")
    res, _ = _decide_oof("STACKED", L, R, q, y, ctx)
    use = res["oof_f05"] > base["oof_f05"] + 0.0001
    log(f"  out-of-fold macro F0.5: Stage 3 {base['oof_f05']:.5f} vs stacked {res['oof_f05']:.5f} "
        f"-> use stacker: {use}")
    res.update(use_stack=bool(use), best_iters=[int(m.best_iteration) for m in ms], inputs=zcols)
    jdump(res, "stack_eval.json")


def step_final():
    """PRODUCTION models, trained ONCE on 100% of the labelled pairs (no split); tree
    counts from cross-validation x1.15 for the doubled data. Test uses only these."""
    asm = os.path.join(WORK, "train_fin_asm")
    X = np.load(os.path.join(asm, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(asm, "y.npy"))
    ev = jload("stage3_eval.json")
    rounds = int(np.mean(ev["best_iters"]) * 1.15) + 1
    try:
        m = _train_slice(X, y, 0, len(y), None, rounds, f"PRODUCTION Stage 3 on all {len(y):,} labelled pairs")
        ev["production_features"] = "all"
    except (MemoryError, Exception) as e:          # LightGBM raises LightGBMError on allocation failure
        # every pair is still used; only the least useful features (CV gain) are dropped
        import lightgbm as lgb
        log(f"  production model on all features failed ({type(e).__name__}: {e}); retrying with top features")
        cv = lgb.Booster(model_file=os.path.join(MODELS, "stage3_cv_m0.txt"))
        # importances are in column order (models trained from a matrix name them Column_N)
        gain = pd.Series(cv.feature_importance("gain"), index=ev["features"])
        keep = [c for c in gain.sort_values(ascending=False).index[:60]]
        idx = sorted(ev["features"].index(c) for c in keep)
        Xs = np.lib.format.open_memmap(os.path.join(asm, "X_top.npy"), mode="w+", dtype=np.float32,
                                       shape=(len(y), len(idx)))
        for s in range(0, len(y), 1_000_000):
            Xs[s:s + 1_000_000] = X[s:s + 1_000_000][:, idx]
        Xs.flush()
        m = _train_slice(Xs, y, 0, len(y), None, rounds, f"PRODUCTION Stage 3 (top {len(idx)} features)")
        ev["production_features"] = [ev["features"][i] for i in idx]
    jdump(ev, "stage3_eval.json")
    m.save_model(os.path.join(MODELS, "stage3_full.txt"))
    sp = os.path.join(MODELS, "stack_eval.json")
    stale = os.path.join(MODELS, "stack_full.txt")
    if os.path.exists(stale):
        os.remove(stale)
    if os.path.exists(sp) and jload("stack_eval.json")["use_stack"]:
        sv = jload("stack_eval.json")
        Z = np.load(os.path.join(asm, "Z_oof.npy"), mmap_mode="r")
        rs = int(np.mean(sv["best_iters"]) * 1.15) + 1
        ms = _train_slice(Z, y, 0, len(y), None, rs, "PRODUCTION stacker on all labelled pairs", STACK_PARAMS)
        ms.save_model(os.path.join(MODELS, "stack_full.txt"))
        del Z
        gc.collect()
    log("production models written: stage3_full.txt" + (" + stack_full.txt" if os.path.exists(
        os.path.join(MODELS, "stack_full.txt")) else ""))


def step_test():
    """Stage 1-2 candidates, assembly and features on test, then scoring + decision."""
    log("==== sub-step cands_test")
    step_cands_test()
    step_test_rest()


def step_test_resume():
    """After a crash inside the test candidate pass: finish it, then the rest."""
    log("==== sub-step cands_test_resume")
    step_cands_test_resume()
    step_test_rest()


def step_test_rest():
    log("==== sub-step assemble / features")
    step_assemble("test_fin")
    step_feats("test_fin")
    step_test_predict()


def step_test_predict():
    """Score the cached test candidates with the production models and write both TSVs."""
    import lightgbm as lgb
    ctx = context("test")
    left, right = ctx["left"], ctx["right"]
    asm = os.path.join(WORK, "test_fin_asm")
    X = np.load(os.path.join(asm, "X.npy"), mmap_mode="r")
    L = np.load(os.path.join(asm, "l.npy")).astype(np.int64)
    R = np.load(os.path.join(asm, "r.npy")).astype(np.int64)
    ev = jload("stage3_eval.json")
    ref = np.load(os.path.join(WORK, "train_fin_asm", "p_oof.npy"))
    sp = os.path.join(MODELS, "stack_eval.json")
    if os.path.exists(sp) and jload("stack_eval.json")["use_stack"]:
        # the stacker was trained on out-of-fold Stage 3 scores, so its test input is the
        # average of the two CV Stage 3 models (same distribution), then the production stacker
        cv = [lgb.Booster(model_file=os.path.join(MODELS, f"stage3_cv_m{k}.txt"))
              for k in range(config.N_FOLDS)]
        p3 = np.mean([_predict(m, X, 0, len(L)) for m in cv], axis=0)
        ev = jload("stack_eval.json")
        Z, zcols = _stack_matrix(asm, "test", p3, L, R, right)
        assert zcols == ev["inputs"], "stacker inputs differ from training"
        p = _predict(lgb.Booster(model_file=os.path.join(MODELS, "stack_full.txt")), Z, 0, len(L))
        ref = np.load(os.path.join(WORK, "train_fin_asm", "q_oof.npy"))
        log("  test scores: production stacker")
    else:
        pf = ev.get("production_features", "all")
        tcols = json.load(open(os.path.join(asm, "X_cols.json")))
        idx = None if pf == "all" else [tcols.index(c) for c in pf]
        p = _predict(lgb.Booster(model_file=os.path.join(MODELS, "stage3_full.txt")), X, 0, len(L), col_idx=idx)
        log(f"  test scores: production Stage 3 ({'all' if idx is None else len(idx)} features)")
    np.save(os.path.join(asm, "p.npy"), p)
    assign_all = decide.assign_mask(R, p)
    c1, lab = ctx["c1"], ctx["lab1"]
    if ev["use_country_thresholds"]:
        mask = np.zeros(len(p), bool)
        for code, cname in enumerate(lab):
            cfg = ev["country_cfg"].get(cname, ev["global_cfg"])      # unseen countries: global
            sel = np.flatnonzero(c1[L] == code)
            mask[sel] = decide.apply(L[sel], R[sel], p[sel], cfg, assign_all[sel])
    else:
        mask = decide.apply(L, R, p, ev["global_cfg"], assign_all)
    log("drift out-of-fold (train) vs test (share of pairs per probability bucket):\n"
        + evaluate.drift_report(ref, p).to_string())
    for code, cname in enumerate(lab):
        log(f"  test {cname}: pairs {int((c1[L] == code).sum()):,}, bucket shares",
            evaluate.drift_report(ref, p[c1[L] == code])["test"].tolist())
    s1_ids = left.list("entity_id")
    rid = right.t["entity_id"]
    os.makedirs(config.OUT, exist_ok=True)
    for name, sel, col in (("candidate_pairs.tsv", np.ones(len(p), bool), "candidate_entity_ids"),
                           ("matching_results.tsv", mask, "matched_entity_ids")):
        idx = np.flatnonzero(sel)
        idx = idx[np.lexsort((-p[idx], L[idx]))]
        r_ids = rid.take(pa.array(R[idx])).to_pylist()
        m = {}
        for a, b_ in zip(L[idx], r_ids):
            m.setdefault(s1_ids[a], []).append(b_)
        write_tsv(s1_ids, m, os.path.join(config.OUT, name), col)
        log(f"  wrote {name}: {len(idx):,} pairs, {len(m):,} of {left.n:,} S1 non-empty")


def step_prepare():
    import prepare
    import translit_dict
    from io_utils import tsv_to_parquet
    os.makedirs(config.RAW, exist_ok=True)
    for split in ("train", "test"):
        for f in sorted(os.listdir(os.path.join(config.DATA, split))):
            dst = config.RAW + f.replace(".tsv", ".parquet")
            if f.endswith(".tsv") and not os.path.exists(dst):
                tsv_to_parquet(os.path.join(config.DATA, split, f), dst)
    if not os.path.exists(config.TDICT):
        translit_dict.learn(config.RAW, config.TDICT)
    prepare.main()


def step_test_cands():
    step_cands_test()


def step_test_assemble():
    step_assemble("test_fin")


def step_test_feats():
    step_feats("test_fin")


ORDER = ["prepare", "prov", "collect", "fit12", "cands_train", "assemble_train", "feats_train", "stage3",
         "stack", "final", "test"]
# `auto`: every step leaves a marker when it finishes; a re-run skips finished steps and
# resumes the interrupted one from its last saved chunk where the step supports it.
EVAL_STEPS = ["prov", "collect", "fit12", "cands_train", "assemble_train", "feats_train", "stage3", "stack"]
FINAL_ONLY = ["final", "test", "test_cands", "test_assemble", "test_feats", "test_predict", "test_resume",
              "test_rest", "cands_test", "cands_test_resume", "feats_test_resume"]
# DEV / VALIDATION stop after the out-of-fold evaluation; only FINAL trains the production
# models and touches the test set
AUTO_ORDER = (["prepare"] + EVAL_STEPS + ["final", "test_cands", "test_assemble", "test_feats", "test_predict"]
              if config.MODE == "FINAL" else EVAL_STEPS)
RESUMABLE = {"collect": "collect_resume", "cands_train": "cands_train_resume",
             "feats_train": "feats_train_resume", "test_cands": "cands_test_resume",
             "test_feats": "feats_test_resume"}
STEP_DIR = os.path.join(MODELS, "steps_done")


def run_step(s):
    if s in FINAL_ONLY and config.MODE != "FINAL":
        raise RuntimeError(f"step {s!r} is FINAL-only (production models / test set); current MODE={config.MODE}")
    log(f"==== step {s}  (disk free {shutil.disk_usage(config.CACHE).free / 1e9:.1f} GB)")
    globals()[f"step_{s}"]()


def log_startup():
    """Which data this run sees: mode, fraction, folds and entity counts."""
    left, right = load_split("train")
    _, n_true = truth_arrays(left, right)
    log(f"MODE={config.MODE}  fraction={config.FRACTION}  folds={config.N_FOLDS}  "
        f"subset_seed={config.SUBSET_SEED}  cache={config.RUN}")
    log(f"TRAIN: S1 entities={left.n:,}  S2 entities={right.sizes[0]:,}  S3 entities={right.sizes[1]:,}  "
        f"true pairs={int(n_true.sum()):,}  singletons={int((n_true == 0).sum()):,}")
    if config.MODE == "FINAL":
        tl, tr = load_split("test")
        log(f"TEST: S1 entities={tl.n:,}  S2 entities={tr.sizes[0]:,}  S3 entities={tr.sizes[1]:,}")
    else:
        log("TEST: not used in this mode (only FINAL touches test_source1/2/3)")


def auto():
    os.makedirs(STEP_DIR, exist_ok=True)
    for s in AUTO_ORDER:
        flag = os.path.join(STEP_DIR, s + ".done")
        if os.path.exists(flag):
            continue
        started = os.path.join(STEP_DIR, s + ".started")
        interrupted = os.path.exists(started)        # it began before and never finished
        open(started, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        run_step(RESUMABLE[s] if (interrupted and s in RESUMABLE) else s)
        open(flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        os.remove(started)
    log("auto: all steps done")


def main(steps):
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    if steps != ["prepare"]:
        log_startup()
    if steps == ["auto"]:
        return auto()
    for s in (AUTO_ORDER if steps == ["all"] else steps):
        run_step(s)


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
