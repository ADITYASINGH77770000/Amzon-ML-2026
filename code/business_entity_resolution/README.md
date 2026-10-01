# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: **normalise → multi-pass blocking (forward + reverse) → Stage 1.5 pre-ranker
→ Stage 2 fast ranker → Stage 3 matcher → decision layer**. Every model is cross-fitted with
GroupKFold by Source 1 entity for evaluation only; the production models are trained on all
labelled data (no holdout).

Only the provided files are used (no external data, APIs, geocoding or LLMs). All
models are LightGBM (MIT licence).

## Environment

- Python 3.13 (tested on 3.13.2, Windows 10); `pip install -r requirements.txt`
- Built to run on a small laptop (2-core i3, 8 GB RAM): every stage streams in chunks,
  normalised tables and blocking indexes are memory-mapped from disk.
- Disk: ~25 GB free for the cache (normalised tables, indexes, feature matrices).

The code expects the challenge folder layout (`dataset/train`, `dataset/test`) three
levels above `src/` (i.e. this folder sits at `<root>/code/business_entity_resolution`).
Set `ER_ROOT` to point elsewhere. Results are cached under `<root>/cache`; every step logs to
`cache/pipeline_log.txt`.

## Reproduce the submitted files

The submitted `output/matching_results.tsv` (public leaderboard 0.920) and
`output/candidate_pairs.tsv` were produced by the key-candidate matcher (`src/key_matcher.py`);
the full pipeline's test-side pass did not fit in the time available.

```
cd src
python run_pipeline.py prepare      # TSV -> parquet, transliteration map, normalised Arrow tables
python key_matcher.py train         # 2-fold GroupKFold by S1 over all labelled S1: fold models,
                                    # out-of-fold macro F0.5 (0.934) and decision thresholds
python key_matcher.py test          # writes <root>/output/matching_results.tsv + candidate_pairs.tsv
```

Test pairs are scored with the average of the two fold models (together they were trained on all
labelled data), exactly as for the submitted files. Run times on a 2-core i3 / 8 GB laptop:
`prepare` ~1 h, `train` ~45 min, `test` ~30 min. The feature matrices are written as memory-mapped
files under `<root>/cache/key_matcher/` (~3.5 GB train, ~3.1 GB test).

- Candidates: 13 exact within-country keys on the normalised fields (name + street / house / city /
  state, alternate name, street + city, full address, 6-character name prefix / suffix + house /
  city / postal); a key links a right record to every S1 carrying it if at most 3–5 do.
- 43 features: key flags and ambiguity, rapidfuzz name / address similarities and gaps to the best
  rival, house / city / state / postal agreement, source flags.
- LightGBM (2-fold GroupKFold by S1), then the one-owner t_open / t_add decision tuned for macro F0.5.

`fast_fallback.py` (unique exact keys, 0.762 on the leaderboard) and `key_stage2.py` (rival-feature
second stage, 0.929 OOF, worse than stage 1 and not used) are kept for reference.

## Full pipeline

`ER_MODE` selects the data a run sees; all modes share one code path, and only `FINAL` trains
the production models or touches the test set.

| `ER_MODE` | Data | Use |
|---|---|---|
| `DEV_FAST` (default) | 5 % of S1 per country + their matches + 5 % of the pool | quick experiments |
| `DEV_MEDIUM` | 20 % | confirmation |
| `VALIDATION` | 20 % (`ER_VALIDATION_FRACTION`), 5-fold GroupKFold | out-of-fold estimate |
| `FINAL` | 100 % | production models + test submission |

```
cd src
set ER_MODE=FINAL
python run_pipeline.py auto         # resumable: finished steps are skipped, interrupted ones resume
```

| Step | What it does |
|---|---|
| `prepare` | TSV → parquet; learn native-script → Latin token map from train pairs; normalise every record |
| `prov` | small provisional Stage 1.5, used only to choose which negatives the collect pass samples |
| `collect` | blocking over every labelled S1: all positive pairs + weighted negatives |
| `fit12` | cross-fitted Stage 1.5 and Stage 2; choose M / budget / tau with the ≤ 0.0005 oracle-loss guard |
| `cands_train` | Stage 1 + 1.5 + 2 over every labelled S1 (out-of-fold filter scores) |
| `assemble_train`, `feats_train` | order candidates, rival features, full Stage 3 feature matrix on disk |
| `stage3` | cross-fitted Stage 3 matcher, decision thresholds tuned on out-of-fold predictions |
| `stack` | stacker evaluation (kept for reference; no gain) |
| `final` | Stage 3 retrained on all labelled data with the frozen settings |
| `test_cands`, `test_assemble`, `test_feats`, `test_predict` | the same stages on test → `output/*.tsv` |

Then validate (from the `student_resource` folder):

```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Development tools (not needed to reproduce)

- `recall_lab.py` — per-pass recall / candidate-oracle F0.5 on a dev sample;
  chooses the passes and k written to `cache/models/passes.json`.
- `quick_f05.py`, `loss_analysis.py`, `decision_lab.py`, `matcher_lab.py`, `prerank_speed.py`,
  `prefilter_lab.py` — evaluation and ablation scripts.
- `make_smoke.py` — miniature dataset (with an unseen country) for an end-to-end smoke run:
  `python make_smoke.py <dir>` then `ER_ROOT=<dir> python run_pipeline.py all`.
- `eda_report.py`, `miss_analysis.py`, `profile_blocking.py`, `prerank_misses.py` — analysis scripts.

## Source layout

| File | Role |
|---|---|
| `config.py` | modes, paths, seeds, versioned cache directories |
| `io_utils.py` | TSV ↔ parquet, output writer (one row per S1, always) |
| `normalize.py` | name / address normalisers, Indic transliteration, state codes |
| `translit_dict.py` | learns native-script → Latin tokens from the training pairs |
| `prepare.py` | parallel normalisation into the cache |
| `data.py` | memory-mapped tables, ground-truth arrays, mode subsets |
| `split.py` | GroupKFold folds by S1 (stratified by country and match count) |
| `blocking.py`, `blocking_keys.py` | pass definitions, key builders, sparse-join helpers |
| `candidates.py` | streaming Stage 1 / 1.5 / 2 candidate builder |
| `signatures.py` | MinHash record signatures for the pre-ranker |
| `monotone.py` | monotone constraints for the filter models |
| `features.py` | cheap and full pair features |
| `crossfit.py` | cross-fitted models, rival features |
| `decide.py` | assignment, thresholds, coarse-to-fine tuning |
| `expected_f.py` | expected-F0.5 decision (evaluated; not used) |
| `evaluate.py` | macro F0.5, oracle metrics |
| `key_matcher.py` | key-candidate matcher that produced the submitted files |
| `fast_fallback.py`, `key_stage2.py` | earlier fallback and an unused second stage (reference) |
| `run_pipeline.py` | the steps above |
