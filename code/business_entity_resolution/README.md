# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: **normalise → multi-pass blocking (forward + reverse) → Stage 1.5 pre-ranker
→ Stage 2 fast ranker → Stage 3 matcher → decision layer**, all cross-fitted and
validated on a stratified 80 / 10 / 10 split of the labelled Source 1 entities.

Only the provided files are used (no external data, APIs, geocoding or LLMs). All
models are LightGBM (MIT licence).

## Environment

- Python 3.13 (tested on 3.13.2, Windows 10); `pip install -r requirements.txt`
- Built to run on a small laptop (2-core i3, 8 GB RAM): every stage streams in chunks,
  normalised tables and blocking indexes are memory-mapped from disk.
- Disk: ~25 GB free for the cache (normalised tables, indexes, feature matrices).

## Reproduce end to end

The code expects the challenge folder layout (`dataset/train`, `dataset/test`) three
levels above `src/` (i.e. this folder sits at `<root>/code/business_entity_resolution`).
Set `ER_ROOT` to point elsewhere.

```
cd src
python run_pipeline.py all
```

`all` runs these steps in order (each can be run on its own; results are cached under
`<root>/cache`):

| Step | What it does | Output |
|---|---|---|
| `prepare` | TSV → parquet; learn native-script → Latin token map from train pairs; normalise every record; Arrow IPC copies for memory-mapping | `cache/raw`, `cache/norm`, `cache/translit_dict.json` |
| `stage12` | raw blocking on a train sample; fit Stage 1.5 (pass scores only) and Stage 2 (cheap string features), choose M / budget / tau with the oracle guard | `cache/models/stage15*`, `stage2*`, `stage12.json` |
| `cands_train` | Stage 1 + 1.5 + 2 over every train S1 (out-of-fold scores for train rows) | `cache/work/train_cands` |
| `assemble_train` | order rows by partition, rival features on p1 | `cache/work/train_asm` |
| `feats_train` | full Stage 3 pair features into an on-disk float32 matrix | `cache/work/train_asm/X.npy` |
| `stage3` | cross-fitted LightGBM, threshold tuning on validation, holdout scored once | `cache/models/stage3_eval.json` |
| `refit` | Stage 3 on 100 % of labelled S1 with the frozen settings | `cache/models/stage3_final_m*.txt` |
| `test` | Stage 1–4 on test with the final artefacts | `output/candidate_pairs.tsv`, `output/matching_results.tsv` |

Then validate (from the `student_resource` folder):

```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Every step logs to `cache/pipeline_log.txt`.

## Development tools (not needed to reproduce)

- `recall_lab.py` — per-pass recall / candidate-oracle F0.5 on a validation dev sample;
  chooses the passes and k written to `cache/models/passes.json`.
- `make_smoke.py` — miniature dataset (with an unseen country) for an end-to-end smoke run:
  `python make_smoke.py <dir>` then `ER_ROOT=<dir> python run_pipeline.py all`.
- `eda_report.py`, `miss_analysis.py`, `profile_blocking.py` — analysis scripts.

## Source layout

| File | Role |
|---|---|
| `config.py` | paths, seeds, versioned cache directories |
| `io_utils.py` | TSV ↔ parquet, output writer (one row per S1, always) |
| `normalize.py` | name / address normalisers, Indic transliteration, state codes |
| `translit_dict.py` | learns native-script → Latin tokens from the training pairs |
| `prepare.py` | parallel normalisation into the cache |
| `data.py` | memory-mapped tables, ground-truth arrays |
| `split.py` | stratified train / validation / holdout split |
| `blocking.py`, `blocking_keys.py` | pass definitions, key builders, sparse-join helpers |
| `candidates.py` | streaming Stage 1 / 1.5 / 2 candidate builder |
| `features.py` | cheap and full pair features |
| `crossfit.py` | two-fold cross-fitted models, rival features |
| `decide.py` | assignment, thresholds, coarse-to-fine tuning |
| `evaluate.py` | macro F0.5, oracle metrics, drift report |
| `run_pipeline.py` | the steps above |
