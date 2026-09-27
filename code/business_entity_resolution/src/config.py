"""Paths, seeds, execution mode and data fractions — every run-level choice in one place.

Execution modes (environment variable ER_MODE; the SAME code path runs in every mode,
only the entity / pool subset and the cache folder change):

  DEV_FAST    ~5% of training S1 entities   - rapid iterations (normalisation, blocking,
                                               features, models)
  DEV_MEDIUM  ~20% of training S1 entities  - serious experiments, model comparisons
  VALIDATION  GroupKFold by S1 over the chosen fraction (ER_VALIDATION_FRACTION,
              default = DEV_MEDIUM), complete pipeline on every fold, exact macro F0.5
  FINAL       100% of the labelled training data for the final models, then 100% of
              test_source1/2/3 for candidates, matching and both output files

In the subset modes every sampled S1 keeps ALL of its true S2/S3 matches, and the rest
of the S2/S3 pool (records that match no sampled S1) is sampled at the same fraction.
The hidden test data is used only in FINAL, and never for training or tuning.
"""
import os

ROOT = os.environ.get("ER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA = os.path.join(ROOT, "dataset")
CACHE = os.path.join(ROOT, "cache")
RAW = os.path.join(CACHE, "raw") + os.sep
NORM = os.path.join(CACHE, "norm") + os.sep
TDICT = os.path.join(CACHE, "translit_dict.json")
# bump when normalisation changes: derived caches live under versioned directories
NORM_VERSION = 2
SEED = 42
N_JOBS = int(os.environ.get("ER_JOBS", "4"))

# ---------------------------------------------------------------- execution mode
DEV_FAST = 0.05
DEV_MEDIUM = 0.20
FINAL = 1.0
MODES = ("DEV_FAST", "DEV_MEDIUM", "VALIDATION", "FINAL")
MODE = os.environ.get("ER_MODE", "DEV_FAST").upper()        # FINAL must be chosen explicitly
if MODE not in MODES:
    raise ValueError(f"ER_MODE must be one of {MODES}, got {MODE!r}")
MODE_FRACTION = {"DEV_FAST": DEV_FAST, "DEV_MEDIUM": DEV_MEDIUM,
                 "VALIDATION": float(os.environ.get("ER_VALIDATION_FRACTION", DEV_MEDIUM)),
                 "FINAL": FINAL}
FRACTION = MODE_FRACTION[MODE]                 # share of labelled S1 entities (and unmatched pool)
SUBSET_SEED = 2026                             # fixed: the same subset in every run of a mode
# GroupKFold by S1 for cross-fitting / out-of-fold evaluation
N_FOLDS = int(os.environ.get("ER_FOLDS", "5" if MODE == "VALIDATION" else "2"))

# ---------------------------------------------------------------- mode-specific caches
# FINAL keeps the historical full-data locations (so cached full-data candidates resume);
# every other mode gets its own folder, so subsets never mix with full-data artefacts.
RUN = CACHE if MODE == "FINAL" else os.path.join(CACHE, "runs", MODE)
INDEX = os.path.join(RUN, f"index_v{NORM_VERSION}")
MODELS = os.path.join(RUN, "models")
WORK = os.path.join(RUN, "work")
SIG = RUN                                      # signatures.py adds its versioned sub-folder
OUT = os.path.join(ROOT, "output") if MODE == "FINAL" else os.path.join(RUN, "output")
LOG = os.path.join(RUN, "pipeline_log.txt")
