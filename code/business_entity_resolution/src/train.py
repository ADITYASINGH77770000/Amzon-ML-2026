"""LightGBM training for Stage 2 (fast ranker) and Stage 3 (matcher), grouped CV by S1."""
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

import config

STAGE2_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=100,
                     feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                     num_threads=config.N_JOBS, seed=config.SEED)
STAGE3_PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_data_in_leaf=80,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                     verbose=-1, num_threads=config.N_JOBS, seed=config.SEED, max_bin=127)


def fit_cv(X, y, groups, params, n_splits=5, rounds=3000, early=100, weight=None, log=print):
    """Grouped K-fold: returns out-of-fold probabilities and the fold models."""
    oof = np.zeros(len(y), np.float32)
    models = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits).split(X, y, groups)):
        w = None if weight is None else weight[tr]
        dtr = lgb.Dataset(X.iloc[tr], y[tr], weight=w, free_raw_data=True)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        models.append(m)
        log(f"  fold {k}: best_iter {m.best_iteration}, valid logloss {m.best_score['valid_0']['binary_logloss']:.5f}")
    return oof, models


def fit_full(X, y, params, rounds):
    return lgb.train(params, lgb.Dataset(X, y), rounds)


def predict_avg(models, X):
    p = np.zeros(len(X), np.float64)
    for m in models:
        p += m.predict(X, num_iteration=m.best_iteration or None)
    return (p / len(models)).astype(np.float32)
