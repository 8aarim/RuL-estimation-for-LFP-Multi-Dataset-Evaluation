"""The 8 algorithms (MATR-notebook settings), tuning spaces, fitting and cell-wise CV."""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from rul.common import SEED, SMOKE, build, cell_folds, prep

SVR_MAX_ROWS = 20000     # ponytail: SVR is ~O(n^2) (82 min per fit on 97k HUST rows); random sample
SMOKE_MAX_ROWS = 3000

DEFAULTS = {   # the MATR notebook's settings (XGBoost = its baseline model)
    'XGBoost':       dict(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.9,
                          colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=1.0,
                          random_state=SEED, verbosity=0),
    'LightGBM':      dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                          random_state=SEED, verbosity=-1),
    'CatBoost':      dict(iterations=500, depth=6, learning_rate=0.03, random_seed=SEED, verbose=0),
    'Grad. Boost':   dict(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=SEED),
    'Random Forest': dict(n_estimators=500, random_state=SEED, n_jobs=-1),
    'SVR':           dict(kernel='rbf', C=100, epsilon=0.1),
    'MLP':           dict(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED),
    'Decision Tree': dict(random_state=SEED),
}
CLASSES = {'XGBoost': xgb.XGBRegressor, 'LightGBM': lgb.LGBMRegressor, 'CatBoost': CatBoostRegressor,
           'Grad. Boost': GradientBoostingRegressor, 'Random Forest': RandomForestRegressor,
           'SVR': SVR, 'MLP': MLPRegressor, 'Decision Tree': DecisionTreeRegressor}
TREE_MODELS = {'XGBoost', 'LightGBM', 'CatBoost', 'Grad. Boost', 'Random Forest', 'Decision Tree'}

SPACES = {
    'XGBoost': dict(n_estimators=[300, 500, 800, 1000, 1500], max_depth=[3, 4, 5, 6, 7, 8],
                    learning_rate=[0.01, 0.02, 0.03, 0.05, 0.1], subsample=[0.6, 0.7, 0.8, 0.9, 1.0],
                    colsample_bytree=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], reg_lambda=[0, 1, 3, 5, 10],
                    reg_alpha=[0, 0.5, 1, 2], min_child_weight=[1, 3, 5, 10]),
    'LightGBM': dict(n_estimators=[300, 500, 800, 1200], num_leaves=[15, 31, 63, 127],
                     max_depth=[-1, 4, 6, 8], learning_rate=[0.01, 0.02, 0.03, 0.05, 0.1],
                     subsample=[0.6, 0.7, 0.8, 0.9, 1.0], subsample_freq=[1],
                     colsample_bytree=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], reg_lambda=[0, 1, 5, 10],
                     min_child_samples=[10, 20, 50, 100]),
    'CatBoost': dict(iterations=[300, 500, 800, 1200], depth=[4, 5, 6, 7, 8],
                     learning_rate=[0.01, 0.02, 0.03, 0.05, 0.1], l2_leaf_reg=[1, 3, 5, 10],
                     bagging_temperature=[0, 0.5, 1]),
    'Grad. Boost': dict(n_estimators=[200, 500, 800], max_depth=[3, 4, 5, 6],
                        learning_rate=[0.01, 0.03, 0.05, 0.1], subsample=[0.7, 0.8, 1.0],
                        min_samples_leaf=[1, 5, 20]),
    'Random Forest': dict(n_estimators=[300, 500], max_depth=[None, 10, 20, 30],
                          min_samples_leaf=[1, 2, 5, 10], max_features=[1.0, 0.5, 'sqrt']),
    'SVR': dict(C=[10, 100, 1000], epsilon=[0.1, 1, 10], gamma=['scale', 0.01, 0.1]),
    'MLP': dict(hidden_layer_sizes=[(64,), (128, 64), (256, 128)], alpha=[1e-4, 1e-3, 1e-2],
                learning_rate_init=[1e-3, 3e-3]),
    'Decision Tree': dict(max_depth=[None, 5, 10, 20], min_samples_leaf=[1, 5, 20, 50]),
}


def fit_model(name, params, X, y):
    cap = SVR_MAX_ROWS if name == 'SVR' else None
    if SMOKE:
        cap = min(cap or SMOKE_MAX_ROWS, SMOKE_MAX_ROWS)
    if cap and len(X) > cap:
        idx = np.random.RandomState(SEED).choice(len(X), cap, replace=False)
        X, y = X[idx], y[idx]
    return CLASSES[name](**{**DEFAULTS[name], **(params or {})}).fit(X, y)


def cv_run(df, cells, feats, name, params, seed):
    """Cell-wise CV on `cells`: per-fold R2 and out-of-fold predictions (cell, cycle, y, p)."""
    fold_r2, parts = [], []
    for tr, te in cell_folds(df, cells, seed):
        Xa, ya, _ = build(df, tr, feats)
        Xb, yb, sb = build(df, te, feats)
        Xa, Xb = prep(Xa, Xb)
        p = fit_model(name, params, Xa, ya).predict(Xb)
        fold_r2.append(r2_score(yb, p))
        parts.append(pd.DataFrame({'cell': sb['cell'].values, 'cycle': sb['cycle'].values, 'y': yb, 'p': p}))
    return np.array(fold_r2), pd.concat(parts, ignore_index=True)
