"""Step 8 (after steps 1-6 on every dataset): one feature set that works on ALL datasets.

Adds protocol-normalised features (rul.preprocess.add_relative: each cell vs its own first 10
cycles), ranks the 40 + 16 features by SHAP on held-out folds of each dataset's training cells,
combines the ranks, then scores the SAME subsets on every dataset: 5-fold CV on training cells
(choice) and once on test cells (check). Model = each dataset's step-3 CatBoost.
Writes compare/common_ranking.csv, compare/common_cv.csv, compare/common_test.csv.
    python -m steps.s8_common
"""
import os

import numpy as np
import pandas as pd
import shap

from rul.common import CV_SEED, DATASETS, ROOT, SMOKE, build, cell_bootstrap_ci, cell_folds, load_json, load_stage1, metrics, prep
from rul.models import cv_run, fit_model
from rul.preprocess import REL_FEATURES, add_relative

SHAP_ROWS = 1000 if SMOKE else 2000
sfx = '_smoke' if SMOKE else ''
OUT = os.path.join(ROOT, 'compare' + sfx)
os.makedirs(OUT, exist_ok=True)


def shap_share(df, cells, feats, algo, params):
    shares = []
    for k, (tr, te) in enumerate(cell_folds(df, cells, CV_SEED)):
        Xa, ya, _ = build(df, tr, feats)
        Xb, _, _ = build(df, te, feats)
        Xa, Xb = prep(Xa, Xb)
        idx = np.random.RandomState(k).choice(len(Xb), min(SHAP_ROWS, len(Xb)), replace=False)
        imp = np.abs(shap.TreeExplainer(fit_model(algo, params, Xa, ya)).shap_values(Xb[idx])).mean(0)
        shares.append(imp / imp.sum())
    return pd.Series(np.mean(shares, 0), feats)


data = {}
for ds in DATASETS:
    res = os.path.join(ROOT, ds, 'results' + sfx)
    df, train, test, main, _ = load_stage1(res)
    fp = load_json(os.path.join(res, 'final_params.json'))
    data[ds] = (add_relative(df), train, test, main, fp['algo'], fp['params'])
    print(f'{ds}: {df.cell.nunique()} cells, relative features added', flush=True)

assert all(sorted(d[3]) == sorted(data[DATASETS[0]][3]) for d in data.values()), 'main sets differ'
POOL = data[DATASETS[0]][3] + REL_FEATURES
share = {}
for ds, (df, train, _, _, algo, params) in data.items():
    share[ds] = shap_share(df, train, POOL, algo, params)
    print(f'{ds}: SHAP top-8 of {len(POOL)}: {share[ds].sort_values(ascending=False).index[:8].tolist()}', flush=True)
rank = pd.DataFrame({f'{ds}_rank': s.rank(ascending=False).astype(int) for ds, s in share.items()})
for ds, s in share.items():
    rank[f'{ds}_share'] = s.round(4)
cols = [f'{ds}_rank' for ds in DATASETS]
rank['mean_rank'] = rank[cols].mean(axis=1)
rank['worst_rank'] = rank[cols].max(axis=1)
rank['relative'] = rank.index.isin(REL_FEATURES)
rank = rank.sort_values(['mean_rank', 'worst_rank'])
rank.to_csv(os.path.join(OUT, 'common_ranking.csv'), index_label='Feature')
print(rank.head(20).to_string())

by_mean = rank.index.tolist()
by_worst = rank.sort_values(['worst_rank', 'mean_rank']).index.tolist()
old = pd.read_csv(os.path.join(OUT, 'ranking_combined.csv')).Feature.tolist()     # step 7, 40 features
subsets = {'all 40': data[DATASETS[0]][3], 'all 40 + relative': POOL,
           'relative + cycle_index': REL_FEATURES + ['cycle_index'], 'step-7 common top-10': old[:10]}
for k in (5, 8, 10, 15):
    subsets[f'common top-{k} (mean rank)'] = by_mean[:k]
    subsets[f'common top-{k} (worst rank)'] = by_worst[:k]

cv_rows, test_rows = [], []
for ds, (df, train, test, _, algo, params) in data.items():
    for label, feats in subsets.items():
        r2, _ = cv_run(df, train, feats, algo, params, CV_SEED)
        cv_rows.append({'dataset': ds, 'subset': label, 'k': len(feats), 'cv_r2': r2.mean(), 'cv_sd': r2.std(ddof=1)})
        Xa, ya, _ = build(df, train, feats)
        Xb, yb, sb = build(df, test, feats)
        Xa, Xb = prep(Xa, Xb)
        p = fit_model(algo, params, Xa, ya).predict(Xb)
        lo, hi = cell_bootstrap_ci(yb, p, sb['cell'].to_numpy())
        test_rows.append({'dataset': ds, 'subset': label, 'k': len(feats), **metrics(yb, p, sb), 'R2_lo': lo, 'R2_hi': hi})
        print(f'{ds:6s} {label:30s} k={len(feats):2d}  CV R2 {r2.mean():.4f}  test R2 {test_rows[-1]["R2"]:.4f}', flush=True)

cv = pd.DataFrame(cv_rows)
cv.to_csv(os.path.join(OUT, 'common_cv.csv'), index=False)
test = pd.DataFrame(test_rows)
test.to_csv(os.path.join(OUT, 'common_test.csv'), index=False)
wide = cv.pivot(index='subset', columns='dataset', values='cv_r2')[list(DATASETS)]
wide['worst'] = wide.min(axis=1)
print('\nCV R2 (training cells) per subset, sorted by the worst dataset:')
print(wide.sort_values('worst', ascending=False).round(4).to_string())
for col in ('R2', 'RMSE', 'MAE', 'MAPE'):
    print(f'\ntest-cell {col} (scored once):')
    print(test.pivot(index='subset', columns='dataset', values=col)[list(DATASETS)].round(4).to_string())
print('\nsubsets:')
for label, feats in subsets.items():
    if len(feats) <= 15:
        print(f'  {label}: {feats}')
