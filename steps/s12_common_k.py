"""Step 12: one k for all datasets. Same feature order everywhere (features_shap.xlsx 'Overall' = mean SHAP rank over
the 4 datasets), top-k for k = 5..40, 5-fold cell-wise CV on training cells (choice); test cells scored once per k (check).
Sweet spot = smallest k whose CV R2 is within 1 SE of that dataset's best k on EVERY dataset.
Writes compare/common_k_cv.csv, compare/common_k_test.csv.
    python -m steps.s12_common_k
"""
import os

import numpy as np
import pandas as pd

from rul.common import TOP8, shap_ranks, CV_SEED, DATASETS, ROOT, build, cell_bootstrap_ci, load_json, load_stage1, metrics, prep
import xgboost as xgb

from rul.models import cv_run, fit_model

KS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 40)
order = shap_ranks().index.tolist()   # same rule as features_shap.xlsx 'Overall' (mean rank, tie -> worst rank)
assert order[:8] == TOP8, order[:8]

DEV = {'device': 'cuda'} if xgb.build_info().get('USE_CUDA') else {}   # GPU on DICC (same trees, faster)
PART = os.path.join(ROOT, 'compare', 'common_k_parts.pkl')   # resumable across 1-h HPC jobs: {(ds, k): (cv_row, test_row)}
done = pd.read_pickle(PART) if os.path.exists(PART) else {}
cv_rows, test_rows = [], []
for ds in DATASETS:
    res = os.path.join(ROOT, ds, 'results')
    df, train, test, main, _ = load_stage1(res)
    assert sorted(order) == sorted(main), f'{ds}: workbook features differ from the main set'
    fp = load_json(os.path.join(res, 'final_params.json'))
    fp['params'] = {**fp['params'], **DEV} if fp['algo'] == 'XGBoost' else fp['params']
    for k in KS:
        if (ds, k) in done:
            cv_rows.append(done[ds, k][0]); test_rows.append(done[ds, k][1])
            continue
        feats = order[:k]
        r2, _ = cv_run(df, train, feats, fp['algo'], fp['params'], CV_SEED)
        cv_rows.append({'dataset': ds, 'k': k, 'cv_r2': r2.mean(), 'se': r2.std(ddof=1) / np.sqrt(len(r2))})
        Xa, ya, _ = build(df, train, feats)
        Xb, yb, sb = build(df, test, feats)
        Xa, Xb = prep(Xa, Xb)
        p = fit_model(fp['algo'], fp['params'], Xa, ya).predict(Xb)
        lo, hi = cell_bootstrap_ci(yb, p, sb['cell'].to_numpy())
        test_rows.append({'dataset': ds, 'k': k, **metrics(yb, p, sb), 'R2_lo': lo, 'R2_hi': hi})
        done[ds, k] = (cv_rows[-1], test_rows[-1])
        pd.to_pickle(done, PART)
        print(f'{ds:6s} k={k:2d}  CV R2 {r2.mean():.4f}  test R2 {test_rows[-1]["R2"]:.4f}  MAPE {test_rows[-1]["MAPE"]:.4f}', flush=True)

cv = pd.DataFrame(cv_rows)
cv.to_csv(os.path.join(ROOT, 'compare', 'common_k_cv.csv'), index=False)
pd.DataFrame(test_rows).to_csv(os.path.join(ROOT, 'compare', 'common_k_test.csv'), index=False)
best = cv.loc[cv.groupby('dataset').cv_r2.idxmax()].set_index('dataset')
cv['ok'] = cv.apply(lambda r: r.cv_r2 >= best.cv_r2[r.dataset] - best.se[r.dataset], axis=1)
ok_all = cv.groupby('k').ok.all()
print(cv.pivot(index='k', columns='dataset', values='cv_r2')[list(DATASETS)].round(4).to_string())
print('best k per dataset:', best.k.to_dict())
print('k within 1 SE of best on every dataset:', ok_all[ok_all].index.tolist())
