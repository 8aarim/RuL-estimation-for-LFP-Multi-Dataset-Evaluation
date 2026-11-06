"""Step 22: accuracy of the LODO core features (step 21) as fixed subsets on every dataset.

    python -m steps.s22_core          -> core_subsets.csv

core4 = in the top 5 of every leave-one-dataset-out ordering; core6 = core4 + the two others present in every
held-out k_H subset (CMT, CC_length). top8 (step 12) and all-40 are rerun here with the same xgboost build so the
rows compare like for like. Same split, folds (CV_SEED), preprocessing and tuned params as steps 4-12.
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from rul.common import TOP8, CV_SEED, DATASETS, ROOT, build, cell_folds, load_json, load_stage1, metrics, prep
from rul.models import DEFAULTS

CORE4 = ['Cum_Delta_Q', 'Discharge_capacity_slope', 'Discharge_capacity', 'Charge_time']   # paper's ablation sets (fixed)
SUBSETS = {'core4': CORE4, 'core6': CORE4 + ['CMT', 'CC_length'], 'top8': TOP8, 'all40': None,
           'core4+cycle_index': CORE4 + ['cycle_index'],   # item 9: how much of top8 is cycle_index?
           'top8-cycle_index': [f for f in TOP8 if f != 'cycle_index'],
           'top8-cycle_index-Cum_Delta_Q': [f for f in TOP8 if f not in ('cycle_index', 'Cum_Delta_Q')]}
DEV = 'cuda' if xgb.build_info().get('USE_CUDA') else 'cpu'

rows = []
for ds in DATASETS:
    res = os.path.join(ROOT, ds, 'results')
    df, train, test, main, _ = load_stage1(res)
    params = {**DEFAULTS['XGBoost'], **load_json(os.path.join(res, 'final_params.json'))['params'], 'device': DEV}
    fit = lambda X, y: xgb.XGBRegressor(**params).fit(X, y)
    for name, feats in SUBSETS.items():
        feats = feats or main
        r2 = []
        for tr, te in cell_folds(df, train, CV_SEED):
            (Xa, ya, _), (Xb, yb, _) = build(df, tr, feats), build(df, te, feats)
            Xa, Xb = prep(Xa, Xb)
            r2.append(r2_score(yb, fit(Xa, ya).predict(Xb)))
        (Xa, ya, _), (Xb, yb, sb) = build(df, train, feats), build(df, test, feats)
        Xa, Xb = prep(Xa, Xb)
        m = metrics(yb, fit(Xa, ya).predict(Xb), sb)
        rows.append(dict(ds=ds, subset=name, k=len(feats), cv_r2=np.mean(r2), cv_se=np.std(r2, ddof=1) / np.sqrt(len(r2)),
                         test_R2=m['R2'], test_RMSE=m['RMSE'], test_MAPE=100 * m['MAPE']))
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, 'core_subsets.csv'), index=False)
print(out.pivot(index='subset', columns='ds', values='test_MAPE').round(2).to_string())
print(out.pivot(index='subset', columns='ds', values='cv_r2').round(4).to_string())
