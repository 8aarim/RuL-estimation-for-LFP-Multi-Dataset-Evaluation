"""Step 25: stage-stratified SHAP on the four datasets, from the SHAP values saved by step 24.

    python -m steps.s25_shap_stage      -> shap4/stage_ranks.csv, shap4/stage_rows.csv

Stage = fraction of the cell's life already used, cycle / (cycle + RUL): early < 1/3, mid 1/3-2/3, late > 2/3
(life fractions, not fixed RUL cut-offs, because median life ranges from 736 to 3,865 cycles).
Rows are re-derived by replaying step 24's folds and random draws, so they line up with the saved SHAP values.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from rul.common import TOP8, CV_SEED, DATASETS, ROOT, build, cell_folds, load_stage1

OUT = os.path.join(ROOT, 'shap4')
SHAP_ROWS = 4000
STAGES = ('early', 'mid', 'late')
rows, counts = [], []
for ds in DATASETS:
    res = os.path.join(ROOT, ds, 'results')
    z = np.load(os.path.join(res, 'shap_values.npz'))
    sv, feats = z['sv'], list(z['feats'])
    df, train, _, _, _ = load_stage1(res)
    frac = []
    for k, (tr, te) in enumerate(cell_folds(df, train, CV_SEED)):
        _, y, s = build(df, te, feats)
        idx = np.random.RandomState(k).choice(len(y), min(SHAP_ROWS, len(y)), replace=False)
        c = s['cycle'].to_numpy()[idx]
        frac.append(c / (c + y[idx]))
    frac = np.concatenate(frac)
    assert len(frac) == len(sv), f'{ds}: rows do not line up'
    stage = np.digitize(frac, [1 / 3, 2 / 3])
    for st, name in enumerate(STAGES):
        m = stage == st
        imp = np.abs(sv[m]).mean(0)
        rk = stats.rankdata(-imp)
        counts.append(dict(ds=ds, stage=name, rows=int(m.sum())))
        rows += [dict(ds=ds, stage=name, feature=f, share=imp[j] / imp.sum(), rank=int(rk[j])) for j, f in enumerate(feats)]
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, 'stage_ranks.csv'), index=False)
pd.DataFrame(counts).to_csv(os.path.join(OUT, 'stage_rows.csv'), index=False)
t = d[d.feature.isin(TOP8)].pivot_table(index='feature', columns=['ds', 'stage'], values='rank').loc[TOP8]
t = t[[(ds, s) for ds in DATASETS for s in STAGES]]
print(pd.DataFrame(counts).pivot(index='ds', columns='stage', values='rows')[list(STAGES)])
print(t.to_string())
for ds in DATASETS:
    top = {s: d[(d.ds == ds) & (d.stage == s)].nsmallest(3, 'rank').feature.tolist() for s in STAGES}
    print(ds, top)
