"""Step 6: how many features? Top-k CV sweep (training cells), then subsets scored ONCE on test cells.

k* = smallest k whose CV R2 is within one standard error of the best k (1-SE rule). The SHAP order
came from the same training folds, so the sweep is mildly optimistic; the test-cell rows are the
unbiased check. MATR also reports its all-informative set (46 features, incl. temperature).
Writes topk_cv.csv, subsets_test.csv, fig_topk_cv.png.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul.common import CV_SEED, build, cell_bootstrap_ci, load_json, load_stage1, metrics, prep, save_json, setup
from rul.models import cv_run, fit_model

ds, cfg, RES, FIG = setup()
df, TRAIN, TEST, FEATURES, ALL = load_stage1(RES)
fp = load_json(os.path.join(RES, 'final_params.json'))
algo, params = fp['algo'], fp['params']
order = pd.read_csv(os.path.join(RES, 'shap_ranking.csv')).Feature.tolist()

ks = sorted({k for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30) if k < len(order)} | {len(order)})
rows = []
for k in ks:
    fr2, _ = cv_run(df, TRAIN, order[:k], algo, params, CV_SEED)
    rows.append({'k': k, 'added': order[k - 1], 'cv_r2': fr2.mean(), 'cv_r2_sd': fr2.std(ddof=1),
                 'se': fr2.std(ddof=1) / np.sqrt(len(fr2))})
    print(f'k={k:2d}  +{order[k - 1]:<26s} CV R2 = {fr2.mean():.4f} +/- {fr2.std(ddof=1):.4f}', flush=True)
sweep = pd.DataFrame(rows)
best = sweep.loc[sweep.cv_r2.idxmax()]
k_star = int(sweep[sweep.cv_r2 >= best.cv_r2 - best.se].k.min())
sweep.to_csv(os.path.join(RES, 'topk_cv.csv'), index=False)
print(f'\nbest k = {int(best.k)} (CV R2 {best.cv_r2:.4f}); 1-SE rule -> k* = {k_star}: {order[:k_star]}')

subsets = {f'all {len(FEATURES)}': FEATURES, f'top-{k_star} (1-SE)': order[:k_star],
           'top-5': order[:5], 'top-10': order[:10]}
if len(ALL) > len(FEATURES):
    subsets[f'all informative {len(ALL)} (incl. temperature)'] = ALL
test_rows = []
for label, feats in subsets.items():
    if any(r['features'] == ';'.join(feats) for r in test_rows):
        continue
    Xa, ya, _ = build(df, TRAIN, feats)
    Xb, yb, sb = build(df, TEST, feats)
    Xa, Xb = prep(Xa, Xb)
    p = fit_model(algo, params, Xa, ya).predict(Xb)
    lo, hi = cell_bootstrap_ci(yb, p, sb['cell'].to_numpy())
    test_rows.append({'subset': label, 'k': len(feats), **metrics(yb, p, sb), 'R2_lo': lo, 'R2_hi': hi,
                      'features': ';'.join(feats)})
test = pd.DataFrame(test_rows)
test.to_csv(os.path.join(RES, 'subsets_test.csv'), index=False)
save_json({'k_star': k_star, 'k_star_features': order[:k_star]}, os.path.join(RES, 'topk.json'))
print(test.drop(columns='features').round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.errorbar(sweep.k, sweep.cv_r2, yerr=sweep.se, marker='o', ms=3, capsize=2, color='#1f77b4')
ax.axhline(best.cv_r2 - best.se, ls=':', color='grey', lw=0.8, label='best $-$ 1 SE')
ax.axvline(k_star, ls='--', color='#d62728', lw=0.8, label=f'$k^*$ = {k_star}')
ax.set(xlabel='Number of top-SHAP features, k', ylabel='5-fold CV $R^2$ (training cells)')
ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, 'fig_topk_cv.png'))
