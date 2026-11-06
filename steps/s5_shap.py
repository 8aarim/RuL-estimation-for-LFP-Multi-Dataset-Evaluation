"""Step 5: SHAP feature ranking on held-out folds of the training cells (test cells never used).

For each CV fold the final model is refit on the other folds and explained on the held-out fold
(up to SHAP_ROWS rows). Importance = share of total mean |SHAP|; per-fold ranks give stability.
Writes shap_ranking.csv, fig_shap_ranking.png, fig_shap_beeswarm.png.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats

from rul.common import CV_SEED, SMOKE, build, cell_folds, load_json, load_stage1, prep, setup
from rul.models import TREE_MODELS, fit_model

SHAP_ROWS = 1000 if SMOKE else 4000

ds, cfg, RES, FIG = setup()
df, TRAIN, TEST, FEATURES, _ = load_stage1(RES)
fp = load_json(os.path.join(RES, 'final_params.json'))
algo, params = fp['algo'], fp['params']
assert algo in TREE_MODELS, f'TreeSHAP needs a tree model, got {algo}'

shares, ranks, sv_pool, x_pool = [], [], [], []
for k, (tr, te) in enumerate(cell_folds(df, TRAIN, CV_SEED)):
    Xa, ya, _ = build(df, tr, FEATURES)
    Xb, _, _ = build(df, te, FEATURES)
    Xa, Xb = prep(Xa, Xb)
    model = fit_model(algo, params, Xa, ya)
    idx = np.random.RandomState(k).choice(len(Xb), min(SHAP_ROWS, len(Xb)), replace=False)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(Xb[idx]))
    imp = np.abs(sv).mean(0)
    shares.append(imp / imp.sum())
    ranks.append(stats.rankdata(-imp))
    sv_pool.append(sv)
    x_pool.append(Xb[idx])
    print(f'fold {k}: explained {len(idx)} held-out rows', flush=True)

shares, ranks = np.array(shares), np.array(ranks)
rank = pd.DataFrame({'Feature': FEATURES, 'share': shares.mean(0), 'share_sd': shares.std(0),
                     'rank_mean': ranks.mean(0), 'rank_sd': ranks.std(0)}).sort_values('share', ascending=False)
rank.insert(0, 'rank', np.arange(1, len(rank) + 1))
rank.to_csv(os.path.join(RES, 'shap_ranking.csv'), index=False)
print(rank.head(15).round(4).to_string(index=False))

top = rank.head(20)[::-1]
fig, ax = plt.subplots(figsize=(4.5, 5))
ax.barh(top.Feature, top.share, xerr=top.share_sd, color='#1f77b4', capsize=2)
ax.set_xlabel('Share of mean |SHAP| (held-out folds, mean $\\pm$ sd)')
ax.set_title(f'{ds.upper()} {algo}: top-20 features')
fig.savefig(os.path.join(FIG, 'fig_shap_ranking.png'))

plt.figure()
shap.summary_plot(np.vstack(sv_pool), np.vstack(x_pool), feature_names=FEATURES, max_display=20, show=False)
plt.title(f'{ds.upper()} {algo}: SHAP (held-out training folds)')
plt.savefig(os.path.join(FIG, 'fig_shap_beeswarm.png'))
