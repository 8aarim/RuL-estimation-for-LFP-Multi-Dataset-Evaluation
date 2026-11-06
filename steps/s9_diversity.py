"""Step 9: how different are the datasets, and how has the data changed from older to newer campaigns?

1. Feature-space diversity, on the 40 shared features (raw units) and on the 16 unit-free relative features
   (each cell vs its own first 10 cycles): PCA map, pairwise distance (mean 1-D Wasserstein over pooled
   z-scored features), and a dataset classifier (cell-grouped CV: how well can a cycle be told apart?).
2. Timeline / cell generation, oldest test campaign first.
Writes compare/diversity/{fig_pca.png, fig_timeline.png, distances.csv, classifier.csv, timeline.csv}.
    python -m steps.s9_diversity
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from rul.common import DATASETS, ROOT, SEED, load_stage1
from rul.preprocess import REL_FEATURES, add_relative

OUT = os.path.join(ROOT, 'compare', 'diversity')
os.makedirs(OUT, exist_ok=True)
ORDER = ['matr', 'snl', 'hust', 'uconn']   # by test campaign: 2017-18, 2017-19, ~2021, 2022-23
COL = {'matr': '#1f77b4', 'snl': '#9467bd', 'hust': '#2ca02c', 'uconn': '#d62728'}
N = 3000                                   # cycles sampled per dataset (equal weight)

frames, lives = [], {}
for ds in DATASETS:
    df, *_ , main, _ = load_stage1(os.path.join(ROOT, ds, 'results'))
    df = add_relative(df)
    lives[ds] = (df.RUL + df.cycle).groupby(df.cell).max()
    frames.append(df.sample(min(N, len(df)), random_state=SEED).assign(ds=ds, cell=lambda d: ds + '/' + d.cell.astype(str)))
pool = pd.concat(frames, ignore_index=True)
SETS = {'raw 40 features': main, '16 relative features': REL_FEATURES}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
dist_rows, clf_rows = [], []
for ax, (name, feats) in zip(axes, SETS.items()):
    X = pool[feats].astype(float)
    X = X.fillna(X.median())
    Z = ((X - X.mean()) / X.std().replace(0, 1)).clip(-6, 6).to_numpy()
    P = PCA(2, random_state=SEED).fit(Z)
    xy = P.transform(Z)
    for ds in ORDER:
        m = (pool.ds == ds).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.35, color=COL[ds], label=ds.upper(), rasterized=True)
    ax.set(title=name, xlabel=f'PC1 ({P.explained_variance_ratio_[0]:.0%})', ylabel=f'PC2 ({P.explained_variance_ratio_[1]:.0%})')
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            ma, mb = (pool.ds == a).to_numpy(), (pool.ds == b).to_numpy()
            d = np.mean([wasserstein_distance(Z[ma, j], Z[mb, j]) for j in range(Z.shape[1])])
            dist_rows.append({'features': name, 'pair': f'{a.upper()}-{b.upper()}', 'distance': d})
    pred = np.empty(len(pool), dtype=object)    # cell-grouped: no cycle of a test cell is seen in training
    for tr, te in GroupKFold(5).split(Z, pool.ds, pool.cell):
        pred[te] = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(Z[tr], pool.ds[tr]).predict(Z[te])
    clf_rows.append({'features': name, 'balanced_accuracy': balanced_accuracy_score(pool.ds, pred), 'chance': 1 / len(DATASETS)})
axes[0].legend(markerscale=4, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_pca.png'))

dist = pd.DataFrame(dist_rows)
dist.to_csv(os.path.join(OUT, 'distances.csv'), index=False)
pd.DataFrame(clf_rows).to_csv(os.path.join(OUT, 'classifier.csv'), index=False)
print(dist.pivot(index='pair', columns='features', values='distance').round(3).to_string())
print(pd.DataFrame(clf_rows).round(3).to_string(index=False))

# 2. timeline, oldest campaign first. Hardcoded facts from the source papers / READMEs (see dataset table).
META = {
    'matr': ('2017-2018', 2019, 'A123 APR18650M1A', 1.1, 'fast-charge protocol', '72 charge policies, 3 batches', 'per cycle (interpolated)', 'yes'),
    'snl': ('2017-2019', 2020, 'A123 APR18650M1A', 1.1, 'temperature, discharge rate', '3 conditions (25 C 3C; 35 C 1C, 2C)', '120 s', 'yes'),
    'hust': ('~2021', 2022, 'LFP 18650 (A123 M1A spec; part no. not stated)', 1.1, 'discharge protocol', '77 multi-stage discharge protocols', '5 s', 'no'),
    'uconn': ('2022-2023', 2025, 'LithiumWerks APR18650M1B', 1.2, 'charge rate, discharge rate, DoD', '11 groups', '1 s', 'no'),
}
init = {'uconn': 1.21}   # C/3 reference test, cell median (cycling capacity is a partial-DoD window)
rows = []
for ds in ORDER:
    df = pd.read_pickle(os.path.join(ROOT, ds, 'results', 'features_v2.pkl'))
    q0 = init.get(ds, df.sort_values('cycle').groupby('cell').Discharge_capacity.apply(lambda s: s.iloc[:20].median()).median())
    L = lives[ds]
    yrs, pub, cell, nom, varies, conds, log, temp = META[ds]
    rows.append({'dataset': ds.upper(), 'test campaign': yrs, 'published': pub, 'cell': cell, 'nominal Ah': nom,
                 'initial Ah (median)': round(q0, 3), 'what varies': varies, 'conditions': conds, 'logging': log,
                 'temperature logged': temp, 'cells': L.size, 'cycles (rows)': len(df), 'cycle life median': int(L.median()),
                 'cycle life range': f'{int(L.min())}-{int(L.max())}', 'fade to EOL (%/100 cycles, median cell)': round(20 / L.median() * 100, 2)})
tl = pd.DataFrame(rows)
tl.to_csv(os.path.join(OUT, 'timeline.csv'), index=False)
print(tl.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.boxplot([lives[d].to_numpy() for d in ORDER], widths=0.5)
ax.set_xticks(range(1, 5), [f'{d.upper()}\n{META[d][0]}\n{META[d][2].split()[-1]}' for d in ORDER], fontsize=8)
ax.set(ylabel='Cycle life to 80% (cycles)', title='Cycle life by test campaign (oldest -> newest)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_timeline.png'))
