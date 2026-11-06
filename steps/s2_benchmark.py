"""Step 2: 8-algorithm comparison, 5-fold cell-wise CV on the training cells.

Winner = highest mean fold R2. "Tied" = per-cell out-of-fold R2 not significantly different from the
winner's (Wilcoxon signed-rank, p >= 0.05). Each algorithm's result is cached (bench_*.pkl), so an
interrupted run resumes; delete a cache file to rerun that algorithm.
Writes benchmark_cv.csv, benchmark.json, fig_benchmark_cv.png.
"""
import os
import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from rul.common import CV_SEED, load_stage1, metrics, per_cell_r2, save_json, setup
from rul.models import DEFAULTS, cv_run

ds, cfg, RES, FIG = setup()
df, TRAIN, TEST, FEATURES, _ = load_stage1(RES)

bench = {}
for name in DEFAULTS:
    path = os.path.join(RES, f'bench_{name.replace(" ", "_").replace(".", "")}.pkl')
    if os.path.exists(path):
        bench[name] = pickle.load(open(path, 'rb'))
        print(f'{name:14s} cached   CV R2 = {bench[name]["fold_r2"].mean():.4f}')
        continue
    t0 = time.time()
    fr2, oof = cv_run(df, TRAIN, FEATURES, name, None, CV_SEED)
    bench[name] = dict(fold_r2=fr2, oof=oof, secs=time.time() - t0)
    pickle.dump(bench[name], open(path, 'wb'))
    print(f'{name:14s} {bench[name]["secs"]:7.0f}s  CV R2 = {fr2.mean():.4f} +/- {fr2.std(ddof=1):.4f}', flush=True)

winner = max(bench, key=lambda n: bench[n]['fold_r2'].mean())
pc = {n: per_cell_r2(b['oof']) for n, b in bench.items()}
rows = []
for n, b in bench.items():
    pooled = metrics(b['oof']['y'].values, b['oof']['p'].values, b['oof'])
    p_w = float('nan') if n == winner else stats.wilcoxon(pc[winner], pc[n]).pvalue
    rows.append({'Model': n, 'CV_R2': b['fold_r2'].mean(), 'CV_R2_sd': b['fold_r2'].std(ddof=1),
                 'OOF_R2': pooled['R2'], 'OOF_RMSE': pooled['RMSE'], 'OOF_MAE': pooled['MAE'],
                 'median_cell_R2': pc[n].median(), 'p_vs_winner': p_w,
                 'tied_with_winner': n == winner or p_w >= 0.05, 'seconds': b['secs']})
tab = pd.DataFrame(rows).sort_values('CV_R2', ascending=False)
tab.to_csv(os.path.join(RES, 'benchmark_cv.csv'), index=False)
tied = tab[tab.tied_with_winner].Model.tolist()
save_json({'winner': winner, 'tied': tied}, os.path.join(RES, 'benchmark.json'))
print()
print(tab.round(4).to_string(index=False))
print(f'\nCV winner: {winner} | tied with it: {tied}')

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.barh(tab.Model[::-1], tab.CV_R2[::-1], xerr=tab.CV_R2_sd[::-1], capsize=3,
        color=['#1f77b4' if t else '#b0b0b0' for t in tab.tied_with_winner[::-1]])
ax.set_xlim(max(0, tab.CV_R2.min() - 0.1), 1)
ax.set_xlabel('5-fold cell-wise CV $R^2$ (training cells)')
ax.set_title(f'{ds.upper()}: algorithm comparison (blue = tied with best)')
fig.savefig(os.path.join(FIG, 'fig_benchmark_cv.png'))
