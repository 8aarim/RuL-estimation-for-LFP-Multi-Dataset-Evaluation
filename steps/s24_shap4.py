"""Step 24: four-dataset SHAP figures and direction check.

    python -m steps.s24_shap4            recompute SHAP values (same folds, rows and model as step 5) -> <ds>/results/shap_values.npz,
                                         then figures and tables
    python -m steps.s24_shap4 fig        figures and tables only, from the saved values

Direction of a feature = Spearman rho between its (standardised) value and its SHAP value over the explained held-out rows:
rho > 0 -> higher value pushes predicted RUL up. Outputs in shap4/: direction.csv, check_vs_step5.csv,
fig_shap4_beeswarm.png, fig_shap4_rank_direction.png.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from rul.common import shap_ranks, CV_SEED, DATASETS, ROOT, build, cell_folds, load_json, load_stage1, prep
from rul.models import fit_model

SHAP_ROWS = 4000                                     # as step 5
OUT = os.path.join(ROOT, 'shap4')
NAME = {'matr': 'MATR', 'hust': 'HUST', 'uconn': 'UConn-ISU-ILCC', 'snl': 'SNL'}
MIN_SHARE = 0.02                                     # below 2% of mean |SHAP| the direction is not interpreted
res = lambda ds: os.path.join(ROOT, ds, 'results')


def compute(ds):
    df, train, _, feats, _ = load_stage1(res(ds))
    fp = load_json(os.path.join(res(ds), 'final_params.json'))
    sv_pool, x_pool = [], []
    for k, (tr, te) in enumerate(cell_folds(df, train, CV_SEED)):
        Xa, ya, _ = build(df, tr, feats)
        Xb, _, _ = build(df, te, feats)
        Xa, Xb = prep(Xa, Xb)
        model = fit_model(fp['algo'], fp['params'], Xa, ya)
        idx = np.random.RandomState(k).choice(len(Xb), min(SHAP_ROWS, len(Xb)), replace=False)
        sv_pool.append(np.asarray(shap.TreeExplainer(model).shap_values(Xb[idx])))
        x_pool.append(Xb[idx])
        print(f'{ds} fold {k}: {len(idx)} rows', flush=True)
    np.savez_compressed(os.path.join(res(ds), 'shap_values.npz'), sv=np.vstack(sv_pool), x=np.vstack(x_pool),
                        feats=np.array(feats), fold=np.concatenate([np.full(len(s), k) for k, s in enumerate(sv_pool)]))


def load(ds):
    z = np.load(os.path.join(res(ds), 'shap_values.npz'))
    return z['sv'], z['x'], list(z['feats']), z['fold']


def tables():
    rows, check = [], []
    for ds in DATASETS:
        sv, x, feats, fold = load(ds)
        share = np.abs(sv).mean(0) / np.abs(sv).mean(0).sum()
        step5 = pd.read_csv(os.path.join(res(ds), 'shap_ranking.csv')).set_index('Feature')
        rank_here = pd.Series(stats.rankdata(-share), index=feats)
        check.append(dict(ds=ds, spearman_vs_step5=stats.spearmanr(rank_here[step5.index], step5['rank'])[0],
                          top8_same=len(set(rank_here.nsmallest(8).index) & set(step5.index[:8]))))
        for j, f in enumerate(feats):
            rho = stats.spearmanr(x[:, j], sv[:, j])[0] if np.std(x[:, j]) > 0 else np.nan
            rows.append(dict(ds=ds, feature=f, share=share[j], rank=int(step5.loc[f, 'rank']), rho=rho))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'direction.csv'), index=False)
    pd.DataFrame(check).to_csv(os.path.join(OUT, 'check_vs_step5.csv'), index=False)
    print(pd.DataFrame(check).to_string(index=False))
    return d


def figures(d):
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    # (1) four-panel beeswarm, top 15 per dataset
    fig = plt.figure(figsize=(11, 11))
    for k, ds in enumerate(DATASETS):
        sv, x, feats, _ = load(ds)
        ax = fig.add_subplot(2, 2, k + 1)
        plt.sca(ax)
        shap.summary_plot(sv, x, feature_names=feats, max_display=15, show=False, plot_size=None, color_bar=(k % 2 == 1))
        ax.set_title(f'({"abcd"[k]}) {NAME[ds]}', fontsize=10, loc='left')
        ax.set_xlabel('SHAP value (cycles of RUL)', fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_shap4_beeswarm.png'), dpi=300)
    plt.close(fig)

    # (2) rank heatmap (all 40, common order) + direction heatmap (common top 12)
    order = shap_ranks().index.tolist()
    R = d.pivot(index='feature', columns='ds', values='rank').loc[order, list(DATASETS)]
    rho = d.pivot(index='feature', columns='ds', values='rho').loc[order[:12], list(DATASETS)]
    sh = d.pivot(index='feature', columns='ds', values='share').loc[order[:12], list(DATASETS)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 7.6), gridspec_kw=dict(width_ratios=[1, 1.05]))
    blues = LinearSegmentedColormap.from_list('b', ['#0d366b', '#1c5cab', '#3987e5', '#9ec5f4', '#eaf2fd'])
    a1.imshow(R.values, cmap=blues, vmin=1, vmax=40, aspect='auto')
    for i in range(R.shape[0]):
        for j in range(4):
            v = R.values[i, j]
            a1.text(j, i, int(v), ha='center', va='center', fontsize=6.5, color='white' if v <= 12 else '#1a1a1a')
    a1.axhline(7.5, color='#eb6834', lw=1.4)
    a1.set_xticks(range(4), [NAME[s].split('-')[0] for s in DATASETS])
    a1.set_yticks(range(len(order)), [f.replace('_', '\\_') if False else f for f in order], fontsize=6.5)
    a1.set_title('(a) SHAP rank per dataset (1 = most important)', fontsize=8.5, loc='left')
    for s in a1.spines.values():
        s.set_visible(False)
    div = LinearSegmentedColormap.from_list('d', ['#c4421a', '#f3c9b5', '#f2f2f0', '#b9d3f2', '#1c5cab'])
    a2.imshow(rho.values, cmap=div, vmin=-1, vmax=1, aspect='auto')
    for i in range(rho.shape[0]):
        for j in range(4):
            v, s_ = rho.values[i, j], sh.values[i, j]
            faint = s_ < MIN_SHARE
            a2.text(j, i, f'{v:+.2f}' + ('' if not faint else '*'), ha='center', va='center', fontsize=7,
                    color='#8a8a8a' if faint else ('white' if abs(v) > 0.6 else '#1a1a1a'))
    a2.set_xticks(range(4), [NAME[s].split('-')[0] for s in DATASETS])
    a2.set_yticks(range(12), order[:12], fontsize=7)
    a2.set_title('(b) Direction: Spearman ρ(value, SHAP), common top 12', fontsize=8.5, loc='left')
    for s in a2.spines.values():
        s.set_visible(False)
    a2.text(0, 12.3, 'ρ > 0: higher value → longer predicted RUL;  * share of |SHAP| < 2%', fontsize=6.5, va='top', color='#555555')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_shap4_rank_direction.png'), dpi=300)
    plt.close(fig)
    print(rho.round(2).to_string())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    only = [a for a in sys.argv[1:] if a in DATASETS]   # e.g. `s24_shap4 matr`: recompute one dataset, reuse the rest
    if 'fig' not in sys.argv[1:]:
        for ds in only or DATASETS:
            compute(ds)
    figures(tables())
