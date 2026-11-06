"""Phase B statistics (local, from downloaded s32 folders; no training).

    python -m steps.s32_stats matr_merged=s32_results/matr_merged_s32 hust=s32_results/hust_s32 [--out DIR]

Per test cell error = mean |p - y| / cell life (LMAE), averaged over seeds; same as s18_stats.py.
1. table.csv       dataset x input x model: LMAE / RMSE / R2 mean and sd over seeds, per-cell LMAE, Wilcoxon p vs the
                   dataset's best (tied = p >= 0.05)
2. eng_vs_raw.csv  same model family, engineered vs raw. Equal information: E40 vs RSH/RCH (both have history), E36 vs RS/RC
                   (neither). Unequal, kept for the paper's E40 vs RS table: E40 vs RS/RC, E36 vs RSH/RCH. Plus best-vs-best
                   per family; paired Wilcoxon over test cells
3. dl_vs_xgb.csv   each tuned deep model vs XGBoost on the same input, paired Wilcoxon over test cells
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+', help='ds=path/to/s32')
ap.add_argument('--out', default='s32_stats')
A = ap.parse_args()
dirs = dict(a.split('=', 1) for a in A.runs)
os.makedirs(A.out, exist_ok=True)

FAM = {'MLP': 'MLP/1D-CNN', '1D-CNN': 'MLP/1D-CNN', 'LSTM': 'LSTM', 'CNN-LSTM': 'LSTM',
       'Transformer': 'Transformer', 'CNN-Transformer': 'Transformer', 'TCN': 'TCN', 'CNN-TCN': 'TCN',
       'PatchTST': 'PatchTST', 'CNN-PatchTST': 'PatchTST'}
PAIRS = [('E40', 'RSH'), ('E36', 'RS'), ('E40', 'RCH'), ('E36', 'RC'),   # equal information: history on both sides or neither
         ('E40', 'RS'), ('E36', 'RSH'), ('E40', 'RC'), ('E36', 'RCH')]   # unequal: only one side has history
ENG, RAW = ('E40', 'E36'), ('RS', 'RSH', 'RC', 'RCH')


def cell_err(path, r):
    d = pd.read_csv(os.path.join(path, 'preds', f'{r.protocol}_{r.input}_{r.model}_{r.seed}_{r.fold}.csv.gz'))
    life = (d.y + d.cycle).groupby(d.cell).transform('max')
    return ((d.p - d.y).abs() / life).groupby(d.cell).mean()


def test(a, b):
    a, b = a.align(b, join='inner')
    return a, b, (float(wilcoxon(a, b).pvalue) if len(a) >= 6 and (a != b).any() else np.nan)


tab, er, dx = [], [], []
for ds, path in dirs.items():
    runs = pd.read_csv(os.path.join(path, 'runs.csv'))
    per = {k: pd.concat([cell_err(path, r) for r in g.itertuples()], axis=1).mean(1)
           for k, g in runs.groupby(['input', 'model'])}
    agg = runs.groupby(['input', 'model'])[['MAPE', 'RMSE', 'R2']].agg(['mean', 'std'])
    best = min(per, key=lambda k: per[k].mean())
    for k, v in per.items():
        p = np.nan if k == best else test(per[best], v)[2]
        tab.append(dict(dataset=ds, input=k[0], model=k[1], seeds=int((runs.input == k[0]).mul(runs.model == k[1]).sum()),
                        cells=len(v), LMAE=round(agg.loc[k, ('MAPE', 'mean')] * 100, 2),
                        LMAE_sd=round(agg.loc[k, ('MAPE', 'std')] * 100, 2), RMSE=round(agg.loc[k, ('RMSE', 'mean')], 1),
                        R2=round(agg.loc[k, ('R2', 'mean')], 3), best=' '.join(best), p_vs_best=p,
                        tied_with_best=k == best or p >= 0.05))
    fam = lambda m: FAM.get(m, m)
    for f in sorted({fam(m) for _, m in per}):
        c = {i: v for (i, m), v in per.items() if fam(m) == f}
        pairs = [(f'{e}-vs-{r}', e, r) for e, r in PAIRS if e in c and r in c]
        E, R = [i for i in ENG if i in c], [i for i in RAW if i in c]
        if E and R:
            pairs.append(('best-vs-best', min(E, key=lambda i: c[i].mean()), min(R, key=lambda i: c[i].mean())))
        for how, e, r in pairs:
            a, b, p = test(c[e], c[r])
            er.append(dict(dataset=ds, family=f, pairing=how, eng=e, raw=r, cells=len(a), eng_LMAE=round(a.mean() * 100, 2),
                           raw_LMAE=round(b.mean() * 100, 2), raw_minus_eng=round((b - a).mean() * 100, 2),
                           cells_eng_better=int((a < b).sum()), p=p))
    for (i, m), v in per.items():
        if m != 'XGBoost' and (i, 'XGBoost') in per:
            a, b, p = test(per[(i, 'XGBoost')], v)
            dx.append(dict(dataset=ds, input=i, model=m, cells=len(a), xgb_LMAE=round(a.mean() * 100, 2),
                           dl_LMAE=round(b.mean() * 100, 2), dl_minus_xgb=round((b - a).mean() * 100, 2),
                           cells_dl_better=int((b < a).sum()), p=p))

verdict = lambda d, col, pos, neg: np.where(d.p >= 0.05, 'n.s.', np.where(d[col] > 0, pos, neg))
tab = pd.DataFrame(tab).sort_values(['dataset', 'LMAE'])
er = pd.DataFrame(er)
er['verdict'] = verdict(er, 'raw_minus_eng', 'engineered', 'raw')
dx = pd.DataFrame(dx)
dx['verdict'] = verdict(dx, 'dl_minus_xgb', 'XGBoost', 'deep')
for name, d in (('table', tab), ('eng_vs_raw', er), ('dl_vs_xgb', dx)):
    d.to_csv(os.path.join(A.out, f'{name}.csv'), index=False)

pd.set_option('display.width', 200)
for ds, g in tab.groupby('dataset', sort=False):
    print(f'\n{ds}: best = {g.best.iloc[0]} ({g.cells.iloc[0]} test cells); tied with best (p >= 0.05):')
    print('   ' + ', '.join(f'{a} {b} {m:.2f}%' for a, b, m in g[g.tied_with_best][['input', 'model', 'LMAE']].to_numpy()))
print('\nEngineered vs raw, same family (per-cell paired Wilcoxon):')
print(er.to_string(index=False))
print('\nVerdicts:\n' + er.groupby(['dataset', 'pairing']).verdict.value_counts().unstack(fill_value=0).to_string())
print('\nTuned deep vs XGBoost, same input:\n' + dx.groupby(['dataset', 'input']).verdict.value_counts().unstack(fill_value=0).to_string())
print(f'\n-> {A.out}')
