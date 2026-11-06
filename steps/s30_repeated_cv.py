"""Step 30 (review R1-M4/M6, R4-M1/M7, DA-C1, R3-M6): XGBoost over many splits instead of one 70/30 split.

    python -m steps.s30_repeated_cv                    # all datasets, all protocols (GPU xgboost: ~1-3 h)
    python -m steps.s30_repeated_cv --ds hust --protocols random --repeats 5    # quick look

Inputs (one XGBoost per input, tuned params of step 3 = same hyperparameters for every input, as in the paper):
  E40   40 features              E36  E40 without history (cycle_index, Cum_Delta_Q, both slopes)  <- DA-C1
  E8    common top-8             E7   top-8 without cycle_index (compact set)
  RS    raw V,I [2x128] flat     RC   raw Q(V) + dQ(V) [2x100] flat
  RSH / RCH  RS / RC + the 4 history features (+ cycle duration if <ds>/results/raw_dur.npz exists, step 13b)
  E40T  E40 + the temperature features (MATR, SNL only)    T6  temperature features alone  <- thermal book chapter
        e.g. python -m steps.s30_repeated_cv --ds matr,snl --inputs E40,E40T,T6
  RST   RS + raw temperature trace [128]    E40TR  E40 + raw temperature trace (need raw_temp.npz, step 13 'temp')
Protocols:
  random   --repeats random 70/30 cell splits (SNL: 6/3), seeds 0..R-1; split seed 42 = the paper's split is added
  grouped  MATR: leave-one-batch-out (3 folds); UConn: leave-one-condition-out; SNL: leave-one-cell-out; HUST has
           one protocol per cell, so grouped = 5-fold cell CV
Caveat: the step-3 hyperparameters were tuned on the seed-42 training cells, which overlap the test cells of other
splits; this favours no input in particular (all inputs share them), but state it when reporting.
Output baselines-free; see s29 for baselines. -> repcv/runs.csv (one row per split x input) and repcv/pairs.csv:
paired differences (input minus E40) of per-split MAPE, mean and 95 % interval over splits, and a cell-bootstrap CI of
the pooled difference for the grouped protocol.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from rul.common import TOP8, DATASETS, ROOT, cell_folds, load_json, load_stage1, metrics
from rul.models import DEFAULTS

ap = argparse.ArgumentParser()
ap.add_argument('--ds', default=','.join(DATASETS))
ap.add_argument('--protocols', default='random,grouped')
ap.add_argument('--inputs', default='E40,E36,E8,E7,RS,RC,RSH,RCH')
ap.add_argument('--repeats', type=int, default=20)
A = ap.parse_args()

OUT = os.path.join(ROOT, 'repcv')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if xgb.build_info().get('USE_CUDA') else 'cpu'
HIST = ['cycle_index', 'Cum_Delta_Q', 'Discharge_capacity_slope', 'V_std_slope']


def inputs(df, main, res, allf):
    E = {'E40': df[main].to_numpy(np.float32), 'E36': df[[f for f in main if f not in HIST]].to_numpy(np.float32),
         'E8': df[TOP8].to_numpy(np.float32), 'E7': df[[f for f in TOP8 if f != 'cycle_index']].to_numpy(np.float32)}
    temp = [f for f in allf if f not in main and f in df]
    if temp:   # datasets that log cell temperature
        E.update(E40T=df[main + temp].to_numpy(np.float32), T6=df[temp].to_numpy(np.float32))
    if not os.path.exists(os.path.join(res, 'raw_inputs.npz')):
        print('no raw_inputs.npz: engineered inputs only', flush=True)
        return E
    z = np.load(os.path.join(res, 'raw_inputs.npz'))
    c = pd.DataFrame(z['curve'])
    c = c.loc[:, c.notna().mean() > 0.5].ffill(axis=1).bfill(axis=1)
    cells = df.cell.to_numpy(str)
    c = c.groupby(cells).ffill()
    ref = c.groupby(cells).transform('first')
    hist = df[HIST].to_numpy(np.float32)
    dur_f = os.path.join(res, 'raw_dur.npz')
    if os.path.exists(dur_f):
        d = np.load(dur_f)
        assert (d['cell'] == cells).all()
        hist = np.column_stack([hist, d['dur']])
    rs = z['sig'].reshape(len(df), -1)
    rc = np.column_stack([c.to_numpy(np.float32), (c - ref).to_numpy(np.float32)])
    E.update(RS=rs, RC=rc, RSH=np.column_stack([rs, hist]), RCH=np.column_stack([rc, hist]))
    tf = os.path.join(res, 'raw_temp.npz')   # step 13 'temp': raw temperature trace [128] (MATR, SNL)
    if os.path.exists(tf):
        t = np.load(tf)
        assert (t['cell'] == cells).all()
        E.update(RST=np.column_stack([rs, t['temp']]), E40TR=np.column_stack([E['E40'], t['temp']]))
    return E


def splits(ds, df, train, test, protocol):
    ds = ds.split('_')[0]   # matr_merged -> matr
    cells = train + test
    if protocol == 'random':
        n_tr = 6 if ds == 'snl' else int(0.7 * len(cells))
        out = [('seed42', train, test)]
        for r in range(A.repeats):
            p = list(np.random.default_rng(r).permutation(cells))
            out.append((f'r{r}', p[:n_tr], p[n_tr:]))
        return out
    if ds == 'snl':
        return [(f'loco_{c}', [k for k in cells if k != c], [c]) for c in cells]
    if ds == 'hust':
        return [(f'cv{i}', a, b) for i, (a, b) in enumerate(cell_folds(df, cells, 0))]
    if ds == 'matr':
        g = [c.split('_')[0] for c in cells]
    else:
        g = [__import__('uconn.config', fromlist=['STRATA']).STRATA[c] for c in cells]
    n = len(set(g))
    return [(f'g{i}', [cells[j] for j in a], [cells[j] for j in b])
            for i, (a, b) in enumerate(GroupKFold(n_splits=min(n, 10)).split(cells, groups=g))]


log = os.path.join(OUT, 'runs.csv')
done = set(pd.read_csv(log)[['ds', 'protocol', 'split', 'input']].astype(str).agg('|'.join, 1)) if os.path.exists(log) else set()
for ds in A.ds.split(','):
    res = os.path.join(ROOT, ds, 'results')
    df, train, test, main, allf = load_stage1(res)
    params = {**DEFAULTS['XGBoost'], **load_json(os.path.join(res, 'final_params.json'))['params'], 'device': DEV}
    X, y, cellcol = inputs(df, main, res, allf), df.RUL.to_numpy(float), df.cell.to_numpy(str)
    for protocol in A.protocols.split(','):
        for name, tr_c, te_c in splits(ds, df, train, test, protocol):
            tr, te = np.isin(cellcol, tr_c), np.isin(cellcol, te_c)
            s = df.loc[te, ['cell', 'cycle']]
            for inp in [i for i in A.inputs.split(',') if i in X]:
                key = f'{ds}|{protocol}|{name}|{inp}'
                if key in done:
                    continue
                Xi = X[inp]
                med = np.nanmedian(Xi[tr], 0)
                Xi = np.where(np.isnan(Xi), med, Xi)   # tree model: no scaling needed
                p = xgb.XGBRegressor(**params).fit(Xi[tr], y[tr]).predict(Xi[te])
                m = metrics(y[te], p, s)
                row = dict(ds=ds, protocol=protocol, split=name, input=inp, n_train=len(tr_c), n_test=len(te_c), **m)
                pd.DataFrame([row]).to_csv(log, mode='a', header=not os.path.exists(log), index=False)
                pd.DataFrame({'cell': s.cell.values, 'cycle': s.cycle.values, 'y': y[te], 'p': p}).to_csv(
                    os.path.join(OUT, f'pred_{key.replace("|", "_")}.csv.gz'), index=False)
                print(key, f'MAPE {100 * m["MAPE"]:.2f}', flush=True)

# ---- summary: per-split paired differences vs E40
r = pd.read_csv(log)
r['MAPE'] *= 100
piv = r.pivot_table(index=['ds', 'protocol', 'split'], columns='input', values='MAPE')
rows = []
for (ds, protocol), g in piv.groupby(level=[0, 1]):
    for inp in g.columns:
        d = (g[inp] - g['E40']).dropna()
        if inp == 'E40' or d.empty:
            continue
        rows.append(dict(ds=ds, protocol=protocol, input=inp, n_splits=len(d), mean_MAPE=g[inp].mean(), E40_MAPE=g['E40'].mean(),
                         diff_mean=d.mean(), diff_lo=d.quantile(0.025), diff_hi=d.quantile(0.975), frac_worse=(d > 0).mean()))
pairs = pd.DataFrame(rows)
pairs.to_csv(os.path.join(OUT, 'pairs.csv'), index=False)
print(r.groupby(['ds', 'protocol', 'input']).MAPE.agg(['mean', 'std', 'count']).round(2).to_string())
print(pairs.round(2).to_string(index=False))
