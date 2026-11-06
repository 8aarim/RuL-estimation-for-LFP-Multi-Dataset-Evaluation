"""Step 31: where is the model reliable? Uncertainty + stratified and hold-out testing across conditions.
(Review R1-M9, R3-M2/M3/M4, user request 2026-10-14.) Local, GPU xgboost recommended (~1-2 h for all).

    python -m steps.s31_reliability                       # all datasets, E40 and E7
    python -m steps.s31_reliability --ds snl --sets E7    # quick

A. Out-of-fold predictions for EVERY cell (cell-wise 5-fold CV over all cells; SNL leave-one-cell-out), so strata
   have enough cells. Inside each fold, 20 % of the training cells calibrate the intervals (90 % target):
     split      split conformal on |residual|                               (constant width)
     norm       normalised conformal on |residual| / (p + 0.1 * mean life)  (width grows with predicted RUL)
     cqr        conformalised quantile regression (XGBoost 5 % / 95 % quantiles)
   OOD score = mean distance to the 10 nearest training rows (standardised features), divided by the 95th percentile
   of the same distance for calibration rows: <= 1 in-domain, > 1 outside what the model was trained on.
B. Reliability map: error (MAPE, life-normalised as in the paper) + coverage + width per stratum:
     condition   MATR batch, charge policy group; UConn charge C-rate, discharge C-rate, DoD; SNL temperature,
                 discharge rate; HUST discharge-power tercile (ragone/cells.csv)
     ageing path life tercile; fade-acceleration tercile (compare/trends/cells.csv 'accel'); knee position tercile
     life stage  early / mid / late third of each cell's life
     OOD         in-domain / edge (1-1.5) / outside (> 1.5)
C. Factor hold-out: train without one level of a factor, test on that level (extrapolation to an unseen
   temperature, duty cycle, batch), with the same three intervals; compare to the in-distribution OOF numbers.
D. Selective prediction: MAPE and coverage when only the most reliable fraction of predictions is kept
   (ranked by normalised interval width or by OOD score).
-> reliability/{oof.csv.gz, map.csv, holdout.csv, selective.csv}
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from rul.common import COMPACT7, DATASETS, ROOT, cell_folds, load_json, load_stage1, meta
from rul.models import DEFAULTS

ap = argparse.ArgumentParser()
ap.add_argument('--ds', default=','.join(DATASETS))
ap.add_argument('--sets', default='E40,E7')
A = ap.parse_args()

OUT = os.path.join(ROOT, 'reliability')
os.makedirs(OUT, exist_ok=True)
ALPHA, SEED = 0.1, 42
DEV = 'cuda' if xgb.build_info().get('USE_CUDA') else 'cpu'
TOP7 = COMPACT7


# ---------------- one train/test fit with three calibrated intervals + OOD
def fit_predict(df, feats, tr_cells, te_cells, params, mean_life):
    rng = np.random.default_rng(SEED)
    tc = list(rng.permutation(tr_cells))
    n_cal = max(2 if len(tc) < 10 else 3, int(0.2 * len(tc)))
    cal, inner = tc[:n_cal], tc[n_cal:]
    get = lambda cs: df[df.cell.isin(cs)]  # noqa: E731
    a, c, t = get(inner), get(cal), get(te_cells)
    med = a[feats].median()
    Xa, Xc, Xt = (d[feats].fillna(med).to_numpy(float) for d in (a, c, t))
    m = xgb.XGBRegressor(**params).fit(Xa, a.RUL)
    q = xgb.XGBRegressor(**{**params, 'objective': 'reg:quantileerror', 'quantile_alpha': np.array([0.05, 0.95])}).fit(Xa, a.RUL)
    pc, pt = m.predict(Xc), m.predict(Xt)
    qc, qt = q.predict(Xc), q.predict(Xt)
    n = len(c)
    k = min(int(np.ceil((n + 1) * (1 - ALPHA))), n) - 1
    s_split = np.sort(np.abs(c.RUL - pc))[k]
    sig = lambda p: np.clip(p, 0, None) + 0.1 * mean_life  # noqa: E731
    s_norm = np.sort(np.abs(c.RUL - pc) / sig(pc))[k]
    s_cqr = np.sort(np.maximum(qc[:, 0] - c.RUL, c.RUL - qc[:, 1]))[k]
    sc = StandardScaler().fit(Xa)
    sub = sc.transform(Xa[rng.choice(len(Xa), min(20000, len(Xa)), replace=False)])
    nn = NearestNeighbors(n_neighbors=10).fit(sub)
    dist = lambda X: nn.kneighbors(sc.transform(X))[0].mean(1)  # noqa: E731
    d95 = np.percentile(dist(Xc), 95)
    return pd.DataFrame({'cell': t.cell.values, 'cycle': t.cycle.values, 'y': t.RUL.values, 'p': pt,
                         'split_lo': pt - s_split, 'split_hi': pt + s_split,
                         'norm_lo': pt - s_norm * sig(pt), 'norm_hi': pt + s_norm * sig(pt),
                         'cqr_lo': qt[:, 0] - s_cqr, 'cqr_hi': qt[:, 1] + s_cqr, 'ood': dist(Xt) / d95})


def summarise(d, life):
    L = d.cell.map(life).to_numpy()
    out = dict(n_cells=d.cell.nunique(), n_rows=len(d), MAPE=100 * np.mean(np.abs(d.p - d.y) / L))
    for mth in ('split', 'norm', 'cqr'):
        lo, hi = d[f'{mth}_lo'], d[f'{mth}_hi']
        out[f'cov_{mth}'] = 100 * np.mean((d.y >= lo) & (d.y <= hi))
        out[f'width_{mth}_pct_life'] = 100 * np.mean((hi - lo) / L)
    return out


oof_all, maps, holds, sel = [], [], [], []
for ds in A.ds.split(','):
    res = os.path.join(ROOT, ds, 'results')
    df, train, test, main, _ = load_stage1(res)
    params = {**DEFAULTS['XGBoost'], **load_json(os.path.join(res, 'final_params.json'))['params'], 'device': DEV, 'random_state': SEED}
    cells = train + test
    life = (df.groupby('cell').cycle.max() + df.groupby('cell').RUL.last()).astype(float)
    M = meta(ds, cells, life)
    folds = [([c for c in cells if c != h], [h]) for h in cells] if ds == 'snl' else cell_folds(df, cells, 0)
    for fs in A.sets.split(','):
        part = os.path.join(OUT, 'parts', f'{ds}_{fs}.pkl')   # resumable across 1-h HPC sessions
        if os.path.exists(part):
            o, m_, h_, s_ = pd.read_pickle(part)
            oof_all.append(o); maps += m_; holds += h_; sel += s_
            print(ds, fs, 'loaded from', part, flush=True)
            continue
        n_maps, n_holds, n_sel = len(maps), len(holds), len(sel)
        feats = main if fs == 'E40' else TOP7
        oof = pd.concat([fit_predict(df, feats, tr, te, params, life[tr].mean()) for tr, te in folds])
        oof['ds'], oof['set'] = ds, fs
        frac = oof.cycle / oof.cell.map(life)
        oof['stage'] = pd.cut(frac, [-1, 1 / 3, 2 / 3, 9], labels=['early', 'mid', 'late']).astype(str)
        oof['ood_bin'] = pd.cut(oof.ood, [-1, 1, 1.5, np.inf], labels=['in', 'edge', 'out']).astype(str)
        oof_all.append(oof)
        maps.append(dict(ds=ds, set=fs, factor='all', level='all', **summarise(oof, life)))
        for f in list(M.columns):
            for lev, g in oof.groupby(oof.cell.map(M[f])):
                maps.append(dict(ds=ds, set=fs, factor=f, level=lev, **summarise(g, life)))
        for f in ('stage', 'ood_bin'):
            for lev, g in oof.groupby(f):
                maps.append(dict(ds=ds, set=fs, factor=f, level=lev, **summarise(g, life)))
        # D. selective prediction
        w = (oof.norm_hi - oof.norm_lo) / oof.cell.map(life)
        for rank_by, score in [('width', w), ('ood', oof.ood)]:
            for keep in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                g = oof[score <= score.quantile(keep)]
                sel.append(dict(ds=ds, set=fs, rank_by=rank_by, keep=keep, **summarise(g, life)))
        # C. factor hold-out: train on the other levels, test on this one
        for f in [c for c in M.columns if c not in ('life', 'fade_accel', 'knee_pos')] + ['life']:
            for lev in M[f].unique():
                te = list(M.index[M[f] == lev])
                tr = [c for c in cells if c not in te]
                if len(te) < 1 or len(tr) < 4:
                    continue
                d = fit_predict(df, feats, tr, te, params, life[tr].mean())
                ref = oof[oof.cell.isin(te)]
                holds.append(dict(ds=ds, set=fs, factor=f, held_out=lev, n_train_cells=len(tr),
                                  **{f'{k}': v for k, v in summarise(d, life).items()},
                                  MAPE_in_distribution=summarise(ref, life)['MAPE'],
                                  cov_norm_in_distribution=summarise(ref, life)['cov_norm']))
        os.makedirs(os.path.dirname(part), exist_ok=True)
        pd.to_pickle((oof, maps[n_maps:], holds[n_holds:], sel[n_sel:]), part)
        print(ds, fs, 'done', flush=True)

oof_all, maps, holds, sel = [], [], [], []   # tables from every finished part, not only this session's datasets
for f in sorted(os.listdir(os.path.join(OUT, 'parts'))):
    o, m_, h_, s_ = pd.read_pickle(os.path.join(OUT, 'parts', f))
    oof_all.append(o); maps += m_; holds += h_; sel += s_
pd.concat(oof_all).to_csv(os.path.join(OUT, 'oof.csv.gz'), index=False)
for name, rows in [('map', maps), ('holdout', holds), ('selective', sel)]:
    pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
print(pd.DataFrame(maps).query("factor in ('all','stage','ood_bin')").round(1).to_string(index=False))
print(pd.DataFrame(holds).round(1).to_string(index=False))
