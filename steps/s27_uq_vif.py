"""Step 27: uncertainty quantification and redundancy (Pearson / VIF) on all four datasets, 40 common features.
Not in the paper (exploratory, 2026-10-14).

    python -m steps.s27_uq_vif        -> uq_vif/{uq.csv, uq_stage.csv, vif.csv, pairs.csv, summary.txt}

UQ = the three interval methods of the MATR notebook (cell 34), at a 90% target, with tuned XGBoost:
  * MATR, HUST, UConn: training cells split 80/20 (seed 42) into inner-train / calibration cells;
    the inner model predicts the test cells.
  * SNL (9 cells): leave-one-cell-out; calibration residuals come from an inner leave-one-cell-out over the other 8.
  Methods: Gaussian residual (sigma from calibration residuals), single-pool split conformal (finite-sample
  quantile of pooled calibration |residuals|), cell-stratified conformal (mean of per-cell 90% quantiles).
  'gauss_test_sigma' repeats the notebook's Gaussian interval, whose sigma came from the TEST residuals
  (leaks the test set; kept only to show how much that inflates coverage).
Coverage CI = 95% cell bootstrap. uq_stage.csv = coverage by life third (rows with cycle/life in thirds).
VIF = diag(inverse correlation matrix) of the median-imputed training rows; pairs = |Pearson r| >= 0.85.
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from rul.common import DATASETS, ROOT, build, load_json, load_stage1, prep
from rul.models import DEFAULTS

OUT = os.path.join(ROOT, 'uq_vif')
os.makedirs(OUT, exist_ok=True)
ALPHA, Z, SEED = 0.1, 1.645, 42
DEV = 'cuda' if xgb.build_info().get('USE_CUDA') else 'cpu'
TOP8 = ['Cum_Delta_Q', 'Discharge_capacity_slope', 'Discharge_capacity', 'Charge_time',
        'cycle_index', 'V_std_slope', 'VMT', 'I_std']


def conf_q(res):
    """Finite-sample split-conformal quantile of absolute residuals."""
    n = len(res)
    return float(np.sort(res)[min(int(np.ceil((n + 1) * (1 - ALPHA))) - 1, n - 1)])


def intervals(cal_res, cal_cells):
    """Half-widths of the three methods from signed calibration residuals and their cells."""
    a = np.abs(cal_res)
    per_cell = pd.Series(a).groupby(np.asarray(cal_cells)).quantile(1 - ALPHA)
    return {'gauss_cal': Z * float(np.std(cal_res)), 'split_conformal': conf_q(a), 'cell_stratified': float(per_cell.mean())}


def boot_ci(hit, cells, n=1000):
    rng, cells = np.random.default_rng(SEED), np.asarray(cells)
    u = np.unique(cells)
    g = {c: hit[cells == c] for c in u}
    vals = [np.concatenate([g[c] for c in rng.choice(u, len(u))]).mean() for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


uq, stage, vif_rows, pair_rows = [], [], [], []
for ds in DATASETS:
    res = os.path.join(ROOT, ds, 'results')
    df, train, test, main, _ = load_stage1(res)
    params = {**DEFAULTS['XGBoost'], **load_json(os.path.join(res, 'final_params.json'))['params'], 'device': DEV}
    fit = lambda X, y: xgb.XGBRegressor(**params).fit(X, y)  # noqa: E731
    life = df.groupby('cell').cycle.max()

    if ds != 'snl':
        rng = np.random.default_rng(SEED)
        tc = list(rng.permutation(train))
        n_cal = max(6, int(0.2 * len(tc)))
        cal, inner = tc[-n_cal:], tc[:-n_cal]
        (Xa, ya, _), (Xc, yc, sc), (Xt, yt, st) = build(df, inner, main), build(df, cal, main), build(df, test, main)
        Xa, Xc = prep(Xa, Xc)
        _, Xt = prep(build(df, inner, main)[0], Xt)
        m = fit(Xa, ya)
        cal_res, pt = yc - m.predict(Xc), m.predict(Xt)
        hw = intervals(cal_res, sc['cell'])
        hw['gauss_test_sigma'] = Z * float(np.std(yt - pt))
        y, p, s, n_info = yt, pt, st, f'{len(inner)} inner / {n_cal} cal / {len(test)} test cells'
        HW = {k: np.full(len(y), v) for k, v in hw.items()}
    else:  # leave-one-cell-out with inner leave-one-cell-out calibration
        cells = list(train) + list(test)
        ys, ps, ss, HW = [], [], [], {k: [] for k in ['gauss_cal', 'split_conformal', 'cell_stratified', 'gauss_test_sigma']}
        for c in cells:
            rest = [k for k in cells if k != c]
            cr, cc = [], []
            for k in rest:
                inn = [j for j in rest if j != k]
                (Xa, ya, _), (Xk, yk, _) = build(df, inn, main), build(df, [k], main)
                Xa, Xk = prep(Xa, Xk)
                cr.append(yk - fit(Xa, ya).predict(Xk)); cc += [k] * len(yk)
            cr = np.concatenate(cr)
            (Xa, ya, _), (Xt, yt, st) = build(df, rest, main), build(df, [c], main)
            Xa, Xt = prep(Xa, Xt)
            pt = fit(Xa, ya).predict(Xt)
            hw = intervals(cr, cc)
            for k in ['gauss_cal', 'split_conformal', 'cell_stratified']:
                HW[k].append(np.full(len(yt), hw[k]))
            ys.append(yt); ps.append(pt); ss.append(st)
        y, p, s = np.concatenate(ys), np.concatenate(ps), pd.concat(ss)
        HW = {k: np.concatenate(v) for k, v in HW.items() if v}
        HW['gauss_test_sigma'] = np.full(len(y), Z * float(np.std(y - p)))
        n_info = '9 cells, leave-one-cell-out (inner LOCO calibration)'

    cl = s['cell'].to_numpy()
    mean_life = float(np.mean([life[c] for c in np.unique(cl)]))
    frac = s['cycle'].to_numpy() / np.array([life[c] for c in cl])
    third = np.digitize(frac, [1 / 3, 2 / 3])
    for meth, h in HW.items():
        hit = ((y >= p - h) & (y <= p + h)).astype(float)
        lo, hi = boot_ci(hit, cl)
        uq.append(dict(ds=ds, method=meth, coverage=100 * hit.mean(), cov_ci_lo=100 * lo, cov_ci_hi=100 * hi,
                       width_cycles=float(2 * h.mean()), width_pct_mean_life=float(200 * h.mean() / mean_life),
                       meets_90=hit.mean() >= 0.9, design=n_info))
        for t, name in enumerate(['early', 'mid', 'late']):
            stage.append(dict(ds=ds, method=meth, stage=name, coverage=100 * hit[third == t].mean()))
    print(ds, n_info, flush=True)

    # redundancy on the training rows (SNL: all 9 cells)
    X = pd.DataFrame(build(df, list(train) + (list(test) if ds == 'snl' else []), main)[0], columns=main)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    C = X.corr().to_numpy()
    vif = np.diag(np.linalg.pinv(C))
    for f, v in zip(X.columns, vif):
        vif_rows.append(dict(ds=ds, feature=f, vif=float(v), top8=f in TOP8))
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(C[i, j]) >= 0.85:
                pair_rows.append(dict(ds=ds, a=cols[i], b=cols[j], r=float(C[i, j])))

uq, stage, vif, pairs = map(pd.DataFrame, (uq, stage, vif_rows, pair_rows))
uq.to_csv(os.path.join(OUT, 'uq.csv'), index=False)
stage.to_csv(os.path.join(OUT, 'uq_stage.csv'), index=False)
vif.to_csv(os.path.join(OUT, 'vif.csv'), index=False)
pairs.to_csv(os.path.join(OUT, 'pairs.csv'), index=False)

lines = ['UQ (90% target, 40 features, tuned XGBoost)', uq.round(1).to_string(index=False), '',
         'Coverage by life third', stage.pivot_table(index=['ds', 'method'], columns='stage', values='coverage')[['early', 'mid', 'late']].round(1).to_string(), '',
         'VIF > 10 / > 100 per dataset', vif.groupby('ds').vif.agg(n_gt10=lambda v: int((v > 10).sum()), n_gt100=lambda v: int((v > 100).sum()), n='size').to_string(), '',
         'VIF of the top-8 features', vif[vif.top8].pivot(index='feature', columns='ds', values='vif').round(1).to_string(), '',
         'Pairs with |r| >= 0.85 per dataset', pairs.groupby('ds').size().to_string(), '',
         'Pairs involving top-8 features', pairs[pairs.a.isin(TOP8) | pairs.b.isin(TOP8)].round(3).to_string(index=False)]
open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
