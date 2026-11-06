"""Step 29 (review: trivial and physics baselines, R1-M1 / R2 / R3-M5 / R4). Same split, SNL 9-cell CV, metrics.

    python -m steps.s29_baselines        -> baselines/baselines.csv

Baselines (all causal: cycle k uses cycles <= k only):
  mean_life    RUL = mean life of the training cells - cycle (clipped at 0). Knows nothing about the cell.
  cond_life    as mean_life, but the mean over training cells of the same test condition (MATR charge policy,
               UConn condition, SNL temperature/rate); falls back to mean_life if the condition has no training cell.
  cap_lin      Discharge_capacity (5-cycle rolling median) extrapolated linearly over the last 100 cycles to
               0.8 x nominal; RUL clipped to [0, longest training life]; before cycle 20 uses mean_life.
  cap_exp      same with a fit of log(Q0 - Q) (power-law/exponential fade); falls back to cap_lin when unstable.
  xgb_cycle    XGBoost on cycle_index only.
  xgb_capacity XGBoost on cycle_index + Discharge_capacity + Cum_Delta_Q + Discharge_capacity_slope.
  xgb_E40      reference: tuned XGBoost, 40 features (should reproduce the paper).
UConn note: Discharge_capacity of 60/80 % DoD cells is throughput, not capacity, so cap_* is reported but expected
to fail there (R2-M2); EOL in UConn comes from RPTs.
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from rul.common import DATASETS, ROOT, build, load_json, load_stage1, metrics, prep
from rul.models import DEFAULTS

OUT = os.path.join(ROOT, 'baselines')
os.makedirs(OUT, exist_ok=True)
NOMINAL = {'matr': 1.1, 'hust': 1.1, 'uconn': 1.2, 'snl': 1.1}
DEV = 'cuda' if xgb.build_info().get('USE_CUDA') else 'cpu'
CAP = ['cycle_index', 'Discharge_capacity', 'Cum_Delta_Q', 'Discharge_capacity_slope']


def condition(ds, cell):
    ds = ds.split('_')[0]   # matr_merged -> matr
    if ds == 'matr':
        return json.load(open(os.path.join(ROOT, 'matr', 'results', 'policies.json')))[cell]
    if ds == 'snl':
        return cell.rsplit('_', 1)[0]            # e.g. 25C_0.5-3C
    if ds == 'uconn':
        cfg = __import__('uconn.config', fromlist=['STRATA'])
        return getattr(cfg, 'STRATA', {}).get(cell, cell)
    return cell                                  # HUST: one protocol per cell -> no shared condition


def extrapolate(g, qe, life_cap, fallback, kind):
    q = g.Discharge_capacity.rolling(5, min_periods=1).median().to_numpy()
    c = g.cycle.to_numpy(float)
    out = np.empty(len(g))
    for i in range(len(g)):
        lo = max(0, i - 99)
        if i < 20:
            out[i] = fallback[i]
            continue
        x, y = c[lo:i + 1], q[lo:i + 1]
        r = np.nan
        if kind == 'exp':
            d = q[0] - y
            ok = d > 1e-4
            if ok.sum() >= 10:
                b, a = np.polyfit(x[ok], np.log(d[ok]), 1)
                if b > 0 and q[0] - qe > 0:
                    r = (np.log(q[0] - qe) - a) / b - c[i]
        if not np.isfinite(r):
            b, a = np.polyfit(x, y, 1)
            r = (qe - a) / b - c[i] if b < 0 else life_cap
        out[i] = np.clip(r, 0, life_cap)
    return out


rows = []
for ds in os.environ.get('DS', ','.join(DATASETS)).split(','):   # DS=matr_merged for the merged MATR
    res = os.path.join(ROOT, ds, 'results')
    df, train, test, main, _ = load_stage1(res)
    params = {**DEFAULTS['XGBoost'], **load_json(os.path.join(res, 'final_params.json'))['params'], 'device': DEV}
    life = df.groupby('cell').cycle.max() + df.groupby('cell').RUL.last()   # EOL cycle of each cell
    splits = [(train, test)] if ds != 'snl' else [([c for c in train + test if c != h], [h]) for h in train + test]
    preds = {}
    for tr, te in splits:
        mean_life = life[tr].mean()
        cond_tr = pd.Series({c: condition(ds, c) for c in tr})
        for c in te:
            g = df[df.cell == c].sort_values('cycle')
            base = np.clip(mean_life - g.cycle.to_numpy(), 0, None)
            same = cond_tr[cond_tr == condition(ds, c)].index
            cl = life[same].mean() if len(same) else mean_life
            cap = life[tr].max()   # longest training life: a causal prior, not a free extrapolation
            for name, p in [('mean_life', base), ('cond_life', np.clip(cl - g.cycle.to_numpy(), 0, None)),
                            ('cap_lin', extrapolate(g, 0.8 * NOMINAL[ds.split('_')[0]], cap, base, 'lin')),
                            ('cap_exp', extrapolate(g, 0.8 * NOMINAL[ds.split('_')[0]], cap, base, 'exp'))]:
                preds.setdefault(name, []).append(pd.DataFrame({'cell': c, 'cycle': g.cycle.values, 'y': g.RUL.values, 'p': p}))
        for name, feats in [('xgb_cycle', ['cycle_index']), ('xgb_capacity', CAP), ('xgb_E40', main)]:
            (Xa, ya, _), (Xb, yb, sb) = build(df, tr, feats), build(df, te, feats)
            Xa, Xb = prep(Xa, Xb)
            p = xgb.XGBRegressor(**params).fit(Xa, ya).predict(Xb)
            preds.setdefault(name, []).append(pd.DataFrame({'cell': sb.cell.values, 'cycle': sb.cycle.values, 'y': yb, 'p': p}))
    for name, parts in preds.items():
        d = pd.concat(parts)
        m = metrics(d.y.to_numpy(), d.p.to_numpy(), d[['cell', 'cycle']])
        rows.append(dict(ds=ds, baseline=name, protocol='split' if ds != 'snl' else 'LOCO-9', **m, MAPE_pct=100 * m['MAPE']))
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, f"baselines_{os.environ.get('DS', 'all').replace(',', '_')}.csv"), index=False)
print(out.pivot(index='baseline', columns='ds', values='MAPE_pct').round(2).to_string())
