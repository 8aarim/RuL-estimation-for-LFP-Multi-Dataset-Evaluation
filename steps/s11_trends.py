"""Step 11 (data only): degradation trends across the four datasets, oldest campaign -> newest.
Writes compare/trends/{cells.csv, median_curves.csv, stress.csv, early.csv, resistance.csv, summary.json}.
Needs compare/trends/uconn_rpt.csv (python -m steps.uconn_rpt_curves) and compare/data_inventory/cells.csv (s10).
    python -m steps.s11_trends
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from rul.common import ROOT

OUT = os.path.join(ROOT, 'compare', 'trends')
NOM = {'MATR': 1.1, 'SNL': 1.1, 'HUST': 1.1, 'UConn': 1.2}
BAD = {f'b1_cell_{i}' for i in range(5)}        # MATR cells stopped at 88-95% SOH but labelled EOL
GRID = np.linspace(0, 1, 21)


def curves():
    """{dataset: {cell: (cycle, SOH, life)}}; MATR/SNL/HUST from per-cycle discharge capacity, UConn from C/3 RPT."""
    out = {}
    for ds in ('matr', 'snl', 'hust'):
        df = pd.read_pickle(os.path.join(ROOT, ds, 'results', 'features_v2.pkl'))
        name = ds.upper()
        out[name] = {c: (g.cycle.to_numpy(float), g.Discharge_capacity.to_numpy(float) / NOM[name], float((g.RUL + g.cycle).max()))
                     for c, g in df.sort_values('cycle').groupby('cell') if c not in BAD}
    r = pd.read_csv(os.path.join(OUT, 'uconn_rpt.csv'), dtype={'cell': str})
    u = pd.read_pickle(os.path.join(ROOT, 'uconn', 'results', 'features_v2.pkl'))
    life = (u.RUL + u.cycle).groupby(u.cell).max()
    out['UConn'] = {c: (g.cycle.to_numpy(float), g.capacity_Ah.to_numpy(float) / 1.2, float(life[c]))
                    for c, g in r.sort_values('cycle').groupby('cell')}
    return out


def slope(x, y, lo, hi):
    """Fade rate (% SOH per 100 cycles) from a straight-line fit on lo <= x <= hi."""
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    return -np.polyfit(x[m], y[m], 1)[0] * 1e4 if m.sum() >= 3 else np.nan


def knee(x, y):
    """Two-line (hinge) least-squares fit over a breakpoint grid; -> breakpoint cycle (second line steeper)."""
    m = np.isfinite(y)
    x, y = x[m], y[m]
    best = (np.inf, np.nan)
    for b in np.linspace(x[0] + 0.1 * (x[-1] - x[0]), x[-1] - 0.05 * (x[-1] - x[0]), 60):
        X = np.c_[np.ones_like(x), x, np.maximum(x - b, 0)]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = float(np.sum((X @ beta - y) ** 2))
        if sse < best[0] and beta[2] < 0:
            best = (sse, b)
    return best[1]


C = curves()
rows, med = [], []
for ds, cells in C.items():
    grid = []
    for c, (x, y, life) in cells.items():
        y0 = np.nanmedian(y[:10]) if ds != 'UConn' else y[0]
        keep = x <= life
        x, y = x[keep], y[keep]
        k = knee(x, y)
        rows.append(dict(dataset=ds, cell=c, life=life, SOH0=y0,
                         r_0_20=slope(x, y, 0, 0.2 * life), r_40_60=slope(x, y, 0.4 * life, 0.6 * life),
                         r_80_100=slope(x, y, 0.8 * life, life), knee_frac=k / life,
                         avg_rate=(y0 - 0.8) / life * 1e4))
        grid.append(np.interp(GRID, x / life, y))
    g = np.array(grid)
    for i, t in enumerate(GRID):
        med.append(dict(dataset=ds, life_frac=round(t, 2), median=np.median(g[:, i]),
                        q25=np.percentile(g[:, i], 25), q75=np.percentile(g[:, i], 75)))
cells = pd.DataFrame(rows)
cells['accel'] = cells.r_80_100 / cells.r_0_20
cells.to_csv(os.path.join(OUT, 'cells.csv'), index=False)
pd.DataFrame(med).to_csv(os.path.join(OUT, 'median_curves.csv'), index=False)
summ = cells.groupby('dataset').agg(n=('cell', 'size'), life_med=('life', 'median'), life_q25=('life', lambda s: s.quantile(.25)),
                                    life_q75=('life', lambda s: s.quantile(.75)), life_cv=('life', lambda s: s.std() / s.mean()),
                                    SOH0_med=('SOH0', 'median'), r_0_20=('r_0_20', 'median'), r_40_60=('r_40_60', 'median'),
                                    r_80_100=('r_80_100', 'median'), accel=('accel', 'median'), knee=('knee_frac', 'median'),
                                    knee_q25=('knee_frac', lambda s: s.quantile(.25)), knee_q75=('knee_frac', lambda s: s.quantile(.75)),
                                    avg_rate=('avg_rate', 'median'))
print(summ.round(3).to_string())
kw = stats.kruskal(*[g.life for _, g in cells.groupby('dataset')])
print('Kruskal-Wallis life', kw)

# stress factors vs life (test conditions from s10)
inv = pd.read_csv(os.path.join(ROOT, 'compare', 'data_inventory', 'cells.csv'), dtype={'cell': str})
inv = inv[inv.status.str.startswith('used') & ~inv.cell.isin(BAD)].dropna(subset=['cycle_life'])
st = []
for ds, fac in (('MATR', 'charge_C'), ('HUST', 'discharge_C'), ('UConn', 'charge_C'),
                ('UConn', 'discharge_C'), ('UConn', 'DoD'), ('SNL', 'discharge_C'), ('SNL', 'temperature_C')):
    d = inv[inv.dataset == ds].dropna(subset=[fac])
    rho, p = stats.spearmanr(d[fac], d.cycle_life)
    st.append(dict(dataset=ds, factor=fac, n=len(d), rho=rho, p=p, lo=d[fac].min(), hi=d[fac].max()))
st = pd.DataFrame(st)
st.to_csv(os.path.join(OUT, 'stress.csv'), index=False)
print(st.round(4).to_string())
u = inv[inv.dataset == 'UConn']
ols = sm.OLS(np.log(u.cycle_life), sm.add_constant(u[['charge_C', 'discharge_C', 'DoD']])).fit()
print(ols.summary().tables[1], 'R2', ols.rsquared)
# a 60% DoD cycle moves 60% of the charge: compare in equivalent full cycles (EFC = cycles x DoD) too
u = u.assign(efc=u.cycle_life * u.DoD)
ole = sm.OLS(np.log(u.efc), sm.add_constant(u[['charge_C', 'discharge_C', 'DoD']])).fit()
print(ole.summary().tables[1], 'R2 EFC', ole.rsquared)
print('Spearman EFC vs DoD', stats.spearmanr(u.DoD, u.efc))
m = inv[inv.dataset == 'MATR'].dropna(subset=['charge_C'])
olm = sm.OLS(np.log(m.cycle_life), sm.add_constant(m[['charge_C']])).fit()
print('MATR log-life slope per C', olm.params.to_dict(), olm.pvalues.to_dict(), olm.rsquared)
# UConn group means for the plot
ug = u.groupby(['charge_C', 'discharge_C', 'DoD']).agg(life_med=('cycle_life', 'median'), life_min=('cycle_life', 'min'),
                                                    life_max=('cycle_life', 'max'), efc_med=('efc', 'median'), n=('cell', 'size')).reset_index()
ug.to_csv(os.path.join(OUT, 'uconn_groups.csv'), index=False)
print(ug.to_string())

# early-life indicator: capacity change cycle 10 -> 100 vs life (capacity-only analogue of Severson et al.)
ear = []
for ds in ('MATR', 'SNL', 'HUST'):
    for c, (x, y, life) in C[ds].items():
        q10, q100 = np.interp([10, 100], x, y)
        ear.append(dict(dataset=ds, cell=c, life=life, dQ_10_100=(q100 - q10) * NOM[ds]))
for c, (x, y, life) in C['UConn'].items():     # RPT at cycle 0 vs interpolated at cycle 100
    ear.append(dict(dataset='UConn', cell=c, life=life, dQ_10_100=(np.interp(100, x, y) - y[0]) * 1.2))
ear = pd.DataFrame(ear)
ear.to_csv(os.path.join(OUT, 'early.csv'), index=False)
er = {ds: dict(zip(('rho', 'p'), stats.spearmanr(g.dQ_10_100, g.life))) for ds, g in ear.groupby('dataset')}
print(er)

# resistance and coulombic efficiency: first 10% of life vs last 10%
rc = []
for ds in ('matr', 'snl', 'hust', 'uconn'):
    df = pd.read_pickle(os.path.join(ROOT, ds, 'results', 'features_v2.pkl'))
    df = df[~df.cell.isin(BAD)]
    df = df.assign(f=df.cycle / (df.RUL + df.cycle))
    for c, g in df.groupby('cell'):
        e, l = g[g.f <= 0.1], g[g.f >= 0.9]
        rc.append(dict(dataset='UConn' if ds == 'uconn' else ds.upper(), cell=c,
                       dcir_rise=(l.DCIR_median.median() / e.DCIR_median.median() - 1) * 100,
                       ce_early=e.CE.median(), ce_late=l.CE.median()))
rc = pd.DataFrame(rc)
rc.to_csv(os.path.join(OUT, 'resistance.csv'), index=False)
rcs = rc.groupby('dataset').agg(n=('cell', 'size'), dcir_rise=('dcir_rise', 'median'), dcir_q25=('dcir_rise', lambda s: s.quantile(.25)),
                                dcir_q75=('dcir_rise', lambda s: s.quantile(.75)), ce_early=('ce_early', 'median'), ce_late=('ce_late', 'median'))
print(rcs.round(4).to_string())
json.dump(dict(summary=summ.reset_index().to_dict('records'), kruskal=dict(H=kw.statistic, p=kw.pvalue),
               uconn_ols_efc=dict(params=ole.params.to_dict(), p=ole.pvalues.to_dict(), ci=ole.conf_int().values.tolist(), r2=ole.rsquared),
               uconn_ols=dict(params=ols.params.to_dict(), p=ols.pvalues.to_dict(), ci=ols.conf_int().values.tolist(), r2=ols.rsquared, n=int(ols.nobs)),
               matr_ols=dict(params=olm.params.to_dict(), p=olm.pvalues.to_dict(), r2=olm.rsquared, n=int(olm.nobs)),
               early=er, resistance=rcs.reset_index().to_dict('records')),
          open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
