"""Causal cleaning + trend features (MATR notebook "Improvements 1, 1b, 2", past cycles only).

Changes vs the notebook, all to remove look-ahead:
  * median window k-4..k (was centred k-2..k+2)
  * gaps forward-filled (was the whole-cell median)
  * DCIR_drift baseline at cycle k < 5 uses cycles 1..k (was cycles 1..5)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

CLEANED = ('Discharge_capacity', 'DCIR_median', 'DCIR_std')


def clean_trend_causal(df):
    out = []
    for _, g in df.groupby('cell', sort=False):
        g = g.sort_values('cycle').copy()
        orig_dcap, orig_ce = g['Discharge_capacity'].to_numpy(float), g['CE'].to_numpy(float)

        for f in CLEANED:
            s = pd.Series(g[f].to_numpy(float)).ffill()          # gap -> last past value
            lead = s.isna().to_numpy()                             # before the first measurement
            if (~lead).sum() < 5:
                continue
            v = median_filter(s.fillna(s.dropna().iloc[0]).to_numpy(), size=5, mode='nearest',
                              origin=2)                            # origin=2 -> window k-4..k
            v[lead] = np.nan
            g[f] = v

        cap = g['Discharge_capacity'].to_numpy(float)
        g['cycle_index'] = g['cycle'].astype(float)
        g['Cum_Delta_Q'] = cap - cap[0]
        g['Delta_Q_cycle'] = np.r_[0.0, np.diff(cap)]
        for f in ('Discharge_capacity', 'V_std'):                  # expanding-window slope
            v = g[f].to_numpy(float)
            g[f + '_slope'] = [float(np.polyfit(np.arange(k + 1, dtype=float), v[:k + 1], 1)[0])
                               if k >= 2 else 0.0 for k in range(len(v))]

        ok = ~np.isnan(orig_dcap) & ~np.isnan(orig_ce) & (orig_ce != 0) & (orig_dcap != 0)
        g.loc[ok, 'CE'] = cap[ok] / (orig_dcap[ok] / orig_ce[ok])  # same charge capacity, cleaned discharge

        d = g['DCIR_median'].to_numpy(float)
        g['DCIR_min_so_far'] = np.fmin.accumulate(d)
        base = [np.nanmedian(d[:min(k + 1, 5)]) if np.isfinite(d[:min(k + 1, 5)]).any() else np.nan
                for k in range(len(d))]
        g['DCIR_drift'] = d - np.array(base)
        out.append(g)
    return pd.concat(out, ignore_index=True)


REL_BASE = 10     # baseline = median of a cell's first 10 cycles (expanding before cycle 10: causal)
REL_RATIO = ('Discharge_capacity', 'Qc', 'Charge_time', 'CC_length', 'CV_length', 'CMT', 'VMT',
             'V_std', 'IC_max', 'DVA_mean', 'DCIR_median')
REL_DIFF = ('CV_frac', 'V_mean', 'CE')


def add_relative(df):
    """Protocol-normalised features: each cell against its own early baseline (change vs early cycles
    transfers across chemistries/protocols: Paulson 2022, Li 2023, Ma 2022).
    x_rel = x / baseline, x_d = x - baseline; plus the mean fade rate so far and a 50-cycle fade slope."""
    out = []
    for _, g in df.groupby('cell', sort=False):
        g = g.sort_values('cycle').copy()
        g['Qc'] = g['Discharge_capacity'] / g['CE']
        g['CV_frac'] = g['CV_length'] / (g['CC_length'] + g['CV_length'])
        for f in REL_RATIO + REL_DIFF:
            b = g[f].expanding().median().to_numpy()
            b[REL_BASE:] = b[min(REL_BASE, len(b)) - 1]
            if f in REL_RATIO:
                g[f + '_rel'] = g[f].to_numpy() / np.where(b != 0, b, np.nan)
            else:
                g[f + '_d'] = g[f].to_numpy() - b
        q, x = g['Discharge_capacity_rel'], g['cycle_index']
        g['fade_rate'] = (1 - q) / x
        g['Q_rel_slope50'] = (x.rolling(50, min_periods=3).cov(q) / x.rolling(50, min_periods=3).var()).fillna(0.0)
        out.append(g.drop(columns=['Qc', 'CV_frac']))
    return pd.concat(out, ignore_index=True).replace([np.inf, -np.inf], np.nan)


REL_FEATURES = [f + '_rel' for f in REL_RATIO] + [f + '_d' for f in REL_DIFF] + ['fade_rate', 'Q_rel_slope50']


def assert_causal(cell_df, n_future=3):
    """Changing a cell's last `n_future` cycles must not change any earlier preprocessed row."""
    changed = cell_df.copy()
    rows = changed.index[-n_future:]
    for f in (*CLEANED, 'CE', 'V_std'):
        changed.loc[rows, f] = changed.loc[rows, f] * 1.5
    a = clean_trend_causal(cell_df).drop(columns='cell').iloc[:-n_future].to_numpy(float)
    b = clean_trend_causal(changed).drop(columns='cell').iloc[:-n_future].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True), 'preprocessing leaks future cycles into past rows'
