"""Preprocessing must use past cycles only.  Run:  python -m tests.test_preprocess"""
import numpy as np
import pandas as pd

import os

from rul.preprocess import REL_FEATURES, add_relative, assert_causal, clean_trend_causal


def toy_cell(n=40, seed=0):
    rng = np.random.RandomState(seed)
    d = pd.DataFrame({'cell': 'c1', 'cycle': np.arange(1, n + 1), 'RUL': np.arange(n, 0, -1),
                      'Discharge_capacity': 1.1 - 0.002 * np.arange(n) + rng.normal(0, 0.003, n),
                      'DCIR_median': 0.02 + rng.normal(0, 0.001, n), 'DCIR_std': rng.rand(n) * 1e-3,
                      'CE': 1 - rng.rand(n) * 1e-3, 'V_std': 0.5 + rng.normal(0, 0.01, n)})
    d.loc[[0, 1, 10], 'DCIR_median'] = np.nan          # leading gap + a mid-life gap
    return d


if __name__ == '__main__':
    d = toy_cell()
    assert_causal(d)                                   # future changes don't move past rows

    # window is k-4..k: a spike at cycle 20 can only affect cycles 20..24 of the cleaned signal
    spiked = d.copy()
    spiked.loc[19, 'Discharge_capacity'] += 1.0
    diff = clean_trend_causal(spiked)['Discharge_capacity'].to_numpy() - clean_trend_causal(d)['Discharge_capacity'].to_numpy()
    assert np.all(diff[:19] == 0), 'filter looks ahead'
    assert np.all(diff[24:] == 0), 'filter window longer than 5 cycles'

    out = clean_trend_causal(d)
    assert np.isnan(out['DCIR_median'].iloc[:2]).all(), 'values invented before the first measurement'
    assert {'cycle_index', 'Cum_Delta_Q', 'Discharge_capacity_slope', 'V_std_slope'} <= set(out.columns)

    # relative features: perturbing the last 3 cycles of a real cell must not move earlier rows
    path = os.path.join('matr', 'results', 'features_v2.pkl')
    if os.path.exists(path):
        c = pd.read_pickle(path)
        c = c[c.cell == c.cell.iloc[0]].reset_index(drop=True)
        num = [k for k in c.columns if k not in ('cell', 'cycle', 'RUL', 'cycle_index')]
        c[num] = c[num].astype(float)
        changed = c.copy()
        changed.loc[changed.index[-3:], num] *= 1.5
        a, b = (add_relative(x)[REL_FEATURES].iloc[:-3].to_numpy(float) for x in (c, changed))
        assert np.allclose(a, b, equal_nan=True), 'relative features leak future cycles'
    print('test_preprocess: all checks passed')
