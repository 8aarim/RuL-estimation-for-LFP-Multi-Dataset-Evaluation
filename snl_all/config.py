"""SNL with all 18 LFP cells. 9 reach the 80% EOL (measured, BatteryLife labels); the other 9 stop at
83-88% SOH, so their EOL is EXTRAPOLATED: linear fit of SOH (25-cycle rolling median of regular-cycle
discharge capacity / 1.1 Ah) over each cell's last 500 cycles, extended to 0.80. That reaches 1.2-2.2x past
the last measured cycle and ignores any late-life knee, so those lives are likely overestimated.
Split is stratified measured/extrapolated; step 4 also scores the measured test cells alone (the honest number).
Not in DATASETS (steps 7-8 keep the 9-cell `snl`). Features: snl/extract.py (python -m snl.extract).
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'snl')
BEST_ALGO = 'XGBoost'
NOMINAL, EOL_SOH, FIT_LAST = 1.1, 0.80, 500   # ponytail: linear extrapolation; a knee-aware fit is the upgrade


def _extrapolated_eol(g):
    g = g.sort_values('cycle')
    soh = g.Discharge_capacity.rolling(25, min_periods=1).median() / NOMINAL
    a, b = np.polyfit(g.cycle.tail(FIT_LAST), soh.tail(FIT_LAST), 1)
    assert a < 0, f'{g.cell.iloc[0]}: capacity not fading, EOL cannot be extrapolated'
    return int(round((EOL_SOH - b) / a))


cen = pd.read_csv(os.path.join(HERE, 'features_snl_censored.csv'))
EXTRAPOLATED = {c: _extrapolated_eol(g) for c, g in cen.groupby('cell')}   # cell -> estimated EOL cycle


def load_raw():
    """-> (frame with cell, cycle, 42 features, RUL; cell order used for the 70/30 split)."""
    lab = pd.read_csv(os.path.join(HERE, 'features_snl.csv'))
    c = cen.copy()
    c['RUL'] = c.cell.map(EXTRAPOLATED) - c.cycle
    raw = pd.concat([lab, c], ignore_index=True).sort_values(['cell', 'cycle'], ignore_index=True)
    return raw, list(dict.fromkeys(raw.cell))


STRATA = {c: 'extrapolated' if c in EXTRAPOLATED else 'measured' for c in load_raw()[1]}
