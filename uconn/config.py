"""UConn-ISU-ILCC (Nowacki et al. 2025, doi 10.1016/j.jpowsour.2024.235813): 64 LFP cells
(LithiumWerks APR18650M1B, 1.2 Ah), 11 conditions (charge C-rate, discharge C-rate, DoD), all to 80%.

Features: data/features/features_uconn.csv written by features/multi_dataset.py uconn with the MATR
notebook's extraction code. Differences, all set there:
- cycle = cumulative cycle count (Cycle Number restarts after every weekly RPT);
- EOL = C/3 RPT capacity crossing 0.96 Ah (80% of nominal), interpolated between RPTs (~100 cycles apart);
- partial-DoD cycling: dQ(V) grid 3.17-3.5 V (window every cell covers), so dQV_at_2V = value at 3.17 V;
- CV threshold 3.595 V (as HUST); no temperature -> the six temperature features drop out.
"""
import os

import pandas as pd

from rul.paths import FEATURES

SOURCE = os.path.join(FEATURES, 'features_uconn.csv')   # python -m features.multi_dataset uconn
PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CyclingParameters.csv')   # copy of the dataset's CyclingParameters.csv
# cell -> cycling group (11 conditions); step 1 stratifies the 70/30 split by it. The plain seeded
# shuffle put 5 of group 4's 6 cells in test and none of group 10's.
STRATA = {f'{c:02d}': g for c, g in pd.read_csv(PARAMS, encoding='utf-8-sig')[['Cell ID', 'Group Number']].to_numpy()}
BEST_ALGO = 'XGBoost'   # step 2: the only algorithm tied with the best on all four datasets


def load_raw():
    """-> (frame with cell, cycle, 42 feature columns, RUL; cell order used for the 70/30 split)."""
    raw = pd.read_csv(SOURCE, dtype={'cell': str})
    return raw, list(dict.fromkeys(raw.cell))   # sorted cell ids
