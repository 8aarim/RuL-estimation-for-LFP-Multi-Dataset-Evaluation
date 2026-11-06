"""SNL (Preger et al. 2020, via BatteryLife): 18650 LFP/graphite, 1.1 Ah, 15/25/35 C, 0.5C charge,
0.5-3C discharge, 0-100% SOC. Only the 9 cells with a BatteryLife 80%-EOL label are used (the other 9
end at 83-88%); so ~3 test cells: report the all-cell CV from step 4 as the main SNL number.

Features: snl/features_snl.csv written by snl/extract.py (MATR notebook feature code; SNL input
handling and its calibration knobs are listed there). Temperature is logged (missing in ~15% of cycles).
"""
import os

import pandas as pd

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features_snl.csv')   # python -m snl.extract
BEST_ALGO = 'XGBoost'   # step 2: the only algorithm tied with the best on all four datasets


def load_raw():
    """-> (frame with cell, cycle, 42 features, RUL; cell order used for the 70/30 split)."""
    raw = pd.read_csv(SOURCE)
    return raw, list(dict.fromkeys(raw.cell))   # sorted names


STRATA = {c: c.rsplit('_', 1)[0] for c in pd.read_csv(SOURCE, usecols=['cell']).cell.unique()}   # condition, e.g. 25C_0.5-3C
