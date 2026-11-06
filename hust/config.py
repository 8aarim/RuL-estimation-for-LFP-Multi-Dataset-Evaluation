"""HUST (Ma et al. 2022, Mendeley nsc7hnsg4s): 77 LFP/graphite 18650 cells, 1.1 Ah (same spec as A123 APR18650M1A; the source does not
state the part number), cycled to 80% at 30 °C.

Features: data/features/features_hust.csv written by features/multi_dataset.py with the MATR
notebook's extraction code (CC/CV split at V_CV = 3.595 V; HUST logs no temperature, so the six
temperature columns are empty and dropped from every feature set).
"""
import os

import pandas as pd

from rul.paths import FEATURES

SOURCE = os.path.join(FEATURES, 'features_hust.csv')   # python -m features.multi_dataset hust
BEST_ALGO = 'XGBoost'   # step 2: the only algorithm tied with the best on all four datasets


def load_raw():
    """-> (frame with cell, cycle, 42 feature columns, RUL; cell order used for the 70/30 split)."""
    raw = pd.read_csv(SOURCE, dtype={'cell': str})
    return raw, list(dict.fromkeys(raw.cell))   # sorted file order -> same split as the HUST notebook
