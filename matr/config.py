"""MATR with b1_cell_0..4 joined to their batch-2 continuations (b2c7, 8, 9, 15, 16), as Severson's loader (step 28).

Features: 42 per-cycle features from features/multi_dataset.py (port of the MATR feature notebook).
"""
import os
import pickle

from rul.common import to_frame
from rul.paths import FEATURES

SOURCE = os.path.join(FEATURES, 'all_data_42features_merged.pkl')   # features.multi_dataset matr, then steps.s28_matr_merge build
BEST_ALGO = 'XGBoost'   # step 2: the only algorithm tied with the best on all four datasets


def load_raw():
    """-> (frame with cell, cycle, 42 features, RUL; cell order used for the 70/30 split)."""
    ref = pickle.load(open(SOURCE, 'rb'))
    return to_frame(ref), list(ref)     # notebook order -> same split as the MATR notebook
