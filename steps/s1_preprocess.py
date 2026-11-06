"""Step 1: load features, causal cleaning + trend features, lock the 70/30 cell split.

Writes features_v2.pkl, features.json (main = 40 features both datasets have; all = every
informative one), split.json.
"""
import glob
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from rul.common import SEED, SMOKE, TEMP_FEATURES, load_json, save_json, setup
from rul.preprocess import assert_causal, clean_trend_causal

ds, cfg, RES, FIG = setup()
raw, cell_order = cfg.load_raw()
print(f'{ds}: {raw.cell.nunique()} cells, {len(raw)} cycles')

assert_causal(raw[raw.cell == cell_order[0]])
print('causality check passed')

t0 = time.time()
df = clean_trend_causal(raw)
info = [c for c in df.columns if c not in ('cell', 'cycle', 'RUL') and df[c].nunique() > 1]
main = [c for c in info if c not in TEMP_FEATURES]
assert len(main) == 40, main
print(f'preprocessed in {time.time() - t0:.0f}s: main set {len(main)} features, all informative {len(info)}')

cells = list(cell_order)[:25] if SMOKE else list(cell_order)
strata = getattr(cfg, 'STRATA', None)
if strata and not SMOKE:   # every cycling condition in both sets
    train, test = train_test_split(cells, train_size=0.7, stratify=[strata[c] for c in cells], random_state=SEED)
else:                      # MATR/HUST: same split as their notebooks
    np.random.seed(SEED)
    np.random.shuffle(cells)
    n_tr = int(0.7 * len(cells))
    train, test = cells[:n_tr], cells[n_tr:]
print(f'train cells {len(train)} | test cells {len(test)} (locked until step 4)')
old = os.path.join(RES, 'split.json')
if os.path.exists(old) and load_json(old) != {'train': train, 'test': test}:
    for f in glob.glob(os.path.join(RES, 'bench_*.pkl')):   # step-2 cache belongs to the old split
        os.remove(f)
    print('split changed: cleared the step-2 cache')

df.to_pickle(os.path.join(RES, 'features_v2.pkl'))
save_json({'main': main, 'all': info}, os.path.join(RES, 'features.json'))
save_json({'train': train, 'test': test}, os.path.join(RES, 'split.json'))
