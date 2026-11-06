"""Step 3: tune the best algorithm on the training cells.

Random search (TUNE_SEED folds) with the default settings as candidate 0, so a tuned setting is only
picked if it beats the default. The pick is re-checked against the default on 15 fresh folds
(CONFIRM_SEEDS) and kept only if it still wins there.
Writes tuning_search.csv, final_params.json.
"""
import json
import os
import random

import numpy as np
import pandas as pd
from scipy import stats

from rul.common import CONFIRM_SEEDS, SEED, SMOKE, TUNE_SEED, load_json, load_stage1, save_json, setup
from rul.models import SPACES, cv_run

N_TUNE = 3 if SMOKE else 30

ds, cfg, RES, FIG = setup()
df, TRAIN, TEST, FEATURES, _ = load_stage1(RES)
bench = load_json(os.path.join(RES, 'benchmark.json'))
algo = cfg.BEST_ALGO or bench['winner']
print(f'{ds}: tuning {algo}' + (' (config override)' if cfg.BEST_ALGO else f' (CV winner; tied: {bench["tied"]})'))

rng = random.Random(SEED)
candidates = [{}] + [{k: rng.choice(v) for k, v in SPACES[algo].items()} for _ in range(N_TUNE)]
rows = []
for i, cand in enumerate(candidates):
    fr2, _ = cv_run(df, TRAIN, FEATURES, algo, cand, TUNE_SEED)
    rows.append({'candidate': i, 'params': json.dumps(cand, default=str) if cand else 'default',
                 'cv_r2': fr2.mean(), 'cv_r2_sd': fr2.std(ddof=1)})
    print(f'[{i:2d}/{N_TUNE}] R2={fr2.mean():.4f}  {rows[-1]["params"]}', flush=True)
search = pd.DataFrame(rows).sort_values('cv_r2', ascending=False)
search.to_csv(os.path.join(RES, 'tuning_search.csv'), index=False)
best = candidates[int(search.iloc[0]['candidate'])]

confirm = None
if best:
    d = np.concatenate([cv_run(df, TRAIN, FEATURES, algo, {}, s)[0] for s in CONFIRM_SEEDS])
    t = np.concatenate([cv_run(df, TRAIN, FEATURES, algo, best, s)[0] for s in CONFIRM_SEEDS])
    confirm = {'default_r2': d.mean(), 'default_sd': d.std(ddof=1), 'tuned_r2': t.mean(),
               'tuned_sd': t.std(ddof=1), 'p_paired_t': stats.ttest_rel(t, d).pvalue, 'n_folds': len(d)}
    print(f'confirmation ({len(d)} fresh folds): default {d.mean():.4f} | tuned {t.mean():.4f} | '
          f'p = {confirm["p_paired_t"]:.4f}')
    if t.mean() <= d.mean():
        best = {}
print(f'FINAL: {algo} with ' + (json.dumps(best, default=str) if best else 'default settings'))
save_json({'algo': algo, 'params': best, 'confirmation': confirm}, os.path.join(RES, 'final_params.json'))
