"""Do SNL's 9 censored cells (test stopped before 80%) improve RUL on the 9 labelled cells?

    python -m snl.extract     # writes features_snl.csv + features_snl_censored.csv
    python -m snl.censored    # after run_all.py snl (needs snl/results/features.json)

Censored cells only bound RUL from below (EOL > last cycle), so they enter through CatBoost's
survival loss (SurvivalAft: label = [lower, upper], upper = -1 for 'not reached'). Evaluation is
always on labelled cells only: cell-wise CV over the 9 labelled cells, censored cells added to every
training fold, 4 fold seeds. Variants separate the loss from the extra data:
  rmse          current loss, labelled cells only (the step-4 'all cells CV' model, default params)
  rmse + naive  censored cells with RUL = lower bound as if exact (biased low: shows why not)
  aft           survival loss, labelled cells only
  aft + cens    survival loss, labelled + censored cells
Survival predictions are capped at 2x the longest life seen in training (they can explode).
Writes snl/results/censored_cv.csv.
"""
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from rul.common import CONFIRM_SEEDS, CV_SEED, ROOT, SEED, build, cell_folds, load_json, metrics, prep
from rul.preprocess import clean_trend_causal

HERE = os.path.join(ROOT, 'snl')
BASE = dict(iterations=500, depth=6, learning_rate=0.03, random_seed=SEED, verbose=0)   # step-2 CatBoost defaults
AFT = 'SurvivalAft:dist=Normal;scale=1.2'   # catboost default AFT; ponytail: dist/scale untuned


def fit_predict(variant, Xa, lo, hi, Xb, cap):
    if variant.startswith('rmse'):
        return CatBoostRegressor(**BASE).fit(Xa, lo).predict(Xb)
    m = CatBoostRegressor(**BASE, loss_function=AFT).fit(Xa, np.c_[lo, hi])
    # predictions are exp(log-life): on a few rows they explode (seen: >1e40); cap at 2x the longest
    # life seen in training. ponytail: blunt physical bound, a tuned dist/scale is the upgrade
    return np.minimum(m.predict(Xb, prediction_type='Exponent'), cap)


lab = clean_trend_causal(pd.read_csv(os.path.join(HERE, 'features_snl.csv')))
cen = clean_trend_causal(pd.read_csv(os.path.join(HERE, 'features_snl_censored.csv')))
feats = load_json(os.path.join(HERE, 'results', 'features.json'))['main']
cells = list(dict.fromkeys(lab.cell))
Xc, yc, _ = build(cen, list(dict.fromkeys(cen.cell)), feats)

rows = []
for variant in ('rmse', 'rmse + naive', 'aft', 'aft + cens'):
    use_cens = variant in ('rmse + naive', 'aft + cens')
    oof, fold_r2 = [], []
    for seed in (CV_SEED, *CONFIRM_SEEDS):
        for tr, te in cell_folds(lab, cells, seed):
            Xa, ya, sa = build(lab, tr, feats)
            Xb, yb, sb = build(lab, te, feats)
            lo, hi = ya, ya.copy()
            cap = 2 * (sa.RUL + sa.cycle).max()
            if use_cens:
                Xa, lo, hi = np.vstack([Xa, Xc]), np.r_[lo, yc], np.r_[hi, np.full(len(yc), -1.0)]
                cap = max(cap, 2 * (cen.RUL + cen.cycle).max())
            Xa, Xb = prep(Xa, Xb)
            p = fit_predict(variant, Xa, lo, hi, Xb, cap)
            fold_r2.append(metrics(yb, p)['R2'])
            if seed == CV_SEED:
                oof.append(pd.DataFrame({'cell': sb.cell.values, 'cycle': sb.cycle.values, 'y': yb, 'p': p}))
    o = pd.concat(oof)
    rows.append({'variant': variant, 'fold_R2': np.mean(fold_r2), 'fold_R2_sd': np.std(fold_r2, ddof=1),
                 **metrics(o.y.to_numpy(), o.p.to_numpy(), o)})
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, 'results', 'censored_cv.csv'), index=False)
print(out.round(4).to_string(index=False))
