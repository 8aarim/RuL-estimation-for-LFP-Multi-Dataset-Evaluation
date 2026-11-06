"""Step 4: final base model -- fit on training cells, scored ONCE on the locked test cells.

Also the literature-comparable lifetime error at cycle 100 (most papers predict life once from 100 cycles).
Writes final_test.json, final_test_segments.csv, final_test_predictions.csv, fig_final_parity.png.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul.common import CV_SEED, build, cell_bootstrap_ci, load_json, load_stage1, metrics, prep, save_json, segment_report, setup
from rul.models import cv_run, fit_model

ds, cfg, RES, FIG = setup()
df, TRAIN, TEST, FEATURES, _ = load_stage1(RES)
fp = load_json(os.path.join(RES, 'final_params.json'))
algo, params = fp['algo'], fp['params']

Xtr, ytr, _ = build(df, TRAIN, FEATURES)
Xte, yte, ste = build(df, TEST, FEATURES)
Xtr, Xte = prep(Xtr, Xte)
yp = fit_model(algo, params, Xtr, ytr).predict(Xte)

m = metrics(yte, yp, ste)
lo, hi = cell_bootstrap_ci(yte, yp, ste['cell'].to_numpy())
# literature-comparable number: whole-life prediction (p + cycle) once per cell, at its first cycle >= 100
early = pd.DataFrame({'cell': ste['cell'].values, 'cycle': ste['cycle'].values, 'life': yte + ste['cycle'].values,
                      'pred': yp + ste['cycle'].values})
early = early[early.cycle >= 100].groupby('cell').first()
e = early.pred - early.life
extra = {'at_cycle_100': {'n_cells': len(early), 'RMSE': float(np.sqrt((e ** 2).mean())), 'MAE': float(e.abs().mean()),
                          'MAPE': float((e.abs() / early.life).mean()), 'mean_life': float(early.life.mean())}}
print(f'{ds}: lifetime predicted at cycle 100 ({len(early)} test cells): RMSE = {extra["at_cycle_100"]["RMSE"]:.0f}  '
      f'MAE = {extra["at_cycle_100"]["MAE"]:.0f}  MAPE = {extra["at_cycle_100"]["MAPE"]:.1%}')
if len(TRAIN) + len(TEST) < 20:   # a few test cells can't carry the result: also CV over every cell
    _, oof = cv_run(df, TRAIN + TEST, FEATURES, algo, params, CV_SEED)
    extra['all_cells_cv'] = metrics(oof['y'].to_numpy(), oof['p'].to_numpy(), oof)
    print(f'{ds}: {len(TRAIN) + len(TEST)} cells only -> CV over all cells: ' +
          '  '.join(f'{k} = {v:.4g}' for k, v in extra['all_cells_cv'].items()))
est = getattr(cfg, 'EXTRAPOLATED', {})   # cells whose EOL is estimated: their error is vs. an estimate, not truth
if est:
    ok = ~ste['cell'].isin(est).to_numpy()
    extra['test_measured_cells'] = metrics(yte[ok], yp[ok], ste[ok])
    print(f'{ds}: test cells with MEASURED EOL only ({ste.cell[ok].nunique()}): ' +
          '  '.join(f'{k} = {v:.4g}' for k, v in extra['test_measured_cells'].items()))
    if 'all_cells_cv' in extra:
        okc = ~oof['cell'].isin(est)
        extra['all_cells_cv_measured'] = metrics(oof.y[okc].to_numpy(), oof.p[okc].to_numpy(), oof[okc])
        print(f'{ds}: CV over all cells, scored on MEASURED cells only: ' +
              '  '.join(f'{k} = {v:.4g}' for k, v in extra['all_cells_cv_measured'].items()))
save_json({'algo': algo, 'params': params, 'n_features': len(FEATURES), **m, 'R2_lo': lo, 'R2_hi': hi, **extra},
          os.path.join(RES, 'final_test.json'))
pd.DataFrame({'cell': ste['cell'], 'cycle': ste['cycle'], 'y': yte, 'p': yp}).to_csv(
    os.path.join(RES, 'final_test_predictions.csv'), index=False)
seg = segment_report(yte, yp)
seg.to_csv(os.path.join(RES, 'final_test_segments.csv'), index=False)
print(f'{ds}: {algo}, {len(FEATURES)} features, {len(TEST)} test cells: R2 = {m["R2"]:.4f} [{lo:.3f}, {hi:.3f}]  '
      f'RMSE = {m["RMSE"]:.1f} ({m["nRMSE"]:.1%} of life)  MAE = {m["MAE"]:.1f} ({m["nMAE"]:.1%})  MAPE = {m["MAPE"]:.1%}')
print(seg.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(3.5, 3.4))
ax.scatter(yte, yp, s=2, alpha=0.25, color='#1f77b4', rasterized=True)
lim = [0, max(yte.max(), yp.max()) * 1.02]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set(xlim=lim, ylim=lim, xlabel='Actual RUL (cycles)', ylabel='Predicted RUL (cycles)')
ax.text(0.04, 0.93, f'{ds.upper()} {algo}  $R^2$ = {m["R2"]:.3f}', transform=ax.transAxes, fontsize=8)
fig.savefig(os.path.join(FIG, 'fig_final_parity.png'))
