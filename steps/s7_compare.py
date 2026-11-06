"""Step 7 (after all datasets): which features are best on MATR, HUST and UConn together?

Combined rank = mean of the per-dataset SHAP ranks (same 40 features in all).
Writes compare/ranking_combined.csv, compare/summary.csv.
    python -m steps.s7_compare
"""
import os

import pandas as pd

from rul.common import DATASETS, ROOT, SMOKE, load_json

sfx = '_smoke' if SMOKE else ''
out = os.path.join(ROOT, 'compare' + sfx)
os.makedirs(out, exist_ok=True)

res = {ds: os.path.join(ROOT, ds, 'results' + sfx) for ds in DATASETS}
ranks = {ds: pd.read_csv(os.path.join(r, 'shap_ranking.csv')).set_index('Feature') for ds, r in res.items()}
tab = pd.DataFrame({f'{ds}_rank': r['rank'] for ds, r in ranks.items()})
for ds, r in ranks.items():
    tab[f'{ds}_share'] = r['share']
tab['mean_rank'] = tab[[f'{ds}_rank' for ds in res]].mean(axis=1)
tab['worst_rank'] = tab[[f'{ds}_rank' for ds in res]].max(axis=1)
tab = tab.sort_values(['mean_rank', 'worst_rank'])
tab.to_csv(os.path.join(out, 'ranking_combined.csv'))
print(tab.head(15).round(4).to_string())

rows = []
for ds, r in res.items():
    ft, tk = load_json(os.path.join(r, 'final_test.json')), load_json(os.path.join(r, 'topk.json'))
    rows.append({'dataset': ds, 'algo': ft['algo'], 'tuned': bool(ft['params']), 'test_R2': ft['R2'],
                 'test_RMSE': ft['RMSE'], 'k_star': tk['k_star'], 'k_star_features': ';'.join(tk['k_star_features'])})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out, 'summary.csv'), index=False)
print()
print(summary.round(4).to_string(index=False))
if summary.algo.nunique() > 1:
    print('\nNOTE: different algorithms per dataset; if tied in step 2, set BEST_ALGO in all configs '
          'and rerun steps 3-6 so the rankings come from the same model type.')
