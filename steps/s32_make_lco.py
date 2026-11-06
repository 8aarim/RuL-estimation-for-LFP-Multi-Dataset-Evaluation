"""Leave-condition-out inputs for s32_tuned_deep.py --cv --policy-cv, one pair per dataset (no re-tuning):
    python -m steps.s32_make_lco   -> configs/lco/policies_<ds>.json, tune_lco_<ds>.csv
Groups: UConn = DoD + charge rate + discharge rate; merged MATR = charge policy; HUST = five discharge-power bands
(GroupKFold needs at least 5 groups, and HUST has one policy per cell).
The tune file is the finished phase B/Bs tuning with its keys renamed, so the same 30-trial settings are reused."""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = os.environ.get('S32_RESULTS', os.path.join(ROOT, 's32_results'))
TUNE = {'uconn': 'uconn_s32', 'matr_merged': 'matr_merged_s32', 'hust': 'hust_s32'}

u = pd.read_csv(os.path.join(ROOT, 'uconn', 'CyclingParameters.csv'), encoding='utf-8-sig')
r = pd.read_csv(os.path.join(ROOT, 'ragone', 'cells.csv'), dtype={'cell': str})
r = r[r.ds == 'hust'].set_index('cell').P_W
pols = {'uconn': {f'{c:02d}': f'{d:.2f}|{ch:g}|{dc:g}' for c, d, ch, dc in zip(u['Cell ID'], u['DoD 1'], u['Chg C-Rate 1'], u['DChg C-Rate 1'])},
        'matr_merged': json.load(open(os.path.join(ROOT, 'matr', 'results', 'policies.json'))),
        'hust': pd.qcut(r, 5, labels=['p1', 'p2', 'p3', 'p4', 'p5']).astype(str).to_dict()}

for ds, pol in pols.items():
    json.dump(pol, open(os.path.join(ROOT, 'configs', 'lco', f'policies_{ds}.json'), 'w'), indent=0)
    t = pd.read_csv(os.path.join(REV, TUNE[ds], 'tune.csv'))
    assert t.key.str.startswith('split|').all() and t.groupby('key').size().eq(30).all(), ds
    t['key'] = 'cvpol|' + t.key.str[len('split|'):]
    t.to_csv(os.path.join(ROOT, 'configs', 'lco', f'tune_lco_{ds}.csv'), index=False)
    train = json.load(open(os.path.join(ROOT, ds, 'results', 'split.json')))['train']
    g = pd.Series({c: pol[c] for c in train})     # KeyError here = a training cell has no group
    print(f'{ds}: {len(train)} training cells in {g.nunique()} groups, largest {g.value_counts().iloc[0]}, tune rows {len(t)}')
