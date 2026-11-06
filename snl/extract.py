"""SNL feature extraction: BatteryLife pickles -> snl/features_snl.csv (42 per-cycle features + RUL).

    python -m snl.extract

Same feature code as MATR/HUST/UConn (features/multi_dataset.py `featurize`, the MATR
notebook's), SNL-specific input handling here:
- cells: the 9 LFP cells with a BatteryLife 80%-EOL label (the other 9 end at 83-88%, never reach EOL);
- RUL = BatteryLife label - cycle; cycles from the label on are dropped;
- every ~200-250 cycles a 4-cycle 0.5C capacity check is logged at ~10 s (regular cycles: 120 s):
  skipped, it is another protocol and its features jump;
- 120 s logging leaves ~4 CV-phase points per charge: CV needs 3 points (notebook: 10);
- CV ends at 0.049 A = 9% of the 0.549 A peak: CMT threshold 10% of peak (notebook: 5%, never reached);
- ~15% of cycles have no temperature at all: T = None, so temperature features are NaN, not TMT = 0.
"""
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.multi_dataset import featurize, to_frame   # noqa: E402  the one copy of the notebook feature code

from rul.paths import RAW  # noqa: E402
DATA = os.path.join(RAW, 'SNL', 'BatteryLife_SNL_LFP')
LABELS = os.path.join(DATA, 'life_labels', 'Life labels', 'SNL_labels.json')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features_snl.csv')
OUT_CENSORED = OUT.replace('.csv', '_censored.csv')   # 9 cells that never reach 80%: RUL = lower bound
V_CV, MIN_CV, CMT_FRAC = 3.595, 3, 0.1   # calibration knobs, see docstring
CHECK_DT = 60                            # s: median logging step below this = capacity-check cycle


def cycles(d, eol):
    for c in d['cycle_data']:
        cyc = c['cycle_number']
        if cyc >= eol:
            return
        t = np.asarray(c['time_in_s'], float)
        if len(t) < 2 or np.median(np.diff(t)) < CHECK_DT:
            continue
        I, V, Qc, Qd, T = (np.asarray(c[k], float) for k in ('current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah',
                                                             'discharge_capacity_in_Ah', 'temperature_in_C'))
        yield cyc, I, V, Qc, Qd, None if np.isnan(T).all() else T, t - t[0], eol - cyc


def extract(name):
    """Labelled cell: RUL = label - cycle. Censored cell (test stopped before 80%): EOL is only known to
    be after the last cycle, so RUL holds the lower bound last + 1 - cycle."""
    d = pickle.load(open(os.path.join(DATA, 'SNL', 'SNL', name + '.pkl'), 'rb'))
    eol = json.load(open(LABELS)).get(name + '.pkl')
    censored = eol is None
    if censored:
        eol = d['cycle_data'][-1]['cycle_number'] + 1
    feats = featurize(cycles(d, eol), v_cv=V_CV, min_cv=MIN_CV, cmt_frac=CMT_FRAC)
    print(f'  {name}: {len(feats)} cycles ({"censored, ran to " + str(eol - 1) if censored else f"EOL {eol}"})', flush=True)
    return name.replace('SNL_18650_LFP_', '').replace('_0-100', ''), feats, censored


if __name__ == '__main__':
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(DATA, 'SNL', 'SNL', 'SNL_18650_LFP_*.pkl')))
    with Pool(6) as p:
        res = p.map(extract, names)
    for path, cens in ((OUT, False), (OUT_CENSORED, True)):
        df = to_frame({n: f for n, f, c in res if c == cens})
        assert df.TMT.isna().equals(df.T_mean.isna()), 'a cycle without temperature got a TMT value'
        df.to_csv(path, index=False)
        print(f'wrote {path}: {df.cell.nunique()} cells, {len(df)} cycles')
