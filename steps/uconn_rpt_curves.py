"""UConn C/3 RPT capacity vs cumulative cycle for all 64 cells -> compare/trends/uconn_rpt.csv (slow: reads ~100 GB of CSV).
    python -m steps.uconn_rpt_curves
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rul.common import ROOT

from rul.paths import RAW  # noqa: E402
UCONN = os.path.join(RAW, 'uni coni')


def curve(cell):
    # same block/RPT bookkeeping as features/multi_dataset.py uconn_rpt, but keeps the whole curve
    fs = glob.glob(os.path.join(UCONN, 'cycling_part*', f'cycling_cell_{cell}_part*.csv'))
    blk = pd.concat(pd.read_csv(f, usecols=['Week Number', 'Cycle Number']) for f in fs).groupby('Week Number')['Cycle Number'].max()
    fs = glob.glob(os.path.join(UCONN, 'rpt_data', 'rpt_data', f'rpt_cell_{cell}_part*.csv'))
    d = pd.concat(pd.read_csv(f, usecols=['Week Number', 'Segment Key', 'Capacity (Ah)']) for f in fs)
    q = d[d['Segment Key'] == 'ref_dchg'].groupby('Week Number')['Capacity (Ah)'].max()
    cum = [blk[blk.index < w].sum() for w in q.index]
    print(cell, len(q), flush=True)
    return pd.DataFrame({'cell': cell, 'week': q.index, 'cycle': cum, 'capacity_Ah': q.to_numpy()})


if __name__ == '__main__':
    P = pd.read_csv(os.path.join(UCONN, 'CyclingParameters.csv'), encoding='utf-8-sig')
    cells = [f"{int(c):02d}" for c in P['Cell ID']]
    with Pool(6) as p:
        out = pd.concat(p.map(curve, cells))
    out.to_csv(os.path.join(ROOT, 'compare', 'trends', 'uconn_rpt.csv'), index=False)
