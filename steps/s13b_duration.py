"""Step 13b (review DA-C1): cycle duration for the raw inputs. RS is V, I over NORMALISED cycle time, so it carries
no duration; this adds the duration of each cycle (t_end - t_start, in the dataset's own time unit) for the same rows.

    python -m steps.s13b_duration matr|hust|uconn|snl      -> <ds>/results/raw_dur.npz (cell, cycle, dur)

Reuses step 13's loaders (same cycles); only the per-cycle function differs. Needs the raw data on D:.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rul.common import ROOT
from steps import s13_raw as s13


def dur_of(cycles, keep, grid):
    out = {}
    for cyc, I, V, Qc, Qd, T, t, rul in cycles:
        if cyc in keep and len(t):
            out[cyc] = float(np.nanmax(t) - np.nanmin(t))
    return out


s13.raw_of = dur_of   # module global used by s13._matr/_hust/_uconn/_snl; workers import this module, so it applies there


def job(args):
    ds, cell, keep = args
    return {'matr': s13._matr, 'matr_merged': s13._matr_merged, 'hust': s13._hust, 'uconn': s13._uconn, 'snl': s13._snl}[ds]((cell, keep))

if __name__ == '__main__':
    ds = sys.argv[1]
    res = os.path.join(ROOT, ds, 'results')
    df = pd.read_pickle(os.path.join(res, 'features_v2.pkl'))[['cell', 'cycle']]
    jobs = [(ds, c, set(g.cycle.astype(int))) for c, g in df.groupby('cell', sort=False)]
    got = {}
    with Pool(int(os.environ.get("NPROC", 6))) as p:   # UConn: ~1 GB per worker
        for cell, d in p.imap_unordered(job, jobs):
            got[cell] = d
    dur = np.array([got[c].get(int(y), np.nan) for c, y in zip(df.cell, df.cycle)], np.float32)
    np.savez_compressed(os.path.join(res, 'raw_dur.npz'), cell=df.cell.to_numpy(str), cycle=df.cycle.to_numpy(int), dur=dur)
    print(f'{ds}: {len(df)} rows, duration missing in {np.isnan(dur).sum()}; median {np.nanmedian(dur):.1f}')
