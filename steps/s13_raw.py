"""Step 13: raw-signal inputs for the raw-vs-engineered comparison, for exactly the (cell, cycle) rows of step 1.

    python -m steps.s13_raw matr|hust|uconn|snl
    python -m steps.s13_raw matr temp          # only the temperature trace [128] -> raw_temp.npz (thermal chapter)

Per cycle:
  sig   [2, 128]  voltage and current over normalised cycle time (temperature left out: HUST/UConn have none)
  curve [100]     discharge capacity Q(V) on the dataset's voltage grid (same function as the dQV features)
Writes <ds>/results/raw_inputs.npz (cell, cycle, sig, curve; float32). Uses the loaders that made the features
(features/multi_dataset.py, snl/extract.py), so the cycles are the same ones.
"""
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from features import multi_dataset as md  # noqa: E402

from rul.common import ROOT  # noqa: E402

N_SIG, N_CURVE = 128, 100
GRID = {'uconn': np.linspace(3.17, 3.5, N_CURVE)}           # UConn partial-DoD window, as its dQV features
DEFAULT_GRID = np.linspace(2.0, 3.5, N_CURVE)


def raw_of(cycles, keep, grid):
    """cycles: loader iterator (cyc, I, V, Qc, Qd, T, t, rul); keep: set of cycle numbers to return."""
    out, tg = {}, np.linspace(0, 1, N_SIG)
    for cyc, I, V, Qc, Qd, T, t, rul in cycles:
        if cyc not in keep:
            continue
        o = np.argsort(t, kind='stable')
        t, V, I, Qd = t[o], V[o], I[o], Qd[o]
        tn = (t - t[0]) / max(t[-1] - t[0], 1e-9)
        tn, u = np.unique(tn, return_index=True)
        sig = np.stack([np.interp(tg, tn, V[u]), np.interp(tg, tn, I[u])]).astype(np.float32)
        q = md.discharge_qv_curve(V, I, Qd, grid)
        temp = np.full(N_SIG, np.nan, np.float32)
        if T is not None and np.size(T) == len(o):   # HUST/UConn have no temperature
            T = np.asarray(T, float)[o][u]
            ok = ~np.isnan(T)
            if ok.any():
                temp = np.interp(tg, tn[ok], T[ok]).astype(np.float32)
        out[cyc] = (sig, np.full(N_CURVE, np.nan, np.float32) if q is None else q.astype(np.float32), temp)
    return out


def _matr(args):
    import h5py
    cell, keep = args
    b, i = cell.split('_cell_')
    path = {'b1': 'MATR_batch_20170512.mat', 'b2': 'MATR_batch_20170630.mat', 'b3': 'MATR_batch_20180412.mat'}[b]
    with h5py.File(os.path.join(md.DATA, 'TRI', 'MATR', path), 'r') as f:
        return cell, raw_of(md.matr_cycles(f, int(i)), keep, DEFAULT_GRID)


def _matr_merged(args):
    """matr_merged (step 28): b1_cell_0..4 continue in batch 2; the other cells as _matr."""
    import h5py
    from steps import s28_matr_merge as s28
    cell, keep = args
    b, i = cell.split('_cell_')
    if b != 'b1' or int(i) not in s28.PAIRS:
        return _matr(args)
    i = int(i)
    with h5py.File(s28.B1, 'r') as f1, h5py.File(s28.B2, 'r') as f2:
        n1 = f1[f1['batch']['cycles'][i, 0]]['I'].shape[0] - 1
        cyc = list(s28.all_cycles(f1, i)) + list(s28.all_cycles(f2, s28.PAIRS[i], offset=n1))
        return cell, raw_of(cyc, keep, DEFAULT_GRID)


def _hust(args):
    import zipfile
    cell, keep = args
    z = zipfile.ZipFile(md.HUST_ZIP)
    n = next(x for x in z.namelist() if x.endswith(f'/{cell}.pkl') or x == f'{cell}.pkl')
    return cell, raw_of(md.hust_cycles(pickle.loads(z.read(n))[cell]), keep, DEFAULT_GRID)


def _uconn(args):
    cell, keep = args
    return cell, raw_of(md.uconn_cycles(cell), keep, GRID['uconn'])


def _snl(args):
    from snl import extract as se
    cell, keep = args
    name = next(n for n in json.load(open(se.LABELS)) if n.replace('SNL_18650_LFP_', '').replace('_0-100', '')[:-4] == cell)[:-4]
    d = pickle.load(open(os.path.join(se.DATA, 'SNL', 'SNL', name + '.pkl'), 'rb'))
    eol = json.load(open(se.LABELS))[name + '.pkl']
    return cell, raw_of(se.cycles(d, eol), keep, DEFAULT_GRID)


if __name__ == '__main__':
    ds = sys.argv[1]
    res = os.path.join(ROOT, ds, 'results')
    df = pd.read_pickle(os.path.join(res, 'features_v2.pkl'))[['cell', 'cycle']]
    jobs = [(c, set(g.cycle.astype(int))) for c, g in df.groupby('cell', sort=False)]
    fn = {'matr': _matr, 'matr_merged': _matr_merged, 'hust': _hust, 'uconn': _uconn, 'snl': _snl}[ds]
    with Pool(int(os.environ.get("NPROC", 6))) as p:                                        # ponytail: ~1 GB RAM per worker on UConn
        got = {}
        for cell, d in p.imap_unordered(fn, jobs):
            got[cell] = d
            print(f'  {cell}: {len(d)} cycles', flush=True)
    sig = np.full((len(df), 2, N_SIG), np.nan, np.float32)
    curve = np.full((len(df), N_CURVE), np.nan, np.float32)
    temp = np.full((len(df), N_SIG), np.nan, np.float32)
    miss = 0
    for r, (c, y) in enumerate(zip(df.cell, df.cycle.astype(int))):
        v = got[c].get(y)
        if v is None:
            miss += 1
            continue
        sig[r], curve[r], temp[r] = v
    assert miss == 0, f'{miss} feature rows have no raw cycle'
    if 'temp' in sys.argv[2:]:   # thermal chapter: only the temperature trace, raw_inputs.npz left as it is
        np.savez_compressed(os.path.join(res, 'raw_temp.npz'), cell=df.cell.to_numpy(str), cycle=df.cycle.to_numpy(int),
                            temp=temp)
        print(f'{ds}: {len(df)} rows; temperature missing in {np.isnan(temp).all(1).sum()} rows')
        sys.exit()
    np.savez_compressed(os.path.join(res, 'raw_inputs.npz'), cell=df.cell.to_numpy(str), cycle=df.cycle.to_numpy(int),
                        sig=sig, curve=curve)
    print(f'{ds}: {len(df)} rows; curve missing in {np.isnan(curve).all(1).sum()} rows')
