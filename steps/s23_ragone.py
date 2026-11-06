"""Step 23: discharge energy and mean discharge power per cycle (operating-condition map, Ragone axes).

    python -m steps.s23_ragone matr|hust|uconn|snl      -> ragone/<ds>.csv (cell, cycle, E_Wh, P_W, Q_Ah, Qd_max)
    python -m steps.s23_ragone fig                      -> ragone/fig_ragone.png (+ cells.csv)

Same raw loaders and (cell, cycle) rows as step 13. Discharge = samples with I < -I_MIN; E = sum V|I|dt,
P = E / discharge time (time-weighted mean power, rests excluded), Q_Ah = sum |I|dt as a units check against the
loader's Qd. Per cell (figure): median over the first EARLY feature cycles.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rul.common import ROOT
from steps import s13_raw as s13

OUT = os.path.join(ROOT, 'ragone')
I_MIN = 0.01                                     # A; below this a sample is rest
TO_H = {'matr': 1 / 60, 'hust': 1 / 3600, 'uconn': 1 / 3600, 'snl': 1 / 3600}   # MATR t is in minutes
EARLY = 100
I_SCALE = {'matr': 1.1}                         # MATR logs current in C-rate units (4.0 at 4C); x 1.1 Ah -> A


def energy_of(ds):
    def run(cycles, keep, grid):
        out = {}
        for cyc, I, V, Qc, Qd, T, t, rul in cycles:
            if cyc not in keep:
                continue
            o = np.argsort(t, kind='stable')
            t, V, I, Qd = t[o], V[o], I[o] * I_SCALE.get(ds, 1), Qd[o]
            dt = np.diff(t) * TO_H[ds]
            d = (I[1:] < -I_MIN) & (dt > 0)      # right-point rule: SNL logs every 120 s, the sample closes the interval
            h = dt[d].sum()
            vi = (V[1:] * -I[1:])[d] * dt[d]
            q = -I[1:][d] * dt[d]
            out[cyc] = (vi.sum(), vi.sum() / h if h > 0 else np.nan, q.sum(), np.nanmax(Qd) if len(Qd) else np.nan)
        return out
    return run


def job(args):
    ds, cell, keep = args
    s13.raw_of = energy_of(ds)                   # same loaders as step 13, energy instead of raw signals
    return getattr(s13, '_' + ds)((cell, keep))


def extract(ds):
    df = pd.read_pickle(os.path.join(ROOT, ds, 'results', 'features_v2.pkl'))[['cell', 'cycle']]
    jobs = [(ds, c, set(g.cycle.astype(int))) for c, g in df.groupby('cell', sort=False)]
    rows = []
    with Pool(6) as p:
        for cell, d in p.imap_unordered(job, jobs):
            rows += [(cell, cyc, *v) for cyc, v in d.items()]
            print(f'  {cell}: {len(d)} cycles', flush=True)
    out = pd.DataFrame(rows, columns=['cell', 'cycle', 'E_Wh', 'P_W', 'Q_Ah', 'Qd_max']).sort_values(['cell', 'cycle'])
    out.to_csv(os.path.join(OUT, f'{ds}.csv'), index=False)
    r = (out.Q_Ah / out.Qd_max).median()
    print(f'{ds}: {len(out)} of {len(df)} rows; median integrated Ah / loader Qd = {r:.3f} (expect ~1)')


def fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False,
                                'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                                'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    # ponytail: 4 series in a scatter exceed the 3-hue all-pairs cap -> 4th is neutral ink, all differ by marker too
    STYLE = {'matr': ('MATR', '#2a78d6', 'o'), 'hust': ('HUST', '#eb6834', 's'),
             'uconn': ('UConn-ISU-ILCC', '#1baf7a', '^'), 'snl': ('SNL', '#52514e', 'D')}
    cells = []
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for ds, (lab, col, mk) in STYLE.items():
        d = pd.read_csv(os.path.join(OUT, f'{ds}.csv'))
        d = d[d.groupby('cell').cycle.rank(method='first') <= EARLY]
        c = d.groupby('cell')[['E_Wh', 'P_W', 'Q_Ah']].median().assign(ds=ds)
        cells.append(c)
        ax.scatter(c.P_W, c.E_Wh, s=16, marker=mk, facecolor=col, edgecolor='white', linewidth=0.5, alpha=0.9,
                   label=f'{lab} ({len(c)})', zorder=3)
    pd.concat(cells).to_csv(os.path.join(OUT, 'cells.csv'))
    ax.set_xscale('log')
    ax.set_xticks([3, 4, 6, 8, 10, 12], ['3', '4', '6', '8', '10', '12'])
    ax.minorticks_off()
    ax.set_xlabel('Mean discharge power (W)')
    ax.set_ylabel('Discharge energy per cycle (Wh)')
    ax.grid(True, which='both', color='#e6e5e0', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower left', fontsize=7, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_ragone.png'), dpi=300)
    print(pd.concat(cells).groupby('ds')[['E_Wh', 'P_W', 'Q_Ah']].describe().round(2).T.to_string())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig() if sys.argv[1] == 'fig' else extract(sys.argv[1])
