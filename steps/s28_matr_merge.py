"""Step 28 (review R2-M1): are MATR b1_cell_0..4 continued in batch 2? Severson et al.'s loader merges
b1c0<-b2c7, b1c1<-b2c8, b1c2<-b2c9, b1c3<-b2c15, b1c4<-b2c16; the original feature cache kept the batch-1 part only
and dropped the five batch-2 parts.

    python -m steps.s28_matr_merge check    # policies + capacity at the junction + where 80 % is reached (minutes)
    python -m steps.s28_matr_merge build    # writes data/features/all_data_42features_merged.pkl

build re-featurises the five merged cells with the same featurize() as the cache (history features such as
DCIR_min_so_far run over the joined cycles), sets EOL = first cycle with discharge capacity <= 0.88 Ah (80 % of
1.1 Ah, Severson's definition) and RUL = EOL - cycle, and copies the other 119 cells unchanged. It does NOT change
matr/config.py: to adopt, point SOURCE there to the merged file and rerun the MATR steps (s1 onwards).
"""
import os
import pickle
import sys

import h5py
import numpy as np

from features import multi_dataset as md  # noqa: E402

PAIRS = {0: 7, 1: 8, 2: 9, 3: 15, 4: 16}
B1 = os.path.join(md.DATA, 'TRI', 'MATR', 'MATR_batch_20170512.mat')
B2 = os.path.join(md.DATA, 'TRI', 'MATR', 'MATR_batch_20170630.mat')
SRC = os.path.join(md.RES, 'all_data_42features.pkl')
DST = os.path.join(md.RES, 'all_data_42features_merged.pkl')
Q_EOL = 0.88


def summary_qd(f, cell):
    b = f['batch']
    return np.hstack(f[b['summary'][cell, 0]]['QDischarge'][()]).astype(float)


def policy(f, cell):
    b = f['batch']
    return ''.join(map(chr, np.array(f[b['policy_readable'][cell, 0]][()]).flatten()))


def all_cycles(f, cell, offset=0):
    """Every recorded cycle (no cycle_life cap), cycle number shifted by offset; rul filled later."""
    cg = f[f['batch']['cycles'][cell, 0]]
    for j in range(1, cg['I'].shape[0]):
        try:
            a = [np.hstack(f[cg[k][j, 0]][()]) for k in ('I', 'V', 'Qc', 'Qd', 'T', 't')]
        except Exception:
            continue
        yield (offset + j, *a, 0)


def check():
    with h5py.File(B1, 'r') as f1, h5py.File(B2, 'r') as f2:
        for c1, c2 in PAIRS.items():
            q1, q2 = summary_qd(f1, c1), summary_qd(f2, c2)
            q1 = q1[q1 > 0.5]
            q2 = q2[q2 > 0.5]
            merged = np.concatenate([q1, q2])
            hit = np.where(merged <= Q_EOL)[0]
            print(f'b1c{c1} [{policy(f1, c1)}] <- b2c{c2} [{policy(f2, c2)}]')
            print(f'   batch-1: {len(q1)} cycles, last 5 QD {np.round(q1[-5:], 4)} (min {q1.min():.4f}, = {q1[-1] / 1.1:.1%} of 1.1 Ah)')
            print(f'   batch-2: {len(q2)} cycles, first 5 QD {np.round(q2[:5], 4)}, last {q2[-1]:.4f}')
            print(f'   junction step {q2[:5].mean() - q1[-5:].mean():+.4f} Ah; 0.88 Ah reached at merged cycle '
                  f'{hit[0] + 1 if len(hit) else "never"} (batch-1 alone: {len(q1)})')


def build():
    ref = pickle.load(open(SRC, 'rb'))
    out = dict(ref)
    with h5py.File(B1, 'r') as f1, h5py.File(B2, 'r') as f2:
        for c1, c2 in PAIRS.items():
            n1 = f1[f1['batch']['cycles'][c1, 0]]['I'].shape[0] - 1
            feats = md.featurize(list(all_cycles(f1, c1)) + list(all_cycles(f2, c2, offset=n1)))
            cyc = sorted(feats)
            # EOL as for every other MATR cell: the source stops each cell at its end of life (batch-2 part ends at
            # 0.880-0.883 Ah), and matr_cycles labels RUL = cycle_life - j; here cycle_life = last merged cycle + 1
            eol = cyc[-1] + 1
            out[f'b1_cell_{c1}'] = {c: {**feats[c], 'RUL': eol - c} for c in cyc if c < eol}
            print(f'b1_cell_{c1}: {len(ref[f"b1_cell_{c1}"])} -> {len(out[f"b1_cell_{c1}"])} cycles, EOL {eol}')
    pickle.dump(out, open(DST, 'wb'))
    print('wrote', DST)


if __name__ == '__main__':
    {'check': check, 'build': build}[sys.argv[1]]()
