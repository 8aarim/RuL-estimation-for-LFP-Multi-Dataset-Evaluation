"""Step 10 (data only, no features/model): cell inventory, variables per dataset, and how the datasets differ
across test campaigns / cell generations in test conditions and capacity fade.
Writes compare/data_inventory/{cells.csv, variables.csv, generations.csv, fig_conditions.png, fig_fade.png}.
    python -m steps.s10_data_inventory
"""
import os
import pickle
import re
import zipfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul.common import ROOT

OUT = os.path.join(ROOT, 'compare', 'data_inventory')
os.makedirs(OUT, exist_ok=True)
from rul.paths import RAW as DATA  # noqa: E402
ORDER = ['MATR', 'SNL', 'HUST', 'UConn']   # oldest test campaign first
COL = {'MATR': '#1f77b4', 'SNL': '#9467bd', 'HUST': '#2ca02c', 'UConn': '#d62728'}
GEN = {'MATR': ('2017-2018', 'A123 APR18650M1A'), 'SNL': ('2017-2019', 'A123 APR18650M1A'),
       'HUST': ('~2021', 'LFP 18650 (A123 M1A spec; part no. not stated)'), 'UConn': ('2022-2023', 'LithiumWerks APR18650M1B')}


def lives(ds):
    df = pd.read_pickle(os.path.join(ROOT, ds, 'results', 'features_v2.pkl'))
    return (df.RUL + df.cycle).groupby(df.cell).max(), df


def charge_rate(policy):
    """Mean C-rate from 0 to 80% SOC for MATR policies 'aC(b%)-cC' (2 steps) or 'a-b-c-d' (4 x 20% SOC)."""
    p = policy.replace('-newstructure', '')
    m = re.match(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C', p)
    if m:
        c1, q1, c2 = map(float, m.groups())
        return 80 / (q1 / c1 + (80 - q1) / c2)
    steps = [float(x) for x in re.findall(r'[\d.]+', p)]
    return 80 / sum(20 / c for c in steps) if len(steps) == 4 else np.nan


rows = []
# MATR: all 140 cells in the batch files; 124 kept by the original notebook
life, mdf = lives('matr')
end_soh = mdf.sort_values('cycle').groupby('cell').Discharge_capacity.apply(lambda s: s.tail(5).median()) / 1.1
# 5 batch-1 cells stop at 88-95% capacity; the source sets cycle_life = last recorded cycle (test stopped, not EOL)
for b, fn in enumerate(('MATR_batch_20170512.mat', 'MATR_batch_20170630.mat', 'MATR_batch_20180412.mat'), 1):
    f = h5py.File(os.path.join(DATA, 'TRI', 'MATR', fn), 'r')
    g = f['batch']
    for i in range(g['policy_readable'].shape[0]):
        pol = ''.join(map(chr, np.array(f[g['policy_readable'][i, 0]][()]).flatten()))
        cid = f'b{b}_cell_{i}'
        raw_life = float(np.array(f[g['cycle_life'][i, 0]][()]).flatten()[0])   # NaN: no EOL in the file
        pol = pol if '(' in pol else pol + ' [truncated in source file]'
        rows.append(dict(dataset='MATR', cell=cid, batch=fn[11:19], protocol=f'charge {pol}; discharge 4C', charge_C=charge_rate(pol),
                         discharge_C=4.0, DoD=1.0, temperature_C=30, cycle_life=life.get(cid, raw_life),
                         status=('excluded by source notebook' if cid not in life else 'used' if end_soh[cid] < 0.85
                                 else f'used, but test stopped at {end_soh[cid]:.0%} capacity (source labels the stop as EOL)')))
# SNL: 18 LFP cells, conditions from file names
life, _ = lives('snl')
for f in sorted(os.listdir(os.path.join(DATA, 'SNL', 'BatteryLife_SNL_LFP', 'SNL', 'SNL'))):
    m = re.match(r'SNL_18650_LFP_(\d+)C_0-100_([\d.]+)-([\d.]+)C_(\w)\.pkl', f)
    if m:
        cid = f'{m[1]}C_{m[2]}-{m[3]}C_{m[4]}'
        rows.append(dict(dataset='SNL', cell=cid, protocol=f'{m[1]} °C, {m[2]}C charge / {m[3]}C discharge, 0-100% SOC', charge_C=float(m[2]),
                         discharge_C=float(m[3]), DoD=1.0, temperature_C=int(m[1]), cycle_life=life.get(cid, np.nan),
                         status='used (EOL measured)' if cid in life else 'never reaches 80% (censored)'))
# HUST: protocol is not stored in the files -> measured from one mid-life cycle per cell
life, _ = lives('hust')
z = zipfile.ZipFile(os.path.join(DATA, 'nsc7hnsg4s-2', 'nsc7hnsg4s-2', 'our_data.zip'))
for n in sorted(x for x in z.namelist() if x.endswith('.pkl')):
    name = os.path.basename(n)[:-4]
    c = pickle.loads(z.read(n))[name]
    d = c['data'][sorted(c['data'])[len(c['data']) // 2]]
    I, V = d['Current (mA)'].to_numpy() / 1000, d['Voltage (V)'].to_numpy()
    dis = -I[I < -0.01]
    lv = np.round(dis / 1.1 * 4) / 4                 # discharge current in C, nearest 0.25C
    n_at = pd.Series(lv).value_counts()
    runs = [x for i, x in enumerate(lv) if (i == 0 or x != lv[i - 1]) and n_at[x] >= 3]
    runs = [x for i, x in enumerate(runs) if i == 0 or x != runs[i - 1]]
    rows.append(dict(dataset='HUST', cell=name, protocol='discharge stages ' + '-'.join(f'{x:g}C' for x in runs) + ' (switch SOCs differ per cell)',
                     charge_C=I[(I > 0.01) & (V < 3.595)].mean() / 1.1, discharge_C=dis.mean() / 1.1, DoD=1.0,
                     temperature_C=np.nan, cycle_life=life.get(name, np.nan), status='used'))
# UConn: conditions from CyclingParameters.csv
life, _ = lives('uconn')
P = pd.read_csv(os.path.join(DATA, 'uni coni', 'CyclingParameters.csv'), encoding='utf-8-sig')
for _, r in P.iterrows():
    cid = f"{int(r['Cell ID']):02d}"
    rows.append(dict(dataset='UConn', cell=cid, batch=f"group {int(r['Group Number'])}",
                     protocol=f"{r['Chg C-Rate 1']:.3g}C charge / {r['DChg C-Rate 1']:.3g}C discharge, {r['DoD 1']:.0%} DoD",
                     charge_C=r['Chg C-Rate 1'], discharge_C=r['DChg C-Rate 1'], DoD=r['DoD 1'], temperature_C=np.nan,
                     cycle_life=life.get(cid, np.nan), status='used'))
cells = pd.DataFrame(rows)[['dataset', 'cell', 'batch', 'status', 'protocol', 'charge_C', 'discharge_C', 'DoD', 'temperature_C', 'cycle_life']]
cells.to_csv(os.path.join(OUT, 'cells.csv'), index=False)
print(cells.groupby(['dataset', 'status']).size().to_string())

# generation comparison: spread of test conditions and outcomes per dataset
gen = []
for ds in ORDER:
    c = cells[cells.dataset == ds]
    u = c[c.status.str.startswith('used')]
    rng = lambda s: f'{s.min():.2g}-{s.max():.2g}' if s.notna().any() else 'not logged'
    gen.append({'dataset': ds, 'test campaign': GEN[ds][0], 'cell': GEN[ds][1], 'cells tested': len(c), 'cells used': len(u),
                # HUST: 77 protocols per source; the stage sequence alone (switch SOCs not shown) gives fewer
                'distinct protocols': 77 if ds == 'HUST' else c.protocol.nunique(), 'charge C-rate range': rng(c.charge_C),
                'discharge C-rate range': rng(c.discharge_C), 'DoD values': ', '.join(f'{x:.0%}' for x in sorted(c.DoD.unique())),
                'temperatures (°C)': ', '.join(str(int(x)) for x in sorted(c.temperature_C.dropna().unique())) or 'not logged',
                'cycle life range': f'{u.cycle_life.min():.0f}-{u.cycle_life.max():.0f}',
                'cycle life spread (std/mean)': round(u.cycle_life.std() / u.cycle_life.mean(), 2)})
gen = pd.DataFrame(gen)
gen.to_csv(os.path.join(OUT, 'generations.csv'), index=False)
print(gen.to_string(index=False))

# variables inside each dataset (raw files)
V = [('MATR', 'cell', 'policy / policy_readable', '-', 'charging policy'), ('MATR', 'cell', 'cycle_life', 'cycles', 'cycles to 80%'),
     ('MATR', 'cell', 'barcode, channel_id', '-', 'cell barcode and tester channel'),
     ('MATR', 'cell', 'Vdlin', 'V', 'common voltage grid for interpolated discharge curves'),
     ('MATR', 'per-cycle summary', 'cycle', '-', 'cycle number'), ('MATR', 'per-cycle summary', 'QCharge, QDischarge', 'Ah', 'charge / discharge capacity'),
     ('MATR', 'per-cycle summary', 'IR', 'ohm', 'internal resistance'), ('MATR', 'per-cycle summary', 'Tavg, Tmin, Tmax', '°C', 'cell temperature'),
     ('MATR', 'per-cycle summary', 'chargetime', 'min', 'charge duration'),
     ('MATR', 'time series per cycle', 'I', 'A', 'current'), ('MATR', 'time series per cycle', 'V', 'V', 'voltage'),
     ('MATR', 'time series per cycle', 'Qc, Qd', 'Ah', 'charge / discharge capacity'), ('MATR', 'time series per cycle', 'T', '°C', 'temperature'),
     ('MATR', 'time series per cycle', 't', 'min', 'time'), ('MATR', 'time series per cycle', 'Qdlin, Tdlin', 'Ah, °C', 'discharge capacity / temperature on Vdlin grid'),
     ('MATR', 'time series per cycle', 'discharge_dQdV', 'Ah/V', 'incremental capacity of the discharge'),
     ('SNL', 'cell', 'cell_id, cathode/anode material, form_factor', '-', 'cell description'),
     ('SNL', 'cell', 'nominal_capacity_in_Ah', 'Ah', 'nominal capacity'), ('SNL', 'cell', 'max/min_voltage_limit_in_V', 'V', 'voltage window'),
     ('SNL', 'cell', 'charge_protocol, discharge_protocol, SOC_interval', '-', 'test protocol'),
     ('SNL', 'time series per cycle', 'cycle_number', '-', 'cycle number'), ('SNL', 'time series per cycle', 'current_in_A', 'A', 'current'),
     ('SNL', 'time series per cycle', 'voltage_in_V', 'V', 'voltage'),
     ('SNL', 'time series per cycle', 'charge_capacity_in_Ah, discharge_capacity_in_Ah', 'Ah', 'charge / discharge capacity'),
     ('SNL', 'time series per cycle', 'time_in_s', 's', 'time'), ('SNL', 'time series per cycle', 'temperature_in_C', '°C', 'temperature (missing in ~15% of cycles)'),
     ('SNL', 'time series per cycle', 'internal_resistance_in_ohm', 'ohm', 'empty (None) in this copy'),
     ('HUST', 'cell', 'rul', 'cycles', 'remaining cycles per cycle (source label)'), ('HUST', 'cell', 'dq', 'Ah', 'per-cycle capacity (source)'),
     ('HUST', 'time series per cycle', 'Status', '-', 'step type (charge / discharge / rest)'), ('HUST', 'time series per cycle', 'Cycle number', '-', 'cycle number'),
     ('HUST', 'time series per cycle', 'Current (mA)', 'mA', 'current'), ('HUST', 'time series per cycle', 'Voltage (V)', 'V', 'voltage'),
     ('HUST', 'time series per cycle', 'Capacity (mAh)', 'mAh', 'single counter: up on charge, down on discharge'),
     ('HUST', 'time series per cycle', 'Time (s)', 's', 'time'),
     ('UConn', 'cell', 'Cell ID, Group Number, Group Channel', '-', 'cell and test group (CyclingParameters.csv)'),
     ('UConn', 'cell', 'Chg C-Rate 1, DChg C-Rate 1, DoD 1', 'C, C, fraction', 'cycling condition'),
     ('UConn', 'time series (cycling CSV)', 'Week Number', 'week', 'week of the test block'), ('UConn', 'time series (cycling CSV)', 'Life', '-', '1st life (only 1st in this release)'),
     ('UConn', 'time series (cycling CSV)', 'Date', 'datetime', 'timestamp'), ('UConn', 'time series (cycling CSV)', 'Cycle Number', '-', 'cycle within the block (restarts after each RPT)'),
     ('UConn', 'time series (cycling CSV)', 'State', '-', 'CCCV Chg / Rest / CC DChg'), ('UConn', 'time series (cycling CSV)', 'Time (s)', 's', 'time since block start'),
     ('UConn', 'time series (cycling CSV)', 'Voltage (V), Current (A)', 'V, A', 'voltage, current'),
     ('UConn', 'time series (cycling CSV)', 'Capacity (Ah)', 'Ah', 'capacity, resets each step'),
     ('UConn', 'time series (RPT CSV)', 'RPT Number, Step Number', '-', 'reference test index and step'),
     ('UConn', 'time series (RPT CSV)', 'Segment Key', '-', 'ref_chg / ref_dchg / slow, fast, ultrafast pulse'),
     ('UConn', 'time series (RPT CSV)', 'Pulse Type, Pulse SOC', '-, %', 'pulse direction and target SOC'),
     ('UConn', 'time series (RPT CSV)', 'Num Cycles', 'cycles', 'cycles since the previous RPT')]
pd.DataFrame(V, columns=['dataset', 'level', 'variable', 'unit', 'description']).to_csv(os.path.join(OUT, 'variables.csv'), index=False)

# figure 1: test-condition space (each point = one cell)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
mk = {1.0: 'o', 0.8: 's', 0.6: '^'}
for ds in ORDER:
    c = cells[cells.dataset == ds]
    for dod, g in c.groupby('DoD'):
        ax.scatter(g.charge_C, g.discharge_C, s=28, marker=mk.get(dod, 'o'), color=COL[ds], alpha=0.6, edgecolor='k', lw=0.3,
                   label=f'{ds} ({GEN[ds][0]})' if dod == c.DoD.max() else None)
for dod, m in mk.items():
    ax.scatter([], [], marker=m, color='grey', label=f'DoD {dod:.0%}')
ax.set(xlabel='Charge C-rate (mean, 0-80% SOC)', ylabel='Discharge C-rate (mean)', title='Test conditions per cell, by campaign')
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_conditions.png'))

# figure 2: capacity fade, capacity / nominal vs cycle (UConn: weekly C/3 reference test)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, ds in zip(axes, ORDER):
    if ds == 'UConn':
        r = pd.read_csv(os.path.join(ROOT, 'uconn', 'rpt_capacity.csv'), comment='#', dtype={'cell': str})
        for _, g in r.groupby('cell'):
            ax.plot(g.cycles_done, g.q_C3_Ah / 1.2, color=COL[ds], lw=0.6, alpha=0.5)
    else:
        _, df = lives(ds.lower())
        for _, g in df.groupby('cell'):
            ax.plot(g.cycle, g.Discharge_capacity / 1.1, color=COL[ds], lw=0.5, alpha=0.4)
    ax.axhline(0.8, ls='--', color='k', lw=0.8)
    ax.set(title=f'{ds} ({GEN[ds][0]})', xlabel='Cycle', ylim=(0.7, 1.12))
axes[0].set_ylabel('Capacity / nominal')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_fade.png'))
