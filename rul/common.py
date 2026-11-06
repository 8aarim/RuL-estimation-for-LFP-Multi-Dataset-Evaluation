"""Shared settings and helpers: dataset setup, matrices, cell-wise folds, metrics."""
import builtins
import importlib
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import numpy as np
import pandas as pd
import shap.explainers._tree as _shap_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = 42                     # split and models
CV_SEED = 0                   # folds for the algorithm comparison, SHAP and top-k sweep
TUNE_SEED = 1                 # folds for the tuning search
CONFIRM_SEEDS = (2, 3, 4)     # fresh folds for tuned-vs-default confirmation
N_FOLDS = 5
DATASETS = ('matr', 'hust', 'uconn', 'snl')
SMOKE = os.environ.get('PIPE_SMOKE') == '1'   # ponytail: smoke-test knob (25 cells, 3000-row cap)
TEMP_FEATURES = ['TMT', 'T_mean', 'T_max', 'T_std', 'T_range', 'Cum_T_integral']

mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],   # DejaVu: Linux/HPC
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})

# shap 0.49 float()s xgboost>=3's base_score string '[4.37E2]'; patch that one parse
# (values checked identical to xgboost's own pred_contribs / pred_interactions).
_shap_tree.float = lambda x=0.0: builtins.float(x.strip('[]') if isinstance(x, str) else x)


def setup():
    """Parse the dataset from argv; return (ds, config module, results dir, figures dir)."""
    if len(sys.argv) < 2 or not os.path.exists(os.path.join(ROOT, sys.argv[1], 'config.py')):   # any dataset folder;
        sys.exit(f'usage: python -m {sys.modules["__main__"].__spec__.name} {"|".join(DATASETS)}')  # DATASETS = steps 7-8 set
    ds = sys.argv[1]
    cfg = importlib.import_module(f'{ds}.config')
    suffix = '_smoke' if SMOKE else ''
    res = os.path.join(ROOT, ds, 'results' + suffix)
    fig = os.path.join(ROOT, ds, 'figures' + suffix)
    os.makedirs(res, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    return ds, cfg, res, fig


def save_json(obj, path):
    json.dump(obj, open(path, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def load_json(path):
    return json.load(open(path))


def load_stage1(res):
    """Outputs of steps.s1_preprocess: preprocessed frame, split, feature lists."""
    df = pd.read_pickle(os.path.join(res, 'features_v2.pkl'))
    split = load_json(os.path.join(res, 'split.json'))
    feats = load_json(os.path.join(res, 'features.json'))
    return df, split['train'], split['test'], feats['main'], feats['all']


def to_frame(all_data):
    rows = [{'cell': k, 'cycle': c, **f} for k, cd in all_data.items() for c, f in sorted(cd.items())]
    return pd.DataFrame(rows)


def build(df, cells, feats):
    s = df.set_index('cell').loc[list(cells)].reset_index()     # cell by cell, cycles ascending
    return s[feats].to_numpy(float), s['RUL'].to_numpy(float), s


def prep(Xtr, Xte):
    imp, sc = SimpleImputer(strategy='median'), StandardScaler()  # fitted on training rows only
    Xtr = sc.fit_transform(imp.fit_transform(Xtr))
    return Xtr, sc.transform(imp.transform(Xte))


def cell_folds(df, cells, seed, n=N_FOLDS):
    """Cell-wise stratified k-fold, strata = cell-life terciles (as in the MATR notebook)."""
    life = df.groupby('cell').cycle.max()
    lives = np.array([life[c] for c in cells])
    strata = np.digitize(lives, np.percentile(lives, [33.3, 66.6]), right=True)
    if np.bincount(strata, minlength=3).min() >= n:
        split = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(cells)), strata)
    else:   # ponytail: too few cells per tercile to stratify (SNL: 6 training cells) -> plain shuffled k-fold
        split = KFold(n_splits=min(n, len(cells)), shuffle=True, random_state=seed).split(cells)
    return [([cells[i] for i in a], [cells[i] for i in b]) for a, b in split]


def metrics(y, p, s=None):
    """R2, RMSE, MAE (cycles); with s (rows' cell, cycle) also, comparable across datasets:
    nRMSE/nMAE = error / mean cell life; MAPE = mean |p - y| / that cell's life
    (= MAPE of the predicted lifetime p + cycle; MAPE on RUL itself diverges as RUL -> 0)."""
    m = dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
             MAE=float(mean_absolute_error(y, p)))
    if s is not None:
        y, cells = np.asarray(y), s['cell'].to_numpy()
        life = pd.Series(y + s['cycle'].to_numpy()).groupby(cells).transform('max').to_numpy()
        mean_life = pd.Series(life).groupby(cells).first().mean()
        m.update(nRMSE=m['RMSE'] / mean_life, nMAE=m['MAE'] / mean_life,
                 MAPE=float(np.mean(np.abs(np.asarray(p) - y) / life)))
    return m


def cell_bootstrap_ci(y, p, cells_arr, n=1000):
    rng = np.random.RandomState(SEED)
    uc = np.unique(cells_arr)
    idx = {c: np.where(cells_arr == c)[0] for c in uc}
    boot = [r2_score(y[s], p[s]) for s in
            (np.concatenate([idx[c] for c in rng.choice(uc, len(uc), replace=True)]) for _ in range(n))]
    return [float(v) for v in np.percentile(boot, [2.5, 97.5])]


def segment_report(y, p):
    rows = []
    for name, cut in (('Full', None), ('Last400', 400), ('Last200', 200), ('Last100', 100), ('Last50', 50)):
        m = np.ones_like(y, bool) if cut is None else y <= cut
        if m.sum() >= 5:
            rows.append(dict(Segment=name, n=int(m.sum()), **metrics(y[m], p[m])))
    return pd.DataFrame(rows)


# Common top-8 (step 12 rule: mean SHAP rank over the 4 datasets, tie -> lower worst rank), MATR = merged set
# (2026-11-03). With the original 124-cell labels slot 8 was I_std (tied with CMT at mean rank 12.00); with the 5
# continuation cells joined, CMT enters and I_std drops out. Old lists: rul/common_bak_20261103m.py.
TOP8 = ['Cum_Delta_Q', 'Discharge_capacity_slope', 'Discharge_capacity', 'Charge_time', 'cycle_index', 'V_std_slope', 'CMT', 'VMT']
COMPACT7 = [f for f in TOP8 if f != 'cycle_index']


def shap_ranks(datasets=None):
    """Feature x dataset SHAP rank table (step 5 shap_ranking.csv), sorted in the common order (mean, then worst)."""
    datasets = datasets or DATASETS
    r = pd.concat({d: pd.read_csv(os.path.join(ROOT, d, 'results', 'shap_ranking.csv')).set_index('Feature')['rank']
                   for d in datasets}, axis=1)
    return r.loc[r.assign(m=r.mean(1), w=r.max(1)).sort_values(['m', 'w']).index]
TREND_NAME = {'matr': 'MATR', 'hust': 'HUST', 'uconn': 'UConn', 'snl': 'SNL'}


def meta(ds, cells, life):
    """Per-cell operating condition + ageing-path strata (moved from step 31, also used by step 34)."""
    ds = ds.split('_')[0]   # matr_merged -> matr (trend descriptors are from the original labels)
    m = pd.DataFrame(index=pd.Index(cells, name='cell'))
    if ds == 'matr':
        pol = json.load(open(os.path.join(ROOT, 'matr', 'results', 'policies.json')))
        m['batch'] = [c.split('_')[0] for c in cells]
        m['charge_C1'] = [float(pol[c].split('C')[0]) for c in cells]
        m['charge_C1'] = pd.qcut(m.charge_C1, 3, labels=['low', 'mid', 'high'], duplicates='drop').astype(str)
    elif ds == 'uconn':
        p = pd.read_csv(os.path.join(ROOT, 'uconn', 'CyclingParameters.csv'), encoding='utf-8-sig')
        p.index = [f'{c:02d}' for c in p['Cell ID']]
        m['charge_C'] = p.loc[cells, 'Chg C-Rate 1'].map('{:g}C'.format).values
        m['discharge_C'] = p.loc[cells, 'DChg C-Rate 1'].map('{:g}C'.format).values
        m['DoD'] = p.loc[cells, 'DoD 1'].map(lambda v: f'{100 * v:.0f}%').values
    elif ds == 'snl':
        m['temperature'] = [c.split('_')[0] for c in cells]
        m['discharge_C'] = [c.split('_')[1].split('-')[1] for c in cells]
    else:
        r = pd.read_csv(os.path.join(ROOT, 'ragone', 'cells.csv'), dtype={'cell': str})
        r = r[r.ds == 'hust'].set_index('cell').P_W
        m['discharge_power'] = pd.qcut(r.reindex(cells), 3, labels=['low', 'mid', 'high']).astype(str).values
    m['life'] = pd.qcut(life.reindex(cells), 3, labels=['short', 'mid', 'long']).astype(str).values
    t = pd.read_csv(os.path.join(ROOT, 'compare', 'trends', 'cells.csv'), dtype={'cell': str})
    t = t[t.dataset == TREND_NAME[ds]].set_index('cell')
    for col, name in [('accel', 'fade_accel'), ('knee_frac', 'knee_pos')]:
        v = t[col].reindex(cells)
        m[name] = pd.qcut(v, 3, labels=['low', 'mid', 'high'], duplicates='drop').astype(str).values if v.notna().sum() >= 6 else 'na'
    return m


def per_cell_r2(oof):
    return oof.groupby('cell')[['y', 'p']].apply(lambda g: r2_score(g['y'], g['p']) if len(g) > 1 else np.nan)
