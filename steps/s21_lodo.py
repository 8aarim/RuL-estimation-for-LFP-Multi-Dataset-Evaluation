"""Leave-one-dataset-out (LODO) test of the common feature subset. Standalone (numpy/pandas/sklearn/xgboost only).

Step 12 ranked the 40 features by mean SHAP rank over ALL four datasets and showed the top-8 suffices on those same
four. Here the held-out dataset H takes no part in choosing the subset:
  1. order_H = the 40 features sorted by mean SHAP rank over the other three datasets (tie -> lower worst rank,
     the rule that reproduces features_shap.xlsx 'Overall'; checked below on all four).
  2. k_H = smallest k whose CV R2 is within 1 SE of the best k on EVERY source dataset (step-12 rule, sources only).
  3. On H: CV on its training cells + one score of its test cells for every k (where does k_H fall vs H's own
     best?), plus two baselines at k_H: H's own SHAP top-k_H (upper reference) and N random k_H-subsets (floor).
Same split, folds (CV_SEED 0), preprocessing and tuned XGBoost params as steps 4-12.

    python -m steps.s21_lodo                                     # all 4 held-out sets, resumable (root = repository)
    python s21_lodo.py --data matr=~/1job,hust=~/hust,uconn=~/uconn,snl=~/snl --out ~/lodo   # flat DICC folders
    python s21_lodo.py --out ~/lodo --summary                     # tables from what has finished (runs on a laptop)
Each dataset folder (default <root>/<ds>/results) needs features_v2.pkl, split.json, features.json, final_params.json.
SHAP ranks: shap_ranks_4ds.csv next to this script (the 4 shap_ranking.csv 'rank' columns; built from step 5 if missing).
Writes <out>/runs.csv (one row per CV run or test score; rerun skips finished rows) and <out>/summary.csv.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED, CV_SEED, N_FOLDS = 42, 0, 5
DATASETS = ('matr', 'hust', 'uconn', 'snl')
KS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 40)       # as step 12
TOP8 = ['Cum_Delta_Q', 'Discharge_capacity_slope', 'Discharge_capacity', 'Charge_time',
        'cycle_index', 'V_std_slope', 'CMT', 'VMT']                     # step-12 common subset, MATR = merged (2026-11-03)
XGB_DEFAULTS = dict(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, random_state=SEED, verbosity=0)   # rul/models.py DEFAULTS

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
ap.add_argument('--heldout', default=','.join(DATASETS))
ap.add_argument('--random', type=int, default=10, help='random k_H-subsets per held-out dataset')
ap.add_argument('--device', default=None, help='cuda|cpu (default: cuda if xgboost sees a GPU)')
ap.add_argument('--summary', action='store_true')
ap.add_argument('--ks', default=','.join(map(str, KS)), help='k values (smoke test: 5,40)')
ap.add_argument('--out', default=None, help='default <root>/lodo')
ap.add_argument('--data', default='', help='ds=DIR,... overrides <root>/<ds>/results per dataset')
ap.add_argument('--ranks', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shap_ranks_4ds.csv'))
A = ap.parse_args()
KS = tuple(int(v) for v in A.ks.split(','))
OUT = os.path.expanduser(A.out or os.path.join(A.root, 'lodo'))
DIRS = {ds: os.path.join(A.root, ds, 'results') for ds in DATASETS}
DIRS.update({k: os.path.expanduser(v) for k, v in (kv.split('=') for kv in A.data.split(',') if kv)})
os.makedirs(OUT, exist_ok=True)
RUNS = os.path.join(OUT, 'runs.csv')
COLS = ['key', 'heldout', 'role', 'ds', 'k', 'sub', 'kind', 'cv_r2', 'se', 'folds', 'R2', 'RMSE', 'MAE', 'MAPE', 'feats']


# ---------- helpers copied from rul/common.py (keep identical) ----------
def build(df, cells, feats):
    s = df.set_index('cell').loc[list(cells)].reset_index()
    return s[feats].to_numpy(float), s['RUL'].to_numpy(float), s


def prep(Xtr, Xte):
    imp, sc = SimpleImputer(strategy='median'), StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr))
    return Xtr, sc.transform(imp.transform(Xte))


def cell_folds(df, cells, seed, n=N_FOLDS):
    life = df.groupby('cell').cycle.max()
    lives = np.array([life[c] for c in cells])
    strata = np.digitize(lives, np.percentile(lives, [33.3, 66.6]), right=True)
    if np.bincount(strata, minlength=3).min() >= n:
        split = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(cells)), strata)
    else:
        split = KFold(n_splits=min(n, len(cells)), shuffle=True, random_state=seed).split(cells)
    return [([cells[i] for i in a], [cells[i] for i in b]) for a, b in split]


def metrics(y, p, s):
    life = pd.Series(y + s['cycle'].to_numpy()).groupby(s['cell'].to_numpy()).transform('max').to_numpy()
    return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)), MAPE=float(np.mean(np.abs(p - y) / life)))


# ---------- ranking ----------
def ranks(path):
    if not os.path.exists(path):   # build it from step 5 (<root>/<ds>/results/shap_ranking.csv 'rank' columns)
        pd.DataFrame({ds: pd.read_csv(os.path.join(A.root, ds, 'results', 'shap_ranking.csv')).set_index('Feature')['rank']
                      for ds in DATASETS}).rename_axis('Feature').to_csv(path)
    t = pd.read_csv(path).set_index('Feature')
    return {ds: t[ds] for ds in DATASETS}


def order_from(R, sources):
    t = pd.DataFrame({ds: R[ds] for ds in sources})
    return t.assign(m=t.mean(1), w=t.max(1)).sort_values(['m', 'w']).index.tolist()


def pick_k(cv):
    """Step-12 rule on the given rows (one dataset per group): smallest k within 1 SE of best k on every dataset."""
    best = cv.loc[cv.groupby('ds').cv_r2.idxmax()].set_index('ds')
    ok = cv.apply(lambda r: r.cv_r2 >= best.cv_r2[r.ds] - best.se[r.ds], axis=1)
    good = ok.groupby(cv.k).all()
    return int(good[good].index.min())


if not A.summary:
    R = ranks(A.ranks)
    full = order_from(R, DATASETS)
    assert full[:8] == TOP8, f'mean-rank rule does not reproduce the step-12 order: {full[:8]}'
    dev = A.device or ('cuda' if xgb.build_info().get('USE_CUDA') else 'cpu')
    data = {}

    def load(ds):
        if ds not in data:
            d = DIRS[ds]
            j = lambda f: json.load(open(os.path.join(d, f)))
            df = pd.read_pickle(os.path.join(d, 'features_v2.pkl'))
            sp, fe, fp = j('split.json'), j('features.json'), j('final_params.json')
            assert sorted(fe['main']) == sorted(full), f'{ds}: main features differ from the 40 ranked'
            data[ds] = (df, sp['train'], sp['test'], {**XGB_DEFAULTS, **fp['params']})
        return data[ds]

    done = set(pd.read_csv(RUNS).key) if os.path.exists(RUNS) else set()

    def save(row):
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(RUNS, mode='a', header=not os.path.exists(RUNS), index=False)
        done.add(row['key'])

    def fit(params, X, y):
        return xgb.XGBRegressor(**params, device=dev).fit(X, y)

    def cv_row(H, role, ds, k, feats, sub=0):
        key = f'{H}|{role}|{ds}|{k}|{sub}|cv'
        if key in done:
            return
        t0 = time.time()
        df, train, _, params = load(ds)
        r2 = []
        for tr, te in cell_folds(df, train, CV_SEED):
            Xa, ya, _ = build(df, tr, feats)
            Xb, yb, _ = build(df, te, feats)
            Xa, Xb = prep(Xa, Xb)
            r2.append(r2_score(yb, fit(params, Xa, ya).predict(Xb)))
        r2 = np.array(r2)
        save(dict(key=key, heldout=H, role=role, ds=ds, k=k, sub=sub, kind='cv', cv_r2=r2.mean(),
                  se=r2.std(ddof=1) / np.sqrt(len(r2)), folds=' '.join(f'{v:.4f}' for v in r2), feats='|'.join(feats)))
        print(f'{key:40s} CV R2 {r2.mean():.4f}  ({time.time() - t0:.0f}s)', flush=True)

    def test_row(H, role, ds, k, feats, sub=0):
        key = f'{H}|{role}|{ds}|{k}|{sub}|test'
        if key in done:
            return
        df, train, test, params = load(ds)
        Xa, ya, _ = build(df, train, feats)
        Xb, yb, sb = build(df, test, feats)
        Xa, Xb = prep(Xa, Xb)
        m = metrics(yb, fit(params, Xa, ya).predict(Xb), sb)
        save(dict(key=key, heldout=H, role=role, ds=ds, k=k, sub=sub, kind='test', **m, feats='|'.join(feats)))
        print(f'{key:40s} test R2 {m["R2"]:.4f} MAPE {m["MAPE"]:.4f}', flush=True)

    for ds in DATASETS:                                 # fail now, not 40 min in
        for f in ('features_v2.pkl', 'split.json', 'features.json', 'final_params.json'):
            assert os.path.exists(os.path.join(DIRS[ds], f)), f'missing {os.path.join(DIRS[ds], f)}'
    print(f'device {dev}; out {RUNS}; data {DIRS}', flush=True)
    for H in A.heldout.split(','):
        src = [d for d in DATASETS if d != H]
        order = order_from(R, src)
        print(f'\n=== held out {H}; order from {src}: {order[:10]}', flush=True)
        for ds in src:                                  # 1-2: choose k on the sources only
            for k in KS:
                cv_row(H, 'source', ds, k, order[:k])
        runs = pd.read_csv(RUNS)
        kH = pick_k(runs[(runs.heldout == H) & (runs.role == 'source') & (runs.kind == 'cv')])
        print(f'k_{H} = {kH}: {order[:kH]}', flush=True)
        for k in KS:                                    # 3: the held-out dataset, every k
            cv_row(H, 'target', H, k, order[:k])
            test_row(H, 'target', H, k, order[:k])
        own = R[H].sort_values().index.tolist()[:kH]    # H's own SHAP top-k_H (uses H: reference only)
        cv_row(H, 'own', H, kH, own)
        test_row(H, 'own', H, kH, own)
        rng = np.random.RandomState(SEED)
        for i in range(A.random):                       # random k_H-subsets: is the ranking better than chance?
            feats = list(rng.choice(full, kH, replace=False))
            cv_row(H, 'random', H, kH, feats, i)
            test_row(H, 'random', H, kH, feats, i)

# ---------- summary (from whatever has finished) ----------
runs = pd.read_csv(RUNS)
rows = []
for H in DATASETS:
    r = runs[runs.heldout == H]
    s = r[(r.role == 'source') & (r.kind == 'cv')]
    if s.ds.nunique() < 3 or s.groupby('ds').size().min() < len(KS):
        continue
    kH = pick_k(s)
    order = s[s.k == 40].feats.iloc[0].split('|')
    t = r[(r.role == 'target') & (r.kind == 'cv')].set_index('k')
    te = r[(r.role == 'target') & (r.kind == 'test')].set_index('k')
    if kH not in t.index or 40 not in te.index:
        continue
    best = t.loc[t.cv_r2.idxmax()]
    rnd_cv = r[(r.role == 'random') & (r.kind == 'cv')].cv_r2
    rnd_te = r[(r.role == 'random') & (r.kind == 'test')].MAPE
    own = r[r.role == 'own'].set_index('kind')
    rows.append(dict(
        heldout=H, k_H=kH, subset='|'.join(order[:kH]), overlap_top8=len(set(order[:kH]) & set(TOP8)),
        cv_r2_kH=t.cv_r2[kH], cv_r2_best=best.cv_r2, best_k=int(t.cv_r2.idxmax()),
        pct_of_best=100 * t.cv_r2[kH] / best.cv_r2, within_1se=t.cv_r2[kH] >= best.cv_r2 - best.se,
        cv_r2_40=t.cv_r2[40], test_mape_kH=te.MAPE.get(kH), test_mape_40=te.MAPE[40],
        test_r2_kH=te.R2.get(kH), test_r2_40=te.R2[40],
        own_cv_r2=own.cv_r2.get('cv'), own_test_mape=own.MAPE.get('test'),
        random_cv_r2=f'{rnd_cv.mean():.4f} +- {rnd_cv.std():.4f} (n={len(rnd_cv)})' if len(rnd_cv) else '',
        random_cv_beaten=int((rnd_cv < t.cv_r2[kH]).sum()) if len(rnd_cv) else None,
        random_test_mape=f'{rnd_te.mean():.4f} +- {rnd_te.std():.4f}' if len(rnd_te) else ''))
if rows:
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    print('\n' + out.drop(columns='subset').T.to_string())
    for r in rows:
        print(f'{r["heldout"]}: {r["subset"]}')
else:
    print('no held-out dataset finished yet')
