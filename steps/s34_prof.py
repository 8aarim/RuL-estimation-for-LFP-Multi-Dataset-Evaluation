"""Step 34: validation analyses t1-t19 (transfer, label sensitivity, stability, temperature, uncertainty, errors, BMS cost, repro).

    python -m steps.s34_prof t1 t2 ...      # or: all            -> prof/t<N>_*.csv
    python -m steps.s34_prof check           # self-check of the small helpers

 t1  cross-dataset MODEL transfer: train one XGBoost on 3 datasets, test on the 4th without retraining
 t2  MATR EOL: the 5 batch-1 cells that stop at 88-95 % (source label) vs dropped vs merged with batch 2
 t3  feature-selection stability: cell-bootstrap SHAP rankings + cross-dataset and LODO subset overlap
 t4  temperature: error per SNL temperature, temperature hold-out, E40 vs E40+temperature vs temperature-only
 t5  SNL: confidence intervals, censored-cell bias check, labelled vs censored composition
 t6  operating conditions: how much each feature moves with condition (DoD, C-rate, temperature) at equal ageing
 t8  mechanisms: ICA proxies (capacity, IC peak height / position, DCIR) vs compact features
 t9  unseen protocols / batches: grouped vs random splits, factor hold-out (aggregates steps 30-31)
 t10 uncertainty: cell-grouped cross-conformal (all training cells calibrate), Mondrian by predicted life stage
 t11 errors in cycles: MAE / RMSE overall, early / mid / late life, near EOL (RUL <= 200 / 100 / 50)
 t12 correlated features: grouped SHAP (families, correlation clusters), family rank stability, family ablation
 t13 BMS: model size, nodes, comparisons per prediction, CPU latency
 t14 reproducibility: cell ids + split + labels, package versions, file hashes, configs
 (t7 = tuned deep models, step 32 on DICC; t15 = cell-format limitation, text only)

Same split (seed 42), folds, preprocessing and tuned XGBoost params as steps 3-6 unless stated. MATR = matr/ = the
merged set (Severson loader, step 28). t2 compares the
two and needs matr_orig: do not rerun it after the swap.
"""
import hashlib
import json
import os
import platform
import sys
import time
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import r2_score

from rul.common import (COMPACT7, CV_SEED, DATASETS, ROOT, build, cell_folds, load_json, load_stage1, meta,
                        metrics, prep)
from rul.models import DEFAULTS

OUT = os.path.join(ROOT, 'prof')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if xgb.build_info().get('USE_CUDA') else 'cpu'
LODO = os.environ.get('LODO_RUNS', os.path.join(ROOT, 'lodo', 'runs.csv'))   # step 21 output
FAMILIES = {
    'History and trend': ['Cum_Delta_Q', 'Discharge_capacity_slope', 'cycle_index', 'V_std_slope'],
    'Capacity': ['Discharge_capacity', 'Delta_Q_cycle', 'CE'],
    'Charge timing': ['Charge_time', 'VMT', 'CMT'],
    'Voltage and current statistics': ['V_mean', 'V_std', 'I_mean', 'I_std', 'dV_dt_mean', 'dV_dt_std',
                                       'dI_dt_mean', 'dI_dt_std'],
    'Charge phases (CC/CV)': ['CC_length', 'CV_length', 'CC_V_mean', 'CC_V_std', 'CC_I_mean', 'CC_I_std',
                              'CV_I_mean', 'CV_I_std'],
    'Resistance (DCIR)': ['DCIR_median', 'DCIR_std', 'DCIR_min_so_far', 'DCIR_drift'],
    'IC / DV peaks': ['IC_max', 'IC_mean', 'DVA_max', 'DVA_mean'],
    'Delta Q(V) statistics': ['dQV_min', 'dQV_mean', 'dQV_var', 'dQV_skew', 'dQV_kurt', 'dQV_at_2V'],
}
COND = {'matr': ['batch', 'charge_C1'], 'hust': ['discharge_power'], 'uconn': ['DoD', 'charge_C', 'discharge_C'],
        'snl': ['temperature', 'discharge_C']}
res = lambda ds: os.path.join(ROOT, ds, 'results')  # noqa: E731


def params(ds):
    return {**DEFAULTS['XGBoost'], **load_json(os.path.join(res(ds), 'final_params.json'))['params'], 'device': DEV}


def fit(Xa, ya, p):
    return xgb.XGBRegressor(**p).fit(Xa, ya)


def life_of(df):
    return (df.groupby('cell').cycle.max() + df.groupby('cell').RUL.last()).astype(float)


def score(y, p, s):
    m = metrics(np.asarray(y), np.asarray(p), s)
    return dict(R2=m['R2'], RMSE=m['RMSE'], MAE=m['MAE'], LMAE=100 * m['MAPE'])


def train_test(df, tr, te, feats, p):
    (Xa, ya, _), (Xb, yb, sb) = build(df, tr, feats), build(df, te, feats)
    Xa, Xb = prep(Xa, Xb)
    return yb, fit(Xa, ya, p).predict(Xb), sb


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 80):
        print(f'\n== {name}\n', df.round(3).to_string(index=False), flush=True)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def kuncheva(a, b, n):
    """Kuncheva consistency of two equal-size subsets of n features: 1 identical, ~0 chance, < 0 worse than chance."""
    k, r = len(a), len(set(a) & set(b))
    return (r * n - k * k) / (k * (n - k))


def wquantile(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return v[o][min(np.searchsorted(c, q), len(v) - 1)]


def boot_ci(per_cell, stat, n=2000, seed=42):
    """95 % percentile CI of stat(frame) over cell resamples. per_cell: dict cell -> rows."""
    rng, cells = np.random.default_rng(seed), list(per_cell)
    v = [stat(pd.concat([per_cell[c] for c in rng.choice(cells, len(cells))])) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def stage(frac):
    return pd.cut(frac, [-1, 1 / 3, 2 / 3, 9], labels=['early', 'mid', 'late']).astype(str)


# ------------------------------------------------------------------ t1 model transfer
def t1():
    data = {ds: load_stage1(res(ds)) for ds in DATASETS}
    main = data['matr'][3]
    p = {**DEFAULTS['XGBoost'], 'device': DEV}   # no dataset's tuned params: same config everywhere
    rows = []
    for fs, feats in (('E40', main), ('E7', COMPACT7)):
        for tgt in DATASETS:
            dft, tr_t, te_t = data[tgt][:3]
            src = [d for d in DATASETS if d != tgt]
            # pooled: one imputer + scaler on the pooled source rows (plain transfer).
            # per_ds: each dataset standardised with its own feature statistics (target: unlabelled features only).
            for scaling in ('pooled', 'per_ds'):
                Xs, ys = [], []
                for d in src:
                    X, y, _ = build(data[d][0], data[d][1] + data[d][2], feats)
                    Xs.append(prep(X, X)[0] if scaling == 'per_ds' else X)
                    ys.append(y)
                Xa, ya = np.vstack(Xs), np.concatenate(ys)
                for scope, cells in (('target_test', te_t), ('target_all', tr_t + te_t)):
                    Xb, yb, sb = build(dft, cells, feats)
                    if scaling == 'per_ds':
                        Xa_, Xb_ = Xa, prep(build(dft, tr_t + te_t, feats)[0], Xb)[1]
                    else:
                        Xa_, Xb_ = prep(Xa, Xb)
                    pb = fit(Xa_, ya, p).predict(Xb_)
                    rows.append(dict(input=fs, target=tgt, sources='+'.join(src), scaling=scaling, scope=scope,
                                     n_cells=len(cells), **score(yb, pb, sb)))
            yb, pb, sb = train_test(dft, tr_t, te_t, feats, p)   # within-dataset, same untuned config
            rows.append(dict(input=fs, target=tgt, sources=tgt + ' train', scaling='within', scope='target_test',
                             n_cells=len(te_t), **score(yb, pb, sb)))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), 't1_transfer.csv')


# ------------------------------------------------------------------ t2 MATR EOL sensitivity
def t2():
    five = [f'b1_cell_{i}' for i in range(5)]
    src, mrg = load_stage1(res('matr')), load_stage1(res('matr_merged'))
    life_s, life_m = life_of(src[0]), life_of(mrg[0])
    save(pd.DataFrame({'cell': five, 'last_cycle_source': [src[0][src[0].cell == c].cycle.max() for c in five],
                       'life_source_label': life_s[five].values, 'life_merged_80pct': life_m[five].values,
                       'in_test_seed42': [c in src[2] for c in five]}), 't2_five_cells.csv')
    cells = src[1] + src[2]
    splits = [('seed42', src[1], src[2])]
    for s in range(10):
        perm = list(np.random.default_rng(s).permutation(cells))
        splits.append((f'r{s}', perm[len(src[2]):], perm[:len(src[2])]))
    variants = {'source_label': (src[0], []), 'drop5': (src[0], five), 'merged': (mrg[0], [])}
    pm = params('matr')
    rows = []
    for name, tr, te in splits:
        for fs, feats in (('E40', src[3]), ('E7', COMPACT7)):
            for v, (df, drop) in variants.items():
                tr_v = [c for c in tr if c not in drop]
                for scope, te_v in (('test_without_5', [c for c in te if c not in five]),
                                    ('test_all', [c for c in te if c not in drop])):
                    if scope == 'test_all' and v == 'drop5':
                        continue
                    yb, pb, sb = train_test(df, tr_v, te_v, feats, pm)
                    rows.append(dict(split=name, input=fs, variant=v, scope=scope, n_train=len(tr_v),
                                     n_test=len(te_v), n5_in_test=sum(c in five for c in te_v), **score(yb, pb, sb)))
        print(name, 'done', flush=True)
    r = pd.DataFrame(rows)
    save(r, 't2_eol_runs.csv')
    w = r[r.scope == 'test_without_5'].pivot_table(index=['split', 'input'], columns='variant', values='LMAE')
    summ = []
    for fs, g in w.groupby(level='input'):
        for v in ('drop5', 'merged'):
            d = g[v] - g['source_label']
            ci = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
            summ.append(dict(input=fs, variant=v, scope='test_without_5', mean_LMAE=g[v].mean(),
                             ref_source_label=g['source_label'].mean(), diff=d.mean(), ci_lo=ci[0], ci_hi=ci[1],
                             n_splits=len(d), p_wilcoxon=stats.wilcoxon(d).pvalue))
    save(pd.DataFrame(summ), 't2_eol_summary.csv')


# ------------------------------------------------------------------ t3 feature-selection stability
B_BOOT, SHAP_ROWS = 20, 4000


def t3():
    rows, summ = [], []
    for ds in DATASETS:
        df, train, _, main, _ = load_stage1(res(ds))
        p, rng = params(ds), np.random.default_rng(42)
        Xt, _, _ = build(df, train, main)
        idx = rng.choice(len(Xt), min(SHAP_ROWS, len(Xt)), replace=False)
        for b in range(B_BOOT):
            cs = list(rng.choice(train, len(train)))
            Xa, ya, _ = build(df, cs, main)
            Xa, Xs = prep(Xa, Xt[idx])
            booster = fit(Xa, ya, p).get_booster()
            sv = booster.predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]
            rows += [dict(ds=ds, b=b, feature=f, imp=v) for f, v in zip(main, np.abs(sv).mean(0))]
        print(ds, 'bootstrap done', flush=True)
    imp = pd.DataFrame(rows)
    imp.to_csv(os.path.join(OUT, 't3_boot_importance.csv'), index=False)
    n = imp.feature.nunique()
    freq = []
    for ds, g in imp.groupby('ds'):
        R = g.pivot(index='b', columns='feature', values='imp')
        order = [list(R.loc[b].sort_values(ascending=False).index) for b in R.index]
        rk = R.rank(axis=1, ascending=False)
        taus = [stats.kendalltau(rk.loc[a], rk.loc[b])[0] for a, b in combinations(R.index, 2)]
        for k in (7, 8, 15):
            pairs = list(combinations(order, 2))
            summ.append(dict(ds=ds, k=k, jaccard=np.mean([jaccard(a[:k], b[:k]) for a, b in pairs]),
                             kuncheva=np.mean([kuncheva(a[:k], b[:k], n) for a, b in pairs]),
                             kendall_tau_full_rank=np.mean(taus),
                             compact7_in_topk=np.mean([len(set(COMPACT7) & set(o[:k])) for o in order])))
        for f in R.columns:
            freq.append(dict(ds=ds, feature=f, mean_rank=rk[f].mean(), sd_rank=rk[f].std(),
                             p_top7=np.mean([f in o[:7] for o in order]), p_top15=np.mean([f in o[:15] for o in order])))
    save(pd.DataFrame(summ), 't3_stability_within.csv')
    fr = pd.DataFrame(freq).sort_values(['ds', 'mean_rank'])
    fr.to_csv(os.path.join(OUT, 't3_feature_frequency.csv'), index=False)
    save(fr[fr.feature.isin(COMPACT7)], 't3_compact7_frequency.csv')
    # across datasets (step 5 rankings) and across LODO folds (step 21 target orders)
    top = {ds: list(pd.read_csv(os.path.join(res(ds), 'shap_ranking.csv')).Feature) for ds in DATASETS}
    lodo = pd.read_csv(LODO)
    t = lodo[(lodo.role == 'target') & (lodo.kind == 'cv')]
    lo = {h: g.sort_values('k').feats.iloc[-1].split('|') for h, g in t.groupby('heldout')}
    kh = {h: int(g.k.iloc[0]) for h, g in lodo[lodo.role == 'own'].groupby('heldout')}
    across = []
    for kind, orders in (('dataset_shap', top), ('lodo_order', lo)):
        for k in (7, 8, 15):
            pairs = list(combinations(orders, 2))
            across.append(dict(kind=kind, k=k, jaccard=np.mean([jaccard(orders[a][:k], orders[b][:k]) for a, b in pairs]),
                               kuncheva=np.mean([kuncheva(orders[a][:k], orders[b][:k], n) for a, b in pairs]),
                               compact7_overlap=np.mean([len(set(COMPACT7) & set(o[:k])) for o in orders.values()])))
    across.append(dict(kind='lodo_k_H', k=np.nan, jaccard=np.mean([jaccard(lo[a][:kh[a]], lo[b][:kh[b]]) for a, b in
                                                                    combinations(lo, 2)]),
                       kuncheva=np.nan, compact7_overlap=np.nan, note=str(kh)))
    save(pd.DataFrame(across), 't3_stability_across.csv')


# ------------------------------------------------------------------ t4 temperature (SNL)
def t4():
    oof = pd.read_csv(os.path.join(ROOT, 'reliability', 'oof.csv.gz'), dtype={'cell': str})
    o = oof[oof.ds == 'snl'].copy()
    o['T'] = o.cell.str.split('_').str[0]
    life = (o.y + o.cycle).groupby(o.cell).max()
    rows = []
    for (fs, T), g in o.groupby(['set', 'T']):
        rows.append(dict(source='LOCO OOF (step 31)', input=fs, temperature=T, n_cells=g.cell.nunique(),
                         LMAE=100 * np.mean(np.abs(g.p - g.y) / g.cell.map(life)), MAE=np.mean(np.abs(g.p - g.y)),
                         RMSE=np.sqrt(np.mean((g.p - g.y) ** 2)),
                         coverage_norm=100 * np.mean((g.y >= g.norm_lo) & (g.y <= g.norm_hi))))
    # repeated random splits (step 30): with / without temperature features, per temperature of the test cell
    for fs in ('E40', 'E7', 'E40T', 'T6'):
        parts = []
        for f in sorted(os.listdir(os.path.join(ROOT, 'repcv'))):
            if f.startswith('pred_snl_random_') and f.endswith(f'_{fs}.csv.gz'):
                d = pd.read_csv(os.path.join(ROOT, 'repcv', f), dtype={'cell': str})
                d['split'] = f.split('_')[3]
                parts.append(d)
        d = pd.concat(parts)
        d['T'], d['L'] = d.cell.str.split('_').str[0], d.cell.map(life)
        for T, g in d.groupby('T'):
            per = (100 * np.abs(g.p - g.y) / g.L).groupby(g.split).mean()
            rows.append(dict(source='11 random splits (step 30)', input=fs, temperature=T, n_cells=g.cell.nunique(),
                             LMAE=per.mean(), LMAE_sd=per.std(), n_splits=len(per)))
    save(pd.DataFrame(rows), 't4_temperature_error.csv')
    h = pd.read_csv(os.path.join(ROOT, 'reliability', 'holdout.csv'))
    save(h[(h.ds == 'snl') & (h.factor == 'temperature')], 't4_temperature_holdout.csv')
    cen = pd.read_csv(os.path.join(ROOT, 'snl', 'features_snl_censored.csv'), usecols=['cell'])
    comp = pd.DataFrame({'cell': list(life.index) + list(cen.cell.unique())})
    comp['labelled'] = comp.cell.isin(life.index)
    comp['T'] = comp.cell.str.split('_').str[0]
    save(comp.groupby(['T', 'labelled']).size().unstack(fill_value=0).reset_index(), 't4_temperature_cells.csv')


# ------------------------------------------------------------------ t5 SNL small sample + censoring
def t5():
    oof = pd.read_csv(os.path.join(ROOT, 'reliability', 'oof.csv.gz'), dtype={'cell': str})
    rows = []
    for fs, o in oof[oof.ds == 'snl'].groupby('set'):
        life = (o.y + o.cycle).groupby(o.cell).max()
        o = o.assign(L=o.cell.map(life))
        per = {c: g for c, g in o.groupby('cell')}
        stat = {'LMAE': lambda g: 100 * np.mean(np.abs(g.p - g.y) / g.L), 'R2': lambda g: r2_score(g.y, g.p),
                'RMSE': lambda g: np.sqrt(np.mean((g.p - g.y) ** 2))}
        for k, f in stat.items():
            lo, hi = boot_ci(per, f)
            rows.append(dict(source='LOCO OOF, 9 cells, cell bootstrap', input=fs, metric=k, value=f(o), ci_lo=lo,
                             ci_hi=hi))
    rc = pd.read_csv(os.path.join(ROOT, 'repcv', 'runs.csv'))
    for fs, g in rc[(rc.ds == 'snl') & (rc.protocol == 'random') & rc.input.isin(['E40', 'E7'])].groupby('input'):
        for k, v in (('LMAE', 100 * g.MAPE), ('R2', g.R2), ('RMSE', g.RMSE)):
            ci = stats.t.interval(0.95, len(v) - 1, v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
            rows.append(dict(source=f'{len(v)} random 6/3 splits, t-interval', input=fs, metric=k, value=v.mean(),
                             ci_lo=ci[0], ci_hi=ci[1]))
    save(pd.DataFrame(rows), 't5_snl_ci.csv')
    # censored cells: does the labelled-only model respect their lower bound (true RUL >= observed remaining cycles)?
    from rul.preprocess import clean_trend_causal
    df, train, test, main, _ = load_stage1(res('snl'))
    cen = clean_trend_causal(pd.read_csv(os.path.join(ROOT, 'snl', 'features_snl_censored.csv')))
    ccells, rows = list(dict.fromkeys(cen.cell)), []
    for fs, feats in (('E40', main), ('E7', COMPACT7)):
        Xa, ya, _ = build(df, train + test, feats)
        Xc, lb, sc = build(cen, ccells, feats)
        Xa, Xc = prep(Xa, Xc)
        pc = fit(Xa, ya, params('snl')).predict(Xc)
        d = pd.DataFrame({'cell': sc.cell.values, 'cycle': sc.cycle.values, 'lower_bound': lb, 'p': pc})
        for c, g in d.groupby('cell'):
            rows.append(dict(input=fs, cell=c, last_cycle=g.cycle.max(), frac_rows_p_ge_bound=np.mean(g.p >= g.lower_bound),
                             p_life_at_cycle_100=(g.p + g.cycle).iloc[min(99, len(g) - 1)],
                             max_label_life_training=(ya + build(df, train + test, feats)[2].cycle.values).max()))
    save(pd.DataFrame(rows), 't5_censored_bound_check.csv')
    soh = lambda f, lab: pd.read_csv(os.path.join(ROOT, 'snl', f)).groupby('cell').apply(  # noqa: E731
        lambda g: pd.Series({'labelled': lab, 'last_cycle': g.cycle.max(),
                             'final_SOH': g.sort_values('cycle').Discharge_capacity.tail(25).median() / 1.1}))
    comp = pd.concat([soh('features_snl.csv', True), soh('features_snl_censored.csv', False)]).reset_index()
    comp['T'], comp['discharge_C'] = comp.cell.str.split('_').str[0], comp.cell.str.split('_').str[1].str.split('-').str[1]
    save(comp, 't5_snl_composition.csv')
    cv = pd.read_csv(os.path.join(ROOT, 'snl', 'results', 'censored_cv.csv'))
    cv['LMAE'] = 100 * cv.MAPE
    save(cv[['variant', 'R2', 'RMSE', 'MAE', 'LMAE', 'fold_R2', 'fold_R2_sd']], 't5_censored_training.csv')
    a = load_json(os.path.join(ROOT, 'snl_all', 'results', 'final_test.json'))
    save(pd.DataFrame([dict(scope=k, R2=a[k]['R2'], RMSE=a[k]['RMSE'], LMAE=100 * a[k]['MAPE'])
                       for k in ('all_cells_cv', 'all_cells_cv_measured', 'test_measured_cells')]),
         't5_snl_all18_extrapolated.csv')


# ------------------------------------------------------------------ t6 operating-condition sensitivity
def eta2(v, level, bins):
    """Share of a feature's variance explained by the condition level within life-fraction bins (0 none, 1 all)."""
    d = pd.DataFrame({'v': v, 'l': level, 'b': bins}).dropna()
    ssb = sst = 0.0
    for _, g in d.groupby('b'):
        if g.l.nunique() < 2:
            continue
        m = g.v.mean()
        sst += ((g.v - m) ** 2).sum()
        ssb += g.groupby('l').v.agg(lambda s: len(s) * (s.mean() - m) ** 2).sum()
    return ssb / sst if sst > 0 else np.nan


def t6():
    rows = []
    for ds in DATASETS:
        df, train, test, main, _ = load_stage1(res(ds))
        life = life_of(df)
        M = meta(ds, train + test, life)
        bins = (10 * df.cycle / df.cell.map(life)).clip(0, 9.99).astype(int)
        for fac in COND[ds]:
            lev = df.cell.map(M[fac])
            for f in main:
                rows.append(dict(ds=ds, factor=fac, levels='/'.join(sorted(M[fac].unique())), feature=f,
                                 compact7=f in COMPACT7, eta2=eta2(df[f], lev, bins)))
        print(ds, 'done', flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 't6_feature_condition_eta2.csv'), index=False)
    save(r[r.compact7].pivot_table(index='feature', columns=['ds', 'factor'], values='eta2').reset_index(),
         't6_compact7_eta2.csv')
    save(r.groupby(['ds', 'factor', 'compact7']).eta2.median().unstack().reset_index(), 't6_eta2_median.csv')
    m = pd.read_csv(os.path.join(ROOT, 'reliability', 'map.csv'))
    keep = m.factor.isin(sum(COND.values(), []))
    save(m[keep][['ds', 'set', 'factor', 'level', 'n_cells', 'MAPE', 'cov_norm', 'width_norm_pct_life']],
         't6_error_by_condition.csv')


# ------------------------------------------------------------------ t8 degradation-mechanism proxies
PEAK_BAND, PEAK_STEP = (2.8, 3.35), 0.045    # plateau voltages (V); largest move per cycle (V). Fixed before the rerun.


def band_peak(y, grid, lo, hi):
    """Interior local maximum of y inside [lo, hi] with parabolic sub-grid refinement -> (V, height), or None.
    A maximum on the edge of the window, or next to a gap, is not a peak."""
    idx = np.flatnonzero((grid >= lo) & (grid <= hi))
    if len(idx) < 3 or np.isnan(y[idx]).all():
        return None
    j = idx[np.nanargmax(y[idx])]
    if j in (idx[0], idx[-1]) or not np.isfinite(y[j - 1:j + 2]).all():
        return None
    a, b, c = y[j - 1:j + 2]
    bend = a - 2 * b + c
    off = 0.5 * (a - c) / bend if bend < 0 else 0.0
    return grid[j] + off * (grid[1] - grid[0]), b - 0.25 * (a - c) * off


def track_peak(ic, grid, start, band=PEAK_BAND, step=PEAK_STEP):
    """Follow one IC peak through a cell's cycles (rows ascending) from voltage start; NaN where it cannot be found."""
    out, prev = np.full((len(ic), 2), np.nan), start
    for i, y in enumerate(ic):
        p = band_peak(y, grid, max(prev - step, band[0]), min(prev + step, band[1]))
        if p:
            out[i] = p
            prev = p[0]
    return out[:, 0], out[:, 1]


def t8():
    rows, cells_out = [], []
    for ds in DATASETS:
        df, train, test, main, _ = load_stage1(res(ds))
        z = np.load(os.path.join(res(ds), 'raw_inputs.npz'))
        grid = np.linspace(3.17, 3.5, 100) if ds == 'uconn' else np.linspace(2.0, 3.5, 100)
        curve = z['curve'].astype(float)
        ic = np.abs(np.gradient(curve, grid, axis=1))
        r = pd.DataFrame({'cell': z['cell'].astype(str), 'cycle': z['cycle']})
        r['Q'] = np.nanmax(curve, 1) - np.nanmin(curve, 1)
        r['V_peak'], r['IC_peak'] = np.nan, np.nan
        life = life_of(df)
        life.index = life.index.astype(str)
        for c, g in r.groupby('cell').groups.items():
            i = np.asarray(g)[np.argsort(r.cycle.to_numpy()[g])]
            frac = r.cycle.to_numpy()[i] / life.get(c, np.nan)
            early = [band_peak(y, grid, *PEAK_BAND) for y in ic[i[(frac > 0.02) & (frac <= 0.1)]]]
            early = [p[0] for p in early if p]
            if early:
                r.loc[i, 'V_peak'], r.loc[i, 'IC_peak'] = track_peak(ic[i], grid, float(np.median(early)))
        d = df.merge(r, on=['cell', 'cycle'], how='inner')
        d['frac'] = d.cycle / d.cell.map(life)
        for c, g in d.groupby('cell'):
            e, l = g[(g.frac > 0.02) & (g.frac <= 0.1)], g[g.frac >= 0.9]
            if len(e) < 3 or len(l) < 3:
                continue
            cells_out.append(dict(ds=ds, cell=c, life=life[str(c)], tracked_pct=100 * g.V_peak.notna().mean(),
                                  Q_loss_pct=100 * (1 - l.Q.median() / e.Q.median()),
                                  IC_peak_loss_pct=100 * (1 - l.IC_peak.median() / e.IC_peak.median()),
                                  V_peak_shift_mV=1000 * (l.V_peak.median() - e.V_peak.median()),
                                  DCIR_rise_pct=100 * (l.DCIR_median.median() / e.DCIR_median.median() - 1)))
        # per-row relative proxies vs each cell's early median, then Spearman with the compact features
        base = d[(d.frac > 0.02) & (d.frac <= 0.1)].groupby('cell')[['Q', 'IC_peak', 'V_peak', 'DCIR_median']].median()
        prox = pd.DataFrame({'Q_rel': d.Q / d.cell.map(base.Q), 'IC_peak_rel': d.IC_peak / d.cell.map(base.IC_peak),
                             'V_peak_shift': d.V_peak - d.cell.map(base.V_peak),
                             'DCIR_rel': d.DCIR_median / d.cell.map(base.DCIR_median)})
        s = np.random.default_rng(42).choice(len(d), min(20000, len(d)), replace=False)
        for f in COMPACT7:
            for pn in prox.columns:
                a, b = d[f].to_numpy()[s], prox[pn].to_numpy()[s]
                m = np.isfinite(a) & np.isfinite(b)
                rows.append(dict(ds=ds, feature=f, proxy=pn, spearman=stats.spearmanr(a[m], b[m])[0] if m.sum() > 50 else np.nan))
        print(ds, 'done', flush=True)
    c = pd.DataFrame(cells_out)
    c.to_csv(os.path.join(OUT, 't8_cell_mechanism_proxies.csv'), index=False)
    agg = c.groupby('ds')[['tracked_pct', 'Q_loss_pct', 'IC_peak_loss_pct', 'V_peak_shift_mV', 'DCIR_rise_pct']].median().reset_index()
    agg.insert(1, 'n_cells', c.groupby('ds').size().to_numpy())
    agg.insert(2, 'n_with_peak', c.groupby('ds').V_peak_shift_mV.count().to_numpy())
    for p in ('Q_loss_pct', 'IC_peak_loss_pct', 'V_peak_shift_mV', 'DCIR_rise_pct'):
        agg[f'rho_life_{p}'] = [stats.spearmanr(g.life, g[p], nan_policy='omit')[0] for _, g in c.groupby('ds')]
    save(agg, 't8_mechanism_summary.csv')
    save(pd.DataFrame(rows).pivot_table(index=['ds', 'feature'], columns='proxy', values='spearman').reset_index(),
         't8_feature_vs_proxy.csv')


# ------------------------------------------------------------------ t9 unseen protocols / batches
def t9():
    rc = pd.read_csv(os.path.join(ROOT, 'repcv', 'runs.csv'))
    rc = rc[rc.input.isin(['E40', 'E7', 'RS', 'RSH'])]
    g = rc.groupby(['ds', 'protocol', 'input']).agg(n_splits=('MAPE', 'size'), LMAE=('MAPE', lambda v: 100 * v.mean()),
                                                     LMAE_sd=('MAPE', lambda v: 100 * v.std()), R2=('R2', 'mean'),
                                                     RMSE=('RMSE', 'mean')).reset_index()
    save(g, 't9_random_vs_grouped.csv')
    h = pd.read_csv(os.path.join(ROOT, 'reliability', 'holdout.csv'))
    h = h[h.factor.isin(sum(COND.values(), []))]
    save(h[['ds', 'set', 'factor', 'held_out', 'n_train_cells', 'n_cells', 'MAPE', 'MAPE_in_distribution', 'cov_norm',
            'cov_norm_in_distribution']], 't9_factor_holdout.csv')


# ------------------------------------------------------------------ t10 cell-grouped conformal intervals
ALPHA = 0.1


def t10():
    rows, out = [], []
    for ds in DATASETS:
        df, train, test, main, _ = load_stage1(res(ds))
        life, p = life_of(df), params(ds)
        cells = train + test
        outer = [([c for c in cells if c != h], [h]) for h in cells] if ds == 'snl' else [(train, test)]
        for fs, feats in (('E40', main), ('E7', COMPACT7)):
            for tr, te in outer:
                inner = [([c for c in tr if c != h], [h]) for h in tr] if ds == 'snl' else cell_folds(df, tr, CV_SEED)
                cal = []
                for a, b in inner:   # out-of-fold residuals of EVERY training cell = calibration set
                    yb, pb, sb = train_test(df, a, b, feats, p)
                    cal.append(pd.DataFrame({'cell': sb.cell.values, 'cycle': sb.cycle.values, 'y': yb, 'p': pb}))
                cal = pd.concat(cal)
                yt, pt, st = train_test(df, tr, te, feats, p)
                sig = lambda q: np.clip(q, 0, None) + 0.1 * life[tr].mean()  # noqa: E731
                w = 1 / cal.cell.map(cal.cell.value_counts()).to_numpy()      # each cell weighs the same
                # ponytail: plain 90 % weighted quantile. The cell-level finite-sample correction
                # ceil((n+1)(1-a))/n exceeds 1 for n < 9 cells (SNL: 8) -> infinite interval, so it is not applied.
                lvl = 1 - ALPHA
                r_abs, r_norm = np.abs(cal.y - cal.p).to_numpy(), (np.abs(cal.y - cal.p) / sig(cal.p)).to_numpy()
                k = min(int(np.ceil((len(cal) + 1) * (1 - ALPHA))), len(cal)) - 1
                q = {'cv_split': np.sort(r_abs)[k], 'cv_norm': np.sort(r_norm)[k],
                     'cell_norm': wquantile(r_norm, w, lvl)}
                t = pd.DataFrame({'cell': st.cell.values, 'cycle': st.cycle.values, 'y': yt, 'p': pt})
                t['cv_split_lo'], t['cv_split_hi'] = pt - q['cv_split'], pt + q['cv_split']
                for m in ('cv_norm', 'cell_norm'):
                    t[f'{m}_lo'], t[f'{m}_hi'] = pt - q[m] * sig(pt), pt + q[m] * sig(pt)
                # Mondrian: separate cell-weighted quantile per PREDICTED life stage (known at prediction time)
                sh_c, sh_t = stage(cal.cycle / (cal.cycle + cal.p.clip(lower=1))), stage(t.cycle / (t.cycle + t.p.clip(lower=1)))
                qm = {s: wquantile(r_norm[sh_c.values == s], w[sh_c.values == s], lvl) if (sh_c == s).sum() > 20
                      else q['cell_norm'] for s in ('early', 'mid', 'late')}
                qq = sh_t.map(qm).to_numpy()
                t['mondrian_lo'], t['mondrian_hi'] = pt - qq * sig(pt), pt + qq * sig(pt)
                t['ds'], t['set'] = ds, fs
                out.append(t)
            print(ds, fs, 'done', flush=True)
    o = pd.concat(out)
    o['L'] = np.nan
    for ds in DATASETS:
        lf = life_of(load_stage1(res(ds))[0])
        o.loc[o.ds == ds, 'L'] = o.loc[o.ds == ds, 'cell'].map(lf)
    o['stage'] = stage(o.cycle / o.L)
    o.to_csv(os.path.join(OUT, 't10_intervals.csv.gz'), index=False)
    for (ds, fs), g in o.groupby(['ds', 'set']):
        per = {c: x for c, x in g.groupby('cell')}
        for m in ('cv_split', 'cv_norm', 'cell_norm', 'mondrian'):
            cov = lambda x, m=m: 100 * np.mean((x.y >= x[f'{m}_lo']) & (x.y <= x[f'{m}_hi']))  # noqa: E731
            lo, hi = boot_ci(per, cov, n=1000)
            r = dict(ds=ds, input=fs, method=m, coverage=cov(g), cov_ci_lo=lo, cov_ci_hi=hi,
                     width_pct_life=100 * np.mean((g[f'{m}_hi'] - g[f'{m}_lo']) / g.L),
                     width_cycles=np.mean(g[f'{m}_hi'] - g[f'{m}_lo']))
            for s in ('early', 'mid', 'late'):
                x = g[g.stage == s]
                r[f'cov_{s}'], r[f'width_{s}_pct_life'] = cov(x), 100 * np.mean((x[f'{m}_hi'] - x[f'{m}_lo']) / x.L)
            rows.append(r)
    save(pd.DataFrame(rows), 't10_coverage.csv')


# ------------------------------------------------------------------ t11 errors in cycles, by stage and near EOL
def t11():
    rows = []
    for ds in DATASETS + ('matr_merged',):
        for fs in ('E40', 'E7'):
            for f in sorted(os.listdir(os.path.join(ROOT, 'repcv'))):
                if not (f.startswith(f'pred_{ds}_random_') and f.endswith(f'_{fs}.csv.gz')):
                    continue
                if ds == 'matr' and f.startswith('pred_matr_merged'):
                    continue
                d = pd.read_csv(os.path.join(ROOT, 'repcv', f), dtype={'cell': str})
                d['L'] = (d.y + d.cycle).groupby(d.cell).transform('max')
                d['stage'] = stage(d.cycle / d.L)
                groups = [('all', d)] + [(s, d[d.stage == s]) for s in ('early', 'mid', 'late')] + \
                         [(f'RUL<={c}', d[d.y <= c]) for c in (200, 100, 50)]
                for name, g in groups:
                    e = g.p - g.y
                    rows.append(dict(ds=ds, input=fs, split=f.split('_random_')[1].split('_')[0], segment=name,
                                     n_rows=len(g), MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                                     bias=np.mean(e), LMAE=100 * np.mean(np.abs(e) / g.L)))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 't11_errors_all_splits.csv'), index=False)
    s = r.groupby(['ds', 'input', 'segment'], sort=False).agg(
        MAE_seed42=('MAE', lambda v: v[r.loc[v.index, 'split'] == 'seed42'].mean()),
        RMSE_seed42=('RMSE', lambda v: v[r.loc[v.index, 'split'] == 'seed42'].mean()),
        MAE_mean=('MAE', 'mean'), MAE_sd=('MAE', 'std'), RMSE_mean=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
        LMAE_mean=('LMAE', 'mean'), bias_mean=('bias', 'mean'), n_splits=('MAE', 'size')).reset_index()
    save(s, 't11_errors_summary.csv')


# ------------------------------------------------------------------ t12 grouped SHAP, clusters, family ablation
def t12():
    fam_of = {f: k for k, v in FAMILIES.items() for f in v}
    rows, clus = [], []
    for ds in DATASETS:
        z = np.load(os.path.join(res(ds), 'shap_values.npz'), allow_pickle=True)
        sv, x, feats = z['sv'], z['x'], list(z['feats'])
        tot = np.abs(sv).mean(0).sum()
        for k, members in FAMILIES.items():
            i = [feats.index(f) for f in members]
            rows.append(dict(ds=ds, family=k, n=len(i), grouped_share=np.abs(sv[:, i].sum(1)).mean() / tot,
                             sum_individual_share=np.abs(sv[:, i]).mean(0).sum() / tot))
        rho = np.nan_to_num(stats.spearmanr(x, nan_policy='omit')[0])
        dist = 1 - np.abs(rho)
        np.fill_diagonal(dist, 0)
        lab = fcluster(linkage(squareform(dist, checks=False), 'average'), t=0.15, criterion='distance')
        for c in np.unique(lab):
            i = np.where(lab == c)[0]
            clus.append(dict(ds=ds, cluster=int(c), n=len(i), members='|'.join(feats[j] for j in i),
                             grouped_share=np.abs(sv[:, i].sum(1)).mean() / tot,
                             families='|'.join(sorted({fam_of[feats[j]] for j in i}))))
    g = pd.DataFrame(rows)
    g['grouped_rank'] = g.groupby('ds').grouped_share.rank(ascending=False)
    save(g, 't12_family_shap.csv')
    c = pd.DataFrame(clus).sort_values(['ds', 'grouped_share'], ascending=[True, False])
    c.to_csv(os.path.join(OUT, 't12_cluster_shap.csv'), index=False)
    save(c[c.n > 1].groupby('ds').head(6), 't12_cluster_shap_top.csv')
    bi = os.path.join(OUT, 't3_boot_importance.csv')
    if os.path.exists(bi):   # family rank stability over the t3 bootstrap
        b = pd.read_csv(bi)
        b['family'] = b.feature.map(fam_of)
        fr = b.groupby(['ds', 'b', 'family']).imp.sum().reset_index()
        fr['rank'] = fr.groupby(['ds', 'b']).imp.rank(ascending=False)
        save(fr.groupby(['ds', 'family'])['rank'].agg(mean_rank='mean', sd_rank='std',
                                                      p_rank1=lambda v: np.mean(v == 1)).reset_index()
             .sort_values(['ds', 'mean_rank']), 't12_family_rank_stability.csv')
    abl = []
    for ds in DATASETS:
        df, train, test, main, _ = load_stage1(res(ds))
        p = params(ds)
        sets = {'all40': main}
        for k, v in FAMILIES.items():
            sets[f'drop: {k}'] = [f for f in main if f not in v]
            sets[f'only: {k}'] = v
        folds = cell_folds(df, train, CV_SEED)
        for name, feats in sets.items():
            r2 = [r2_score(*train_test(df, a, b, feats, p)[:2]) for a, b in folds]
            yb, pb, sb = train_test(df, train, test, feats, p)
            abl.append(dict(ds=ds, subset=name, k=len(feats), cv_R2=np.mean(r2),
                            cv_se=np.std(r2, ddof=1) / np.sqrt(len(r2)), **score(yb, pb, sb)))
            print(abl[-1], flush=True)
    a = pd.DataFrame(abl)
    ref = a[a.subset == 'all40'].set_index('ds')
    a['dLMAE_vs_all40'] = a.LMAE - a.ds.map(ref.LMAE)
    a['dcvR2_vs_all40'] = a.cv_R2 - a.ds.map(ref.cv_R2)
    save(a, 't12_family_ablation.csv')


# ------------------------------------------------------------------ t13 BMS cost
def leaf_depths(booster):
    t = booster.trees_to_dataframe()
    depth = {}
    for tree, g in t.groupby('Tree'):
        par = {}
        for _, r in g.iterrows():
            for ch in (r.Yes, r.No):
                if isinstance(ch, str):
                    par[ch] = r.ID
        for nid in g.ID:
            d, n = 0, nid
            while n in par:
                n, d = par[n], d + 1
            depth[nid] = d
    return t, depth


def t13():
    rows = []
    for ds in DATASETS:
        df, train, test, main, _ = load_stage1(res(ds))
        for fs, feats in (('E40', main), ('E7', COMPACT7)):
            Xa, ya, _ = build(df, train, feats)
            Xb, _, _ = build(df, test, feats)
            Xa, Xb = prep(Xa, Xb)
            bst = fit(Xa, ya, {**params(ds), 'device': 'cpu'}).get_booster()
            bst.set_param({'nthread': 1})
            t, depth = leaf_depths(bst)
            leaves = bst.predict(xgb.DMatrix(Xb[:2000]), pred_leaf=True).astype(int)
            comps = np.array([[depth[f'{j}-{leaf}'] for j, leaf in enumerate(row)] for row in leaves]).sum(1)
            x1 = Xb[:1].astype(np.float32)
            for _ in range(50):
                bst.inplace_predict(x1)
            ts = []
            for _ in range(2000):
                t0 = time.perf_counter()
                bst.inplace_predict(x1)
                ts.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            bst.inplace_predict(Xb[:10000].astype(np.float32))
            n_split = int((t.Feature != 'Leaf').sum())
            rows.append(dict(ds=ds, input=fs, n_features=len(feats), n_trees=bst.num_boosted_rounds(),
                             n_split_nodes=n_split, n_leaves=int((t.Feature == 'Leaf').sum()),
                             json_kB=len(bst.save_raw('json')) / 1024, ubj_kB=len(bst.save_raw('ubj')) / 1024,
                             # ponytail: flat C array estimate, 8 B per split (f32 threshold, u16 feature, u16 right
                             # child) + 4 B per leaf (f32); a real codegen (treelite / m2cgen) gives the exact figure
                             c_array_kB=(8 * n_split + 4 * int((t.Feature == 'Leaf').sum())) / 1024,
                             comparisons_per_pred=comps.mean(),
                             cpu_latency_us_single=1e6 * np.median(ts),
                             cpu_us_per_row_batch=1e6 * (time.perf_counter() - t0) / min(10000, len(Xb)),
                             # ponytail: rough MCU estimate, ~10 cycles per node visit at 168 MHz (Cortex-M4F class)
                             mcu_est_us_168MHz=comps.mean() * 10 / 168, cpu=platform.processor()))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), 't13_bms_cost.csv')


# ------------------------------------------------------------------ t14 reproducibility pack
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def t14():
    import importlib
    import rul.preprocess as pp
    rep = os.path.join(OUT, 'repro')
    os.makedirs(rep, exist_ok=True)
    cells, files, cfgs = [], {}, {}
    for ds in DATASETS + ('matr_merged',):
        df, train, test, main, _ = load_stage1(res(ds))
        life = life_of(df)
        M = meta(ds, train + test, life)
        for c in train + test:
            cells.append(dict(ds=ds, cell=c, split='train' if c in train else 'test', n_cycles=int((df.cell == c).sum()),
                              eol_cycle=life[c], **{k: M.loc[c, k] for k in COND.get(ds.split('_')[0], [])}))
        cfg = importlib.import_module(f'{ds}.config')
        for p in [getattr(cfg, 'SOURCE', None), *(os.path.join(res(ds), f) for f in
                                                  ('features_v2.pkl', 'split.json', 'features.json', 'final_params.json'))]:
            if p and os.path.exists(p):
                files[os.path.relpath(p, os.path.dirname(ROOT))] = md5(p)
        cfgs[ds] = dict(features=main, xgboost_params=params(ds), split_seed=42, cv_seed=CV_SEED)
    pd.DataFrame(cells).to_csv(os.path.join(rep, 'cells.csv'), index=False)
    import catboost, lightgbm, scipy, shap, sklearn   # noqa: E401
    ver = {m.__name__: m.__version__ for m in (np, pd, xgb, sklearn, scipy, shap, lightgbm, catboost)}
    ver.update(python=sys.version.split()[0], platform=platform.platform())
    pre = {k: v for k, v in vars(pp).items() if k.isupper()}
    json.dump(dict(versions=ver, file_md5=files, preprocessing=pre, configs=cfgs), open(os.path.join(rep, 'manifest.json'), 'w'),
              indent=1, default=str)
    print(pd.DataFrame(cells).groupby(['ds', 'split']).size().to_string(), '\n', json.dumps(ver, indent=1))


# ------------------------------------------------------------------ self-check
# ------------------------------------------------------------------ t15 few-shot calibration (comment 1)
KS, DRAWS = (1, 2, 4, 8), 10
TOPUP = dict(n_estimators=100, max_depth=3, learning_rate=0.05)   # fixed before any result, never tuned on test


def line_fit(p, y):
    """y ~ a p + b by least squares; a non-positive slope is meaningless here -> shift only."""
    a, b = np.polyfit(p, y, 1)
    return (float(a), float(b)) if a > 0 else (1.0, float(np.mean(y - p)))


def draw_cells(pool, k, n, rng):
    """n random k-subsets of pool; every subset once when there are at most n of them."""
    if comb(len(pool), k) <= n:
        return [list(c) for c in combinations(pool, k)]
    return [list(rng.choice(pool, k, replace=False)) for _ in range(n)]


def t15():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    data = {ds: load_stage1(res(ds)) for ds in DATASETS}
    main = data['matr'][3]
    p = {**DEFAULTS['XGBoost'], 'device': DEV}
    scaler = lambda: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())  # noqa: E731
    frac = lambda y, s: y / pd.Series(y + s.cycle.to_numpy()).groupby(s.cell.to_numpy()).transform('max').to_numpy()  # noqa: E731
    rows = []
    for fs, feats in (('E40', main), ('E7', COMPACT7)):
        for tgt in DATASETS:
            dft, tr_t, te_t = data[tgt][:3]
            src = [d for d in DATASETS if d != tgt]
            Xb, yb, sb = build(dft, te_t, feats)
            # source models, fitted once: RUL in cycles on pooled scaling; life fraction on per-dataset scaling
            Xc, yc, Xf, yf = [], [], [], []
            for d in src:
                X, y, s = build(data[d][0], data[d][1] + data[d][2], feats)
                Xc.append(X); yc.append(y); Xf.append(scaler().fit_transform(X)); yf.append(frac(y, s))  # noqa: E702
            pool_sc = scaler().fit(np.vstack(Xc))
            m_cyc = fit(pool_sc.transform(np.vstack(Xc)), np.concatenate(yc), p)
            m_frac = fit(np.vstack(Xf), np.concatenate(yf), p)
            pb_cyc = m_cyc.predict(pool_sc.transform(Xb))
            base = dict(input=fs, target=tgt, n_test=len(te_t))
            rows.append(dict(base, K=0, draw=0, method='direct', **score(yb, pb_cyc, sb)))
            rows.append(dict(base, K=len(tr_t), draw=0, method='within_full', **score(yb, train_pred(dft, tr_t, Xb, feats, p), sb)))
            rng = np.random.default_rng(42)
            for k in KS:
                if k > len(tr_t):
                    continue
                for dr, cal in enumerate(draw_cells(tr_t, k, DRAWS, rng)):
                    Xk, yk, sk = build(dft, cal, feats)
                    life = (yk + sk.cycle.to_numpy())
                    Lbar = pd.Series(life).groupby(sk.cell.to_numpy()).max().mean()
                    cyc_k = pool_sc.transform(Xk)
                    a, b = line_fit(m_cyc.predict(cyc_k), yk)
                    tk = scaler().fit(Xk)                # target scaling from the K calibration cells only
                    Fk, Fb = tk.transform(Xk), tk.transform(Xb)
                    fk, fb = m_frac.predict(Fk), m_frac.predict(Fb)
                    a2, b2 = line_fit(fk * Lbar, yk)
                    topup = lambda m, X, y: xgb.XGBRegressor(**{**p, **TOPUP}).fit(X, y, xgb_model=m.get_booster())  # noqa: E731
                    pred = {
                        'cycles+linear': a * pb_cyc + b,
                        'cycles+topup': topup(m_cyc, cyc_k, yk).predict(pool_sc.transform(Xb)),
                        'frac+life': fb * Lbar,
                        'frac+life+linear': a2 * fb * Lbar + b2,
                        'frac+topup': topup(m_frac, Fk, frac(yk, sk)).predict(Fb) * Lbar,
                        'target_only': fit(Fk, yk, p).predict(Fb),
                        'mean_life': np.maximum(Lbar - sb.cycle.to_numpy(), 0),
                    }
                    rows += [dict(base, K=k, draw=dr, method=m, **score(yb, v, sb)) for m, v in pred.items()]
                print(fs, tgt, f'K={k} done', flush=True)
    r = pd.DataFrame(rows)
    save(r, 't15_fewshot_runs.csv')
    g = r.groupby(['input', 'target', 'K', 'method'])
    out = g.agg(n_test=('n_test', 'first'), draws=('LMAE', 'size'), LMAE=('LMAE', 'mean'), LMAE_sd=('LMAE', 'std'),
                LMAE_worst=('LMAE', 'max'), MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'), R2=('R2', 'mean')).reset_index()
    save(out.sort_values(['input', 'target', 'K', 'LMAE']), 't15_fewshot_summary.csv')


def train_pred(df, tr, Xb, feats, p):
    Xa, ya, _ = build(df, tr, feats)
    Xa, Xb = prep(Xa, Xb)
    return fit(Xa, ya, p).predict(Xb)


# ------------------------------------------------------------------ t16 condition-grouped conformal (comment 10)
def t16():
    """Calibrate on residuals from conditions the inner model never saw, then test on an unseen condition.
    Baseline (random-cell calibration) = cov_norm in t9_factor_holdout.csv and t10_coverage.csv. E40 only."""
    from sklearn.model_selection import GroupKFold
    rows = []
    for ds in DATASETS:
        df, train, test, main, _ = load_stage1(res(ds))
        life, p = life_of(df), params(ds)
        cells = train + test
        M = meta(ds, cells, life)
        cond = M[COND[ds]].astype(str).agg('|'.join, axis=1)
        tasks = [('random', 'test', train, test)]
        for f in COND[ds]:
            for lev in M[f].unique():
                te = list(M.index[M[f] == lev])
                tasks.append((f, lev, [c for c in cells if c not in te], te))
        for f, lev, tr, te in tasks:
            g = cond[tr].to_numpy()
            if len(set(g)) < 2:
                continue
            cal = []
            for a, b in GroupKFold(min(5, len(set(g)))).split(tr, groups=g):
                yb, pb, sb = train_test(df, [tr[i] for i in a], [tr[i] for i in b], main, p)
                cal.append(pd.DataFrame({'cell': sb.cell.values, 'y': yb, 'p': pb}))
            cal = pd.concat(cal)
            sig = lambda q: np.clip(q, 0, None) + 0.1 * life[tr].mean()  # noqa: E731
            w = 1 / cal.cell.map(cal.cell.value_counts()).to_numpy()      # each cell weighs the same
            res_ = np.abs(cal.y - cal.p).to_numpy()
            q_abs, q_norm = wquantile(res_, w, 1 - ALPHA), wquantile(res_ / sig(cal.p).to_numpy(), w, 1 - ALPHA)
            yt, pt, st = train_test(df, tr, te, main, p)
            L = st.cell.map(life).to_numpy()
            for m, h in (('group_abs', q_abs * np.ones(len(pt))), ('group_norm', q_norm * sig(pt))):
                rows.append(dict(ds=ds, factor=f, held_out=lev, n_train=len(tr), n_test=len(te), method=m, LMAE=100 * np.mean(np.abs(pt - yt) / L),
                                 coverage=100 * np.mean(np.abs(pt - yt) <= h), width_pct_life=100 * np.mean(2 * h / L), width_cycles=np.mean(2 * h)))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), 't16_group_conformal.csv')


# ------------------------------------------------------------------ t17 small models, real compiled size (comment 13)
GCC = os.path.expanduser(r'~\.platformio\packages\toolchain-riscv32-esp\bin\riscv32-esp-elf-')
SMALL = [(1500, 6, 0.01), (300, 4, 0.05), (100, 4, 0.1), (50, 3, 0.2)]


def flatten(bst):
    """Depth-first node arrays: left child = next node. -> (feat (255 = leaf), value, right child)."""
    t = bst.trees_to_dataframe().set_index('ID')
    feat, val, right, roots = [], [], [], []

    def walk(i):
        r, k = t.loc[i], len(feat)
        feat.append(255 if r.Feature == 'Leaf' else int(r.Feature[1:]))
        val.append(float(r.Gain if r.Feature == 'Leaf' else r.Split))
        right.append(0)
        if r.Feature != 'Leaf':
            walk(r.Yes)
            right[k] = len(feat)
            walk(r.No)

    for tree in range(bst.num_boosted_rounds()):
        roots.append(len(feat))
        walk(f'{tree}-0')
    return np.array(feat, np.uint8), np.array(val, np.float32), np.array(right, np.uint32), roots


def flat_predict(feat, val, right, roots, x, base):
    out = np.float32(base)
    visits = 0
    for n in roots:
        while feat[n] != 255:
            n, visits = (n + 1 if x[feat[n]] < val[n] else right[n]), visits + 1
        out += val[n]
    return float(out), visits


C_MODEL = """#include <stdint.h>
typedef struct {{ float v; uint32_t right_f; }} node_t;   /* right child in the low 24 bits, feature in the top 8 */
const uint32_t n_trees = {nt};
const float base_score = {base}f;
const uint32_t roots[] = {{{roots}}};
const node_t nodes[] = {{{nodes}}};
float rul_predict(const float *x) {{
    float out = base_score;
    for (uint32_t t = 0; t < n_trees; t++) {{
        uint32_t n = roots[t];
        while ((nodes[n].right_f >> 24) != 255)
            n = x[nodes[n].right_f >> 24] < nodes[n].v ? n + 1 : (nodes[n].right_f & 0xFFFFFF);
        out += nodes[n].v;
    }}
    return out;
}}
"""


def t17(feats=COMPACT7, tag=''):
    import subprocess
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    folder = os.path.join(OUT, 'bms_c')
    os.makedirs(folder, exist_ok=True)
    rows = []
    for ds in DATASETS:
        df, train, test, *_ = load_stage1(res(ds))
        (Xa, ya, _), (Xb, yb, sb) = build(df, train, feats), build(df, test, feats)
        sc = make_pipeline(SimpleImputer(strategy='median'), StandardScaler()).fit(Xa)
        Xa_, Xb_ = sc.transform(Xa).astype(np.float32), sc.transform(Xb).astype(np.float32)
        for n, depth, lr in SMALL:
            m = xgb.XGBRegressor(**{**DEFAULTS['XGBoost'], 'n_estimators': n, 'max_depth': depth, 'learning_rate': lr,
                                    'device': 'cpu'}).fit(Xa_, ya)
            bst = m.get_booster()
            base = float(json.loads(bst.save_config())['learner']['learner_model_param']['base_score'].strip('[]'))
            feat, val, right, roots = flatten(bst)
            pick = np.linspace(0, len(Xb_) - 1, 200).astype(int)
            flat = [flat_predict(feat, val, right, roots, Xb_[i], base) for i in pick]
            diff = np.abs(np.array([f[0] for f in flat]) - bst.inplace_predict(Xb_[pick]))
            assert diff.max() < 1, f'flat tree disagrees with xgboost by {diff.max()}'
            nodes = ','.join(f'{{{v:.7e}f,{(int(f) << 24) | int(r)}u}}' for f, v, r in zip(feat, val, right))
            src = os.path.join(folder, f'rul_{ds}_{n}x{depth}{tag}.c')
            with open(src, 'w') as f:
                f.write(C_MODEL.format(nt=len(roots), base=f'{base:.7e}', roots=','.join(map(str, roots)), nodes=nodes))
            obj = src[:-2] + '.o'
            subprocess.run([GCC + 'gcc', '-Os', '-march=rv32imc', '-mabi=ilp32', '-c', src, '-o', obj], check=True)
            text, data, bss = map(int, subprocess.run([GCC + 'size', obj], check=True, capture_output=True, text=True).stdout.split('\n')[1].split()[:3])
            visits = float(np.mean([f[1] for f in flat]))
            rows.append(dict(ds=ds, trees=n, depth=depth, lr=lr, **score(yb, bst.inplace_predict(Xb_), sb), nodes=len(feat),
                             flash_kB_esp32c3=(text + data) / 1024, ram_B_input=4 * len(feats), node_visits=visits,
                             # ponytail: soft-float compare (no FPU on C3), ~40 cycles per visit assumed, not measured
                             est_ms_esp32c3_160MHz=visits * 40 / 160e3))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), f't17_bms_small_models{tag}.csv')


def t17b():
    """t17 on the common top 9 without cycle_index (8 features), 2026-11-05: the small option in the paper is k = 9."""
    from rul.common import shap_ranks
    top9 = shap_ranks().index[:9].tolist()
    assert top9[8] == 'I_std', top9
    t17([f for f in top9 if f != 'cycle_index'], '_top9')


def t17c():
    """BMS input choice (2026-11-05): top 9 with vs without cycle_index, same small XGBoost models as t17.
    Per-test-cell LMAE (mean |error| / cell life), paired Wilcoxon over test cells (SNL: 3 cells, not testable)."""
    from scipy.stats import wilcoxon
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from rul.common import shap_ranks
    top9 = shap_ranks().index[:9].tolist()
    sets = {'top9': top9, 'top9_no_cycle': [f for f in top9 if f != 'cycle_index']}
    rows = []
    for ds in DATASETS:
        df, train, test, *_ = load_stage1(res(ds))
        for n, depth, lr in SMALL:
            cellerr = {}
            for name, feats in sets.items():
                (Xa, ya, _), (Xb, yb, sb) = build(df, train, feats), build(df, test, feats)
                sc = make_pipeline(SimpleImputer(strategy='median'), StandardScaler()).fit(Xa)
                m = xgb.XGBRegressor(**{**DEFAULTS['XGBoost'], 'n_estimators': n, 'max_depth': depth, 'learning_rate': lr,
                                        'device': 'cpu'}).fit(sc.transform(Xa), ya)
                pb = m.predict(sc.transform(Xb))
                e = pd.DataFrame({'cell': sb.cell.astype(str), 'ae': np.abs(pb - yb), 'cyc': sb.cycle, 'rul': yb})
                life = e.groupby('cell').cyc.max() + e.groupby('cell').rul.last()
                cellerr[name] = 100 * e.groupby('cell').ae.mean() / life
                rows_ = score(yb, pb, sb)
                cellerr[name + '_overall'] = rows_['LMAE']
            a, b = cellerr['top9'], cellerr['top9_no_cycle'].loc[cellerr['top9'].index]
            pv = wilcoxon(a, b).pvalue if len(a) >= 6 else np.nan
            rows.append(dict(ds=ds, trees=n, depth=depth, n_test_cells=len(a), LMAE_top9=cellerr['top9_overall'],
                             LMAE_no_cycle=cellerr['top9_no_cycle_overall'], diff_pt=cellerr['top9_no_cycle_overall'] - cellerr['top9_overall'],
                             cell_median_diff_pt=float(np.median(b - a)), wilcoxon_p=pv))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), 't17c_cycle_counter.csv')


# ------------------------------------------------------------------ t18 SoC-window sensitivity (comment 6)
WINDOWS = {'full': (0, 1), '20-80': (0.2, 0.8), '40-80': (0.4, 0.8), '60-100': (0.6, 1), '0-40': (0, 0.4)}


def window_soc(early):
    """SoC of each voltage grid point from a cell's early Q(V) curve, rising with voltage. NaN points stay NaN."""
    e = early - np.nanmin(early)
    soc = e / np.nanmax(e)
    v = soc[np.isfinite(soc)]
    return 1 - soc if v[0] > v[-1] else soc


def t18():
    """Partial-window capacity cut from FULL cycles (a simulation: no cell here was really cycled in the window).
    Same 4 causal features for each window: window capacity, its change, expanding slope, cycle index."""
    rows = []
    for ds in DATASETS:
        df, train, test, main, _ = load_stage1(res(ds))
        z = np.load(os.path.join(res(ds), 'raw_inputs.npz'))
        curve, cell, cyc = z['curve'].astype(float), z['cell'].astype(str), z['cycle']
        p = params(ds)
        d0 = df[['cell', 'cycle', 'RUL']].assign(cell=df.cell.astype(str))
        for name, (lo, hi) in WINDOWS.items():
            parts = []
            for c in np.unique(cell):
                i = np.flatnonzero(cell == c)
                i = i[np.argsort(cyc[i])]
                ok = np.flatnonzero(np.isfinite(curve[i]).sum(1) > 50)   # empty cycles carry no window
                if len(ok) < 20:
                    continue
                soc = window_soc(np.nanmedian(curve[i[ok[:20]]], 0))
                a, b = np.nanargmin(np.abs(soc - lo)), np.nanargmin(np.abs(soc - hi))
                q = pd.Series(np.abs(curve[i, b] - curve[i, a]), dtype=float)
                x = pd.Series(cyc[i], dtype=float)
                parts.append(pd.DataFrame({'cell': c, 'cycle': cyc[i], 'Q_w': q, 'dQ_w': q - q.iloc[ok[:10]].median(),
                                           'slope_w': (x.expanding(3).cov(q) / x.expanding(3).var()).fillna(0.0), 'cyc': x}))
            d = d0.merge(pd.concat(parts), on=['cell', 'cycle']).dropna(subset=['Q_w'])
            tr, te = d[d.cell.isin(map(str, train))], d[d.cell.isin(map(str, test))]
            for fs, cols in (('with_cycle', ['Q_w', 'dQ_w', 'slope_w', 'cyc']), ('no_cycle', ['Q_w', 'dQ_w', 'slope_w'])):
                pb = fit(tr[cols].to_numpy(float), tr.RUL.to_numpy(float), p).predict(te[cols].to_numpy(float))
                rows.append(dict(ds=ds, window=name, features=fs, n_test_cells=te.cell.nunique(), **score(te.RUL.to_numpy(float), pb, te)))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), 't18_soc_windows.csv')


# ------------------------------------------------------------------ t19 repro: extend pack, rerun from it alone (comment 14)
TUNE_DIRS = {'matr_merged': 'matr_merged_s32', 'hust': 'hust_s32', 'uconn': 'uconn_s32'}
SOURCES = {'matr': 'Severson et al. 2019, Toyota Research Institute / data.matr.io', 'hust': 'Ma et al. 2022, data repository of the paper',
           'uconn': 'UConn-ISU-ILCC LFP/Gr Battery Aging Dataset, UConn Digital Commons, CC BY 4.0',
           'snl': 'Preger et al. 2020, batteryarchive.org, BatteryLife distribution (Tan et al. 2025)'}


def t19():
    rep = os.path.join(OUT, 'repro')
    man = json.load(open(os.path.join(rep, 'manifest.json')))
    man['dataset_sources'] = SOURCES
    deep = {}
    for ds, sub in TUNE_DIRS.items():
        t = pd.read_csv(os.path.join(os.environ.get('S32_RESULTS', os.path.join(ROOT, 's32_results')), sub, 'tune.csv'))
        val = [c for c in t.columns if 'val' in c.lower()][0]
        fin = t[np.isfinite(t[val])]
        deep[ds] = {k: g.loc[g[val].idxmin()].to_dict() for k, g in fin.groupby('key')}
        man['file_md5'][f'revision/{sub}/tune.csv'] = md5(os.path.join(os.environ.get('S32_RESULTS', os.path.join(ROOT, 's32_results')), sub, 'tune.csv'))
        man['file_md5'][f'revision/{sub}/runs.csv'] = md5(os.path.join(os.environ.get('S32_RESULTS', os.path.join(ROOT, 's32_results')), sub, 'runs.csv'))
    man['deep_best_configs'] = deep
    for f in ('steps/s34_prof.py', 'steps/s32_tuned_deep.py', 'steps/s32_stats.py', 'rul/common.py', 'rul/preprocess.py'):
        man['file_md5'][f] = md5(os.path.join(ROOT, f))
    json.dump(man, open(os.path.join(rep, 'manifest.json'), 'w'), indent=1, default=str)
    # rerun: split and parameters come only from the pack; features are checked by MD5 first
    cells, rows = pd.read_csv(os.path.join(rep, 'cells.csv'), dtype={'cell': str}), []
    for ds in DATASETS:
        path = os.path.join(res(ds), 'features_v2.pkl')
        key = [k for k in man['file_md5'] if k.replace('\\', '/').endswith(f'{ds}/results/features_v2.pkl')][0]
        assert md5(path) == man['file_md5'][key], f'{ds} features changed'
        df = pd.read_pickle(path)
        c, cfg = cells[cells.ds == ds], man['configs'][ds]
        tr, te = c[c.split == 'train'].cell.tolist(), c[c.split == 'test'].cell.tolist()
        yb, pb, sb = train_test(df.assign(cell=df.cell.astype(str)), tr, te, cfg['features'], {**cfg['xgboost_params'], 'device': DEV})
        rows.append(dict(ds=ds, n_train=len(tr), n_test=len(te), **score(yb, pb, sb)))
    save(pd.DataFrame(rows), 't19_repro_rerun.csv')


def check():
    soc = window_soc(np.array([9., 7, 5, 3, 1]))
    assert soc[0] == 0 and soc[-1] == 1                   # Q falling with voltage -> SoC still rises with voltage
    a, b = line_fit(np.array([0., 1, 2]), np.array([3., 5, 7]))
    assert abs(a - 2) < 1e-9 and abs(b - 3) < 1e-9
    assert line_fit(np.array([0., 1, 2]), np.array([7., 5, 3]))[0] == 1
    rng = np.random.default_rng(0)
    assert len(draw_cells(list('abc'), 2, 10, rng)) == 3
    assert all(len(set(d)) == 2 for d in draw_cells(list('abcdefgh'), 2, 5, rng))
    assert len(draw_cells([f'c{i}' for i in range(86)], 8, 5, rng)) == 5   # huge pool: must not enumerate
    g = np.linspace(2.0, 3.5, 100)                        # peak drifts 3.15 -> 3.10 V and fades; bigger false spike at 2.0 V
    ic = np.array([(5 - 0.02 * i) * np.exp(-((g - (3.15 - 0.001 * i)) / 0.03) ** 2) for i in range(50)])
    ic[:, :3] = 20
    ic[10] = np.nan                                       # empty cycle must not lose the track
    v, h = track_peak(ic, g, 3.15)
    assert np.isnan(v[10]) and np.nanmax(np.abs(v - (3.15 - 0.001 * np.arange(50)))) < 0.005 and abs(h[0] - 5) < 0.2
    assert band_peak(np.arange(100.0), g, 2.8, 3.35) is None   # rising to the window edge is not a peak
    assert jaccard('abc', 'abd') == 0.5
    assert kuncheva(list('ab'), list('ab'), 10) == 1
    assert abs(wquantile(np.arange(10.0), np.ones(10), 0.9) - 8) < 1e-9
    v = np.r_[np.zeros(5), np.ones(5)]
    assert eta2(v, ['a'] * 5 + ['b'] * 5, np.zeros(10)) == 1
    assert eta2(v, ['a', 'b'] * 5, np.zeros(10)) < 0.1
    print('s34 helpers OK')


TASKS = dict(t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6, t8=t8, t9=t9, t10=t10, t11=t11, t12=t12, t13=t13, t14=t14,
             t15=t15, t16=t16, t17=t17, t17b=t17b, t17c=t17c, t18=t18, t19=t19, check=check)
if __name__ == '__main__':
    todo = sys.argv[1:] or ['check']
    for name in (list(TASKS)[:-1] if todo == ['all'] else todo):
        t0 = time.time()
        print(f'\n######## {name}', flush=True)
        TASKS[name]()
        print(f'######## {name} done in {time.time() - t0:.0f} s', flush=True)
