"""42-feature extraction (port of the MATR feature notebook, verified by `check`), applied per dataset
(train and test within one dataset, no cross-dataset transfer).

    python multi_dataset.py check   # port check vs notebook MATR cache (features + R2 0.9485)
    python multi_dataset.py matr    # featurize the 124 MATR cells -> all_data_42features.pkl (then steps.s28_matr_merge build)
    python multi_dataset.py hust    # featurize HUST (cached) + evaluate

Feature code is copied verbatim from the notebook (cell numbers noted); only the h5 I/O
is replaced by per-dataset loaders yielding (cyc, I, V, Qc, Qd, T, t, rul) in A / Ah / s.
"""
import glob, os, sys, pickle, zipfile, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rul.paths import FEATURES as RES, RAW as DATA  # noqa: E402
FIGS = os.path.join(RES, 'figs')
HUST_ZIP = os.path.join(DATA, 'nsc7hnsg4s-2', 'nsc7hnsg4s-2', 'our_data.zip')
MATR_B2 = os.path.join(DATA, 'TRI', 'MATR', 'MATR_batch_20170630.mat')
MATR_FILES = {'b1': 'MATR_batch_20170512.mat', 'b2': 'MATR_batch_20170630.mat', 'b3': 'MATR_batch_20180412.mat'}
SEED = 42


# ---------------- copied from notebook cell 2 ----------------
def compute_dcir(I, V):
    I = np.array(I)
    V = np.array(V)
    dI = np.diff(I)
    step_idx = np.where(np.abs(dI) > 0.2)[0]
    dcir_values = []
    for idx in step_idx:
        if idx + 1 >= len(V):
            continue
        delta_I = I[idx + 1] - I[idx]
        delta_V = V[idx + 1] - V[idx]
        if abs(delta_I) < 1e-5:
            continue
        r = abs(delta_V / delta_I)
        if 0 < r < 0.5:
            dcir_values.append(r)
    if len(dcir_values) == 0:
        return np.nan, np.nan
    return np.median(dcir_values), np.std(dcir_values)


def extract_features(I, V, Qc, Qd, T, t, v_cv=3.6, min_cv=10, cmt_frac=0.05):
    if len(V) < 30:
        return None
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6
    dV_dt = np.gradient(V) / dt
    dI_dt = np.gradient(I) / dt
    dQ = np.gradient(Qc)
    dV = np.gradient(V)
    dV[dV == 0] = 1e-6
    dQ[dQ == 0] = 1e-6
    ic = dQ / dV
    dvic = dV / dQ
    charge_mask = I > 0
    if charge_mask.sum() < 20:
        return None
    # v_cv: charge cut-off voltage, min_cv: CV-phase points needed (changes vs notebook; MATR defaults kept)
    cc_mask = charge_mask & (V < v_cv)
    cv_mask = charge_mask & (V >= v_cv)
    if np.sum(cc_mask) < 10 or np.sum(cv_mask) < min_cv:
        return None
    V_cc, I_cc = V[cc_mask], I[cc_mask]
    V_cv, I_cv = V[cv_mask], I[cv_mask]
    dcir_median, dcir_std = compute_dcir(I, V)
    vmt = np.nan
    idx = np.where((V >= 3.55) & charge_mask)[0]
    if len(idx) > 0:
        vmt = t[idx[0]]
    cmt = np.nan
    I_charge_peak = np.max(I[charge_mask])
    cmt_threshold = max(cmt_frac * I_charge_peak, 1e-3)
    idx = np.where((I < cmt_threshold) & charge_mask)[0]
    if len(idx) > 0:
        cmt = t[idx[0]]
    tmt = np.nan
    if T is not None and len(T) > 0:
        tmt = t[np.argmax(T)]
    return {
        "V_mean": np.mean(V), "V_std": np.std(V), "I_mean": np.mean(I), "I_std": np.std(I),
        "dV_dt_mean": np.mean(dV_dt), "dV_dt_std": np.std(dV_dt),
        "dI_dt_mean": np.mean(dI_dt), "dI_dt_std": np.std(dI_dt),
        "IC_max": np.max(ic), "IC_mean": np.mean(ic),
        "CC_V_mean": np.mean(V_cc), "CC_V_std": np.std(V_cc),
        "CC_I_mean": np.mean(I_cc), "CC_I_std": np.std(I_cc), "CC_length": len(V_cc),
        "CV_I_mean": np.mean(I_cv), "CV_I_std": np.std(I_cv), "CV_length": len(V_cv),
        "DVA_max": np.max(dvic), "DVA_mean": np.mean(dvic),
        "Discharge_capacity": np.max(Qd),
        "DCIR_median": dcir_median, "DCIR_std": dcir_std,
        "VMT": vmt, "CMT": cmt, "TMT": tmt,
    }


# ---------------- copied from notebook cell 4 ----------------
VOLTAGE_GRID = np.linspace(2.0, 3.5, 1000)
LOOKBACK = 10
DQV_KEYS = ('dQV_min', 'dQV_mean', 'dQV_var', 'dQV_skew', 'dQV_kurt', 'dQV_at_2V')


def discharge_qv_curve(V, I, Qd, grid):
    mask = (I <= 0) & (Qd > 0)
    if mask.sum() < 10:
        return None
    v, q = V[mask], Qd[mask]
    order = np.argsort(v)
    v, q = v[order], q[order]
    v_u, idx = np.unique(v, return_index=True)
    if len(v_u) < 5:
        return None
    return np.interp(grid, v_u, q[idx], left=np.nan, right=np.nan)


def dqv_stats(dq):
    valid = dq[~np.isnan(dq)]
    if len(valid) < 50:
        return {k: np.nan for k in DQV_KEYS}
    m = valid.mean()
    s2 = valid.var(ddof=1)
    s = np.sqrt(s2) if s2 > 0 else 1e-12
    skew = np.mean((valid - m) ** 3) / (s ** 3)
    kurt = np.mean((valid - m) ** 4) / (s2 ** 2)
    at_2v = dq[0]
    return {
        'dQV_min': np.log10(abs(valid.min()) + 1e-12),
        'dQV_mean': np.log10(abs(m) + 1e-12),
        'dQV_var': np.log10(abs(s2) + 1e-12),
        'dQV_skew': np.log10(abs(skew) + 1e-12),
        'dQV_kurt': np.log10(abs(kurt) + 1e-12),
        'dQV_at_2V': np.log10(abs(at_2v) + 1e-12) if not np.isnan(at_2v) else np.nan,
    }


def featurize(cycles, v_cv=3.6, min_cv=10, cmt_frac=0.05):
    """cycles: iterable of (cyc, I, V, Qc, Qd, T, t, rul), cyc ascending.
    Returns {cyc: feat} with the notebook's 42 features in the notebook's key order
    (cell 3 base + Delta_Q_cycle, cell 4 dQV, cell 5 T/CE/Charge_time, cell 6 trends)."""
    tmp, qv, extra = {}, {}, {}
    for cyc, I, V, Qc, Qd, T, t, rul in cycles:
        feat = extract_features(I, V, Qc, Qd, T, t, v_cv, min_cv, cmt_frac)    # cell 3
        if feat is None:
            continue
        feat['RUL'] = rul
        tmp[cyc] = feat
        qv[cyc] = discharge_qv_curve(V, I, Qd, VOLTAGE_GRID)  # cell 4
        extra[cyc] = (T, Qc, I, t)                            # cell 5 inputs
    cycs = sorted(tmp)
    for k, cyc in enumerate(cycs):                            # cell 3: Delta_Q_cycle
        tmp[cyc]['Delta_Q_cycle'] = 0.0 if k == 0 else (
            tmp[cyc]['Discharge_capacity'] - tmp[cycs[k - 1]]['Discharge_capacity'])
    for cyc in cycs:                                          # cell 4
        cur, prev = qv.get(cyc), qv.get(cyc - LOOKBACK)
        tmp[cyc].update({k: np.nan for k in DQV_KEYS} if cur is None or prev is None
                        else dqv_stats(cur - prev))
    for cyc in cycs:                                          # cell 5
        T, Qc, I, t = extra[cyc]
        T = np.array([]) if T is None else T                  # ponytail: HUST has no T -> NaN features
        qc = np.max(Qc) if len(Qc) else np.nan
        qd = tmp[cyc].get('Discharge_capacity', np.nan)
        ce = qd / qc if qc and not np.isnan(qc) and qc > 0 else np.nan
        cm = I > 0
        tmp[cyc].update({
            'T_mean': np.mean(T) if len(T) else np.nan,
            'T_max': np.max(T) if len(T) else np.nan,
            'T_std': np.std(T) if len(T) else np.nan,
            'T_range': (np.max(T) - np.min(T)) if len(T) else np.nan,
            'CE': ce,
            'Charge_time': t[cm].max() - t[cm].min() if cm.sum() >= 2 else np.nan,
        })
    t_sum, dmin = 0.0, np.inf                                 # cell 6
    early = [tmp[c].get('DCIR_median', np.nan) for c in cycs[:5]]
    early = [v for v in early if not np.isnan(v)]
    base = np.median(early) if early else np.nan
    for cyc in cycs:
        f = tmp[cyc]
        if not np.isnan(f['T_mean']):
            t_sum += f['T_mean']
        if not np.isnan(f['DCIR_median']):
            dmin = min(dmin, f['DCIR_median'])
        d = f['DCIR_median']
        f.update({'Cum_T_integral': t_sum,
                  'DCIR_min_so_far': dmin if np.isfinite(dmin) else np.nan,
                  'DCIR_drift': d - base if not np.isnan(d) and not np.isnan(base) else np.nan})
    return tmp


# ---------------- loaders ----------------
def matr_cycles(f, cell):
    """copied from notebook cell 3 (h5 read loop), batch file opened by caller."""
    batch = f['batch']
    eol = int(np.array(f[batch['cycle_life'][cell, 0]][()]).flatten()[0])
    cg = f[batch['cycles'][cell, 0]]
    for j in range(1, cg['I'].shape[0]):
        if j >= eol:
            break
        try:
            a = [np.hstack(f[cg[k][j, 0]][()]) for k in ('I', 'V', 'Qc', 'Qd', 'T', 't')]
        except Exception:
            continue
        yield (j, a[0], a[1], a[2], a[3], a[4], a[5], eol - j)


def hust_cells():
    z = zipfile.ZipFile(HUST_ZIP)
    for n in sorted(x for x in z.namelist() if x.endswith('.pkl')):
        name = os.path.basename(n)[:-4]
        c = pickle.loads(z.read(n))[name]
        yield name, hust_cycles(c)


def hust_cycles(c):
    for cyc in sorted(c['data']):
        d = c['data'][cyc]
        I = d['Current (mA)'].to_numpy() / 1000
        V = d['Voltage (V)'].to_numpy()
        cap = d['Capacity (mAh)'].to_numpy() / 1000
        t = d['Time (s)'].to_numpy().astype(float)
        dis = I < 0
        # HUST logs one capacity counter: counts up on charge, down on discharge.
        # MATR convention: Qc holds at its max through discharge, Qd counts up from 0.
        qmax = cap[~dis].max() if (~dis).any() else 0.0
        Qc = np.where(dis, qmax, cap)
        Qd = np.where(dis, qmax - cap, 0.0)
        yield (cyc, I, V, Qc, Qd, None, t, int(c['rul'][cyc]))


UCONN = os.path.join(DATA, 'uni coni')
UCONN_EOL = 0.96       # EOL (dataset README): C/3 RPT capacity <= 80% of nominal 1.2 Ah, as MATR/HUST
# UConn cycles a partial window (60/80/100% DoD: discharge cut-off 3.163/3.070/2.0 V); the notebook
# grid starts at 2.0 V, so use the window every cell covers. dQV_at_2V = value at the grid's low end.
UCONN_GRID = np.linspace(3.17, 3.5, 1000)


def uconn_rpt(cell):
    """-> (cycles done before each cycling block, keyed by Week Number; interpolated EOL cycle).
    Block lengths come from the cycling files: RPT 'Num Cycles' is empty at week 21 for every cell."""
    fs = glob.glob(os.path.join(UCONN, 'cycling_part*', f'cycling_cell_{cell}_part*.csv'))
    blk = pd.concat(pd.read_csv(f, usecols=['Week Number', 'Cycle Number']) for f in fs
                    ).groupby('Week Number')['Cycle Number'].max()
    fs = glob.glob(os.path.join(UCONN, 'rpt_data', 'rpt_data', f'rpt_cell_{cell}_part*.csv'))
    d = pd.concat(pd.read_csv(f, usecols=['Week Number', 'Segment Key', 'Capacity (Ah)', 'Num Cycles']) for f in fs)
    n = d.groupby('Week Number')['Num Cycles'].first()        # cycles since the previous RPT (logged)
    q = d[d['Segment Key'] == 'ref_dchg'].groupby('Week Number')['Capacity (Ah)'].max()   # C/3 capacity
    assert q.index.equals(n.index), f'cell {cell}: RPT without reference discharge'
    cum = np.array([blk[blk.index < w].sum() for w in q.index], float)   # cycles done when each RPT runs
    last = np.array([blk[blk.index < w].iloc[-1] for w in q.index[1:]], float)   # block just before each RPT
    logged = n.iloc[1:].to_numpy()
    ok = ~np.isnan(logged)                                    # (cell 60 skips the week-38 RPT: 2 blocks, 1 RPT)
    assert (last[ok] == logged[ok]).all(), f'cell {cell}: cycling blocks disagree with RPT Num Cycles'
    q = q.to_numpy()
    k = int(np.argmax(q <= UCONN_EOL))
    assert q[k] <= UCONN_EOL and k > 0, f'cell {cell} never reaches {UCONN_EOL} Ah (min {q.min():.3f})'
    eol = cum[k - 1] + (q[k - 1] - UCONN_EOL) / (q[k - 1] - q[k]) * (cum[k] - cum[k - 1])
    return dict(zip(blk.index, blk.cumsum() - blk)), int(round(eol))


def _part(f):
    return int(f.rsplit('_part', 1)[1][:-4])


def uconn_cycles(cell):
    """Cycling files are 1M-row chunks; Cycle Number restarts after every weekly RPT."""
    offset, eol = uconn_rpt(cell)
    fs = sorted(glob.glob(os.path.join(UCONN, 'cycling_part*', f'cycling_cell_{cell}_part*.csv')), key=_part)
    cols = ['Week Number', 'Cycle Number', 'Time (s)', 'Voltage (V)', 'Current (A)', 'Capacity (Ah)']
    left = None
    for f in fs + [None]:
        if f is None:
            d, left = left, None
        else:
            d = pd.read_csv(f, usecols=cols)
            d['g'] = d['Week Number'].map(offset).astype(int) + d['Cycle Number']
            d = d if left is None else pd.concat([left, d])
            left = d[d.g == d.g.iloc[-1]]                     # last cycle may continue in the next part
            d = d[d.g != d.g.iloc[-1]]
        for cyc, x in d.groupby('g', sort=False):
            if cyc >= eol:
                return
            x = x.sort_values('Time (s)', kind='stable')
            I, V, cap = (x[c].to_numpy(float) for c in ('Current (A)', 'Voltage (V)', 'Capacity (Ah)'))
            t = x['Time (s)'].to_numpy(float)
            # counter resets every step; MATR convention: Qc held through discharge, Qd from 0
            Qc = np.maximum.accumulate(np.where(I > 0, cap, 0.0))
            Qd = np.maximum.accumulate(np.where(I < 0, cap, 0.0))
            yield (int(cyc), I, V, Qc, Qd, None, t - t[0], int(eol - cyc))

def uconn_featurize(cell):
    global VOLTAGE_GRID
    VOLTAGE_GRID = UCONN_GRID                                 # per worker process (spawned on Windows)
    feats = featurize(uconn_cycles(cell), v_cv=3.595)
    print(f'  {cell}: {len(feats)} cycles', flush=True)
    return cell, feats


# ---------------- preprocessing (notebook cells 48, 49, 50) ----------------
def clean_trend(df):
    """v2 features: kernel-5 rolling-median cleaning (cell 48), trend features (cell 49),
    CE / DCIR history recomputed from the cleaned signals (cell 50). Adds 4 features."""
    from scipy.ndimage import median_filter
    out = []
    for _, g in df.groupby('cell', sort=False):
        g = g.sort_values('cycle').copy()
        orig_dcap, orig_ce = g['Discharge_capacity'].to_numpy(float), g['CE'].to_numpy(float)
        for f in ('Discharge_capacity', 'DCIR_median', 'DCIR_std'):            # cell 48
            raw = g[f].to_numpy(float)
            mask = ~np.isnan(raw)
            if mask.sum() < 5:
                continue
            g[f] = median_filter(np.where(mask, raw, np.nanmedian(raw)), size=5, mode='nearest')
        cap = g['Discharge_capacity'].to_numpy(float)                          # cell 49
        g['cycle_index'] = g['cycle'].astype(float)   # notebook name 'cycle' (clashes with our row key)
        g['Cum_Delta_Q'] = cap - cap[0]
        g['Delta_Q_cycle'] = np.r_[0.0, np.diff(cap)]
        for f in ('Discharge_capacity', 'V_std'):
            v = g[f].to_numpy(float)
            g[f + '_slope'] = [float(np.polyfit(np.arange(k + 1, dtype=float), v[:k + 1], 1)[0]) if k >= 2 else 0.0
                               for k in range(len(v))]
        ok = ~np.isnan(orig_dcap) & ~np.isnan(orig_ce) & (orig_ce != 0) & (orig_dcap != 0)   # cell 50
        g.loc[ok, 'CE'] = cap[ok] / (orig_dcap[ok] / orig_ce[ok])
        d = g['DCIR_median'].to_numpy(float)
        g['DCIR_min_so_far'] = np.fmin.accumulate(d)
        e = d[:5][~np.isnan(d[:5])]
        g['DCIR_drift'] = d - np.median(e) if len(e) else np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True)


def segment_report(y, yp):
    """notebook cell 52: errors by proximity to EOL."""
    rows = []
    for name, cut in (('Full', None), ('Last400', 400), ('Last200', 200), ('Last100', 100), ('Last50', 50)):
        m = np.ones_like(y, bool) if cut is None else y <= cut
        if m.sum() >= 5:
            rows.append(dict(Segment=name, MAE=mean_absolute_error(y[m], yp[m]),
                             RMSE=np.sqrt(mean_squared_error(y[m], yp[m])), n=int(m.sum())))
    return pd.DataFrame(rows)


# ---------------- evaluation (notebook cells 7, 8, 27) ----------------
def to_frame(all_data):
    rows = [{'cell': k, 'cycle': c, **f} for k, cd in all_data.items() for c, f in sorted(cd.items())]
    return pd.DataFrame(rows)


def build(df, cells, feats):
    s = df[df.cell.isin(cells)]
    s = s.set_index('cell').loc[cells].reset_index()          # notebook order: cell by cell
    return s[feats].to_numpy(float), s['RUL'].to_numpy(float), s


def prep(Xtr, Xte):
    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr))
    return Xtr, sc.transform(imp.transform(Xte))


def evaluate(df, ds, cell_order=None):
    feats = [c for c in df.columns if c not in ('cell', 'cycle', 'RUL')]
    dropped = [c for c in feats if df[c].nunique() <= 1]    # all-NaN or constant (e.g. no T sensor)
    feats = [c for c in feats if c not in dropped]
    fill = df[feats].notna().mean().sort_values()
    print(f'[{ds}] {df.cell.nunique()} cells, {len(df)} cycles, {len(feats)} features'
          f'  dropped (no information): {dropped}')
    print(f'[{ds}] lowest fill rates:\n{fill.head(6).round(3).to_string()}')
    for f, sign in (('dQV_var', -1), ('CE', 1)):              # notebook cell 4/5 sanity checks
        if f in feats:
            ok = df[[f, 'RUL']].dropna()
            print(f'[{ds}] corr({f}, RUL) = {np.corrcoef(ok[f], ok.RUL)[0, 1]:.3f}  (expected sign {sign:+d})')

    cell_list = list(cell_order or dict.fromkeys(df.cell))
    np.random.seed(SEED)
    np.random.shuffle(cell_list)                              # cell 7 (in place, reused by cell 27)
    split = int(0.7 * len(cell_list))
    tr_c, te_c = cell_list[:split], cell_list[split:]
    Xtr, ytr, _ = build(df, tr_c, feats)
    Xte, yte, te = build(df, te_c, feats)
    Xtr, Xte = prep(Xtr, Xte)
    m = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.9,
                         colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=1.0,
                         objective='reg:squarederror', random_state=SEED, verbosity=0)
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    r2, rmse, mae = r2_score(yte, yp), np.sqrt(mean_squared_error(yte, yp)), mean_absolute_error(yte, yp)
    np.random.seed(SEED)                                      # cell 8 cell-bootstrap CI
    tc = te.cell.to_numpy()
    uc = np.unique(tc)
    idx = {c: np.where(tc == c)[0] for c in uc}
    boot = []
    for _ in range(1000):
        s = np.concatenate([idx[c] for c in np.random.choice(uc, size=len(uc), replace=True)])
        boot.append(r2_score(yte[s], yp[s]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f'[{ds}] split ({len(tr_c)}/{len(te_c)} cells): R2={r2:.4f} [{lo:.3f},{hi:.3f}]'
          f'  RMSE={rmse:.2f}  MAE={mae:.2f}')

    # SHAP via xgboost's own TreeSHAP: shap 0.49 can't parse xgboost 3 base_score (same fix as shap_train_rank.py)
    sv = np.abs(m.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]).mean(0)
    top = pd.DataFrame({'Feature': feats, 'mean_abs_SHAP': sv}).sort_values('mean_abs_SHAP', ascending=False)
    top.head(10).to_csv(os.path.join(RES, f'shap_top10_{ds}.csv'), index=False)
    print(f'[{ds}] SHAP top-10: {top.Feature.head(10).tolist()}')
    parity(yte, yp, r2, ds)

    life = df.groupby('cell').cycle.max()                     # cell 27
    lives = np.array([life[c] for c in cell_list])
    t33, t66 = np.percentile(lives, 33.3), np.percentile(lives, 66.6)
    strata = np.array([0 if v <= t33 else (1 if v <= t66 else 2) for v in lives])
    cr2, crm, cma, oof_t, oof_p = [], [], [], [], []
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for a, b in skf.split(np.arange(len(cell_list)), strata):
            Xa, ya, _ = build(df, [cell_list[i] for i in a], feats)
            Xb, yb, _ = build(df, [cell_list[i] for i in b], feats)
            Xa, Xb = prep(Xa, Xb)
            mm = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.9,
                                  colsample_bytree=0.8, reg_lambda=3.0, random_state=SEED, verbosity=0)
            mm.fit(Xa, ya)
            yh = mm.predict(Xb)
            cr2.append(r2_score(yb, yh)); crm.append(np.sqrt(mean_squared_error(yb, yh)))
            cma.append(mean_absolute_error(yb, yh))
            if seed == 0:                                     # cell 57: seed-0 pooled OOF
                oof_t.append(yb); oof_p.append(yh)
    cr2, crm, cma = map(np.array, (cr2, crm, cma))
    h = stats.t.ppf(0.975, len(cr2) - 1) * cr2.std(ddof=1) / np.sqrt(len(cr2))
    print(f'[{ds}] CV 10x5: R2={cr2.mean():.4f}+/-{cr2.std(ddof=1):.4f} [{cr2.mean()-h:.3f},{cr2.mean()+h:.3f}]'
          f'  RMSE={crm.mean():.2f}+/-{crm.std(ddof=1):.2f}  MAE={cma.mean():.2f}+/-{cma.std(ddof=1):.2f}')

    seg = pd.concat([segment_report(yte, yp).assign(eval='split'),
                     segment_report(np.concatenate(oof_t), np.concatenate(oof_p)).assign(eval='cv_oof_seed0')])
    seg.round(2).to_csv(os.path.join(RES, f'segments_{ds}.csv'), index=False)
    print(f'[{ds}] segment errors:')
    print(seg.round(2).to_string(index=False))

    row = dict(dataset=ds, n_cells=len(cell_list), n_cycles=len(df), n_features=len(feats),
               dropped=';'.join(dropped), split_R2=r2, split_R2_lo=lo, split_R2_hi=hi,
               split_RMSE=rmse, split_MAE=mae, cv_R2=cr2.mean(), cv_R2_sd=cr2.std(ddof=1),
               cv_R2_lo=cr2.mean() - h, cv_R2_hi=cr2.mean() + h, cv_RMSE=crm.mean(), cv_MAE=cma.mean())
    out = os.path.join(RES, 'multi_dataset_summary.csv')
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    old = old[old.get('dataset', pd.Series(dtype=str)) != ds] if len(old) else old
    pd.concat([old, pd.DataFrame([row])]).to_csv(out, index=False)
    return row


def parity(y, yp, r2, ds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.scatter(y, yp, s=2, alpha=0.3, color='#1f77b4', rasterized=True)
    lim = [0, max(y.max(), yp.max()) * 1.02]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set(xlim=lim, ylim=lim, xlabel='Actual RUL (cycles)', ylabel='Predicted RUL (cycles)')
    ax.text(0.04, 0.93, f'{ds.upper()}  $R^2$ = {r2:.3f}', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f'parity_{ds}.png'), dpi=600)
    plt.close(fig)


def featurize_dataset(ds, cells_iter, v_cv=3.6):
    path = os.path.join(RES, f'features_{ds}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    all_data = {}
    for name, cyc_iter in cells_iter:
        all_data[name] = featurize(cyc_iter, v_cv)
        print(f'  {name}: {len(all_data[name])} cycles', flush=True)
    df = to_frame(all_data)
    df.to_csv(path, index=False)
    return df


def check():
    """Port check against the notebook's own MATR output (results/all_data_42features.pkl)."""
    import h5py
    ref = pickle.load(open(os.path.join(RES, 'all_data_42features.pkl'), 'rb'))
    with h5py.File(MATR_B2, 'r') as f:
        for cell in (0, 1):
            mine, theirs = featurize(matr_cycles(f, cell)), ref[f'b2_cell_{cell}']
            assert sorted(mine) == sorted(theirs), 'cycle set differs'
            c = sorted(mine)[len(mine) // 2]
            assert list(mine[c]) == list(theirs[c]) and len(mine[c]) == 43, 'feature keys/order differ'
            for cy in mine:
                a = np.array(list(mine[cy].values()), float)
                b = np.array(list(theirs[cy].values()), float)
                assert np.allclose(a, b, equal_nan=True, rtol=1e-9), f'b2_cell_{cell} cycle {cy} differs'
            print(f'b2_cell_{cell}: all {len(mine)} cycles x 42 features match notebook')
    row = evaluate(to_frame(ref), 'matr', cell_order=list(ref))
    assert abs(row['split_R2'] - 0.9461) < 5e-4, f"split R2 {row['split_R2']:.4f} != 0.9461 (CHECKPOINT: xgboost 3.1.2 baseline)"
    row = evaluate(clean_trend(to_frame(ref)), 'matr_v2', cell_order=list(ref))
    assert abs(row['cv_R2'] - 0.9159) < 0.005, f"v2 CV R2 {row['cv_R2']:.4f} vs notebook 0.9159"
    print('PORT CHECK PASSED')


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)
    ds = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if ds == 'check':
        check()
    elif ds == 'matr':
        # the 124 cells of splits/cells.csv (keys b1/b2/b3_cell_<index in the .mat file>, 41/43/40), source labels;
        # then `python -m steps.s28_matr_merge build` joins b1c0-4 with their batch-2 continuations (the paper's MATR)
        import h5py
        path = os.path.join(RES, 'all_data_42features.pkl')
        cells = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'splits', 'cells.csv'))
        keys = sys.argv[2].split(',') if len(sys.argv) > 2 else list(cells.loc[cells.ds == 'matr', 'cell'])  # smoke: a few keys
        all_data = {}
        for b, fn in MATR_FILES.items():
            with h5py.File(os.path.join(DATA, 'TRI', 'MATR', fn), 'r') as f:
                for k in sorted((k for k in keys if k.startswith(b + '_')), key=lambda k: int(k.split('_')[-1])):
                    all_data[k] = featurize(matr_cycles(f, int(k.split('_')[-1])))
                    print(f'  {k}: {len(all_data[k])} cycles', flush=True)
        pickle.dump(all_data, open(path, 'wb'))
        print(f'wrote {path} ({len(all_data)} cells)')
    elif ds == 'hust':
        # HUST CV plateau logs at 3.5996-3.5999 V (never >= 3.6): 5 mV tolerance, calibration knob
        # screening (results/hust_screening.csv): no gaps, all cells end at 80%, no spikes -> no exclusions
        df = featurize_dataset('hust', hust_cells(), v_cv=3.595)
        evaluate(df, 'hust')
        evaluate(clean_trend(df), 'hust_v2')
    elif ds == 'uconn':
        from multiprocessing import Pool
        path = os.path.join(RES, 'features_uconn.csv')
        cells = sorted({os.path.basename(f).split('_')[2] for f in glob.glob(os.path.join(UCONN, 'cycling_part*', '*.csv'))})
        cells = cells[:int(sys.argv[2])] if len(sys.argv) > 2 else cells   # smoke: first n cells
        with Pool(6) as p:                                    # ponytail: ~1 GB RAM per worker
            to_frame(dict(p.map(uconn_featurize, cells))).to_csv(path, index=False)
        print(f'wrote {path}')
    else:
        sys.exit(f'unknown dataset {ds!r} (use check | matr | hust | uconn)')
