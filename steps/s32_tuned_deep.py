"""Step 32 (journal revision, review 2026-10-14): s18_hpc.py + equal-information inputs + deep-model tuning.
Everything of s18 below still holds; additions:
  inputs  E36  E40 without the 4 history features (cycle_index, Cum_Delta_Q, Discharge_capacity_slope, V_std_slope)
          E7   compact set (top-8 without cycle_index)
          RSH  RS + 5 extra channels, each a constant over the 128 points: the 4 history features + cycle duration
          RCH  RC + the same 5 channels                     (duration from raw_dur.npz, step 13b; NaN -> 0 if absent)
  --tune N  random search of N deep-learning configurations per (input, model) on the validation cells (seed 0,
          --tune-epochs), then the best configuration is trained with every --seeds seed. Search space: lr, width d,
          dropout, weight decay, batch, window (sequence models). Trials go to OUT/tune.csv (resumable).
          XGBoost keeps its step-3 tuned parameters (30 random candidates), so N = 30 matches its budget.
          To fit the DICC budget: tuning runs once per input family (E40, RS, RC; E36, RSH, RCH reuse that setting)
          and, with --cv, on fold 0 only (later folds reuse it).
  models  tcn       TCN / CNN-TCN: causal dilated residual convolutions over the window of cycles (Bai et al. 2018)
          patchtst  PatchTST / CNN-PatchTST: channel-independent patch Transformer over the window (Nie et al. 2023);
                    on E inputs the features themselves are the channels, on R inputs the 1D-CNN cycle embedding
                    projected to 16 channels
          Both take the same encoders, windows, training loop and search space as lstm/transformer; --tune also
          tunes batlinet (dropout is not used by BatLiNet).
Default output folder DATA/s32. Data folder also needs raw_dur.npz for RSH/RCH duration (else history only).

Original s18 header:
Raw signals vs engineered features, XGBoost vs deep learning, for any of the 4 datasets. Standalone.
Models, inputs, seeds and training budget are IDENTICAL to s18_matr.py (the MATR run), so all datasets compare
directly; only the dataset is a parameter. Same output folder name (DIR/s18) and runs.csv columns.

    python s18_hpc.py --dataset hust --data DIR                 # all inputs x models; 3 DL seeds
    python s18_hpc.py --dataset uconn --data DIR --models xgb,single --inputs E8,RC
    python s18_hpc.py --dataset snl --data DIR --cv             # cell-wise CV (training cells; all cells if < 20)
    python s18_hpc.py --dataset matr --data DIR --cv --policy-cv --models xgb   # leave-policy-out CV
    python s18_hpc.py --dataset hust --data DIR --smoke         # 12 cells, 2 epochs: checks the setup in ~1 min

DIR must hold these files from RUL_multidataset/<dataset>/results/ (steps 1, 3, 13):
    features_v2.pkl  split.json  features.json  final_params.json  raw_inputs.npz
Results go to DIR/s18/ (or --out): runs.csv (one row per run), summary_<split|cv>.csv, preds/*.csv.gz.

Inputs (one row = one cycle, the same rows and train/test cells as the paper):
  E40  the 40 main engineered features            E8  common top-8 (step 12 order, listed in TOP8 below)
  RS   raw V(t), I(t) of the cycle [2, 128]
  RC   raw discharge Q(V) and dQ(V) = Q(V) - Q(V) of the cell's first cycle with a curve [2, 100] (Severson's dQ(V))
Models (key -> name on E inputs / on R inputs):
  xgb          XGBoost / XGBoost (flattened)     step-3 tuned params, single cycle
  single       MLP / 1D-CNN                       single cycle
  lstm         LSTM / CNN-LSTM                    last --window cycles of the cell (past only) -> LSTM
  transformer  Transformer / CNN-Transformer      same window -> 2-layer Transformer encoder
  batlinet     BatLiNet (Zhang et al., Nat. Mach. Intell. 7, 270 (2025)) on the same window, see class BatLiNet
Scalers and early stopping use training cells only (early stopping: 10% of the training cells).
Built for 1-hour interactive sessions: finished runs are skipped on restart, and a deep-learning run cut off
mid-training resumes from its last epoch (DIR/s18/ckpt/). Rerun the same command in the next session.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED, CV_SEED = 42, 0
TOP8 = ['Cum_Delta_Q', 'Discharge_capacity_slope', 'Discharge_capacity', 'Charge_time', 'cycle_index',
        'V_std_slope', 'VMT', 'I_std']   # features_shap.xlsx 'Overall' order, first 8 (step 12)
HIST = ['cycle_index', 'Cum_Delta_Q', 'Discharge_capacity_slope', 'V_std_slope']
SPACE = dict(lr=[3e-4, 1e-3, 3e-3], d=[32, 64, 128], dropout=[0.0, 0.1, 0.3], wd=[1e-5, 1e-4, 1e-3],
             batch=[256, 512, 1024], window=[10, 20, 40])
XGB_DEFAULTS = dict(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, random_state=SEED, verbosity=0)   # = rul/models.py
NAMES = {'xgb': ('XGBoost', 'XGBoost'), 'xgbd': ('XGBoost-untuned', 'XGBoost-untuned'),
         'xgb0': ('XGBoost-libdefault', 'XGBoost-libdefault'), 'batlinet': ('BatLiNet', 'BatLiNet'), 'single': ('MLP', '1D-CNN'), 'lstm': ('LSTM', 'CNN-LSTM'),
         'transformer': ('Transformer', 'CNN-Transformer'), 'tcn': ('TCN', 'CNN-TCN'), 'patchtst': ('PatchTST', 'CNN-PatchTST')}
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())))

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['matr', 'hust', 'uconn', 'snl'])
ap.add_argument('--data', required=True, help="folder with the dataset's five result files")
ap.add_argument('--out', help='default: DATA/s32')
ap.add_argument('--inputs', default='E36,E7,RSH,RCH')
ap.add_argument('--models', default='xgb,single,lstm,transformer')
ap.add_argument('--seeds', default='0,1,2', help='DL seeds (XGBoost runs once, seed 42)')
ap.add_argument('--window', type=int, default=20)
ap.add_argument('--epochs', type=int, default=150)
ap.add_argument('--patience', type=int, default=15)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--cv', action='store_true', help='cell-wise CV on the training cells instead of the test split')
ap.add_argument('--policy-cv', action='store_true', help='with --cv: folds group cells by charge policy (MATR policies.json)')
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--tune', type=int, default=0, help='random-search trials per (input, model); 0 = s18 settings')
ap.add_argument('--tune-epochs', type=int, default=60)
A = ap.parse_args()
if A.smoke:
    A.epochs, A.window, A.seeds, A.tune_epochs = 2, 5, '0', 2


# ---------------- data
def load():
    j = lambda f: json.load(open(os.path.join(A.data, f)))
    df = pd.read_pickle(os.path.join(A.data, 'features_v2.pkl'))
    split, main = j('split.json'), j('features.json')['main']
    z = np.load(os.path.join(A.data, 'raw_inputs.npz'))
    assert (z['cell'] == df.cell.to_numpy(str)).all() and (z['cycle'] == df.cycle.to_numpy(int)).all(), \
        'raw_inputs.npz rows differ from features_v2.pkl'
    cells = df.cell.to_numpy(str)
    c = pd.DataFrame(z['curve'])
    c = c.loc[:, c.notna().mean() > 0.5]   # voltage points missing in most cycles (HUST > 3.17 V, SNL > ~2.7 V)
    c = c.ffill(axis=1).bfill(axis=1)       # gaps at the ends of a curve: nearest value of the same cycle
    c = c.groupby(cells).ffill()            # cycle without a curve (MATR 299, SNL 27%): previous cycle of the cell
    ref = c.groupby(cells).transform('first')
    rc = np.stack([c.to_numpy(np.float32), (c - ref).to_numpy(np.float32)], 1)
    h = df[HIST].to_numpy(np.float32)
    dur_f = os.path.join(A.data, 'raw_dur.npz')
    if os.path.exists(dur_f):
        dz = np.load(dur_f)
        assert (dz['cell'] == cells).all(), 'raw_dur.npz rows differ from features_v2.pkl'
        h = np.column_stack([h, dz['dur'].astype(np.float32)])
    else:
        print('raw_dur.npz not found: RSH/RCH get the 4 history channels only', flush=True)
    hb = lambda L: np.repeat(h[:, :, None], L, 2)  # noqa: E731   constant channels
    X = {'E40': df[main].to_numpy(np.float32), 'E8': df[TOP8].to_numpy(np.float32), 'RS': z['sig'], 'RC': rc,
         'E36': df[[f for f in main if f not in HIST]].to_numpy(np.float32),
         'E7': df[[f for f in TOP8 if f != 'cycle_index']].to_numpy(np.float32),
         'RSH': np.concatenate([z['sig'], hb(z['sig'].shape[2])], 1), 'RCH': np.concatenate([rc, hb(rc.shape[2])], 1)}
    pos = df.groupby('cell', sort=False).cumcount().to_numpy()
    return df, split['train'], split['test'], X, pos


def scale(name, X, fit):
    if name.startswith('E'):
        imp = SimpleImputer(strategy='median').fit(X[fit])
        sc = StandardScaler().fit(imp.transform(X[fit]))
        return sc.transform(imp.transform(X)).astype(np.float32)
    mu = np.nanmean(X[fit], (0, 2), keepdims=True)
    sd = np.nanstd(X[fit], (0, 2), keepdims=True) + 1e-8
    return np.nan_to_num((X - mu) / sd).astype(np.float32)


def windows(pos, w):
    """(n, w) row indices of each row's last w cycles, oldest first; a cell's first rows repeat its first cycle."""
    r = np.arange(len(pos))[:, None]
    return r - np.minimum(np.arange(w - 1, -1, -1)[None, :], pos[:, None])


def cell_folds(df, cells, seed, n=5):
    """Cell-wise stratified k-fold, strata = cell-life terciles (= rul/common.py)."""
    life = df.groupby('cell').cycle.max()
    lives = np.array([life[c] for c in cells])
    strata = np.digitize(lives, np.percentile(lives, [33.3, 66.6]), right=True)
    if np.bincount(strata, minlength=3).min() >= n:
        split = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(cells)), strata)
    else:
        split = KFold(n_splits=min(n, len(cells)), shuffle=True, random_state=seed).split(cells)
    return [([cells[i] for i in a], [cells[i] for i in b]) for a, b in split]


def metrics(y, p, s):
    """= rul/common.py: MAPE = mean |p - y| / that cell's life (MAPE of the predicted lifetime)."""
    life = pd.Series(y + s['cycle'].to_numpy()).groupby(s['cell'].to_numpy()).transform('max').to_numpy()
    mean_life = pd.Series(life).groupby(s['cell'].to_numpy()).first().mean()
    rmse, mae = float(np.sqrt(mean_squared_error(y, p))), float(mean_absolute_error(y, p))
    return dict(R2=float(r2_score(y, p)), RMSE=rmse, MAE=mae, nRMSE=rmse / mean_life, nMAE=mae / mean_life,
                MAPE=float(np.mean(np.abs(p - y) / life)))


def at_cycle_100(s, y, p):
    """Whole-life prediction once per cell, at its first cycle >= 100 (the literature's usual number)."""
    e = pd.DataFrame({'cell': s.cell.values, 'cycle': s.cycle.values, 'life': y + s.cycle.values, 'pred': p + s.cycle.values})
    e = e[e.cycle >= 100].groupby('cell').first()
    d = e.pred - e.life
    return dict(life100_RMSE=float(np.sqrt((d ** 2).mean())), life100_MAPE=float((d.abs() / e.life).mean()))


# ---------------- models
class Net(nn.Module):
    def __init__(self, head, shape, d=64, dropout=0.1, window=None):
        super().__init__()
        if len(shape) == 1:   # engineered: (F,)
            self.enc = nn.Sequential(nn.Linear(shape[0], 2 * d), nn.ReLU(), nn.Dropout(dropout if A.tune else 0.0), nn.Linear(2 * d, d), nn.ReLU())
        else:                 # raw: (C, L)
            def blk(i, o, k):
                return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
            self.enc = nn.Sequential(*blk(shape[0], 32, 7), nn.MaxPool1d(2), *blk(32, 64, 5), nn.MaxPool1d(2),
                                     *blk(64, d, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = head
        W = window or A.window
        if head == 'lstm':
            self.seq = nn.LSTM(d, d, num_layers=2, batch_first=True, dropout=dropout)
        elif head == 'transformer':
            self.pe = nn.Parameter(torch.zeros(1, W, d))
            self.seq = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True,
                                                                        norm_first=True), 2)
        elif head == 'tcn':
            self.seq = TCN(d, W, dropout)
        elif head == 'patchtst':
            if len(shape) == 1:   # engineered: the features themselves are PatchTST's channels (no MLP encoder)
                self.enc = nn.Identity()
            else:                 # raw: CNN cycle embedding -> 16 channels (cost grows with channels: d=128 was 12x LSTM)
                self.enc = nn.Sequential(self.enc, nn.Linear(d, 16))
            self.seq = PatchTST(shape[0] if len(shape) == 1 else 16, W, d, dropout)
        self.out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):                      # x: (B, W, *shape)
        B, W = x.shape[:2]
        h = self.enc(x.flatten(0, 1)).view(B, W, -1)
        if self.head == 'lstm':
            h = self.seq(h)[0]
        elif self.head == 'transformer':
            h = self.seq(h + self.pe[:, :W])
        elif self.head in ('tcn', 'patchtst'):
            return self.out(self.seq(h)).squeeze(-1)
        return self.out(h[:, -1]).squeeze(-1)


class TCN(nn.Module):
    """Temporal convolutional network (Bai et al. 2018) over the window of cycles: residual blocks of two causal
    dilated Conv1d (kernel 3), dilations 1, 2, 4, ... until the receptive field covers the window. Output = last cycle."""
    def __init__(self, d, W, dropout):
        super().__init__()
        self.blocks, dil = nn.ModuleList(), 1
        while True:
            self.blocks.append(nn.ModuleList([nn.Conv1d(d, d, 3, dilation=dil, padding=2 * dil),
                                              nn.Conv1d(d, d, 3, dilation=dil, padding=2 * dil)]))
            if 1 + 4 * (2 * dil - 1) >= W:   # receptive field of all blocks so far: 1 + 2 * 2 * (1 + 2 + ... + dil)
                break
            dil *= 2
        self.drop = nn.Dropout(dropout)

    def forward(self, h):                      # (B, W, d) -> (B, d)
        h = h.transpose(1, 2)
        L = h.shape[2]
        for c1, c2 in self.blocks:   # padding both sides then cutting the right end = causal convolution
            z = self.drop(torch.relu(c1(h)[..., :L]))
            z = self.drop(torch.relu(c2(z)[..., :L]))
            h = torch.relu(h + z)
        return h[..., -1]


class PatchTST(nn.Module):
    """PatchTST (Nie et al., ICLR 2023) over the window of cycles: channel-independent, each channel's W-cycle series
    cut into patches (length 4, stride 2), shared linear patch embedding + learned positions, 2-layer Transformer
    encoder shared across channels; the last patch token of every channel is concatenated -> d."""
    P, S, DP = 4, 2, 32   # ponytail: patch length/stride and token width fixed (not tuned); tune if PatchTST is close
    # ponytail: no RevIN instance norm, it would remove each window's level, and the capacity level is the RUL signal

    def __init__(self, C, W, d, dropout):
        super().__init__()
        n = (W - self.P) // self.S + 1
        self.emb, self.pe = nn.Linear(self.P, self.DP), nn.Parameter(torch.zeros(1, n, self.DP))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.DP, 4, 2 * self.DP, dropout, batch_first=True,
                                                                    norm_first=True), 2)
        self.proj = nn.Linear(C * self.DP, d)

    def forward(self, h):                      # (B, W, C) -> (B, d)
        B, W, C = h.shape
        z = h.transpose(1, 2).reshape(B * C, W).unfold(1, self.P, self.S)   # (B*C, n patches, P)
        z = self.emb(z) + self.pe   # chunks: B*C sequences can exceed the CUDA launch-grid limit in one call
        z = torch.cat([self.enc(c)[:, -1] for c in z.split(32768)])          # (B*C, DP)
        return torch.relu(self.proj(z.reshape(B, C * self.DP)))


ALPHA, LAMBDA, K_REF = 0.5, 1.0, 32   # BatLiNet: intra/inter mix ("average of the two branches"), inter-loss weight, refs


class BatLiNet(nn.Module):
    """BatLiNet re-implemented from the paper (arXiv 2310.05052; official code is on Code Ocean only).
    The window of cycles is an image (C, W cycles, L). Intra branch f: RUL of the cell. Inter branch g: RUL difference
    between the cell and a reference training cell at the same cycle, from the difference image. Each encoder = two
    Conv2d, each followed by AvgPool and ReLU; both share the linear head w. Prediction = ALPHA * intra +
    (1 - ALPHA) * median over K_REF references of (y_ref + inter). Adapted from whole-life-at-cycle-100 to RUL at every
    cycle; RC already holds Severson's dQ(V) to the first cycle (the paper's intra-cell difference)."""
    def __init__(self, shape, d=64, window=None):
        super().__init__()
        self.W = window or A.window
        c = 1 if len(shape) == 1 else shape[0]

        def enc():
            conv = nn.Sequential(nn.Conv2d(c, 16, 3, padding=1), nn.AvgPool2d(2, ceil_mode=True), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.AvgPool2d(2, ceil_mode=True), nn.ReLU(), nn.Flatten())
            n = conv(torch.zeros(1, c, self.W, shape[-1])).shape[1]   # ponytail: channel sizes not given in paper
            return nn.Sequential(conv, nn.Linear(n, d), nn.ReLU())
        self.f, self.g, self.w = enc(), enc(), nn.Linear(d, 1)

    @staticmethod
    def img(x):   # (B, W, F) -> (B, 1, W, F);  (B, W, C, L) -> (B, C, W, L)
        return x.unsqueeze(1) if x.dim() == 3 else x.transpose(1, 2)

    def intra(self, x):
        return self.w(self.f(self.img(x))).squeeze(-1)

    def inter(self, dx):
        return self.w(self.g(self.img(dx))).squeeze(-1)


def fit_dl(head, X, idx, y, tr, va, te, seed, ckpt, hp=None, epochs=None):
    hp = {**dict(lr=A.lr, d=64, dropout=0.1, wd=1e-4, batch=A.batch), **(hp or {})}
    epochs = epochs or A.epochs
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, idx = torch.from_numpy(X).to(DEV), torch.from_numpy(idx).to(DEV)
    ys = float(y[tr].std())
    yt = torch.from_numpy((y / ys).astype(np.float32)).to(DEV)
    bl = head == 'batlinet'
    m = (BatLiNet(X.shape[1:], hp['d'], idx.shape[1]) if bl else Net(head, X.shape[1:], hp['d'], hp['dropout'], idx.shape[1])).to(DEV)
    trs = tr[np.argsort(CYC[tr], kind='stable')]   # BatLiNet references: training rows sorted by cycle
    cs, ucs = CYC[trs], np.unique(CYC[trs])

    def refs(rows, rng=np.random):
        """One random training row per row, at the same cycle (or the next cycle any training cell reached)."""
        c = ucs[np.minimum(np.searchsorted(ucs, CYC[rows]), len(ucs) - 1)]
        lo, hi = np.searchsorted(cs, c, 'left'), np.searchsorted(cs, c, 'right')
        return torch.from_numpy(trs[lo + (rng.rand(len(rows)) * (hi - lo)).astype(int)]).to(DEV)

    opt = torch.optim.AdamW(m.parameters(), hp['lr'], weight_decay=hp['wd'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
    best, state, bad, ep0 = np.inf, None, 0, 0
    if os.path.exists(ckpt):   # session ended mid-run (1 h walltime / VPN drop): continue from the last epoch
        c = torch.load(ckpt, map_location='cpu', weights_only=False)
        m.load_state_dict(c['m'])
        opt.load_state_dict(c['opt'])
        sch.load_state_dict(c['sch'])
        best, state, bad, ep0 = c['best'], c['state'], c['bad'], c['ep']
        np.random.set_state(c['np_rng'])
        torch.set_rng_state(c['torch_rng'])
        if DEV.type == 'cuda' and c['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(c['cuda_rng'])
        print(f'      resumed at epoch {ep0}', flush=True)

    def pred(b, rng):
        x = X[idx[torch.from_numpy(b).to(DEV)]]
        if not bl:
            return m(x)
        inter = torch.stack([yt[r] + m.inter(x - X[idx[r]]) for r in (refs(b, rng) for _ in range(K_REF))]).median(0).values
        return ALPHA * m.intra(x) + (1 - ALPHA) * inter

    def predict(rows):
        m.eval()
        with torch.no_grad():
            rng = np.random.RandomState(seed)   # same references every epoch: val loss tracks the model, not the draw
            # chunk by cycles, not windows: 4096 windows x W=40 = 164k cycles OOMed a 12 GB Titan (job 489220)
            n = max(1, -(-len(rows) * idx.shape[1] // 16384))
            return torch.cat([pred(b, rng) for b in np.array_split(rows, n)])

    ep = ep0
    for ep in range(ep0 + 1, epochs + 1):
        if bad >= A.patience:   # had already stopped early when the session ended
            ep -= 1
            break
        m.train()
        for b in np.array_split(np.random.permutation(tr), max(1, len(tr) // hp['batch'])):
            r = refs(b) if bl else None
            b = torch.from_numpy(b).to(DEV)
            if bl:   # intra loss + LAMBDA * inter loss (paper eq.)
                x = X[idx[b]]
                loss = nn.functional.mse_loss(m.intra(x), yt[b]) +                     LAMBDA * nn.functional.mse_loss(m.inter(x - X[idx[r]]), yt[b] - yt[r])
            else:
                loss = nn.functional.mse_loss(m(X[idx[b]]), yt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        v = nn.functional.mse_loss(predict(va), yt[torch.from_numpy(va).to(DEV)]).item()
        sch.step(v)
        if v < best:
            best, bad, state = v, 0, {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
        torch.save(dict(m=m.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), best=best, state=state, bad=bad,
                        ep=ep, np_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None), ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)   # atomic: a kill mid-save keeps the previous checkpoint
        if ep % 10 == 0 or ep == 1:
            print(f'      epoch {ep:3d}  train {loss.item():.4f}  val {v:.4f}  best {best:.4f}', flush=True)
        if bad >= A.patience:
            break
    m.load_state_dict(state)
    return predict(te).cpu().numpy() * ys, dict(epochs=ep, params=sum(p.numel() for p in m.parameters()), val_loss=float(best))


def fit_xgb(X, y, tr, te, mk):
    """xgb = step-3 tuned; xgbd = project defaults before tuning (rul/models.py); xgb0 = xgboost library defaults."""
    fp = json.load(open(os.path.join(A.data, 'final_params.json')))
    X = X.reshape(len(X), -1)
    kw = {'xgb': {**XGB_DEFAULTS, **fp['params']}, 'xgbd': XGB_DEFAULTS, 'xgb0': dict(random_state=SEED)}[mk]
    m = xgb.XGBRegressor(**kw, device='cuda' if DEV.type == 'cuda' else 'cpu')
    m.fit(X[tr], y[tr])
    return m.predict(X[te]), dict(epochs=0, params=int(m.get_params()['n_estimators'] or 100))   # None = lib default 100


# ---------------- runs
df, train, test, X, pos = load()
if A.smoke:
    train, test = train[:8], test[:4]
out = A.out or os.path.join(A.data, 's32_smoke' if A.smoke else 's32')
for d in ('preds', 'ckpt'):
    os.makedirs(os.path.join(out, d), exist_ok=True)
log = os.path.join(out, 'runs.csv')
done = set(pd.read_csv(log)[['protocol', 'input', 'model', 'seed', 'fold']].astype(str).agg('|'.join, 1)) \
    if os.path.exists(log) else set()
protocol = ('cvpol' if A.policy_cv else 'cv') if A.cv else 'split'
cv_cells = train + test if len(train) + len(test) < 20 else train   # SNL (9 cells): CV over all, as step 4
if A.policy_cv:   # leave-policy-out: no test cell shares its charge policy with a training cell
    pol = json.load(open(os.path.join(A.data, 'policies.json')))
    gkf = GroupKFold(n_splits=5).split(cv_cells, groups=[pol[c] for c in cv_cells])
    splits = [([cv_cells[i] for i in a], [cv_cells[i] for i in b]) for a, b in gkf]
else:
    splits = cell_folds(df, cv_cells, CV_SEED) if A.cv else [(train, test)]
y, cellcol, CYC = df.RUL.to_numpy(float), df.cell.to_numpy(str), df.cycle.to_numpy(int)
idx1, idxw = windows(pos, 1), windows(pos, A.window)
print(f'===== {A.dataset.upper()}: {len(df)} rows, {protocol}, {len(done)} runs already done, device {DEV}'
      f'{" (" + torch.cuda.get_device_name(0) + ")" if DEV.type == "cuda" else ""} =====', flush=True)

tune_log = os.path.join(out, 'tune.csv')
TUNE_FAMILY = {'E36': 'E40', 'RSH': 'RS', 'RCH': 'RC'}   # same input minus/plus history channels -> same setting


def tune(mk, inp, Xs, fold, tr, va):
    """Random search on the validation cells (never the test cells); seed 0; resumable via tune.csv."""
    rng = np.random.RandomState(1000 + fold)
    trials = [{k: v[rng.randint(len(v))] for k, v in SPACE.items()} for _ in range(A.tune)]
    t = pd.read_csv(tune_log) if os.path.exists(tune_log) else pd.DataFrame(columns=['key', 'trial', 'hp', 'val_loss'])
    key = f'{protocol}|{inp}|{mk}|{fold}'
    have = t[t.key == key]
    for i, hp in enumerate(trials):
        if i in set(have.trial):
            continue
        idx = idx1 if mk == 'single' else windows(pos, hp['window'])
        ck = os.path.join(out, 'ckpt', f'tune_{key.replace("|", "_")}_{i}.pt')
        try:
            _, info = fit_dl(mk, Xs, idx, y, tr, va, va, 0, ck, hp, A.tune_epochs)
        except torch.OutOfMemoryError:   # setting too big for this GPU: record as failed (never chosen), go on
            torch.cuda.empty_cache()
            info = dict(val_loss=float('inf'))
            print(f'      tune {key} trial {i}: CUDA out of memory, recorded as failed', flush=True)
        pd.DataFrame([dict(key=key, trial=i, hp=json.dumps(hp), val_loss=info['val_loss'])]).to_csv(
            tune_log, mode='a', header=not os.path.exists(tune_log), index=False)
        if os.path.exists(ck):
            os.remove(ck)
        print(f'      tune {key} trial {i}: val {info["val_loss"]:.4f}  {hp}', flush=True)
    t = pd.read_csv(tune_log)
    best = t[t.key == key].sort_values('val_loss').iloc[0]
    return json.loads(best.hp)


for fold, (tr_c, te_c) in enumerate(splits):
    va_c = set(np.random.RandomState(SEED).choice(tr_c, max(1, len(tr_c) // 10), replace=False))
    tr_all = np.where(np.isin(cellcol, tr_c))[0]
    te = np.where(np.isin(cellcol, te_c))[0]
    va = tr_all[np.isin(cellcol[tr_all], list(va_c))]
    tr = tr_all[~np.isin(cellcol[tr_all], list(va_c))]
    s = df.iloc[te][['cell', 'cycle']]
    for inp in A.inputs.split(','):
        Xs = scale(inp, X[inp], tr_all)
        for mk in A.models.split(','):
            name = NAMES[mk][inp.startswith('R')]
            hp, idx_mk = None, (idx1 if mk == 'single' else idxw)
            if A.tune and not mk.startswith('xgb'):
                if fold == 0:   # CV: tune on fold 0 only, later folds reuse its setting (tune.csv key has fold 0)
                    tune_split = (tr, va)
                fam = TUNE_FAMILY.get(inp, inp)   # E36/RSH/RCH reuse the setting tuned on E40/RS/RC
                hp = tune(mk, fam, Xs if fam == inp else scale(fam, X[fam], tr_all), 0, *tune_split)
                if mk != 'single':
                    idx_mk = windows(pos, hp['window'])
            for seed in ([SEED] if mk.startswith('xgb') else [int(v) for v in A.seeds.split(',')]):
                key = '|'.join(map(str, (protocol + ('_tuned' if A.tune and not mk.startswith('xgb') else ''), inp, name, seed, fold)))
                if key in done:
                    continue
                print(f'  {protocol} fold {fold}  {inp:4s} {name:16s} seed {seed}', flush=True)
                t0 = time.time()
                ckpt = os.path.join(out, 'ckpt', key.replace('|', '_') + '.pt')
                if mk.startswith('xgb'):
                    p, info = fit_xgb(Xs, y, tr_all, te, mk)
                else:
                    p, info = fit_dl(mk, Xs, idx_mk, y, tr, va, te, seed, ckpt, hp)

                info = {'val_loss': np.nan, **info, 'hp': json.dumps(hp) if hp else ''}   # same columns on every row
                row = dict(dataset=A.dataset, protocol=protocol + ('_tuned' if A.tune and not mk.startswith('xgb') else ''), input=inp, model=name, seed=seed, fold=fold,
                           n_train_cells=len(tr_c), n_test_cells=len(te_c), train_s=round(time.time() - t0, 1),
                           **info, **metrics(y[te], p, s), **at_cycle_100(s, y[te], p))
                pd.DataFrame({'cell': s.cell.values, 'cycle': s.cycle.values, 'y': y[te], 'p': p}).to_csv(
                    os.path.join(out, 'preds', key.replace('|', '_') + '.csv.gz'), index=False)
                pd.DataFrame([row]).to_csv(log, mode='a', header=not os.path.exists(log), index=False)   # last: marks done
                if os.path.exists(ckpt):
                    os.remove(ckpt)
                print(f'    -> MAPE {row["MAPE"]:.4f}  RMSE {row["RMSE"]:.1f}  R2 {row["R2"]:.4f}  '
                      f'({row["train_s"]:.0f} s, {info["epochs"]} epochs)', flush=True)

r = pd.read_csv(log)
r = r[r.protocol.str.startswith(protocol)]
summ = r.groupby(['protocol', 'input', 'model'])[['MAPE', 'RMSE', 'R2', 'life100_MAPE']].agg(['mean', 'std']).round(4)
summ.to_csv(os.path.join(out, f'summary_{protocol}.csv'))
print(f'\n{A.dataset.upper()} {protocol}: mean over seeds (and folds)\n{summ.to_string()}\nS32 DONE', flush=True)
