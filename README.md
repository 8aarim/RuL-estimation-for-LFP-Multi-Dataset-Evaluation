# Explainable Machine Learning for LFP Battery Remaining Useful Life Prediction: Multi-Dataset Evaluation

Code for the paper of the same title (manuscript in preparation). It predicts the remaining useful life (RUL) of
LiFePO4/graphite cells from per-cycle engineered features, on four public datasets, and asks which features carry
over between datasets.

| Dataset | Cells used | Source |
|---|---|---|
| MATR | 124 (b1c0-4 merged with their batch-2 continuations, as in the Severson et al. data loader) | Severson et al. 2019, Toyota Research Institute / data.matr.io |
| HUST | 77 | Ma et al. 2022, data repository of the paper |
| UConn-ISU-ILCC | 64 | UConn-ISU-ILCC LFP/Gr Battery Aging Dataset, UConn Digital Commons (CC BY 4.0) |
| SNL | 9 that reach end of life | Preger et al. 2020, batteryarchive.org, BatteryLife distribution (Tan et al. 2025) |

The exact cell list, train/test split and test conditions are in `splits/cells.csv`.

## Setup

Python 3.10. The versions used for the paper are pinned in `requirements.txt`.

```
pip install -r requirements.txt
python -m tests.test_preprocess        # checks that preprocessing uses past cycles only
```

The data is not included. Download it as described in `data/README.md`, then either put it under `data/raw/` or
point the environment variables at it:

```
RUL_RAW=/path/to/raw/datasets          # default data/raw
RUL_FEATURES=/path/to/feature/files    # default data/features
```

## Pipeline

All commands run from the repository root. `<ds>` is one of `matr`, `hust`, `uconn`, `snl`.

1. Per-cycle features (42 per cycle):
   ```
   python -m features.multi_dataset matr       # 124 MATR cells, source labels
   python -m steps.s28_matr_merge build         # merged MATR, the version used in the paper
   python -m features.multi_dataset hust
   python -m features.multi_dataset uconn
   python -m snl.extract
   ```
2. Per-dataset model, steps 1-6 (preprocessing and locked 70/30 split, 8-algorithm benchmark, tuning, final test,
   SHAP ranking, top-k sweep):
   ```
   python run_all.py <ds>
   ```
3. Cross-dataset analyses, after step 2 has run on all four datasets, e.g.
   ```
   python -m steps.s8_common
   python -m steps.s12_common_k
   python -m steps.s21_lodo
   ```
   See the table below for every step.
4. Tuned deep models and leave-condition-out tests (GPU, long): `steps/s32_tuned_deep.py` (options in its header),
   leave-condition-out inputs from `steps/s32_make_lco.py`, statistics with `steps/s32_stats.py`. Condition groups
   and tuned settings are in `configs/lco/`.

The fixed settings used in the paper (feature lists, split, tuned XGBoost parameters) are in `configs/<ds>/`.

## Steps

| Step | What it does |
|---|---|
| s1_preprocess | load features, causal cleaning and trend features, lock the 70/30 cell split |
| s2_benchmark | 8-algorithm comparison, 5-fold cell-wise CV on the training cells |
| s3_tune | tune the best algorithm on the training cells |
| s4_final_model | final model, fit on the training cells, scored once on the locked test cells |
| s5_shap | SHAP feature ranking on held-out folds of the training cells |
| s6_topk | top-k feature sweep; subsets scored once on the test cells |
| s7_compare, s8_common | features that work on several / all datasets |
| s9_diversity, s10_data_inventory, s11_trends | how the datasets differ (data only) |
| s12_common_k | one common feature ranking and number of features for all datasets |
| s13_raw, s13b_duration | raw-signal inputs for the raw-vs-engineered comparison |
| s21_lodo | leave-one-dataset-out test of the common feature subset |
| s22_core | the leave-one-dataset-out core features as fixed subsets |
| s23_ragone | discharge energy and power per cycle (operating-condition map) |
| s24_shap4, s25_shap_stage | four-dataset SHAP figures; SHAP by degradation stage |
| s27_uq_vif | uncertainty and feature redundancy (Pearson, VIF) |
| s28_matr_merge | merge MATR b1c0-4 with their batch-2 continuations |
| s29_baselines | trivial and physics baselines |
| s30_repeated_cv | XGBoost over many splits instead of one |
| s31_reliability | uncertainty, stratified and held-out tests across conditions |
| s32_make_lco, s32_tuned_deep, s32_stats | tuned deep models, leave-condition-out, statistics |
| s34_prof | validation analyses (transfer, label sensitivity, stability, temperature, uncertainty, errors in cycles, BMS model size, reproducibility) |
| uconn_rpt_curves | UConn reference-test capacity curves (reads about 100 GB of CSV) |

## Citation

See `CITATION.cff`. The paper reference will be added on publication.

## License

MIT, see `LICENSE`. The datasets keep their own licences.
