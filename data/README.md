# Data

Not included in the repository. Download each dataset and arrange it under `data/raw/` (or under the folder given by
`RUL_RAW`) as below. The folder names are the ones the loaders expect.

```
data/raw/
  TRI/MATR/MATR_batch_20170512.mat         MATR batch 1 (Severson et al. 2019, data.matr.io)
  TRI/MATR/MATR_batch_20170630.mat         MATR batch 2
  TRI/MATR/MATR_batch_20180412.mat         MATR batch 3
  nsc7hnsg4s-2/nsc7hnsg4s-2/our_data.zip   HUST (Ma et al. 2022, Mendeley Data nsc7hnsg4s, version 2)
  uni coni/                                UConn-ISU-ILCC LFP/Gr Battery Aging Dataset (UConn Digital Commons)
    CyclingParameters.csv
    cycling_part*/*.csv
  SNL/BatteryLife_SNL_LFP/                 SNL LFP cells as distributed by BatteryLife (Tan et al. 2025)
    SNL/SNL/SNL_18650_LFP_*.pkl
    life_labels/Life labels/SNL_labels.json
```

Feature files written by `features/` and `snl/` go to `data/features/` (or `RUL_FEATURES`). Both folders are
git-ignored.
