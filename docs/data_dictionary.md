# Data dictionary

## Published aggregate tables (`results/tables/`)

All tables describe the 368 evaluated BraTS 2020 training cases (five patient-level
out-of-fold folds). They contain no imaging data and no per-patient rows.

### `aggregate_metrics.csv`
One row per (region, metric).

| Column | Meaning |
|---|---|
| `region` | `ET` enhancing tumour (label 3); `TC` tumour core (labels 2+3); `WT` whole tumour (labels 1+2+3), using internal nnU-Net labels |
| `metric` | `dice`, `iou`, `sensitivity`, `precision`, `specificity`, `hd95_mm` (pooled bidirectional 95th-percentile Hausdorff distance, mm), `abs_volume_error_pct` (absolute relative volume error, %) |
| `mean`, `sd`, `median`, `q1`, `q3` | statistics over valid cases; `sd` uses n-1 |
| `ci95_low`, `ci95_high` | percentile 95% interval of the mean from 10,000 patient-level bootstrap resamples (seed 20260829) |
| `valid_n`, `total_n` | cases with a defined value / all evaluated cases (undefined values are excluded, not set to 0 or 1) |

### `per_fold_metrics.csv`
One row per fold (0-4). `n` = cases held out in that fold. For each region: mean and SD
(n-1) of `dice`, `iou`, `sensitivity`, `precision`, `hd95_mm` over defined cases.
`macro_dice_mean` = fold mean of case-level macro Dice (mean over the regions with defined Dice).

### `fold_valid_counts.csv`
`total_n` and the number of cases with defined Dice per region and fold.
ET is lower than the fold size because Dice is undefined when both reference and prediction ET are empty.

### `volume_dice_correlations.csv`
Spearman correlation between log10(reference region volume) and Dice, over cases with positive
reference volume (`n_positive`). `rho_ci95_*` use 5,000 bootstrap resamples; `p_value_exploratory` is
two-sided, uncorrected and descriptive only.

## Configuration files (`configs/`)
* `model_configuration.json` – selected fields of the executed nnU-Net v2.8.1 `3d_fullres` plan and `dataset.json`.
* `experiment_summary.json` – cohort, training, inference and evaluation settings (see manuscript).
* The case-to-fold assignment (nnU-Net `splits_final.json`) is **not** included in this release; only fold sizes are reported (`experiment_summary.json`).

## Local-only per-case table (NOT in this repository)
`brats_eval.evaluate` writes `oof_case_metrics.csv`; keep it local. Columns:
`case`, `fold`, `spacing`, then for each region R in {ET, TC, WT}:
`R_dice, R_iou, R_sensitivity, R_precision, R_specificity, R_hd95_mm, R_gt_volume_ml,
R_pred_volume_ml, R_abs_volume_error_pct, R_tp, R_fp, R_fn, R_tn`, and `macro_dice`.
Empty-region policy: see README.
