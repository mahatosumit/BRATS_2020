# Validation notes

What was run, where, and what it does and does not establish.

## Environment used for the checks
* Interpreter: CPython 3.10.12 (GCC 11.4.0), Linux virtual machine on the authors' computer.
* Packages (exact): numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, nibabel 5.4.2, pytest 9.2.0.
* The original analysis was executed on Windows with the same numpy/scipy/pandas/nibabel versions (Python 3.10).

## 1. Unit tests (no BraTS data)
`python -m pytest tests` -> **19 passed**. They cover hand-computable masks (perfect/disjoint/half overlap), the empty-region policy,
the pooled bidirectional HD95 (anisotropic spacing, single voxels, pooled-vs-directed maximum), ARVE with voxel spacing, bootstrap
determinism, valid-n tables and a synthetic-NIfTI end-to-end run of `evaluate.py` (discovery, uniqueness, missing references).

## 2. Real-volume validation of `evaluate.py` and `summarize.py`
* `evaluate.py` was run on the real BraTS 2020 reference segmentations and the stored nnU-Net out-of-fold predictions for all five folds (368 cases; run fold by fold because of the interactive time limit).
* Result: all 368 rows and all 41 numeric columns equal the previously stored per-case table (maximum absolute difference 0.0; identical NaN pattern; identical case order and columns).
* `summarize.py` on that table reproduced `aggregate_metrics.csv`, `per_fold_metrics.csv`, `fold_valid_counts.csv` and `volume_dice_correlations.csv` exactly (maximum absolute difference 0.0).
* Limits: this verifies the code against the authors' own predictions and reference labels. It is not a third-party reproduction, and it does not regenerate the predictions (that requires retraining, which is not packaged here).

## 3. Correlation p-values
`volume_dice_correlations.csv` reports Spearman's rho between log10(reference volume) and Dice for cases with positive reference volume
(ET n = 341, TC n = 368, WT n = 368), with a 5,000-resample bootstrap CI and the two-sided `scipy.stats.spearmanr` p-value (t-distribution approximation with n-2 degrees of freedom; exploratory and uncorrected).
Comparing a re-run on Linux with the original Windows-generated table, rho, n and the ET and WT p-values were bit-identical; the only difference was one value, the TC p-value (1.3172674509472126e-05 versus 1.3172674509472124e-05), an absolute difference of 1.7e-21, i.e. exactly one unit in the last place of a double, with a relative difference of 1.3e-16 (below machine epsilon, 2.2e-16). An independent 60-digit evaluation of the t-distribution p-value agreed with every scipy value to about 1e-14 relative or better. The difference is therefore floating-point platform noise in the final digit, with no effect on any reported number (P values are reported to two significant figures). The Linux Python 3.10 re-run reproduces the Linux Python 3.13 value, consistent with a platform rather than a version effect.

## 4. Not done
* No independent third-party reproduction.
* No retraining of the nnU-Net models.
* Gitleaks was not available; secret scanning used detect-secrets and a regular-expression sweep.
