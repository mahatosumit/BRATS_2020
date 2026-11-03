# Reproduction guide

Two different things can be reproduced, with very different requirements.

## A. Reproduce the metric analysis (what this repository supports)
Requires: Python 3.10+ and the pinned packages. Everything else below needs **your own** copy of BraTS 2020 and **your own** out-of-fold nnU-Net predictions, because neither is redistributed here.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src                                  # Windows PowerShell: $env:PYTHONPATH="src"

# 1. software checks (no imaging data needed)
python -m pytest -q tests

# 2. per-case metrics from predictions + reference labels (needs data you hold)
python -m brats_eval.evaluate \
  --results-dir <nnUNet_results>/Dataset500_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --gt-dir      <nnUNet_preprocessed>/Dataset500_BraTS2020/gt_segmentations \
  --output      outputs/oof_case_metrics.csv

# 3. aggregate tables (compare with results/tables)
python -m brats_eval.summarize --case-metrics outputs/oof_case_metrics.csv --output-dir outputs/tables
```

Status: on the authors' machine (CPython 3.10.12, pinned versions) steps 2 and 3 reproduced all 368 per-case rows and all four aggregate tables exactly from the stored out-of-fold predictions and BraTS reference segmentations. This is a check of the evaluation code against the authors' own predictions, not an independent third-party re-run.

With the pinned package versions, step 3 reproduces the published `results/tables/*.csv` exactly when
run on the original per-case table; a different set of predictions will give different numbers.
The bootstrap uses one seeded generator (20260829); do not reorder the calls in `summarize.py`.

## B. Retrain the five-fold model (NOT packaged here)
This repository does **not** contain training code, checkpoints, preprocessed data or training logs. The
configuration summaries in `configs/` document what was run (nnU-Net v2.8.1, `3d_fullres`, 1,000 epochs,
folds 0-4; the case-to-fold assignment is withheld from this release), but retraining additionally requires BraTS 2020 data obtained through the official access process under its applicable terms and a GPU. Training was interrupted and resumed in several folds and was not bit-wise
deterministic, so retraining is not expected to reproduce the reported numbers exactly.
