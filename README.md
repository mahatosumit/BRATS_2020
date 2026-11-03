# BraTS 2020 nnU-Net five-fold evaluation: code and aggregate results

Evaluation code, configuration summaries and **aggregate** results supporting the manuscript
*"Evidence-Audited Five-Fold Evaluation of a Self-Configuring 3D nnU-Net for Multimodal Glioma Segmentation on BraTS 2020"*
(Mahato, Gupta, Sikidar; KPR Institute of Engineering and Technology). The manuscript is not yet published.

> **This repository does not redistribute the BraTS MRI dataset**, reference or predicted segmentation volumes, model
> checkpoints, or per-patient metric tables. See [Data access and redistribution](#data-access-and-redistribution).

## Research objective
To reconstruct and re-assess a completed five-fold nnU-Net v2 experiment on BraTS 2020: document the executed
configuration, recompute patient-level overlap and boundary metrics from out-of-fold predictions, state the empty-region policy,
and quantify uncertainty, without claiming architectural superiority, external validity or clinical applicability.

## Study overview
* **Dataset:** BraTS 2020 training data (multimodal MRI: FLAIR, T1, T1ce, T2; expert segmentations of edema, necrotic/non-enhancing core and enhancing tumour). The imaging data are not distributed here; they must be obtained through the official BraTS 2020 access process (see [Data access and redistribution](#data-access-and-redistribution)). This repository does not state which download route supplied the authors' local copy.
* **Evaluated cohort:** **368 cases** from 369 source directories. `BraTS20_Training_355` has a valid segmentation with a noncanonical filename that the conversion step did not detect; it was not converted, trained or evaluated.
* **Model:** nnU-Net v2.8.1, `3d_fullres`, six-stage PlainConvUNet (features 32/64/128/256/320/320), 128³ patches, batch size 2, region-based (WT/TC/ET) sigmoid output, Dice + BCE loss with deep supervision, 1,000 epochs.
* **Evaluation:** five-fold patient-level cross-validation (74/74/74/73/73 held out); every case has exactly one out-of-fold prediction generated from `checkpoint_final` (identified from logs, timestamps and source code; command lines for folds 1–4 were not archived), Gaussian-weighted sliding window (step 0.5), mirroring on all axes, sigmoid threshold 0.5, no postprocessing. The held-out fold was also used for epoch-wise patch-level monitoring and `checkpoint_best` selection, which was not used for the reported predictions.

## Main validated results
Out-of-fold, patient-level (undefined values excluded; valid n shown). CI = patient-level percentile bootstrap of the mean (10,000 resamples).

| Region | Dice mean (95% CI) | Dice SD | IoU mean | HD95 median [IQR], mm | Dice valid / total n |
|---|---|---|---|---|---|
| ET | 0.791 (0.766–0.816) | 0.247 | 0.704 | 1.41 [1.00–2.24] | 357/368 |
| TC | 0.875 (0.858–0.891) | 0.161 | 0.804 | 2.24 [1.41–4.24] | 368/368 |
| WT | 0.919 (0.910–0.926) | 0.078 | 0.857 | 2.24 [1.41–4.58] | 368/368 |

Fold-wise mean Dice ranged from 0.744 to 0.846 (ET), 0.851 to 0.912 (TC) and 0.910 to 0.928 (WT); see `results/tables/per_fold_metrics.csv`.
These are internal cross-validation estimates on the prepared BraTS training distribution only.

## Repository structure
```
BRATS_2020/
├── README.md
├── CITATION.cff
├── .gitignore
├── requirements.txt
├── environment.yml
├── SHA256SUMS.txt
├── src/brats_eval/        metrics.py, evaluate.py, summarize.py
├── tests/                 test_metrics.py, test_summarize.py, test_evaluate.py
├── configs/               model_configuration.json, experiment_summary.json
├── results/tables/        aggregate_metrics.csv, per_fold_metrics.csv, fold_valid_counts.csv, volume_dice_correlations.csv
├── figures/               workflow, architecture, fold_stability, metric_summary, volume_dice_association_summary (SVG/PDF)
└── docs/                  methodology.md, data_dictionary.md, reproduction_guide.md, validation_notes.md
```

## Requirements
* Python 3.10 with NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, nibabel 5.4.2 (see `requirements.txt` / `environment.yml`). The 19 tests were run on CPython 3.10.12 with exactly these versions (see [`docs/validation_notes.md`](docs/validation_notes.md)).
* Metric analysis runs on a CPU. The original training used one NVIDIA RTX A4000 GPU, nnU-Net 2.8.1 and PyTorch 2.6.0+cu124; training code is **not** part of this repository.

## Installation
```bash
git clone https://github.com/mahatosumit/BRATS_2020.git
cd BRATS_2020
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduction (two different things)
**Reproducing the metric analysis** needs your own BraTS 2020 reference labels and out-of-fold predictions:
```bash
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"
python -m pytest -q tests
python -m brats_eval.evaluate  --results-dir <trainer_results_dir> --gt-dir <gt_segmentations_dir> --output outputs/oof_case_metrics.csv
python -m brats_eval.summarize --case-metrics outputs/oof_case_metrics.csv --output-dir outputs/tables
```
**Validation status of the evaluation code.** `evaluate.py` and `summarize.py` were run on the authors' own BraTS 2020 reference segmentations and their stored out-of-fold predictions (CPython 3.10.12, pinned packages): all 368 per-case rows and all four aggregate tables were reproduced exactly. This verifies the code against the authors' own predictions; it is not an independent re-run by a third party, and it does not re-create the predictions, which require retraining. Synthetic-data unit tests run without any BraTS data. Details: [`docs/validation_notes.md`](docs/validation_notes.md).

**Retraining the five-fold model is not reproducible from this repository** (no training code, data, checkpoints or logs are included; training was interrupted and resumed in several folds). See [`docs/reproduction_guide.md`](docs/reproduction_guide.md).

## Metrics and empty-region policy
Dice, IoU, sensitivity, precision, specificity, HD95 (95th percentile of pooled bidirectional 6-connected surface distances, mm) and absolute relative volume error (ARVE).
* Dice and IoU are **undefined (excluded, not scored 1)** when both reference and prediction are empty; Dice is 0 when exactly one is empty.
* Sensitivity is undefined for an empty reference, precision for an empty prediction, ARVE for an empty reference.
* HD95 is 0 mm when both masks are empty and undefined when exactly one is empty.
* Macro Dice is the mean over the regions with defined Dice for a case.
* HD95 uses a pooled percentile, which differs from taking the maximum of two directed 95th percentiles; values are not directly comparable with leaderboard HD95.
Full definitions: [`docs/data_dictionary.md`](docs/data_dictionary.md), [`docs/methodology.md`](docs/methodology.md).

## Data access and redistribution
**This repository does not redistribute the BraTS 2020 imaging data, the reference segmentations, any predicted segmentation volumes or any model checkpoints.** Users must obtain the dataset themselves through the official BraTS 2020 access process (registration: <https://www.med.upenn.edu/cbica/brats2020/registration.html>; data page: <https://www.med.upenn.edu/cbica/brats2020/data.html>), subject to its applicable terms. The organisers state that the challenge data may be used for all purposes provided the challenge is appropriately referenced by citing:
1. Menze BH, et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE Trans Med Imaging 2015;34(10):1993–2024. doi:10.1109/TMI.2014.2377694
2. Bakas S, et al. Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. Sci Data 2017;4:170117. doi:10.1038/sdata.2017.117
3. Bakas S, et al. Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge. arXiv:1811.02629 (2018).

Copies of the data on third-party sites (for example Kaggle uploads) carry labels set by their uploaders. Such a label does not establish that the uploader holds the rights in the original BraTS material, and it is not relied on here.

**What is published:** evaluation code and tests, configuration summaries, methodology and reproduction documentation, aggregate and per-fold summary tables, and vector diagrams showing aggregate statistics only (34 files including `SHA256SUMS.txt`).
**What is withheld:** the case-to-fold assignment (which BraTS case belongs to which fold; fold sizes of 74/74/74/73/73 are reported), per-patient metric tables, patient-level plots and BraTS-derived qualitative images. Redistribution permission for these has not been established. The evaluation code reads fold membership from the user's own nnU-Net results directory and does not need the withheld file.

## Licence and rights
**No reuse licence has been assigned to this repository at this release.** The repository owner intends to dedicate the authors' original project materials (code, documentation, aggregate tables and aggregate-only figures) to the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). Because the work is jointly authored, that dedication requires the agreement of all three authors, which has not yet been documented; until a `LICENSE` file is added, default copyright applies and no reuse permission is granted. If CC0 is later applied, it will cover only original project materials. It will **not** apply to the BraTS data or any third-party material, and it is **not** a licence for the BraTS dataset, whose use is governed by the organisers' terms above.

## Citation
See [`CITATION.cff`](CITATION.cff). A DOI will be added only after one has been minted for an archived release.

## Limitations
Internal cross-validation on the BraTS training distribution only; no external, prospective or clinical validation; no comparator models; single reference standard without inter-rater data; HD95 convention not directly comparable with challenge leaderboards; ET Dice is undefined for 11 cases without reference ET and is excluded; training provenance (checkpoint choice for folds 1–4, resumed runs, early fold-0 epoch-time difference) is reconstructed from logs rather than from archived command histories.

## Authors and institution
Sumit Kumar Mahato (corresponding author; 23mi058@kpriet.ac.in; ORCID 0009-0001-5445-9568), Ashutosh Kumar Gupta (ORCID 0009-0001-7377-6403), Arnab Sikidar (ORCID 0000-0002-6817-5179).
KPR Institute of Engineering and Technology, Coimbatore 641407, Tamil Nadu, India. Repository owner: [@mahatosumit](https://github.com/mahatosumit).
