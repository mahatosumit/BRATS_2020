"""Compute per-case metrics from out-of-fold predictions and reference labels.

This step needs the BraTS 2020 reference segmentations and your own nnU-Net
predictions. Neither is distributed with this repository. Output is a
per-case CSV that should stay local (see docs/data_dictionary.md).

Example
-------
python -m brats_eval.evaluate \
    --results-dir  <nnUNet_results>/Dataset500_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --gt-dir       <nnUNet_preprocessed>/Dataset500_BraTS2020/gt_segmentations \
    --output       outputs/oof_case_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import REGIONS, binary_metrics

N_FOLDS = 5
EXPECTED_CASES = 368


def discover_cases(results_dir: Path, gt_dir: Path, expected: int | None = EXPECTED_CASES) -> list[dict]:
    cases = []
    for fold in range(N_FOLDS):
        validation = results_dir / f"fold_{fold}" / "validation"
        for pred_path in sorted(validation.glob("*.nii.gz")):
            gt_path = gt_dir / pred_path.name
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing reference segmentation for {pred_path.name}")
            cases.append({"fold": fold, "case": pred_path.name[:-7], "prediction": pred_path, "ground_truth": gt_path})
    ids = [c["case"] for c in cases]
    if len(set(ids)) != len(ids):
        raise RuntimeError("A case appears in more than one fold's validation output")
    if expected is not None and len(ids) != expected:
        raise RuntimeError(f"Expected {expected} unique out-of-fold cases; found {len(ids)}")
    return cases


def evaluate(cases: list[dict]) -> pd.DataFrame:
    import nibabel as nib  # imported lazily so metric tests need no imaging stack

    rows = []
    for index, item in enumerate(cases, start=1):
        pred_img = nib.load(str(item["prediction"]))
        gt_img = nib.load(str(item["ground_truth"]))
        if pred_img.shape != gt_img.shape:
            raise ValueError(f"Shape mismatch for {item['case']}: {pred_img.shape} vs {gt_img.shape}")
        if not np.allclose(pred_img.affine, gt_img.affine, atol=1e-4):
            raise ValueError(f"Affine mismatch for {item['case']}")
        pred = np.asanyarray(pred_img.dataobj).astype(np.uint8)
        gt = np.asanyarray(gt_img.dataobj).astype(np.uint8)
        spacing = tuple(float(x) for x in gt_img.header.get_zooms()[:3])
        row = {"case": item["case"], "fold": item["fold"], "spacing": "x".join(f"{x:g}" for x in spacing)}
        region_dice = []
        for region, labels in REGIONS.items():
            m = binary_metrics(np.isin(pred, labels), np.isin(gt, labels), spacing)
            region_dice.append(m["dice"])
            for metric, value in m.items():
                row[f"{region}_{metric}"] = value
        row["macro_dice"] = float(np.nanmean(region_dice))  # mean over regions with defined Dice
        rows.append(row)
        if index % 25 == 0 or index == len(cases):
            print(f"Evaluated {index}/{len(cases)}", flush=True)
    return pd.DataFrame(rows).sort_values(["fold", "case"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True, help="nnU-Net trainer results directory containing fold_0..fold_4")
    ap.add_argument("--gt-dir", type=Path, required=True, help="directory of reference segmentations (<case>.nii.gz)")
    ap.add_argument("--output", type=Path, default=Path("outputs/oof_case_metrics.csv"))
    ap.add_argument("--expected-cases", type=int, default=EXPECTED_CASES, help="set 0 to disable the cohort-size check")
    args = ap.parse_args()
    cases = discover_cases(args.results_dir, args.gt_dir, args.expected_cases or None)
    df = evaluate(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
