"""Aggregate per-case metrics into the tables reported in the manuscript.

Input is a per-case CSV produced by ``brats_eval.evaluate`` (columns documented
in docs/data_dictionary.md). Outputs are aggregate tables only.

Statistical conventions
-----------------------
* Undefined (NaN) values are excluded from each statistic; valid and total n
  are reported.
* SD uses n-1. Confidence intervals are percentile intervals from 10,000
  patient-level bootstrap resamples of the mean (single seeded generator,
  seed 20260829, consumed in the order: aggregate table, then correlations).
* Spearman correlation is computed between log10(reference volume) and Dice
  for cases with positive reference volume; its CI uses 5,000 bootstrap
  resamples and its P value is exploratory and uncorrected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import REGIONS

SEED = 20260829
METRICS = ("dice", "iou", "sensitivity", "precision", "specificity", "hd95_mm", "abs_volume_error_pct")


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    lo, hi = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def aggregate_table(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for metric in METRICS:
            values = df[f"{region}_{metric}"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            lo, hi = bootstrap_mean_ci(values, rng)
            rows.append({
                "region": region, "metric": metric,
                "mean": float(np.mean(finite)), "sd": float(np.std(finite, ddof=1)),
                "median": float(np.median(finite)),
                "q1": float(np.percentile(finite, 25)), "q3": float(np.percentile(finite, 75)),
                "ci95_low": lo, "ci95_high": hi,
                "valid_n": int(len(finite)), "total_n": int(len(values)),
            })
    return pd.DataFrame(rows)


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in df.groupby("fold"):
        row = {"fold": int(fold), "n": int(len(group))}
        for region in REGIONS:
            for metric in ("dice", "iou", "sensitivity", "precision", "hd95_mm"):
                values = group[f"{region}_{metric}"].to_numpy(dtype=float)
                row[f"{region}_{metric}_mean"] = float(np.nanmean(values))
                row[f"{region}_{metric}_sd"] = float(np.nanstd(values, ddof=1))
        row["macro_dice_mean"] = float(group["macro_dice"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def valid_count_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in df.groupby("fold"):
        row = {"fold": int(fold), "total_n": int(len(group))}
        for region in REGIONS:
            row[f"{region}_dice_valid_n"] = int(group[f"{region}_dice"].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame, rng: np.random.Generator, draws: int = 5000) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        x = df[f"{region}_gt_volume_ml"].to_numpy(dtype=float)
        y = df[f"{region}_dice"].to_numpy(dtype=float)
        keep = (x > 0) & np.isfinite(y)  # positive reference volume and defined Dice
        # The published analysis used positive volume only; Dice is defined whenever the
        # reference is non-empty, so the two filters coincide for these data.
        log_x, y_pos = np.log10(x[keep]), y[keep]
        rho, p = spearmanr(log_x, y_pos)
        boot = []
        for _ in range(draws):
            idx = rng.integers(0, len(log_x), len(log_x))
            value = spearmanr(log_x[idx], y_pos[idx]).statistic
            if np.isfinite(value):
                boot.append(value)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"region": region, "spearman_rho_log_volume_vs_dice": float(rho),
                     "rho_ci95_low": float(lo), "rho_ci95_high": float(hi),
                     "p_value_exploratory": float(p), "n_positive": int(keep.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-metrics", type=Path, required=True, help="per-case CSV from brats_eval.evaluate (kept local)")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = ap.parse_args()
    df = pd.read_csv(args.case_metrics)
    rng = np.random.default_rng(SEED)
    summary = aggregate_table(df, rng)
    corr = correlation_table(df, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    fold_table(df).to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    valid_count_table(df).to_csv(args.output_dir / "fold_valid_counts.csv", index=False)
    corr.to_csv(args.output_dir / "volume_dice_correlations.csv", index=False)
    print(f"Wrote aggregate tables to {args.output_dir}")


if __name__ == "__main__":
    main()
