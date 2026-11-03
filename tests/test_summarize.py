"""Tests for aggregate-table construction on a small synthetic per-case table."""
import numpy as np
import pandas as pd
import pytest

from brats_eval.summarize import (SEED, aggregate_table, bootstrap_mean_ci, correlation_table,
                                  fold_table, valid_count_table)


def synthetic(n=60):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"case": [f"C{i:03d}" for i in range(n)], "fold": np.arange(n) % 5})
    for r in ("ET", "TC", "WT"):
        vol = rng.uniform(0.5, 80, n)
        for m in ("dice", "iou", "sensitivity", "precision", "specificity", "hd95_mm", "abs_volume_error_pct"):
            df[f"{r}_{m}"] = rng.uniform(0.2, 1.0, n)
        df[f"{r}_gt_volume_ml"] = vol
    df.loc[[3, 8], "ET_dice"] = np.nan          # two undefined ET Dice values
    df["macro_dice"] = df[["ET_dice", "TC_dice", "WT_dice"]].mean(axis=1, skipna=True)
    return df


def test_aggregate_counts_and_exclusion_of_nan():
    df = synthetic()
    t = aggregate_table(df, np.random.default_rng(SEED))
    row = t[(t.region == "ET") & (t.metric == "dice")].iloc[0]
    assert row.valid_n == 58 and row.total_n == 60
    expected = df["ET_dice"].dropna().mean()
    assert row["mean"] == pytest.approx(expected)
    assert row.sd == pytest.approx(df["ET_dice"].dropna().std(ddof=1))
    assert row.ci95_low < row["mean"] < row.ci95_high


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    v = np.linspace(0, 1, 40)
    a = bootstrap_mean_ci(v, np.random.default_rng(SEED), draws=500)
    b = bootstrap_mean_ci(v, np.random.default_rng(SEED), draws=500)
    assert a == b


def test_bootstrap_ignores_nan_and_handles_all_nan():
    lo_nan, hi_nan = bootstrap_mean_ci(np.array([np.nan, np.nan]), np.random.default_rng(0))
    assert np.isnan(lo_nan) and np.isnan(hi_nan)
    lo, hi = bootstrap_mean_ci(np.array([1.0, np.nan, 1.0]), np.random.default_rng(0), draws=200)
    assert lo == hi == 1.0


def test_fold_and_valid_count_tables():
    df = synthetic()
    folds = fold_table(df)
    assert folds.n.tolist() == [12] * 5
    counts = valid_count_table(df)
    assert counts.ET_dice_valid_n.sum() == 58
    assert counts.TC_dice_valid_n.tolist() == [12] * 5
    # fold mean of ET Dice excludes NaN
    f0 = df[df.fold == 0]["ET_dice"].mean()
    assert folds.loc[0, "ET_dice_mean"] == pytest.approx(f0)


def test_correlation_positive_for_monotone_relation():
    df = synthetic(80)
    df["WT_dice"] = np.log10(df["WT_gt_volume_ml"]) / 3
    c = correlation_table(df, np.random.default_rng(SEED), draws=200)
    wt = c[c.region == "WT"].iloc[0]
    assert wt.spearman_rho_log_volume_vs_dice == pytest.approx(1.0)
    assert wt.n_positive == 80
