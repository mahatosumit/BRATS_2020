"""Unit tests for brats_eval.metrics using small masks with hand-computable answers."""
import math

import numpy as np
import pytest

from brats_eval.metrics import binary_metrics, hd95, region_masks, surface

SP = (1.0, 1.0, 1.0)


def box(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def test_perfect_overlap():
    m = box((8, 8, 8), (2, 2, 2), (6, 6, 6))
    r = binary_metrics(m, m, SP)
    assert r["dice"] == 1.0 and r["iou"] == 1.0
    assert r["sensitivity"] == 1.0 and r["precision"] == 1.0
    assert r["hd95_mm"] == 0.0 and r["abs_volume_error_pct"] == 0.0


def test_disjoint_masks():
    a = box((8, 8, 8), (0, 0, 0), (2, 2, 2))
    b = box((8, 8, 8), (5, 5, 5), (7, 7, 7))
    r = binary_metrics(a, b, SP)
    assert r["dice"] == 0.0 and r["iou"] == 0.0 and r["sensitivity"] == 0.0 and r["precision"] == 0.0


def test_half_overlap_known_values():
    # pred = 4 voxels, gt = 4 voxels, intersection = 2 voxels
    p = np.zeros((1, 1, 8), bool); p[0, 0, 0:4] = True
    g = np.zeros((1, 1, 8), bool); g[0, 0, 2:6] = True
    r = binary_metrics(p, g, SP)
    assert r["tp"] == 2 and r["fp"] == 2 and r["fn"] == 2 and r["tn"] == 2
    assert r["dice"] == pytest.approx(0.5)            # 2*2/(4+4)
    assert r["iou"] == pytest.approx(2 / 6)            # 2/(2+2+2)
    assert r["sensitivity"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx(0.5)
    assert r["specificity"] == pytest.approx(0.5)      # TN/(TN+FP) = 2/4


def test_both_empty_is_undefined_not_perfect():
    e = np.zeros((4, 4, 4), bool)
    r = binary_metrics(e, e, SP)
    assert math.isnan(r["dice"]) and math.isnan(r["iou"])
    assert math.isnan(r["sensitivity"]) and math.isnan(r["precision"])
    assert math.isnan(r["abs_volume_error_pct"])
    assert r["hd95_mm"] == 0.0
    assert r["specificity"] == 1.0


def test_reference_empty_prediction_present_is_false_positive():
    e = np.zeros((4, 4, 4), bool)
    p = box((4, 4, 4), (0, 0, 0), (2, 2, 2))
    r = binary_metrics(p, e, SP)
    assert r["dice"] == 0.0                            # defined and zero
    assert math.isnan(r["sensitivity"]) and math.isnan(r["abs_volume_error_pct"])
    assert math.isnan(r["hd95_mm"])


def test_prediction_empty_reference_present_is_missed_region():
    e = np.zeros((4, 4, 4), bool)
    g = box((4, 4, 4), (0, 0, 0), (2, 2, 2))
    r = binary_metrics(e, g, SP)
    assert r["dice"] == 0.0 and r["sensitivity"] == 0.0
    assert math.isnan(r["precision"]) and math.isnan(r["hd95_mm"])


def test_arve_and_volume_use_voxel_spacing():
    g = box((10, 10, 10), (0, 0, 0), (4, 5, 5))        # 100 voxels
    p = box((10, 10, 10), (0, 0, 0), (3, 5, 5))        # 75 voxels
    r = binary_metrics(p, g, (2.0, 1.0, 1.0))          # 2 mm^3 per voxel
    assert r["gt_volume_ml"] == pytest.approx(0.2)
    assert r["pred_volume_ml"] == pytest.approx(0.15)
    assert r["abs_volume_error_pct"] == pytest.approx(25.0)


def test_surface_of_single_voxel_and_cube():
    one = box((5, 5, 5), (2, 2, 2), (3, 3, 3))
    assert surface(one).sum() == 1
    cube = box((7, 7, 7), (1, 1, 1), (6, 6, 6))        # 5^3 voxels, interior 3^3
    assert surface(cube).sum() == 5 ** 3 - 3 ** 3


def test_hd95_single_voxels_with_anisotropic_spacing():
    a = np.zeros((6, 3, 3), bool); a[0, 1, 1] = True
    b = np.zeros((6, 3, 3), bool); b[3, 1, 1] = True
    assert hd95(a, b, (2.0, 1.0, 1.0)) == pytest.approx(6.0)   # 3 voxels * 2 mm
    assert hd95(a, b, (1.0, 1.0, 1.0)) == pytest.approx(3.0)


def test_hd95_is_pooled_percentile_not_max_of_directed():
    # A single-voxel prediction versus a long line: pooled distances are
    # dominated by the many reference points, so HD95 < max directed distance.
    pred = np.zeros((1, 1, 41), bool); pred[0, 0, 0] = True
    gt = np.zeros((1, 1, 41), bool); gt[0, 0, :] = True
    # surfaces: pred -> its voxel; gt -> all 41 voxels in a 1-voxel-thick line
    pooled = hd95(pred, gt, SP)
    assert pooled < 40.0                     # max directed distance would be 40
    assert pooled == pytest.approx(np.percentile(np.concatenate([[0.0], np.arange(41.0)]), 95))


def test_region_masks_are_hierarchical():
    labels = np.array([[[0, 1, 2, 3]]])
    m = region_masks(labels)
    assert m["WT"].tolist() == [[[False, True, True, True]]]
    assert m["TC"].tolist() == [[[False, False, True, True]]]
    assert m["ET"].tolist() == [[[False, False, False, True]]]
