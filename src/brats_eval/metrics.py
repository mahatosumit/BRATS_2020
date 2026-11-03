"""Binary segmentation metrics used in the BraTS 2020 nnU-Net out-of-fold audit.

Conventions (identical to the analysis reported in the manuscript):

* Dice / IoU are NaN (undefined) when both prediction and reference are empty.
* Sensitivity is NaN when the reference is empty; precision is NaN when the
  prediction is empty; specificity is 1.0 when there are no negatives.
* HD95 is the 95th percentile of the pooled bidirectional surface distances
  (6-connected inner surface, distances in millimetres). It is 0.0 when both
  masks are empty and NaN when exactly one mask is empty.
* Absolute relative volume error (ARVE, %) is NaN when the reference is empty.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree

#: Hierarchical BraTS evaluation regions in terms of internal nnU-Net labels
#: (1 = edema, 2 = necrotic/non-enhancing core, 3 = enhancing tumour).
REGIONS = {"ET": (3,), "TC": (2, 3), "WT": (1, 2, 3)}


def surface(mask: np.ndarray) -> np.ndarray:
    """Return the inner surface voxels (mask minus its 6-connected erosion)."""
    structure = generate_binary_structure(mask.ndim, 1)
    return np.logical_xor(mask, binary_erosion(mask, structure=structure, border_value=0))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Pooled bidirectional 95th-percentile Hausdorff distance in mm."""
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("nan")
    scale = np.asarray(spacing, dtype=np.float32)
    pred_points = np.argwhere(surface(pred)).astype(np.float32) * scale
    gt_points = np.argwhere(surface(gt)).astype(np.float32) * scale
    d_pred, _ = cKDTree(gt_points).query(pred_points, k=1, workers=-1)
    d_gt, _ = cKDTree(pred_points).query(gt_points, k=1, workers=-1)
    return float(np.percentile(np.concatenate([d_pred, d_gt]), 95))


def binary_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """Overlap, boundary, volume and confusion-matrix metrics for one region."""
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    pred_n = tp + fp
    gt_n = tp + fn
    dice = float("nan") if pred_n + gt_n == 0 else 2.0 * tp / (pred_n + gt_n)
    iou = float("nan") if tp + fp + fn == 0 else tp / (tp + fp + fn)
    sensitivity = float("nan") if gt_n == 0 else tp / gt_n
    precision = float("nan") if pred_n == 0 else tp / pred_n
    specificity = 1.0 if tn + fp == 0 else tn / (tn + fp)
    voxel_volume_mm3 = float(np.prod(spacing))
    gt_volume_ml = gt_n * voxel_volume_mm3 / 1000.0
    pred_volume_ml = pred_n * voxel_volume_mm3 / 1000.0
    arve = float("nan") if gt_volume_ml == 0 else 100.0 * abs(pred_volume_ml - gt_volume_ml) / gt_volume_ml
    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "precision": precision,
        "specificity": specificity,
        "hd95_mm": hd95(pred, gt, spacing),
        "gt_volume_ml": gt_volume_ml,
        "pred_volume_ml": pred_volume_ml,
        "abs_volume_error_pct": arve,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def region_masks(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean ET/TC/WT masks from a hierarchical label map (labels 0-3)."""
    return {name: np.isin(label_map, labels) for name, labels in REGIONS.items()}
