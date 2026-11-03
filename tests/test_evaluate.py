"""End-to-end test of case discovery and evaluation on tiny synthetic NIfTI files."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from brats_eval.evaluate import discover_cases, evaluate  # noqa: E402


def write(path, arr, spacing=(1, 1, 1)):
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), affine), str(path))


def test_discovery_uniqueness_and_metrics(tmp_path):
    res, gt = tmp_path / "res", tmp_path / "gt"
    gt.mkdir()
    ref = np.zeros((10, 10, 10), np.uint8)
    ref[2:8, 2:8, 2:8] = 1
    ref[3:7, 3:7, 3:7] = 2
    ref[4:6, 4:6, 4:6] = 3
    for fold in range(5):
        (res / f"fold_{fold}" / "validation").mkdir(parents=True)
        case = f"CASE_{fold}"
        write(gt / f"{case}.nii.gz", ref)
        write(res / f"fold_{fold}" / "validation" / f"{case}.nii.gz", ref)   # perfect prediction
    cases = discover_cases(res, gt, expected=5)
    assert [c["fold"] for c in cases] == [0, 1, 2, 3, 4]
    df = evaluate(cases)
    assert df.shape[0] == 5
    assert (df[["ET_dice", "TC_dice", "WT_dice"]] == 1.0).all().all()
    assert (df["macro_dice"] == 1.0).all()


def test_duplicate_case_across_folds_is_rejected(tmp_path):
    res, gt = tmp_path / "res", tmp_path / "gt"
    gt.mkdir()
    arr = np.ones((4, 4, 4), np.uint8)
    write(gt / "DUP.nii.gz", arr)
    for fold in (0, 1):
        (res / f"fold_{fold}" / "validation").mkdir(parents=True)
        write(res / f"fold_{fold}" / "validation" / "DUP.nii.gz", arr)
    for fold in (2, 3, 4):
        (res / f"fold_{fold}" / "validation").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        discover_cases(res, gt, expected=None)


def test_missing_reference_is_an_error(tmp_path):
    res, gt = tmp_path / "res", tmp_path / "gt"
    gt.mkdir()
    for fold in range(5):
        (res / f"fold_{fold}" / "validation").mkdir(parents=True)
    write(res / "fold_0" / "validation" / "X.nii.gz", np.ones((3, 3, 3), np.uint8))
    with pytest.raises(FileNotFoundError):
        discover_cases(res, gt, expected=None)
