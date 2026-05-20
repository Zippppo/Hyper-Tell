"""Compare VoxTell inference outputs against ground-truth segmentations.

Example
-------
python tests/evaluate_segmentation.py \
    --pred-dir outputs/BDMAP_00000001_test \
    --gt-dir niigz_data/BDMAP_00000001/segmentations \
    --pairs ct_liver=liver ct_spleen=spleen \
            ct_right_kidney=kidney_right ct_left_kidney=kidney_left \
    --csv outputs/BDMAP_00000001_test/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


DEFAULT_PAIRS = {
    "ct_liver": "liver",
    "ct_spleen": "spleen",
    "ct_right_kidney": "kidney_right",
    "ct_left_kidney": "kidney_left",
}


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    mask = data > 0
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return mask, spacing, img.affine


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def symmetric_surface_distances(
    pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """Return concatenated surface-to-surface distances in mm (both directions)."""
    pred_surf = surface_voxels(pred)
    gt_surf = surface_voxels(gt)

    if not pred_surf.any() or not gt_surf.any():
        return np.array([], dtype=np.float64)

    dist_to_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dist_to_gt[pred_surf]
    d_gt_to_pred = dist_to_pred[gt_surf]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def compute_metrics(
    pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]
) -> dict[str, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())

    union = tp + fp + fn
    iou = tp / union if union > 0 else float("nan")
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    voxel_vol_ml = float(np.prod(spacing)) / 1000.0  # mm^3 -> mL
    pred_vol_ml = pred_b.sum() * voxel_vol_ml
    gt_vol_ml = gt_b.sum() * voxel_vol_ml
    rel_vol_diff = (pred_vol_ml - gt_vol_ml) / gt_vol_ml if gt_vol_ml > 0 else float("nan")

    distances = symmetric_surface_distances(pred_b, gt_b, spacing)
    if distances.size > 0:
        assd = float(distances.mean())
        hd = float(distances.max())
        hd95 = float(np.percentile(distances, 95))
    else:
        assd = hd = hd95 = float("nan")

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "assd_mm": assd,
        "hd_mm": hd,
        "hd95_mm": hd95,
        "rel_vol_diff": rel_vol_diff,
        "pred_vol_ml": pred_vol_ml,
        "gt_vol_ml": gt_vol_ml,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def parse_pairs(items: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--pairs entry '{item}' must be of the form pred_stem=gt_stem"
            )
        pred_stem, gt_stem = item.split("=", 1)
        pairs[pred_stem.strip()] = gt_stem.strip()
    return pairs


def resolve_path(folder: Path, stem: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No file '{stem}.nii(.gz)' under {folder}")


def format_row(name: str, m: dict[str, float]) -> str:
    return (
        f"{name:<22} "
        f"{m['dice']:.4f}  {m['iou']:.4f}  "
        f"{m['precision']:.4f}  {m['recall']:.4f}  "
        f"{m['hd95_mm']:>7.2f}  {m['assd_mm']:>6.2f}  "
        f"{m['rel_vol_diff']:>+7.2%}  "
        f"{m['pred_vol_ml']:>8.1f}  {m['gt_vol_ml']:>8.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with predicted *.nii.gz masks")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory with ground-truth *.nii.gz masks")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[f"{k}={v}" for k, v in DEFAULT_PAIRS.items()],
        help="Mapping pred_stem=gt_stem (without .nii.gz). Defaults match the demo case.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    pairs = parse_pairs(args.pairs)

    header = (
        f"{'label':<22} {'Dice':>6}  {'IoU':>6}  "
        f"{'Prec':>6}  {'Recall':>6}  {'HD95':>7}  {'ASSD':>6}  "
        f"{'VolDiff':>7}  {'PredmL':>8}  {'GTmL':>8}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    dice_scores: list[float] = []
    iou_scores: list[float] = []

    for pred_stem, gt_stem in pairs.items():
        pred_path = resolve_path(args.pred_dir, pred_stem)
        gt_path = resolve_path(args.gt_dir, gt_stem)

        pred_mask, pred_spacing, pred_affine = load_mask(pred_path)
        gt_mask, gt_spacing, gt_affine = load_mask(gt_path)

        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"Shape mismatch for {pred_stem} vs {gt_stem}: {pred_mask.shape} vs {gt_mask.shape}"
            )
        if not np.allclose(pred_affine, gt_affine, atol=1e-3):
            print(
                f"  [warn] {pred_stem}: affines differ slightly from {gt_stem}; "
                "using GT spacing for distance metrics",
                file=sys.stderr,
            )

        metrics = compute_metrics(pred_mask, gt_mask, gt_spacing)
        print(format_row(f"{pred_stem} -> {gt_stem}", metrics))

        row = {"pred": pred_stem, "gt": gt_stem, **metrics}
        rows.append(row)
        if not np.isnan(metrics["dice"]):
            dice_scores.append(metrics["dice"])
        if not np.isnan(metrics["iou"]):
            iou_scores.append(metrics["iou"])

    print("-" * len(header))
    if dice_scores:
        print(
            f"{'mean':<22} {np.mean(dice_scores):.4f}  {np.mean(iou_scores):.4f}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {args.csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"JSON written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
