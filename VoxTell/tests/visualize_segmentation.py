"""Render axial PNG previews of VoxTell segmentation masks overlaid on a CT volume.

Example:
    python tests/visualize_segmentation.py \\
        --ct niigz_data/BDMAP_00000001/ct.nii.gz \\
        --seg-dir outputs/BDMAP_00000001_test \\
        --out outputs/BDMAP_00000001_test/preview \\
        --slices 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

# Distinct colors for up to ~10 masks (RGB 0-1).
PALETTE = [
    (1.00, 0.20, 0.20),  # red
    (0.20, 0.80, 0.30),  # green
    (0.30, 0.55, 1.00),  # blue
    (1.00, 0.75, 0.10),  # amber
    (0.85, 0.30, 0.95),  # magenta
    (0.20, 0.85, 0.90),  # cyan
    (1.00, 0.50, 0.00),  # orange
    (0.55, 0.90, 0.20),  # lime
    (0.95, 0.45, 0.65),  # pink
    (0.60, 0.40, 0.85),  # purple
]


def load_volume(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata()


def window_ct(vol: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """Apply soft-tissue CT window and normalize to [0, 1]."""
    lo, hi = wl - ww / 2, wl + ww / 2
    out = np.clip(vol, lo, hi)
    return (out - lo) / (hi - lo)


def pick_slices(mask_union: np.ndarray, n: int) -> list[int]:
    """Pick `n` axial slice indices spanning the range where any mask is present."""
    z_has = np.where(mask_union.any(axis=(0, 1)))[0]
    if z_has.size == 0:
        # Fallback: evenly across the volume.
        return np.linspace(0, mask_union.shape[2] - 1, n, dtype=int).tolist()
    z_lo, z_hi = int(z_has.min()), int(z_has.max())
    if z_lo == z_hi:
        return [z_lo]
    return np.linspace(z_lo, z_hi, n, dtype=int).tolist()


def to_display(slc: np.ndarray) -> np.ndarray:
    """Rotate axial slice for radiological convention (head up, patient-left on viewer-right)."""
    return np.rot90(slc)


def render(
    ct_path: Path,
    seg_dir: Path,
    out_dir: Path,
    n_slices: int,
    alpha: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = load_volume(ct_path)
    ct_win = window_ct(ct)

    ct_stem = ct_path.name.replace(".nii.gz", "").replace(".nii", "")
    mask_files = sorted(seg_dir.glob(f"{ct_stem}_*.nii.gz"))
    if not mask_files:
        raise SystemExit(
            f"No mask files matched '{ct_stem}_*.nii.gz' in {seg_dir}. "
            "Did you point --seg-dir at the prediction output folder?"
        )

    masks: list[tuple[str, np.ndarray, tuple[float, float, float]]] = []
    for i, mf in enumerate(mask_files):
        label = mf.name[len(ct_stem) + 1 : -len(".nii.gz")].replace("_", " ")
        arr = load_volume(mf) > 0.5
        if arr.shape != ct.shape:
            print(f"[skip] {mf.name}: shape {arr.shape} != CT {ct.shape}")
            continue
        masks.append((label, arr, PALETTE[i % len(PALETTE)]))

    union = np.zeros_like(ct, dtype=bool)
    for _, m, _ in masks:
        union |= m

    slice_idxs = pick_slices(union, n_slices)
    print(f"CT: {ct.shape}, masks: {[m[0] for m in masks]}, slices: {slice_idxs}")

    n_cols = len(slice_idxs)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(3.2 * n_cols, 3.6), squeeze=False, facecolor="black"
    )

    for col, z in enumerate(slice_idxs):
        ax = axes[0, col]
        ax.imshow(to_display(ct_win[:, :, z]), cmap="gray", vmin=0, vmax=1)
        for _, m, color in masks:
            m_slc = to_display(m[:, :, z])
            if not m_slc.any():
                continue
            cmap = ListedColormap([(0, 0, 0, 0), (*color, alpha)])
            ax.imshow(m_slc, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"z={z}", color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend below the panel.
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=c, label=lbl, markersize=10)
               for lbl, _, c in masks]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(masks), 5),
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        fontsize=9,
    )
    fig.suptitle(f"{ct_stem} — VoxTell segmentation", color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    overview = out_dir / f"{ct_stem}_overview.png"
    fig.savefig(overview, dpi=140, facecolor="black")
    plt.close(fig)
    print(f"wrote {overview}")

    # Per-slice full-resolution PNGs (handy for screenshots / reports).
    for z in slice_idxs:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
        ax.imshow(to_display(ct_win[:, :, z]), cmap="gray", vmin=0, vmax=1)
        for _, m, color in masks:
            m_slc = to_display(m[:, :, z])
            if not m_slc.any():
                continue
            cmap = ListedColormap([(0, 0, 0, 0), (*color, alpha)])
            ax.imshow(m_slc, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{ct_stem} z={z}", color="white", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        path = out_dir / f"{ct_stem}_z{z:04d}.png"
        fig.savefig(path, dpi=140, facecolor="black")
        plt.close(fig)
        print(f"wrote {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct", required=True, type=Path, help="Original CT NIfTI file")
    p.add_argument("--seg-dir", required=True, type=Path, help="Folder with VoxTell mask outputs")
    p.add_argument("--out", required=True, type=Path, help="Folder to write PNGs into")
    p.add_argument("--slices", type=int, default=6, help="Number of axial slices to render")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity (0-1)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.ct, args.seg_dir, args.out, args.slices, args.alpha)
