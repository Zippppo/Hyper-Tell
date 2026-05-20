"""Interactive HTML viewer for VoxTell CT + segmentation, rendered with Plotly.

Builds a single self-contained HTML file with an axial slice slider and
play/pause controls. Each frame is encoded as a JPEG data-URI so the file
stays small even for hundreds of slices.

Example:
    python tests/visualize_segmentation_html.py \\
        --ct niigz_data/BDMAP_00000001/ct.nii.gz \\
        --seg-dir outputs/BDMAP_00000001_test \\
        --out outputs/BDMAP_00000001_test/preview.html
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from PIL import Image as PILImage

# RGB 0-255 colors for up to ~10 masks.
PALETTE = [
    (255, 60, 60),    # red
    (60, 200, 80),    # green
    (80, 140, 255),   # blue
    (255, 190, 30),   # amber
    (220, 80, 240),   # magenta
    (60, 220, 230),   # cyan
    (255, 130, 0),    # orange
    (140, 230, 60),   # lime
    (240, 120, 170),  # pink
    (160, 110, 220),  # purple
]


def load_vol(p: Path) -> np.ndarray:
    return nib.load(str(p)).get_fdata()


def window_uint8(vol: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    return (np.clip((vol - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def axial_z_range(union: np.ndarray, pad: int) -> tuple[int, int]:
    z = np.where(union.any(axis=(0, 1)))[0]
    if z.size == 0:
        return 0, union.shape[2] - 1
    return (
        max(0, int(z.min()) - pad),
        min(union.shape[2] - 1, int(z.max()) + pad),
    )


def display(slc: np.ndarray) -> np.ndarray:
    """Axial slice (H, W) -> rotated for radiologic display (head up)."""
    return np.rot90(slc)


def blend_slice(ct_slc: np.ndarray, mask_slices, alpha: float) -> np.ndarray:
    rgb = np.stack([ct_slc] * 3, axis=-1).astype(np.float32)
    for m, color in mask_slices:
        sel = m > 0
        if not sel.any():
            continue
        rgb[sel] = (1 - alpha) * rgb[sel] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def encode_jpeg(rgb: np.ndarray, quality: int) -> str:
    buf = BytesIO()
    PILImage.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build(
    ct_path: Path,
    seg_dir: Path,
    out_html: Path,
    alpha: float,
    downsample: int,
    pad_z: int,
    jpeg_quality: int,
    wl: float,
    ww: float,
) -> None:
    ct = load_vol(ct_path)
    ct_u8 = window_uint8(ct, wl=wl, ww=ww)

    stem = ct_path.name.replace(".nii.gz", "").replace(".nii", "")
    mask_files = sorted(seg_dir.glob(f"{stem}_*.nii.gz"))
    if not mask_files:
        raise SystemExit(
            f"No mask files matching '{stem}_*.nii.gz' in {seg_dir}. "
            "Point --seg-dir at the VoxTell prediction folder."
        )

    masks: list[tuple[str, np.ndarray, tuple[int, int, int]]] = []
    for i, mf in enumerate(mask_files):
        label = mf.name[len(stem) + 1 : -len(".nii.gz")].replace("_", " ")
        arr = load_vol(mf) > 0.5
        if arr.shape != ct.shape:
            print(f"[skip] {mf.name}: shape {arr.shape} != CT {ct.shape}")
            continue
        masks.append((label, arr, PALETTE[i % len(PALETTE)]))

    union = np.zeros_like(ct, dtype=bool)
    for _, m, _ in masks:
        union |= m
    z_lo, z_hi = axial_z_range(union, pad_z)

    d = max(downsample, 1)
    ct_ds = ct_u8[::d, ::d]
    masks_ds = [(lbl, m[::d, ::d], col) for lbl, m, col in masks]

    z_indices = list(range(z_lo, z_hi + 1))
    h, w = display(ct_ds[:, :, 0]).shape
    print(
        f"frames: {len(z_indices)}  (z {z_lo}-{z_hi})  "
        f"in-plane: {ct_ds.shape[:2]} -> display {w}x{h}  "
        f"masks: {[lbl for lbl, _, _ in masks_ds]}"
    )

    sources: list[str] = []
    for z in z_indices:
        slc = display(ct_ds[:, :, z])
        msks = [(display(m[:, :, z]), col) for _, m, col in masks_ds]
        rgb = blend_slice(slc, msks, alpha)
        sources.append(encode_jpeg(rgb, jpeg_quality))

    frames = [
        go.Frame(data=[go.Image(source=src)], name=str(z))
        for z, src in zip(z_indices, sources)
    ]

    legend_html = "&nbsp;&nbsp;".join(
        f"<span style='color:rgb{col}; font-size:18px'>&#9632;</span> {lbl}"
        for lbl, _, col in masks_ds
    )

    fig = go.Figure(
        data=[go.Image(source=sources[0])],
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"{stem} &mdash; VoxTell segmentation", x=0.5),
            sliders=[dict(
                active=0,
                steps=[dict(
                    method="animate",
                    label=str(z),
                    args=[[str(z)], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    )],
                ) for z in z_indices],
                currentvalue=dict(prefix="axial z = ", visible=True),
                x=0.05, y=-0.02, len=0.9,
                pad=dict(t=30, b=10),
            )],
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.05, y=1.08, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, dict(
                        frame=dict(duration=80, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    )]),
                ],
            )],
            xaxis=dict(visible=False, constrain="domain"),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=10, r=10, t=70, b=110),
            paper_bgcolor="#111", plot_bgcolor="#111",
            font=dict(color="white"),
            annotations=[dict(
                text=legend_html, xref="paper", yref="paper",
                x=0.5, y=-0.18, showarrow=False,
                font=dict(color="white", size=13),
                xanchor="center",
            )],
            width=900, height=950,
        ),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", auto_play=False)
    size_mb = out_html.stat().st_size / 1e6
    print(f"wrote {out_html}  ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ct", required=True, type=Path, help="Original CT NIfTI file")
    p.add_argument("--seg-dir", required=True, type=Path, help="Folder with VoxTell mask outputs")
    p.add_argument("--out", required=True, type=Path, help="Output .html file path")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity 0-1 (default 0.45)")
    p.add_argument("--downsample", type=int, default=2, help="In-plane downsample factor (default 2)")
    p.add_argument("--pad-z", type=int, default=5, help="Extra slices outside mask z-range (default 5)")
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-95 (default 80)")
    p.add_argument("--wl", type=float, default=40.0, help="CT window level (default 40 HU)")
    p.add_argument("--ww", type=float, default=400.0, help="CT window width (default 400 HU)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(a.ct, a.seg_dir, a.out, a.alpha, a.downsample, a.pad_z, a.jpeg_quality, a.wl, a.ww)
