#!/usr/bin/env python3
"""Analyze stored voxel grid shapes and recommend economical training sizes."""

from __future__ import annotations

import argparse
import ast
import json
import math
import struct
import sys
import time
import zipfile
from datetime import date
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.vocabulary import read_json, write_json  # noqa: E402


AxisShape = Tuple[int, int, int]


def read_npy_header_from_npz(path: Path, key: str) -> Dict[str, Any]:
    member_name = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as zf:
        if member_name not in zf.namelist():
            raise KeyError(f"{member_name} not found in {path}")
        with zf.open(member_name) as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError(f"{member_name} in {path} is not an NPY array")
            version = f.read(2)
            if version == b"\x01\x00":
                header_len = struct.unpack("<H", f.read(2))[0]
            elif version in {b"\x02\x00", b"\x03\x00"}:
                header_len = struct.unpack("<I", f.read(4))[0]
            else:
                raise ValueError(f"Unsupported NPY version {version!r} in {path}")
            header = f.read(header_len)
    encoding = "utf-8" if version == b"\x03\x00" else "latin1"
    parsed = ast.literal_eval(header.decode(encoding).strip())
    if not isinstance(parsed, dict):
        raise ValueError(f"Unexpected NPY header in {path}: {parsed!r}")
    return parsed


def shape_from_npz_header(path: Path, key: str) -> AxisShape:
    header = read_npy_header_from_npz(path, key)
    shape = tuple(int(x) for x in header["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected 3D {key} in {path}, got shape {shape}")
    return shape  # type: ignore[return-value]


def empirical_quantile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(int(x) for x in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def parse_quantiles(values: str) -> List[float]:
    quantiles: List[float] = []
    for raw in values.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if value > 1:
            value /= 100.0
        if not 0 < value <= 1:
            raise argparse.ArgumentTypeError("quantiles must be in (0, 1] or (0, 100]")
        quantiles.append(value)
    if not quantiles:
        raise argparse.ArgumentTypeError("at least one quantile is required")
    return quantiles


def parse_shape(value: str) -> AxisShape:
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must have three integers, for example 164,129,256")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape  # type: ignore[return-value]


def volume(shape: Sequence[int]) -> int:
    return int(shape[0]) * int(shape[1]) * int(shape[2])


def shape_metrics(shapes: np.ndarray, candidate: AxisShape) -> Dict[str, Any]:
    candidate_array = np.asarray(candidate, dtype=np.int64)
    fits_axis = shapes <= candidate_array
    fits_all = np.all(fits_axis, axis=1)
    original_volumes = np.prod(shapes, axis=1).astype(np.float64)
    kept_shapes = np.minimum(shapes, candidate_array)
    kept_volumes = np.prod(kept_shapes, axis=1).astype(np.float64)
    candidate_volume = float(volume(candidate))
    padding_fraction = (candidate_volume - kept_volumes) / candidate_volume
    crop_fraction = 1.0 - (kept_volumes / original_volumes)
    cropped = ~fits_all
    return {
        "shape": list(int(x) for x in candidate),
        "volume": int(candidate_volume),
        "axis_coverage": [float(np.mean(fits_axis[:, axis])) for axis in range(3)],
        "joint_coverage": float(np.mean(fits_all)),
        "cropped_case_count": int(np.sum(cropped)),
        "mean_padding_fraction": float(np.mean(padding_fraction)),
        "mean_crop_fraction": float(np.mean(crop_fraction)),
        "mean_crop_fraction_cropped_only": (
            float(np.mean(crop_fraction[cropped])) if np.any(cropped) else 0.0
        ),
        "retained_source_volume_fraction": float(np.sum(kept_volumes) / np.sum(original_volumes)),
    }


def minimum_volume_joint_box(shapes: np.ndarray, target_coverage: float) -> Dict[str, Any]:
    n_cases = int(shapes.shape[0])
    required = max(1, int(math.ceil(target_coverage * n_cases)))
    unique_d = np.unique(shapes[:, 0])
    unique_h = np.unique(shapes[:, 1])
    best_shape: AxisShape | None = None
    best_volume: int | None = None

    for d in unique_d:
        d_subset = shapes[shapes[:, 0] <= d]
        if d_subset.shape[0] < required:
            continue
        for h in unique_h:
            dh_widths = d_subset[d_subset[:, 1] <= h, 2]
            if dh_widths.shape[0] < required:
                continue
            w = int(np.partition(dh_widths, required - 1)[required - 1])
            candidate = (int(d), int(h), w)
            candidate_volume = volume(candidate)
            if best_volume is None or candidate_volume < best_volume:
                best_shape = candidate
                best_volume = candidate_volume

    if best_shape is None:
        raise RuntimeError(f"Could not find a box covering {target_coverage:.3f}")
    result = shape_metrics(shapes, best_shape)
    result["target_coverage"] = float(target_coverage)
    result["required_cases"] = required
    return result


def summarize_axis(values: Sequence[int], quantiles: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(np.min(array)),
        "max": int(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "quantiles": {str(q): empirical_quantile(values, q) for q in quantiles},
    }


def selected_split_files(args: argparse.Namespace) -> Tuple[List[Path], Dict[str, Any]]:
    if args.split_file is None:
        return sorted(args.voxel_dir.glob("*.npz")), {
            "split_file": None,
            "selected_splits": [],
            "declared_count": None,
            "missing_count": 0,
            "missing_files_preview": [],
        }

    split_data = read_json(args.split_file)
    selected_names: List[str] = []
    for split in args.splits:
        selected_names.extend(str(name) for name in split_data.get(split, []))

    files: List[Path] = []
    missing: List[str] = []
    for name in selected_names:
        path = Path(name)
        if not path.is_absolute():
            if path.parent == Path("."):
                path = args.voxel_dir / path
            else:
                path = ROOT / path
        if path.exists():
            files.append(path)
        else:
            missing.append(name)

    return files, {
        "split_file": str(args.split_file),
        "selected_splits": list(args.splits),
        "declared_count": len(selected_names),
        "missing_count": len(missing),
        "missing_files_preview": missing[:20],
    }


def collect_shapes(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    files, selection_summary = selected_split_files(args)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"No .npz files found in {args.voxel_dir}")

    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for index, path in enumerate(files, start=1):
        try:
            shape = shape_from_npz_header(path, args.key)
        except Exception as exc:  # noqa: BLE001
            if args.strict:
                raise
            warnings.append(f"{path.name}: {exc}")
            continue
        records.append({"case_id": path.stem, "filename": path.name, "shape": list(shape)})
        if args.progress_every and index % args.progress_every == 0:
            print(f"scanned {index}/{len(files)} files", flush=True)
    if not records:
        raise ValueError("No usable shape records were collected")
    return records, warnings, selection_summary


def build_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    records, warnings, selection_summary = collect_shapes(args)
    shapes = np.asarray([record["shape"] for record in records], dtype=np.int64)
    quantiles = args.quantiles
    axis_names = ["D", "H", "W"]

    axis_summary = {
        axis_names[axis]: summarize_axis(shapes[:, axis].tolist(), quantiles)
        for axis in range(3)
    }

    axis_quantile_boxes: List[Dict[str, Any]] = []
    for q in quantiles:
        candidate = tuple(empirical_quantile(shapes[:, axis].tolist(), q) for axis in range(3))
        result = shape_metrics(shapes, candidate)  # type: ignore[arg-type]
        result["quantile"] = float(q)
        axis_quantile_boxes.append(result)

    joint_optimized_boxes = [
        minimum_volume_joint_box(shapes, q)
        for q in args.joint_coverages
    ]

    reference_shapes: List[Dict[str, Any]] = []
    seen_references = set()
    for label, shape in args.reference_shape:
        if (label, shape) in seen_references:
            continue
        seen_references.add((label, shape))
        result = shape_metrics(shapes, shape)
        result["label"] = label
        reference_shapes.append(result)

    unique_shapes, counts = np.unique(shapes, axis=0, return_counts=True)
    top_shape_indices = np.argsort(counts)[::-1][: args.top_shapes]
    top_shapes = [
        {
            "shape": [int(x) for x in unique_shapes[index]],
            "count": int(counts[index]),
            "fraction": float(counts[index] / shapes.shape[0]),
            "volume": volume(unique_shapes[index]),
        }
        for index in top_shape_indices
    ]

    return {
        "source_voxel_dir": str(args.voxel_dir),
        "selection": selection_summary,
        "array_key": args.key,
        "measurement_scope": {
            "shape_source": f"{args.key}.npy header inside each .npz",
            "measures": (
                "native stored voxel grid extents used as the shared grid for labels "
                "and sensor-point occupancy"
            ),
            "training_semantics": (
                "Body-Tell voxelizes sensor_pc into voxel_labels.shape, then applies "
                "the same volume_size crop/pad to voxel_labels and occupancy."
            ),
            "does_not_measure": [
                "tight bounding box of nonzero anatomical label voxels",
                "tight bounding box of sensor_pc/body-surface points",
                "union tight bounding box of labels and body surface points",
            ],
        },
        "num_cases": int(shapes.shape[0]),
        "quantiles": [float(q) for q in quantiles],
        "joint_coverages": [float(q) for q in args.joint_coverages],
        "axis_summary": axis_summary,
        "axis_quantile_boxes": axis_quantile_boxes,
        "joint_optimized_boxes": joint_optimized_boxes,
        "reference_shapes": reference_shapes,
        "top_shapes": top_shapes,
        "records": records if args.include_records else [],
        "warnings": warnings[:200],
        "warning_count": len(warnings),
        "elapsed_seconds": float(time.time() - started),
    }


def fmt_int(value: int | float) -> str:
    return f"{int(round(float(value))):,}"


def fmt_float(value: int | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: int | float) -> str:
    return f"{100.0 * float(value):.2f}%"


def shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(x)) for x in shape)


def render_candidate_rows(rows: Iterable[Dict[str, Any]], label_key: str) -> str:
    rendered: List[str] = []
    for item in rows:
        if label_key == "quantile":
            label = f"axis p{100 * float(item[label_key]):g}"
        elif label_key == "target_coverage":
            label = f"joint p{100 * float(item[label_key]):g}"
        else:
            label = str(item[label_key])
        axis_cov = " / ".join(fmt_pct(x) for x in item["axis_coverage"])
        rendered.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td><code>{shape_text(item['shape'])}</code></td>"
            f"<td>{fmt_int(item['volume'])}</td>"
            f"<td>{fmt_pct(item['joint_coverage'])}</td>"
            f"<td>{axis_cov}</td>"
            f"<td>{fmt_int(item['cropped_case_count'])}</td>"
            f"<td>{fmt_pct(item['mean_padding_fraction'])}</td>"
            f"<td>{fmt_pct(item['mean_crop_fraction'])}</td>"
            f"<td>{fmt_pct(item['retained_source_volume_fraction'])}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def render_html(analysis: Dict[str, Any], output_json: Path | None) -> str:
    axis_rows: List[str] = []
    for axis, summary in analysis["axis_summary"].items():
        quantile_text = "<br>".join(
            f"p{100 * float(q):g}: {fmt_int(v)}"
            for q, v in summary["quantiles"].items()
        )
        axis_rows.append(
            "<tr>"
            f"<td>{escape(axis)}</td>"
            f"<td>{fmt_int(summary['min'])}</td>"
            f"<td>{fmt_float(summary['mean'])}</td>"
            f"<td>{fmt_float(summary['median'])}</td>"
            f"<td>{quantile_text}</td>"
            f"<td>{fmt_int(summary['max'])}</td>"
            "</tr>"
        )

    top_rows = "\n".join(
        "<tr>"
        f"<td><code>{shape_text(item['shape'])}</code></td>"
        f"<td>{fmt_int(item['count'])}</td>"
        f"<td>{fmt_pct(item['fraction'])}</td>"
        f"<td>{fmt_int(item['volume'])}</td>"
        "</tr>"
        for item in analysis["top_shapes"]
    )

    json_note = (
        f"<p>Machine-readable JSON: <code>{escape(str(output_json))}</code></p>"
        if output_json is not None
        else ""
    )
    warning_note = ""
    if analysis["warning_count"]:
        warning_items = "".join(f"<li>{escape(item)}</li>" for item in analysis["warnings"][:20])
        warning_note = (
            f"<section><h2>Warnings</h2><p>{analysis['warning_count']} warnings were collected.</p>"
            f"<ul>{warning_items}</ul></section>"
        )

    selection = analysis["selection"]
    scope = analysis["measurement_scope"]
    if selection["split_file"] is None:
        selection_text = "all .npz files found in the data directory"
    else:
        selection_text = (
            f"files listed in <code>{escape(selection['split_file'])}</code> "
            f"for splits <code>{escape(','.join(selection['selected_splits']))}</code>"
        )
        if selection["missing_count"]:
            selection_text += (
                f"; missing declared files: {fmt_int(selection['missing_count'])}"
            )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Body-Tell S2I Volume Shape Distribution</title>
  <style>
    body {{ margin: 32px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; }}
    h1 {{ margin-bottom: 4px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    p {{ line-height: 1.55; }}
    code {{ background: #f3f5f7; padding: 2px 5px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px 10px; vertical-align: top; text-align: left; }}
    th {{ background: #edf1f5; }}
    .meta {{ color: #52606d; }}
    .callout {{ border-left: 4px solid #2c7be5; background: #f5f9ff; padding: 12px 14px; }}
  </style>
</head>
<body>
  <h1>Body-Tell S2I Volume Shape Distribution</h1>
  <p class="meta">Source: <code>{escape(analysis['source_voxel_dir'])}</code> · selection: {selection_text} · array key: <code>{escape(analysis['array_key'])}</code> · cases: {fmt_int(analysis['num_cases'])} · generated {date.today().isoformat()}</p>
  <div class="callout">
    <p>
      <strong>Measurement scope.</strong>
      Shapes are read from <code>{escape(scope['shape_source'])}</code>.
      This measures {escape(scope['measures'])}.
      It does not measure a tight bounding box of only nonzero label voxels, only body-surface points,
      or the union of labels and body surface.
    </p>
    <p>
      Training semantics: {escape(scope['training_semantics'])}
    </p>
    <p>
      The axis-quantile boxes answer "what D/H/W independently covers p% of each axis".
      The joint-optimized boxes answer "what minimum-volume D/H/W covers p% of complete cases".
      Use the joint-optimized rows when the objective is the most economical fixed <code>volume_size</code>.
    </p>
  </div>

  <section>
    <h2>Axis Distribution</h2>
    <table>
      <thead><tr><th>Axis</th><th>Min</th><th>Mean</th><th>Median</th><th>Quantiles</th><th>Max</th></tr></thead>
      <tbody>
        {''.join(axis_rows)}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Axis-Quantile Boxes</h2>
    <table>
      <thead>
        <tr><th>Rule</th><th>Shape</th><th>Volume</th><th>Joint Coverage</th><th>Axis Coverage D/H/W</th><th>Cropped Cases</th><th>Mean Padding</th><th>Mean Crop</th><th>Retained Source Volume</th></tr>
      </thead>
      <tbody>
        {render_candidate_rows(analysis['axis_quantile_boxes'], 'quantile')}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Joint-Optimized Boxes</h2>
    <table>
      <thead>
        <tr><th>Target</th><th>Shape</th><th>Volume</th><th>Joint Coverage</th><th>Axis Coverage D/H/W</th><th>Cropped Cases</th><th>Mean Padding</th><th>Mean Crop</th><th>Retained Source Volume</th></tr>
      </thead>
      <tbody>
        {render_candidate_rows(analysis['joint_optimized_boxes'], 'target_coverage')}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Reference Shapes</h2>
    <table>
      <thead>
        <tr><th>Label</th><th>Shape</th><th>Volume</th><th>Joint Coverage</th><th>Axis Coverage D/H/W</th><th>Cropped Cases</th><th>Mean Padding</th><th>Mean Crop</th><th>Retained Source Volume</th></tr>
      </thead>
      <tbody>
        {render_candidate_rows(analysis['reference_shapes'], 'label')}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Most Common Native Shapes</h2>
    <table>
      <thead><tr><th>Shape</th><th>Count</th><th>Fraction</th><th>Volume</th></tr></thead>
      <tbody>{top_rows}</tbody>
    </table>
  </section>

  {warning_note}
  {json_note}
</body>
</html>
"""


def default_report_path() -> Path:
    today = date.today()
    report_dir = ROOT / "reports" / today.strftime("%y-%m-%d")
    return report_dir / f"s2i_volume_shape_distribution_report_{today.isoformat()}.html"


def parse_labeled_shape(value: str) -> Tuple[str, AxisShape]:
    if "=" in value:
        label, raw_shape = value.split("=", 1)
        label = label.strip()
    else:
        raw_shape = value
        label = value
    if not label:
        raise argparse.ArgumentTypeError("reference shape label cannot be empty")
    return label, parse_shape(raw_shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-dir", type=Path, default=ROOT / "S2I-Dataset-70cls" / "data")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument(
        "--splits",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=["train", "val", "test"],
        help="Comma-separated split names to use when --split-file is provided.",
    )
    parser.add_argument("--key", type=str, default="voxel_labels")
    parser.add_argument("--output-html", type=Path, default=default_report_path())
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--quantiles", type=parse_quantiles, default=parse_quantiles("0.90,0.95,0.975,0.99,1.0"))
    parser.add_argument("--joint-coverages", type=parse_quantiles, default=parse_quantiles("0.95,0.99,1.0"))
    parser.add_argument(
        "--reference-shape",
        type=parse_labeled_shape,
        action="append",
        default=[
            ("current_config", (164, 129, 256)),
            ("previous_config", (144, 129, 256)),
        ],
        help="Reference fixed shape as label=D,H,W. Can be repeated.",
    )
    parser.add_argument("--top-shapes", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = build_analysis(args)

    output_json = args.output_json
    if output_json is None and args.output_html is not None:
        output_json = args.output_html.with_suffix(".json")

    if output_json is not None:
        write_json(analysis, output_json)
        print(f"wrote {output_json}")

    if args.output_html is not None:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(render_html(analysis, output_json), encoding="utf-8")
        print(f"wrote {args.output_html}")

    print(f"cases={analysis['num_cases']} elapsed={analysis['elapsed_seconds']:.2f}s")
    for item in analysis["joint_optimized_boxes"]:
        print(
            f"joint p{100 * item['target_coverage']:g}: "
            f"{shape_text(item['shape'])} "
            f"coverage={fmt_pct(item['joint_coverage'])} "
            f"volume={fmt_int(item['volume'])}"
        )


if __name__ == "__main__":
    main()
