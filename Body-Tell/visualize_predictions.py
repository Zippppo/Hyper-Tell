#!/usr/bin/env python3
"""
Interactive visualization for Body-Tell predictions using Plotly.
Generates an HTML file with 3D volume rendering comparing predictions with ground truth.

Usage:
python Body-Tell/visualize_predictions.py \
--input Body-Tell/outputs/inference_s2i_smoke/S2I_00001/S2I_00001_predictions.npz \
--original Body-Tell/S2I-Dataset-70cls/data/S2I_00001.npz \
--opacity 0.4 \
--downsample-3d 3

Note: Higher downsample-3d reduces file size significantly.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent


def _to_python_str(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def load_default_aggregate_targets():
    """Load aggregate prompt target ids from the default vocabulary when available."""
    vocab_path = ROOT / 'configs' / 'label_vocab.json'
    if not vocab_path.exists():
        return {}

    with vocab_path.open('r', encoding='utf-8') as f:
        vocab = json.load(f)

    return {
        str(aggregate['id']): [int(x) for x in aggregate.get('component_class_ids', [])]
        for aggregate in vocab.get('aggregates', [])
    }


def normalize_target_class_ids(value):
    """Convert stored target id arrays into a list of per-prompt class id lists."""
    array = np.asarray(value)
    if array.ndim == 0:
        return [[int(array.item())]] if int(array.item()) >= 0 else [[]]
    if array.ndim == 1:
        return [[int(x)] if int(x) >= 0 else [] for x in array]

    target_ids = []
    for row in array:
        target_ids.append([int(x) for x in np.ravel(row) if int(x) >= 0])
    return target_ids


def infer_target_class_ids(prompt_ids, aggregate_targets=None):
    """Infer GT label ids for older prediction files that only stored prompt ids."""
    aggregate_targets = aggregate_targets or load_default_aggregate_targets()
    inferred = []

    for prompt_id in prompt_ids:
        text = _to_python_str(prompt_id)
        class_match = re.match(r'^class_(\d+)_prompt_\d+$', text)
        if class_match:
            inferred.append([int(class_match.group(1))])
            continue

        if text.isdigit():
            inferred.append([int(text)])
            continue

        aggregate_match = re.match(r'^(.+)_prompt_\d+$', text)
        aggregate_id = aggregate_match.group(1) if aggregate_match else text
        inferred.append(list(aggregate_targets.get(aggregate_id, [])))

    return inferred


def build_gt_mask(voxel_labels, target_ids):
    """Build a GT mask from one or more integer voxel label ids."""
    ids = [int(x) for x in target_ids if int(x) >= 0]
    if not ids:
        return np.zeros(voxel_labels.shape, dtype=np.float32)
    if len(ids) == 1:
        return (voxel_labels == ids[0]).astype(np.float32)
    return np.isin(voxel_labels, ids).astype(np.float32)


def load_predictions(npz_path):
    """Load prediction data from npz file."""
    data = np.load(npz_path)
    prompt_ids = data['prompt_ids']
    if 'prompt_target_class_ids' in data.files:
        target_class_ids = normalize_target_class_ids(data['prompt_target_class_ids'])
    elif 'target_class_ids' in data.files:
        target_class_ids = normalize_target_class_ids(data['target_class_ids'])
    elif 'prompt_class_ids' in data.files:
        target_class_ids = normalize_target_class_ids(data['prompt_class_ids'])
    else:
        target_class_ids = infer_target_class_ids(prompt_ids)

    pred_combined = (
        data['pred_combined']
        if 'pred_combined' in data.files
        else np.zeros(data['pred_masks'].shape[1:], dtype=np.uint16)
    )
    case_path = _to_python_str(data['case_path']) if 'case_path' in data.files else None
    return {
        'pred_masks': data['pred_masks'],
        'pred_combined': pred_combined,
        'prompt_texts': data['prompt_texts'],
        'prompt_ids': prompt_ids,
        'target_class_ids': target_class_ids,
        'threshold': float(data['threshold']),
        'voxel_size': data['grid_voxel_size'],
        'grid_world_min': data.get('grid_world_min', np.array([0., 0., 0.])),
        'case_path': case_path,
    }


def load_original_data(original_npz_path):
    """Load original voxel data including sensor point cloud."""
    if not original_npz_path.exists():
        return None
    data = np.load(original_npz_path)
    return {
        'sensor_pc': data['sensor_pc'],
        'voxel_labels': data['voxel_labels'],
        'grid_world_min': data['grid_world_min'],
    }


def infer_case_id(input_path):
    stem = input_path.stem
    if stem.endswith('_predictions'):
        return stem[:-len('_predictions')]
    return input_path.parent.name


def _candidate_roots(input_path):
    roots = []
    resolved = input_path.resolve()
    for parent in resolved.parents:
        if parent.name == ROOT.name or (parent / 'S2I-Dataset-70cls').exists() or (parent / 'Dataset').exists():
            roots.append(parent)
    roots.append(ROOT)

    unique_roots = []
    seen = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique_roots.append(root)
    return unique_roots


def find_original_data_path(input_path, pred_data):
    """Find the original voxel file for GT overlays."""
    case_path = pred_data.get('case_path')
    if case_path:
        for path in (Path(case_path), ROOT.parent / case_path, ROOT / case_path):
            if path.exists():
                return path

    case_id = infer_case_id(input_path)
    data_dirs = (
        Path('S2I-Dataset-70cls') / 'data',
        Path('Dataset') / 'voxel_data',
    )
    for root in _candidate_roots(input_path):
        for data_dir in data_dirs:
            candidate = root / data_dir / f'{case_id}.npz'
            if candidate.exists():
                return candidate
    return None


def create_prediction_only_view(pred_masks, prompt_texts, voxel_size, grid_world_min, original_data=None, opacity=0.3, downsample_factor=2):
    """Create 3D visualization showing only predictions.

    Args:
        downsample_factor: Factor to downsample 3D data (2 = 8x fewer points, 3 = 27x fewer)
    """
    pred_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    fig = go.Figure()

    # Add body surface point cloud if available
    if original_data is not None and 'sensor_pc' in original_data:
        sensor_pc = original_data['sensor_pc']
        pc_downsample = max(1, len(sensor_pc) // 20000)
        sensor_pc_sampled = sensor_pc[::pc_downsample]
        fig.add_trace(go.Scatter3d(
            x=sensor_pc_sampled[:, 0],
            y=sensor_pc_sampled[:, 1],
            z=sensor_pc_sampled[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.15),
            name='Body Surface',
            hoverinfo='skip',
        ))

    # Add predicted organ masks
    for i, (mask, organ_name) in enumerate(zip(pred_masks, prompt_texts)):
        if mask.sum() == 0:
            continue

        if downsample_factor > 1:
            mask_downsampled = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
            voxel_size_downsampled = voxel_size * downsample_factor
        else:
            mask_downsampled = mask
            voxel_size_downsampled = voxel_size

        X, Y, Z = np.mgrid[0:mask_downsampled.shape[0], 0:mask_downsampled.shape[1], 0:mask_downsampled.shape[2]]
        X = X * voxel_size_downsampled[0] + grid_world_min[0]
        Y = Y * voxel_size_downsampled[1] + grid_world_min[1]
        Z = Z * voxel_size_downsampled[2] + grid_world_min[2]

        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=mask_downsampled.flatten(),
            isomin=0.5,
            isomax=1.0,
            opacity=opacity,
            surface_count=1,
            colorscale=[[0, pred_colors[i % len(pred_colors)]], [1, pred_colors[i % len(pred_colors)]]],
            showscale=False,
            name=organ_name,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    fig.update_layout(
        title='Predictions',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=True,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def create_gt_only_view(prompt_texts, target_class_ids, voxel_size, grid_world_min, original_data, opacity=0.3, downsample_factor=2):
    """Create 3D visualization showing only ground truth.

    Args:
        downsample_factor: Factor to downsample 3D data (2 = 8x fewer points, 3 = 27x fewer)
    """
    gt_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    fig = go.Figure()

    if original_data is None or 'voxel_labels' not in original_data:
        # No GT available
        fig.add_annotation(
            text="Ground Truth not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title='Ground Truth')
        return fig

    voxel_labels = original_data['voxel_labels']

    # Add body surface point cloud if available
    if 'sensor_pc' in original_data:
        sensor_pc = original_data['sensor_pc']
        pc_downsample = max(1, len(sensor_pc) // 20000)
        sensor_pc_sampled = sensor_pc[::pc_downsample]
        fig.add_trace(go.Scatter3d(
            x=sensor_pc_sampled[:, 0],
            y=sensor_pc_sampled[:, 1],
            z=sensor_pc_sampled[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.15),
            name='Body Surface',
            hoverinfo='skip',
        ))

    # Add GT organ masks
    for i, (organ_name, target_ids) in enumerate(zip(prompt_texts, target_class_ids)):
        gt_mask = build_gt_mask(voxel_labels, target_ids)

        if gt_mask.sum() == 0:
            continue

        if downsample_factor > 1:
            gt_mask_downsampled = gt_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
            voxel_size_downsampled = voxel_size * downsample_factor
        else:
            gt_mask_downsampled = gt_mask
            voxel_size_downsampled = voxel_size

        X, Y, Z = np.mgrid[0:gt_mask_downsampled.shape[0], 0:gt_mask_downsampled.shape[1], 0:gt_mask_downsampled.shape[2]]
        X = X * voxel_size_downsampled[0] + grid_world_min[0]
        Y = Y * voxel_size_downsampled[1] + grid_world_min[1]
        Z = Z * voxel_size_downsampled[2] + grid_world_min[2]

        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=gt_mask_downsampled.flatten(),
            isomin=0.5,
            isomax=1.0,
            opacity=opacity,
            surface_count=1,
            colorscale=[[0, gt_colors[i % len(gt_colors)]], [1, gt_colors[i % len(gt_colors)]]],
            showscale=False,
            name=organ_name,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    fig.update_layout(
        title='Ground Truth',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=True,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def create_difference_view(pred_masks, prompt_texts, target_class_ids, voxel_size, grid_world_min, original_data, opacity=0.3, downsample_factor=2):
    """Create 3D visualization showing prediction vs GT differences.

    Shows three types of regions:
    - Green: Correct predictions (TP - True Positive)
    - Red: False positives (predicted but not in GT)
    - Blue: False negatives (in GT but not predicted)
    """
    fig = go.Figure()

    if original_data is None or 'voxel_labels' not in original_data:
        # No GT available, just show a message
        fig.add_annotation(
            text="Ground Truth not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title='Prediction vs Ground Truth Difference')
        return fig

    voxel_labels = original_data['voxel_labels']

    # Add body surface point cloud if available
    if 'sensor_pc' in original_data:
        sensor_pc = original_data['sensor_pc']
        pc_downsample = max(1, len(sensor_pc) // 20000)
        sensor_pc_sampled = sensor_pc[::pc_downsample]
        fig.add_trace(go.Scatter3d(
            x=sensor_pc_sampled[:, 0],
            y=sensor_pc_sampled[:, 1],
            z=sensor_pc_sampled[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.15),
            name='Body Surface',
            hoverinfo='skip',
        ))

    # Process each organ
    for i, (pred_mask, organ_name, target_ids) in enumerate(zip(pred_masks, prompt_texts, target_class_ids)):
        gt_mask = build_gt_mask(voxel_labels, target_ids)

        # Calculate differences
        tp_mask = (pred_mask > 0) & (gt_mask > 0)  # True Positive (correct)
        fp_mask = (pred_mask > 0) & (gt_mask == 0)  # False Positive (over-segmentation)
        fn_mask = (pred_mask == 0) & (gt_mask > 0)  # False Negative (under-segmentation)

        # Downsample
        if downsample_factor > 1:
            tp_downsampled = tp_mask[::downsample_factor, ::downsample_factor, ::downsample_factor].astype(np.float32)
            fp_downsampled = fp_mask[::downsample_factor, ::downsample_factor, ::downsample_factor].astype(np.float32)
            fn_downsampled = fn_mask[::downsample_factor, ::downsample_factor, ::downsample_factor].astype(np.float32)
            voxel_size_downsampled = voxel_size * downsample_factor
        else:
            tp_downsampled = tp_mask.astype(np.float32)
            fp_downsampled = fp_mask.astype(np.float32)
            fn_downsampled = fn_mask.astype(np.float32)
            voxel_size_downsampled = voxel_size

        X, Y, Z = np.mgrid[0:tp_downsampled.shape[0], 0:tp_downsampled.shape[1], 0:tp_downsampled.shape[2]]
        X = X * voxel_size_downsampled[0] + grid_world_min[0]
        Y = Y * voxel_size_downsampled[1] + grid_world_min[1]
        Z = Z * voxel_size_downsampled[2] + grid_world_min[2]

        # Add True Positives (green - correct predictions)
        if tp_downsampled.sum() > 0:
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=tp_downsampled.flatten(),
                isomin=0.5,
                isomax=1.0,
                opacity=opacity * 0.7,
                surface_count=1,
                colorscale=[[0, '#00CC00'], [1, '#00CC00']],  # Green
                showscale=False,
                name=f'{organ_name} (Correct)',
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))

        # Add False Positives (red - over-segmentation)
        if fp_downsampled.sum() > 0:
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=fp_downsampled.flatten(),
                isomin=0.5,
                isomax=1.0,
                opacity=opacity,
                surface_count=1,
                colorscale=[[0, '#FF0000'], [1, '#FF0000']],  # Red
                showscale=False,
                name=f'{organ_name} (False Pos)',
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))

        # Add False Negatives (blue - under-segmentation)
        if fn_downsampled.sum() > 0:
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=fn_downsampled.flatten(),
                isomin=0.5,
                isomax=1.0,
                opacity=opacity,
                surface_count=1,
                colorscale=[[0, '#0000FF'], [1, '#0000FF']],  # Blue
                showscale=False,
                name=f'{organ_name} (False Neg)',
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))

    fig.update_layout(
        title='Prediction vs GT Difference',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=True,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig






def create_combined_html(pred_data, original_data, output_path, opacity=0.3, downsample_3d=2):
    """Create HTML report with three side-by-side 3D visualizations."""
    pred_masks = pred_data['pred_masks']
    pred_combined = pred_data['pred_combined']
    prompt_texts = pred_data['prompt_texts']
    target_class_ids = pred_data['target_class_ids']
    voxel_size = pred_data['voxel_size']
    grid_world_min = pred_data['grid_world_min']

    # Create all three visualizations
    fig_pred = create_prediction_only_view(pred_masks, prompt_texts, voxel_size, grid_world_min, original_data, opacity, downsample_3d)
    fig_gt = create_gt_only_view(prompt_texts, target_class_ids, voxel_size, grid_world_min, original_data, opacity, downsample_3d)
    fig_diff = create_difference_view(pred_masks, prompt_texts, target_class_ids, voxel_size, grid_world_min, original_data, opacity, downsample_3d)

    # Combine into single HTML with three-column layout
    html_parts = [
        '<html><head><title>Body-Tell Visualization</title>',
        '<meta charset="utf-8">',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }',
        'h1 { color: #333; text-align: center; }',
        '.info-box { background: white; padding: 15px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.viz-container { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 20px 0; }',
        '.viz-panel { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.note { color: #666; font-style: italic; margin-top: 10px; }',
        '.legend-box { background: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #4ECDC4; }',
        '.legend-item { margin: 5px 0; }',
        '.color-indicator { display: inline-block; width: 15px; height: 15px; margin-right: 8px; vertical-align: middle; border: 1px solid #ccc; }',
        '</style>',
        '</head><body>',
        '<h1>Body-Tell Segmentation Results</h1>',
        '<div class="info-box">',
        f'<p><strong>Organs Segmented:</strong> {", ".join(prompt_texts)}</p>',
        f'<p><strong>Volume Shape:</strong> {pred_combined.shape}</p>',
        f'<p><strong>Threshold:</strong> {pred_data["threshold"]:.2f}</p>',
        f'<p><strong>Voxel Size:</strong> {voxel_size} mm</p>',
        '<div class="legend-box">',
        '<p><strong>Difference View Legend:</strong></p>',
        '<div class="legend-item"><span class="color-indicator" style="background-color: #00CC00;"></span>Green = Correct predictions (True Positive)</div>',
        '<div class="legend-item"><span class="color-indicator" style="background-color: #FF0000;"></span>Red = False positives (over-segmentation)</div>',
        '<div class="legend-item"><span class="color-indicator" style="background-color: #0000FF;"></span>Blue = False negatives (under-segmentation)</div>',
        '</div>',
        '<p class="note">Tip: Use mouse to rotate, zoom, and pan. Click legend items to show/hide specific organs.</p>',
        '</div>',
        '<div class="viz-container">',
        '<div class="viz-panel">',
        fig_pred.to_html(full_html=False, include_plotlyjs='cdn'),
        '</div>',
        '<div class="viz-panel">',
        fig_gt.to_html(full_html=False, include_plotlyjs='cdn'),
        '</div>',
        '<div class="viz-panel">',
        fig_diff.to_html(full_html=False, include_plotlyjs='cdn'),
        '</div>',
        '</div>',
        '</body></html>',
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    print(f"[OK] Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Body-Tell predictions')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to predictions npz file')
    parser.add_argument('--original', type=str, default=None,
                        help='Path to original voxel data npz file (for ground truth overlay)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path (default: same dir as input)')
    parser.add_argument('--opacity', type=float, default=0.4,
                        help='3D visualization opacity (0-1)')
    parser.add_argument('--downsample-3d', type=int, default=3,
                        help='Downsample factor for 3D view (2=8x smaller, 3=27x smaller, default=3)')

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading predictions from: {input_path}")
    pred_data = load_predictions(input_path)

    # Load original data if provided or try to find it automatically
    original_data = None
    if args.original:
        original_path = Path(args.original)
        if original_path.exists():
            print(f"Loading original data from: {original_path}")
            original_data = load_original_data(original_path)
    else:
        auto_original_path = find_original_data_path(input_path, pred_data)
        if auto_original_path is not None:
            print(f"Auto-detected original data: {auto_original_path}")
            original_data = load_original_data(auto_original_path)
        else:
            print("Warning: Original data not found. Visualization will show predictions only.")

    # Determine output path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_visualization.html"
    else:
        output_path = Path(args.output)

    # Create visualization
    print("Creating visualizations...")
    create_combined_html(pred_data, original_data, output_path, args.opacity, args.downsample_3d)

    print(f"\n[DONE] Open the file in your browser:")
    print(f"   file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
