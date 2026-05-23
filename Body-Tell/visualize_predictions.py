#!/usr/bin/env python3
"""
Interactive visualization for Body-Tell predictions using Plotly.
Generates an HTML file with 3D volume rendering and 2D slice views.

Usage:
python Body-Tell/visualize_predictions.py \
--input Body-Tell/outputs/inference_demo/BDMAP_00000001/BDMAP_00000001_predictions.npz \
--original Body-Tell/Dataset/voxel_data/BDMAP_00000001.npz \
--opacity 0.4 \
--downsample-3d 3 \
--max-slices 40

Note: Higher downsample-3d and lower max-slices reduce file size significantly.
"""

import argparse
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import json


def load_predictions(npz_path):
    """Load prediction data from npz file."""
    data = np.load(npz_path)
    return {
        'pred_masks': data['pred_masks'],
        'pred_combined': data['pred_combined'],
        'prompt_texts': data['prompt_texts'],
        'prompt_ids': data['prompt_ids'],
        'threshold': float(data['threshold']),
        'voxel_size': data['grid_voxel_size'],
        'grid_world_min': data.get('grid_world_min', np.array([0., 0., 0.])),
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


def create_3d_visualization(pred_masks, prompt_texts, voxel_size, grid_world_min, original_data=None, opacity=0.3, downsample_factor=2):
    """Create 3D isosurface visualization for each organ with optional body surface.

    Args:
        downsample_factor: Factor to downsample 3D data (2 = 8x fewer points, 3 = 27x fewer)
    """
    # Color palette for different organs (bright colors)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    fig = go.Figure()

    # Add body surface point cloud if available (light gray, semi-transparent)
    # Downsample point cloud to reduce file size
    if original_data is not None and 'sensor_pc' in original_data:
        sensor_pc = original_data['sensor_pc']
        # Downsample to every Nth point
        pc_downsample = max(1, len(sensor_pc) // 20000)  # Keep ~20k points max
        sensor_pc_sampled = sensor_pc[::pc_downsample]
        fig.add_trace(go.Scatter3d(
            x=sensor_pc_sampled[:, 0],
            y=sensor_pc_sampled[:, 1],
            z=sensor_pc_sampled[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='lightgray',
                opacity=0.15,
            ),
            name='Body Surface',
            hoverinfo='skip',
        ))

    # Skip ground truth 3D visualization to save space
    # (Ground truth is still shown in 2D slice viewers)

    # Add predicted organ masks (bright, more opaque)
    for i, (mask, organ_name) in enumerate(zip(pred_masks, prompt_texts)):
        if mask.sum() == 0:
            continue

        # Downsample the mask to reduce data size
        if downsample_factor > 1:
            mask_downsampled = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
            voxel_size_downsampled = voxel_size * downsample_factor
        else:
            mask_downsampled = mask
            voxel_size_downsampled = voxel_size

        # Convert voxel indices to real-world coordinates with proper origin offset
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
            colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]],
            showscale=False,
            name=f'{organ_name} (predicted)',
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    fig.update_layout(
        title='3D Organ Segmentation (Predictions in Color)',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def create_interactive_slice_viewer(pred_masks, prompt_texts, original_data=None, axis='axial', organ_idx=0, max_slices=50):
    """Create an interactive slice viewer for a single organ with slider for a specific axis.

    Args:
        max_slices: Maximum number of slices to include (reduces file size)
    """
    # Get the specific organ mask
    organ_mask = pred_masks[organ_idx]
    organ_name = prompt_texts[organ_idx]

    # Color for target organ (bright) and gray for others
    target_color = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][organ_idx % 5]

    # Get ground truth if available
    voxel_labels = original_data['voxel_labels'] if original_data is not None else None

    # Determine axis configuration
    if axis == 'axial':
        num_slices = organ_mask.shape[0]
        axis_labels = ('Y (Coronal)', 'Z (Sagittal)')
        title_prefix = 'Axial'
    elif axis == 'sagittal':
        num_slices = organ_mask.shape[1]
        axis_labels = ('X (Axial)', 'Z (Sagittal)')
        title_prefix = 'Sagittal'
    else:  # coronal
        num_slices = organ_mask.shape[2]
        axis_labels = ('X (Axial)', 'Y (Coronal)')
        title_prefix = 'Coronal'

    # Downsample slices if too many
    slice_step = max(1, num_slices // max_slices)
    slice_indices = list(range(0, num_slices, slice_step))

    # Create frames for selected slices only
    frames = []
    for i in slice_indices:
        if axis == 'axial':
            organ_slice = organ_mask[i, :, :]
            gt_slice = voxel_labels[i, :, :] if voxel_labels is not None else None
        elif axis == 'sagittal':
            organ_slice = organ_mask[:, i, :]
            gt_slice = voxel_labels[:, i, :] if voxel_labels is not None else None
        else:  # coronal
            organ_slice = organ_mask[:, :, i]
            gt_slice = voxel_labels[:, :, i] if voxel_labels is not None else None

        # Create RGB image: target organ in color, GT in gray
        h, w = organ_slice.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Add ground truth in gray (background)
        if gt_slice is not None:
            gray_mask = (gt_slice > 0) & (organ_slice == 0)
            rgb_image[gray_mask] = [128, 128, 128]

        # Add target organ in bright color (foreground)
        if target_color == '#FF6B6B':  # Red
            rgb_image[organ_slice > 0] = [255, 107, 107]
        elif target_color == '#4ECDC4':  # Cyan
            rgb_image[organ_slice > 0] = [78, 205, 196]
        elif target_color == '#45B7D1':  # Blue
            rgb_image[organ_slice > 0] = [69, 183, 209]
        elif target_color == '#FFA07A':  # Orange
            rgb_image[organ_slice > 0] = [255, 160, 122]
        else:  # Yellow
            rgb_image[organ_slice > 0] = [152, 216, 200]

        frames.append(go.Frame(
            data=[go.Image(z=rgb_image)],
            name=str(i)
        ))

    # Create initial figure with middle slice
    mid_idx = len(slice_indices) // 2
    initial_slice_idx = slice_indices[mid_idx]
    if axis == 'axial':
        initial_organ = organ_mask[initial_slice_idx, :, :]
        initial_gt = voxel_labels[initial_slice_idx, :, :] if voxel_labels is not None else None
    elif axis == 'sagittal':
        initial_organ = organ_mask[:, initial_slice_idx, :]
        initial_gt = voxel_labels[:, initial_slice_idx, :] if voxel_labels is not None else None
    else:
        initial_organ = organ_mask[:, :, initial_slice_idx]
        initial_gt = voxel_labels[:, :, initial_slice_idx] if voxel_labels is not None else None

    h, w = initial_organ.shape
    initial_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if initial_gt is not None:
        gray_mask = (initial_gt > 0) & (initial_organ == 0)
        initial_rgb[gray_mask] = [128, 128, 128]

    if target_color == '#FF6B6B':
        initial_rgb[initial_organ > 0] = [255, 107, 107]
    elif target_color == '#4ECDC4':
        initial_rgb[initial_organ > 0] = [78, 205, 196]
    elif target_color == '#45B7D1':
        initial_rgb[initial_organ > 0] = [69, 183, 209]
    elif target_color == '#FFA07A':
        initial_rgb[initial_organ > 0] = [255, 160, 122]
    else:
        initial_rgb[initial_organ > 0] = [152, 216, 200]

    fig = go.Figure(
        data=[go.Image(z=initial_rgb)],
        frames=frames
    )

    # Add slider
    sliders = [dict(
        active=mid_idx,
        yanchor="top",
        y=0,
        xanchor="left",
        x=0.1,
        currentvalue=dict(
            prefix=f"{title_prefix} Slice: ",
            visible=True,
            xanchor="right"
        ),
        transition=dict(duration=0),
        pad=dict(b=10, t=50),
        len=0.8,
        steps=[dict(
            args=[[f.name], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0)
            )],
            method="animate",
            label=str(slice_indices[k])
        ) for k, f in enumerate(frames)]
    )]

    fig.update_layout(
        title=f'{organ_name} - {title_prefix} View (Target in color, others in gray)',
        xaxis=dict(showticklabels=False, title=axis_labels[0]),
        yaxis=dict(showticklabels=False, title=axis_labels[1], scaleanchor='x', scaleratio=1),
        sliders=sliders,
        height=600,
    )

    return fig




def create_combined_html(pred_data, original_data, output_path, opacity=0.3, downsample_3d=2, max_slices=50):
    """Create a comprehensive HTML report with all visualizations."""
    pred_masks = pred_data['pred_masks']
    pred_combined = pred_data['pred_combined']
    prompt_texts = pred_data['prompt_texts']
    voxel_size = pred_data['voxel_size']
    grid_world_min = pred_data['grid_world_min']

    # Create 3D visualization
    fig_3d = create_3d_visualization(pred_masks, prompt_texts, voxel_size, grid_world_min, original_data, opacity, downsample_3d)

    # Create interactive slice viewers for each organ and each axis
    slice_viewers_html = []
    for organ_idx, organ_name in enumerate(prompt_texts):
        slice_viewers_html.append(f'<h3>{organ_name}</h3>')
        slice_viewers_html.append('<div class="organ-slices">')

        fig_axial = create_interactive_slice_viewer(pred_masks, prompt_texts, original_data, axis='axial', organ_idx=organ_idx, max_slices=max_slices)
        slice_viewers_html.append(fig_axial.to_html(full_html=False, include_plotlyjs='cdn'))

        fig_sagittal = create_interactive_slice_viewer(pred_masks, prompt_texts, original_data, axis='sagittal', organ_idx=organ_idx, max_slices=max_slices)
        slice_viewers_html.append(fig_sagittal.to_html(full_html=False, include_plotlyjs='cdn'))

        fig_coronal = create_interactive_slice_viewer(pred_masks, prompt_texts, original_data, axis='coronal', organ_idx=organ_idx, max_slices=max_slices)
        slice_viewers_html.append(fig_coronal.to_html(full_html=False, include_plotlyjs='cdn'))

        slice_viewers_html.append('</div>')

    # Combine into single HTML
    html_parts = [
        '<html><head><title>Body-Tell Visualization</title>',
        '<meta charset="utf-8">',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }',
        'h1 { color: #333; text-align: center; }',
        'h2 { color: #555; border-bottom: 2px solid #4ECDC4; padding-bottom: 10px; }',
        'h3 { color: #666; margin-top: 30px; }',
        '.info-box { background: white; padding: 15px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.viz-container { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.note { color: #666; font-style: italic; margin-top: 10px; }',
        '.organ-slices { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 30px; }',
        '</style>',
        '</head><body>',
        '<h1>Body-Tell Segmentation Results</h1>',
        '<div class="info-box">',
        f'<p><strong>Organs Segmented:</strong> {", ".join(prompt_texts)}</p>',
        f'<p><strong>Volume Shape:</strong> {pred_combined.shape}</p>',
        f'<p><strong>Threshold:</strong> {pred_data["threshold"]:.2f}</p>',
        f'<p><strong>Voxel Size:</strong> {voxel_size} mm</p>',
        '<p class="note">Note: Target organ shown in color, other organs in gray. Ground truth (if available) also shown in gray.</p>',
        '</div>',
        '<div class="viz-container">',
        '<h2>3D Volume Rendering</h2>',
        f'<p>Interactive 3D view - use mouse to rotate, zoom, and pan. Downsampled by {downsample_3d}x for performance.</p>',
        fig_3d.to_html(full_html=False, include_plotlyjs='cdn'),
        '</div>',
        '<div class="viz-container">',
        '<h2>Interactive Slice Viewers</h2>',
        f'<p>Use the sliders to navigate through each plane (showing up to {max_slices} slices per axis). Each organ is shown separately with target organ in color and others in gray.</p>',
    ]

    html_parts.extend(slice_viewers_html)

    html_parts.extend([
        '</div>',
        '</body></html>',
    ])

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
    parser.add_argument('--max-slices', type=int, default=40,
                        help='Maximum slices per axis in slice viewer (default=40)')

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
        # Try to auto-detect original data file
        # Assume structure: Body-Tell/outputs/.../CASE_ID/CASE_ID_predictions.npz
        # Original should be at: Body-Tell/Dataset/voxel_data/CASE_ID.npz
        case_id = input_path.parent.name
        body_tell_root = input_path.parents[2]  # Go up from outputs/inference_demo/CASE_ID
        auto_original_path = body_tell_root / 'Dataset' / 'voxel_data' / f'{case_id}.npz'

        if auto_original_path.exists():
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
    create_combined_html(pred_data, original_data, output_path, args.opacity, args.downsample_3d, args.max_slices)

    print(f"\n[DONE] Open the file in your browser:")
    print(f"   file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
