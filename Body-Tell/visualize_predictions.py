#!/usr/bin/env python3
"""
Interactive visualization for Body-Tell predictions using Plotly.
Generates an HTML file with 3D volume rendering and 2D slice views.
"""

import argparse
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    }


def create_3d_visualization(pred_masks, prompt_texts, opacity=0.3):
    """Create 3D isosurface visualization for each organ."""
    # Color palette for different organs
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    fig = go.Figure()

    for i, (mask, organ_name) in enumerate(zip(pred_masks, prompt_texts)):
        if mask.sum() == 0:
            continue

        # Create isosurface
        X, Y, Z = np.mgrid[0:mask.shape[0], 0:mask.shape[1], 0:mask.shape[2]]

        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=mask.flatten(),
            isomin=0.5,
            isomax=1.0,
            opacity=opacity,
            surface_count=1,
            colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]],
            showscale=False,
            name=organ_name,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    fig.update_layout(
        title='3D Organ Segmentation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        height=700,
    )

    return fig


def create_slice_visualization(pred_combined, prompt_texts, slice_axis='axial', slice_idx=None):
    """Create 2D slice visualization."""
    # Get slice
    if slice_idx is None:
        slice_idx = pred_combined.shape[{'axial': 0, 'sagittal': 1, 'coronal': 2}[slice_axis]] // 2

    if slice_axis == 'axial':
        slice_data = pred_combined[slice_idx, :, :]
        axis_labels = ('Y (Coronal)', 'Z (Sagittal)')
    elif slice_axis == 'sagittal':
        slice_data = pred_combined[:, slice_idx, :]
        axis_labels = ('X (Axial)', 'Z (Sagittal)')
    else:  # coronal
        slice_data = pred_combined[:, :, slice_idx]
        axis_labels = ('X (Axial)', 'Y (Coronal)')

    # Create discrete colormap
    colors_discrete = ['#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=slice_data,
        colorscale=[[i/(len(colors_discrete)-1), c] for i, c in enumerate(colors_discrete)],
        showscale=True,
        colorbar=dict(
            title="Organ ID",
            tickmode='array',
            tickvals=list(range(len(prompt_texts) + 1)),
            ticktext=['Background'] + list(prompt_texts),
        ),
        hovertemplate=f'{slice_axis.capitalize()} slice {slice_idx}<br>' +
                      f'{axis_labels[0]}: %{{x}}<br>' +
                      f'{axis_labels[1]}: %{{y}}<br>' +
                      'Label: %{z}<extra></extra>',
    ))

    fig.update_layout(
        title=f'{slice_axis.capitalize()} Slice (index: {slice_idx})',
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        height=500,
        yaxis=dict(scaleanchor='x', scaleratio=1),
    )

    return fig


def create_multi_slice_view(pred_combined, prompt_texts):
    """Create a combined view with all three orthogonal slices."""
    # Get middle slices
    axial_idx = pred_combined.shape[0] // 2
    sagittal_idx = pred_combined.shape[1] // 2
    coronal_idx = pred_combined.shape[2] // 2

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Axial (Z={axial_idx})',
            f'Sagittal (Y={sagittal_idx})',
            f'Coronal (X={coronal_idx})',
            'Legend'
        ),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'table'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Color mapping
    colors_discrete = ['#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    colorscale = [[i/(len(colors_discrete)-1), c] for i, c in enumerate(colors_discrete)]

    # Axial slice
    axial_slice = pred_combined[axial_idx, :, :]
    fig.add_trace(
        go.Heatmap(z=axial_slice, colorscale=colorscale, showscale=False, name='Axial'),
        row=1, col=1
    )

    # Sagittal slice
    sagittal_slice = pred_combined[:, sagittal_idx, :]
    fig.add_trace(
        go.Heatmap(z=sagittal_slice, colorscale=colorscale, showscale=False, name='Sagittal'),
        row=1, col=2
    )

    # Coronal slice
    coronal_slice = pred_combined[:, :, coronal_idx]
    fig.add_trace(
        go.Heatmap(z=coronal_slice, colorscale=colorscale, showscale=False, name='Coronal'),
        row=2, col=1
    )

    # Legend table
    legend_data = {
        'Label ID': list(range(len(prompt_texts) + 1)),
        'Organ': ['Background'] + list(prompt_texts),
        'Color': ['Black'] + ['Red', 'Cyan', 'Blue', 'Orange', 'Yellow'][:len(prompt_texts)],
    }

    fig.add_trace(
        go.Table(
            header=dict(values=list(legend_data.keys()), fill_color='paleturquoise', align='left'),
            cells=dict(values=list(legend_data.values()), fill_color='lavender', align='left')
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, scaleanchor='x', scaleratio=1)

    fig.update_layout(
        title_text='Multi-Planar Reconstruction (MPR)',
        height=900,
        showlegend=False,
    )

    return fig


def create_combined_html(pred_data, output_path):
    """Create a comprehensive HTML report with all visualizations."""
    pred_masks = pred_data['pred_masks']
    pred_combined = pred_data['pred_combined']
    prompt_texts = pred_data['prompt_texts']

    # Create individual figures
    fig_3d = create_3d_visualization(pred_masks, prompt_texts)
    fig_multi_slice = create_multi_slice_view(pred_combined, prompt_texts)

    # Combine into single HTML
    html_parts = [
        '<html><head><title>Body-Tell Visualization</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }',
        'h1 { color: #333; text-align: center; }',
        '.info-box { background: white; padding: 15px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.viz-container { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '</style>',
        '</head><body>',
        '<h1>Body-Tell Segmentation Results</h1>',
        '<div class="info-box">',
        f'<p><strong>Organs Segmented:</strong> {", ".join(prompt_texts)}</p>',
        f'<p><strong>Volume Shape:</strong> {pred_combined.shape}</p>',
        f'<p><strong>Threshold:</strong> {pred_data["threshold"]:.2f}</p>',
        f'<p><strong>Voxel Size:</strong> {pred_data["voxel_size"]}</p>',
        '</div>',
        '<div class="viz-container">',
        '<h2>3D Volume Rendering</h2>',
        fig_3d.to_html(full_html=False, include_plotlyjs='cdn'),
        '</div>',
        '<div class="viz-container">',
        '<h2>2D Slice Views</h2>',
        fig_multi_slice.to_html(full_html=False, include_plotlyjs='cdn'),
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
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path (default: same dir as input)')
    parser.add_argument('--opacity', type=float, default=0.3,
                        help='3D visualization opacity (0-1)')

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading predictions from: {input_path}")
    pred_data = load_predictions(input_path)

    # Determine output path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_visualization.html"
    else:
        output_path = Path(args.output)

    # Create visualization
    print("Creating visualizations...")
    create_combined_html(pred_data, output_path)

    print(f"\n[DONE] Open the file in your browser:")
    print(f"   file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
