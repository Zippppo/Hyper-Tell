from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.crop_s2i_dataset import (
    CropStats,
    crop_dataset,
    crop_npz_file,
    plan_crop,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_case(path: Path) -> np.ndarray:
    labels = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    sensor_pc = np.array(
        [
            [12.0, 17.0, 30.0],
            [17.9, 31.9, 45.9],
            [10.0, 20.0, 30.0],
            [18.0, 20.0, 30.0],
            [14.0, 35.0, 30.0],
        ],
        dtype=np.float32,
    )
    grid_world_min = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    grid_voxel_size = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    grid_occ_size = np.array(labels.shape, dtype=np.int32)
    grid_world_max = grid_world_min + grid_occ_size * grid_voxel_size
    np.savez(
        path,
        sensor_pc=sensor_pc,
        voxel_labels=labels,
        grid_world_min=grid_world_min,
        grid_world_max=grid_world_max.astype(np.float32),
        grid_voxel_size=grid_voxel_size,
        grid_occ_size=grid_occ_size,
        format_version=np.array("1.0"),
        label_schema_version=np.array("1.0"),
    )
    return labels


def test_plan_crop_tracks_source_and_destination_offsets() -> None:
    plan = plan_crop((5, 4, 3), (3, 6, 4))

    assert [item.src for item in plan] == [slice(1, 4), slice(0, 4), slice(0, 3)]
    assert [item.dst for item in plan] == [slice(0, 3), slice(1, 5), slice(0, 3)]
    assert [item.grid_offset for item in plan] == [1, -1, 0]
    assert [item.crop_before for item in plan] == [1, 0, 0]
    assert [item.crop_after for item in plan] == [1, 0, 0]
    assert [item.pad_before for item in plan] == [0, 1, 0]
    assert [item.pad_after for item in plan] == [0, 1, 1]


def test_crop_npz_file_updates_grid_metadata_and_filters_sensor_points(tmp_path: Path) -> None:
    src = tmp_path / "input.npz"
    dst = tmp_path / "output.npz"
    labels = _write_case(src)

    stats = crop_npz_file(src, dst, target_shape=(3, 6, 4))

    assert isinstance(stats, CropStats)
    assert stats.source_shape == (5, 4, 3)
    assert stats.target_shape == (3, 6, 4)
    assert stats.cropped
    assert stats.axis_crop_before == (1, 0, 0)
    assert stats.axis_crop_after == (1, 0, 0)

    with np.load(dst, allow_pickle=False) as data:
        out_labels = data["voxel_labels"]
        assert out_labels.shape == (3, 6, 4)
        assert np.array_equal(out_labels[:, 1:5, 0:3], labels[1:4, :, :])
        assert np.count_nonzero(out_labels[:, 0, :]) == 0
        assert np.count_nonzero(out_labels[:, 5, :]) == 0
        assert np.count_nonzero(out_labels[:, :, 3]) == 0

        assert data["grid_occ_size"].tolist() == [3, 6, 4]
        assert np.allclose(data["grid_voxel_size"], [2.0, 3.0, 4.0])
        assert np.allclose(data["grid_world_min"], [12.0, 17.0, 30.0])
        assert np.allclose(data["grid_world_max"], [18.0, 35.0, 46.0])
        assert np.all(data["sensor_pc"] >= data["grid_world_min"])
        assert np.all(data["sensor_pc"] < data["grid_world_max"])
        assert data["sensor_pc"].tolist() == [
            [12.0, 17.0, 30.0],
            [17.899999618530273, 31.899999618530273, 45.900001525878906],
        ]
        assert str(data["format_version"]) == "1.0"
        assert str(data["label_schema_version"]) == "1.0"


def test_crop_dataset_copies_package_metadata_and_reports_dry_run(tmp_path: Path) -> None:
    input_root = tmp_path / "S2I-Dataset-70cls"
    output_root = tmp_path / "S2I-Dataset-70cls-crop"
    data_dir = input_root / "data"
    data_dir.mkdir(parents=True)
    _write_case(data_dir / "S2I_00001.npz")
    _write_json(input_root / "dataset_info.json", {"num_classes": 2})
    _write_json(input_root / "dataset_split.json", {"train": ["S2I_00001.npz"]})
    _write_json(input_root / "class_presence.json", {"num_cases": 1})

    dry_stats = crop_dataset(
        input_root=input_root,
        output_root=output_root,
        target_shape=(3, 6, 4),
        dry_run=True,
        workers=1,
    )

    assert dry_stats.total_files == 1
    assert dry_stats.cropped_files == 1
    assert dry_stats.full_coverage_fraction == 0.0
    assert not output_root.exists()

    stats = crop_dataset(
        input_root=input_root,
        output_root=output_root,
        target_shape=(3, 6, 4),
        dry_run=False,
        workers=1,
    )

    assert stats.total_files == 1
    assert (output_root / "data" / "S2I_00001.npz").exists()
    assert json.loads((output_root / "dataset_info.json").read_text()) == {"num_classes": 2}
    assert json.loads((output_root / "dataset_split.json").read_text()) == {
        "train": ["S2I_00001.npz"]
    }
    assert json.loads((output_root / "class_presence.json").read_text()) == {"num_cases": 1}


def test_crop_dataset_rejects_in_place_output(tmp_path: Path) -> None:
    input_root = tmp_path / "S2I-Dataset-70cls"
    (input_root / "data").mkdir(parents=True)

    with pytest.raises(ValueError, match="in-place"):
        crop_dataset(
            input_root=input_root,
            output_root=input_root,
            target_shape=(3, 6, 4),
            dry_run=True,
            workers=1,
        )
