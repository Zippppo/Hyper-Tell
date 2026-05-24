from __future__ import annotations

from pathlib import Path

import numpy as np

from visualize_predictions import build_gt_mask, find_original_data_path, load_predictions


def _write_minimal_prediction(path: Path, prompt_id: str = "class_001_prompt_000") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pred_masks=np.zeros((1, 2, 2, 2), dtype=np.uint8),
        prompt_texts=np.asarray(["liver"]),
        prompt_ids=np.asarray([prompt_id]),
        threshold=np.asarray(0.5, dtype=np.float32),
        grid_voxel_size=np.ones(3, dtype=np.float32),
        grid_world_min=np.zeros(3, dtype=np.float32),
    )


def test_load_predictions_infers_target_ids_from_string_prompt_ids(tmp_path: Path) -> None:
    prediction_path = tmp_path / "S2I_00002_predictions.npz"
    _write_minimal_prediction(prediction_path)

    pred_data = load_predictions(prediction_path)
    labels = np.array([[[0, 1], [1, 2]]], dtype=np.int64)

    assert pred_data["target_class_ids"] == [[1]]
    assert int(build_gt_mask(labels, pred_data["target_class_ids"][0]).sum()) == 2


def test_find_original_data_path_prefers_s2i_dataset(tmp_path: Path) -> None:
    root = tmp_path / "Body-Tell"
    prediction_path = root / "outputs" / "inference_s2i_smoke" / "S2I_00002" / "S2I_00002_predictions.npz"
    original_path = root / "S2I-Dataset-70cls" / "data" / "S2I_00002.npz"
    _write_minimal_prediction(prediction_path)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_bytes(b"placeholder")

    assert find_original_data_path(prediction_path, {"case_path": None}) == original_path
