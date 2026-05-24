from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from inference import resolve_case_paths, resolve_data_path, save_case_outputs


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_s2i_root(tmp_path: Path) -> Path:
    root = tmp_path / "Body-Tell"
    data_dir = root / "S2I-Dataset-70cls" / "data"
    data_dir.mkdir(parents=True)
    for name in ("S2I_00001.npz", "S2I_00002.npz"):
        (data_dir / name).write_bytes(b"placeholder")
    _write_json(
        root / "S2I-Dataset-70cls" / "dataset_split.json",
        {
            "train": ["S2I_00001.npz"],
            "val": ["S2I_00002.npz"],
            "test": [],
        },
    )
    return root


def _args(**overrides) -> Namespace:
    values = {
        "input": None,
        "case_id": None,
        "split": None,
        "limit": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_case_id_uses_configured_s2i_voxel_dir(tmp_path: Path) -> None:
    root = _make_s2i_root(tmp_path)
    data_cfg = {
        "voxel_dir": "S2I-Dataset-70cls/data",
        "split_path": "S2I-Dataset-70cls/dataset_split.json",
    }

    paths = resolve_case_paths(_args(case_id=["S2I_00001"]), root, data_cfg)

    assert paths == [root / "S2I-Dataset-70cls" / "data" / "S2I_00001.npz"]


def test_resolve_split_uses_configured_s2i_split_file(tmp_path: Path) -> None:
    root = _make_s2i_root(tmp_path)
    data_cfg = {
        "voxel_dir": "S2I-Dataset-70cls/data",
        "split_path": "S2I-Dataset-70cls/dataset_split.json",
    }

    paths = resolve_case_paths(_args(split="val"), root, data_cfg)

    assert paths == [root / "S2I-Dataset-70cls" / "data" / "S2I_00002.npz"]


def test_resolve_data_path_uses_configured_relative_artifact(tmp_path: Path) -> None:
    root = tmp_path / "Body-Tell"
    expected = root / "artifacts" / "text_embeddings" / "prompt_embeddings.pt"

    actual = resolve_data_path(
        root,
        "artifacts/text_embeddings/prompt_embeddings.pt",
        "unused/default.pt",
    )

    assert actual == expected


def test_save_case_outputs_stores_prompt_target_class_ids(tmp_path: Path) -> None:
    labels = np.array(
        [
            [[0, 1], [3, 4]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.int64,
    )
    masks_model = np.stack(
        [
            labels == 1,
            np.isin(labels, [3, 4]),
        ],
        axis=0,
    ).astype(np.uint8)
    args = Namespace(
        output=tmp_path / "outputs",
        threshold=0.5,
        save_model_shape=False,
        save_combined=True,
        evaluate=True,
        checkpoint=tmp_path / "checkpoint.pt",
    )
    case = {
        "case_id": "S2I_00002",
        "case_path": str(tmp_path / "Body-Tell" / "S2I-Dataset-70cls" / "data" / "S2I_00002.npz"),
        "original_shape": labels.shape,
        "labels": labels,
        "metadata": {},
    }
    prompt_records = [
        {
            "text": "liver",
            "prompt_id": "class_001_prompt_000",
            "index": 10,
            "source_type": "class",
            "class_id": 1,
            "aggregate_id": None,
        },
        {
            "text": "kidneys",
            "prompt_id": "agg_kidneys_prompt_000",
            "index": 20,
            "source_type": "aggregate",
            "class_id": None,
            "aggregate_id": "agg_kidneys",
        },
    ]
    vocab = {
        "aggregates": [
            {
                "id": "agg_kidneys",
                "component_class_ids": [3, 4],
            }
        ]
    }

    save_case_outputs(args, case, masks_model, None, prompt_records, vocab, {})

    npz_path = args.output / "S2I_00002" / "S2I_00002_predictions.npz"
    with np.load(npz_path) as data:
        np.testing.assert_array_equal(data["prompt_class_ids"], np.array([1, -1]))
        np.testing.assert_array_equal(data["prompt_target_class_ids"], np.array([[1, -1], [3, 4]]))
        assert str(data["case_path"].item()).endswith("S2I-Dataset-70cls/data/S2I_00002.npz")
