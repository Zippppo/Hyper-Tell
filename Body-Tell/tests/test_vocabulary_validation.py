from __future__ import annotations

import json
from pathlib import Path

from body_tell.data.vocabulary import load_label_vocab, validate_label_vocab


ROOT = Path(__file__).resolve().parents[1]


def test_static_label_vocab_satisfies_step5_acceptance() -> None:
    vocab = load_label_vocab(ROOT / "configs" / "label_vocab.json")
    dataset_info = json.loads(
        (ROOT / "S2I-Dataset-70cls" / "dataset_info.json").read_text(encoding="utf-8")
    )

    validation = validate_label_vocab(vocab, dataset_info=dataset_info)
    underexpanded = [
        (cls["id"], cls["source_name"], len(cls.get("prompts", [])))
        for cls in vocab["classes"]
        if cls.get("train_as_positive", False) and len(cls.get("prompts", [])) < 5
    ]
    aggregate_component_gaps = [
        aggregate.get("id")
        for aggregate in vocab.get("aggregates", [])
        if not aggregate.get("component_class_ids")
    ]

    assert validation["errors"] == []
    assert underexpanded == []
    assert aggregate_component_gaps == []


def test_validate_label_vocab_rejects_step5_acceptance_gaps() -> None:
    vocab = {
        "version": "phase0-2026-05-20",
        "classes": [
            {
                "id": 0,
                "source_name": "inside_body_empty",
                "canonical": "inside body empty space",
                "prompts": ["inside body empty space"],
                "train_as_positive": False,
            },
            {
                "id": 1,
                "source_name": "liver",
                "canonical": "liver",
                "prompts": [
                    "liver",
                    "hepatic organ",
                    "liver tissue",
                    "hepatic tissue",
                ],
                "train_as_positive": True,
            },
        ],
        "aggregates": [
            {
                "id": "agg_empty",
                "canonical": "unsupported combined label",
                "prompts": ["unsupported combined label"],
                "component_class_ids": [],
            }
        ],
        "ignore_as_positive": [0],
    }
    dataset_info = {
        "num_classes": 2,
        "class_names": ["inside_body_empty", "liver"],
    }

    validation = validate_label_vocab(vocab, dataset_info=dataset_info)

    assert any(
        "class 1" in error and "at least 5 prompts" in error
        for error in validation["errors"]
    )
    assert any(
        "aggregate agg_empty" in error and "component_class_ids" in error
        for error in validation["errors"]
    )
