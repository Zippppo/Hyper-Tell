#!/usr/bin/env python3
"""Build Body-Tell Phase 0 label vocabulary from dataset_info.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.vocabulary import (  # noqa: E402
    INSTRUCTION,
    TEXT_ENCODER,
    VOCAB_VERSION,
    read_json,
    validate_label_vocab,
    write_json,
)


SOURCE_DATASET_INFO = "Body-Tell/Dataset/dataset_info.json"

ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
}

CLASS_PROMPTS: Dict[str, List[str]] = {
    "inside_body_empty": [
        "inside body empty space",
        "empty unlabeled body interior",
        "non-organ space inside the body",
    ],
    "liver": ["liver", "hepatic organ", "liver tissue"],
    "spleen": ["spleen", "splenic organ", "spleen tissue"],
    "kidney_left": ["left kidney", "left renal organ", "left kidney tissue"],
    "kidney_right": ["right kidney", "right renal organ", "right kidney tissue"],
    "stomach": ["stomach", "gastric organ", "stomach wall"],
    "pancreas": ["pancreas", "pancreatic organ", "pancreas tissue"],
    "gallbladder": ["gallbladder", "gall bladder", "biliary gallbladder"],
    "urinary_bladder": ["urinary bladder", "bladder", "urinary bladder wall"],
    "prostate": ["prostate", "prostate gland", "prostatic tissue"],
    "heart": ["heart", "cardiac organ", "myocardium and heart"],
    "brain": ["brain", "brain tissue", "cerebral tissue"],
    "thyroid_gland": ["thyroid gland", "thyroid", "thyroid tissue"],
    "spinal_cord": ["spinal cord", "spinal cord tissue", "cord within spine"],
    "lung": ["lung", "lung tissue", "pulmonary tissue"],
    "esophagus": ["esophagus", "oesophagus", "esophageal tube"],
    "trachea": ["trachea", "windpipe", "tracheal airway"],
    "small_bowel": ["small bowel", "small intestine", "small bowel loops"],
    "duodenum": ["duodenum", "duodenal segment", "duodenal bowel"],
    "colon": ["colon", "large bowel", "colon segment"],
    "adrenal_gland_left": [
        "left adrenal gland",
        "left suprarenal gland",
        "left adrenal gland tissue",
    ],
    "adrenal_gland_right": [
        "right adrenal gland",
        "right suprarenal gland",
        "right adrenal gland tissue",
    ],
    "spine": ["spine", "vertebral column", "spinal column bones"],
    "skull": ["skull", "cranium", "skull bone"],
    "sternum": ["sternum", "breastbone", "sternal bone"],
    "costal_cartilages": [
        "costal cartilages",
        "rib cartilages",
        "costal cartilage structures",
    ],
    "scapula_left": ["left scapula", "left shoulder blade", "left scapular bone"],
    "scapula_right": ["right scapula", "right shoulder blade", "right scapular bone"],
    "clavicula_left": ["left clavicle", "left collarbone", "left clavicular bone"],
    "clavicula_right": ["right clavicle", "right collarbone", "right clavicular bone"],
    "humerus_left": ["left humerus", "left upper arm bone", "left humeral bone"],
    "humerus_right": ["right humerus", "right upper arm bone", "right humeral bone"],
    "hip_left": ["left hip bone", "left pelvic bone", "left os coxae"],
    "hip_right": ["right hip bone", "right pelvic bone", "right os coxae"],
    "femur_left": ["left femur", "left thigh bone", "left femoral bone"],
    "femur_right": ["right femur", "right thigh bone", "right femoral bone"],
    "gluteus_maximus_left": [
        "left gluteus maximus muscle",
        "left gluteus maximus",
        "left gluteus maximus tissue",
    ],
    "gluteus_maximus_right": [
        "right gluteus maximus muscle",
        "right gluteus maximus",
        "right gluteus maximus tissue",
    ],
    "gluteus_medius_left": [
        "left gluteus medius muscle",
        "left gluteus medius",
        "left gluteus medius tissue",
    ],
    "gluteus_medius_right": [
        "right gluteus medius muscle",
        "right gluteus medius",
        "right gluteus medius tissue",
    ],
    "gluteus_minimus_left": [
        "left gluteus minimus muscle",
        "left gluteus minimus",
        "left gluteus minimus tissue",
    ],
    "gluteus_minimus_right": [
        "right gluteus minimus muscle",
        "right gluteus minimus",
        "right gluteus minimus tissue",
    ],
    "autochthon_left": [
        "left autochthonous back muscles",
        "left intrinsic back muscles",
        "left paraspinal autochthonous muscles",
    ],
    "autochthon_right": [
        "right autochthonous back muscles",
        "right intrinsic back muscles",
        "right paraspinal autochthonous muscles",
    ],
    "iliopsoas_left": [
        "left iliopsoas muscle",
        "left iliopsoas",
        "left psoas iliacus muscle",
    ],
    "iliopsoas_right": [
        "right iliopsoas muscle",
        "right iliopsoas",
        "right psoas iliacus muscle",
    ],
}


def rib_prompts(source_name: str) -> List[str]:
    parts = source_name.split("_")
    side = parts[1]
    rib_number = int(parts[2])
    ordinal = ORDINALS[rib_number]
    return [
        f"{side} {ordinal} rib",
        f"{side} rib {rib_number}",
        f"{side} {ordinal} rib bone",
        f"{side} {ordinal} costal bone",
    ]


def prompts_for_source(source_name: str) -> List[str]:
    if source_name.startswith("rib_"):
        return rib_prompts(source_name)
    if source_name not in CLASS_PROMPTS:
        raise KeyError(f"No prompt mapping defined for {source_name}")
    return CLASS_PROMPTS[source_name]


def build_aggregates() -> List[Dict[str, Any]]:
    return [
        {
            "id": "agg_kidneys",
            "canonical": "kidneys",
            "prompts": ["kidneys", "bilateral kidneys", "renal organs"],
            "component_class_ids": [3, 4],
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Inference/analysis concept composed from left and right kidney labels.",
        },
        {
            "id": "agg_adrenal_glands",
            "canonical": "adrenal glands",
            "prompts": ["adrenal glands", "bilateral adrenal glands", "suprarenal glands"],
            "component_class_ids": [20, 21],
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from left and right adrenal gland labels.",
        },
        {
            "id": "agg_ribs",
            "canonical": "ribs",
            "prompts": ["ribs", "left and right ribs", "rib bones"],
            "component_class_ids": list(range(23, 47)),
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from individual left and right rib labels 1 through 12.",
        },
        {
            "id": "agg_left_ribs",
            "canonical": "left ribs",
            "prompts": ["left ribs", "left rib bones", "left costal bones"],
            "component_class_ids": list(range(23, 35)),
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from left rib labels 1 through 12.",
        },
        {
            "id": "agg_right_ribs",
            "canonical": "right ribs",
            "prompts": ["right ribs", "right rib bones", "right costal bones"],
            "component_class_ids": list(range(35, 47)),
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from right rib labels 1 through 12.",
        },
        {
            "id": "agg_lungs",
            "canonical": "lungs",
            "prompts": ["lungs", "whole lung label", "pulmonary organ"],
            "component_class_ids": [14],
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Body-Tell currently exposes one lung class rather than left/right lung labels.",
        },
        {
            "id": "agg_hips",
            "canonical": "hip bones",
            "prompts": ["hip bones", "bilateral hip bones", "pelvic hip bones"],
            "component_class_ids": [56, 57],
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from left and right hip labels.",
        },
        {
            "id": "agg_femurs",
            "canonical": "femurs",
            "prompts": ["femurs", "bilateral femurs", "thigh bones"],
            "component_class_ids": [58, 59],
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from left and right femur labels.",
        },
        {
            "id": "agg_gluteal_muscles",
            "canonical": "gluteal muscles",
            "prompts": ["gluteal muscles", "bilateral gluteal muscles", "buttock muscles"],
            "component_class_ids": [60, 61, 62, 63, 64, 65],
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from gluteus maximus, medius, and minimus labels.",
        },
        {
            "id": "agg_iliopsoas_muscles",
            "canonical": "iliopsoas muscles",
            "prompts": ["iliopsoas muscles", "bilateral iliopsoas", "psoas iliacus muscles"],
            "component_class_ids": [68, 69],
            "train_as_positive": False,
            "eval_as_foreground": False,
            "notes": "Composed from left and right iliopsoas labels.",
        },
    ]


def build_vocab(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    classes = []
    class_names = dataset_info["class_names"]
    for class_id, source_name in enumerate(class_names):
        prompts = prompts_for_source(source_name)
        train_as_positive = class_id != 0
        classes.append(
            {
                "id": class_id,
                "source_name": source_name,
                "canonical": prompts[0],
                "prompts": prompts,
                "train_as_positive": train_as_positive,
                "eval_as_foreground": train_as_positive,
                "notes": (
                    "Background/empty interior label; excluded from foreground training "
                    "and foreground mean Dice."
                    if class_id == 0
                    else "Base class label as provided by Body-Tell/Dataset/voxel_data."
                ),
            }
        )

    return {
        "version": VOCAB_VERSION,
        "source_dataset_info": SOURCE_DATASET_INFO,
        "language": "en",
        "text_encoder": TEXT_ENCODER,
        "instruction": INSTRUCTION,
        "classes": classes,
        "aggregates": build_aggregates(),
        "ignore_as_positive": [0],
        "review": {
            "status": "pending_manual_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "notes": "Auto-built Phase 0 vocabulary; review prompts and aggregates before Phase 1.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-info",
        default=ROOT / "Dataset" / "dataset_info.json",
        type=Path,
        help="Path to Body-Tell/Dataset/dataset_info.json",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "configs" / "label_vocab.json",
        type=Path,
        help="Output label_vocab.json path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_info = read_json(args.dataset_info)
    vocab = build_vocab(dataset_info)
    validation = validate_label_vocab(vocab, dataset_info=dataset_info, strict=True)
    write_json(vocab, args.output)
    print(f"wrote {args.output}")
    print(f"classes={len(vocab['classes'])} aggregates={len(vocab['aggregates'])}")
    if validation["warnings"]:
        print("warnings:")
        for warning in validation["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
