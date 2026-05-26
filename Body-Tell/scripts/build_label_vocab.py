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


SOURCE_DATASET_INFO = "Body-Tell/S2I-Dataset-70cls/dataset_info.json"

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

CLASS_PROMPTS: Dict[str, List[str]] = {'inside_body_empty': ['inside body empty space',
                       'empty unlabeled body interior',
                       'non-organ space inside the body'],
 'liver': ['liver', 'hepatic organ', 'liver tissue', 'hepatic tissue', 'liver organ'],
 'spleen': ['spleen', 'splenic organ', 'spleen tissue', 'splenic tissue', 'spleen organ'],
 'kidney_left': ['left kidney',
                 'left renal organ',
                 'left kidney tissue',
                 'left renal tissue',
                 'left kidney organ'],
 'kidney_right': ['right kidney',
                  'right renal organ',
                  'right kidney tissue',
                  'right renal tissue',
                  'right kidney organ'],
 'stomach': ['stomach',
             'gastric organ',
             'stomach structure',
             'gastric structure',
             'stomach organ'],
 'pancreas': ['pancreas',
              'pancreatic organ',
              'pancreas tissue',
              'pancreatic tissue',
              'pancreas organ'],
 'gallbladder': ['gallbladder',
                 'gall bladder',
                 'biliary gallbladder',
                 'biliary gall bladder',
                 'gallbladder organ'],
 'urinary_bladder': ['urinary bladder',
                     'bladder',
                     'urinary bladder organ',
                     'vesical organ',
                     'urinary bladder structure'],
 'prostate': ['prostate',
              'prostate gland',
              'prostatic tissue',
              'prostatic gland',
              'prostate organ'],
 'heart': ['heart', 'cardiac organ', 'heart tissue', 'cardiac structure', 'heart organ'],
 'brain': ['brain', 'brain tissue', 'cerebral tissue', 'cerebral organ', 'brain organ'],
 'thyroid_gland': ['thyroid gland',
                   'thyroid',
                   'thyroid tissue',
                   'thyroid gland tissue',
                   'thyroid organ'],
 'spinal_cord': ['spinal cord',
                 'spinal cord tissue',
                 'cord within spine',
                 'spinal neural cord',
                 'spinal cord structure'],
 'lung': ['lung', 'lung tissue', 'pulmonary tissue', 'lung organ', 'pulmonary structure'],
 'esophagus': ['esophagus',
               'oesophagus',
               'esophageal tube',
               'esophageal structure',
               'esophagus structure'],
 'trachea': ['trachea', 'windpipe', 'tracheal airway', 'tracheal tube', 'trachea airway'],
 'small_bowel': ['small bowel',
                 'small intestine',
                 'small bowel loops',
                 'small bowel structure',
                 'small intestinal loops'],
 'duodenum': ['duodenum',
              'duodenal segment',
              'duodenal bowel',
              'duodenum segment',
              'duodenal structure'],
 'colon': ['colon', 'large bowel', 'colon segment', 'large intestine', 'colonic segment'],
 'adrenal_gland_left': ['left adrenal gland',
                        'left suprarenal gland',
                        'left adrenal gland tissue',
                        'left adrenal tissue',
                        'left adrenal organ'],
 'adrenal_gland_right': ['right adrenal gland',
                         'right suprarenal gland',
                         'right adrenal gland tissue',
                         'right adrenal tissue',
                         'right adrenal organ'],
 'spine': ['spine',
           'vertebral column',
           'spinal column bones',
           'spinal bony column',
           'spine structure'],
 'skull': ['skull', 'cranium', 'skull bone', 'cranial bone', 'cranial structure'],
 'sternum': ['sternum', 'breastbone', 'sternal bone', 'sternum bone', 'sternal structure'],
 'costal_cartilages': ['costal cartilages',
                       'rib cartilages',
                       'costal cartilage structures',
                       'rib cartilage structures',
                       'costal cartilage set'],
 'scapula_left': ['left scapula',
                  'left shoulder blade',
                  'left scapular bone',
                  'left scapula bone',
                  'left scapular structure'],
 'scapula_right': ['right scapula',
                   'right shoulder blade',
                   'right scapular bone',
                   'right scapula bone',
                   'right scapular structure'],
 'clavicula_left': ['left clavicle',
                    'left collarbone',
                    'left clavicular bone',
                    'left clavicle bone',
                    'left clavicular structure'],
 'clavicula_right': ['right clavicle',
                     'right collarbone',
                     'right clavicular bone',
                     'right clavicle bone',
                     'right clavicular structure'],
 'humerus_left': ['left humerus',
                  'left upper arm bone',
                  'left humeral bone',
                  'left humerus bone',
                  'left upper arm skeletal bone'],
 'humerus_right': ['right humerus',
                   'right upper arm bone',
                   'right humeral bone',
                   'right humerus bone',
                   'right upper arm skeletal bone'],
 'hip_left': ['left hip bone',
              'left pelvic bone',
              'left os coxae',
              'left pelvic osseous structure',
              'left hip osseous structure'],
 'hip_right': ['right hip bone',
               'right pelvic bone',
               'right os coxae',
               'right pelvic osseous structure',
               'right hip osseous structure'],
 'femur_left': ['left femur',
                'left thigh bone',
                'left femoral bone',
                'left femur bone',
                'left thigh skeletal bone'],
 'femur_right': ['right femur',
                 'right thigh bone',
                 'right femoral bone',
                 'right femur bone',
                 'right thigh skeletal bone'],
 'gluteus_maximus_left': ['left gluteus maximus muscle',
                          'left gluteus maximus',
                          'left gluteus maximus tissue',
                          'left gluteus maximus structure',
                          'left gluteus maximus musculature'],
 'gluteus_maximus_right': ['right gluteus maximus muscle',
                           'right gluteus maximus',
                           'right gluteus maximus tissue',
                           'right gluteus maximus structure',
                           'right gluteus maximus musculature'],
 'gluteus_medius_left': ['left gluteus medius muscle',
                         'left gluteus medius',
                         'left gluteus medius tissue',
                         'left gluteus medius structure',
                         'left gluteus medius musculature'],
 'gluteus_medius_right': ['right gluteus medius muscle',
                          'right gluteus medius',
                          'right gluteus medius tissue',
                          'right gluteus medius structure',
                          'right gluteus medius musculature'],
 'gluteus_minimus_left': ['left gluteus minimus muscle',
                          'left gluteus minimus',
                          'left gluteus minimus tissue',
                          'left gluteus minimus structure',
                          'left gluteus minimus musculature'],
 'gluteus_minimus_right': ['right gluteus minimus muscle',
                           'right gluteus minimus',
                           'right gluteus minimus tissue',
                           'right gluteus minimus structure',
                           'right gluteus minimus musculature'],
 'autochthon_left': ['left autochthonous back muscles',
                     'left intrinsic back muscles',
                     'left paraspinal autochthonous muscles',
                     'left autochthonous spine muscles',
                     'left deep back muscles'],
 'autochthon_right': ['right autochthonous back muscles',
                      'right intrinsic back muscles',
                      'right paraspinal autochthonous muscles',
                      'right autochthonous spine muscles',
                      'right deep back muscles'],
 'iliopsoas_left': ['left iliopsoas muscle',
                    'left iliopsoas',
                    'left psoas iliacus muscle',
                    'left iliopsoas musculature',
                    'left iliacus psoas muscle'],
 'iliopsoas_right': ['right iliopsoas muscle',
                     'right iliopsoas',
                     'right psoas iliacus muscle',
                     'right iliopsoas musculature',
                     'right iliacus psoas muscle']}

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
        f"{side} {ordinal} rib structure",
    ]


def prompts_for_source(source_name: str) -> List[str]:
    if source_name.startswith("rib_"):
        return rib_prompts(source_name)
    if source_name not in CLASS_PROMPTS:
        raise KeyError(f"No prompt mapping defined for {source_name}")
    return CLASS_PROMPTS[source_name]


AGGREGATES: List[Dict[str, Any]] = [{'id': 'agg_kidneys',
  'canonical': 'kidneys',
  'prompts': ['kidneys',
              'bilateral kidneys',
              'renal organs',
              'both kidneys',
              'left and right kidneys'],
  'component_class_ids': [3, 4],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Inference/analysis concept composed from left and right kidney labels.'},
 {'id': 'agg_adrenal_glands',
  'canonical': 'adrenal glands',
  'prompts': ['adrenal glands',
              'bilateral adrenal glands',
              'suprarenal glands',
              'both adrenal glands',
              'left and right adrenal glands'],
  'component_class_ids': [20, 21],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right adrenal gland labels.'},
 {'id': 'agg_ribs',
  'canonical': 'ribs',
  'prompts': ['ribs', 'left and right ribs', 'rib bones', 'all ribs', 'bilateral rib bones'],
  'component_class_ids': [23,
                          24,
                          25,
                          26,
                          27,
                          28,
                          29,
                          30,
                          31,
                          32,
                          33,
                          34,
                          35,
                          36,
                          37,
                          38,
                          39,
                          40,
                          41,
                          42,
                          43,
                          44,
                          45,
                          46],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from individual left and right rib labels 1 through 12.'},
 {'id': 'agg_left_ribs',
  'canonical': 'left ribs',
  'prompts': ['left ribs',
              'left rib bones',
              'left costal bones',
              'left rib set',
              'left rib group'],
  'component_class_ids': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left rib labels 1 through 12.'},
 {'id': 'agg_right_ribs',
  'canonical': 'right ribs',
  'prompts': ['right ribs',
              'right rib bones',
              'right costal bones',
              'right rib set',
              'right rib group'],
  'component_class_ids': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from right rib labels 1 through 12.'},
 {'id': 'agg_lungs',
  'canonical': 'lungs',
  'prompts': ['lungs',
              'whole lung label',
              'pulmonary organ',
              'combined lung structure',
              'complete lung structure'],
  'component_class_ids': [14],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Body-Tell currently exposes one lung class rather than left/right lung labels.'},
 {'id': 'agg_hips',
  'canonical': 'hip bones',
  'prompts': ['hip bones',
              'bilateral hip bones',
              'pelvic hip bones',
              'left and right hip bones',
              'paired hip bones'],
  'component_class_ids': [56, 57],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right hip labels.'},
 {'id': 'agg_femurs',
  'canonical': 'femurs',
  'prompts': ['femurs',
              'bilateral femurs',
              'thigh bones',
              'left and right femurs',
              'paired femur bones'],
  'component_class_ids': [58, 59],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right femur labels.'},
 {'id': 'agg_gluteal_muscles',
  'canonical': 'gluteal muscles',
  'prompts': ['gluteal muscles',
              'bilateral gluteal muscles',
              'buttock muscles',
              'gluteus muscle group',
              'gluteal musculature'],
  'component_class_ids': [60, 61, 62, 63, 64, 65],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from gluteus maximus, medius, and minimus labels.'},
 {'id': 'agg_iliopsoas_muscles',
  'canonical': 'iliopsoas muscles',
  'prompts': ['iliopsoas muscles',
              'bilateral iliopsoas',
              'psoas iliacus muscles',
              'left and right iliopsoas muscles',
              'iliopsoas musculature'],
  'component_class_ids': [68, 69],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right iliopsoas labels.'},
 {'id': 'agg_shoulder_girdle_bones',
  'canonical': 'shoulder girdle bones',
  'prompts': ['shoulder girdle bones',
              'scapulae and clavicles',
              'bilateral scapulae and clavicles',
              'shoulder blade and collarbone bones',
              'pectoral girdle bones'],
  'component_class_ids': [50, 51, 52, 53],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right scapula and clavicle labels.'},
 {'id': 'agg_humeri',
  'canonical': 'humeri',
  'prompts': ['humeri',
              'bilateral humeri',
              'upper arm bones',
              'left and right humerus bones',
              'paired humerus bones'],
  'component_class_ids': [54, 55],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right humerus labels.'},
 {'id': 'agg_hip_and_femur_bones',
  'canonical': 'hip and femur bones',
  'prompts': ['hip and femur bones',
              'bilateral hip and femur bones',
              'pelvic and thigh bones',
              'proximal lower limb bones',
              'paired hip bones and femurs'],
  'component_class_ids': [56, 57, 58, 59],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right hip and femur labels.'},
 {'id': 'agg_autochthonous_back_muscles',
  'canonical': 'autochthonous back muscles',
  'prompts': ['autochthonous back muscles',
              'bilateral intrinsic back muscles',
              'paraspinal autochthonous muscles',
              'left and right autochthonous back muscles',
              'deep back muscles'],
  'component_class_ids': [66, 67],
  'train_as_positive': False,
  'eval_as_foreground': False,
  'notes': 'Composed from left and right autochthonous back muscle labels.'}]


def build_aggregates() -> List[Dict[str, Any]]:
    return [
        {
            **aggregate,
            "prompts": list(aggregate["prompts"]),
            "component_class_ids": list(aggregate["component_class_ids"]),
        }
        for aggregate in AGGREGATES
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
                    else "Base class label as provided by Body-Tell/S2I-Dataset-70cls/data."
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
            "status": "step5_static_review_complete",
            "reviewed_by": "sp_b_sampling_vocab_worker",
            "reviewed_at": "2026-05-26",
            "notes": (
                "Expanded under VoxTell Appendix C principles: anatomy-level synonyms "
                "and rephrasings only, laterality retained for left/right labels, "
                "and aggregate concepts limited to unions of existing component_class_ids. "
                "Independent clinical expert review is still recommended before publication claims."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-info",
        default=ROOT / "S2I-Dataset-70cls" / "dataset_info.json",
        type=Path,
        help="Path to Body-Tell/S2I-Dataset-70cls/dataset_info.json",
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
