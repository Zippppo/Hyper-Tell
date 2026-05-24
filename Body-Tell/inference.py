#!/usr/bin/env python3
"""Run Body-Tell prompt-conditioned inference on HyperBody ``.npz`` cases.

python Body-Tell/inference.py \
--checkpoint Body-Tell/checkpoints/test-S2I/best.pt \
--case-id S2I_00001 \
--prompts liver \
--output Body-Tell/outputs/inference_s2i_smoke \
--device cuda --gpu 0 --amp \
--prompt-batch-size 1 \
--evaluate --save-combined

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.amp import autocast
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.dataset import fit_array_to_shape, voxelize_sensor_points  # noqa: E402
from body_tell.data.vocabulary import (  # noqa: E402
    EMBEDDING_DIM,
    INSTRUCTION,
    TEXT_ENCODER,
    build_target_mask,
    flatten_prompt_records,
    foreground_eval_class_ids,
    load_label_vocab,
    read_json,
)
from body_tell.models.voxtell_body_model import VoxTellBodyConfig, VoxTellBodyModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Body-Tell inference for prompt-conditioned occupancy segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "phase1_voxtell_body.yaml",
        help="Training config used to build the model and resolve data artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "phase1_bz2" / "best.pt",
        help="Checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="One or more HyperBody .npz case files.",
    )
    parser.add_argument(
        "--case-id",
        nargs="*",
        default=None,
        help="Case ids or filenames under the configured data.voxel_dir, e.g. S2I_00001.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="Run inference on a dataset split from dataset_split.json.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of cases.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "inference",
        help="Output directory.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help=(
            "Prompt texts or aliases from label_vocab.json. Exact cached prompts, "
            "canonical names, and source names like kidney_left are supported."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Text file with one prompt per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--all-foreground",
        action="store_true",
        help="Use one canonical prompt for each eval foreground class.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Use one canonical prompt for every class, including class 0.",
    )
    parser.add_argument(
        "--include-aggregates",
        action="store_true",
        help="Append canonical aggregate prompts when using --all-foreground or --all-classes.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Path to prompt_embeddings.pt. Defaults to data.embedding_cache_path or <root>/artifacts/text_embeddings/prompt_embeddings.pt.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Path to label_vocab.json. Defaults to data.vocab_path or <root>/configs/label_vocab.json.",
    )
    parser.add_argument(
        "--encode-missing",
        action="store_true",
        help="Encode prompts that are not present in the cache with the Qwen text encoder.",
    )
    parser.add_argument("--text-model", default=TEXT_ENCODER)
    parser.add_argument("--text-device", default="auto", help="auto, cpu, cuda, or cuda:<id>.")
    parser.add_argument(
        "--text-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Dtype for optional text encoding.",
    )
    parser.add_argument("--text-cache-dir", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=4,
        help="Number of prompts evaluated per model forward pass.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:<id>. Auto if omitted.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id used when --device is cuda.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for model inference.")
    parser.add_argument("--save-probs", action="store_true", help="Save sigmoid probabilities as float16.")
    parser.add_argument(
        "--save-model-shape",
        action="store_true",
        help="Also save masks before restoring them to the original case shape.",
    )
    parser.add_argument(
        "--save-combined",
        action="store_true",
        help="Save a combined label volume where prompt i is encoded as i+1.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If voxel_labels are present and prompt targets are known, compute per-prompt Dice.",
    )
    parser.add_argument(
        "--ignore-checkpoint-config",
        action="store_true",
        help="Always use --config instead of the config embedded in the checkpoint.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_body_root(root_value: str | Path) -> Path:
    candidate = Path(root_value)
    if candidate.exists():
        return candidate
    repo_relative = ROOT.parent / candidate
    if repo_relative.exists():
        return repo_relative
    if candidate.name == ROOT.name:
        return ROOT
    return candidate


def resolve_data_path(
    body_root: Path,
    configured_path: str | Path | None,
    default_relative_path: str | Path,
) -> Path:
    path = Path(configured_path) if configured_path is not None else Path(default_relative_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    repo_relative = ROOT.parent / path
    if repo_relative.exists():
        return repo_relative
    return body_root / path


def resolve_device(device_arg: str | None, gpu: int) -> torch.device:
    if device_arg is None:
        return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device(f"cuda:{gpu}")
    return torch.device(device_arg)


def build_model(cfg: Mapping[str, Any]) -> VoxTellBodyModel:
    mc = cfg["model"]
    config = VoxTellBodyConfig(
        input_channels=mc["input_channels"],
        encoder_channels=tuple(mc["encoder_channels"]),
        text_embedding_dim=mc["text_embedding_dim"],
        query_dim=mc["query_dim"],
        text_projection_hidden_dim=mc["text_projection_hidden_dim"],
        transformer_num_heads=mc["transformer_num_heads"],
        transformer_layers=mc["transformer_layers"],
        transformer_feedforward_dim=mc["transformer_feedforward_dim"],
        transformer_dropout=mc.get("transformer_dropout", 0.1),
        decoder_layer=mc["decoder_layer"],
        num_maskformer_stages=mc["num_maskformer_stages"],
        num_heads=mc["num_heads"],
        deep_supervision=mc["deep_supervision"],
    )
    return VoxTellBodyModel(config)


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def load_model(checkpoint: Mapping[str, Any], cfg: Mapping[str, Any], device: torch.device) -> VoxTellBodyModel:
    model = build_model(cfg)
    state_dict = strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def canonical_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ").strip().casefold())


def first_canonical_record(
    records: Sequence[Mapping[str, Any]],
    *,
    source_type: str,
    class_id: int | None = None,
    aggregate_id: str | None = None,
) -> Mapping[str, Any]:
    for record in records:
        if record["source_type"] != source_type:
            continue
        if class_id is not None and int(record["class_id"]) != int(class_id):
            continue
        if aggregate_id is not None and str(record["aggregate_id"]) != str(aggregate_id):
            continue
        if bool(record.get("is_canonical", False)):
            return record
    raise KeyError(f"No canonical prompt record for {source_type}:{class_id or aggregate_id}")


def build_prompt_aliases(
    vocab: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    aliases: dict[str, Mapping[str, Any]] = {}

    for record in records:
        aliases.setdefault(canonical_key(str(record["text"])), record)
        aliases.setdefault(canonical_key(str(record["prompt_id"])), record)

    for cls in vocab.get("classes", []):
        record = first_canonical_record(records, source_type="class", class_id=int(cls["id"]))
        values = [
            str(cls["id"]),
            str(cls.get("source_name", "")),
            str(cls.get("canonical", "")),
            *[str(prompt) for prompt in cls.get("prompts", [])],
        ]
        for value in values:
            if value:
                aliases.setdefault(canonical_key(value), record)

    for aggregate in vocab.get("aggregates", []):
        aggregate_id = str(aggregate["id"])
        record = first_canonical_record(records, source_type="aggregate", aggregate_id=aggregate_id)
        values = [
            aggregate_id,
            str(aggregate.get("canonical", "")),
            *[str(prompt) for prompt in aggregate.get("prompts", [])],
        ]
        for value in values:
            if value:
                aliases.setdefault(canonical_key(value), record)

    return aliases


def read_prompt_file(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def select_prompt_records(
    args: argparse.Namespace,
    vocab: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    selected: list[Mapping[str, Any]] = []
    missing_texts: list[str] = []

    if args.all_foreground or args.all_classes:
        if args.all_classes:
            class_ids = [int(cls["id"]) for cls in vocab.get("classes", [])]
        else:
            class_ids = foreground_eval_class_ids(vocab)
        for class_id in class_ids:
            selected.append(first_canonical_record(records, source_type="class", class_id=class_id))
        if args.include_aggregates:
            for aggregate in vocab.get("aggregates", []):
                selected.append(
                    first_canonical_record(
                        records,
                        source_type="aggregate",
                        aggregate_id=str(aggregate["id"]),
                    )
                )

    prompt_texts = list(args.prompts or [])
    if args.prompt_file is not None:
        prompt_texts.extend(read_prompt_file(args.prompt_file))

    aliases = build_prompt_aliases(vocab, records)
    for prompt in prompt_texts:
        record = aliases.get(canonical_key(prompt))
        if record is None:
            missing_texts.append(prompt)
        else:
            selected.append(record)

    if not selected and not missing_texts:
        raise ValueError("No prompts selected. Use --prompts, --prompt-file, --all-foreground, or --all-classes.")

    deduped: list[Mapping[str, Any]] = []
    seen: set[tuple[str, int | str]] = set()
    for record in selected:
        key = (str(record["source_type"]), int(record["index"]))
        if key not in seen:
            seen.add(key)
            deduped.append(record)
    return deduped, missing_texts


def resolve_text_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_text_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def wrap_with_instruction(prompts: Sequence[str]) -> list[str]:
    return [f"Instruct: {INSTRUCTION}\nQuery: {prompt}" for prompt in prompts]


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


@torch.inference_mode()
def encode_free_text_prompts(args: argparse.Namespace, prompts: Sequence[str]) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    device = resolve_text_device(args.text_device)
    dtype = resolve_text_dtype(args.text_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model,
        padding_side="left",
        cache_dir=str(args.text_cache_dir) if args.text_cache_dir else None,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.text_model,
        cache_dir=str(args.text_cache_dir) if args.text_cache_dir else None,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    model = model.to(device)

    tokens = tokenizer(
        wrap_with_instruction(prompts),
        padding=True,
        truncation=True,
        max_length=args.max_text_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    output = model(**tokens)
    embeddings = last_token_pool(output.last_hidden_state, tokens["attention_mask"])
    embeddings = embeddings.detach().to("cpu", dtype=torch.float32)
    if tuple(embeddings.shape) != (len(prompts), EMBEDDING_DIM):
        raise ValueError(f"Expected missing prompt embeddings {(len(prompts), EMBEDDING_DIM)}, got {tuple(embeddings.shape)}")
    return embeddings


def load_prompt_embeddings(
    args: argparse.Namespace,
    cache_path: Path,
    records: Sequence[Mapping[str, Any]],
    missing_texts: Sequence[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached_embeddings = cache["embeddings"].float().contiguous()
    if cached_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {cached_embeddings.shape[1]}")

    prompt_records = [dict(record) for record in records]
    embeddings = [cached_embeddings[int(record["index"])] for record in records]

    if missing_texts:
        if not args.encode_missing:
            available = ", ".join(str(record["text"]) for record in prompt_records[:12])
            raise ValueError(
                "Prompts not found in embedding cache: "
                f"{list(missing_texts)}. Use cached vocabulary prompts or pass --encode-missing. "
                f"Cache examples: {available}"
            )
        free_embeddings = encode_free_text_prompts(args, missing_texts)
        for offset, text in enumerate(missing_texts):
            prompt_records.append(
                {
                    "index": -1,
                    "prompt_id": f"free_text_{offset:03d}",
                    "source_type": "free_text",
                    "class_id": None,
                    "aggregate_id": None,
                    "text": text,
                    "is_canonical": False,
                }
            )
            embeddings.append(free_embeddings[offset])

    return torch.stack(embeddings, dim=0), prompt_records


def resolve_split_entry(body_root: Path, voxel_dir: Path, split_entry: str | Path) -> Path:
    path = Path(split_entry)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return voxel_dir / path
    if path.exists():
        return path
    repo_relative = ROOT.parent / path
    if repo_relative.exists():
        return repo_relative
    return body_root / path


def resolve_input_path(body_root: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    repo_relative = ROOT.parent / path
    if repo_relative.exists():
        return repo_relative
    return body_root / path


def resolve_case_paths(
    args: argparse.Namespace,
    body_root: Path,
    data_cfg: Mapping[str, Any] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    data_cfg = data_cfg or {}
    voxel_dir = resolve_data_path(body_root, data_cfg.get("voxel_dir"), "Dataset/voxel_data")

    if args.input:
        paths.extend(resolve_input_path(body_root, path) for path in args.input)

    if args.case_id:
        for case_id in args.case_id:
            name = str(case_id)
            if not name.endswith(".npz"):
                name = f"{name}.npz"
            paths.append(voxel_dir / name)

    if args.split:
        split_path = resolve_data_path(
            body_root,
            data_cfg.get("split_path"),
            "Dataset/dataset_split.json",
        )
        split_data = read_json(split_path)
        paths.extend(resolve_split_entry(body_root, voxel_dir, name) for name in split_data[args.split])

    if not paths:
        raise ValueError("No input cases selected. Use --input, --case-id, or --split.")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)

    if args.limit is not None:
        deduped = deduped[: args.limit]
    missing = [str(path) for path in deduped if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input case files: {missing[:5]}")
    return deduped


def load_case(case_path: Path, volume_size: Sequence[int]) -> dict[str, Any]:
    with np.load(case_path) as data:
        if "voxel_labels" in data.files:
            labels = np.asarray(data["voxel_labels"], dtype=np.int64)
            original_shape = tuple(int(x) for x in labels.shape)
        else:
            labels = None
            original_shape = tuple(int(x) for x in np.asarray(data["grid_occ_size"]).tolist())

        required = ("sensor_pc", "grid_world_min", "grid_voxel_size")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{case_path} is missing required arrays: {missing}")

        occupancy = voxelize_sensor_points(
            np.asarray(data["sensor_pc"], dtype=np.float32),
            original_shape,
            np.asarray(data["grid_world_min"], dtype=np.float32),
            np.asarray(data["grid_voxel_size"], dtype=np.float32),
        )
        metadata = {
            key: np.asarray(data[key]).copy()
            for key in ("grid_world_min", "grid_world_max", "grid_voxel_size", "grid_occ_size")
            if key in data.files
        }

    occupancy_model = fit_array_to_shape(occupancy, volume_size, pad_value=False)
    labels_model = fit_array_to_shape(labels, volume_size, pad_value=0) if labels is not None else None
    return {
        "case_id": case_path.stem,
        "case_path": str(case_path),
        "original_shape": original_shape,
        "occupancy": torch.from_numpy(occupancy_model.astype(np.float32, copy=False))[None, None],
        "labels": labels,
        "labels_model_shape": labels_model,
        "metadata": metadata,
    }


def restore_to_original_shape(array: np.ndarray, original_shape: Sequence[int]) -> np.ndarray:
    """Invert ``fit_array_to_shape`` for arrays shaped ``(..., D, H, W)``."""

    original_shape = tuple(int(x) for x in original_shape)
    model_shape = tuple(int(x) for x in array.shape[-3:])
    output = np.zeros((*array.shape[:-3], *original_shape), dtype=array.dtype)

    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for original_size, model_size in zip(original_shape, model_shape):
        if original_size >= model_size:
            dst_start = (original_size - model_size) // 2
            src_slices.append(slice(0, model_size))
            dst_slices.append(slice(dst_start, dst_start + model_size))
        else:
            src_start = (model_size - original_size) // 2
            src_slices.append(slice(src_start, src_start + original_size))
            dst_slices.append(slice(0, original_size))

    output[(..., *dst_slices)] = array[(..., *src_slices)]
    return output


@torch.inference_mode()
def predict_masks(
    model: VoxTellBodyModel,
    occupancy: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    device: torch.device,
    threshold: float,
    prompt_batch_size: int,
    use_amp: bool,
    save_probs: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    masks: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []
    occupancy = occupancy.to(device, non_blocking=True)
    prompt_batch_size = max(1, int(prompt_batch_size))

    for start in range(0, prompt_embeddings.shape[0], prompt_batch_size):
        end = min(start + prompt_batch_size, prompt_embeddings.shape[0])
        text = prompt_embeddings[start:end][None].to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(occupancy, text)
            if isinstance(logits, list):
                logits = logits[0]
        probs = torch.sigmoid(logits.squeeze(0).float()).cpu()
        masks.append((probs.numpy() > threshold).astype(np.uint8))
        if save_probs:
            probs_out.append(probs.numpy().astype(np.float16))
        del text, logits, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    masks_array = np.concatenate(masks, axis=0)
    probs_array = np.concatenate(probs_out, axis=0) if save_probs else None
    return masks_array, probs_array


def target_class_ids(record: Mapping[str, Any], vocab: Mapping[str, Any]) -> list[int]:
    if record["source_type"] == "class" and record.get("class_id") is not None:
        return [int(record["class_id"])]
    if record["source_type"] == "aggregate":
        aggregate_id = str(record["aggregate_id"])
        for aggregate in vocab.get("aggregates", []):
            if str(aggregate["id"]) == aggregate_id:
                return [int(x) for x in aggregate.get("component_class_ids", [])]
    return []


def dice_score(mask: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    mask_bool = mask.astype(bool, copy=False)
    target_bool = target.astype(bool, copy=False)
    denominator = float(mask_bool.sum() + target_bool.sum())
    if denominator == 0.0:
        return 1.0
    intersection = float(np.logical_and(mask_bool, target_bool).sum())
    return (2.0 * intersection + eps) / (denominator + eps)


def combined_label_volume(masks: np.ndarray) -> np.ndarray:
    combined = np.zeros(masks.shape[1:], dtype=np.uint16)
    for index, mask in enumerate(masks):
        combined[mask.astype(bool, copy=False)] = index + 1
    return combined


def padded_target_class_ids(targets: Sequence[Sequence[int]]) -> np.ndarray:
    width = max((len(ids) for ids in targets), default=0)
    encoded = np.full((len(targets), width), -1, dtype=np.int64)
    for row, ids in enumerate(targets):
        if ids:
            encoded[row, : len(ids)] = [int(x) for x in ids]
    return encoded


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_case_outputs(
    args: argparse.Namespace,
    case: Mapping[str, Any],
    masks_model: np.ndarray,
    probs_model: np.ndarray | None,
    prompt_records: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    case_dir = args.output / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)

    masks_original = restore_to_original_shape(masks_model, case["original_shape"])
    prompt_target_ids = [target_class_ids(record, vocab) for record in prompt_records]
    payload: dict[str, Any] = {
        "pred_masks": masks_original.astype(np.uint8, copy=False),
        "prompt_texts": np.asarray([str(record["text"]) for record in prompt_records]),
        "prompt_ids": np.asarray([str(record["prompt_id"]) for record in prompt_records]),
        "prompt_class_ids": np.asarray(
            [int(record["class_id"]) if record.get("class_id") is not None else -1 for record in prompt_records],
            dtype=np.int64,
        ),
        "prompt_target_class_ids": padded_target_class_ids(prompt_target_ids),
        "prompt_cache_indices": np.asarray([int(record["index"]) for record in prompt_records], dtype=np.int64),
        "threshold": np.asarray(args.threshold, dtype=np.float32),
        "case_path": np.asarray(str(Path(case["case_path"]).resolve())),
        "original_shape": np.asarray(case["original_shape"], dtype=np.int64),
        "model_volume_size": np.asarray(masks_model.shape[-3:], dtype=np.int64),
    }
    payload.update(case["metadata"])
    if args.save_model_shape:
        payload["pred_masks_model_shape"] = masks_model.astype(np.uint8, copy=False)
    if probs_model is not None:
        payload["pred_probs_model_shape"] = probs_model
    if args.save_combined:
        payload["pred_combined"] = combined_label_volume(masks_original)

    npz_path = case_dir / f"{case['case_id']}_predictions.npz"
    np.savez_compressed(npz_path, **payload)

    prompt_summaries: list[dict[str, Any]] = []
    for index, record in enumerate(prompt_records):
        summary = {
            "output_label": index + 1,
            "prompt_id": str(record["prompt_id"]),
            "prompt_text": str(record["text"]),
            "source_type": str(record["source_type"]),
            "class_id": record.get("class_id"),
            "aggregate_id": record.get("aggregate_id"),
            "positive_voxels": int(masks_original[index].sum()),
        }
        ids = prompt_target_ids[index]
        if ids:
            summary["target_class_ids"] = ids
        if args.evaluate and case["labels"] is not None and ids:
            target = build_target_mask(case["labels"], ids)
            summary["dice"] = float(dice_score(masks_original[index], target))
            summary["target_voxels"] = int(target.sum())
        prompt_summaries.append(summary)

    eval_dices = [item["dice"] for item in prompt_summaries if "dice" in item]
    summary_payload = {
        "case_id": case["case_id"],
        "input": case["case_path"],
        "output_npz": str(npz_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "threshold": args.threshold,
        "original_shape": list(case["original_shape"]),
        "model_volume_size": list(masks_model.shape[-3:]),
        "num_prompts": len(prompt_records),
        "mean_dice": float(np.mean(eval_dices)) if eval_dices else None,
        "prompts": prompt_summaries,
    }
    with (case_dir / f"{case['case_id']}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2, default=json_ready)
        f.write("\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    if not args.ignore_checkpoint_config and checkpoint.get("config"):
        cfg = checkpoint["config"]

    body_root = resolve_body_root(cfg["data"]["root"])
    volume_size = tuple(int(x) for x in cfg["data"]["volume_size"])
    data_cfg = cfg.get("data", {})
    vocab_path = args.vocab or resolve_data_path(
        body_root,
        data_cfg.get("vocab_path"),
        "configs/label_vocab.json",
    )
    cache_path = args.embedding_cache or resolve_data_path(
        body_root,
        data_cfg.get("embedding_cache_path"),
        "artifacts/text_embeddings/prompt_embeddings.pt",
    )
    vocab = load_label_vocab(vocab_path)
    cache_records = flatten_prompt_records(vocab)
    selected_records, missing_texts = select_prompt_records(args, vocab, cache_records)
    prompt_embeddings, prompt_records = load_prompt_embeddings(args, cache_path, selected_records, missing_texts)

    device = resolve_device(args.device, args.gpu)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    case_paths = resolve_case_paths(args, body_root, data_cfg)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'N/A')}")
    print(f"body_root={body_root}")
    print(f"volume_size={volume_size}")
    print(f"prompts={len(prompt_records)} cases={len(case_paths)}")

    model = load_model(checkpoint, cfg, device)

    run_summaries: list[dict[str, Any]] = []
    for case_path in tqdm(case_paths, desc="Body-Tell inference"):
        case = load_case(case_path, volume_size)
        masks_model, probs_model = predict_masks(
            model=model,
            occupancy=case["occupancy"],
            prompt_embeddings=prompt_embeddings,
            device=device,
            threshold=args.threshold,
            prompt_batch_size=args.prompt_batch_size,
            use_amp=args.amp,
            save_probs=args.save_probs,
        )
        save_case_outputs(
            args=args,
            case=case,
            masks_model=masks_model,
            probs_model=probs_model,
            prompt_records=prompt_records,
            vocab=vocab,
            checkpoint=checkpoint,
        )
        run_summaries.append(
            {
                "case_id": case["case_id"],
                "output_dir": str(args.output / str(case["case_id"])),
                "positive_voxels": [int(mask.sum()) for mask in masks_model],
            }
        )

    with (args.output / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "config": str(args.config),
                "body_root": str(body_root),
                "threshold": args.threshold,
                "num_cases": len(case_paths),
                "num_prompts": len(prompt_records),
                "prompts": [dict(record) for record in prompt_records],
                "cases": run_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=json_ready,
        )
        f.write("\n")

    print(f"wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
