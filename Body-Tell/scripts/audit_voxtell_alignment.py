#!/usr/bin/env python3
"""SP-A.P0 compatibility audit for Body-Tell and VoxTell v1.1."""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BODY_TELL_ROOT = Path(__file__).resolve().parents[1]
VOXTELL_ROOT = REPO_ROOT / "VoxTell"
DEFAULT_BODY_CONFIG = BODY_TELL_ROOT / "configs" / "phase1_voxtell_body.yaml"
DEFAULT_VOXTELL_DIR = VOXTELL_ROOT / "models-weight" / "voxtell" / "voxtell_v1.1"
PROJECT_TO_DECODER_RE = re.compile(r"^project_to_decoder_channels\.(\d+)\.2\.weight$")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


def _prefix(name: str) -> str:
    return name.split(".", 1)[0]


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for item in shape:
        total *= int(item)
    return total


def load_body_config(config_path: Path) -> Any:
    import yaml

    if str(BODY_TELL_ROOT) not in sys.path:
        sys.path.insert(0, str(BODY_TELL_ROOT))
    from body_tell.models.voxtell_body_model import VoxTellBodyConfig

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_raw = dict(raw.get("model", {}))
    model_raw.pop("name", None)
    cfg = asdict(VoxTellBodyConfig())
    cfg.update(model_raw)
    cfg["encoder_channels"] = tuple(int(x) for x in cfg["encoder_channels"])
    return VoxTellBodyConfig(**cfg)


def body_projection_rows(config: Any) -> list[dict[str, Any]]:
    fused_stage_count = min(
        int(config.num_maskformer_stages),
        len(config.encoder_channels) - 1,
    )
    rows = []
    for stage_index, channels in enumerate(config.encoder_channels[:fused_stage_count]):
        output_dim = int(channels) if stage_index == 0 else int(channels) * int(config.num_heads)
        rows.append(
            {
                "stage": stage_index,
                "channels": int(channels),
                "output_dim": output_dim,
                "weight_shape": (output_dim, int(config.text_projection_hidden_dim)),
                "bias_shape": (output_dim,),
                "key": f"project_to_decoder_channels.{stage_index}.2",
                "target_key": f"project_to_decoder_channels.{stage_index}.2",
            }
        )
    return rows


def load_body_state_shapes(config: Any) -> dict[str, tuple[int, ...]]:
    import torch

    if str(BODY_TELL_ROOT) not in sys.path:
        sys.path.insert(0, str(BODY_TELL_ROOT))
    from body_tell.models.voxtell_body_model import VoxTellBodyModel

    with torch.device("meta"):
        model = VoxTellBodyModel(config)
    return {name: _shape(tensor) for name, tensor in model.state_dict().items()}


def parse_voxtell_model_source(source_path: Path) -> dict[str, Any]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    model_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VoxTellModel"
    )
    decoder_configs: dict[int, dict[str, Any]] = {}
    defaults: dict[str, Any] = {}

    for node in model_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DECODER_CONFIGS":
                    decoder_configs = _literal(node.value) or {}
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            positional_args = [arg.arg for arg in node.args.args]
            default_values = list(node.args.defaults)
            default_names = positional_args[-len(default_values) :] if default_values else []
            defaults = {
                name: _literal(default)
                for name, default in zip(default_names, default_values)
            }
            break

    return {
        "decoder_configs": decoder_configs,
        "defaults": defaults,
    }


def parse_voxtell_predictor_overrides(predictor_path: Path) -> dict[str, Any]:
    tree = ast.parse(predictor_path.read_text(encoding="utf-8"))
    overrides: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_model_call = (
            isinstance(func, ast.Name)
            and func.id == "VoxTellModel"
            or isinstance(func, ast.Attribute)
            and func.attr == "VoxTellModel"
        )
        if not is_model_call:
            continue
        for keyword in node.keywords:
            if keyword.arg is not None:
                value = _literal(keyword.value)
                if value is not None:
                    overrides[keyword.arg] = value
        break
    return overrides


def voxtell_projection_rows(
    decoder_configs: dict[int, dict[str, Any]],
    num_maskformer_stages: int,
    num_heads: int,
    hidden_dim: int,
) -> list[dict[str, Any]]:
    rows = []
    for stage_index, config in list(sorted(decoder_configs.items()))[:num_maskformer_stages]:
        channels = int(config["channels"])
        output_dim = channels if int(stage_index) == 0 else channels * int(num_heads)
        rows.append(
            {
                "stage": int(stage_index),
                "channels": channels,
                "output_dim": output_dim,
                "weight_shape": (output_dim, int(hidden_dim)),
                "bias_shape": (output_dim,),
                "key": f"project_to_decoder_channels.{stage_index}.2",
            }
        )
    return rows


def load_checkpoint_shapes(voxtell_dir: Path) -> dict[str, Any]:
    import torch

    checkpoint_path = voxtell_dir / "fold_0" / "checkpoint_final.pth"
    if not checkpoint_path.exists():
        return {
            "available": False,
            "checkpoint_path": str(checkpoint_path),
            "error": "checkpoint file does not exist",
        }

    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )

    state_dict = checkpoint.get("network_weights", checkpoint)
    shapes = {name: _shape(tensor) for name, tensor in state_dict.items()}
    rows = []
    for name, shape in sorted(shapes.items()):
        match = PROJECT_TO_DECODER_RE.match(name)
        if not match:
            continue
        stage = int(match.group(1))
        rows.append(
            {
                "stage": stage,
                "output_dim": int(shape[0]),
                "weight_shape": shape,
                "bias_shape": shapes.get(f"project_to_decoder_channels.{stage}.2.bias"),
                "key": f"project_to_decoder_channels.{stage}.2",
            }
        )

    return {
        "available": True,
        "checkpoint_path": str(checkpoint_path),
        "key_count": len(shapes),
        "state_shapes": shapes,
        "projection_rows": sorted(rows, key=lambda row: row["stage"]),
        "pos_embed_shape": shapes.get("pos_embed"),
    }


def infer_num_heads_from_projection_rows(
    rows: list[dict[str, Any]],
    decoder_configs: dict[int, dict[str, Any]],
) -> int | None:
    ratios = []
    for row in rows:
        stage = int(row["stage"])
        if stage == 0:
            continue
        channels = int(decoder_configs[stage]["channels"])
        output_dim = int(row["output_dim"])
        if output_dim % channels != 0:
            return None
        ratios.append(output_dim // channels)
    if not ratios:
        return None
    first = ratios[0]
    return first if all(item == first for item in ratios) else None


def compare_state_shapes(
    checkpoint_shapes: dict[str, tuple[int, ...]] | None,
    body_shapes: dict[str, tuple[int, ...]] | None,
) -> dict[str, Any]:
    if checkpoint_shapes is None or body_shapes is None:
        return {"available": False}

    matching = []
    mismatched = []
    for name, body_shape in sorted(body_shapes.items()):
        checkpoint_shape = checkpoint_shapes.get(name)
        if checkpoint_shape is None:
            continue
        if checkpoint_shape == body_shape:
            matching.append(name)
        else:
            mismatched.append(
                {
                    "name": name,
                    "body_shape": body_shape,
                    "checkpoint_shape": checkpoint_shape,
                }
            )

    missing_in_checkpoint = sorted(set(body_shapes) - set(checkpoint_shapes))
    unexpected_in_checkpoint = sorted(set(checkpoint_shapes) - set(body_shapes))
    prefix_rows = []
    for prefix in sorted({_prefix(name) for name in set(body_shapes) | set(checkpoint_shapes)}):
        body_keys = {name for name in body_shapes if _prefix(name) == prefix}
        checkpoint_keys = {name for name in checkpoint_shapes if _prefix(name) == prefix}
        matching_keys = {name for name in matching if _prefix(name) == prefix}
        prefix_rows.append(
            {
                "prefix": prefix,
                "body_keys": len(body_keys),
                "checkpoint_keys": len(checkpoint_keys),
                "shape_compatible_same_name": len(matching_keys),
            }
        )

    matched_param_count = sum(_numel(body_shapes[name]) for name in matching)
    body_param_count = sum(_numel(shape) for shape in body_shapes.values())

    return {
        "available": True,
        "matching_count": len(matching),
        "mismatch_count": len(mismatched),
        "missing_in_checkpoint_count": len(missing_in_checkpoint),
        "unexpected_in_checkpoint_count": len(unexpected_in_checkpoint),
        "matching": matching,
        "mismatched": mismatched,
        "missing_in_checkpoint": missing_in_checkpoint,
        "unexpected_in_checkpoint": unexpected_in_checkpoint,
        "prefix_rows": prefix_rows,
        "matched_param_count": matched_param_count,
        "body_param_count": body_param_count,
    }


def audit_transformer_decoder_prefix(
    checkpoint_shapes: dict[str, tuple[int, ...]] | None,
    body_shapes: dict[str, tuple[int, ...]] | None,
    prefix: str = "transformer_decoder.",
) -> dict[str, Any]:
    if checkpoint_shapes is None or body_shapes is None:
        return {
            "available": False,
            "prefix": prefix,
            "status": "unavailable",
        }

    source_shapes = {
        name[len(prefix) :]: shape
        for name, shape in checkpoint_shapes.items()
        if name.startswith(prefix)
    }
    target_shapes = {
        name[len(prefix) :]: shape
        for name, shape in body_shapes.items()
        if name.startswith(prefix)
    }
    missing = sorted(set(target_shapes) - set(source_shapes))
    unexpected = sorted(set(source_shapes) - set(target_shapes))
    shape_mismatches = [
        {
            "key": name,
            "checkpoint_shape": source_shapes[name],
            "model_shape": target_shapes[name],
        }
        for name in sorted(set(source_shapes) & set(target_shapes))
        if source_shapes[name] != target_shapes[name]
    ]
    loaded_parameter_count = sum(_numel(shape) for shape in source_shapes.values())
    passed = bool(source_shapes) and not missing and not unexpected and not shape_mismatches
    return {
        "available": True,
        "prefix": prefix,
        "status": "pass" if passed else "fail",
        "loaded_tensor_count": len(source_shapes),
        "loaded_parameter_count": loaded_parameter_count,
        "target_tensor_count": len(target_shapes),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "excluded_prefixes": [
            "encoder.",
            "decoder.",
            "project_bottleneck_embed.",
            "project_text_embed.",
            "project_to_decoder_channels.",
            "pos_embed",
        ],
    }


def build_whitelist_input(
    *,
    pos_embed_policy: str,
    body_num_heads: int,
    source_default_num_heads: int,
    checkpoint_num_heads: int | None,
) -> dict[str, Any]:
    effective_checkpoint_heads = checkpoint_num_heads
    exact_skips = []
    conditional_skips = []
    no_skip_notes = []

    if pos_embed_policy == "dynamic_hwd":
        exact_skips.append(
            {
                "pattern": "pos_embed",
                "reason": "Body-Tell current non-192^3 crop uses dynamic positional encoding.",
            }
        )

    if effective_checkpoint_heads is not None and effective_checkpoint_heads != body_num_heads:
        conditional_skips.extend(
            [
                {
                    "pattern": "project_to_decoder_channels.[1-4].*",
                    "reason": "projection output channels scale with num_heads for stages 1-4",
                },
                {
                    "pattern": "decoder.transpconvs.[1-4].*",
                    "reason": "intermediate decoder inputs include num_heads fused mask channels",
                },
                {
                    "pattern": "decoder.seg_layers.*",
                    "reason": "segmentation heads take input_features_skip + num_heads channels",
                },
            ]
        )
    elif effective_checkpoint_heads is not None:
        no_skip_notes.append(
            "checkpoint-derived num_heads matches Body-Tell current num_heads; "
            "do not skip project_to_decoder_channels.* or decoder mask-fusion keys for this reason"
        )

    if source_default_num_heads != body_num_heads:
        conditional_skips.append(
            {
                "pattern": "project_to_decoder_channels.[1-4].*, decoder.transpconvs.[1-4].*, decoder.seg_layers.*",
                "reason": (
                    "only apply if loading a checkpoint generated from the VoxTellModel source "
                    "default num_heads=1 into a Body-Tell num_heads=32 model"
                ),
            }
        )

    return {
        "pos_embed_policy": pos_embed_policy,
        "body_num_heads": body_num_heads,
        "voxtell_source_default_num_heads": source_default_num_heads,
        "voxtell_checkpoint_num_heads": checkpoint_num_heads,
        "required_exact_skip_patterns": exact_skips,
        "conditional_skip_patterns": conditional_skips,
        "no_skip_notes": no_skip_notes,
    }


def make_audit(args: argparse.Namespace) -> dict[str, Any]:
    body_config = load_body_config(args.config)
    body_rows = body_projection_rows(body_config)
    source_info = parse_voxtell_model_source(VOXTELL_ROOT / "voxtell" / "model" / "voxtell_model.py")
    predictor_overrides = parse_voxtell_predictor_overrides(
        VOXTELL_ROOT / "voxtell" / "inference" / "predictor.py"
    )

    defaults = source_info["defaults"]
    decoder_configs = source_info["decoder_configs"]
    source_default_num_heads = int(defaults["num_heads"])
    source_default_hidden_dim = int(defaults["project_to_decoder_hidden_dim"])
    source_default_num_stages = int(defaults["num_maskformer_stages"])
    source_rows = voxtell_projection_rows(
        decoder_configs,
        source_default_num_stages,
        source_default_num_heads,
        source_default_hidden_dim,
    )

    predictor_num_heads = int(predictor_overrides.get("num_heads", source_default_num_heads))
    predictor_hidden_dim = int(
        predictor_overrides.get("project_to_decoder_hidden_dim", source_default_hidden_dim)
    )
    predictor_num_stages = int(
        predictor_overrides.get("num_maskformer_stages", source_default_num_stages)
    )
    predictor_rows = voxtell_projection_rows(
        decoder_configs,
        predictor_num_stages,
        predictor_num_heads,
        predictor_hidden_dim,
    )

    checkpoint = None if args.skip_checkpoint else load_checkpoint_shapes(args.voxtell_dir)
    checkpoint_num_heads = None
    if checkpoint and checkpoint["available"]:
        checkpoint_num_heads = infer_num_heads_from_projection_rows(
            checkpoint["projection_rows"],
            decoder_configs,
        )
        for row in checkpoint["projection_rows"]:
            stage = row["stage"]
            row["channels"] = int(decoder_configs[stage]["channels"])

    body_shapes = None if args.skip_state_dict_compare else load_body_state_shapes(body_config)
    checkpoint_shapes = (
        checkpoint.get("state_shapes")
        if checkpoint is not None and checkpoint.get("available")
        else None
    )
    state_compare = compare_state_shapes(checkpoint_shapes, body_shapes)
    transformer_decoder_prefix = audit_transformer_decoder_prefix(
        checkpoint_shapes,
        body_shapes,
    )

    whitelist_input = build_whitelist_input(
        pos_embed_policy=args.pos_embed_policy,
        body_num_heads=int(body_config.num_heads),
        source_default_num_heads=source_default_num_heads,
        checkpoint_num_heads=checkpoint_num_heads,
    )

    return {
        "body": {
            "config_path": str(args.config),
            "num_heads": int(body_config.num_heads),
            "num_maskformer_stages_config": int(body_config.num_maskformer_stages),
            "fused_stage_count_actual": len(body_rows),
            "encoder_channels": list(body_config.encoder_channels),
            "projection_prefix": "project_to_decoder_channels",
            "target_projection_prefix": "project_to_decoder_channels",
            "projection_rows": body_rows,
        },
        "voxtell": {
            "source_default": {
                "num_heads": source_default_num_heads,
                "hidden_dim": source_default_hidden_dim,
                "projection_rows": source_rows,
            },
            "predictor_v1_1": {
                "num_heads": predictor_num_heads,
                "hidden_dim": predictor_hidden_dim,
                "projection_rows": predictor_rows,
                "overrides": predictor_overrides,
            },
            "checkpoint": checkpoint,
            "checkpoint_num_heads": checkpoint_num_heads,
        },
        "state_compare_current_body_vs_checkpoint": state_compare,
        "transformer_decoder_prefix_audit": transformer_decoder_prefix,
        "whitelist_input": whitelist_input,
    }


def fmt_shape(shape: Any) -> str:
    if shape is None:
        return "n/a"
    return "(" + ", ".join(str(int(x)) for x in shape) + ")"


def projection_line(row: dict[str, Any]) -> str:
    return (
        f"  stage {row['stage']}: channels={row.get('channels', 'n/a')} "
        f"output_dim={row['output_dim']} weight={fmt_shape(row['weight_shape'])} "
        f"key={row['key']}"
    )


def print_text_report(audit: dict[str, Any]) -> None:
    body = audit["body"]
    voxtell = audit["voxtell"]
    print("SP-A.P0 VoxTell compatibility baseline")
    print()
    print("Body-Tell current")
    print(f"  config: {body['config_path']}")
    print(f"  num_heads: {body['num_heads']}")
    print(
        "  fused projection stages: "
        f"{body['fused_stage_count_actual']} "
        f"(configured {body['num_maskformer_stages_config']})"
    )
    for row in body["projection_rows"]:
        print(projection_line(row))
    print()
    print("VoxTell v1.1 source default")
    print(f"  num_heads: {voxtell['source_default']['num_heads']}")
    for row in voxtell["source_default"]["projection_rows"]:
        print(projection_line(row))
    print()
    print("VoxTell v1.1 predictor/checkpoint profile")
    print(f"  predictor num_heads: {voxtell['predictor_v1_1']['num_heads']}")
    if voxtell["checkpoint"] and voxtell["checkpoint"].get("available"):
        print(f"  checkpoint: {voxtell['checkpoint']['checkpoint_path']}")
        print(f"  checkpoint keys: {voxtell['checkpoint']['key_count']}")
        print(f"  checkpoint pos_embed: {fmt_shape(voxtell['checkpoint']['pos_embed_shape'])}")
        print(f"  checkpoint inferred num_heads: {voxtell['checkpoint_num_heads']}")
        for row in voxtell["checkpoint"]["projection_rows"]:
            print(projection_line(row))
    elif voxtell["checkpoint"]:
        print(f"  checkpoint unavailable: {voxtell['checkpoint']['error']}")
    else:
        print("  checkpoint inspection skipped")
        for row in voxtell["predictor_v1_1"]["projection_rows"]:
            print(projection_line(row))
    print()
    print("P6 whitelist input")
    for item in audit["whitelist_input"]["required_exact_skip_patterns"]:
        print(f"  required skip: {item['pattern']} ({item['reason']})")
    for item in audit["whitelist_input"]["conditional_skip_patterns"]:
        print(f"  conditional skip: {item['pattern']} ({item['reason']})")
    for note in audit["whitelist_input"]["no_skip_notes"]:
        print(f"  note: {note}")
    compare = audit["state_compare_current_body_vs_checkpoint"]
    if compare["available"]:
        print()
        print("Current Body-Tell same-name checkpoint shape coverage")
        print(f"  matching keys: {compare['matching_count']}")
        print(f"  shape mismatches: {compare['mismatch_count']}")
        print(f"  missing in checkpoint: {compare['missing_in_checkpoint_count']}")
        print(f"  unexpected in checkpoint: {compare['unexpected_in_checkpoint_count']}")
        print(f"  matched params: {compare['matched_param_count']:,} / {compare['body_param_count']:,}")
    prefix_audit = audit["transformer_decoder_prefix_audit"]
    print()
    print("Transformer decoder prefix audit")
    print(f"  status: {prefix_audit['status'].upper()}")
    print(f"  prefix: {prefix_audit['prefix']}")
    if prefix_audit["available"]:
        print(f"  loaded tensors: {prefix_audit['loaded_tensor_count']}")
        print(f"  loaded params: {prefix_audit['loaded_parameter_count']:,}")
        print(f"  missing: {len(prefix_audit['missing_keys'])}")
        print(f"  unexpected: {len(prefix_audit['unexpected_keys'])}")
        print(f"  shape mismatches: {len(prefix_audit['shape_mismatches'])}")


def table_row(cells: list[Any], code_first: bool = False) -> str:
    rendered = []
    for index, cell in enumerate(cells):
        value = html.escape(str(cell))
        if code_first and index == 0:
            value = f"<code>{value}</code>"
        rendered.append(f"<td>{value}</td>")
    return "<tr>" + "".join(rendered) + "</tr>"


def projection_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        table_row(
            [
                row["stage"],
                row.get("channels", "n/a"),
                row["output_dim"],
                fmt_shape(row["weight_shape"]),
                row["key"],
            ]
        )
        for row in rows
    )
    return (
        "<table><thead><tr>"
        "<th>Stage</th><th>Channels</th><th>Output dim</th><th>Final linear weight</th><th>Key</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
    )


def render_html_report(audit: dict[str, Any]) -> str:
    body = audit["body"]
    voxtell = audit["voxtell"]
    compare = audit["state_compare_current_body_vs_checkpoint"]
    prefix_audit = audit["transformer_decoder_prefix_audit"]
    whitelist = audit["whitelist_input"]
    checkpoint = voxtell["checkpoint"]
    checkpoint_rows = (
        checkpoint["projection_rows"]
        if checkpoint and checkpoint.get("available")
        else voxtell["predictor_v1_1"]["projection_rows"]
    )
    checkpoint_label = "checkpoint-derived" if checkpoint and checkpoint.get("available") else "predictor-derived"

    prefix_table = ""
    if compare["available"]:
        prefix_rows = "\n".join(
            table_row(
                [
                    row["prefix"],
                    row["body_keys"],
                    row["checkpoint_keys"],
                    row["shape_compatible_same_name"],
                ],
                code_first=True,
            )
            for row in compare["prefix_rows"]
        )
        prefix_table = (
            "<table><thead><tr><th>Prefix</th><th>Body keys</th><th>Checkpoint keys</th>"
            "<th>Same-name shape-compatible</th></tr></thead><tbody>"
            f"{prefix_rows}</tbody></table>"
        )

    required_skip_rows = "\n".join(
        table_row([item["pattern"], item["reason"]], code_first=True)
        for item in whitelist["required_exact_skip_patterns"]
    )
    conditional_skip_rows = "\n".join(
        table_row([item["pattern"], item["reason"]], code_first=True)
        for item in whitelist["conditional_skip_patterns"]
    )
    no_skip_notes = "".join(
        f"<li>{html.escape(note)}</li>" for note in whitelist["no_skip_notes"]
    )
    prefix_status_class = "ok" if prefix_audit["status"] == "pass" else "warn"
    prefix_shape_rows = "\n".join(
        table_row(
            [
                item["key"],
                fmt_shape(item["checkpoint_shape"]),
                fmt_shape(item["model_shape"]),
            ],
            code_first=True,
        )
        for item in prefix_audit.get("shape_mismatches", [])
    )
    if not prefix_shape_rows:
        prefix_shape_rows = "<tr><td colspan=\"3\">None</td></tr>"

    embedded_json = html.escape(json.dumps(audit, indent=2, sort_keys=True, default=str))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SP-A.P0 VoxTell Alignment Audit</title>
  <style>
    body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17202a; background: #f6f8fa; }}
    main {{ max-width: 1160px; margin: 0 auto; padding: 32px 24px 56px; }}
    section {{ background: #fff; border: 1px solid #d8e0e7; border-radius: 8px; padding: 18px 20px; margin: 16px 0; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #d8e0e7; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f7; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    code {{ background: #edf3f7; border-radius: 4px; padding: 1px 4px; }}
    pre {{ overflow-x: auto; background: #111827; color: #f8fafc; border-radius: 8px; padding: 14px; }}
    .meta {{ color: #5d6875; }}
    .warn {{ border-left: 4px solid #9a4a17; background: #fff7ed; padding: 10px 12px; }}
    .ok {{ border-left: 4px solid #276749; background: #eef9f0; padding: 10px 12px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>SP-A.P0 VoxTell Compatibility Baseline Audit</h1>
    <p class="meta">Body config: <code>{html.escape(body['config_path'])}</code>; VoxTell profile: <code>{html.escape(str(DEFAULT_VOXTELL_DIR))}</code>.</p>
  </header>

  <section>
    <h2>Projection Shape Baseline</h2>
    <p><strong>Body-Tell current</strong>: <code>num_heads={body['num_heads']}</code>, actual fused projection stages <code>{body['fused_stage_count_actual']}</code> from configured <code>{body['num_maskformer_stages_config']}</code>; prefix is <code>project_to_decoder_channels</code>.</p>
    {projection_table(body['projection_rows'])}
    <p><strong>VoxTell source default</strong>: <code>num_heads={voxtell['source_default']['num_heads']}</code>. This is the class default in <code>voxtell_model.py</code>, not necessarily the shipped v1.1 checkpoint profile.</p>
    {projection_table(voxtell['source_default']['projection_rows'])}
    <p><strong>VoxTell v1.1 {checkpoint_label}</strong>: predictor override <code>num_heads={voxtell['predictor_v1_1']['num_heads']}</code>; checkpoint inferred <code>num_heads={voxtell['checkpoint_num_heads']}</code>; pos_embed shape <code>{fmt_shape(checkpoint.get('pos_embed_shape') if checkpoint else None)}</code>.</p>
    {projection_table(checkpoint_rows)}
    <p class="warn">Audit finding: the model class default is <code>num_heads=1</code>, but the v1.1 predictor and checkpoint metadata use <code>num_heads={voxtell['checkpoint_num_heads']}</code>. P4/P6 should key off the actual checkpoint being loaded.</p>
  </section>

  <section>
    <h2>Weight-Key Audit Input</h2>
    <p>Current Body-Tell can only directly reuse same-name/same-shape subsets before SP-A architecture work. This table is a baseline input for P6, not a claim that current Body-Tell is already checkpoint-compatible.</p>
    {prefix_table}
    <p>Same-name compatible current Body-Tell keys: <code>{compare.get('matching_count', 'n/a')}</code>; shape mismatches: <code>{compare.get('mismatch_count', 'n/a')}</code>; checkpoint keys unexpected to current Body-Tell: <code>{compare.get('unexpected_in_checkpoint_count', 'n/a')}</code>.</p>
  </section>

  <section>
    <h2>Transformer Decoder Prefix Audit</h2>
    <p class="{prefix_status_class}"><strong>Status:</strong> <code>{html.escape(prefix_audit['status'])}</code>; prefix <code>{html.escape(prefix_audit['prefix'])}</code>; loaded tensors <code>{prefix_audit.get('loaded_tensor_count', 'n/a')}</code>; loaded params <code>{prefix_audit.get('loaded_parameter_count', 'n/a')}</code>.</p>
    <p>Missing keys: <code>{len(prefix_audit.get('missing_keys', []))}</code>; unexpected keys: <code>{len(prefix_audit.get('unexpected_keys', []))}</code>; shape mismatches: <code>{len(prefix_audit.get('shape_mismatches', []))}</code>. The audited load target is only <code>model.transformer_decoder.state_dict()</code>; excluded prefixes include <code>encoder.</code>, <code>decoder.</code>, <code>project_bottleneck_embed.</code>, <code>project_text_embed.</code>, <code>project_to_decoder_channels.</code>, and <code>pos_embed</code>.</p>
    <table><thead><tr><th>Key</th><th>Checkpoint shape</th><th>Model shape</th></tr></thead><tbody>{prefix_shape_rows}</tbody></table>
  </section>

  <section>
    <h2>P6 Whitelist Input</h2>
    <h3>Required exact skips</h3>
    <table><thead><tr><th>Pattern</th><th>Reason</th></tr></thead><tbody>{required_skip_rows}</tbody></table>
    <h3>Conditional skips</h3>
    <table><thead><tr><th>Pattern</th><th>Reason</th></tr></thead><tbody>{conditional_skip_rows}</tbody></table>
    <ul>{no_skip_notes}</ul>
  </section>

  <section>
    <h2>Machine-Readable Audit Payload</h2>
    <pre>{embedded_json}</pre>
  </section>
</main>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Body-Tell/VoxTell v1.1 num_heads, projection shapes, "
            "current same-name state-dict coverage, and P6 whitelist input."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BODY_CONFIG,
        help="Body-Tell YAML config to audit.",
    )
    parser.add_argument(
        "--voxtell-dir",
        type=Path,
        default=DEFAULT_VOXTELL_DIR,
        help="VoxTell v1.1 model directory containing plans.json and fold_0/checkpoint_final.pth.",
    )
    parser.add_argument(
        "--pos-embed-policy",
        choices=("dynamic_hwd", "fixed_voxtell_192"),
        default="dynamic_hwd",
        help="Position embedding loading policy to reflect in the P6 whitelist input.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip checkpoint metadata inspection and use source/predictor-derived shapes only.",
    )
    parser.add_argument(
        "--skip-state-dict-compare",
        action="store_true",
        help="Skip current Body-Tell meta-device state-dict shape comparison.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        help="Optional HTML report path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = make_audit(args)
    print_text_report(audit)
    if args.html_output:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_html_report(audit), encoding="utf-8")
        print()
        print(f"Wrote HTML report: {args.html_output}")
    prefix_audit = audit["transformer_decoder_prefix_audit"]
    if prefix_audit["available"] and prefix_audit["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
