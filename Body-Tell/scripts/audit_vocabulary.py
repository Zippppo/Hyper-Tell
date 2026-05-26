#!/usr/bin/env python3
"""Generate a readable HTML audit report for Phase 0 vocabulary artifacts."""

from __future__ import annotations

import argparse
import html
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.vocabulary import (  # noqa: E402
    file_sha256,
    flatten_prompt_records,
    load_class_presence,
    load_label_vocab,
    read_json,
    validate_label_vocab,
)


def esc(value: Any) -> str:
    return html.escape(str(value))


def load_embedding_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = cache.get("embeddings")
    return {
        "exists": True,
        "model_name": cache.get("model_name"),
        "embedding_dim": cache.get("embedding_dim"),
        "vocab_version": cache.get("vocab_version"),
        "vocab_hash": cache.get("vocab_hash"),
        "num_prompts": cache.get("num_prompts"),
        "is_qwen_cache": cache.get("is_qwen_cache"),
        "fallback_error": cache.get("fallback_error"),
        "embedding_shape": list(embeddings.shape) if embeddings is not None else None,
    }


def duplicate_prompts(vocab: Mapping[str, Any]) -> Dict[str, List[str]]:
    owners: Dict[str, List[str]] = defaultdict(list)
    for record in flatten_prompt_records(vocab):
        owner = (
            f"class {record['class_id']}"
            if record["source_type"] == "class"
            else f"aggregate {record['aggregate_id']}"
        )
        owners[record["text"].casefold()].append(owner)
    return {prompt: values for prompt, values in owners.items() if len(values) > 1}


def render(args: argparse.Namespace) -> str:
    vocab = load_label_vocab(args.vocab)
    dataset_info = read_json(args.dataset_info)
    validation = validate_label_vocab(vocab, dataset_info=dataset_info, strict=False)
    presence = load_class_presence(args.presence) if args.presence.exists() else None
    embedding_meta = load_embedding_meta(args.embedding_cache)
    prompt_records = flatten_prompt_records(vocab)
    duplicates = duplicate_prompts(vocab)
    vocab_hash = file_sha256(args.vocab)

    rows = []
    for cls in vocab["classes"]:
        stats = presence["classes"].get(str(cls["id"]), {}) if presence else {}
        flags = []
        if not cls.get("train_as_positive", False):
            flags.append("not train positive")
        if stats.get("is_rare"):
            flags.append("rare")
        if stats.get("is_small_structure"):
            flags.append("small")
        rows.append(
            "<tr>"
            f"<td>{cls['id']}</td>"
            f"<td><code>{esc(cls['source_name'])}</code></td>"
            f"<td>{esc(cls['canonical'])}</td>"
            f"<td>{esc(', '.join(cls['prompts']))}</td>"
            f"<td>{esc(stats.get('case_count', 'n/a'))}</td>"
            f"<td>{esc(stats.get('median_voxels_per_present_case', 'n/a'))}</td>"
            f"<td>{esc(', '.join(flags) if flags else 'ok')}</td>"
            "</tr>"
        )

    aggregate_rows = []
    for aggregate in vocab.get("aggregates", []):
        aggregate_rows.append(
            "<tr>"
            f"<td><code>{esc(aggregate['id'])}</code></td>"
            f"<td>{esc(aggregate['canonical'])}</td>"
            f"<td>{esc(', '.join(aggregate['prompts']))}</td>"
            f"<td>{esc(aggregate['component_class_ids'])}</td>"
            f"<td>{esc(aggregate.get('notes', ''))}</td>"
            "</tr>"
        )

    warning_items = validation["warnings"] + [
        f"duplicate prompt {prompt!r}: {owners}" for prompt, owners in duplicates.items()
    ]
    if embedding_meta.get("exists") and not embedding_meta.get("is_qwen_cache"):
        warning_items.append("Embedding cache is deterministic fallback, not Qwen last-token embeddings.")
    if embedding_meta.get("exists") and embedding_meta.get("vocab_hash") != vocab_hash:
        warning_items.append("Embedding cache vocab_hash does not match current label_vocab.json.")
    if not warning_items:
        warning_items = ["No static warnings."]

    shape_summary = presence.get("shape_summary", {}) if presence else {}
    split_summary = presence.get("split_summary", {}) if presence else {}

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Body-Tell Vocabulary Audit</title>
  <style>
    body {{ margin: 0; background: #f7f9fc; color: #172033; font: 14px/1.55 Inter, system-ui, sans-serif; }}
    main {{ width: min(1180px, calc(100% - 36px)); margin: 0 auto; padding: 34px 0 56px; }}
    h1 {{ margin: 0 0 6px; font-size: 32px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    section {{ background: white; border: 1px solid #d8e1ea; border-radius: 10px; padding: 18px 20px; margin: 14px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e0e7ef; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef4f8; }}
    code {{ background: #edf4fb; border: 1px solid #c9d7e6; border-radius: 5px; padding: 1px 5px; }}
    .meta {{ color: #5f6d7e; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .panel {{ background: #eef4f8; border-radius: 8px; padding: 12px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Body-Tell Vocabulary Audit</h1>
  <p class="meta">Vocabulary version {esc(vocab.get('version'))} · hash <code>{esc(vocab_hash)}</code></p>

  <section>
    <h2>Artifact Summary</h2>
    <div class="grid">
      <div class="panel">Classes<br><strong>{len(vocab.get('classes', []))}</strong></div>
      <div class="panel">Aggregates<br><strong>{len(vocab.get('aggregates', []))}</strong></div>
      <div class="panel">Prompt records<br><strong>{len(prompt_records)}</strong></div>
      <div class="panel">Scanned cases<br><strong>{esc(presence.get('num_cases', 'missing') if presence else 'missing')}</strong></div>
      <div class="panel">Recommended volume size<br><strong>{esc(shape_summary.get('recommended_volume_size', 'missing'))}</strong></div>
      <div class="panel">Embedding shape<br><strong>{esc(embedding_meta.get('embedding_shape', 'missing'))}</strong></div>
    </div>
  </section>

  <section>
    <h2>Warnings</h2>
    <ul>{"".join(f"<li>{esc(item)}</li>" for item in warning_items)}</ul>
    <p>Validation errors: {esc(validation['errors'] if validation['errors'] else 'none')}</p>
  </section>

  <section>
    <h2>Embedding Cache</h2>
    <table><tbody>
      {"".join(f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>" for key, value in embedding_meta.items())}
    </tbody></table>
  </section>

  <section>
    <h2>Split And Shape Summary</h2>
    <p><strong>Split:</strong> {esc(split_summary)}</p>
    <p><strong>Shapes:</strong> {esc(shape_summary)}</p>
  </section>

  <section>
    <h2>Base Classes</h2>
    <table>
      <thead><tr><th>ID</th><th>Source</th><th>Canonical</th><th>Prompts</th><th>Cases</th><th>Median Voxels</th><th>Flags</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Aggregate Concepts</h2>
    <table>
      <thead><tr><th>ID</th><th>Canonical</th><th>Prompts</th><th>Components</th><th>Notes</th></tr></thead>
      <tbody>{"".join(aggregate_rows)}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-info",
        type=Path,
        default=ROOT / "S2I-Dataset-70cls" / "dataset_info.json",
    )
    parser.add_argument("--vocab", type=Path, default=ROOT / "configs" / "label_vocab.json")
    parser.add_argument(
        "--presence",
        type=Path,
        default=ROOT / "S2I-Dataset-70cls" / "class_presence.json",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=ROOT / "artifacts" / "text_embeddings" / "prompt_embeddings.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "vocabulary_audit.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
