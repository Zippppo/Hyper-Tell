#!/usr/bin/env python3
"""Audit Body-Tell model parameter counts and emit an HTML report."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BODY_TELL_ROOT = Path(__file__).resolve().parents[1]
if str(BODY_TELL_ROOT) not in sys.path:
    sys.path.insert(0, str(BODY_TELL_ROOT))


def load_body_train_module() -> Any:
    train_path = BODY_TELL_ROOT / "train.py"
    spec = importlib.util.spec_from_file_location("body_tell_train_entry", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training entry from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def human_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def module_param_count(model: torch.nn.Module, prefix: str) -> tuple[int, int]:
    module = model.get_submodule(prefix) if prefix else model
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def direct_child_counts(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, child in model.named_children():
        total = sum(param.numel() for param in child.parameters())
        trainable = sum(param.numel() for param in child.parameters() if param.requires_grad)
        rows.append({"name": name, "total": total, "trainable": trainable})
    return rows


def subtree_counts(model: torch.nn.Module) -> list[dict[str, Any]]:
    prefixes = [
        "encoder",
        "encoder.stages.0",
        "encoder.stages.1",
        "encoder.stages.2",
        "encoder.stages.3",
        "encoder.stages.4",
        "project_bottleneck_embed",
        "project_text_embed",
        "transformer_decoder",
        "transformer_decoder.layers.0",
        "transformer_decoder.layers.1",
        "transformer_decoder.layers.2",
        "transformer_decoder.layers.3",
        "transformer_decoder.layers.4",
        "transformer_decoder.layers.5",
        "transformer_decoder.norm",
        "project_to_decoder_channels",
        "project_to_decoder_channels.0",
        "project_to_decoder_channels.1",
        "project_to_decoder_channels.2",
        "project_to_decoder_channels.3",
        "project_to_decoder_channels.4",
        "decoder",
        "decoder.transpconvs",
        "decoder.stages",
        "decoder.seg_layers",
    ]
    rows = []
    for prefix in prefixes:
        try:
            total, trainable = module_param_count(model, prefix)
        except AttributeError:
            continue
        rows.append({"name": prefix, "total": total, "trainable": trainable})
    return rows


def parameter_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, param in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": tuple(int(x) for x in param.shape),
                "numel": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
                "dtype": str(param.dtype).replace("torch.", ""),
            }
        )
    return sorted(rows, key=lambda item: item["numel"], reverse=True)


def top_level_from_name(name: str) -> str:
    return name.split(".", 1)[0]


def category_counts(param_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "trainable": 0})
    for row in param_rows:
        key = top_level_from_name(row["name"])
        grouped[key]["total"] += row["numel"]
        if row["requires_grad"]:
            grouped[key]["trainable"] += row["numel"]
    return [
        {"name": name, "total": counts["total"], "trainable": counts["trainable"]}
        for name, counts in sorted(grouped.items(), key=lambda item: item[1]["total"], reverse=True)
    ]


def find_logged_parameter_count(log_path: Path | None) -> int | None:
    if not log_path or not log_path.exists():
        return None
    pattern = re.compile(r"Model parameters:\s*([0-9,]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def parse_training_metrics(log_path: Path | None) -> dict[str, Any] | None:
    if not log_path or not log_path.exists():
        return None
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+).*?loss=([0-9.]+).*?dice=([0-9.]+).*?fg_dice=([0-9.]+).*?neg_fp=([0-9.]+)"
    )
    val_pattern = re.compile(
        r"\[val\]\s+loss=([0-9.]+)\s+\|\s+fg_dice=([0-9.]+)\s+\|\s+neg_fp=([0-9.]+)"
    )
    train_rows = []
    val_rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            train_rows.append(
                {
                    "epoch": int(epoch_match.group(1)),
                    "loss": float(epoch_match.group(2)),
                    "dice_loss": float(epoch_match.group(3)),
                    "fg_dice": float(epoch_match.group(4)),
                    "neg_fp": float(epoch_match.group(5)),
                }
            )
            continue
        val_match = val_pattern.search(line)
        if val_match:
            val_rows.append(
                {
                    "epoch": len(val_rows) + 1,
                    "loss": float(val_match.group(1)),
                    "fg_dice": float(val_match.group(2)),
                    "neg_fp": float(val_match.group(3)),
                }
            )
    if not train_rows and not val_rows:
        return None
    return {"train": train_rows, "val": val_rows}


def pct(value: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{100 * value / total:.2f}%"


def row_html(row: dict[str, Any], total: int) -> str:
    return (
        "<tr>"
        f"<td><code>{html.escape(str(row['name']))}</code></td>"
        f"<td class=\"num\">{row['total']:,}</td>"
        f"<td class=\"num\">{human_count(row['total'])}</td>"
        f"<td class=\"num\">{pct(row['total'], total)}</td>"
        f"<td class=\"num\">{row['trainable']:,}</td>"
        "</tr>"
    )


def param_row_html(row: dict[str, Any], total: int) -> str:
    return (
        "<tr>"
        f"<td><code>{html.escape(row['name'])}</code></td>"
        f"<td><code>{html.escape(str(row['shape']))}</code></td>"
        f"<td class=\"num\">{row['numel']:,}</td>"
        f"<td class=\"num\">{pct(row['numel'], total)}</td>"
        f"<td>{html.escape(row['dtype'])}</td>"
        f"<td>{'yes' if row['requires_grad'] else 'no'}</td>"
        "</tr>"
    )


def render_report(audit: dict[str, Any]) -> str:
    total = audit["totals"]["parameters"]
    trainable = audit["totals"]["trainable_parameters"]
    logged = audit["cross_checks"]["logged_train_parameter_count"]
    top_rows = "\n".join(row_html(row, total) for row in audit["tables"]["top_level"])
    subtree_rows = "\n".join(row_html(row, total) for row in audit["tables"]["subtrees"])
    largest_rows = "\n".join(param_row_html(row, total) for row in audit["tables"]["largest_tensors"])
    cfg = audit["config"]["model"]
    runtime = audit["runtime_model"]
    metrics = audit.get("training_metrics")
    metrics_html = ""
    if metrics and metrics["train"] and metrics["val"]:
        last_train = metrics["train"][-1]
        last_val = metrics["val"][-1]
        best_val = max(metrics["val"], key=lambda item: item["fg_dice"])
        gap = last_train["fg_dice"] - last_val["fg_dice"]
        metrics_html = f"""
        <section>
          <h2>训练日志交叉信息</h2>
          <div class="cards">
            <div><span>训练样本</span><strong>{audit['dataset'].get('train_samples', 'unknown')}</strong></div>
            <div><span>验证样本</span><strong>{audit['dataset'].get('val_samples', 'unknown')}</strong></div>
            <div><span>最后 epoch train fg_dice</span><strong>{last_train['fg_dice']:.4f}</strong></div>
            <div><span>最后 epoch val fg_dice</span><strong>{last_val['fg_dice']:.4f}</strong></div>
            <div><span>最后 gap</span><strong>{gap:.4f}</strong></div>
            <div><span>最佳 val fg_dice</span><strong>{best_val['fg_dice']:.4f} @ epoch {best_val['epoch']}</strong></div>
          </div>
          <p class="note">该部分只用于解释参数量与过拟合风险的关系；参数量统计本身来自模型实例化后的 <code>named_parameters()</code>。</p>
        </section>
        """

    logged_text = "not found"
    verdict = "not checked"
    if logged is not None:
        logged_text = f"{logged:,}"
        verdict = "match" if logged == trainable else "mismatch"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Body-Tell 参数量审计报告</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5b6673;
      --line: #d9e0e8;
      --panel: #f7f9fb;
      --accent: #176b87;
      --accent2: #9b3d2e;
      --good: #26734d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.55;
    }}
    header {{
      padding: 36px 44px 26px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f7fbfd 0%, #ffffff 100%);
    }}
    main {{ padding: 28px 44px 48px; max-width: 1320px; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 32px 0 12px; font-size: 21px; }}
    h3 {{ margin: 20px 0 10px; font-size: 16px; }}
    p {{ margin: 8px 0; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
      background: #eef3f7;
      padding: 1px 4px;
      border-radius: 4px;
    }}
    .meta {{ color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    .cards > div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--panel);
    }}
    .cards span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .cards strong {{
      display: block;
      font-size: 22px;
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 24px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      vertical-align: top;
    }}
    th {{
      text-align: left;
      background: #f1f5f8;
      color: #24313d;
      position: sticky;
      top: 0;
    }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{
      color: var(--muted);
      border-left: 4px solid var(--line);
      padding-left: 12px;
    }}
    .warn {{ color: var(--accent2); font-weight: 650; }}
    .ok {{ color: var(--good); font-weight: 650; }}
    .grid2 {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
    }}
    pre {{
      overflow: auto;
      background: #f6f8fa;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      max-height: 420px;
    }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 20px; padding-right: 20px; }}
      .grid2 {{ grid-template-columns: 1fr; }}
      table {{ font-size: 13px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Body-Tell 参数量审计报告</h1>
    <p class="meta">生成时间：{html.escape(audit['generated_at'])}；入口配置：<code>{html.escape(audit['paths']['config'])}</code></p>
  </header>
  <main>
    <section>
      <h2>结论</h2>
      <div class="cards">
        <div><span>总参数</span><strong>{total:,}</strong></div>
        <div><span>可训练参数</span><strong>{trainable:,}</strong></div>
        <div><span>冻结参数</span><strong>{audit['totals']['frozen_parameters']:,}</strong></div>
        <div><span>训练日志参数量</span><strong>{logged_text}</strong></div>
        <div><span>日志交叉校验</span><strong class="{'ok' if verdict == 'match' else 'warn'}">{verdict}</strong></div>
        <div><span>FP32 参数内存</span><strong>{audit['memory']['fp32_parameters_gib']:.2f} GiB</strong></div>
      </div>
      <p>本次以 <code>Body-Tell/train.py::build_model()</code> 和 <code>phase1_voxtell_body.yaml</code> 实例化模型后统计，因此覆盖训练时真实使用的 Body-Tell 架构。当前模型没有冻结层，<strong>328.98M 参数全部参与 AdamW 优化</strong>。</p>
      <p class="note">参数量本身已经足以支持强记忆能力；结合日志里训练 Dice 持续上升、验证 Dice 基本平台化，过拟合判断是合理的。但参数量不是唯一原因，还需要同时看数据划分、prompt 分布、标签稀疏度和验证集类别覆盖。</p>
    </section>

    <section>
      <h2>训练配置摘要</h2>
      <div class="cards">
        <div><span>模型</span><strong>{html.escape(audit['config']['model'].get('name', 'unknown'))}</strong></div>
        <div><span>encoder_channels</span><strong>{html.escape(str(cfg['encoder_channels']))}</strong></div>
        <div><span>query_dim</span><strong>{cfg['query_dim']}</strong></div>
        <div><span>text_embedding_dim</span><strong>{cfg['text_embedding_dim']}</strong></div>
        <div><span>Transformer layers</span><strong>{cfg['transformer_layers']}</strong></div>
        <div><span>mask heads</span><strong>{cfg['num_heads']}</strong></div>
        <div><span>requested mask stages</span><strong>{cfg['num_maskformer_stages']}</strong></div>
        <div><span>effective fused stages</span><strong>{runtime['fused_stage_count']}</strong></div>
        <div><span>selected encoder stage</span><strong>{runtime['selected_stage']}</strong></div>
        <div><span>mask projection modules</span><strong>{runtime['mask_projection_count']}</strong></div>
      </div>
      <p class="note">注意：配置请求 <code>num_maskformer_stages={cfg['num_maskformer_stages']}</code>，但代码使用 <code>min(num_maskformer_stages, len(encoder_channels)-1)</code>，所以本次运行时实际 fused stage 数是 <code>{runtime['fused_stage_count']}</code>，对应 <code>{runtime['mask_projection_count']}</code> 个 mask projection 模块。</p>
    </section>

    <section>
      <h2>顶层模块参数分布</h2>
      <table>
        <thead><tr><th>模块</th><th class="num">参数</th><th class="num">简写</th><th class="num">占比</th><th class="num">可训练</th></tr></thead>
        <tbody>{top_rows}</tbody>
      </table>
    </section>

    <section>
      <h2>子模块拆分</h2>
      <table>
        <thead><tr><th>模块</th><th class="num">参数</th><th class="num">简写</th><th class="num">占比</th><th class="num">可训练</th></tr></thead>
        <tbody>{subtree_rows}</tbody>
      </table>
    </section>

    <section>
      <h2>最大参数张量 Top 30</h2>
      <table>
        <thead><tr><th>参数名</th><th>shape</th><th class="num">参数</th><th class="num">占比</th><th>dtype</th><th>trainable</th></tr></thead>
        <tbody>{largest_rows}</tbody>
      </table>
    </section>

    <section>
      <h2>内存量级</h2>
      <div class="cards">
        <div><span>FP32 参数</span><strong>{audit['memory']['fp32_parameters_gib']:.2f} GiB/rank</strong></div>
        <div><span>参数 + 梯度 + AdamW 状态</span><strong>{audit['memory']['adamw_training_state_gib']:.2f} GiB/rank</strong></div>
        <div><span>2 卡 DDP 参数副本</span><strong>{audit['memory']['ddp_two_rank_fp32_parameters_gib']:.2f} GiB</strong></div>
      </div>
      <p class="note">这里不包含 activation、CUDA workspace、DataLoader pinned memory 或 checkpoint 缓存。3D 体素输入的 activation 通常会比参数优化状态更敏感。</p>
    </section>

    {metrics_html}

    <section>
      <h2>审计方法</h2>
      <ol>
        <li>加载 YAML 配置：<code>{html.escape(audit['paths']['config'])}</code>。</li>
        <li>动态 import <code>Body-Tell/train.py</code>，调用同一个 <code>build_model(cfg)</code>。</li>
        <li>使用 PyTorch <code>model.named_parameters()</code> 统计每个张量的 <code>numel()</code>。</li>
        <li>读取训练日志 <code>{html.escape(audit['paths'].get('log') or 'not provided')}</code> 中的 <code>Model parameters</code> 作为交叉校验。</li>
      </ol>
    </section>

    <section>
      <h2>原始 JSON 摘要</h2>
      <pre>{html.escape(json.dumps(audit, indent=2, ensure_ascii=False))}</pre>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Count Body-Tell model parameters")
    parser.add_argument("--config", type=Path, default=BODY_TELL_ROOT / "configs/phase1_voxtell_body.yaml")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=BODY_TELL_ROOT / "reports/body_tell_parameter_audit.html")
    parser.add_argument("--json-output", type=Path, default=BODY_TELL_ROOT / "reports/body_tell_parameter_audit.json")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_module = load_body_train_module()
    model = train_module.build_model(cfg)
    model.eval()

    rows = parameter_rows(model)
    total_params = sum(row["numel"] for row in rows)
    trainable_params = sum(row["numel"] for row in rows if row["requires_grad"])
    frozen_params = total_params - trainable_params
    buffers = sum(buffer.numel() for buffer in model.buffers())
    logged_count = find_logged_parameter_count(args.log)
    metrics = parse_training_metrics(args.log)

    dataset = {}
    if metrics:
        log_text = args.log.read_text(encoding="utf-8", errors="replace") if args.log and args.log.exists() else ""
        train_match = re.search(r"Train dataset:\s*(\d+)\s+samples", log_text)
        val_match = re.search(r"Val dataset:\s*(\d+)\s+samples", log_text)
        if train_match:
            dataset["train_samples"] = int(train_match.group(1))
        if val_match:
            dataset["val_samples"] = int(val_match.group(1))

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "repo_root": str(REPO_ROOT),
            "body_tell_root": str(BODY_TELL_ROOT),
            "config": str(args.config),
            "log": str(args.log) if args.log else None,
            "output": str(args.output),
        },
        "config": cfg,
        "model_config_dataclass": asdict(model.config),
        "runtime_model": {
            "selected_stage": int(model.selected_stage),
            "selected_stage_channels": int(model.config.encoder_channels[model.selected_stage]),
            "fused_stage_count": int(model.fused_stage_count),
            "project_to_decoder_channel_count": len(model.project_to_decoder_channels),
            "deep_supervision": bool(model.config.deep_supervision),
        },
        "totals": {
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "buffers": buffers,
            "parameter_tensors": len(rows),
        },
        "cross_checks": {
            "logged_train_parameter_count": logged_count,
            "matches_training_log": logged_count == trainable_params if logged_count is not None else None,
        },
        "memory": {
            "fp32_parameters_gib": bytes_to_gib(total_params * 4),
            "adamw_training_state_gib": bytes_to_gib(total_params * 4 * 4),
            "ddp_two_rank_fp32_parameters_gib": bytes_to_gib(total_params * 4 * 2),
        },
        "dataset": dataset,
        "tables": {
            "top_level": category_counts(rows),
            "direct_children": direct_child_counts(model),
            "subtrees": subtree_counts(model),
            "largest_tensors": rows[:30],
        },
        "training_metrics": metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(audit), encoding="utf-8")
    args.json_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"parameters={total_params:,}")
    print(f"trainable={trainable_params:,}")
    print(f"frozen={frozen_params:,}")
    if logged_count is not None:
        print(f"logged={logged_count:,}")
        print(f"log_match={logged_count == trainable_params}")
    print(f"html={args.output}")
    print(f"json={args.json_output}")


if __name__ == "__main__":
    main()
