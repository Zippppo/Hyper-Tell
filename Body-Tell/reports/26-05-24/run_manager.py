#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required. Run inside the project environment.") from exc


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPORT_DIR / "workflow_manifest.yaml"
STATUS_JSON_PATH = REPORT_DIR / "workflow_status.json"
STATUS_HTML_PATH = REPORT_DIR / "workflow_status.html"
LEAD_LOG_START = "<!-- WORKFLOW_PROGRESS_LOG_START -->"
LEAD_LOG_END = "<!-- WORKFLOW_PROGRESS_LOG_END -->"
ATTENTION_STATUSES = {"assigned", "submitted", "needs_fix", "gate_failed", "blocked"}


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_manifest(manifest: dict[str, Any]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True, sort_keys=False, width=120)
    refresh_status_files(manifest)


def get_task(manifest: dict[str, Any], task_id: str) -> dict[str, Any]:
    tasks = manifest.get("tasks", {})
    if task_id not in tasks:
        raise SystemExit(f"Unknown task id: {task_id}")
    return tasks[task_id]


def accepted_tasks(manifest: dict[str, Any]) -> set[str]:
    return {
        task_id
        for task_id, task in manifest.get("tasks", {}).items()
        if task.get("status") == "accepted"
    }


def is_runnable(manifest: dict[str, Any], task_id: str) -> bool:
    task = get_task(manifest, task_id)
    if task.get("status") not in {"todo", "ready", "needs_fix", "gate_failed"}:
        return False
    accepted = accepted_tasks(manifest)
    return all(dep in accepted for dep in task.get("depends_on", []))


def runnable_task_ids(manifest: dict[str, Any]) -> list[str]:
    return [
        task_id
        for task_id in manifest.get("tasks", {})
        if is_runnable(manifest, task_id)
    ]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def report_path(relative_path: str) -> Path:
    return REPORT_DIR / relative_path


def task_result_path(task: dict[str, Any]) -> Path:
    return report_path(task["result_file"])


def file_identity(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def file_was_updated(path: Path, previous_identity: tuple[int, int] | None) -> bool:
    current_identity = file_identity(path)
    return current_identity is not None and current_identity != previous_identity


def task_run_dir(task_id: str, task: dict[str, Any]) -> Path:
    result = task_result_path(task)
    if result.name:
        return result.parent
    return REPORT_DIR / "runs" / task_id


def bullet(items: list[str] | None) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def render_prompt(manifest: dict[str, Any], task_id: str, *, kind: str) -> str:
    task = get_task(manifest, task_id)
    template_name = "worker_prompt.md" if kind == "worker" else "review_prompt.md"
    template = (REPORT_DIR / "templates" / template_name).read_text(encoding="utf-8")
    result_file = task_result_path(task)
    review_file = task_run_dir(task_id, task) / "REVIEW.html"
    replacements = {
        "task_id": task_id,
        "title": task.get("title", ""),
        "owner": task.get("owner", ""),
        "repo_root": str(REPO_ROOT),
        "lead_file": rel(REPORT_DIR / manifest["lead_plan"]),
        "manifest_file": rel(MANIFEST_PATH),
        "plan_file": rel(REPORT_DIR / task["plan_file"]),
        "scope": task.get("scope", ""),
        "dependencies": bullet(task.get("depends_on", [])),
        "acceptance": bullet(task.get("acceptance", [])),
        "verification": bullet(task.get("verification", [])),
        "last_note": task.get("last_note", "-"),
        "result_file": rel(result_file),
        "review_file": rel(review_file),
        "gate_file": rel(task_run_dir(task_id, task) / "GATE.html"),
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def write_prompt(manifest: dict[str, Any], task_id: str, *, kind: str) -> Path:
    task = get_task(manifest, task_id)
    run_dir = task_run_dir(task_id, task)
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_name = "PROMPT.md" if kind == "worker" else "REVIEW_PROMPT.md"
    prompt_path = run_dir / prompt_name
    prompt_path.write_text(render_prompt(manifest, task_id, kind=kind), encoding="utf-8")
    return prompt_path


def run_codex(
    prompt: str,
    run_dir: Path,
    *,
    final_name: str,
    events_name: str,
    sandbox: str,
    bypass_permissions: bool = False,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / events_name
    final_path = run_dir / final_name
    if bypass_permissions:
        cmd = [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            "exec",
            "-C",
            str(REPO_ROOT),
            "--json",
            "-o",
            str(final_path),
            prompt,
        ]
    else:
        cmd = [
            "codex",
            "-a",
            "never",
            "exec",
            "-s",
            sandbox,
            "-C",
            str(REPO_ROOT),
            "--json",
            "-o",
            str(final_path),
            prompt,
        ]
    with events_path.open("w", encoding="utf-8") as events:
        return subprocess.run(cmd, cwd=REPO_ROOT, stdout=events, stderr=subprocess.STDOUT, text=True).returncode


def effective_sandbox(args: argparse.Namespace) -> str:
    if getattr(args, "full_access", False):
        return "danger-full-access"
    return args.sandbox


def run_worker_task(manifest: dict[str, Any], task_id: str, *, sandbox: str, bypass_permissions: bool) -> int:
    task = get_task(manifest, task_id)
    result_path = task_result_path(task)
    result_before = file_identity(result_path)
    prompt_path = write_prompt(manifest, task_id, kind="worker")
    mark_status(manifest, task_id, "assigned", f"Started Codex worker with {rel(prompt_path)}")
    save_manifest(manifest)
    code = run_codex(
        prompt_path.read_text(encoding="utf-8"),
        task_run_dir(task_id, task),
        final_name="codex-final.md",
        events_name="events.jsonl",
        sandbox=sandbox,
        bypass_permissions=bypass_permissions,
    )
    manifest = load_manifest()
    result_path = task_result_path(get_task(manifest, task_id))
    if code == 0 and file_was_updated(result_path, result_before):
        mark_status(manifest, task_id, "submitted", "Worker finished and RESULT.html exists.")
    elif code == 0 and result_path.exists():
        mark_status(manifest, task_id, "needs_fix", "Worker finished but RESULT.html was not updated.")
    elif code == 0:
        mark_status(manifest, task_id, "needs_fix", "Worker finished but RESULT.html is missing.")
    else:
        mark_status(manifest, task_id, "needs_fix", f"Codex worker exited with code {code}.")
    save_manifest(manifest)
    return code


def run_review_task(manifest: dict[str, Any], task_id: str, *, sandbox: str, bypass_permissions: bool) -> int:
    task = get_task(manifest, task_id)
    review_path = review_path_for(task_id, task)
    review_before = file_identity(review_path)
    prompt_path = write_prompt(manifest, task_id, kind="review")
    code = run_codex(
        prompt_path.read_text(encoding="utf-8"),
        task_run_dir(task_id, task),
        final_name="review-final.md",
        events_name="review-events.jsonl",
        sandbox=sandbox,
        bypass_permissions=bypass_permissions,
    )
    if code == 0 and not file_was_updated(review_path, review_before):
        return 1
    return code


def review_path_for(task_id: str, task: dict[str, Any]) -> Path:
    return task_run_dir(task_id, task) / "REVIEW.html"


def extract_review_verdict(review_path: Path) -> str | None:
    if not review_path.exists():
        return None
    text = review_path.read_text(encoding="utf-8", errors="ignore").lower()
    compact = re.sub(r"<[^>]+>", " ", text)
    compact = html.unescape(compact)
    match = re.search(r"\bverdict\b\s*[:：-]?\s*(accept|needs_fix|blocked)\b", compact)
    if match:
        return match.group(1)
    for verdict in ("needs_fix", "blocked", "accept"):
        if re.search(rf"\b{verdict}\b", compact):
            return verdict
    return None


def extract_phase1_slurm_gate(review_path: Path) -> tuple[bool, str]:
    gate_path = review_path.parent / "GATE.html"
    text_parts: list[str] = []
    if review_path.exists():
        text_parts.append(review_path.read_text(encoding="utf-8", errors="ignore"))
    if gate_path.exists():
        text_parts.append(gate_path.read_text(encoding="utf-8", errors="ignore"))
    if not text_parts:
        return False, "REVIEW.html and GATE.html are missing."
    text = "\n".join(text_parts).lower()
    compact = re.sub(r"<[^>]+>", " ", text)
    compact = html.unescape(compact)
    compact = re.sub(r"\s+", " ", compact)
    mentions_gate = "phase1 slurm gate" in compact or "body-tell-phase1.sh" in compact
    pass_marker = re.search(r"\bphase1\s+slurm\s+gate\b\s*[:：-]?\s*(pass|passed)\b", compact)
    completed_state = re.search(r"\bsacct\s+final\s+state\b\s*[:：-]?\s*completed\b", compact)
    exit_zero_field = re.search(r"\bsacct\s+exit\s+code\b\s*[:：-]?\s*0:0\b", compact)
    completed_sacct_line = re.search(r"\b\d+(?:\.batch)?\s+completed\s+0:0\b", compact)
    if mentions_gate and (pass_marker or (completed_state and exit_zero_field) or completed_sacct_line):
        return True, "Phase1 Slurm gate evidence shows COMPLETED with exit code 0:0."
    if not mentions_gate:
        return False, "REVIEW.html lacks required phase1 Slurm gate evidence."
    if not completed_state and not completed_sacct_line:
        return False, "Phase1 Slurm gate evidence does not show COMPLETED."
    return False, "Phase1 Slurm gate evidence does not show exit code 0:0."


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_timestamp(value: Any) -> dt.datetime | None:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        try:
            parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def compact_text(value: Any, *, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def file_snapshot(path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": rel(path),
        "exists": path.exists(),
    }
    if snapshot["exists"]:
        stat = path.stat()
        snapshot["bytes"] = stat.st_size
        snapshot["modified_at"] = dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).replace(microsecond=0).isoformat()
    return snapshot


def task_snapshot(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    review_path = review_path_for(task_id, task)
    return {
        "task_id": task_id,
        "title": task.get("title", ""),
        "owner": task.get("owner", ""),
        "status": task.get("status", ""),
        "updated_at": task.get("updated_at", ""),
        "last_note": compact_text(task.get("last_note", "")),
        "result": file_snapshot(task_result_path(task)),
        "review": file_snapshot(review_path),
        "verdict": extract_review_verdict(review_path) or "-",
    }


def latest_task_id(manifest: dict[str, Any]) -> str | None:
    latest_id: str | None = None
    latest_at: dt.datetime | None = None
    for task_id, task in manifest.get("tasks", {}).items():
        parsed = parse_timestamp(task.get("updated_at"))
        if parsed is None:
            continue
        if latest_at is None or parsed > latest_at:
            latest_id = task_id
            latest_at = parsed
    return latest_id


def ordered_attention_task_ids(manifest: dict[str, Any]) -> list[str]:
    status_priority = {
        "blocked": 0,
        "gate_failed": 1,
        "needs_fix": 2,
        "submitted": 3,
        "assigned": 4,
    }
    task_items = [
        (task_id, task)
        for task_id, task in manifest.get("tasks", {}).items()
        if task.get("status") in ATTENTION_STATUSES
    ]

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, str]:
        task_id, task = item
        parsed = parse_timestamp(task.get("updated_at"))
        timestamp = parsed.timestamp() if parsed else 0.0
        return (status_priority.get(task.get("status", ""), 99), -timestamp, task_id)

    return [task_id for task_id, _ in sorted(task_items, key=sort_key)]


def status_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in manifest.get("tasks", {}).values():
        status = str(task.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_workflow_status(manifest: dict[str, Any]) -> dict[str, Any]:
    tasks = manifest.get("tasks", {})
    ready_ids = runnable_task_ids(manifest)
    attention_ids = ordered_attention_task_ids(manifest)
    latest_id = latest_task_id(manifest)
    focus_id = attention_ids[0] if attention_ids else ready_ids[0] if ready_ids else latest_id

    focus = task_snapshot(focus_id, tasks[focus_id]) if focus_id else None
    latest = task_snapshot(latest_id, tasks[latest_id]) if latest_id else None
    next_task = task_snapshot(ready_ids[0], tasks[ready_ids[0]]) if ready_ids else None
    attention = [task_snapshot(task_id, tasks[task_id]) for task_id in attention_ids[:5]]

    if focus:
        result_state = "yes" if focus["result"]["exists"] else "no"
        review_state = "yes" if focus["review"]["exists"] else "no"
        focus_text = f"{focus['task_id']} {focus['status']}"
    else:
        result_state = "no"
        review_state = "no"
        focus_text = "-"
    attention_text = ",".join(item["task_id"] for item in attention) if attention else "none"
    next_text = next_task["task_id"] if next_task else "-"
    latest_text = latest["updated_at"] if latest else "-"
    summary = (
        f"focus={focus_text}; latest_update={latest_text}; "
        f"result={result_state}; review={review_state}; "
        f"verdict={(focus or {}).get('verdict', '-')}; next={next_text}; attention={attention_text}"
    )

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_manifest": rel(MANIFEST_PATH),
        "lead_safe": True,
        "rule": "Leader agents must use this compact status and must not tail events.jsonl or review-events.jsonl.",
        "summary": summary,
        "counts": status_counts(manifest),
        "latest_update": latest,
        "focus": focus,
        "attention": attention,
        "next_task": next_task,
        "ready_task_ids": ready_ids[:10],
    }


def bool_word(snapshot: dict[str, Any] | None, key: str) -> str:
    if not snapshot:
        return "no"
    return "yes" if snapshot[key]["exists"] else "no"


def format_snapshot(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return "-"
    return (
        f"{snapshot['task_id']} {snapshot['status']} "
        f"result={bool_word(snapshot, 'result')} "
        f"review={bool_word(snapshot, 'review')} "
        f"verdict={snapshot.get('verdict', '-')} "
        f"updated={snapshot.get('updated_at') or '-'}"
    )


def render_status_lines(status: dict[str, Any]) -> list[str]:
    counts = " ".join(f"{key}={value}" for key, value in status["counts"].items()) or "-"
    attention = ", ".join(format_snapshot(item) for item in status["attention"]) or "none"
    ready = ", ".join(status["ready_task_ids"]) or "-"
    return [
        f"generated_at: {status['generated_at']}",
        f"summary: {status['summary']}",
        f"latest_update: {format_snapshot(status['latest_update'])}",
        f"focus: {format_snapshot(status['focus'])}",
        f"attention: {attention}",
        f"next_task: {format_snapshot(status['next_task'])}",
        f"ready: {ready}",
        f"counts: {counts}",
        "rule: do not tail events.jsonl or review-events.jsonl from the long-lived leader context",
    ]


def render_status_html(status: dict[str, Any]) -> str:
    lines = render_status_lines(status)
    escaped_lines = "\n".join(f"<li>{html.escape(line)}</li>" for line in lines)
    json_path = html.escape(rel(STATUS_JSON_PATH))
    generated_at = html.escape(status["generated_at"])
    summary = html.escape(status["summary"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Body-Tell Workflow Status</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; line-height: 1.45; }}
    main {{ max-width: 980px; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    .summary {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f3f5f7; padding: 10px; border: 1px solid #d8dee4; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <main>
    <h1>Body-Tell Workflow Status</h1>
    <p><strong>Generated:</strong> {generated_at}</p>
    <p class="summary">{summary}</p>
    <ul>
{escaped_lines}
    </ul>
    <p>JSON source: {json_path}</p>
  </main>
</body>
</html>
"""


def refresh_status_files(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest is None:
        manifest = load_manifest()
    status = build_workflow_status(manifest)
    STATUS_JSON_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATUS_HTML_PATH.write_text(render_status_html(status), encoding="utf-8")
    return status


def mark_status(manifest: dict[str, Any], task_id: str, status: str, note: str = "") -> None:
    task = get_task(manifest, task_id)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    task["status"] = status
    task["updated_at"] = now
    if note:
        task["last_note"] = note
    history = task.setdefault("history", [])
    history.append({"at": now, "status": status, "note": note})


def append_lead_log(manifest: dict[str, Any], task_id: str, status: str, note: str) -> None:
    lead_path = REPORT_DIR / manifest["lead_plan"]
    content = lead_path.read_text(encoding="utf-8")
    task = get_task(manifest, task_id)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = rel(task_result_path(task))
    escaped_note = html.escape(note or task.get("last_note", ""))
    escaped_title = html.escape(task.get("title", ""))
    entry = (
        "\n"
        '<article class="workflow-progress-entry">\n'
        f"<h3>{html.escape(now)} - {html.escape(task_id)} - {html.escape(status)}</h3>\n"
        f"<p><strong>Title:</strong> {escaped_title}</p>\n"
        f"<p><strong>Owner:</strong> {html.escape(task.get('owner', ''))}</p>\n"
        f"<p><strong>Result:</strong> {html.escape(result)}</p>\n"
        f"<p><strong>Note:</strong> {escaped_note}</p>\n"
        "</article>\n"
    )
    if LEAD_LOG_START not in content:
        block = (
            "\n<section id=\"agent-workflow-progress\">\n"
            "<h2>Agent Workflow Progress Log</h2>\n"
            f"{LEAD_LOG_START}\n"
            f"{LEAD_LOG_END}\n"
            "</section>\n"
        )
        body_close = content.lower().rfind("</body>")
        if body_close == -1:
            content += block
        else:
            content = content[:body_close] + block + content[body_close:]
    content = content.replace(LEAD_LOG_END, entry + LEAD_LOG_END)
    lead_path.write_text(content, encoding="utf-8")


def cmd_status(args: argparse.Namespace) -> int:
    status = refresh_status_files(load_manifest()) if not args.no_write else build_workflow_status(load_manifest())
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        for line in render_status_lines(status):
            print(line)
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    try:
        while True:
            status = refresh_status_files(load_manifest())
            if not args.quiet:
                print(f"{status['generated_at']} {status['summary']}", flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


def cmd_list(_: argparse.Namespace) -> int:
    manifest = load_manifest()
    accepted = accepted_tasks(manifest)
    for task_id, task in manifest.get("tasks", {}).items():
        deps = task.get("depends_on", [])
        missing = [dep for dep in deps if dep not in accepted]
        state = "runnable" if is_runnable(manifest, task_id) else "waiting"
        if task.get("status") in {"accepted", "assigned", "submitted", "gate_failed", "blocked"}:
            state = task.get("status")
        missing_text = ",".join(missing) if missing else "-"
        print(f"{task_id:12} {task.get('status', '-'):10} {state:10} missing_deps={missing_text} {task.get('title', '')}")
    return 0


def cmd_ready(_: argparse.Namespace) -> int:
    manifest = load_manifest()
    for task_id in runnable_task_ids(manifest):
        task = get_task(manifest, task_id)
        print(f"{task_id}\t{task.get('owner')}\t{task.get('title')}")
    return 0


def cmd_next(_: argparse.Namespace) -> int:
    manifest = load_manifest()
    ready = runnable_task_ids(manifest)
    if not ready:
        print("No runnable tasks.")
        return 1
    task_id = ready[0]
    task = get_task(manifest, task_id)
    print(f"{task_id}\t{task.get('owner')}\t{task.get('title')}")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    prompt_path = write_prompt(manifest, args.task_id, kind="worker")
    print(rel(prompt_path))
    return 0


def cmd_review_prompt(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    prompt_path = write_prompt(manifest, args.task_id, kind="review")
    print(rel(prompt_path))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    if not args.force and not is_runnable(manifest, args.task_id):
        raise SystemExit(f"Task is not runnable: {args.task_id}")
    return run_worker_task(
        manifest,
        args.task_id,
        sandbox=effective_sandbox(args),
        bypass_permissions=args.bypass_permissions,
    )


def cmd_review(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    task = get_task(manifest, args.task_id)
    prompt_path = write_prompt(manifest, args.task_id, kind="review")
    if not args.execute:
        print(rel(prompt_path))
        return 0
    return run_review_task(
        manifest,
        args.task_id,
        sandbox=effective_sandbox(args),
        bypass_permissions=args.bypass_permissions,
    )


def cmd_auto(args: argparse.Namespace) -> int:
    completed = 0
    while completed < args.max_tasks:
        manifest = load_manifest()
        ready = runnable_task_ids(manifest)
        if not ready:
            print("No runnable tasks remain.")
            return 0

        task_id = ready[0]
        task = get_task(manifest, task_id)
        print(f"[auto] running {task_id}: {task.get('title', '')}")
        worker_code = run_worker_task(
            manifest,
            task_id,
            sandbox=effective_sandbox(args),
            bypass_permissions=args.bypass_permissions,
        )
        manifest = load_manifest()
        task = get_task(manifest, task_id)
        if worker_code != 0 or task.get("status") != "submitted":
            print(f"[auto] worker did not submit a valid result for {task_id}; status={task.get('status')}")
            if args.continue_on_failure:
                completed += 1
                continue
            return worker_code or 1

        print(f"[auto] reviewing {task_id}")
        review_code = run_review_task(
            manifest,
            task_id,
            sandbox=effective_sandbox(args),
            bypass_permissions=args.bypass_permissions,
        )
        manifest = load_manifest()
        task = get_task(manifest, task_id)
        review_path = review_path_for(task_id, task)
        verdict = extract_review_verdict(review_path)
        if review_code != 0:
            mark_status(manifest, task_id, "needs_fix", f"Reviewer exited with code {review_code}.")
            save_manifest(manifest)
            print(f"[auto] reviewer failed for {task_id}")
            if args.continue_on_failure:
                completed += 1
                continue
            return review_code

        if verdict == "accept":
            gate_ok, gate_note = extract_phase1_slurm_gate(review_path)
            if not gate_ok:
                note = f"Reviewer accept rejected: {gate_note} See {rel(review_path)}."
                mark_status(manifest, task_id, "gate_failed", note)
                save_manifest(manifest)
                print(f"[auto] {task_id} -> gate_failed; {gate_note}")
                if args.continue_on_failure:
                    completed += 1
                    continue
                return 1
            note = f"Accepted by reviewer with phase1 Slurm gate pass. See {rel(review_path)}."
            mark_status(manifest, task_id, "accepted", note)
            append_lead_log(manifest, task_id, "accepted", note)
            save_manifest(manifest)
            print(f"[auto] accepted {task_id}")
        elif verdict in {"needs_fix", "blocked"}:
            note = f"Reviewer verdict: {verdict}. See {rel(review_path)}."
            mark_status(manifest, task_id, verdict, note)
            save_manifest(manifest)
            print(f"[auto] {task_id} -> {verdict}")
            if not args.continue_on_failure:
                return 1
        else:
            note = f"Could not parse reviewer verdict. See {rel(review_path)}."
            mark_status(manifest, task_id, "needs_fix", note)
            save_manifest(manifest)
            print(f"[auto] {note}")
            if not args.continue_on_failure:
                return 1

        completed += 1

    print(f"Reached max task limit: {args.max_tasks}")
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    task = get_task(manifest, args.task_id)
    if not args.allow_missing_result and not task_result_path(task).exists():
        raise SystemExit(f"Cannot accept {args.task_id}: missing {rel(task_result_path(task))}")
    gate_ok, gate_note = extract_phase1_slurm_gate(review_path_for(args.task_id, task))
    if not gate_ok:
        raise SystemExit(f"Cannot accept {args.task_id}: {gate_note}")
    note = f"{args.note} Phase1 Slurm gate verified: {gate_note}"
    mark_status(manifest, args.task_id, "accepted", note)
    append_lead_log(manifest, args.task_id, "accepted", note)
    save_manifest(manifest)
    print(f"accepted {args.task_id}")
    return 0


def cmd_fail(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    mark_status(manifest, args.task_id, "needs_fix", args.note)
    save_manifest(manifest)
    print(f"needs_fix {args.task_id}")
    return 0


def cmd_gate_fail(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    mark_status(manifest, args.task_id, "gate_failed", args.note)
    save_manifest(manifest)
    print(f"gate_failed {args.task_id}")
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    mark_status(manifest, args.task_id, "blocked", args.note)
    save_manifest(manifest)
    print(f"blocked {args.task_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the Body-Tell long-running agent workflow.")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("status", help="Write and print compact leader-safe workflow status.")
    p.add_argument("--json", action="store_true", help="Print the full compact JSON status.")
    p.add_argument("--no-write", action="store_true", help="Do not refresh workflow_status.json/html.")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("monitor", help="Continuously refresh compact status files without reading event logs.")
    p.add_argument("--interval", type=float, default=30.0, help="Seconds between status refreshes.")
    p.add_argument("--once", action="store_true", help="Refresh once and exit.")
    p.add_argument("--quiet", action="store_true", help="Do not print the one-line summary on each refresh.")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("list", help="List all tasks and dependency state.")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("ready", help="List runnable tasks.")
    p.set_defaults(func=cmd_ready)

    p = sub.add_parser("next", help="Print the first runnable task.")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("prompt", help="Generate a worker prompt without running Codex.")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("review-prompt", help="Generate a reviewer prompt without running Codex.")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_review_prompt)

    p = sub.add_parser("run", help="Run a worker task through Codex CLI.")
    p.add_argument("task_id")
    p.add_argument("--force", action="store_true", help="Run even if dependencies are not accepted.")
    p.add_argument("--sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    p.add_argument("--full-access", action="store_true", help="Alias for --sandbox danger-full-access.")
    p.add_argument("--bypass-permissions", action="store_true", help="Use Codex's full approval/sandbox bypass flag.")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("review", help="Generate or execute a reviewer task through Codex CLI.")
    p.add_argument("task_id")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    p.add_argument("--full-access", action="store_true", help="Alias for --sandbox danger-full-access.")
    p.add_argument("--bypass-permissions", action="store_true", help="Use Codex's full approval/sandbox bypass flag.")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("auto", help="Run manager loop: worker, reviewer, status update, then next runnable task.")
    p.add_argument("--max-tasks", type=int, default=999)
    p.add_argument("--sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    p.add_argument("--full-access", action="store_true", help="Alias for --sandbox danger-full-access.")
    p.add_argument("--bypass-permissions", action="store_true", help="Use Codex's full approval/sandbox bypass flag.")
    p.add_argument("--continue-on-failure", action="store_true")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("accept", help="Accept a submitted task and append progress to lead.html.")
    p.add_argument("task_id")
    p.add_argument("--note", required=True)
    p.add_argument("--allow-missing-result", action="store_true")
    p.set_defaults(func=cmd_accept)

    p = sub.add_parser("fail", help="Return a task for fixes after review failed.")
    p.add_argument("task_id")
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_fail)

    p = sub.add_parser("gate-fail", help="Return a task for fixes after a required integration gate failed.")
    p.add_argument("task_id")
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_gate_fail)

    p = sub.add_parser("block", help="Mark a task as blocked.")
    p.add_argument("task_id")
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_block)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
