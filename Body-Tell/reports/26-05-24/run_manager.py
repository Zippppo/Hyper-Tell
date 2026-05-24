#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required. Run inside the project environment.") from exc


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPORT_DIR / "workflow_manifest.yaml"
LEAD_LOG_START = "<!-- WORKFLOW_PROGRESS_LOG_START -->"
LEAD_LOG_END = "<!-- WORKFLOW_PROGRESS_LOG_END -->"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_manifest(manifest: dict[str, Any]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True, sort_keys=False, width=120)


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
    if task.get("status") not in {"todo", "ready", "needs_fix"}:
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
        "result_file": rel(result_file),
        "review_file": rel(review_file),
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
    if code == 0 and result_path.exists():
        mark_status(manifest, task_id, "submitted", "Worker finished and RESULT.html exists.")
    elif code == 0:
        mark_status(manifest, task_id, "needs_fix", "Worker finished but RESULT.html is missing.")
    else:
        mark_status(manifest, task_id, "needs_fix", f"Codex worker exited with code {code}.")
    save_manifest(manifest)
    return code


def run_review_task(manifest: dict[str, Any], task_id: str, *, sandbox: str, bypass_permissions: bool) -> int:
    task = get_task(manifest, task_id)
    prompt_path = write_prompt(manifest, task_id, kind="review")
    return run_codex(
        prompt_path.read_text(encoding="utf-8"),
        task_run_dir(task_id, task),
        final_name="review-final.md",
        events_name="review-events.jsonl",
        sandbox=sandbox,
        bypass_permissions=bypass_permissions,
    )


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


def cmd_list(_: argparse.Namespace) -> int:
    manifest = load_manifest()
    accepted = accepted_tasks(manifest)
    for task_id, task in manifest.get("tasks", {}).items():
        deps = task.get("depends_on", [])
        missing = [dep for dep in deps if dep not in accepted]
        state = "runnable" if is_runnable(manifest, task_id) else "waiting"
        if task.get("status") in {"accepted", "assigned", "submitted", "blocked"}:
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
            note = f"Accepted by reviewer. See {rel(review_path)}."
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
    mark_status(manifest, args.task_id, "accepted", args.note)
    append_lead_log(manifest, args.task_id, "accepted", args.note)
    save_manifest(manifest)
    print(f"accepted {args.task_id}")
    return 0


def cmd_fail(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    mark_status(manifest, args.task_id, "needs_fix", args.note)
    save_manifest(manifest)
    print(f"needs_fix {args.task_id}")
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

    p = sub.add_parser("fail", help="Return a task for fixes.")
    p.add_argument("task_id")
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_fail)

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
