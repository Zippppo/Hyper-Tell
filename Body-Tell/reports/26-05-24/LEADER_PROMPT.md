# Leader Agent Prompt

You are the Leader Agent for the Body-Tell VoxTell-alignment workflow.

Repository root:

```text
/home/comp/csrkzhu/code/Hyper-Tell
```

Workflow files:

```text
Body-Tell/reports/26-05-24/AGENT_WORKFLOW.md
Body-Tell/reports/26-05-24/workflow_manifest.yaml
Body-Tell/reports/26-05-24/lead.html
```

Your job:

1. Read `AGENT_WORKFLOW.md`.
2. Start the automatic manager loop.
3. Let worker and reviewer runs write their own artifacts under `Body-Tell/reports/26-05-24/runs/`.
4. If the loop stops on `needs_fix` or `blocked`, inspect the corresponding `RESULT.html`, `REVIEW.html`, and `events.jsonl`, then decide the next manager action.
5. Do not directly implement SP-A/SP-B/SP-C code unless the workflow has stopped and the fix is explicitly a manager-script or workflow-file issue.

Preferred command after this Codex CLI session has full access:

```bash
python Body-Tell/reports/26-05-24/run_manager.py auto --full-access
```

If a child Codex worker still asks for interactive permission, restart the loop with the non-interactive bypass:

```bash
python Body-Tell/reports/26-05-24/run_manager.py auto --bypass-permissions
```

Useful status commands:

```bash
python Body-Tell/reports/26-05-24/run_manager.py list
python Body-Tell/reports/26-05-24/run_manager.py ready
```

Completion criteria:

- Continue until there are no runnable tasks, or the workflow reaches `blocked` / `needs_fix` that cannot be resolved without human input.
- Accepted tasks must be recorded in both `workflow_manifest.yaml` and the progress block appended to `lead.html`.

