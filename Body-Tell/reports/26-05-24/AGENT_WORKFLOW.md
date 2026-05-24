# Body-Tell Agent Workflow

This directory turns the existing four HTML plans into an executable workflow for long-running Codex work.

## Files

- `lead.html`: human-readable lead plan and final progress log. Only the manager may edit it.
- `workflow_manifest.yaml`: machine-readable task graph, owners, dependencies, status, and acceptance criteria.
- `LEADER_PROMPT.md`: prompt for running one interactive Codex CLI session as the leader.
- `bodytell_voxtell_alignment_plan_2026-05-24.html`: SP-A worker plan.
- `bodytell_sampling_vocabulary_optimization_report.html`: SP-B worker plan.
- `bodytell_training_loop_eval_alignment_plan_2026-05-24.html`: SP-C worker plan.
- `runs/<TASK_ID>/PROMPT.md`: exact prompt sent to the worker.
- `runs/<TASK_ID>/RESULT.html`: worker completion report.
- `runs/<TASK_ID>/REVIEW.html`: reviewer/manager gate report.
- `runs/<TASK_ID>/events.jsonl`: optional `codex --json` event stream.
- `runs/<TASK_ID>/codex-final.md`: final Codex worker message.

## Agents

### Manager

The manager owns `lead.html` and `workflow_manifest.yaml`.

Responsibilities:

- Choose tasks whose dependencies are accepted.
- Generate a worker prompt from the manifest and the corresponding sub-plan.
- Run or assign exactly one task at a time unless a human explicitly chooses parallel execution.
- Review `RESULT.html`, git diff, and verification output.
- Mark the task as `accepted`, `needs_fix`, or `blocked`.
- Append accepted progress to `lead.html`.

The manager must not implement SP-A/SP-B/SP-C code directly.

### SP-A Architecture Worker

Source of truth:

- `bodytell_voxtell_alignment_plan_2026-05-24.html`

Primary code areas:

- `Body-Tell/body_tell/models/**`
- `Body-Tell/body_tell/losses/**`
- `Body-Tell/configs/**`
- `Body-Tell/scripts/**`
- `Body-Tell/tests/**`

### SP-B Sampling/Vocabulary Worker

Source of truth:

- `bodytell_sampling_vocabulary_optimization_report.html`

Primary code areas:

- `Body-Tell/body_tell/data/**`
- `Body-Tell/configs/**`
- `Body-Tell/scripts/**`
- `Body-Tell/tests/**`

### SP-C Training/Eval Worker

Source of truth:

- `bodytell_training_loop_eval_alignment_plan_2026-05-24.html`

Primary code areas:

- `Body-Tell/train.py`
- `Body-Tell/body_tell/losses/**`
- `Body-Tell/body_tell/metrics/**`
- `Body-Tell/configs/**`
- `Body-Tell/tests/**`

### Reviewer

The reviewer checks worker output and writes `REVIEW.html`. The reviewer may not edit implementation files or `lead.html`.

## State Machine

```text
todo -> ready -> assigned -> submitted -> accepted
                              -> needs_fix
                              -> blocked
```

Only the manager changes status. Workers write `RESULT.html`; they do not update status themselves.

## Worker Rules

Every worker prompt must enforce these rules:

- Use only the assigned task from the assigned sub-plan.
- Do not modify `lead.html` or `workflow_manifest.yaml`.
- Do not do adjacent refactors.
- Use `conda run -n voxtell ...` for verification commands.
- Write an HTML result report to the task result path.
- Include changed files, verification commands, observed outputs, blockers, and residual risks.

## Commands

List the task graph:

```bash
python Body-Tell/reports/26-05-24/run_manager.py list
```

Show runnable tasks:

```bash
python Body-Tell/reports/26-05-24/run_manager.py ready
```

Generate a worker prompt without running Codex:

```bash
python Body-Tell/reports/26-05-24/run_manager.py prompt SP-B.Step1
```

Run a worker through Codex CLI:

```bash
python Body-Tell/reports/26-05-24/run_manager.py run SP-B.Step1
```

Run the manager loop automatically:

```bash
python Body-Tell/reports/26-05-24/run_manager.py auto --full-access
```

If Codex still asks for interactive permission, use the stronger non-interactive bypass:

```bash
python Body-Tell/reports/26-05-24/run_manager.py auto --bypass-permissions
```

To run this through one interactive leader Codex session, open Codex CLI in the repository, set full access once, then give it `LEADER_PROMPT.md`.

Generate a review prompt:

```bash
python Body-Tell/reports/26-05-24/run_manager.py review-prompt SP-B.Step1
```

Accept a reviewed task and append progress to `lead.html`:

```bash
python Body-Tell/reports/26-05-24/run_manager.py accept SP-B.Step1 --note "Dynamic sampling accepted after dataset tests passed."
```

Return a task for fixes:

```bash
python Body-Tell/reports/26-05-24/run_manager.py fail SP-B.Step1 --note "Prompt RNG test is missing cross-epoch evidence."
```

## Parallelism

The safe default is sequential execution. If parallelism is needed, run at most one task per lane at a time:

- one SP-A task
- one SP-B task
- one SP-C task

Do not run two workers that can touch the same files in the same worktree. For real parallel execution, use separate git worktrees per worker branch and merge only after manager acceptance.
