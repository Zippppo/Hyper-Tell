# Body-Tell Agent Workflow

This directory defines a subagent-first workflow for the Body-Tell VoxTell-alignment plans. The main conversation is only the dispatcher; worker, reviewer, and diagnostic subagents do the work and write the artifacts.

## Files

- `lead.html`: human-readable plan and accepted progress log.
- `workflow_manifest.yaml`: task graph, owners, dependencies, status, and acceptance criteria.
- `LEADER_PROMPT.md`: short prompt for the dispatcher conversation.
- `bodytell_voxtell_alignment_plan_2026-05-24.html`: SP-A source-of-truth plan.
- `bodytell_sampling_vocabulary_optimization_report.html`: SP-B source-of-truth plan.
- `bodytell_training_loop_eval_alignment_plan_2026-05-24.html`: SP-C source-of-truth plan.
- `templates/worker_prompt.md`: worker subagent prompt template.
- `templates/review_prompt.md`: reviewer subagent prompt template.
- `runs/<TASK_ID>/PROMPT.md`: optional saved copy of the exact worker prompt.
- `runs/<TASK_ID>/RESULT.html`: worker report.
- `runs/<TASK_ID>/REVIEW.html`: reviewer report and gate verdict.
- `runs/<TASK_ID>/GATE.html`: optional compact gate evidence when separated from `REVIEW.html`.
- `runs/archive/<TASK_ID>/...`: archived accepted artifacts.

Legacy `events.jsonl` and `review-events.jsonl` files may exist from the old CLI-manager workflow. They are not authoritative. Do not read them into the dispatcher context; use a diagnostic subagent if legacy forensics are required.

## Dispatcher Loop

1. Read only `workflow_manifest.yaml` and targeted sections of `lead.html` or the relevant plan.
2. Select a `ready` task whose dependencies are accepted.
3. Save the worker prompt to `runs/<TASK_ID>/PROMPT.md` if an audit trail is useful.
4. Spawn one worker subagent with the task id, owner, plan file, scope, acceptance criteria, verification commands, current manager note, writable paths, and TDD requirements.
5. Wait for the worker to write `runs/<TASK_ID>/RESULT.html` with test-first evidence and return a compact summary.
6. Spawn one reviewer subagent with the result path, acceptance criteria, relevant diff scope, TDD test-quality requirements, and gate requirements.
7. Wait for the reviewer to write `runs/<TASK_ID>/REVIEW.html` and return `accept`, `needs_fix`, or `blocked`.
8. Accept only when the reviewer records passing test quality, passing targeted pytest, and a passing phase1 Slurm gate.
9. Update `workflow_manifest.yaml` and append compact progress to `lead.html`.
10. If the verdict is `needs_fix` or `gate_failed`, rerun the same task with the reviewer note.
11. If the verdict is `blocked`, stop and report the blocker.

## State Machine

```text
todo -> ready -> assigned -> submitted -> accepted
                              -> needs_fix
                              -> gate_failed
                              -> blocked
```

Only the dispatcher changes task status. Workers write `RESULT.html`; reviewers write `REVIEW.html`; neither updates `workflow_manifest.yaml` or `lead.html`.

## Roles

### Dispatcher

- Owns `lead.html` and `workflow_manifest.yaml`.
- Selects runnable tasks.
- Spawns worker, reviewer, and diagnostic subagents.
- Records accepted progress.
- Does not implement SP-A/SP-B/SP-C code.
- Does not perform full reviews in the main context.

### SP-A Architecture Worker

- Source of truth: `bodytell_voxtell_alignment_plan_2026-05-24.html`
- Primary writable areas:
  - `Body-Tell/body_tell/models/**`
  - `Body-Tell/body_tell/losses/**`
  - `Body-Tell/configs/**`
  - `Body-Tell/scripts/**`
  - `Body-Tell/tests/**`

### SP-B Sampling/Vocabulary Worker

- Source of truth: `bodytell_sampling_vocabulary_optimization_report.html`
- Primary writable areas:
  - `Body-Tell/body_tell/data/**`
  - `Body-Tell/configs/**`
  - `Body-Tell/scripts/**`
  - `Body-Tell/tests/**`

### SP-C Training/Eval Worker

- Source of truth: `bodytell_training_loop_eval_alignment_plan_2026-05-24.html`
- Primary writable areas:
  - `Body-Tell/train.py`
  - `Body-Tell/body_tell/losses/**`
  - `Body-Tell/body_tell/metrics/**`
  - `Body-Tell/configs/**`
  - `Body-Tell/tests/**`

### Reviewer

- Reviews the worker result, relevant diff, verification evidence, and acceptance criteria.
- Does not edit implementation files, `lead.html`, or `workflow_manifest.yaml`.
- Reviews test quality before implementation details.
- Must run the required phase1 Slurm gate before returning `accept`.

## TDD Development Contract

Every implementation task is test-first by default.

Worker sequence:

1. Read the assigned acceptance criteria and identify the focused regression behavior.
2. Add or update the smallest meaningful pytest/regression test for that behavior.
3. Run the focused test before the implementation change and record red evidence. If a true red run is impossible because the behavior is already covered or the task is a pure audit, explain that explicitly in `RESULT.html`.
4. Implement the smallest task-scoped code change.
5. Run the focused pytest target again and record green evidence.
6. Run any broader suggested verification that is cheap enough for the task.
7. Write `RESULT.html` with test files, red evidence, green evidence, changed implementation files, acceptance checklist, blockers, and residual risk.

Reviewer sequence:

1. Review tests before implementation.
2. Confirm the tests map to the acceptance criteria and would fail for the relevant regression.
3. Reject tests that are tautological, only check superficial shapes when semantics are required, mock away the core behavior, or merely assert the implementation's current constants.
4. Run or verify the focused pytest target and record `Targeted pytest: pass`, `failed`, or `blocked`.
5. Review the implementation diff only after test quality is acceptable.
6. Run the phase1 Slurm gate last.

Acceptance requires all of:

- `Test quality: pass`.
- `Targeted pytest: pass`.
- `Phase1 Slurm gate: pass`.

## Required Gate

Every accepted task requires:

```bash
sbatch slurm/body-tell-phase1.sh
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed -n
```

`REVIEW.html` or `GATE.html` must record:

- `Phase1 Slurm gate: pass`, `failed`, or `blocked`.
- `Slurm command: sbatch slurm/body-tell-phase1.sh`.
- `Slurm job id: <JOB_ID>`.
- `sacct final state: <STATE>`.
- `sacct exit code: <EXIT_CODE>`.
- The smallest relevant failure excerpt, only when failed.

A task may be accepted only when test quality passes, targeted pytest passes, and the main Slurm job is `COMPLETED` with exit code `0:0`. If the reviewer says `accept` without this evidence, record `gate_failed` and rerun the same task with the missing-evidence note.

## Worker Rules

Every worker prompt must enforce:

- Use only the assigned task from the assigned sub-plan.
- You are not alone in the codebase; do not revert unrelated edits.
- Read the current manager note; if rerun after `needs_fix` or `gate_failed`, address it directly.
- Follow the TDD sequence: focused test first, red evidence, implementation, green evidence.
- Do not edit `lead.html` or `workflow_manifest.yaml`.
- Do not do adjacent refactors.
- Use `conda run -n voxtell ...` for verification commands.
- Write an HTML result report to the requested result path.
- Include test files, red/green evidence, changed implementation files, verification commands, important output, blockers, and residual risks.

## Context Rules

- Keep dispatcher summaries under 10 lines whenever possible.
- Subagents return only verdict, artifact paths, changed or inspected files, commands run, blockers, and next action.
- Do not paste full diffs, full HTML reports, full Slurm logs, or long test logs into the dispatcher context.
- On `needs_fix`, `gate_failed`, or `blocked`, inspect targeted sections of `RESULT.html`, `REVIEW.html`, or `GATE.html` only.
- If raw event-log forensics are needed, spawn a short-lived diagnostic subagent and bring back only the conclusion plus the smallest relevant excerpt.

## Parallelism

The safe default is one worker/reviewer cycle at a time. If the user explicitly asks for parallel work, run at most one task per lane:

- one SP-A task
- one SP-B task
- one SP-C task

Do not run two workers that can touch the same files in the same worktree. For true parallel implementation, use separate git worktrees per worker branch and merge only after review and gate acceptance.
