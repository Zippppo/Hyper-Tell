# Body-Tell Agent Workflow

This directory defines a subagent-first workflow for the Body-Tell VoxTell-alignment plans. The main conversation is a pure dispatcher; worker, reviewer, post-accept, and diagnostic subagents do the work, write artifacts, update workflow documents, and commit accepted changes.

## Files

- `lead.html`: human-readable plan and accepted progress log.
- `workflow_manifest.yaml`: task graph, owners, dependencies, status, and acceptance criteria.
- `LEADER_PROMPT.md`: short prompt for the dispatcher conversation.
- `bodytell_voxtell_alignment_plan_2026-05-24.html`: SP-A source-of-truth plan.
- `bodytell_sampling_vocabulary_optimization_report.html`: SP-B source-of-truth plan.
- `bodytell_training_loop_eval_alignment_plan_2026-05-24.html`: SP-C source-of-truth plan.
- `templates/worker_prompt.md`: worker subagent prompt template.
- `templates/review_prompt.md`: reviewer subagent prompt template.
- `templates/post_accept_prompt.md`: post-accept documentation-sync and commit prompt template.
- `runs/<TASK_ID>/PROMPT.md`: optional saved copy of the exact worker prompt.
- `runs/<TASK_ID>/RESULT.html`: worker report.
- `runs/<TASK_ID>/REVIEW.html`: reviewer report and gate verdict.
- `runs/<TASK_ID>/GATE.html`: optional compact gate evidence when separated from `REVIEW.html`.
- `runs/<TASK_ID>/POST_ACCEPT.html`: post-accept documentation sync, manifest update, and commit evidence.
- `runs/archive/<TASK_ID>/...`: archived accepted artifacts.

Legacy `events.jsonl` and `review-events.jsonl` files may exist from the old CLI-manager workflow. They are not authoritative. Do not read them into the dispatcher context; use a diagnostic subagent if legacy forensics are required.

## Pure Dispatcher Loop

1. Read only `workflow_manifest.yaml`, `AGENT_WORKFLOW.md`, and small targeted snippets needed to choose the next action.
2. Select a `ready` task whose dependencies are accepted.
3. Spawn one worker subagent with the task id, owner, plan file, scope, acceptance criteria, verification commands, current manager note, writable paths, and TDD requirements. The worker follows `templates/worker_prompt.md` and writes `runs/<TASK_ID>/RESULT.html`.
4. Wait for the worker to return only compact status.
5. Spawn one reviewer subagent with the result path, acceptance criteria, relevant diff scope, TDD test-quality requirements, and gate requirements. The reviewer follows `templates/review_prompt.md` and writes `runs/<TASK_ID>/REVIEW.html`.
6. Wait for the reviewer to return only compact status.
7. Treat `accept` as eligible only when `REVIEW.html` records passing test quality, passing targeted pytest, and a phase1 Slurm main job `COMPLETED` with exit code `0:0`.
8. If eligible for accept, spawn one post-accept subagent. It follows `templates/post_accept_prompt.md`, verifies the review gate evidence, updates `workflow_manifest.yaml`, appends compact progress to `lead.html`, synchronizes all relevant reports under this directory, writes `runs/<TASK_ID>/POST_ACCEPT.html`, and creates a git commit.
9. Continue to the next `ready` task only after the post-accept subagent returns a commit hash.
10. If the verdict is `needs_fix` or `gate_failed`, rerun the same task with the reviewer note. Do not spawn post-accept and do not commit.
11. If the verdict is `blocked`, stop and report the blocker in no more than 10 lines.

The dispatcher must not write files, apply patches, update docs, update manifests, run commits, inspect full diffs, or read long logs. If a file mutation is required, it must be delegated to a bounded subagent.

## State Machine

```text
todo -> ready -> assigned -> submitted -> accepted
                              -> needs_fix
                              -> gate_failed
                              -> blocked
```

Only the post-accept subagent changes accepted task status, dependent ready states, `workflow_manifest.yaml`, `lead.html`, related reports, and git commits. Workers write `RESULT.html`; reviewers write `REVIEW.html`; neither updates workflow documents or commits. The main dispatcher never mutates files.

## Roles

### Dispatcher

- Selects runnable tasks.
- Spawns worker, reviewer, post-accept, and diagnostic subagents.
- Waits for compact status and decides the next dispatch action.
- Does not implement SP-A/SP-B/SP-C code.
- Does not perform full reviews in the main context.
- Does not edit files, update documentation, update manifests, or commit.

### Post-Accept Maintainer

- Source of truth: `templates/post_accept_prompt.md`.
- Runs only after reviewer `accept` and required gate evidence exists in `REVIEW.html`.
- Verifies `Test quality: pass`, `Targeted pytest: pass`, and phase1 Slurm main job `COMPLETED` with exit code `0:0`.
- Updates `workflow_manifest.yaml`, appends compact progress to `lead.html`, synchronizes all relevant reports in `Body-Tell/reports/26-05-24`, and writes `runs/<TASK_ID>/POST_ACCEPT.html`.
- Marks newly unblocked tasks `ready`.
- Creates a git commit containing the accepted implementation changes, review artifacts, workflow state, and documentation sync for that task.
- If accepted changes cannot be isolated from unrelated dirty files, writes `POST_ACCEPT.html` with `blocked` and does not commit.

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
- Scope intent: training control-plane and evaluation protocol work. Task 0/1 define budget, step semantics, logging, and resume clocks; they are not direct performance-improvement claims.
- Primary writable areas:
  - `Body-Tell/train.py`
  - `Body-Tell/body_tell/losses/**`
  - `Body-Tell/body_tell/metrics/**`
  - `Body-Tell/configs/**`
  - `Body-Tell/tests/**`

### Reviewer

- Reviews the worker result, relevant diff, verification evidence, and acceptance criteria.
- Does not edit implementation files, `lead.html`, or `workflow_manifest.yaml`.
- Does not synchronize documents and does not commit.
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
8. Do not update workflow documents and do not commit.

Reviewer sequence:

1. Review tests before implementation.
2. Confirm the tests map to the acceptance criteria and would fail for the relevant regression.
3. Reject tests that are tautological, only check superficial shapes when semantics are required, mock away the core behavior, or merely assert the implementation's current constants.
4. Run or verify the focused pytest target and record `Targeted pytest: pass`, `failed`, or `blocked`.
5. Review the implementation diff only after test quality is acceptable.
6. Run the phase1 Slurm gate last.
7. Do not update workflow documents and do not commit.

Acceptance requires all of:

- `Test quality: pass`.
- `Targeted pytest: pass`.
- `Phase1 Slurm gate: pass`.

Post-accept requires all of:

- Review verdict is `accept`.
- `REVIEW.html` records `Test quality: pass`.
- `REVIEW.html` records `Targeted pytest: pass`.
- `REVIEW.html` records `slurm/body-tell-phase1.sh` main job `COMPLETED` with exit code `0:0`.
- Documentation sync completed for all relevant `Body-Tell/reports/26-05-24` files.
- A git commit was created for the accepted task.

## Required Gate

Every reviewer accept requires:

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

A task may enter post-accept only when test quality passes, targeted pytest passes, and the main Slurm job is `COMPLETED` with exit code `0:0`. If the reviewer says `accept` without this evidence, treat it as `gate_failed` and rerun the same task with the missing-evidence note.

## Worker Rules

Every worker prompt must enforce:

- Use only the assigned task from the assigned sub-plan.
- You are not alone in the codebase; do not revert unrelated edits.
- Read the current manager note; if rerun after `needs_fix` or `gate_failed`, address it directly.
- Follow the TDD sequence: focused test first, red evidence, implementation, green evidence.
- Do not edit `lead.html` or `workflow_manifest.yaml`.
- Do not synchronize reports and do not commit.
- Do not do adjacent refactors.
- Use `conda run -n voxtell ...` for verification commands.
- Write an HTML result report to the requested result path.
- Include test files, red/green evidence, changed implementation files, verification commands, important output, blockers, and residual risks.

## Post-Accept Rules

Every post-accept prompt must enforce:

- Verify reviewer gate evidence directly from `REVIEW.html` before changing state.
- Do not edit implementation code.
- Update `workflow_manifest.yaml` for the accepted task, including `updated_at`, `last_note`, and `history`.
- Mark dependent tasks `ready` when all dependencies are accepted.
- Append compact progress to `lead.html`.
- Synchronize all relevant reports under `Body-Tell/reports/26-05-24` so current status, next tasks, accepted evidence paths, and config names are not stale.
- Keep historical progress entries factual; revise misleading present-tense "next" statements when they become stale.
- Run lightweight document consistency checks, including stale task/config greps and `git diff --check`.
- Commit only the accepted task changes, review artifacts, workflow state, documentation sync, and post-accept artifact. Leave unrelated dirty files untouched.
- Write `runs/<TASK_ID>/POST_ACCEPT.html` with gate evidence, updated docs, commit hash, residual risks, and next ready tasks.

## Context Rules

- Keep dispatcher summaries under 10 lines whenever possible.
- Subagents return only verdict, artifact paths, changed or inspected files, commands run, Slurm job id/state/exit code, commit hash when applicable, blockers, and next action.
- Do not paste full diffs, full HTML reports, full Slurm logs, or long test logs into the dispatcher context.
- On `needs_fix`, `gate_failed`, or `blocked`, inspect targeted sections of `RESULT.html`, `REVIEW.html`, `GATE.html`, or `POST_ACCEPT.html` only.
- If raw event-log forensics are needed, spawn a short-lived diagnostic subagent and bring back only the conclusion plus the smallest relevant excerpt.

## Parallelism

The safe default is one worker/reviewer/post-accept cycle at a time. If the user explicitly asks for parallel work, run at most one task per lane:

- one SP-A task
- one SP-B task
- one SP-C task

Do not run two workers that can touch the same files in the same worktree. For true parallel implementation, use separate git worktrees per worker branch and merge only after review and gate acceptance.
