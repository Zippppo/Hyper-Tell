You are the reviewer subagent for the Body-Tell workflow.

Review task: CFG.CanonicalPhase1 - Collapse Phase1 VoxTell training config ambiguity

Repository root:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell

Lead plan:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/lead.html

Sub-plan:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/bodytell_voxtell_alignment_plan_2026-05-24.html

Plan authority:
- SP-A accepted work established `Body-Tell/configs/phase1_voxtell_aligned.yaml` as the VoxTell-aligned Phase1 architecture/config path.
- This manager override resolves a cross-plan ambiguity before SP-B.Step2 continues: keep only one Phase1 VoxTell config file, with `phase1_voxtell_aligned.yaml` as canonical.
- If any plan text still references `phase1_voxtell_body.yaml`, treat that as stale and record the conflict in REVIEW.html.

Worker result:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/CFG.CanonicalPhase1/RESULT.html

Current manager note / previous failure context:
SP-B.Step2 was interrupted before RESULT.html because it edited `phase1_voxtell_body.yaml` while SP-A accepted work/tests rely on `phase1_voxtell_aligned.yaml`. Review whether the worker eliminated the active ambiguity by making `phase1_voxtell_aligned.yaml` the single canonical active Phase1 VoxTell config. Do not expand into SP-B.Step2 implementation review.

Acceptance criteria:
- `Body-Tell/configs/phase1_voxtell_aligned.yaml` is the single canonical Phase1 VoxTell training config.
- `Body-Tell/configs/phase1_voxtell_body.yaml` is removed from active use and no active code/test/script default still references it.
- Training, inference, audit scripts, and tests that need the Phase1 config point to `phase1_voxtell_aligned.yaml`.
- Canonical config preserves SP-A aligned architecture-sensitive fields such as encoder channels/block layout, avoiding regression to the old body config.
- Any SP-B.Step2 config-only fields already introduced by the interrupted worker, such as `patch_size` and `foreground_oversample_prob`, are present only in the canonical aligned config if they remain needed.

Suggested verification commands:
- conda run -n voxtell python -m pytest Body-Tell/tests/test_model_shapes.py Body-Tell/tests/test_train_loop.py
- conda run -n voxtell python Body-Tell/scripts/audit_voxtell_alignment.py --config Body-Tell/configs/phase1_voxtell_aligned.yaml
- find Body-Tell -type f \\( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' -o -name '*.sh' \\) -print | xargs grep -n "phase1_voxtell_body" 2>/dev/null

Rules:
- Do not edit implementation files.
- Do not edit `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/lead.html`.
- Do not edit `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/workflow_manifest.yaml`.
- Inspect tests before implementation. Test quality is a required acceptance gate.
- Confirm tests map to the acceptance criteria and would fail for the relevant regression.
- Reject tautological tests, tests that only assert superficial shapes when semantics are required, tests that mock away the core behavior, and tests that only assert current implementation constants.
- Inspect the worker result, test diff, implementation diff, and verification evidence.
- Use `conda run -n voxtell ...` if running verification.
- Run or verify the focused pytest target before the Slurm gate. Verdict may be `accept` only when test quality and targeted pytest pass.
- Run the required phase1 Slurm gate before accepting:
  - Submit `sbatch slurm/body-tell-phase1.sh`.
  - Wait for the job to leave the queue.
  - Record final accounting with `sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed -n`.
  - Write the compact gate artifact to `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/CFG.CanonicalPhase1/GATE.html` or include the same fields in `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/CFG.CanonicalPhase1/REVIEW.html`.
  - Verdict may be `accept` only when the main job is `COMPLETED` with exit code `0:0`.
  - If the job fails because of code behavior, verdict must be `needs_fix`.
  - If Slurm is unavailable or the job cannot be submitted for external cluster reasons, verdict must be `blocked`.

Required output:
Write `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/CFG.CanonicalPhase1/REVIEW.html` as an HTML review containing:
1. Verdict: `accept`, `needs_fix`, or `blocked`.
2. Test quality section with these exact fields:
   - `Test quality: pass`, `failed`, or `blocked`.
   - `Focused tests reviewed: <paths>`.
   - `Regression covered: <yes/no and reason>`.
   - `Red evidence reviewed: <yes/no and reason>`.
   - `Targeted pytest: pass`, `failed`, or `blocked`.
3. Evidence for each acceptance criterion.
4. Commands run and important output.
5. Required fix list if not accepted.
6. Any residual risk the manager should record.
7. A phase1 Slurm gate section with these exact fields:
   - `Phase1 Slurm gate: pass`, `failed`, or `blocked`.
   - `Slurm command: sbatch slurm/body-tell-phase1.sh`.
   - `Slurm job id: <JOB_ID>`.
   - `sacct final state: <STATE>`.
   - `sacct exit code: <EXIT_CODE>`.
   - For failures, include only the smallest relevant stderr/log excerpt needed to identify the root cause.
