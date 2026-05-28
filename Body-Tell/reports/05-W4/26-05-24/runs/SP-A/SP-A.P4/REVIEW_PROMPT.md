You are the Reviewer for the Body-Tell workflow.

Review task: SP-A.P4 - Align projection, mask head, num_heads decision, and forward layout

Repository root:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell

Lead plan:
Body-Tell/reports/26-05-24/lead.html

Sub-plan:
Body-Tell/reports/26-05-24/bodytell_voxtell_alignment_plan_2026-05-24.html

Worker result:
Body-Tell/reports/26-05-24/runs/SP-A.P4/RESULT.html

Current manager note / previous failure context:
Post-accept phase1 Slurm gate job 37719 failed after SP-A.P4 review acceptance; no commit made. Failure remains VoxTellDecoder stride-compatible input mismatch at stage 0: transposed output spatial (18, 14, 32) vs skip spatial (17, 14, 32). Rerun SP-A.P4 before advancing to SP-A.P5.

Acceptance criteria:
- num_heads path decision is recorded in RESULT.html.
- Non-cubic input shape tests prove axes are not swapped.
- P6 skip whitelist candidate is explicit.

Suggested verification commands:
- conda run -n voxtell python -m pytest Body-Tell/tests/test_model_shapes.py

Rules:
- Do not edit implementation files.
- Do not edit `Body-Tell/reports/26-05-24/lead.html`.
- Do not edit `Body-Tell/reports/26-05-24/workflow_manifest.yaml`.
- Inspect the worker result, relevant git diff, and verification evidence.
- Use `conda run -n voxtell ...` if running verification.
- Run the required phase1 Slurm gate before accepting:
  - Submit `sbatch slurm/body-tell-phase1.sh`.
  - Wait for the job to leave the queue.
  - Record final accounting with `sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed -n`.
  - Write the compact gate artifact to `Body-Tell/reports/26-05-24/runs/SP-A.P4/GATE.html` or include the same fields in `Body-Tell/reports/26-05-24/runs/SP-A.P4/REVIEW.html`.
  - Verdict may be `accept` only when the main job is `COMPLETED` with exit code `0:0`.
  - If the job fails because of code behavior, verdict must be `needs_fix`.
  - If Slurm is unavailable or the job cannot be submitted for external cluster reasons, verdict must be `blocked`.

Required output:
Write `Body-Tell/reports/26-05-24/runs/SP-A.P4/REVIEW.html` as an HTML review containing:
1. Verdict: `accept`, `needs_fix`, or `blocked`.
2. Evidence for each acceptance criterion.
3. Commands run and important output.
4. Required fix list if not accepted.
5. Any residual risk the manager should record.
6. A phase1 Slurm gate section with these exact fields:
   - `Phase1 Slurm gate: pass`, `failed`, or `blocked`.
   - `Slurm command: sbatch slurm/body-tell-phase1.sh`.
   - `Slurm job id: <JOB_ID>`.
   - `sacct final state: <STATE>`.
   - `sacct exit code: <EXIT_CODE>`.
   - For failures, include only the smallest relevant stderr/log excerpt needed to identify the root cause.
