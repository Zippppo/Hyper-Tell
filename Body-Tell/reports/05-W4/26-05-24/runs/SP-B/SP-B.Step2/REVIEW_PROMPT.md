You are the reviewer subagent for the Body-Tell workflow.

Review task: SP-B.Step2 - Implement patch crop and patch-level 2 positive plus 1 negative prompts

Repository root:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell

Lead plan:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/lead.html

Sub-plan:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/bodytell_sampling_vocabulary_optimization_report.html

Plan authority:
- The task section in the source-of-truth sub-plan is authoritative.
- The acceptance criteria below are a synchronized manifest summary. If they conflict with the sub-plan, review against the sub-plan and record the conflict in REVIEW.html.
- Manager override after CFG.CanonicalPhase1: `Body-Tell/configs/phase1_voxtell_aligned.yaml` is the only active Phase1 VoxTell config. Reject any active code/test/script/config default that recreates or relies on `phase1_voxtell_body.yaml`.

Worker result:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-B.Step2/RESULT.html

Current manager note / previous failure context:
This is the official SP-B.Step2 review after an interrupted prior worker. `CFG.CanonicalPhase1` is accepted: Test quality pass, Targeted pytest pass, and phase1 Slurm gate job 37958 COMPLETED with exit code 0:0. Review only SP-B.Step2 behavior and ensure the implementation uses `phase1_voxtell_aligned.yaml` only. The worker reported true red evidence was not possible because inherited interrupted Step2 tests and implementation already passed before cleanup; judge whether the resulting regression tests are still meaningful.

Acceptance criteria:
- patch_size, foreground_oversample_prob live in phase1_voxtell_aligned.yaml (no hardcoded 192).
- Output patch shape equals configured patch_size.
- Foreground/random crop ratio is auditable and close to 85/15.
- Positive masks are not empty inside the sampled patch.
- Setting patch_size=[128,128,256] degrades sampler to whole-case behavior (regression path).

Suggested verification commands:
- conda run -n voxtell python -m pytest Body-Tell/tests/test_dataset_prompt_masks.py Body-Tell/tests/test_crop_s2i_dataset.py

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
  - Write the compact gate artifact to `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-B.Step2/GATE.html` or include the same fields in `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-B.Step2/REVIEW.html`.
  - Verdict may be `accept` only when the main job is `COMPLETED` with exit code `0:0`.
  - If the job fails because of code behavior, verdict must be `needs_fix`.
  - If Slurm is unavailable or the job cannot be submitted for external cluster reasons, verdict must be `blocked`.

Required output:
Write `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-B.Step2/REVIEW.html` as an HTML review containing:
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
