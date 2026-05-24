You are sp_a_architecture_worker.

Assigned task: SP-A.P4 - Align projection, mask head, num_heads decision, and forward layout

Repository root:
/home/comp/csrkzhu/code/Hyper-Tell

Lead plan:
Body-Tell/reports/26-05-24/lead.html

Your source-of-truth sub-plan:
Body-Tell/reports/26-05-24/bodytell_voxtell_alignment_plan_2026-05-24.html

Task scope:
SP-A P4: execute path A or path B for num_heads, projection shape, mask fusion, and axis layout.

Current manager note / previous failure context:
Post-accept phase1 Slurm gate job 37719 failed after SP-A.P4 review acceptance; no commit made. Failure remains VoxTellDecoder stride-compatible input mismatch at stage 0: transposed output spatial (18, 14, 32) vs skip spatial (17, 14, 32). Rerun SP-A.P4 before advancing to SP-A.P5.

Dependencies already accepted:
- SP-A.P0
- SP-A.P3

Acceptance criteria:
- num_heads path decision is recorded in RESULT.html.
- Non-cubic input shape tests prove axes are not swapped.
- P6 skip whitelist candidate is explicit.

Suggested verification commands:
- conda run -n voxtell python -m pytest Body-Tell/tests/test_model_shapes.py

Global rules:
- Use conda environment `voxtell` for verification. Prefer commands like `conda run -n voxtell python -m pytest ...`.
- Keep all reports and run logs in HTML.
- Do not edit `Body-Tell/reports/26-05-24/lead.html`.
- Do not edit `Body-Tell/reports/26-05-24/workflow_manifest.yaml`.
- Do not modify unrelated code or perform adjacent refactors.
- If this is a rerun after `needs_fix` or `gate_failed`, directly address the current manager note and explain the fix in RESULT.html.
- If the assigned task is impossible, write the blocker clearly in the result report instead of expanding scope.

Required output:
Write `Body-Tell/reports/26-05-24/runs/SP-A.P4/RESULT.html` as an HTML report containing:
1. Task id and summary.
2. Files changed.
3. Verification commands run and important output.
4. Acceptance criteria checklist.
5. Blockers or residual risks.
6. Recommendation: `ready_for_review`, `needs_fix`, or `blocked`.

Start by reading the lead plan and the source-of-truth sub-plan. Then complete only the assigned task.
