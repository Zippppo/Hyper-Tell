You are sp_a_architecture_worker.

Assigned task: SP-A.P2 - Replace encoder with VoxTell residual encoder

Repository root:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell

Lead plan:
Body-Tell/reports/26-05-24/lead.html

Your source-of-truth sub-plan:
Body-Tell/reports/26-05-24/bodytell_voxtell_alignment_plan_2026-05-24.html

Task scope:
SP-A P2: residual encoder channels, blocks, norm, activation, and bias policy.

Dependencies already accepted:
- SP-A.P1

Acceptance criteria:
- Encoder channel/block structure matches the SP-A plan or records an explicit domain-exemption decision.
- Existing model shape tests are updated and passing.

Suggested verification commands:
- conda run -n voxtell python -m pytest Body-Tell/tests/test_model_shapes.py

Global rules:
- Use conda environment `voxtell` for verification. Prefer commands like `conda run -n voxtell python -m pytest ...`.
- Keep all reports and run logs in HTML.
- Do not edit `Body-Tell/reports/26-05-24/lead.html`.
- Do not edit `Body-Tell/reports/26-05-24/workflow_manifest.yaml`.
- Do not modify unrelated code or perform adjacent refactors.
- If the assigned task is impossible, write the blocker clearly in the result report instead of expanding scope.

Required output:
Write `Body-Tell/reports/26-05-24/runs/SP-A.P2/RESULT.html` as an HTML report containing:
1. Task id and summary.
2. Files changed.
3. Verification commands run and important output.
4. Acceptance criteria checklist.
5. Blockers or residual risks.
6. Recommendation: `ready_for_review`, `needs_fix`, or `blocked`.

Start by reading the lead plan and the source-of-truth sub-plan. Then complete only the assigned task.

