You are sp_a_architecture_worker.

Assigned task: SP-A.P5-P6-lite - Load VoxTell transformer decoder prefix

Repository root:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell

Lead plan:
Body-Tell/reports/26-05-24/lead.html

Your source-of-truth sub-plan:
Body-Tell/reports/26-05-24/bodytell_voxtell_alignment_plan_2026-05-24.html

Plan authority:
- The task section in the source-of-truth sub-plan is authoritative.
- The manifest scope and acceptance criteria below are only a synchronized summary. If they conflict with the sub-plan, follow the sub-plan and record the conflict in RESULT.html.

Task scope:
SP-A P5/P6-lite: runtime HWD position/memory contract plus --init-transformer-from-voxtell transformer_decoder.* prefix loading.

Current manager note / previous failure context:
Reviewer verdict: needs_fix. See Body-Tell/reports/26-05-24/runs/SP-A.P5-P6-lite/REVIEW.html.

Dependencies already accepted:
- SP-A.P4

Acceptance criteria:
- P5 is not an independent pos_embed implementation stage.
- Do not implement fixed_voxtell_192 and do not add a pos_embed state-dict loading path.
- Runtime position encoding is generated from the selected feature's actual H,W,D and matches memory token count, batch, dtype, device, and channel dimension.
- Non-cubic input coverage verifies the VoxTell-compatible flatten contract b c d h w -> b h w d c -> (h w d) b c.
- Loader reads fold_0/checkpoint_final.pth network_weights and filters only transformer_decoder.* keys.
- Filtered keys map exactly to model.transformer_decoder.state_dict() with no missing, unexpected, or shape mismatch entries.
- Loading uses model.transformer_decoder.load_state_dict(..., strict=True).
- Loader explicitly does not load encoder.*, decoder.*, project_bottleneck_embed.*, project_text_embed.*, project_to_decoder_channels.*, or pos_embed; no full-model whitelist is used.
- Logs and saved checkpoint config record init_transformer_from_voxtell metadata: checkpoint path, prefix, loaded tensor count, and loaded parameter count.
- CPU prefix audit passes; at least one forward/loss/backward smoke passes if GPU is available.

Suggested verification commands:
- conda run -n voxtell python -m pytest Body-Tell/tests/test_model_shapes.py
- conda run -n voxtell python Body-Tell/scripts/audit_voxtell_alignment.py --config Body-Tell/configs/phase1_voxtell_aligned.yaml

Global rules:
- Use conda environment `voxtell` for verification. Prefer commands like `conda run -n voxtell python -m pytest ...`.
- Keep all reports and run logs in HTML.
- Do not edit `Body-Tell/reports/26-05-24/lead.html`.
- Do not edit `Body-Tell/reports/26-05-24/workflow_manifest.yaml`.
- Do not modify unrelated code or perform adjacent refactors.
- If this is a rerun after `needs_fix` or `gate_failed`, directly address the current manager note and explain the fix in RESULT.html.
- If the assigned task is impossible, write the blocker clearly in the result report instead of expanding scope.

Required output:
Write `Body-Tell/reports/26-05-24/runs/SP-A.P5-P6-lite/RESULT.html` as an HTML report containing:
1. Task id and summary.
2. Files changed.
3. Verification commands run and important output.
4. Acceptance criteria checklist.
5. Blockers or residual risks.
6. Recommendation: `ready_for_review`, `needs_fix`, or `blocked`.

Start by reading the lead plan and the source-of-truth sub-plan. Then complete only the assigned task.
