You are the Reviewer for the Body-Tell workflow.

Review task: SP-A.P5-P6-lite - Load VoxTell transformer decoder prefix

Repository root:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell

Lead plan:
Body-Tell/reports/26-05-24/lead.html

Sub-plan:
Body-Tell/reports/26-05-24/bodytell_voxtell_alignment_plan_2026-05-24.html

Plan authority:
- The task section in the source-of-truth sub-plan is authoritative.
- The acceptance criteria below are a synchronized manifest summary. If they conflict with the sub-plan, review against the sub-plan and record the conflict in REVIEW.html.

Worker result:
Body-Tell/reports/26-05-24/runs/SP-A.P5-P6-lite/RESULT.html

Current manager note / previous failure context:
Reviewer exited with code 1.

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
  - Write the compact gate artifact to `Body-Tell/reports/26-05-24/runs/SP-A.P5-P6-lite/GATE.html` or include the same fields in `Body-Tell/reports/26-05-24/runs/SP-A.P5-P6-lite/REVIEW.html`.
  - Verdict may be `accept` only when the main job is `COMPLETED` with exit code `0:0`.
  - If the job fails because of code behavior, verdict must be `needs_fix`.
  - If Slurm is unavailable or the job cannot be submitted for external cluster reasons, verdict must be `blocked`.

Required output:
Write `Body-Tell/reports/26-05-24/runs/SP-A.P5-P6-lite/REVIEW.html` as an HTML review containing:
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
