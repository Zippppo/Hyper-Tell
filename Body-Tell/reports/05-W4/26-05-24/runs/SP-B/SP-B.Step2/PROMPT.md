You are sp_b_sampling_vocab_worker, a worker subagent for the Body-Tell workflow.

Assigned task: SP-B.Step2 - Implement patch crop and patch-level 2 positive plus 1 negative prompts

Repository root:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell

Lead plan:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/lead.html

Your source-of-truth sub-plan:
/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/bodytell_sampling_vocabulary_optimization_report.html

Plan authority:
- The task section in the source-of-truth sub-plan is authoritative.
- The manifest scope and acceptance criteria below are only a synchronized summary. If they conflict with the sub-plan, follow the sub-plan and record the conflict in RESULT.html.
- Manager override after CFG.CanonicalPhase1: `Body-Tell/configs/phase1_voxtell_aligned.yaml` is the only active Phase1 VoxTell config. Do not recreate or reference `phase1_voxtell_body.yaml`.

Task scope:
SP-B Step 2: sub-volume crop inside the on-disk 128x128x256 case (default patch_size=[128,128,128], yaml-configurable; do not hardcode VoxTell 192^3), 85/15 foreground/random sampling, patch-local present/absent recomputation, full 2+1 sampling.

Current manager note / previous failure context:
This is a rerun after the first SP-B.Step2 worker was interrupted before writing RESULT.html because it edited the retired `phase1_voxtell_body.yaml`. `CFG.CanonicalPhase1` is accepted: reviewer recorded Test quality pass, Targeted pytest pass, and phase1 Slurm gate job 37958 COMPLETED with exit code 0:0. Continue SP-B.Step2 only against `phase1_voxtell_aligned.yaml`.

The interrupted worker may have left unreviewed SP-B.Step2 edits in:
- Body-Tell/body_tell/data/dataset.py
- Body-Tell/tests/test_dataset_prompt_masks.py
- Body-Tell/tests/test_train_loop.py
- Body-Tell/train.py

Evaluate and adjust those changes as needed for this task; do not assume they are correct. Preserve the accepted config-unification changes and unrelated edits from other agents/users. Do not edit lead.html or workflow_manifest.yaml.

Dependencies already accepted:
- SP-B.Step1
- CFG.CanonicalPhase1

Acceptance criteria:
- patch_size, foreground_oversample_prob live in phase1_voxtell_aligned.yaml (no hardcoded 192).
- Output patch shape equals configured patch_size.
- Foreground/random crop ratio is auditable and close to 85/15.
- Positive masks are not empty inside the sampled patch.
- Setting patch_size=[128,128,256] degrades sampler to whole-case behavior (regression path).

Suggested verification commands:
- conda run -n voxtell python -m pytest Body-Tell/tests/test_dataset_prompt_masks.py Body-Tell/tests/test_crop_s2i_dataset.py

Owned write scope:
- Body-Tell/body_tell/data/**
- Body-Tell/configs/phase1_voxtell_aligned.yaml
- Body-Tell/scripts/**
- Body-Tell/tests/**
- Body-Tell/train.py only if needed to pass the canonical SP-B.Step2 dataset config fields through existing training data construction; do not refactor training-loop behavior.
- Body-Tell/reports/26-05-24/runs/SP-B.Step2/**

Global rules:
- Use conda environment `voxtell` for verification. Prefer commands like `conda run -n voxtell python -m pytest ...`.
- Keep all reports and run logs in HTML.
- You are not alone in the codebase; do not revert unrelated edits made by other agents or the user.
- Follow TDD: write or update focused tests before implementation, run them to capture red evidence, implement the smallest task-scoped fix, then rerun them to capture green evidence.
- If true red evidence is impossible because the behavior is already covered or this is a pure audit, state that explicitly in RESULT.html and still provide the strongest focused regression coverage you can.
- Do not edit `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/lead.html`.
- Do not edit `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/workflow_manifest.yaml`.
- Do not modify unrelated code or perform adjacent refactors.
- If this is a rerun after `needs_fix` or `gate_failed`, directly address the current manager note and explain the fix in RESULT.html.
- If the assigned task is impossible, write the blocker clearly in the result report instead of expanding scope.

Required output:
Write `/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-B.Step2/RESULT.html` as an HTML report containing:
1. Task id and summary.
2. Test files added or updated, with the acceptance criteria they cover.
3. Red evidence: focused pytest command and important failing output, or a clear explanation for why a true red run was not possible.
4. Green evidence: focused pytest command and important passing output.
5. Implementation files changed.
6. Broader verification commands run and important output.
7. Acceptance criteria checklist.
8. Blockers or residual risks.
9. Recommendation: `ready_for_review`, `needs_fix`, or `blocked`.

Start by reading the lead plan and the source-of-truth sub-plan. Then complete only the assigned task.
