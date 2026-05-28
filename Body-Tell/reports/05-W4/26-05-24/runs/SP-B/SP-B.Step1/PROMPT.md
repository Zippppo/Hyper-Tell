# SP-B.Step1 Effective Worker Prompt / Audit Trail

Worker: sp_b_sampling_vocab_worker

Repository root: /home/xiaoqingguo/Rongkun/code/Hyper-Tell

Controlling template read first:

- /home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/templates/worker_prompt.md

Expanded task:

- Task ID: SP-B.Step1
- Title: Add dynamic sampling RNG infrastructure
- Lead plan: Body-Tell/reports/26-05-24/lead.html
- Source-of-truth sub-plan: Body-Tell/reports/26-05-24/bodytell_sampling_vocabulary_optimization_report.html
- Scope: SP-B Step 1: dataset epoch/iteration-aware prompt RNG instead of fixed seed+index.
- Dependencies: none
- Current manager note: reviewer verdict `needs_fix`. Dataset-side criteria passed, and Phase1 Slurm gate already passed for reviewer job 37823. The authoritative sub-plan requires the training loop to call the dataset epoch setter each epoch. `Body-Tell/train.py` only called `train_sampler.set_epoch(epoch)` and did not call `train_dataset.set_epoch(epoch)` or handle smoke-mode `Subset.dataset`.

Acceptance criteria:

1. Dataset exposes set_epoch(epoch) or equivalent.
2. Same case can sample different prompt text across epochs.
3. DDP-compatible deterministic behavior is documented.

Verification command:

```bash
conda run -n voxtell python -m pytest Body-Tell/tests/test_dataset_prompt_masks.py
```

Global constraints applied:

- Use conda environment `voxtell` for verification.
- Keep reports and run logs in HTML.
- Do not revert unrelated edits by other agents or the user.
- Do not edit Body-Tell/reports/26-05-24/lead.html.
- Do not edit Body-Tell/reports/26-05-24/workflow_manifest.yaml.
- Keep changes limited to writable scopes:
  - Body-Tell/body_tell/data/**
  - Body-Tell/configs/**
  - Body-Tell/scripts/**
  - Body-Tell/tests/**
  - Body-Tell/reports/26-05-24/runs/**
- Narrow reviewer-requested exception: Body-Tell/train.py may be edited only to call the dataset epoch setter alongside sampler epoch handling, including smoke-mode torch.utils.data.Subset handling.

Audit notes:

- Read the lead plan and SP-B source-of-truth sub-plan before code edits.
- Found pre-existing modified worktree state in Body-Tell/reports/26-05-24/workflow_manifest.yaml and left it untouched.
- Rerun fix: wired Body-Tell/train.py to set both sampler epoch and underlying dataset epoch each training epoch. The helper unwraps smoke-mode torch.utils.data.Subset before calling set_epoch.
- Added a focused regression test that verifies sampler.set_epoch and the underlying dataset set_epoch are both updated when smoke mode wraps the dataset in Subset.
