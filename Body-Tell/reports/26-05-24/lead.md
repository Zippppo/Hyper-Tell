# Body-Tell Training Loop/Eval Alignment Progress

Updated: 2026-05-28T16:44:21+08:00

## Current status

- T1 - Clarify deep_supervision model output contract: accepted.
- T2 - Fix prompt metrics foreground Dice aggregation: accepted.
- T3 - Unify training and inference model construction: accepted.
- T4 - Reduce DDP metrics before logging and best checkpoint decisions: accepted.
- Plan status: complete. No ready or todo tasks remain.

## T1 accepted evidence

- Reviewer verdict: accept.
- Test quality: pass.
- Focused pytest: pass, 3 passed.
- Required pytest gate: pass, 15 passed for `Body-Tell/tests/test_train_loop.py`, `Body-Tell/tests/test_prompt_loss_backward.py`, and `Body-Tell/tests/test_inference_paths.py`.
- Training smoke: pass for `bash h20b-sh-scripts/test-train.sh`; reviewer evidence shows the real training entry reached `epoch 1: 89%`, satisfying the source-plan smoke requirement without requiring full training completion.

## T2 accepted evidence

- Reviewer verdict: accept.
- Test quality: pass.
- Focused pytest: pass, 1 passed for `Body-Tell/tests/test_prompt_loss_backward.py::test_prompt_metrics_nan_safe_foreground_dice_aggregation`.
- Required pytest gate: pass, 16 passed for `Body-Tell/tests/test_train_loop.py`, `Body-Tell/tests/test_prompt_loss_backward.py`, and `Body-Tell/tests/test_inference_paths.py`.
- Training smoke: pass for `bash h20b-sh-scripts/test-train.sh` under the `voxtell` environment; reviewer evidence shows the real training entry loaded config, built train/val datasets, built the model, entered epoch 1, and reached `7/1259` before intentional smoke interruption.

## T3 accepted evidence

- Reviewer verdict: accept.
- Test quality: pass.
- Focused pytest: pass, 1 passed for `Body-Tell/tests/test_inference_paths.py::test_train_and_inference_build_model_match_aligned_config`.
- Required pytest gate: pass, 17 passed for `Body-Tell/tests/test_train_loop.py`, `Body-Tell/tests/test_prompt_loss_backward.py`, and `Body-Tell/tests/test_inference_paths.py`.
- Training smoke: pass for `bash h20b-sh-scripts/test-train.sh` under the `voxtell` environment; reviewer evidence shows the real training entry loaded config, built datasets, built the model, initialized the training flow, and reached `epoch 1 ... 41/1259` before intentional smoke interruption.

## T4 accepted evidence

- Reviewer verdict: accept.
- Test quality: pass.
- Focused pytest: pass, 1 passed for `Body-Tell/tests/test_train_loop.py::test_ddp_metric_reduce_helper`.
- Required pytest gate: pass, 18 passed for `Body-Tell/tests/test_train_loop.py`, `Body-Tell/tests/test_prompt_loss_backward.py`, and `Body-Tell/tests/test_inference_paths.py`.
- Training smoke: pass for `bash h20b-sh-scripts/test-train.sh` under the `voxtell` environment; reviewer evidence shows the real training entry entered DDP training, built train/val datasets, built the model, and reached `epoch 1 ... 18/1259` before intentional smoke interruption.

## Next ready tasks

- None.

## Notes

- The old generic Phase1 Slurm gate from the template does not apply to this plan.
- Local changes to `h20b-sh-scripts/test-train.sh` are not part of the T1, T2, T3, or T4 acceptance commits and are left untouched.
- Local ignored `.codex/SKILL/workflow.md` routing documentation is outside the T4 acceptance commit scope.
