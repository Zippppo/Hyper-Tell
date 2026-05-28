Implemented and reported SP-A.P5-P6-lite.

Updated:
- [voxtell_body_model.py](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/body_tell/models/voxtell_body_model.py): HWD flatten contract, dtype-safe runtime position encoding, strict `transformer_decoder.*` prefix loader.
- [train.py](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/train.py): `--init-transformer-from-voxtell`, `--amp-dtype`, `--grad-clip-norm`, metadata logging/checkpoint config.
- [test_model_shapes.py](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/tests/test_model_shapes.py): non-cubic HWD contract, CUDA bfloat16 smoke, loader filtering coverage.
- [audit_voxtell_alignment.py](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/scripts/audit_voxtell_alignment.py): prefix audit/reporting.

Wrote the required HTML result report:
[RESULT.html](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-A.P5-P6-lite/RESULT.html)

Verification passed:
- `pytest Body-Tell/tests/test_model_shapes.py`: `11 passed`
- VoxTell prefix audit: `PASS`, 110 tensors, 251,858,944 params, 0 missing/unexpected/shape mismatches
- CUDA bfloat16 real aligned-model prefix smoke: `PASS`
- Slurm gate `37779`: `COMPLETED`, exit code `0:0`, elapsed `00:43:45`
- `git diff --check`: passed

Recommendation in the report: `ready_for_review`.