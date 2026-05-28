Completed SP-A.P1 and wrote the report at [RESULT.html](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-A.P1/RESULT.html:1).

Changed [transformer.py](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/body_tell/models/transformer.py:137) so the active pre-norm decoder path matches VoxTell: `norm2` cross-attention, then `norm3` FFN, with self-attention parameters preserved but not executed. Changed [voxtell_body_model.py](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/body_tell/models/voxtell_body_model.py:69) to construct the transformer layer with ReLU. Updated the P0 golden parity test in [test_voxtell_transformer_compat.py](/home/xiaoqingguo/Rongkun/code/Hyper-Tell/Body-Tell/tests/test_voxtell_transformer_compat.py:66) from expected-fail to active validation.

Verification passed:

```bash
conda run -n voxtell python -m pytest Body-Tell/tests/test_voxtell_transformer_compat.py
# 1 passed in 13.90s
```

Recommendation in the report: `ready_for_review`. I did not edit the protected `lead.html` or `workflow_manifest.yaml`; they were already dirty in the worktree.