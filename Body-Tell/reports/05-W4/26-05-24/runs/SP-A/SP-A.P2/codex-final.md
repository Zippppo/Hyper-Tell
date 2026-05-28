Implemented SP-A.P2 and wrote the required report at `Body-Tell/reports/26-05-24/runs/SP-A.P2/RESULT.html`.

Changed:
- `Body-Tell/body_tell/models/voxtell_body_model.py`
- `Body-Tell/train.py`
- `Body-Tell/tests/test_model_shapes.py`
- `Body-Tell/configs/phase1_voxtell_aligned.yaml`
- `Body-Tell/reports/26-05-24/runs/SP-A.P2/RESULT.html`

Verification passed:

```text
conda run -n voxtell python -m pytest Body-Tell/tests/test_model_shapes.py
5 passed in 15.74s
```

Also checked full SP-A encoder metadata: channels `(32, 64, 128, 256, 320, 320)`, blocks `(1, 3, 4, 6, 6, 6)`, params `90296576`.

I did not edit `lead.html` or `workflow_manifest.yaml`. Note that several report workflow files were already dirty in the worktree before my changes, including `workflow_manifest.yaml`; I left them untouched.