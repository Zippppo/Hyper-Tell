Implemented SP-A.P3 and wrote the required report: [RESULT.html](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-A.P3/RESULT.html).

Changed:
- [voxtell_body_model.py](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/body_tell/models/voxtell_body_model.py): replaced the custom decoder path with `VoxTellDecoder`, using `stages`, `transpconvs`, always-built `seg_layers`, and shared `decoder.encoder` state-dict structure.
- [test_model_shapes.py](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/tests/test_model_shapes.py): added five-output decoder coverage, decoder key coverage, always-built `seg_layers` coverage, and explicit stride-mismatch coverage.

Verification:
- `conda run -n voxtell python -m pytest Body-Tell/tests/test_model_shapes.py Body-Tell/tests/test_prompt_loss_backward.py`
- Result: `11 passed in 16.30s`
- Extra smoke: five decoder outputs flow through `PromptSegmentationLoss` and backward with finite loss.

Recommendation in the report: `ready_for_review`. The remaining explicit risk is data/config-side: strict VoxTell-style decoding now rejects non-stride-compatible volume sizes such as current `[129, 108, 256]` instead of interpolating.