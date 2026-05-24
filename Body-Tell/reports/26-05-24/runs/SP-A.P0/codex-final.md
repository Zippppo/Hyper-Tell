Implemented SP-A.P0 and wrote the required report: [RESULT.html](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-A.P0/RESULT.html).

Changed:
- [audit_voxtell_alignment.py](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/scripts/audit_voxtell_alignment.py)
- [test_voxtell_transformer_compat.py](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/tests/test_voxtell_transformer_compat.py)
- [audit_voxtell_alignment.html](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-A.P0/audit_voxtell_alignment.html)
- [RESULT.html](/home/comp/csrkzhu/code/Hyper-Tell/Body-Tell/reports/26-05-24/runs/SP-A.P0/RESULT.html)

Verification passed in `conda run -n voxtell`:
- `audit_voxtell_alignment.py --help`: OK
- `py_compile`: OK
- audit generation: OK
- transformer golden test: `1 xfailed` as intended for SP-A.P1
- existing model shape tests: `3 passed`

Key finding: VoxTell source default is `num_heads=1`, but the shipped v1.1 predictor and checkpoint are `num_heads=32`, with checkpoint projection outputs `[32, 2048, 4096, 8192, 10240]`. The result report recommends `ready_for_review`.