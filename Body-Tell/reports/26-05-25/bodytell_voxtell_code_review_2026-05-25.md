# Body-Tell / VoxTell 对齐代码 Review

日期：2026-05-25  
范围：`main...HEAD` 中 Body-Tell 非报告代码改动，10 个文件，约 `+1939/-126`。本次 review 只读代码与日志，未修改业务代码。

参考材料：

- `Body-Tell/reports/26-05-24/bodytell_voxtell_alignment_plan_2026-05-24.html`
- `Body-Tell/reports/26-05-24/runs/SP-A.P0` 到 `SP-A.P5-P6-lite`
- 当前分支 `adjust-to-voxtell`

## 总体结论

当前实现基本围绕计划中的 SP-A 收敛：transformer decoder 语义与 VoxTell 对齐，P6-lite loader 严格限制在 `transformer_decoder.*`，runtime HWD flatten 与 position dtype/device 契约都有测试覆盖。没有发现明显的 `pdb`、`breakpoint`、临时 `TODO/FIXME`、调试打印遗留在运行时模型或训练主路径中。

主要问题不在功能正确性，而在工程收尾：

1. 新增运行时依赖没有写入根环境声明，fresh install 风险高。
2. 参数审计脚本在 shared encoder 结构下会误报 decoder 参数，且没有覆盖 6-stage residual encoder 的完整分解。
3. 审计/报告脚本体量偏大，HTML 渲染与审计逻辑混在一个 826 行脚本中，短期可接受，长期建议拆分。
4. 模型中存在若干“看似臃肿”的 unused 参数，但多数是 checkpoint key 兼容所需，不应简单删掉。

## Findings

### 1. High: 新增 `dynamic_network_architectures` 运行时依赖未声明

证据：

- `Body-Tell/body_tell/models/voxtell_body_model.py:107` 默认构造 `VoxTellDecoder`。
- `Body-Tell/body_tell/models/voxtell_body_model.py:329` 到 `:333` 中 `VoxTellDecoder` 无条件 import `dynamic_network_architectures`。
- `Body-Tell/body_tell/models/voxtell_body_model.py:255` 到 `:260` 中 residual encoder 也依赖该包。
- 根 `requirements.txt` 只包含 torch/numpy/scipy/sklearn/matplotlib/tqdm/pyyaml/wandb/pytest，没有 `dynamic-network-architectures`。

影响：

即使使用旧的 `backbone: conv` 默认路径，模型实例化仍会进入 `VoxTellDecoder` 并要求 `dynamic_network_architectures`。当前 conda 环境中该包存在，所以测试通过；但新环境按 `requirements.txt` 安装会直接失败。原计划 P2 也明确要求把 `dynamic-network-architectures>=0.4.1,<0.5` 写入 Body-Tell 环境声明。

建议：

把 `dynamic-network-architectures>=0.4.1,<0.5` 加入根依赖文件。若后续代码直接依赖 VoxTell 同款工具链，也同步确认是否需要 `einops`。

### 2. Medium: `parameter_audit.py` 会把共享 encoder 计入 decoder，参数审计结果会误导

证据：

- `Body-Tell/body_tell/models/voxtell_body_model.py:325` 在 `VoxTellDecoder` 中保存 `self.encoder = encoder`。
- `Body-Tell/scripts/parameter_audit.py:52` 到 `:56` 的 `module_param_count()` 直接统计 `module.parameters()`。
- `Body-Tell/scripts/parameter_audit.py:59` 到 `:65` 的 `direct_child_counts()` 会统计 top-level `decoder` 子树。
- `Body-Tell/scripts/parameter_audit.py:92` 把 `decoder` 作为独立 subtree 输出。

影响：

由于 decoder 持有共享 encoder 引用，`decoder` subtree 统计会包含 encoder 参数。原计划里已经提醒“统计 decoder 参数时需扣除共享 encoder，避免误读为 `102.47M`”。当前审计脚本没有做这个扣除，后续参数报告容易把兼容性共享引用误读成真实 decoder 膨胀。

建议：

在参数审计脚本中对 shared module 做去重，或专门把 `decoder.encoder.*` 从 decoder subtree 统计中排除。报告中同时显示“raw subtree count”和“deduplicated decoder count”会更稳。

### 3. Medium: `parameter_audit.py` 的 encoder stage 明细仍按 5-stage 写死

证据：

- `Body-Tell/scripts/parameter_audit.py:69` 到 `:76` 只列出 `encoder.stages.0` 到 `encoder.stages.4`。
- `Body-Tell/configs/phase1_voxtell_aligned.yaml:7` 到 `:8` 配置了 6 个 encoder stage 和 6 个 block count。

影响：

aligned config 下的 `encoder.stages.5` 以及 residual encoder 的 stem 不会进入 subtree 明细。总参数仍可由 top-level `encoder` 看到，但分 stage 审计不完整，容易掩盖 6-stage VoxTell 对齐是否真的落地。

建议：

subtree 明细应从 `model.encoder.stages` 动态生成，并在 residual encoder 模式下单独列出 `encoder.stem`。

## Bloat / Debug Review

### 可接受的“兼容性臃肿”

以下内容看起来像冗余，但和 VoxTell checkpoint key/shape 兼容有关，不建议作为 debug 残留删除：

- `TransformerDecoderLayer` 仍保留 `self_attn`、`norm1` 等参数，但 `forward_pre()` 按 VoxTell 路径跳过 self-attention。证据：`Body-Tell/body_tell/models/transformer.py:82` 到 `:94`、`:137` 到 `:161`。
- `VoxTellDecoder` 即使 `deep_supervision=False` 也构造所有 `seg_layers`。证据：`Body-Tell/body_tell/models/voxtell_body_model.py:408` 到 `:421`。
- 测试显式记录这些参数没有梯度。证据：`Body-Tell/tests/test_model_shapes.py:434` 到 `:450`。

这些会带来 DDP `find_unused_parameters=True` 和参数量观感上的成本，但属于有意的 checkpoint 兼容成本，不是 debug 代码。

### 偏臃肿但不阻塞的实现

- `Body-Tell/scripts/audit_voxtell_alignment.py` 目前 826 行，把 AST 解析、checkpoint shape 审计、文本输出、HTML 渲染、嵌入 JSON 都放在一个脚本里。`render_html_report()` 从 `:633` 到 `:764`，占比较高。
- 这是离线审计脚本，不影响训练主路径；但如果后续 SP-B/SP-C 也继续复制这种报告脚本模式，reports/scripts 会快速膨胀。

建议：

保留当前脚本作为 SP-A 审计产物可以接受；下一轮新增审计时，应抽出 shared helper，例如 checkpoint shape loading、state-dict prefix compare、HTML table rendering。

### Debug 残留检查

检查对象为本次改动的 10 个代码/测试/config 文件。结果：

- 未发现 `pdb`、`ipdb`、`breakpoint`、`set_trace`。
- 未发现临时 `TODO` / `FIXME`。
- `print()` 只出现在 CLI 审计脚本 `audit_voxtell_alignment.py` 和 `parameter_audit.py`，属于命令行输出，不是训练主路径 debug 打印。
- 工作区存在若干 `__pycache__` / `.pyc` 文件，但 `git ls-files` 未显示它们被跟踪；属于本地/忽略产物，不是本次代码 diff 的组成部分。

## 测试与验证

已执行：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n voxtell python -m pytest \
  Body-Tell/tests/test_model_shapes.py \
  Body-Tell/tests/test_voxtell_transformer_compat.py \
  Body-Tell/tests/test_train_loop.py
```

结果：`16 passed in 59.36s`

已执行：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n voxtell python \
  Body-Tell/scripts/audit_voxtell_alignment.py \
  --config Body-Tell/configs/phase1_voxtell_aligned.yaml \
  --skip-checkpoint --skip-state-dict-compare
```

结果：命令成功返回；轻量路径确认 aligned config 下 5 个 projection stage 和 P6 whitelist 输入可生成。完整 checkpoint audit 已由 runs 中的 SP-A.P5-P6-lite 记录为通过。

## 结论

功能层面：当前 SP-A 对齐实现可以继续进入后续 review/实验，不建议为“瘦身”删除 transformer self-attn、norm1、decoder seg_layers 或 shared encoder 引用，这些是 VoxTell 兼容设计的一部分。

工程层面：建议在继续 SP-B/SP-C 前先补两个收尾项：声明 `dynamic-network-architectures` 依赖；修正 `parameter_audit.py` 的 shared encoder 去重和 6-stage 明细。这样可以避免后续环境复现失败和参数报告误读。
