# Portable Subagent Development Workflow

这是一份可迁移的 subagent 开发流程指导。它抽象自 `Body-Tell/reports/26-05-24`，重点不是复刻 Body-Tell 的细节，而是保留一套稳定的协作结构：主对话只调度，具体工作由专门 agent 完成。

## 1. 一句话流程

```text
Dispatcher -> Worker(TDD) -> Reviewer
                         -> accept -> Commit/Record -> next ready task
                         -> needs_fix/gate_failed -> same Worker rerun
                         -> blocked -> stop and report blocker
```

## 2. 核心原则

- 主对话是 pure dispatcher，只读少量状态，只分配任务，只根据 compact verdict 决定下一步。
- Worker 只做单个任务的实现，按 TDD 执行，写 result artifact，不更新全局状态，不 commit。
- Reviewer 独立验收，先看测试质量，再看实现和 gate，不修改实现，不 commit。
- Review 不通过时，只把同一个任务交回原 worker/fix worker，不进入 commit。
- Review 通过后，由 commit/status agent 验证证据、提交代码、同步状态、记录 commit hash。
- 任务状态只能由 post-accept/status recorder 维护，不能由 worker 或主对话直接改。
- 每个 agent 只拿到完成自己职责所需的上下文，避免主对话长期上下文膨胀。

## 3. 推荐文件结构

```text
reports/<workflow-name>/
  AGENT_WORKFLOW.md
  LEADER_PROMPT.md
  workflow_manifest.yaml
  lead.md
  templates/
    worker_prompt.md
    review_prompt.md
    post_accept_prompt.md
  runs/<TASK_ID>/
    RESULT.md
    REVIEW.md
    POST_ACCEPT.md
```

文件职责：

- `AGENT_WORKFLOW.md`：长期流程规则。
- `LEADER_PROMPT.md`：给主对话的简短调度 prompt。
- `workflow_manifest.yaml`：任务图、依赖、owner、状态、验收标准、验证命令。
- `lead.md`：人类可读的总控计划和进度摘要。
- `templates/*`：不同角色的 prompt 模板。
- `runs/<TASK_ID>/*`：每个任务的证据链。

## 4. 角色边界

### Dispatcher

只负责：

- 读取必要的 manifest 片段。
- 选择下一个 `ready` task。
- 启动 worker、reviewer、post-accept 或 diagnostic agent。
- 等待 compact status。
- 根据 verdict 决定继续、返工、停止或上报 blocker。

禁止：

- 写代码。
- 做完整 review。
- 读长日志、完整 diff、完整报告。
- 改 manifest、lead、报告或源码。
- commit。

### Worker

负责单个任务的 TDD 实现：

1. 读任务 scope 和 acceptance criteria。
2. 先写或更新 focused regression test。
3. 运行测试并记录 red evidence。
4. 做最小实现改动。
5. 重跑测试并记录 green evidence。
6. 写 `RESULT`，列出改动、验证、风险和建议 verdict。

Worker 不同步文档、不改 manifest、不 commit、不做相邻重构。

### Reviewer

负责独立验收：

1. 先审测试质量。
2. 确认测试覆盖验收标准，并能暴露目标 regression。
3. 运行或核验 focused test。
4. 审实现 diff。
5. 运行项目要求的 gate。
6. 写 `REVIEW`，返回 `accept`、`needs_fix`、`gate_failed` 或 `blocked`。

Reviewer 不修改实现。代码问题返回 `needs_fix`；gate 证据缺失或 gate 需要重跑返回 `gate_failed`；外部条件不可用返回 `blocked`。

### Post-Accept / Commit / Status

这个角色可以合并，也可以拆成 commit agent 和 status recorder。

负责：

- 直接检查 `REVIEW` 中的 accept、测试通过、gate 通过证据。
- 确认可提交内容能和无关 dirty files 隔离。
- 创建 task-scoped commit。
- 更新 manifest、lead 和必要报告。
- 记录 commit hash、next ready tasks、残余风险。

如果证据不齐或改动无法隔离，返回 `gate_failed` 或 `blocked`，不 commit。

## 5. 状态机

推荐状态：

```text
todo -> ready -> assigned -> submitted -> accepted
                              -> needs_fix
                              -> gate_failed
                              -> blocked
```

`accepted` 的含义必须明确：review 通过、必要 gate 通过、代码已 commit、状态已记录。不要把 “reviewer 说 accept” 单独当成任务完成。

## 6. 主循环

1. 读 workflow 规则和 manifest 的必要片段。
2. 选择一个依赖已满足的 `ready` task。
3. 派发 worker。
4. worker 返回 compact status 和 `RESULT` 路径。
5. 派发 reviewer。
6. reviewer 返回 compact verdict 和 `REVIEW` 路径。
7. 若 `needs_fix` 或 `gate_failed`，带 review note 重新派发同一任务。
8. 若 `blocked`，停止并报告 blocker。
9. 若 `accept`，派发 post-accept。
10. post-accept 返回 commit hash 后，才进入下一个 `ready` task。

Subagent 返回内容必须简短：

```text
verdict, artifact paths, changed/inspected files, commands run,
gate result, commit hash, blockers, next action
```

## 7. 验收底线

每个实现任务默认要求：

- `Test quality: pass`
- `Targeted test: pass`
- `Required gate: pass` 或明确记录 `skipped` 及原因
- `Reviewer verdict: accept`
- `Commit hash: <hash>`
- manifest/lead 已记录状态

没有这些证据，不进入下一个任务。

## 8. 并行规则

默认不并行。只有在用户明确要求时才并行，并且：

- 每条 lane 的可写路径不重叠。
- 不在同一个 worktree 同时改同一批文件。
- 最好每条 lane 使用独立 worktree/branch。
- 每条 lane 都必须独立 review 和 gate。

## 9. 迁移检查

迁移到新项目时只需要替换：

- 项目任务拆分和依赖关系。
- owner/lane 定义。
- source-of-truth plan 路径。
- acceptance criteria。
- focused test 命令。
- required gate。
- 可写路径。
- artifact 格式和目录。

不要把项目特有知识写死在主对话里。主对话只需要知道如何调度，项目知识应主要存在于 manifest、plan 和 task prompt 中。

## 10. 最重要的不变量

- 主对话不做实质工作。
- Worker 做 TDD，但不改全局状态。
- Reviewer 独立验收，但不修代码。
- Post-accept 只在证据齐全后 commit 和记录状态。
- Review 失败回到同一个任务。
- 每个任务都有 artifact 链：`RESULT -> REVIEW -> POST_ACCEPT`。
- 只有拿到 commit hash 且状态已记录，任务才算完成。
