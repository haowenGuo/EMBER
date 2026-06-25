# EMBER 架构说明

## 1. EMBER 动态评估框架

EMBER 将偏见评估从静态问答扩展为多轮对抗交互。系统包含目标模型、对抗智能体、对话竞技场、偏见评估器和结果分析模块。目标模型在连续轮次中受到对抗智能体追问和挑战，系统只抽取目标模型回答进行偏见评分，从而观察偏见随轮次变化的动态轨迹。

## 2. EMBER-Agent

EMBER-Agent 面向推理期缓偏。其基本闭环为：

```text
Think -> Reflect -> Observe -> Act
```

- Think：根据当前上下文生成候选回答。
- Reflect：检查候选回答中的偏见、刻板印象或不平衡表述。
- Observe：判断当前回答是否达标，并形成修改反馈。
- Act：输出最终回答，或根据反馈重新生成。

该机制适用于多轮对抗场景，因为它不只依赖对话开始前的一次性安全提示，而是在每次输出前进行动态自检。

## 3. EMBER-Harness

EMBER-Harness 是将 EMBER-Agent 工程化部署到复杂 agent 系统中的控制层。它不替换底座模型，也不要求重写业务逻辑，而是在关键阶段外包裹 Stage Manager 与 EMBER Gate。

典型阶段包括：

```text
input_parse
memory_read
retrieval
planning
tool_result
draft_response
final_response
```

每个阶段执行后，系统先保存阶段快照，再进行风险评审。通过评审的快照被标记为 committed；未通过评审的快照被标记为 failed，并触发回退。

## 4. 阶段快照与回退

阶段快照保存：

- 阶段名称
- 阶段产物
- 元数据
- 上一稳定快照 ID
- 风险评审结果

当 EMBER Gate 发现风险时，Harness 不直接丢弃整条任务轨迹，而是回退到最近的 committed 快照。这样可以只重做当前阶段或后续阶段，减少最终输出检查导致的大规模返工。

## 5. 三种自检位置

| 策略 | 优点 | 缺点 |
| --- | --- | --- |
| 最终输出自检 | 检查次数少，接入简单 | 风险发现晚，难定位风险源，返工范围大 |
| 逐调用自检 | 检出及时，覆盖最细 | 检查次数和 token 开销最高 |
| 阶段门自检 | 在安全性和开销之间折中，支持局部回退 | 对阶段划分和状态管理有要求 |

## 6. 公开实现主线

当前仓库的主线实现位于 `src/ember`：

- `providers/`：统一 provider 层，支持离线规则 provider、OpenAI-compatible provider、本地 Transformers provider。
- `agent.py`：实现 EMBER-Agent 自检与重写闭环。
- `harness.py`：实现阶段快照、EMBER Gate、审计日志和回退指针。
- `runner.py`：将阶段计划真正跑起来，并支持 `final_only`、`per_call`、`stage_gate` 三种策略。
- `cli.py`：提供 `ember-agent-demo`、`ember-harness-run`、`ember-benchmark` 三个命令。

在 `stage_gate` 策略中，runner 会为每个阶段先创建 `pending` 快照并执行
风险评估；通过后快照状态更新为 `committed`，失败后状态更新为 `failed`，
并在决策中记录 `rollback_to` 指向最近的安全快照。若设置 `state_dir`，
这些快照会以 JSON 文件落盘，同时追加写入 `audit.jsonl`，便于复现实验轨迹和
分析回退次数。

旧 `code/` 目录保留为论文实验脚本，不再作为公开仓库的主要入口。
