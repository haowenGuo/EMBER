# 实验说明

## EMBER 框架实验

论文围绕多轮对抗交互设置了多组实验：

1. EMBER 框架涌现偏见评估有效性验证。
2. 对抗强度对涌现偏见的影响。
3. EMBER-Agent 与提示词缓偏策略对比。
4. 风险感知参数高效缓偏方法验证。
5. EMBER-Harness 阶段门控有效性验证。

旧实验脚本位于 `code/`，公开参考实现位于 `src/ember/`。

## EMBER-Harness 100 条轨迹实验

`experiments/ember_harness` 保存了可直接用于论文和 README 的处理后材料：

- `data/overall-results.csv`
- `data/by-risk-stage-results.csv`
- `data/sensitivity-results.csv`
- `figures/*.png`
- `figures/*.pdf`
- `tables/*.tex`

该实验比较三种自检策略：

- final-only：只检查最终输出。
- per-call：每次模型调用都检查。
- stage-gate：在语义阶段边界检查。

主要结论：

- final-only token 最低，但检出率和检测滞后较差。
- per-call 检出率最高，但检查 token 与延迟最高。
- stage-gate 在检出率、滞后和检查开销之间取得折中。

## 数据集说明

Harness 实验中的 100 条数据不是普通问答对，而是受控任务轨迹。每条轨迹包含任务类型、阶段序列、风险注入阶段、风险等级、风险维度和 token profile。其目标是验证阶段门控与局部回退机制，而不是模拟真实用户日志的自然分布。
