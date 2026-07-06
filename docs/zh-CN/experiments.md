# 实验说明

## EMBER 框架实验

论文围绕多轮对抗交互设置了多组实验：

1. EMBER 框架涌现偏见评估有效性验证。
2. 对抗强度对涌现偏见的影响。
3. EMBER-Agent 与提示词缓偏策略对比。
4. 风险感知参数高效缓偏方法验证。
5. EMBER-Harness 阶段门控有效性验证。

旧实验脚本位于 `code/`，公开参考实现位于 `src/ember/`。

## 论文结果总览

`experiments/thesis_results` 保存了论文级结果图表和摘要 CSV，覆盖以下实验：

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 多轮偏见自然演化 | `figures/fig_01_dynamic_bias_round_lines.png` | 展示 Qwen3-4B、Llama-3.1-8B、Mistral-7B 在双智能体和多智能体辩论中的轮次变化。 |
| 偏见维度贡献 | `figures/fig_02_bias_dimension_contribution.png` | 展示政治、种族/文化、宗教、性别、残障、年龄六个维度的贡献比例。 |
| 对抗强度影响 | `figures/fig_03_attack_strength_lines.png` | 比较中立、温和、强对抗提示词对偏见得分的影响。 |
| EMBER-Agent 缓偏 | `figures/fig_05_qwen_strategy_lines.png` 至 `figures/fig_07_mistral_strategy_lines.png` | 比较基线、提示词缓偏和 EMBER-Agent 的综合偏见得分。 |
| 风险感知参数高效缓偏 | `figures/fig_08_risk_peft_overall_table.png`、`figures/fig_09_risk_peft_round_lines.png` | 展示 EMBER-Prompt、EMBER-Agent、SFT、SFT+RL 的训练期缓偏效果。 |
| 评估器与人工校验 | `figures/fig_10_evaluator_human_alignment.png` | 汇总 BiasExpert、GPT-5.5 与人工抽样的趋势一致性。 |
| EMBER-Harness | `figures/fig_11_harness_safety_cost_tradeoff.png` | 展示最终检查、逐调用检查和阶段门控的安全-成本权衡。 |

该目录只保存摘要级结果，不包含大体积原始生成 JSONL。完整复现实验可根据
`data/*.csv`、`src/ember/` 和早期 `code/` 脚本继续扩展。

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
