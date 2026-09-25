# 0-5轮偏见演化与图表分析（论文正文草稿）

## 表X 0-5轮总偏见分数均值

总偏见分数为六个维度偏见等级分数之和；数值越低表示偏见风险越低。高风险率定义为总偏见分数不低于6的样本占比。

| 模型 | 方法 | 第0轮 | 第1轮 | 第2轮 | 第3轮 | 第4轮 | 第5轮 | 第5轮-第0轮 | 第5轮高风险率 | 第5轮相对BASE降幅 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen | BASE | 1.707 | 2.260 | 2.793 | 3.073 | 3.127 | 3.093 | 1.386 | 10.0% | - |
| Qwen | PROMPT | 2.027 | 2.267 | 2.500 | 2.727 | 2.807 | 2.780 | 0.753 | 8.7% | 10.1% |
| Qwen | AGENT | 1.693 | 1.940 | 2.213 | 2.420 | 2.527 | 2.420 | 0.727 | 6.7% | 21.8% |
| Llama | BASE | 2.240 | 2.067 | 2.053 | 1.987 | 2.000 | 1.967 | -0.273 | 4.0% | - |
| Llama | PROMPT | 2.227 | 1.973 | 1.993 | 1.953 | 1.953 | 1.920 | -0.307 | 2.0% | 2.4% |
| Llama | AGENT | 2.053 | 1.907 | 1.913 | 1.840 | 1.800 | 1.813 | -0.240 | 1.3% | 7.8% |

## 图X 0-5轮偏见分数曲线

![0-5轮偏见分数曲线](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/final_experiment_ember_300/results/codex_gpt55_simplified_v2_first_eval_full_20260531/figures/paper_round_curve_qwen_llama_combined.png)

图注建议：图X展示了FIRST_EVAL中Qwen与Llama在BASE、EMBER-PROMPT和EMBER-AGENT三种设置下的0-5轮平均总偏见分数。每个点对应150个topic在该轮的平均得分，分数越高表示偏见风险越强。

## 图Y 相对BASE的逐轮降幅

![相对BASE的逐轮降幅](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/final_experiment_ember_300/results/codex_gpt55_simplified_v2_first_eval_full_20260531/figures/paper_round_reduction_vs_base_combined.png)

图注建议：图Y展示了EMBER-PROMPT和EMBER-AGENT相对BASE的逐轮偏见分数下降量。纵轴为BASE均分减方法均分，正值表示该方法降低了偏见分数，负值表示该方法高于BASE。

## 正文分析草稿

从0-5轮的变化趋势来看，Qwen与Llama呈现出明显不同的多轮偏见演化模式。对于Qwen，BASE设置下的平均总偏见分数从第0轮的1.707快速上升到第5轮的3.093，累计增加1.387分，并在第4轮达到3.127的峰值。这说明在不加入额外缓解机制时，多轮辩论式交互会持续放大Qwen的偏见表达，尤其在中后期轮次中更容易被攻击者诱导出更强的立场化、泛化或刻板化表述。与均值曲线一致，Qwen BASE的高风险率也从第0轮的0.7%上升到第4轮的13.3%，第5轮仍保持在10.0%，表明风险并不只是均值抬升，而是高分尾部样本同步增加。

EMBER-PROMPT在Qwen上的效果具有滞后性。第0轮和第1轮中，PROMPT的偏见均分分别为2.027和2.267，并未低于BASE；从第2轮开始，PROMPT才相对BASE表现出稳定下降，第5轮相对BASE下降10.1%。这说明单纯提示词约束可以缓解部分后期偏见累积，但对初始轮次或早期诱导并不稳定。相比之下，EMBER-AGENT在Qwen上表现出更强且更持续的缓解效果：除第0轮基本持平外，第1-5轮均明显低于BASE，第5轮均分为2.420，相对BASE下降21.8%；同时其高风险率在第5轮为6.7%，低于BASE的10.0%。因此，AGENT机制不仅降低了总体偏见水平，也更有效地压制了多轮交互后期的风险尾部。

Llama的轮次曲线则不同。Llama BASE的平均总偏见分数从第0轮的2.240下降到第5轮的1.967，累计变化为-0.273，并没有出现Qwen式的逐轮放大。这表明在本FIRST_EVAL设置下，Llama的基础响应在多轮交互中更倾向于保持或略微降低偏见分数，因此其可被缓解的空间天然小于Qwen。PROMPT和AGENT在Llama上仍然带来下降，但幅度较小：第5轮PROMPT相对BASE下降2.4%，AGENT下降7.8%。从图Y可以看出，Llama的AGENT降幅在各轮大致保持稳定，说明该机制对Llama也有一致的风险控制作用，只是由于BASE曲线本身没有明显恶化，最终均值改善不如Qwen显著。

综合两组模型可以看出，EMBER-AGENT的优势主要体现在多轮交互诱发偏见增长的场景中。当模型在BASE设置下随轮次逐渐偏离中性、平衡表达时，AGENT能够通过轮次内的风险识别与干预显著压低中后期得分；当模型本身的轮次曲线较平时，AGENT仍能降低偏见分数和高风险尾部，但表现为较温和的稳定收益。因此，0-5轮结果支持本文的核心判断：风险头感知的agent式干预不是单纯降低静态输出偏见，而是尤其适合控制多轮互动中的偏见累积和后期风险放大。

## 可写入论文的简短结论

0-5轮结果表明，Qwen在BASE设置下存在显著的多轮偏见放大现象，而EMBER-AGENT能够最有效地抑制这一趋势。相比之下，Llama的BASE曲线本身较为平缓，因此方法收益主要表现为小幅但稳定的风险降低。整体而言，EMBER-AGENT在两个模型上均优于BASE和EMBER-PROMPT，尤其在Qwen的中后期轮次中体现出更强的偏见累积抑制能力。

## 文件

- 论文轮次表: `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\results\codex_gpt55_simplified_v2_first_eval_full_20260531\detailed_statistics\paper_round_analysis_table.csv`
- 曲线图: `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\results\codex_gpt55_simplified_v2_first_eval_full_20260531\figures\paper_round_curve_qwen_llama_combined.png`
- 降幅图: `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\results\codex_gpt55_simplified_v2_first_eval_full_20260531\figures\paper_round_reduction_vs_base_combined.png`
