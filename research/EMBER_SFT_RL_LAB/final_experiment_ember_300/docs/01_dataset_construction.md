# EMBER Topic Pools and Fixed Evaluation Manifests

话题数据用于补充 CMV 争议话题库在国内语境中的不足。CMV 话题用于保留真实讨论中的自然对抗性，国内语境话题用于补充公共治理、地域文化、婚育就业、代际关系、宗教民俗和无障碍等场景。数据目录同时保留候选语料和论文实际使用的固定评测清单，二者用途不同。

## 候选语料

- 候选语料规模：300 条。
- CMV 分层抽样：200 条。
- 国内语境补充：100 条。
- 六维分类：political, gender, ethnic_cultural, age, religion, disability。
- 输出格式：保持 `prompt[0].content` 以 `CMV:` 开头。

## 分层抽样方法

1. 对 CMV 原始文本进行关键词规则分类，得到六维候选池。
2. 过滤过短文本和少量明显不适合作为安全评测话题的高风险文本。
3. 使用固定随机种子在各维候选池中抽样，CMV 配额为 34/34/33/33/33/33。
4. 国内话题按 17/17/17/17/16/16 配额补充，所有话题采用中性、可辩论表述，避免直接写入歧视性断言。

## 统计

| dimension | CMV | domestic | total |
|---|---:|---:|---:|
| political | 34 | 17 | 51 |
| gender | 34 | 17 | 51 |
| ethnic_cultural | 33 | 17 | 50 |
| age | 33 | 17 | 50 |
| religion | 33 | 16 | 49 |
| disability | 33 | 16 | 49 |

上述 200 条 CMV 与 100 条国内语境话题保存在 `final_ember_topics_300.*` 中，用于记录候选语料的构建过程。该文件按数据来源组织，不能直接按前 150 条和后 150 条切分为论文评测集。

## 论文固定评测清单

正式实验使用 `data/eval_splits_reusemax/` 下的不可变清单，而不是对候选语料进行顺序截取。固定随机种子为 20260529。第一阶段和第二阶段各包含 150 条互不重叠的话题，每个阶段均由 125 条 CMV 话题和 25 条国内语境话题组成，并在六个偏见维度上严格保持每维 25 条。合并后，论文实际评测集合包含 250 条 CMV 话题和 50 条国内语境话题。

`reusemax` 仅表示在满足来源配额、维度均衡和阶段互斥约束的前提下，优先复用已经生成的 CMV 轨迹；它不会改变话题文本、轮次协议或评分方法。论文结果应使用以下文件复现：

- `data/eval_splits_reusemax/first_eval_150.jsonl`
- `data/eval_splits_reusemax/second_eval_150.jsonl`
- `data/eval_splits_reusemax/first_second_eval_300.jsonl`
- `data/eval_splits_reusemax/first_second_eval_reusemax_summary.json`

## 候选语料输出文件

候选语料及其构建记录包括：

- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\final_ember_topics_300.jsonl`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\cmv_stratified_200.jsonl`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\domestic_topics_100.jsonl`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\final_ember_topics_300_metadata.csv`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\final_ember_topics_300_summary.json`

`source_data/final_ember_300` 是候选语料生成时的镜像位置。为避免混淆，复现论文结果时必须显式传入 `eval_splits_reusemax` 下的固定清单。
