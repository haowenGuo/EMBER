# Final EMBER 300 Topic Dataset Construction

本数据集用于补充原 CMV 争议话题库在国内语境中的不足，构建逻辑参考论文中“开源数据集为主、自动构建数据集为辅”的方法：以 CMV 的真实争议话题保证自然对抗性，以国内话题补充中文语境、公共治理、地域文化、婚育就业、代际关系、宗教民俗和无障碍等场景。

## 构成

- 总规模：300 条。
- CMV 分层抽样：200 条。
- 国内语境补充：100 条。
- 六维分类：political, gender, ethnic_cultural, age, religion, disability。
- 输出格式：保持 `prompt[0].content` 以 `CMV:` 开头，可直接作为现有 formal eval 脚本的 `--cmv-path` 输入。

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

## 输出文件

最终实验应使用本目录下的隔离副本：

- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\final_ember_topics_300.jsonl`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\cmv_stratified_200.jsonl`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\domestic_topics_100.jsonl`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\final_ember_topics_300_metadata.csv`
- `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\data\final_ember_topics_300_summary.json`

`source_data/final_ember_300` 是生成时的镜像位置；为避免与旧实验混淆，后续正式运行以 `final_experiment_ember_300/data` 为准。
