# EMBER Final 300 Experiment

本目录是后续正式实验的唯一入口，用于隔离旧的 CMV001-100、CMV101-200、Doubao 历史实验、Llama 旧协议实验等混杂文件。除非特别回溯历史结果，后续实验应优先只参考本目录和本文档列出的脚本。

## Final Scope

- 数据集：`data/final_ember_topics_300.jsonl`
- 规模：300 条，其中 CMV 分层抽样 200 条，国内语境补充 100 条。
- 六维：`political`, `gender`, `ethnic_cultural`, `age`, `religion`, `disability`
- 目标模型：Qwen3-4B-Instruct、Llama-3.1-8B-Instruct，按同一协议比较。
- 攻击模型：优先使用本地 ModelScope `Qwen3.5-9B`，Doubao 仅作为历史对照。
- 评估器：`EmergentMethods/Qwen3-4B-BiasExpert`
- 方法：`BASE`, `EMBER-PROMPT`, `EMBER-AGENT`, `SFT`, `SFT+RL`

## Directory Layout

```text
final_experiment_ember_300/
  data/
    final_ember_topics_300.jsonl
    cmv_stratified_200.jsonl
    domestic_topics_100.jsonl
    final_ember_topics_300_metadata.csv
    final_ember_topics_300_summary.json
  docs/
    01_dataset_construction.md
    02_prompt_and_agent_methods.md
    03_attacker_and_evaluation_protocol.md
    04_code_inventory.md
  configs/
    final_experiment_config.json
  code/
    formal_eval_qwen/
    formal_eval_llama/
    pipeline_scripts/
    attacker_probe/
    src/risk_gated_alignment/
    configs/
    runtime/
  runbooks/
    01_server_runbook.md
  scripts/
    build_final_ember_topic_dataset.py
    run_qwen_final150_round05_three_methods_local_attacker.sh
    run_llama_final150_round05_three_methods_local_attacker.sh
    run_qwen_final_300_local_attacker.sh
    run_llama_final_300_local_attacker.sh
  results/
    reserved for new final-run outputs copied back from fat20
```

`code/` 是最终实验代码快照，用于审计和防止误用旧脚本；`scripts/` 是推荐运行入口。

## Current Run Profile

最新正式运行口径：

- 数据范围：`final_001` 到 `final_150`，也就是 `--topic-start 1 --topic-end 150`。
- 对话轮次：`0-5`，也就是 `--max-debate-round 5`。
- 方法：`BASE`, `EMBER-PROMPT`, `EMBER-AGENT`。
- 场景：双 Agent 辩论，目标模型 vs. Qwen3.5-9B 攻击模型。
- 目标模型：Qwen3-4B-Instruct 与 Llama-3.1-8B-Instruct。

启动脚本：

```text
scripts/run_qwen_final150_round05_three_methods_local_attacker.sh
scripts/run_llama_final150_round05_three_methods_local_attacker.sh
```

## Important Boundary

旧目录中的结果只用于历史排查，不作为最终实验口径：

- `formal_eval_cmv101_200_fiveway_exact_fullcontext`
- `formal_eval_cmv101_200_fiveway_exact_fullcontext_llama`
- `formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/outputs/*`
- `outputs/provocateur_probe/*`

后续正式输出建议统一命名为：

```text
final_ember300_<target-model>_qwen3_5_9b_attacker_<date>
```

例如：

```text
final_ember300_qwen3_4b_qwen3_5_9b_attacker_20260528
final_ember300_llama3_1_8b_qwen3_5_9b_attacker_20260528
```

## Compatibility Note

已对 Qwen 和 Llama 的正式评估脚本做了向后兼容补丁：当 JSONL 行中存在 `topic_id`, `source`, `primary_dimension` 等字段时，评估输出会保留这些元数据；原始 CMV 文件没有这些字段时仍按旧逻辑自动生成 `cmv_001` 等编号。
