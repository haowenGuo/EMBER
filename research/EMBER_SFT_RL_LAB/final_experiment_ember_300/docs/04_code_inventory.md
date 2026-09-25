# Code Inventory

本目录封存最终实验相关代码，避免后续从旧实验文件夹误用早期脚本。

## Execution Launchers

这些脚本是服务器上推荐的启动入口：

| path | purpose |
|---|---|
| `scripts/run_qwen_final_300_local_attacker.sh` | 启动 Qwen3-4B 在 final 300 数据集上的五方法评估 |
| `scripts/run_llama_final_300_local_attacker.sh` | 启动 Llama-3.1-8B 在 final 300 数据集上的五方法评估 |
| `scripts/run_qwen_final150_round05_three_methods_local_attacker.sh` | 启动 Qwen3-4B 在前 150 条、0-5 轮、三方法评估 |
| `scripts/run_llama_final150_round05_three_methods_local_attacker.sh` | 启动 Llama-3.1-8B 在前 150 条、0-5 轮、三方法评估 |
| `scripts/build_final_ember_topic_dataset.py` | 复现 final 300 数据集构建 |

## Evaluation Script Snapshots

| path | purpose |
|---|---|
| `code/formal_eval_qwen/run_fiveway_exact_biasexpert_eval.py` | Qwen 目标模型正式五方法评估脚本快照 |
| `code/formal_eval_llama/run_fiveway_exact_biasexpert_eval.py` | Llama-from-Qwen 协议正式五方法评估脚本快照 |

这两个脚本快照包含最终补丁：

- 支持读取 final 300 JSONL 中的 `topic_id`, `source`, `primary_dimension` 等元数据。
- 支持本地 Qwen3.5-9B 攻击模型。
- 本地攻击模型关闭 thinking 输出，避免 `<think>` 或 `Thinking Process` 污染。

实际服务器运行仍使用仓库中的正式位置：

```text
formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py
formal_eval_cmv101_200_fiveway_exact_fullcontext_llama_from_qwen/run_fiveway_exact_biasexpert_eval.py
```

`code/` 下的版本用于审计、归档和必要时恢复。

## Training and Data Pipeline Snapshots

| path | purpose |
|---|---|
| `code/pipeline_scripts/run_server_pipeline.py` | Qwen/Llama 风险感知训练总 pipeline |
| `code/pipeline_scripts/run_llama_server_pipeline.py` | Llama 训练启动包装脚本 |
| `code/pipeline_scripts/prepare_topic_splits.py` | topic split 生成 |
| `code/pipeline_scripts/build_training_corpus.py` | 训练语料构建 |
| `code/pipeline_scripts/audit_training_sources.py` | 训练来源审计 |
| `code/pipeline_scripts/export_alignment_data.py` | alignment 数据导出 |
| `code/pipeline_scripts/train_risk_head.py` | risk head 训练 |
| `code/pipeline_scripts/train_sft_adapters.py` | adapter SFT 训练 |
| `code/pipeline_scripts/train_rl_policy.py` | RL policy 训练 |
| `code/pipeline_scripts/evaluate_policy.py` | policy 评估 |
| `code/pipeline_scripts/plot_metrics.py` | 训练指标绘图 |
| `code/pipeline_scripts/download_modelscope_qwen_attackers.py` | ModelScope Qwen 攻击模型下载 |

## Attacker Probe Snapshot

| path | purpose |
|---|---|
| `code/attacker_probe/probe_local_qwen_provocateur.py` | 本地 Qwen 攻击模型小样本探测脚本 |

## Core Python Package Snapshot

| path | purpose |
|---|---|
| `code/src/risk_gated_alignment/data.py` | topic、训练样本、偏见标签数据逻辑 |
| `code/src/risk_gated_alignment/modeling.py` | RiskGatedCausalLM 与可训练状态加载 |
| `code/src/risk_gated_alignment/rewards.py` | RL reward 相关逻辑 |
| `code/src/risk_gated_alignment/utils.py` | 通用工具函数 |

## Config and Runtime

| path | purpose |
|---|---|
| `code/configs/qwen3_4b_local_server.env` | Qwen 目标模型训练配置 |
| `code/configs/llama3_1_8b_qwen_protocol_server.env` | Llama 使用 Qwen 协议训练配置 |
| `code/runtime/requirements-server.txt` | 服务器 Python 依赖 |
| `code/runtime/pyproject.toml` | 项目 Python 包配置 |

## Do Not Use as Final Protocol

以下旧脚本没有放入最终代码快照，除非回溯历史实验，不应作为最终实验依据：

- `run_exact_ember_biasexpert_fourway.py`
- `resume_exact_ember_biasexpert_fourway.py`
- `run_qwen_formal_heldout15_*`
- `train_kto.py`
- `train_grpo.py`
- 早期 `formal_eval_cmv001_100*` 和 `formal_eval_cmv101_200*` 目录中的脚本
