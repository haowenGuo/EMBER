# vLLM 推理架构说明

本轮正式实验不再使用旧的“生成一轮就立刻调用 BiasExpert 评分”的串行流程，而改为两阶段架构：

1. 生成阶段：使用 vLLM OpenAI-compatible server 批量生成多轮对抗对话，只保存 `generated_rounds.jsonl`。
2. 评估阶段：等 Qwen 与 Llama 的所有生成结果完成后，再统一启动 BiasExpert vLLM 服务，离线生成 `scored_rounds.jsonl` 与 `analysis/*.csv`。

## GPU 分配

生成阶段采用数据并行：

- GPU3：单独运行一个共享的 Qwen3.5-9B attacker vLLM 服务。
- GPU0/GPU1/GPU2：分别运行 3 个 target vLLM 服务，对 topic/variant job 分片并行生成。
- Qwen target 先跑完，再释放 target 服务并跑 Llama target。

评估阶段采用数据并行：

- GPU0/GPU1/GPU2/GPU3：全部运行 BiasExpert vLLM 服务。
- 所有 `generated_rounds.jsonl` 行按 shard 分配到 4 个 BiasExpert endpoint 批量评分。
- BiasExpert 默认使用服务器本地 snapshot：
  `/data2/guohaowen_data/huggingface_cache/hub/models--EmergentMethods--Qwen3-4B-BiasExpert/snapshots/02585c80cc5b324228255d9bbb26017ce649dc43`。

## 关键变化

- `BASE` 与 `EMBER-PROMPT` 保持原始 debate prompt 语义。
- `EMBER-AGENT` 不再在生成时调用 BiasExpert，因为这会把评估模型和算法过程耦合在一起，并显著降低显存与 batch 利用率。
- 新的 `EMBER-AGENT` 使用 target 模型自身的 self-reflection rewrite：先生成初稿，再进行一次安全检查与最小改写，最终 BiasExpert 只在离线阶段评分。
- vLLM 服务通过并发 OpenAI chat 请求自然形成 batch；每个 target endpoint 内部可同时处理多个 topic job。

## 入口脚本

安装 vLLM：

```bash
cd /home/haowen/Lab/EMBER_SFT_RL_LAB
bash final_experiment_ember_300/scripts/install_vllm_runtime.sh
```

安装脚本会创建独立环境 `/home/haowen/Lab/enter/envs/vllm`。vLLM server 使用该环境运行，数据生成与离线评分控制脚本仍使用原有 `/home/haowen/Lab/enter/envs/ai/bin/python`，避免污染旧训练/评估环境。

启动 0-150、0-5 轮、三方法、Qwen 后 Llama 的完整流程：

```bash
cd /home/haowen/Lab/EMBER_SFT_RL_LAB
nohup bash final_experiment_ember_300/scripts/run_vllm_final150_round05_qwen_then_llama.sh \
  > logs/vllm_final150_round05_launcher_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

小规模冒烟测试可用同一个脚本缩小范围：

```bash
cd /home/haowen/Lab/EMBER_SFT_RL_LAB
TOPIC_START=1 TOPIC_END=1 MAX_DEBATE_ROUND=1 TARGET_GPUS=0 BIAS_GPUS=0 \
  bash final_experiment_ember_300/scripts/run_vllm_final150_round05_qwen_then_llama.sh
```

默认 `VLLM_MAX_MODEL_LEN=16384`，避免 0-5 轮中后段上下文超过 8192。若后续继续缩短轮数或上下文，也可以调回 8192 换取更高 batch 容量。

如果某张 GPU 已被其他进程占用但仍想纳入，可降低 vLLM 显存预留比例，例如：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.75 TARGET_GPUS=0,1,2 BIAS_GPUS=0,1,2,3 \
  bash final_experiment_ember_300/scripts/run_vllm_final150_round05_qwen_then_llama.sh
```

若 GPU0 被其他任务持续占用，更稳的启动方式是：

```bash
TARGET_GPUS=1,2 ATTACKER_GPU=3 BIAS_GPUS=1,2,3 \
  bash final_experiment_ember_300/scripts/run_vllm_final150_round05_qwen_then_llama.sh
```

输出目录形如：

```text
final_experiment_ember_300/results/vllm_final150_round05_YYYYmmdd_HHMMSS/
  qwen/
    generation_manifest.json
    generated_rounds.jsonl
    scoring_manifest.json
    scored_rounds.jsonl
    analysis/
  llama/
    generation_manifest.json
    generated_rounds.jsonl
    scoring_manifest.json
    scored_rounds.jsonl
    analysis/
```

## 当前旧运行快照

低并行旧运行已经在服务器端归档到：

```text
/home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/results/partial_archives/before_vllm_20260528_162612
```

该目录保留了当时的 worker JSONL、日志、启动脚本和 GPU/进程状态。旧进程已停止，原始输出目录未删除。
