#!/usr/bin/env bash
set -euo pipefail

cd /home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen

PYTHON="/home/haowen/Lab/enter/envs/ai/bin/python"
ATTACKER="/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B"

"${PYTHON}" run_fiveway_exact_biasexpert_eval.py \
  --topic-start 101 \
  --topic-end 103 \
  --variants BASE \
  --provocateur-engine local \
  --local-provocateur-model-path "${ATTACKER}" \
  --local-provocateur-max-new-tokens 1024 \
  --max-parallel-gpus 1 \
  --output-dir outputs/qwen3_5_9b_local_attacker_reference_protocol_base_101_103
