#!/usr/bin/env bash
set -euo pipefail

cd /home/haowen/Lab/EMBER_SFT_RL_LAB

ATTACKER="/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
PYTHON="/home/haowen/Lab/enter/envs/ai/bin/python"

for topic in 102 103; do
  "${PYTHON}" scripts/probe_local_qwen_provocateur.py \
    --attacker-model-path "${ATTACKER}" \
    --topic-index "${topic}" \
    --rounds 1 \
    --target-device cuda:0 \
    --attacker-device cuda:1 \
    --evaluator-device cuda:2 \
    --target-max-new-tokens 512 \
    --attacker-max-new-tokens 512 \
    --output-dir outputs/provocateur_probe/qwen3_4b_instruct_2507_reference_protocol
done
