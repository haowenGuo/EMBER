#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/haowen/Lab/EMBER_SFT_RL_LAB"
PYTHON="/home/haowen/Lab/enter/envs/ai/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="$ROOT/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen"
DATASET="$ROOT/final_experiment_ember_300/data/final_ember_topics_300.jsonl"
TARGET="/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ATTACKER="/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B"
OUT_DIR="$EVAL_DIR/outputs/final150_round05_qwen3_4b_base_prompt_agent_qwen3_5_9b_attacker_${STAMP}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/final150_round05_qwen3_4b_base_prompt_agent_qwen3_5_9b_attacker_${STAMP}.log"
PID_FILE="$LOG_DIR/final150_round05_qwen3_4b.pid"

mkdir -p "$LOG_DIR"
cd "$EVAL_DIR"

COMMON_ARGS=(
  --model-name-or-path "$TARGET"
  --cmv-path "$DATASET"
  --topic-start 1
  --topic-end 150
  --max-debate-round 5
  --variants BASE,EMBER-PROMPT,EMBER-AGENT
  --provocateur-engine local
  --local-provocateur-model-path "$ATTACKER"
  --local-provocateur-device cuda:1
  --local-provocateur-max-new-tokens 1024
  --output-dir "$OUT_DIR"
)

(
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" run_fiveway_exact_biasexpert_eval.py \
    --worker-mode \
    --worker-id 0 \
    "${COMMON_ARGS[@]}"
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" run_fiveway_exact_biasexpert_eval.py \
    "${COMMON_ARGS[@]}" \
    --max-parallel-gpus 1
) > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
echo "Output: $OUT_DIR"

