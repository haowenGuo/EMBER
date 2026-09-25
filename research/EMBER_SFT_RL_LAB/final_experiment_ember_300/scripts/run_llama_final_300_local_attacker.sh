#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/haowen/Lab/EMBER_SFT_RL_LAB"
PYTHON="/home/haowen/Lab/enter/envs/ai/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="$ROOT/formal_eval_cmv101_200_fiveway_exact_fullcontext_llama_from_qwen"
DATASET="$ROOT/final_experiment_ember_300/data/final_ember_topics_300.jsonl"
ATTACKER="/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B"
SFT_CKPT="$ROOT/runs/llama3_1_8b_instruct_qwen_protocol/sft/best"
RL_CKPT="$ROOT/runs/llama3_1_8b_instruct_qwen_protocol/rl/final"
OUT_DIR="$EVAL_DIR/outputs/final_ember300_llama3_1_8b_qwen3_5_9b_attacker_${STAMP}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/final_ember300_llama3_1_8b_qwen3_5_9b_attacker_${STAMP}.log"
PID_FILE="$LOG_DIR/final_ember300_llama3_1_8b.pid"

mkdir -p "$LOG_DIR"
cd "$EVAL_DIR"

test -d "$SFT_CKPT"
test -d "$RL_CKPT"

nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2 \
  "$PYTHON" run_fiveway_exact_biasexpert_eval.py \
  --cmv-path "$DATASET" \
  --topic-start 1 \
  --topic-end 300 \
  --variants BASE,EMBER-PROMPT,EMBER-AGENT,SFT,SFT+RL \
  --sft-checkpoint-dir "$SFT_CKPT" \
  --rl-checkpoint-dir "$RL_CKPT" \
  --provocateur-engine local \
  --local-provocateur-model-path "$ATTACKER" \
  --local-provocateur-device cuda:1 \
  --local-provocateur-max-new-tokens 1024 \
  --output-dir "$OUT_DIR" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
echo "Output: $OUT_DIR"

