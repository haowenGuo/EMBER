#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
PYTHON="${PYTHON:-/home/haowen/Lab/enter/envs/ai/bin/python}"
FIRST_RUN_ROOT="${FIRST_RUN_ROOT:-$ROOT/final_experiment_ember_300/results/vllm_first_eval150_round05_reusemax_20260529_052000}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/datasets/final_first_eval_mixed}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SEED="${SEED:-42}"

"$PYTHON" "$ROOT/final_experiment_ember_300/scripts/build_first_eval_mixed_training_corpus.py" \
  --qwen-scored "$FIRST_RUN_ROOT/qwen/scored_rounds.jsonl" \
  --llama-scored "$FIRST_RUN_ROOT/llama/scored_rounds.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --train-ratio "$TRAIN_RATIO" \
  --seed "$SEED"
