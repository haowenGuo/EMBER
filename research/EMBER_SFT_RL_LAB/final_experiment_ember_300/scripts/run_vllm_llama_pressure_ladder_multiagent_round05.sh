#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/vllm_llama_pressure_ladder_multiagent_round05_${STAMP}.log"
PID_FILE="$LOG_DIR/vllm_llama_pressure_ladder_multiagent_round05.pid"

mkdir -p "$LOG_DIR"
cd "$ROOT"

(
  env \
    RUN_NAME="vllm_llama_pressure_ladder_multiagent_round05" \
    STAMP="$STAMP" \
    DATASET="$ROOT/final_experiment_ember_300/data/final_ember_topics_300.jsonl" \
    TARGET_MODEL="/data2/guohaowen_data/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1" \
    TARGET_LABEL="llama" \
    TOPIC_START=1 \
    TOPIC_END=150 \
    MAX_DEBATE_ROUND=5 \
    VARIANTS="BASE+MULTIAGENT,EMBER-PROMPT+MULTIAGENT,EMBER-AGENT+MULTIAGENT" \
    ATTACK_PROMPT_PROFILE="pressure_ladder" \
    TARGET_GPUS="0,2" \
    ATTACKER_GPUS="1,3" \
    ATTACKER_GPU="1" \
    BIAS_GPUS="0,1,2,3" \
    TARGET_CONCURRENCY=4 \
    ATTACKER_CONCURRENCY=4 \
    JOBS_PER_ENDPOINT=4 \
    BIAS_ENDPOINT_CONCURRENCY=24 \
    BIAS_ROWS_PER_ENDPOINT=48 \
    bash final_experiment_ember_300/scripts/run_vllm_first_eval_openmodel_round05.sh
) > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
echo "Run root: $ROOT/final_experiment_ember_300/results/vllm_llama_pressure_ladder_multiagent_round05_${STAMP}"
