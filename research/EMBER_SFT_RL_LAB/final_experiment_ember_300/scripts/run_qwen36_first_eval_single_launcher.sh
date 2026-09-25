#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-vllm_first_eval150_round05_qwen36_27b_fp8}"
LOG="$ROOT/logs/${RUN_NAME}_${STAMP}_tmux.log"
mkdir -p "$ROOT/logs"
exec > "$LOG" 2>&1

cd "$ROOT"

export STAMP
export RUN_NAME
export TARGET_MODEL="${TARGET_MODEL:-/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.6-27B-FP8}"
export TARGET_LABEL="${TARGET_LABEL:-qwen36_27b_fp8}"
export TARGET_GPUS="${TARGET_GPUS:-1,2}"
export ATTACKER_GPU="${ATTACKER_GPU:-3}"
export BIAS_GPUS="${BIAS_GPUS:-1,2}"
export TARGET_CONCURRENCY="${TARGET_CONCURRENCY:-2}"
export JOBS_PER_ENDPOINT="${JOBS_PER_ENDPOINT:-2}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export TARGET_VLLM_GPU_MEMORY_UTILIZATION="${TARGET_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export ATTACKER_VLLM_GPU_MEMORY_UTILIZATION="${ATTACKER_VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
export BIAS_VLLM_GPU_MEMORY_UTILIZATION="${BIAS_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
export VLLM_READY_WAIT_STEPS="${VLLM_READY_WAIT_STEPS:-1200}"

exec bash final_experiment_ember_300/scripts/run_vllm_first_eval_openmodel_round05.sh
