#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/final_mixed_train_second_eval_${STAMP}.log"
PID_FILE="$LOG_DIR/final_mixed_train_second_eval.pid"
mkdir -p "$LOG_DIR"

cd "$ROOT"
nohup bash final_experiment_ember_300/scripts/run_final_mixed_train_then_second_eval.sh > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "PID=$(cat "$PID_FILE")"
echo "LOG=$LOG_FILE"
