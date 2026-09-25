#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_TAG="${MODEL_TAG:-qwen3_4b_instruct_2507_local}"
DATASET_BUCKET="${DATASET_BUCKET:-qwen}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/$MODEL_TAG}"

mkdir -p "$RUN_DIR"

python "$ROOT/scripts/prepare_topic_splits.py"
python "$ROOT/scripts/build_training_corpus.py"

python "$ROOT/scripts/train_risk_head.py" \
  --model-name-or-path "$MODEL_NAME" \
  --train-data "$ROOT/datasets/$DATASET_BUCKET/risk_head_train.jsonl" \
  --dev-data "$ROOT/datasets/$DATASET_BUCKET/risk_head_dev.jsonl" \
  --output-dir "$RUN_DIR/risk_head" \
  --batch-size 2 \
  --epochs 3 \
  --max-length 1024

python "$ROOT/scripts/train_sft_adapters.py" \
  --model-name-or-path "$MODEL_NAME" \
  --train-data "$ROOT/datasets/$DATASET_BUCKET/sft_train.jsonl" \
  --dev-data "$ROOT/datasets/$DATASET_BUCKET/sft_dev.jsonl" \
  --risk-head-checkpoint "$RUN_DIR/risk_head/best" \
  --output-dir "$RUN_DIR/sft" \
  --batch-size 1 \
  --epochs 3 \
  --max-length 1024

python "$ROOT/scripts/train_rl_policy.py" \
  --model-name-or-path "$MODEL_NAME" \
  --train-data "$ROOT/datasets/$DATASET_BUCKET/rl_train.jsonl" \
  --sft-checkpoint "$RUN_DIR/sft/best" \
  --output-dir "$RUN_DIR/rl" \
  --batch-size 1 \
  --epochs 1 \
  --samples-per-prompt 2 \
  --max-prompt-length 1024 \
  --max-new-tokens 160

python "$ROOT/scripts/evaluate_policy.py" \
  --model-name-or-path "$MODEL_NAME" \
  --eval-data "$ROOT/datasets/$DATASET_BUCKET/rl_test.jsonl" \
  --output-dir "$RUN_DIR/eval_base" \
  --max-prompt-length 1024 \
  --max-new-tokens 160

python "$ROOT/scripts/evaluate_policy.py" \
  --model-name-or-path "$MODEL_NAME" \
  --checkpoint "$RUN_DIR/rl/final" \
  --eval-data "$ROOT/datasets/$DATASET_BUCKET/rl_test.jsonl" \
  --output-dir "$RUN_DIR/eval_rl" \
  --max-prompt-length 1024 \
  --max-new-tokens 160

python "$ROOT/scripts/plot_metrics.py" \
  --risk-log "$RUN_DIR/risk_head/risk_head_log.jsonl" \
  --sft-log "$RUN_DIR/sft/sft_log.jsonl" \
  --rl-log "$RUN_DIR/rl/rl_log.jsonl" \
  --eval-metrics "$RUN_DIR/eval_base/metrics.json" "$RUN_DIR/eval_rl/metrics.json" \
  --eval-labels "base" "rl" \
  --output-dir "$RUN_DIR/plots"
