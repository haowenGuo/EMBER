#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
PYTHON="${PYTHON:-/home/haowen/Lab/enter/envs/ai/bin/python}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required}"
DATASET_BUCKET="${DATASET_BUCKET:-final_first_eval_mixed}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/$MODEL_TAG}"
DTYPE="${DTYPE:-bf16}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"

RISK_BATCH_SIZE="${RISK_BATCH_SIZE:-2}"
RISK_EPOCHS="${RISK_EPOCHS:-3}"
RISK_LR="${RISK_LR:-1e-4}"

SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_LR="${SFT_LR:-2e-4}"
SFT_RISK_AUX_WEIGHT="${SFT_RISK_AUX_WEIGHT:-0.05}"
SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-0}"
SFT_EARLY_STOPPING_MIN_DELTA="${SFT_EARLY_STOPPING_MIN_DELTA:-0.0}"

RL_BATCH_SIZE="${RL_BATCH_SIZE:-1}"
RL_EPOCHS="${RL_EPOCHS:-1}"
RL_LR="${RL_LR:-5e-5}"
RL_SAMPLES_PER_PROMPT="${RL_SAMPLES_PER_PROMPT:-2}"
RL_RISK_WEIGHT="${RL_RISK_WEIGHT:-1.0}"
RL_STANCE_WEIGHT="${RL_STANCE_WEIGHT:-0.4}"
RL_LENGTH_WEIGHT="${RL_LENGTH_WEIGHT:-0.1}"
RL_REFUSAL_WEIGHT="${RL_REFUSAL_WEIGHT:-0.4}"
RL_REPETITION_WEIGHT="${RL_REPETITION_WEIGHT:-0.4}"
RL_OVERLENGTH_WEIGHT="${RL_OVERLENGTH_WEIGHT:-0.4}"
RL_OVERLENGTH_CHARS="${RL_OVERLENGTH_CHARS:-2200}"
RL_OVERLENGTH_HARD_CHARS="${RL_OVERLENGTH_HARD_CHARS:-3600}"
RL_KL_WEIGHT="${RL_KL_WEIGHT:-0.03}"
RL_USE_SEPARATE_SCORER="${RL_USE_SEPARATE_SCORER:-0}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"

DATASET_DIR="$ROOT/datasets/$DATASET_BUCKET"
LOG_DIR="$ROOT/logs/${MODEL_TAG}_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ ! -s "$DATASET_DIR/risk_head_train.jsonl" || ! -s "$DATASET_DIR/sft_train.jsonl" || ! -s "$DATASET_DIR/rl_train.jsonl" ]]; then
  echo "[train-only] missing training files in $DATASET_DIR" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_dist() {
  local stage="$1"
  shift
  echo "[train-only] stage=$stage model_tag=$MODEL_TAG run_dir=$RUN_DIR"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" "$@" 2>&1 | tee "$LOG_DIR/${stage}.log"
}

run_dist risk_head "$ROOT/scripts/train_risk_head.py" \
  --model-name-or-path "$MODEL_NAME" \
  --train-data "$DATASET_DIR/risk_head_train.jsonl" \
  --dev-data "$DATASET_DIR/risk_head_dev.jsonl" \
  --output-dir "$RUN_DIR/risk_head" \
  --batch-size "$RISK_BATCH_SIZE" \
  --epochs "$RISK_EPOCHS" \
  --learning-rate "$RISK_LR" \
  --max-length "$MAX_LENGTH" \
  --dtype "$DTYPE"

run_dist sft "$ROOT/scripts/train_sft_adapters.py" \
  --model-name-or-path "$MODEL_NAME" \
  --train-data "$DATASET_DIR/sft_train.jsonl" \
  --dev-data "$DATASET_DIR/sft_dev.jsonl" \
  --risk-head-checkpoint "$RUN_DIR/risk_head/best" \
  --output-dir "$RUN_DIR/sft" \
  --batch-size "$SFT_BATCH_SIZE" \
  --epochs "$SFT_EPOCHS" \
  --learning-rate "$SFT_LR" \
  --risk-head-aux-weight "$SFT_RISK_AUX_WEIGHT" \
  --early-stopping-patience "$SFT_EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-delta "$SFT_EARLY_STOPPING_MIN_DELTA" \
  --max-length "$MAX_LENGTH" \
  --dtype "$DTYPE"

run_dist rl "$ROOT/scripts/train_rl_policy.py" \
  --model-name-or-path "$MODEL_NAME" \
  --train-data "$DATASET_DIR/rl_train.jsonl" \
  --sft-checkpoint "$RUN_DIR/sft/best" \
  --output-dir "$RUN_DIR/rl" \
  --batch-size "$RL_BATCH_SIZE" \
  --epochs "$RL_EPOCHS" \
  --samples-per-prompt "$RL_SAMPLES_PER_PROMPT" \
  --learning-rate "$RL_LR" \
  --risk-weight "$RL_RISK_WEIGHT" \
  --stance-weight "$RL_STANCE_WEIGHT" \
  --length-weight "$RL_LENGTH_WEIGHT" \
  --refusal-weight "$RL_REFUSAL_WEIGHT" \
  --repetition-weight "$RL_REPETITION_WEIGHT" \
  --overlength-weight "$RL_OVERLENGTH_WEIGHT" \
  --overlength-chars "$RL_OVERLENGTH_CHARS" \
  --overlength-hard-chars "$RL_OVERLENGTH_HARD_CHARS" \
  --kl-weight "$RL_KL_WEIGHT" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  $(if [[ "$RL_USE_SEPARATE_SCORER" == "1" ]]; then echo "--use-separate-scorer"; fi)

cat > "$RUN_DIR/TRAINING_DONE.txt" <<EOF
model_name=$MODEL_NAME
model_tag=$MODEL_TAG
dataset_bucket=$DATASET_BUCKET
run_dir=$RUN_DIR
log_dir=$LOG_DIR
completed_at=$(date '+%Y-%m-%d %H:%M:%S')
EOF

echo "[train-only] done $RUN_DIR"
