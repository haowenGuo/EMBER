#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
PYTHON="${PYTHON:-/home/haowen/Lab/enter/envs/ai/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/final_experiment_ember_300/results/vllm_first_eval150_round05_reusemax_20260529_052000}"
RUN_NAME="${RUN_NAME:-openai_gpt55_first_eval_bias_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/final_experiment_ember_300/results/$RUN_NAME}"
BIAS_PROMPT_SOURCE="${BIAS_PROMPT_SOURCE:-$ROOT/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py}"
MODEL="${MODEL:-${OPENAI_EVAL_MODEL:-gpt-5.5}}"
MODEL_LABEL="${MODEL_LABEL:-gpt55}"
LEGACY_CONFIG="${LEGACY_CONFIG:-${OPENAI_LEGACY_CONFIG:-}}"
API_MODE="${API_MODE:-responses}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-5000}"
TEMPERATURE="${TEMPERATURE:-0.3}"
TOP_P="${TOP_P:-0.95}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-240}"
MAX_RETRIES="${MAX_RETRIES:-3}"
TOPIC_START="${TOPIC_START:-}"
TOPIC_END="${TOPIC_END:-}"
VARIANTS="${VARIANTS:-}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_ROOT"

if [[ -z "${OPENAI_API_KEY:-}" && -f "$HOME/.bashrc" ]]; then
  # Some fat20 shells define provider keys in .bashrc without exporting them.
  # Source it here so long-running batch scripts see the same credentials as SSH checks.
  set +u
  # shellcheck source=/dev/null
  source "$HOME/.bashrc"
  set -u
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[openai-first-eval] OPENAI_API_KEY is required." >&2
  exit 2
fi

write_paths() {
  cat > "$OUTPUT_ROOT/RUN_PATHS.txt" <<EOF
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
bias_prompt_source=$BIAS_PROMPT_SOURCE
model=$MODEL
model_label=$MODEL_LABEL
legacy_config=$LEGACY_CONFIG
api_mode=$API_MODE
concurrency=$CONCURRENCY
max_output_tokens=$MAX_OUTPUT_TOKENS
temperature=$TEMPERATURE
top_p=$TOP_P
topic_start=$TOPIC_START
topic_end=$TOPIC_END
variants=$VARIANTS
limit=$LIMIT
EOF
}

score_model() {
  local model_name="$1"
  local input_jsonl="$RUN_ROOT/$model_name/scored_rounds.jsonl"
  local output_dir="$OUTPUT_ROOT/$model_name"
  if [[ ! -f "$input_jsonl" ]]; then
    echo "[openai-first-eval] missing input: $input_jsonl" >&2
    exit 3
  fi

  local args=(
    "$ROOT/final_experiment_ember_300/scripts/openai_score_biasexpert_prompt.py"
    --input-jsonl "$input_jsonl"
    --output-dir "$output_dir"
    --bias-prompt-source "$BIAS_PROMPT_SOURCE"
    --model "$MODEL"
    --model-label "$MODEL_LABEL"
    --api-mode "$API_MODE"
    --concurrency "$CONCURRENCY"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --request-timeout "$REQUEST_TIMEOUT"
    --max-retries "$MAX_RETRIES"
  )
  if [[ -n "$LEGACY_CONFIG" ]]; then args+=(--legacy-config "$LEGACY_CONFIG"); fi
  if [[ -n "$TOPIC_START" ]]; then args+=(--topic-start "$TOPIC_START"); fi
  if [[ -n "$TOPIC_END" ]]; then args+=(--topic-end "$TOPIC_END"); fi
  if [[ -n "$VARIANTS" ]]; then args+=(--variants "$VARIANTS"); fi
  if [[ -n "$LIMIT" ]]; then args+=(--limit "$LIMIT"); fi

  echo "[openai-first-eval] scoring $model_name input=$input_jsonl output=$output_dir"
  "$PYTHON" "${args[@]}"
}

cd "$ROOT"
write_paths
score_model qwen
score_model llama
echo "[openai-first-eval] done $OUTPUT_ROOT"
