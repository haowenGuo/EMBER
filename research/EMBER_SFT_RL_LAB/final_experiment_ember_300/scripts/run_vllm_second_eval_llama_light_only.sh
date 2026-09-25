#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
PYTHON="${PYTHON:-/home/haowen/Lab/enter/envs/ai/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/haowen/Lab/enter/envs/vllm/bin/python}"
SCRIPTS_DIR="$ROOT/final_experiment_ember_300/scripts"
DATASET="${DATASET:-$ROOT/final_experiment_ember_300/data/eval_splits_reusemax/second_eval_150.jsonl}"

LLAMA_TARGET="${LLAMA_TARGET:-/data2/guohaowen_data/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1}"
ATTACKER="${ATTACKER:-/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B}"
BIAS_EXPERT="${BIAS_EXPERT:-/data2/guohaowen_data/huggingface_cache/hub/models--EmergentMethods--Qwen3-4B-BiasExpert/snapshots/02585c80cc5b324228255d9bbb26017ce649dc43}"
BIAS_PROMPT_SOURCE="${BIAS_PROMPT_SOURCE:-$ROOT/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py}"

LLAMA_RUN_DIR="${LLAMA_RUN_DIR:-$ROOT/runs/llama3_1_8b_final_first_eval_mixed_light_sft1_rl1_lr2e6_20260601_142311}"
BASE_PROMPT_AGENT_SOURCE="${BASE_PROMPT_AGENT_SOURCE:-$ROOT/final_experiment_ember_300/results/second_eval150_round05_five_methods_from_first_eval_mixed_20260529_164351/llama/base_prompt_agent/generated_rounds.jsonl}"

TOPIC_START="${TOPIC_START:-1}"
TOPIC_END="${TOPIC_END:-150}"
MAX_DEBATE_ROUND="${MAX_DEBATE_ROUND:-5}"

ATTACKER_GPU="${ATTACKER_GPU:-0}"
RISK_GATED_GPUS="${RISK_GATED_GPUS:-1,2,3}"
BIAS_GPUS="${BIAS_GPUS:-0,1,2,3}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-12288}"

RISK_GATED_MAX_PROMPT_LENGTH="${RISK_GATED_MAX_PROMPT_LENGTH:-4096}"
RISK_GATED_TARGET_MAX_TOKENS="${RISK_GATED_TARGET_MAX_TOKENS:-512}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-second_eval150_round05_llama_light_sft1_rl1_full}"
RUN_ROOT="$ROOT/final_experiment_ember_300/results/${RUN_NAME}_${STAMP}"
LOG_DIR="$ROOT/logs/${RUN_NAME}_${STAMP}"
mkdir -p "$RUN_ROOT/llama/base_prompt_agent" "$RUN_ROOT/llama/risk_gated_shards" "$LOG_DIR"

SERVER_PIDS=()

cleanup_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_servers EXIT

wait_vllm() {
  local port="$1"
  local name="$2"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[llama-light-second-eval] ${name} ready on port ${port}"
      return 0
    fi
    sleep 2
  done
  echo "[llama-light-second-eval] ${name} failed to become ready on port ${port}" >&2
  return 1
}

start_vllm_server() {
  local gpu="$1"
  local port="$2"
  local model_path="$3"
  local served_name="$4"
  local log_file="$5"
  echo "[llama-light-second-eval] starting ${served_name} gpu=${gpu} port=${port}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export PATH="$(dirname "$VLLM_PYTHON"):$PATH"
    exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "$port" \
      --model "$model_path" \
      --served-model-name "$served_name" \
      --trust-remote-code \
      --dtype auto \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
      --enable-prefix-caching \
      --no-enable-log-requests
  ) > "$log_file" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  echo "$pid" > "${log_file%.log}.pid"
  wait_vllm "$port" "$served_name"
}

stop_all_servers() {
  cleanup_servers
  SERVER_PIDS=()
  sleep 5
}

run_risk_gated_two_methods() {
  local out_dir="$1"
  IFS=',' read -r -a worker_gpus <<< "$RISK_GATED_GPUS"
  local worker_count="${#worker_gpus[@]}"
  local pids=()

  for worker_index in "${!worker_gpus[@]}"; do
    local gpu="${worker_gpus[$worker_index]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON" "$SCRIPTS_DIR/risk_gated_generate_debates.py" \
        --cmv-path "$DATASET" \
        --topic-start "$TOPIC_START" \
        --topic-end "$TOPIC_END" \
        --max-debate-round "$MAX_DEBATE_ROUND" \
        --variants "SFT,SFT+RL" \
        --model-name-or-path "$LLAMA_TARGET" \
        --sft-checkpoint "$LLAMA_RUN_DIR/sft/best" \
        --rl-checkpoint "$LLAMA_RUN_DIR/rl/final" \
        --dtype bf16 \
        --attacker-endpoint "http://127.0.0.1:8103/v1" \
        --attacker-model attacker \
        --target-name llama \
        --provocateur-name qwen_attacker \
        --output-dir "$out_dir" \
        --worker-index "$worker_index" \
        --worker-count "$worker_count" \
        --max-prompt-length "$RISK_GATED_MAX_PROMPT_LENGTH" \
        --target-max-tokens "$RISK_GATED_TARGET_MAX_TOKENS" \
        --attacker-disable-thinking
    ) > "$LOG_DIR/llama_risk_gated_worker${worker_index}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

merge_generated() {
  local target_dir="$RUN_ROOT/llama"
  local inputs=("$target_dir/base_prompt_agent/generated_rounds.jsonl")
  for shard in "$target_dir/risk_gated_shards"/generated_rounds.worker*.jsonl; do
    inputs+=("$shard")
  done
  "$PYTHON" "$SCRIPTS_DIR/merge_generated_rounds.py" \
    --inputs "${inputs[@]}" \
    --output "$target_dir/generated_rounds.jsonl" \
    > "$LOG_DIR/llama_merge_generated.log" 2>&1
}

start_bias_servers() {
  IFS=',' read -r -a bias_gpu_list <<< "$BIAS_GPUS"
  for offset in "${!bias_gpu_list[@]}"; do
    local gpu="${bias_gpu_list[$offset]}"
    local port=$((8130 + offset))
    start_vllm_server "$gpu" "$port" "$BIAS_EXPERT" "biasexpert" "$LOG_DIR/biasexpert_gpu${gpu}.log"
  done
}

score_llama() {
  local endpoints=()
  IFS=',' read -r -a bias_gpu_list <<< "$BIAS_GPUS"
  for offset in "${!bias_gpu_list[@]}"; do
    local port=$((8130 + offset))
    endpoints+=("http://127.0.0.1:${port}/v1")
  done
  local endpoint_csv
  endpoint_csv=$(IFS=,; echo "${endpoints[*]}")
  "$PYTHON" "$SCRIPTS_DIR/vllm_score_biasexpert.py" \
    --generated-jsonl "$RUN_ROOT/llama/generated_rounds.jsonl" \
    --output-dir "$RUN_ROOT/llama" \
    --bias-endpoints "$endpoint_csv" \
    --bias-model biasexpert \
    --bias-prompt-source "$BIAS_PROMPT_SOURCE" \
    --endpoint-concurrency 24 \
    --rows-per-endpoint 48 \
    > "$LOG_DIR/llama_biasexpert_scoring.log" 2>&1
}

{
  echo "run_root=$RUN_ROOT"
  echo "log_dir=$LOG_DIR"
  echo "dataset=$DATASET"
  echo "llama_run_dir=$LLAMA_RUN_DIR"
  echo "base_prompt_agent_source=$BASE_PROMPT_AGENT_SOURCE"
  echo "expected_rows=4500"
  echo "note=BASE, EMBER-PROMPT, and EMBER-AGENT generations are reused because they do not depend on the trained adapter; SFT and SFT+RL are regenerated from the new Llama run."
  echo "risk_gated_max_prompt_length=$RISK_GATED_MAX_PROMPT_LENGTH"
  echo "risk_gated_target_max_tokens=$RISK_GATED_TARGET_MAX_TOKENS"
} > "$RUN_ROOT/RUN_PATHS.txt"

if [[ ! -f "$BASE_PROMPT_AGENT_SOURCE" ]]; then
  echo "Missing base/prompt/agent source: $BASE_PROMPT_AGENT_SOURCE" >&2
  exit 2
fi

cp "$BASE_PROMPT_AGENT_SOURCE" "$RUN_ROOT/llama/base_prompt_agent/generated_rounds.jsonl"
echo "[llama-light-second-eval] reused base/prompt/agent rows $(wc -l < "$RUN_ROOT/llama/base_prompt_agent/generated_rounds.jsonl")"

start_vllm_server "$ATTACKER_GPU" 8103 "$ATTACKER" "attacker" "$LOG_DIR/attacker_gpu${ATTACKER_GPU}.log"
run_risk_gated_two_methods "$RUN_ROOT/llama/risk_gated_shards"
stop_all_servers

merge_generated
echo "[llama-light-second-eval] merged rows $(wc -l < "$RUN_ROOT/llama/generated_rounds.jsonl")"

start_bias_servers
score_llama
stop_all_servers

echo "[llama-light-second-eval] scored rows $(wc -l < "$RUN_ROOT/llama/scored_rounds.jsonl")"
echo "[llama-light-second-eval] done $RUN_ROOT"
