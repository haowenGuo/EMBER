#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
PYTHON="${PYTHON:-/home/haowen/Lab/enter/envs/ai/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/haowen/Lab/enter/envs/vllm/bin/python}"
SCRIPTS_DIR="$ROOT/final_experiment_ember_300/scripts"
DATASET="${DATASET:-$ROOT/final_experiment_ember_300/data/final_ember_topics_300.jsonl}"
QWEN_TARGET="/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
LLAMA_TARGET="/data2/guohaowen_data/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1"
ATTACKER="/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B"
BIAS_EXPERT="${BIAS_EXPERT:-/data2/guohaowen_data/huggingface_cache/hub/models--EmergentMethods--Qwen3-4B-BiasExpert/snapshots/02585c80cc5b324228255d9bbb26017ce649dc43}"
BIAS_PROMPT_SOURCE="$ROOT/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py"
TOPIC_START="${TOPIC_START:-1}"
TOPIC_END="${TOPIC_END:-150}"
MAX_DEBATE_ROUND="${MAX_DEBATE_ROUND:-5}"
VARIANTS="${VARIANTS:-BASE,EMBER-PROMPT,EMBER-AGENT}"
TARGET_GPUS="${TARGET_GPUS:-0,1,2}"
BIAS_GPUS="${BIAS_GPUS:-0,1,2,3}"
ATTACKER_GPU="${ATTACKER_GPU:-3}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-24576}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-vllm_final150_round05}"
RUN_ROOT="$ROOT/final_experiment_ember_300/results/${RUN_NAME}_${STAMP}"
LOG_DIR="$ROOT/logs/${RUN_NAME}_${STAMP}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

SERVER_PIDS=()

cleanup_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup_servers EXIT

require_vllm() {
  "$VLLM_PYTHON" - <<'PY'
import vllm
print("vllm", getattr(vllm, "__version__", "unknown"))
PY
}

wait_vllm() {
  local port="$1"
  local name="$2"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[vllm-run] ${name} ready on port ${port}"
      return 0
    fi
    sleep 2
  done
  echo "[vllm-run] ${name} failed to become ready on port ${port}" >&2
  return 1
}

start_vllm_server() {
  local gpu="$1"
  local port="$2"
  local model_path="$3"
  local served_name="$4"
  local log_file="$5"
  shift 5
  echo "[vllm-run] starting ${served_name} gpu=${gpu} port=${port}"
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
      --no-enable-log-requests \
      "$@"
  ) > "$log_file" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  echo "$pid" > "${log_file%.log}.pid"
  wait_vllm "$port" "$served_name"
}

stop_current_servers() {
  cleanup_servers
  SERVER_PIDS=()
  sleep 5
}

stop_servers_from() {
  local start_index="$1"
  local total="${#SERVER_PIDS[@]}"
  for ((idx = start_index; idx < total; idx++)); do
    local pid="${SERVER_PIDS[$idx]}"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  SERVER_PIDS=("${SERVER_PIDS[@]:0:start_index}")
  sleep 5
}

run_generation_for_target() {
  local target_kind="$1"
  local target_name="$2"
  local target_model_path="$3"
  local base_port="$4"
  local out_dir="$5"
  local endpoints=()
  local pid_start_index="${#SERVER_PIDS[@]}"
  IFS=',' read -r -a target_gpu_list <<< "$TARGET_GPUS"

  for offset in "${!target_gpu_list[@]}"; do
    local gpu="${target_gpu_list[$offset]}"
    local port=$((base_port + offset))
    start_vllm_server "$gpu" "$port" "$target_model_path" "target" "$LOG_DIR/${target_kind}_target_gpu${gpu}.log"
    endpoints+=("http://127.0.0.1:${port}/v1")
  done

  local endpoint_csv
  endpoint_csv=$(IFS=,; echo "${endpoints[*]}")
  "$PYTHON" "$SCRIPTS_DIR/vllm_generate_debates.py" \
    --cmv-path "$DATASET" \
    --topic-start "$TOPIC_START" \
    --topic-end "$TOPIC_END" \
    --max-debate-round "$MAX_DEBATE_ROUND" \
    --variants "$VARIANTS" \
    --target-endpoints "$endpoint_csv" \
    --attacker-endpoint "http://127.0.0.1:8103/v1" \
    --target-model target \
    --attacker-model attacker \
    --target-name "$target_name" \
    --provocateur-name qwen_attacker \
    --target-concurrency 24 \
    --attacker-concurrency 64 \
    --jobs-per-endpoint 18 \
    --attacker-disable-thinking \
    --output-dir "$out_dir" \
    > "$LOG_DIR/${target_kind}_generation.log" 2>&1

  stop_servers_from "$pid_start_index"
}

run_bias_scoring() {
  local target_kind="$1"
  local out_dir="$2"
  local endpoints=()
  IFS=',' read -r -a bias_gpu_list <<< "$BIAS_GPUS"
  for offset in "${!bias_gpu_list[@]}"; do
    local gpu="${bias_gpu_list[$offset]}"
    local port=$((8130 + offset))
    start_vllm_server "$gpu" "$port" "$BIAS_EXPERT" "biasexpert" "$LOG_DIR/biasexpert_${target_kind}_gpu${gpu}.log"
    endpoints+=("http://127.0.0.1:${port}/v1")
  done
  local endpoint_csv
  endpoint_csv=$(IFS=,; echo "${endpoints[*]}")
  "$PYTHON" "$SCRIPTS_DIR/vllm_score_biasexpert.py" \
    --generated-jsonl "$out_dir/generated_rounds.jsonl" \
    --output-dir "$out_dir" \
    --bias-endpoints "$endpoint_csv" \
    --bias-model biasexpert \
    --bias-prompt-source "$BIAS_PROMPT_SOURCE" \
    --endpoint-concurrency 24 \
    --rows-per-endpoint 48 \
    > "$LOG_DIR/${target_kind}_biasexpert_scoring.log" 2>&1
  stop_current_servers
}

require_vllm

{
  echo "run_root=$RUN_ROOT"
  echo "log_dir=$LOG_DIR"
  echo "dataset=$DATASET"
  echo "qwen_target=$QWEN_TARGET"
  echo "llama_target=$LLAMA_TARGET"
  echo "attacker=$ATTACKER"
  echo "bias_expert=$BIAS_EXPERT"
  echo "topic_start=$TOPIC_START"
  echo "topic_end=$TOPIC_END"
  echo "max_debate_round=$MAX_DEBATE_ROUND"
  echo "variants=$VARIANTS"
  echo "target_gpus=$TARGET_GPUS"
  echo "bias_gpus=$BIAS_GPUS"
  echo "attacker_gpu=$ATTACKER_GPU"
  echo "vllm_gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
  echo "vllm_max_model_len=$VLLM_MAX_MODEL_LEN"
  echo "vllm_max_num_seqs=$VLLM_MAX_NUM_SEQS"
  echo "vllm_max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
} > "$RUN_ROOT/RUN_PATHS.txt"

# One shared attacker service. Target services are data-parallel workers on the other three GPUs.
start_vllm_server "$ATTACKER_GPU" 8103 "$ATTACKER" "attacker" "$LOG_DIR/attacker_gpu${ATTACKER_GPU}.log"
run_generation_for_target "qwen" "qwen" "$QWEN_TARGET" 8110 "$RUN_ROOT/qwen"
run_generation_for_target "llama" "llama" "$LLAMA_TARGET" 8120 "$RUN_ROOT/llama"
stop_current_servers

# BiasExpert is intentionally launched only after all generation is done.
run_bias_scoring "qwen" "$RUN_ROOT/qwen"
run_bias_scoring "llama" "$RUN_ROOT/llama"

echo "[vllm-run] done: $RUN_ROOT"
