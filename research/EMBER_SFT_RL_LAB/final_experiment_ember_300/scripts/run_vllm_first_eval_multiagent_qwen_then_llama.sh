#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
PYTHON="${PYTHON:-/home/haowen/Lab/enter/envs/ai/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/haowen/Lab/enter/envs/vllm/bin/python}"
SCRIPTS_DIR="$ROOT/final_experiment_ember_300/scripts"
DATASET="${DATASET:-$ROOT/final_experiment_ember_300/data/eval_splits_reusemax/first_eval_150.jsonl}"
QWEN_TARGET="/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
LLAMA_TARGET="/data2/guohaowen_data/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1"
ATTACKER="/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B"
BIAS_EXPERT="${BIAS_EXPERT:-/data2/guohaowen_data/huggingface_cache/hub/models--EmergentMethods--Qwen3-4B-BiasExpert/snapshots/02585c80cc5b324228255d9bbb26017ce649dc43}"
BIAS_PROMPT_SOURCE="$ROOT/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py"

TOPIC_START="${TOPIC_START:-1}"
TOPIC_END="${TOPIC_END:-150}"
MAX_DEBATE_ROUND="${MAX_DEBATE_ROUND:-5}"
VARIANTS="${VARIANTS:-BASE+MULTIAGENT,EMBER-PROMPT+MULTIAGENT,EMBER-AGENT+MULTIAGENT}"
MULTIAGENT_PROVOCATEURS="${MULTIAGENT_PROVOCATEURS:-d0,d1,d2}"

TARGET_GPUS="${TARGET_GPUS:-0,1}"
ATTACKER_GPUS="${ATTACKER_GPUS:-2,3}"
BIAS_GPUS="${BIAS_GPUS:-0,1,2,3}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
TARGET_VLLM_GPU_MEMORY_UTILIZATION="${TARGET_VLLM_GPU_MEMORY_UTILIZATION:-$VLLM_GPU_MEMORY_UTILIZATION}"
ATTACKER_VLLM_GPU_MEMORY_UTILIZATION="${ATTACKER_VLLM_GPU_MEMORY_UTILIZATION:-$VLLM_GPU_MEMORY_UTILIZATION}"
BIAS_VLLM_GPU_MEMORY_UTILIZATION="${BIAS_VLLM_GPU_MEMORY_UTILIZATION:-$VLLM_GPU_MEMORY_UTILIZATION}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-24576}"

JOBS_PER_ENDPOINT="${JOBS_PER_ENDPOINT:-8}"
TARGET_CONCURRENCY="${TARGET_CONCURRENCY:-16}"
ATTACKER_CONCURRENCY="${ATTACKER_CONCURRENCY:-32}"
BIAS_ENDPOINT_CONCURRENCY="${BIAS_ENDPOINT_CONCURRENCY:-24}"
BIAS_ROWS_PER_ENDPOINT="${BIAS_ROWS_PER_ENDPOINT:-48}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-vllm_first_eval150_round05_multiagent}"
RUN_ROOT="$ROOT/final_experiment_ember_300/results/${RUN_NAME}_${STAMP}"
LOG_DIR="$ROOT/logs/${RUN_NAME}_${STAMP}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

SERVER_PIDS=()
ATTACKER_ENDPOINTS=()

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
      echo "[vllm-multiagent] ${name} ready on port ${port}"
      return 0
    fi
    sleep 2
  done
  echo "[vllm-multiagent] ${name} failed to become ready on port ${port}" >&2
  return 1
}

start_vllm_server() {
  local gpu="$1"
  local port="$2"
  local model_path="$3"
  local served_name="$4"
  local log_file="$5"
  local gpu_memory_utilization="$6"
  shift 6
  echo "[vllm-multiagent] starting ${served_name} gpu=${gpu} port=${port}"
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
      --gpu-memory-utilization "$gpu_memory_utilization" \
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

start_attacker_servers() {
  local endpoints=()
  IFS=',' read -r -a attacker_gpu_list <<< "$ATTACKER_GPUS"
  for offset in "${!attacker_gpu_list[@]}"; do
    local gpu="${attacker_gpu_list[$offset]}"
    local port=$((8103 + offset))
    start_vllm_server "$gpu" "$port" "$ATTACKER" "attacker" "$LOG_DIR/attacker_gpu${gpu}.log" \
      "$ATTACKER_VLLM_GPU_MEMORY_UTILIZATION"
    endpoints+=("http://127.0.0.1:${port}/v1")
  done
  ATTACKER_ENDPOINTS=("${endpoints[@]}")
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
    start_vllm_server "$gpu" "$port" "$target_model_path" "target" "$LOG_DIR/${target_kind}_target_gpu${gpu}.log" \
      "$TARGET_VLLM_GPU_MEMORY_UTILIZATION"
    endpoints+=("http://127.0.0.1:${port}/v1")
  done

  local endpoint_csv
  endpoint_csv=$(IFS=,; echo "${endpoints[*]}")
  local attacker_endpoint_csv
  attacker_endpoint_csv=$(IFS=,; echo "${ATTACKER_ENDPOINTS[*]}")

  "$PYTHON" "$SCRIPTS_DIR/vllm_generate_debates.py" \
    --cmv-path "$DATASET" \
    --topic-start "$TOPIC_START" \
    --topic-end "$TOPIC_END" \
    --max-debate-round "$MAX_DEBATE_ROUND" \
    --variants "$VARIANTS" \
    --target-endpoints "$endpoint_csv" \
    --attacker-endpoints "$attacker_endpoint_csv" \
    --target-model target \
    --attacker-model attacker \
    --target-name "$target_name" \
    --provocateur-name qwen_attacker \
    --multiagent-provocateur-names "$MULTIAGENT_PROVOCATEURS" \
    --target-concurrency "$TARGET_CONCURRENCY" \
    --attacker-concurrency "$ATTACKER_CONCURRENCY" \
    --jobs-per-endpoint "$JOBS_PER_ENDPOINT" \
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
    start_vllm_server "$gpu" "$port" "$BIAS_EXPERT" "biasexpert" "$LOG_DIR/biasexpert_${target_kind}_gpu${gpu}.log" \
      "$BIAS_VLLM_GPU_MEMORY_UTILIZATION"
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
    --endpoint-concurrency "$BIAS_ENDPOINT_CONCURRENCY" \
    --rows-per-endpoint "$BIAS_ROWS_PER_ENDPOINT" \
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
  echo "multiagent_provocateurs=$MULTIAGENT_PROVOCATEURS"
  echo "target_gpus=$TARGET_GPUS"
  echo "attacker_gpus=$ATTACKER_GPUS"
  echo "bias_gpus=$BIAS_GPUS"
  echo "vllm_gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
  echo "target_vllm_gpu_memory_utilization=$TARGET_VLLM_GPU_MEMORY_UTILIZATION"
  echo "attacker_vllm_gpu_memory_utilization=$ATTACKER_VLLM_GPU_MEMORY_UTILIZATION"
  echo "bias_vllm_gpu_memory_utilization=$BIAS_VLLM_GPU_MEMORY_UTILIZATION"
  echo "vllm_max_model_len=$VLLM_MAX_MODEL_LEN"
  echo "vllm_max_num_seqs=$VLLM_MAX_NUM_SEQS"
  echo "vllm_max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
  echo "jobs_per_endpoint=$JOBS_PER_ENDPOINT"
  echo "target_concurrency=$TARGET_CONCURRENCY"
  echo "attacker_concurrency=$ATTACKER_CONCURRENCY"
} > "$RUN_ROOT/RUN_PATHS.txt"

start_attacker_servers
run_generation_for_target "qwen" "qwen" "$QWEN_TARGET" 8110 "$RUN_ROOT/qwen"
run_generation_for_target "llama" "llama" "$LLAMA_TARGET" 8120 "$RUN_ROOT/llama"
stop_current_servers

run_bias_scoring "qwen" "$RUN_ROOT/qwen"
run_bias_scoring "llama" "$RUN_ROOT/llama"

echo "[vllm-multiagent] done: $RUN_ROOT"
