#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/haowen/Lab/EMBER_SFT_RL_LAB"
PYTHON="/home/haowen/Lab/enter/envs/ai/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="$ROOT/formal_eval_cmv101_200_fiveway_exact_fullcontext_llama_from_qwen"
DATASET="$ROOT/final_experiment_ember_300/data/final_ember_topics_300.jsonl"
TARGET="/data2/guohaowen_data/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1"
ATTACKER="/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B"
LOG_DIR="$ROOT/logs"
TOPIC_START="${TOPIC_START:-1}"
TOPIC_END="${TOPIC_END:-12}"
MAX_DEBATE_ROUND="${MAX_DEBATE_ROUND:-5}"
PROFILES="${PROFILES:-baseline pressure_ladder identity_specificity consistency_trap}"

mkdir -p "$LOG_DIR"
cd "$EVAL_DIR"

echo "[PROMPT-SMOKE] stamp=$STAMP"
echo "[PROMPT-SMOKE] topics=${TOPIC_START}-${TOPIC_END} rounds=0-${MAX_DEBATE_ROUND}"
echo "[PROMPT-SMOKE] profiles=$PROFILES"

for PROFILE in $PROFILES; do
  OUT_DIR="$EVAL_DIR/outputs/prompt_tune_llama_${PROFILE}_t${TOPIC_START}_${TOPIC_END}_r${MAX_DEBATE_ROUND}_${STAMP}"
  echo "[PROMPT-SMOKE] running profile=$PROFILE output=$OUT_DIR"
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" run_fiveway_exact_biasexpert_eval.py \
    --worker-mode \
    --worker-id 0 \
    --model-name-or-path "$TARGET" \
    --cmv-path "$DATASET" \
    --topic-start "$TOPIC_START" \
    --topic-end "$TOPIC_END" \
    --max-debate-round "$MAX_DEBATE_ROUND" \
    --variants BASE \
    --provocateur-engine local \
    --local-provocateur-model-path "$ATTACKER" \
    --local-provocateur-device cuda:1 \
    --local-provocateur-max-new-tokens 1024 \
    --attack-prompt-profile "$PROFILE" \
    --output-dir "$OUT_DIR" \
    --max-parallel-gpus 1
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" run_fiveway_exact_biasexpert_eval.py \
    --model-name-or-path "$TARGET" \
    --cmv-path "$DATASET" \
    --topic-start "$TOPIC_START" \
    --topic-end "$TOPIC_END" \
    --max-debate-round "$MAX_DEBATE_ROUND" \
    --variants BASE \
    --provocateur-engine local \
    --local-provocateur-model-path "$ATTACKER" \
    --local-provocateur-device cuda:1 \
    --local-provocateur-max-new-tokens 1024 \
    --attack-prompt-profile "$PROFILE" \
    --output-dir "$OUT_DIR" \
    --max-parallel-gpus 1
  echo "[PROMPT-SMOKE] done profile=$PROFILE"
done

echo "[PROMPT-SMOKE] completed stamp=$STAMP"
