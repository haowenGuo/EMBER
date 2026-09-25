#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/haowen/Lab/EMBER_SFT_RL_LAB}"
TRAIN_SCRIPT="$ROOT/final_experiment_ember_300/scripts/run_risk_gated_training_only.sh"
EVAL_SCRIPT="$ROOT/final_experiment_ember_300/scripts/run_vllm_second_eval_five_methods_qwen_then_llama.sh"

QWEN_TARGET="${QWEN_TARGET:-/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
LLAMA_TARGET="${LLAMA_TARGET:-/data2/guohaowen_data/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1}"
DATASET_BUCKET="${DATASET_BUCKET:-final_first_eval_mixed}"

QWEN_MODEL_TAG="${QWEN_MODEL_TAG:-qwen3_4b_final_first_eval_mixed}"
LLAMA_MODEL_TAG="${LLAMA_MODEL_TAG:-llama3_1_8b_final_first_eval_mixed}"

echo "[final-pipeline] build/check mixed FIRST_EVAL corpus"
bash "$ROOT/final_experiment_ember_300/scripts/build_first_eval_mixed_training_corpus.sh"

echo "[final-pipeline] train Qwen risk-gated algorithm"
MODEL_NAME="$QWEN_TARGET" \
MODEL_TAG="$QWEN_MODEL_TAG" \
DATASET_BUCKET="$DATASET_BUCKET" \
RUN_DIR="$ROOT/runs/$QWEN_MODEL_TAG" \
bash "$TRAIN_SCRIPT"

echo "[final-pipeline] train Llama risk-gated algorithm"
MODEL_NAME="$LLAMA_TARGET" \
MODEL_TAG="$LLAMA_MODEL_TAG" \
DATASET_BUCKET="$DATASET_BUCKET" \
RUN_DIR="$ROOT/runs/$LLAMA_MODEL_TAG" \
bash "$TRAIN_SCRIPT"

echo "[final-pipeline] run SECOND_EVAL five-method evaluation"
QWEN_RUN_DIR="$ROOT/runs/$QWEN_MODEL_TAG" \
LLAMA_RUN_DIR="$ROOT/runs/$LLAMA_MODEL_TAG" \
RUN_NAME="${RUN_NAME:-second_eval150_round05_five_methods_from_first_eval_mixed}" \
bash "$EVAL_SCRIPT"

echo "[final-pipeline] done"
