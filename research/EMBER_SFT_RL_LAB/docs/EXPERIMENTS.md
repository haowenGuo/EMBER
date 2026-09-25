# Experiment Plan

## Data protocol

Use topic-level split, not sample-level split.

Recommended default:

- train: 70%
- dev: 15%
- test: 15%

All final claims should come from held-out test topics only.

## Recommended base models

Current default main run:

1. `/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`

Recommended comparison runs:

1. `Qwen/Qwen2.5-0.5B-Instruct`
2. `Qwen/Qwen2.5-1.5B-Instruct`

Use the local `Qwen3-4B-Instruct-2507` snapshot for the main server run. Use the smaller Qwen2.5 models only for debugging and lightweight ablations.

## Main ablations

1. Base model
2. Risk head only
3. SFT adapters
4. SFT + RL adapters

Optional:

5. Shared single adapter instead of 6-dim bank
6. Remove gating, use uniform adapter weights

## Metrics

Primary:

- mean total risk score
- mean per-dimension risk

Secondary:

- stance overlap
- refusal rate
- completion length

## Output artifacts

Each run should produce:

- training logs
- saved trainable checkpoints
- evaluation metrics json
- prediction jsonl
- plots

## Result table template

| Model | Method | Mean Risk | Stance | Refusal | Avg Length |
|------|--------|----------:|-------:|--------:|-----------:|
| Qwen3-4B-2507 | Base |  |  |  |  |
| Qwen3-4B-2507 | Risk+SFT |  |  |  |  |
| Qwen3-4B-2507 | Risk+SFT+RL |  |  |  |  |
