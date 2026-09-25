# EMBER Risk-Gated Alignment Lab

This is the isolated server-ready codebase for the EMBER extension based on:

- a 6-dim bias risk head,
- a dimension-aware adapter bank,
- gated residual fusion,
- supervised debiasing,
- lightweight policy optimization.

It does **not** modify the original [EMBER workspace](F:/lab/LLM_Eval/EMBER).

## Project layout

- [src/risk_gated_alignment](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/src/risk_gated_alignment)
  Core package: data handling, model architecture, reward functions.
- [scripts](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts)
  End-to-end pipeline scripts.
- [configs](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/configs)
  Server env presets.
- [docs](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/docs)
  Solution, experiments, development notes.
- [source_data](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/source_data)
  Copied raw EMBER JSONL sources.
- [generated](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/generated)
  Earlier bootstrap exports kept for reference.

## Main pipeline

1. Topic-level split:
   [prepare_topic_splits.py](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts/prepare_topic_splits.py)
2. Corpus build:
   [build_training_corpus.py](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts/build_training_corpus.py)
3. Risk-head training:
   [train_risk_head.py](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts/train_risk_head.py)
4. Adapter SFT:
   [train_sft_adapters.py](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts/train_sft_adapters.py)
5. RL refinement:
   [train_rl_policy.py](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts/train_rl_policy.py)
6. Evaluation:
   [evaluate_policy.py](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts/evaluate_policy.py)
7. Plotting:
   [plot_metrics.py](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/scripts/plot_metrics.py)

## Quick start

```bash
cd F:/lab/LLM_Eval/EMBER_SFT_RL_LAB
pip install -r requirements-server.txt
python scripts/run_server_pipeline.py --env-file configs/qwen3_4b_local_server.env
```

## Documents

- [SOLUTION.md](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/docs/SOLUTION.md)
- [EXPERIMENTS.md](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/docs/EXPERIMENTS.md)
- [DEVELOPMENT.md](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/docs/DEVELOPMENT.md)
