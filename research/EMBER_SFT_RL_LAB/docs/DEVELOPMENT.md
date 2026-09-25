# Development Notes

## Package

Core code lives in:

- [src/risk_gated_alignment](F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/src/risk_gated_alignment)

Main modules:

- `data.py`
- `modeling.py`
- `rewards.py`
- `utils.py`

## Scripts

- `prepare_topic_splits.py`
- `build_training_corpus.py`
- `run_server_pipeline.py`
- `train_risk_head.py`
- `train_sft_adapters.py`
- `train_rl_policy.py`
- `evaluate_policy.py`
- `plot_metrics.py`

## Checkpoint format

Trainable state is saved as:

- `risk_head.pt`
- `adapter_bank.pt`
- `trainable_config.json`

The base model is referenced by path/name and is not duplicated in each checkpoint.

## Extending the method

Low-risk extensions:

1. Move adapters from final hidden states to selected transformer layers.
2. Add optional KL regularization in RL.
3. Replace the simple lexical stance metric with an entailment scorer.
4. Swap the risk-head reward with an external evaluator on larger servers.

## Practical advice

- Keep the first run simple.
- The current default server backbone is `Qwen3-4B-Instruct-2507` from the local snapshot path in `configs/qwen3_4b_local_server.env`.
- Debug with `Qwen2.5-0.5B` when you only want to sanity-check the pipeline.
- Use `Qwen2.5-1.5B` as a lighter comparison baseline if you want a smaller ablation run.
- Do not report test numbers until the entire pipeline is frozen.
