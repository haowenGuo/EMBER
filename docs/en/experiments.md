# Experiments

The thesis experiments cover five directions:

1. Dynamic bias discovery with the EMBER framework.
2. The effect of adversarial pressure.
3. EMBER-Agent versus prompt-based mitigation.
4. Risk-aware parameter-efficient mitigation.
5. EMBER-Harness stage-gate evaluation.

Legacy scripts live under `code/`; the cleaned public implementation lives
under `src/ember`.

## Thesis Result Overview

`experiments/thesis_results` collects paper-facing figures and summary CSVs
for the main thesis experiments:

| Block | Files | Description |
| --- | --- | --- |
| Dynamic bias evolution | `figures/fig_01_dynamic_bias_round_lines.png` | Round-level trends for Qwen3-4B, Llama-3.1-8B, and Mistral-7B under dual-agent and multi-agent debate. |
| Bias dimension contribution | `figures/fig_02_bias_dimension_contribution.png` | Contribution of political, ethnic/cultural, religion, gender, disability, and age dimensions. |
| Attack strength | `figures/fig_03_attack_strength_lines.png` | Neutral, mild, and adversarial prompt strength comparison. |
| EMBER-Agent mitigation | `figures/fig_05_qwen_strategy_lines.png` to `figures/fig_07_mistral_strategy_lines.png` | Baseline, prompt mitigation, and EMBER-Agent composite score comparison. |
| Risk-aware PEFT mitigation | `figures/fig_08_risk_peft_overall_table.png`, `figures/fig_09_risk_peft_round_lines.png` | Training-time mitigation results for EMBER-Prompt, EMBER-Agent, SFT, and SFT+RL. |
| Evaluator-human validation | `figures/fig_10_evaluator_human_alignment.png` | Aggregate consistency check among BiasExpert, GPT-5.5, and human samples. |
| EMBER-Harness | `figures/fig_11_harness_safety_cost_tradeoff.png` | Safety-cost trade-off among final-only, per-call, and stage-gate checking. |

This directory stores summary artifacts only; large raw generation JSONL files
are kept out of the public repository.

## EMBER-Harness Controlled Benchmark

`experiments/ember_harness` contains processed paper-ready artifacts:

- `data/overall-results.csv`
- `data/by-risk-stage-results.csv`
- `data/sensitivity-results.csv`
- `figures/*.png`
- `figures/*.pdf`
- `tables/*.tex`

The benchmark compares:

- final-only checking
- per-call checking
- stage-gate checking

The 100 cases are controlled task trajectories rather than simple prompt-answer
pairs. Each trajectory records task type, stage sequence, injected risk stage,
risk level, risk dimensions, and token profile. This design evaluates stage
gating and local rollback rather than natural user-log frequency.
