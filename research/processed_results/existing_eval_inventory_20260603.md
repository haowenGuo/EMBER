# Existing FIRST_EVAL / SECOND_EVAL Data Inventory

Generated at 2026-06-03 Asia/Shanghai from local results root: `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\results`.

## Artifacts

- Variant-level index: `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\results\existing_eval_variant_index_20260603.csv`
- Run-level summary: `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\results\existing_eval_run_summary_20260603.csv`
- Human-readable inventory: `F:\lab\LLM_Eval\EMBER_SFT_RL_LAB\final_experiment_ember_300\results\existing_eval_inventory_20260603.md`

## High-Level Read

- Qwen SECOND_EVAL Codex/GPT-5.5 review is complete and clean: BASE=1.929; EMBER-PROMPT=1.792; EMBER-AGENT=1.602; SFT=0.936; SFT+RL=0.276; parse failures total=0.
- Qwen SECOND_EVAL BiasExpert baseline is available: BASE=2.204; EMBER-PROMPT=1.957; EMBER-AGENT=1.696; SFT=1.248; SFT+RL=0.685; parse failures total=195.
- Old Llama SECOND_EVAL full run is available but SFT+RL is not improved over SFT there: BASE=1.988; EMBER-PROMPT=1.576; EMBER-AGENT=1.648; SFT=1.638; SFT+RL=1.680; parse failures total=175.
- Llama light SFT1/RL1 full SECOND_EVAL is available as a completed reference: BASE=1.871; EMBER-PROMPT=1.576; EMBER-AGENT=1.621; SFT=2.418; SFT+RL=2.531; parse failures total=81.
- Plateau SFT + short RL smoke is diagnostic only and worse than the light smoke: BASE=2.550; EMBER-PROMPT=1.915; EMBER-AGENT=2.183; SFT=3.220; SFT+RL=2.644; parse failures total=3.
- Strict clean SFT0/short-context smoke must be treated as invalid despite low means: BASE=2.800; EMBER-PROMPT=1.644; EMBER-AGENT=1.983; SFT=0.000; SFT+RL=0.050; parse failures total=72.
- Current Llama score0-only original GPU0 retraining is still an active run and is not included as a completed eval result yet.

## Canonical Completed Results

| category | run | model | status | variant_means | total_parse_failed_rows |
| --- | --- | --- | --- | --- | --- |
| FIRST_EVAL_BIASEXPERT_CANONICAL | vllm_first_eval150_round05_reusemax_20260529_052000 | llama | CANONICAL_COMPLETE | BASE=1.831; EMBER-PROMPT=1.497; EMBER-AGENT=1.495 | 42 |
| FIRST_EVAL_BIASEXPERT_CANONICAL | vllm_first_eval150_round05_reusemax_20260529_052000 | qwen | CANONICAL_COMPLETE | BASE=2.180; EMBER-PROMPT=1.858; EMBER-AGENT=1.639 | 59 |
| FIRST_EVAL_CODEX_GPT55_CANONICAL | codex_gpt55_simplified_v2_first_eval_full_20260531 | llama | CANONICAL_COMPLETE | BASE=2.052; EMBER-PROMPT=2.003; EMBER-AGENT=1.888 | 0 |
| FIRST_EVAL_CODEX_GPT55_CANONICAL | codex_gpt55_simplified_v2_first_eval_full_20260531 | qwen | CANONICAL_COMPLETE | BASE=2.676; EMBER-PROMPT=2.518; EMBER-AGENT=2.202 | 0 |
| SECOND_EVAL_BIASEXPERT_CANONICAL | second_eval150_round05_five_methods_from_first_eval_mixed_20260529_164351 | llama | CANONICAL_COMPLETE | BASE=1.988; EMBER-PROMPT=1.576; EMBER-AGENT=1.648; SFT=1.638; SFT+RL=1.680 | 175 |
| SECOND_EVAL_BIASEXPERT_CANONICAL | second_eval150_round05_five_methods_from_first_eval_mixed_20260529_164351 | qwen | CANONICAL_COMPLETE | BASE=2.204; EMBER-PROMPT=1.957; EMBER-AGENT=1.696; SFT=1.248; SFT+RL=0.685 | 195 |
| SECOND_EVAL_CODEX_GPT55_CANONICAL | codex_gpt55_simplified_v2_second_eval_qwen_full_20260602 | qwen | CANONICAL_COMPLETE | BASE=1.929; EMBER-PROMPT=1.792; EMBER-AGENT=1.602; SFT=0.936; SFT+RL=0.276 | 0 |

## Completed Reference Full Runs

| category | run | model | status | variant_means | total_parse_failed_rows |
| --- | --- | --- | --- | --- | --- |
| PRE_SPLIT_FINAL_EVAL_BIASEXPERT | vllm_final150_round05_20260528_180454 | llama | REFERENCE_ONLY | BASE=1.938; EMBER-PROMPT=1.649; EMBER-AGENT=1.618 | 24 |
| PRE_SPLIT_FINAL_EVAL_BIASEXPERT | vllm_final150_round05_20260528_180454 | qwen | REFERENCE_ONLY | BASE=2.124; EMBER-PROMPT=1.932; EMBER-AGENT=1.777 | 62 |
| SECOND_EVAL_LLAMA_LIGHT_FULL | second_eval150_round05_llama_light_sft1_rl1_full_20260602_055632 | llama | COMPLETE_REFERENCE | BASE=1.871; EMBER-PROMPT=1.576; EMBER-AGENT=1.621; SFT=2.418; SFT+RL=2.531 | 81 |

## Llama / Qwen Smokes And Ablations

| run | model | status | variant_means | total_parse_failed_rows | note |
| --- | --- | --- | --- | --- | --- |
| fair_qwen_llama_sft_rl_eval10_round05_micro24_compare_20260601_121220 | llama | DIAGNOSTIC_ONLY | SFT=1.610; SFT+RL=1.933 | 1 | Micro fair Qwen/Llama SFT/RL comparison; diagnostic only. |
| fair_qwen_llama_sft_rl_eval10_round05_micro24_compare_20260601_121220 | qwen | DIAGNOSTIC_ONLY | SFT=1.750; SFT+RL=1.464 | 4 | Micro fair Qwen/Llama SFT/RL comparison; diagnostic only. |
| light_sft1_rl1_second_eval10_sft_rl_compare_20260601_212032 | llama | DIAGNOSTIC_ONLY | SFT=2.900; SFT+RL=2.383 | 0 | Topics 1-10 smoke for Llama light SFT1/RL1; useful but not full eval. |
| light_sft1_rl1_second_eval10_sft_rl_compare_20260601_212032 | qwen | DIAGNOSTIC_ONLY | SFT=2.692; SFT+RL=3.566 | 15 | Topics 1-10 smoke for Llama light SFT1/RL1; useful but not full eval. |
| llama_clean_sft0_shortctx4096_second_eval10_sft_rl_compare_20260603_074954 | llama | INVALID_DEGENERATE | BASE=2.800; EMBER-PROMPT=1.644; EMBER-AGENT=1.983; SFT=0.000; SFT+RL=0.050 | 72 | Strict clean SFT0/short-context smoke; invalid/degenerate despite low mean because parse failures are massive. |
| llama_conservative_v2_micro24_sft_rl_only_eval3_round05_smoke_20260601_113138 | llama | DIAGNOSTIC_ONLY | SFT=2.056; SFT+RL=1.833 | 0 | Small Llama conservative-v2 smoke; diagnostic only. |
| llama_sftplateau_shortrl_second_eval10_sft_rl_compare_20260603_022824 | llama | DIAGNOSTIC_ONLY | BASE=2.550; EMBER-PROMPT=1.915; EMBER-AGENT=2.183; SFT=3.220; SFT+RL=2.644 | 3 | Topics 1-10 smoke after SFT-plateau + short RL; worse than light smoke, use as diagnostic only. |
| llama_tuned_v1_second_eval3_round05_smoke_20260530_221900 | llama | DIAGNOSTIC_ONLY | BASE=2.056; EMBER-PROMPT=2.278; EMBER-AGENT=1.444; SFT=1.556; SFT+RL=2.944 | 0 |  |
| llama_tuned_v1_second_eval3_round05_smoke_20260530_221900 | qwen | DIAGNOSTIC_ONLY | BASE=1.944; EMBER-PROMPT=1.882; EMBER-AGENT=1.333; SFT=1.944; SFT+RL=2.125 | 3 |  |
| openai_gpt55_first_eval_bias_ps1_smoke_limit1 | llama | DIAGNOSTIC_ONLY | BASE=3.000 | 0 |  |
| openai_gpt55_first_eval_bias_ps1_smoke_limit1 | qwen | DIAGNOSTIC_ONLY | BASE=3.000 | 0 |  |
| openai_gpt55_first_eval_bias_smoke_local4 | qwen | DIAGNOSTIC_ONLY | BASE=3.000 | 0 |  |
| vllm_final150_round05_smoke_vllm_20260528_174428 | llama | DIAGNOSTIC_ONLY | BASE=1.000; EMBER-PROMPT=2.500; EMBER-AGENT=0.000 | 0 | Remote smoke analysis copied back on 2026-06-03; diagnostic only. |
| vllm_final150_round05_smoke_vllm_20260528_174428 | qwen | DIAGNOSTIC_ONLY | BASE=1.000; EMBER-PROMPT=2.000; EMBER-AGENT=1.500 | 0 | Remote smoke analysis copied back on 2026-06-03; diagnostic only. |

## Review Prompt Development / Small Agent Smokes

These are retained for provenance, but should not be used as headline experiment numbers.

| category | run | model | status | variant_means | total_parse_failed_rows |
| --- | --- | --- | --- | --- | --- |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_20260531 | llama | REFERENCE_ONLY | BASE=1.500 | 0 |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_full_20260531 | qwen | REFERENCE_ONLY | BASE=4.076; EMBER-PROMPT=4.470; EMBER-AGENT=3.767 | 0 |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_qwen_aligned_20260531 | qwen | REFERENCE_ONLY | BASE=1.917; EMBER-PROMPT=1.741; EMBER-AGENT=1.095 | 0 |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_qwen_aligned_blindcheck_20260531 | qwen | REFERENCE_ONLY | BASE=0.417; EMBER-PROMPT=0.444; EMBER-AGENT=0.444 | 0 |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_qwen_exact_prompt_blindcheck_20260531 | qwen | REFERENCE_ONLY | BASE=3.875; EMBER-PROMPT=4.611; EMBER-AGENT=4.056 | 0 |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_qwen_simplified_prompt_blindcheck_20260531 | qwen | REFERENCE_ONLY | BASE=2.167; EMBER-PROMPT=2.333; EMBER-AGENT=1.944 | 0 |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_qwen_simplified_prompt_blindcheck_20260531 | qwen | REFERENCE_ONLY | BASE=3.833; EMBER-PROMPT=4.154; EMBER-AGENT=2.795 | 0 |
| REVIEW_PROMPT_DEV_OR_SMOKE | codex_agent_first_eval_bias_review_qwen_simplified_v2_blindcheck_20260531 | qwen | REFERENCE_ONLY | BASE=3.345; EMBER-PROMPT=3.551; EMBER-AGENT=2.513 | 0 |

## Remote Raw Or Partial Results Without Standard Analysis

- `vllm_second_eval150_round05_reusemax_20260529_052000` at `/home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/results/vllm_second_eval150_round05_reusemax_20260529_052000`: Remote raw generated/scored files exist, but no standard analysis directory was present. wc showed qwen/scored_rounds.jsonl=918 lines and llama/scored_rounds.jsonl=918 lines. Treat as raw/provenance, not canonical summary.
- `longsft8_shortrl_second_eval10_sft_rl_compare_20260602_162054` at `/home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/results/longsft8_shortrl_second_eval10_sft_rl_compare_20260602_162054`: Remote partial generation only: qwen/risk_gated_shards/generated_rounds.worker0.jsonl had 112 lines and no standard analysis directory. Treat as incomplete/partial.

## Practical Recommendation

Use only these as current headline tables unless we intentionally rerun analysis:

- FIRST_EVAL BiasExpert: `vllm_first_eval150_round05_reusemax_20260529_052000`
- SECOND_EVAL BiasExpert: `second_eval150_round05_five_methods_from_first_eval_mixed_20260529_164351`
- FIRST_EVAL Codex/GPT-5.5 blind review: `codex_gpt55_simplified_v2_first_eval_full_20260531`
- SECOND_EVAL Codex/GPT-5.5 blind review: `codex_gpt55_simplified_v2_second_eval_qwen_full_20260602` (Qwen only)
- Llama troubleshooting references: `second_eval150_round05_llama_light_sft1_rl1_full_20260602_055632`, `light_sft1_rl1_second_eval10_sft_rl_compare_20260601_212032`, `llama_sftplateau_shortrl_second_eval10_sft_rl_compare_20260603_022824`; exclude `llama_clean_sft0_shortctx4096_second_eval10_sft_rl_compare_20260603_074954` from conclusions because it is degenerate.

## Notes On The Active Score0-Only Run

The ongoing training bucket `final_first_eval_mixed_sft_score0_only` keeps the original corpus construction/truncation and changes only `sft_train`/`sft_dev` to `total_score <= 0` rows: train 1098/2132 kept, dev 278/535 kept. It has no completed eval summary yet.
