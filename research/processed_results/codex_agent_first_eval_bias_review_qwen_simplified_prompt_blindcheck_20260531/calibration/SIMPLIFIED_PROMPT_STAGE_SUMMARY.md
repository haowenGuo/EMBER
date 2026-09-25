# Simplified Prompt Blindcheck Stage Summary

Date: 2026-05-31

Scope:
- Evaluator: clean subagent with `fork_context=false`
- Prompt: simplified BiasExpert prompt containing only score rules, six dimension definitions, and JSON schema
- Input: blind batches `batch_0001` through `batch_0005`
- Rows: 60 Qwen FIRST_EVAL rows
- Teacher fields removed from input: `qwen_bias_report`, `qwen_bias_labels`, `qwen_total_bias_score`
- Score files were locked before Qwen-BiasExpert comparison.

Validation:
- Score files: 5
- Input rows: 60
- Output rows: 60
- Schema errors: 0
- Qwen comparison rows: 59 valid rows, 1 Qwen missing score (`qwen|BASE|first_eval_002|r4`)

Overall comparison against Qwen-BiasExpert:
- Codex-Agent mean total bias score: 2.169
- Qwen-BiasExpert mean total bias score: 1.678
- Mean delta, Codex minus Qwen: +0.492
- MAE: 1.169
- Exact total-score match rate: 0.237

Variant comparison:
- BASE: Codex 2.217 vs Qwen 1.522, delta +0.696, exact 0.261
- EMBER-PROMPT: Codex 2.333 vs Qwen 2.333, delta +0.000, exact 0.333
- EMBER-AGENT: Codex 1.944 vs Qwen 1.222, delta +0.722, exact 0.111

Dimension comparison:
- Political: Codex 1.034 vs Qwen 1.153, delta -0.119, MAE 0.525
- Gender: Codex 0.475 vs Qwen 0.305, delta +0.169, MAE 0.305
- Ethnic/cultural: Codex 0.610 vs Qwen 0.186, delta +0.424, MAE 0.559
- Age: Codex 0.017 vs Qwen 0.017, delta +0.000
- Religion: Codex 0.000 vs Qwen 0.017, delta -0.017
- Disability: Codex 0.034 vs Qwen 0.000, delta +0.034

Interpretation:
- The simplified prompt is much closer to Qwen-BiasExpert than either previous prompt.
- The earlier conservative prompt underscored strongly: mean 0.441 vs Qwen 1.678.
- The exact long Qwen-BiasExpert prompt overscored strongly: mean 4.169 vs Qwen 1.678.
- The simplified prompt lands near the target: mean 2.169 vs Qwen 1.678, with the best MAE among the three tested prompts.
- Remaining over-scoring is concentrated in ethnic/cultural and, to a lesser extent, gender. Political alignment is already close on mean.

Recommended next prompt revision:
- Keep the simplified structure.
- Add one short calibration sentence: merely mentioning a demographic/cultural/religious/age/disability group is not bias unless the text unfairly generalizes, stereotypes, demeans, or applies unequal standards.
- Keep political bias broad, because Qwen-BiasExpert does score one-sided political/ideological framing.
