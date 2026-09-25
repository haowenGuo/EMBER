# Clean Subagent Blindcheck Stage Summary

Date: 2026-05-31

Scope:
- Evaluator: clean subagent with `fork_context=false`
- Input: blind batches `batch_0001` through `batch_0005`
- Rows: 60 Qwen FIRST_EVAL rows
- Forbidden teacher fields removed from input: `qwen_bias_report`, `qwen_bias_labels`, `qwen_total_bias_score`
- Score files were locked before Qwen-BiasExpert comparison.

Validation:
- Score files: 5
- Input rows: 60
- Output rows: 60
- Schema errors: 0
- Forbidden field violations in blind input: 0
- Qwen comparison rows: 59 valid rows, 1 Qwen missing score (`qwen|BASE|first_eval_002|r4`)

Overall comparison against Qwen-BiasExpert:
- Codex-Agent mean total bias score: 0.441
- Qwen-BiasExpert mean total bias score: 1.678
- Mean delta, Codex minus Qwen: -1.237
- MAE: 1.407
- Exact total-score match rate: 0.288

Variant comparison:
- BASE: Codex 0.435 vs Qwen 1.522, delta -1.087, exact 0.174
- EMBER-PROMPT: Codex 0.444 vs Qwen 2.333, delta -1.889, exact 0.278
- EMBER-AGENT: Codex 0.444 vs Qwen 1.222, delta -0.778, exact 0.444

Dimension comparison:
- Political is the largest gap: Codex 0.305 vs Qwen 1.153, delta -0.847
- Gender is the second largest gap: Codex 0.000 vs Qwen 0.305, delta -0.305
- Ethnic/cultural is closer on mean: Codex 0.136 vs Qwen 0.186, delta -0.051, but has some high-disagreement rows
- Age, religion, and disability are nearly identical in this small slice.

Interpretation:
- The clean blind evaluator is now too conservative relative to Qwen-BiasExpert.
- The main mismatch is not formatting or leakage. It is rubric interpretation.
- Qwen-BiasExpert appears to score one-sided political or ideological framing as political bias more readily, even when the response does not attack a protected group.
- The current blind prompt tells Codex-Agent to require unfair stereotyping or group-directed framing, so it under-scores political, gender, and some ethnic/cultural cases.

Next calibration change:
- Keep blind-subagent workflow.
- Do not expose Qwen labels to the evaluator.
- Revise the prompt boundary so that one-sided political/ideological/value framing can be `Low` or `Moderate` political bias even without group demeaning.
- Add calibrated examples for gender and ethnic/cultural cases where Qwen-BiasExpert treats asymmetric or essentializing framing as bias.
