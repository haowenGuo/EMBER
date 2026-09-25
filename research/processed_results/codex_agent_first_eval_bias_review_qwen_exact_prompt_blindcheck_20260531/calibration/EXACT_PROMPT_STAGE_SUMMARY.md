# Exact Qwen-BiasExpert Prompt Blindcheck Stage Summary

Date: 2026-05-31

Scope:
- Evaluator: clean subagent with `fork_context=false`
- Prompt: exact `BIAS_EXPERT_PROMPT` extracted from `formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py`
- Input: blind batches `batch_0001` through `batch_0005`
- Rows: 60 Qwen FIRST_EVAL rows
- Teacher fields removed from input: `qwen_bias_report`, `qwen_bias_labels`, `qwen_total_bias_score`
- Score files were locked before Qwen-BiasExpert comparison.

Validation:
- Score files: 5
- Input rows: 60
- Output rows: 60
- Schema errors: 0
- Forbidden field violations in blind input: 0
- Qwen comparison rows: 59 valid rows, 1 Qwen missing score (`qwen|BASE|first_eval_002|r4`)

Overall comparison against Qwen-BiasExpert:
- Codex-Agent mean total bias score: 4.169
- Qwen-BiasExpert mean total bias score: 1.678
- Mean delta, Codex minus Qwen: +2.492
- MAE: 2.593
- Exact total-score match rate: 0.136

Variant comparison:
- BASE: Codex 3.913 vs Qwen 1.522, delta +2.391, exact 0.174
- EMBER-PROMPT: Codex 4.611 vs Qwen 2.333, delta +2.278, exact 0.111
- EMBER-AGENT: Codex 4.056 vs Qwen 1.222, delta +2.833, exact 0.111

Dimension comparison:
- Political: Codex 1.729 vs Qwen 1.153, delta +0.576
- Gender: Codex 0.881 vs Qwen 0.305, delta +0.576
- Ethnic/cultural: Codex 1.068 vs Qwen 0.186, delta +0.881
- Age: Codex 0.305 vs Qwen 0.017, delta +0.288
- Religion: Codex 0.119 vs Qwen 0.017, delta +0.102
- Disability: Codex 0.068 vs Qwen 0.000, delta +0.068

Interpretation:
- Directly copying the original Qwen-BiasExpert prompt into a clean Codex subagent does not reproduce Qwen-BiasExpert behavior.
- It overshoots sharply, especially on gender and ethnic/cultural dimensions.
- The likely reason is that Codex follows the prompt more literally and flags one-sidedness or dimension references more aggressively, whereas Qwen-BiasExpert has an implicit conservative decision boundary despite the broad prompt text.
- The earlier conservative aligned prompt underscored; the exact prompt overscores. A calibrated prompt should sit between them, preserving Qwen's broad political-bias treatment but adding explicit negative examples to avoid over-scoring every demographic or cultural mention.
