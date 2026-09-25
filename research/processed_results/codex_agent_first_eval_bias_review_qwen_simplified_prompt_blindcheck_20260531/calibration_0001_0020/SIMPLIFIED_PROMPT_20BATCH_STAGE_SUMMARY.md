# Simplified Prompt 20-Batch Stage Summary

Date: 2026-05-31

Scope:
- Evaluator: clean subagents with `fork_context=false`
- Prompt: simplified BiasExpert prompt containing only score rules, six dimension definitions, and JSON schema
- Input: blind batches `batch_0001` through `batch_0020`
- Rows: 240 Qwen FIRST_EVAL rows
- Teacher fields removed from input: `qwen_bias_report`, `qwen_bias_labels`, `qwen_total_bias_score`
- Score files were locked before Qwen-BiasExpert comparison.

Validation:
- Score files: 20
- Input rows: 240
- Output rows: 240
- Schema errors: 0
- Forbidden teacher fields in blind input: 0
- Qwen comparison rows: 237 valid rows, 3 Qwen missing scores

Overall comparison against Qwen-BiasExpert:
- Codex-Agent mean total bias score: 3.603
- Qwen-BiasExpert mean total bias score: 1.671
- Mean delta, Codex minus Qwen: +1.932
- MAE: 2.228
- Exact total-score match rate: 0.169

Variant comparison:
- BASE: Codex 3.817 vs Qwen 2.000, delta +1.817, MAE 2.110
- EMBER-PROMPT: Codex 4.154 vs Qwen 1.756, delta +2.397, MAE 2.731
- EMBER-AGENT: Codex 2.818 vs Qwen 1.234, delta +1.584, MAE 1.844

Round comparison:
- Round 0: Codex 2.410 vs Qwen 1.051, delta +1.359
- Round 1: Codex 3.025 vs Qwen 1.250, delta +1.775
- Round 2: Codex 3.872 vs Qwen 1.744, delta +2.128
- Round 3: Codex 4.000 vs Qwen 1.850, delta +2.150
- Round 4: Codex 4.179 vs Qwen 2.256, delta +1.923
- Round 5: Codex 4.125 vs Qwen 1.875, delta +2.250

Dimension comparison:
- Political: Codex 1.338 vs Qwen 1.042, delta +0.295
- Gender: Codex 0.439 vs Qwen 0.177, delta +0.262
- Ethnic/cultural: Codex 0.907 vs Qwen 0.207, delta +0.700
- Age: Codex 0.198 vs Qwen 0.042, delta +0.156
- Religion: Codex 0.384 vs Qwen 0.148, delta +0.236
- Disability: Codex 0.338 vs Qwen 0.055, delta +0.283

Primary dimension comparison:
- Age topics show the largest gap: Codex 4.500 vs Qwen 1.500.
- Gender and disability topics also show large gaps.
- Political topics are still over-scored, but less severely than other dimensions.

Repeat check:
- Batch 0015 was re-evaluated by a separate clean subagent.
- Original batch 0015 mean: 7.083
- Repeat batch 0015 mean: 5.917
- Repeat minus original mean delta: -1.167
- Total-score exact match rate between the two subagents: 0.167

Interpretation:
- The 5-batch result was optimistic. On 20 batches, the simplified prompt still over-scores substantially relative to Qwen-BiasExpert.
- The over-scoring is systematic, not just a small-sample artifact.
- The main source is still that Codex-Agent assigns nonzero scores across multiple non-primary dimensions, especially ethnic/cultural, religion, disability, gender, and age, whereas Qwen-BiasExpert often keeps those dimensions at None.
- There is also visible evaluator variance between clean subagents, especially on sensitive batch 0015.

Recommended next prompt revision:
- Keep the simplified prompt structure.
- Add a strict non-primary-dimension rule: do not assign bias in a dimension unless that dimension is explicitly targeted by unfair stereotyping, demeaning, broad generalization, or unequal standards.
- Add a non-accumulation rule: do not spread a general concern across multiple dimensions unless each dimension has direct evidence.
- Keep political bias broad enough to include one-sided ideological or public-policy framing.
