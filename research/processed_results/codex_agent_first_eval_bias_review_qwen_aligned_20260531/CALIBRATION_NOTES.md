# Qwen-Aligned Codex-Agent Review Notes

## Status

This directory currently contains two different things that must not be confused.

### 1. Teacher-Anchored Calibration Artifacts

Existing score files for `batch_0001-batch_0010` and `batch_0015-batch_0016` were produced as teacher-anchored calibration artifacts.

They use Qwen-BiasExpert outputs as the boundary anchor and therefore are **not** independent Codex-Agent blind evaluations.

They are useful only for:

- checking result directory structure
- checking merge/statistics scripts
- confirming the desired Qwen-BiasExpert level mapping and conservative boundary

They must not be cited as evidence that Codex-Agent independently matched Qwen-BiasExpert.

### 2. Blind Qwen-Aligned Prompt

The file `codex_agent_review_prompt_qwen_aligned_blind.md` is the corrected prompt for future independent Codex-Agent scoring.

Blind scoring must follow this rule:

- Do not read Qwen-BiasExpert `bias_report`, `bias_labels`, or `total_bias_score` before writing Codex-Agent scores.
- Score using only metadata, topic/context, and target response text.
- Lock the Codex score file first.
- Compare with Qwen-BiasExpert only after the score file has been written.

## Recommended Next Step

Create a separate blind-check root, for example:

`codex_agent_first_eval_bias_review_qwen_aligned_blindcheck_20260531`

Then blind-score `batch_0011-batch_0014` and `batch_0015-batch_0016`, followed by a post-hoc comparison report.

