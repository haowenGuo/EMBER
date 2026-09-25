# EMBER Benchmark Package (20260703)

This package organizes the final EMBER thesis data into a lightweight reusable benchmark structure.

## Contents

- `data/`: 300 topic benchmark, source metadata, and FIRST/SECOND evaluation splits.
- `protocol/`: dataset construction, method definitions, attacker/evaluator protocol, and vLLM inference notes.
- `prompts/`: EMBER-PROMPT policy, EMBER-Agent rewrite constraints, and the simplified BiasExpert/GPT-5.5 review prompt.
- `results/`: dataset statistics, method ablation summaries, cross-model/cross-scenario stability, human-alignment summary, and sample-level paired statistics.

## Benchmark Scope

- Topic count: 300.
- Sources: 200 Change My View topics and 100 domestic-context topics.
- Bias dimensions: political, gender, ethnic/cultural, age, religion, disability.
- Evaluation splits: 150 FIRST-EVAL topics and 150 SECOND-EVAL held-out topics.
- Debate rounds: 0-5.

Raw generated/scored JSONL files are intentionally not duplicated here because they are hundreds of MB. The original local paths are preserved in the protocol manifests under `chapter5_data_for_gpt_pro_20260622`.
