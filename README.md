# EMBER

**Emergent Multi-turn Bias Evaluation and Mitigation in Adversarial Dialogues**

EMBER is a research and engineering framework for evaluating and mitigating
emergent bias in multi-turn adversarial interactions with large language
models. It combines multi-agent debate simulation, BiasExpert-style scoring,
EMBER-Agent reflection-based mitigation, and EMBER-Harness stage-gated
deployment.

## Language / 语言

- [中文说明](README.zh-CN.md)
- [English README](README.en.md)

## What Is In This Repository?

- `src/ember`: the main runnable implementation.
- `src/ember/providers/`: rule-based, OpenAI-compatible, and local Transformers providers.
- `src/ember/runner.py`: stage-plan runner and built-in benchmark loop.
- `examples`: minimal demos that run without private API keys.
- `code`: legacy thesis experiment scripts kept for traceability, not the main entry point.
- `Dataset`: sample datasets already present in the original project.
- `experiments/ember_harness`: paper-ready EMBER-Harness tables, figures, and processed CSV results.
- `docs`: architecture, experiment, and safety notes in Chinese and English.

## Core Ideas

1. **EMBER framework** evaluates bias dynamically instead of relying only on
   static prompt-answer benchmarks.
2. **EMBER-Agent** adds a reflection-revision loop that checks and rewrites
   model outputs during multi-turn interactions.
3. **Risk-aware mitigation** studies parameter-efficient ways to reduce
   inference-time self-check overhead.
4. **EMBER-Harness** moves EMBER-Agent from final-output filtering to semantic
   stage gates with snapshots and local rollback.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
ember-agent-demo --json
ember-harness-run --strategy stage_gate --json
ember-benchmark --output outputs/ember_benchmark.csv
pytest
```

The demos use deterministic rule-based evaluators so the repository can be
tested without private models or API credentials.

## Provider Modes

- `--provider rule`: offline deterministic demo provider.
- `--provider openai`: OpenAI-compatible chat completions provider; configure
  `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.
- `--provider transformers`: local Hugging Face Transformers provider; pass
  `--model` or set `TRANSFORMERS_MODEL`.

## Runnable Closed Loop

`ember-harness-run` executes the same stage plan with three check placements:

- `final_only`: check and repair only after the full trajectory is produced.
- `per_call`: check every modeled LLM call and re-check repaired artifacts.
- `stage_gate`: save each stage as a pending snapshot, run the gate, commit clean
  snapshots, and roll back to the latest committed snapshot when a stage fails.

When `--state-dir` is set, stage-gate runs persist one JSON snapshot per check
and an `audit.jsonl` decision log. `ember-benchmark` runs the built-in
controlled scenarios and exports expected token costs for the three strategies.

## Safety Note

Do not commit real API keys. Use `.env.example` as a template and keep secrets
only in local environment variables. Large raw JSONL experiment dumps should be
released through GitHub Releases, Hugging Face Datasets, or another artifact
store rather than committed directly to the repository.
