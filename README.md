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

- `src/ember`: clean public reference implementation.
- `examples`: minimal demos that run without private API keys.
- `code`: legacy thesis experiment scripts kept for traceability.
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
python examples/minimal_agent_demo.py
python examples/minimal_harness_demo.py
pytest
```

The demos use deterministic rule-based evaluators so the repository can be
tested without private models or API credentials.

## Safety Note

Do not commit real API keys. Use `.env.example` as a template and keep secrets
only in local environment variables. Large raw JSONL experiment dumps should be
released through GitHub Releases, Hugging Face Datasets, or another artifact
store rather than committed directly to the repository.
