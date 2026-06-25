# EMBER: Emergent Multi-turn Bias Evaluation and Mitigation

EMBER is a research and engineering framework for evaluating and mitigating
emergent bias in adversarial multi-turn interactions with large language
models. Instead of treating bias evaluation as a static prompt-answer task,
EMBER models a full interaction trajectory: user input, adversarial pressure,
target-model responses, evaluator scores, mitigation actions, and stage-level
rollback.

## Repository Layout

```text
EMBER/
├── src/ember/                    # Clean public reference implementation
│   ├── arena.py                   # Multi-agent adversarial dialogue arena
│   ├── agent.py                   # EMBER-Agent reflection-revision loop
│   ├── harness.py                 # EMBER-Harness stage gates and rollback
│   └── data.py                    # JSONL data helpers
├── examples/                      # Minimal demos without private API keys
├── tests/                         # Lightweight tests
├── code/                          # Legacy thesis experiment scripts
├── Dataset/                       # Sample datasets
├── experiments/ember_harness/     # Processed Harness experiment artifacts
├── docs/zh-CN/                    # Chinese docs
└── docs/en/                       # English docs
```

## Main Components

### EMBER Evaluation Framework

EMBER uses multi-agent adversarial debate to expose bias that may not appear in
single-turn static benchmarks. A target model interacts with one or more
provocateur agents, and the target model's responses are scored by a
BiasExpert-style evaluator across dimensions such as political, gender,
ethnic/cultural, age, religion, and disability bias.

### EMBER-Agent

EMBER-Agent is an inference-time mitigation loop inspired by ReAct. It first
generates a candidate answer, then checks the answer for bias, observes the
feedback, and rewrites the answer until it passes the check or reaches a
maximum refinement budget.

### Risk-Aware Parameter-Efficient Mitigation

The thesis also studies risk-aware mitigation with a risk head, lightweight
adapters, supervised fine-tuning, and reinforcement learning. This line of work
targets the inference-time overhead introduced by repeated self-checking.

### EMBER-Harness

EMBER-Harness places safety checks at semantic stage boundaries rather than
only at the final output or after every model call. A typical assistant
trajectory contains:

```text
input_parse -> memory_read -> retrieval -> planning -> tool_result -> draft_response -> final_response
```

Each stage is snapshotted before it is committed. If the EMBER Gate detects
risk, the runner rolls back to the latest clean snapshot and repairs the local
stage instead of restarting the whole task.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python examples/minimal_agent_demo.py
python examples/minimal_harness_demo.py
pytest
```

The examples use deterministic rule-based evaluators, so they run without
private models or credentials.

## EMBER-Harness Results

`experiments/ember_harness` contains processed CSV files, LaTeX tables, and
figures for a 100-trajectory controlled benchmark comparing three placements:

| Strategy | Check calls | Check tokens | Expected total tokens | Detection rate | Avg. lag |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final-only | 100 | 91,758 | 538,670 | 78.75% | 3.24 |
| Stage-gate | 504 | 391,824 | 690,608 | 97.50% | 0.18 |
| Per-call | 952 | 804,263 | 1,064,828 | 100.00% | 0.04 |

The correct interpretation is not that Stage-gate is always the cheapest
strategy. It is a safety-cost trade-off: it uses roughly half the check tokens
of Per-call while achieving near-Per-call detection and much lower lag than
Final-only.

![Token cost breakdown](experiments/ember_harness/figures/fig_token_cost_breakdown.png)

## Security

Never commit real API keys. Use `.env.example` as a local template. Private
model paths, service credentials, and large raw JSONL dumps should stay outside
the git repository or be released through a dedicated artifact channel.

## Documentation

- [Architecture](docs/en/architecture.md)
- [Experiments](docs/en/experiments.md)
- [Paper summary](docs/en/paper-summary.md)
- [Security and reproducibility](docs/en/security.md)
- [中文说明](README.zh-CN.md)
