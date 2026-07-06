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
│   ├── providers/                 # Rule, OpenAI-compatible, and local providers
│   ├── runner.py                  # Stage-plan runner and benchmark loop
│   ├── cli.py                     # Command-line entry points
│   └── data.py                    # JSONL data helpers
├── examples/                      # Minimal demos without private API keys
├── tests/                         # Lightweight tests
├── code/                          # Legacy thesis experiment scripts
├── Dataset/                       # Sample datasets
├── experiments/thesis_results/     # Thesis result figures and summary CSVs
├── experiments/ember_harness/     # Processed Harness experiment artifacts
├── docs/zh-CN/                    # Chinese docs
└── docs/en/                       # English docs
```

## Current Public Release Status

This repository is a cleaned public reference release. The actively maintained
runtime is `src/ember`; scripts under `code` are retained for thesis
traceability and are not the recommended entry point. The test suite covers
the rule provider, the EMBER-Agent check-rewrite loop, EMBER-Harness snapshots
and audit logs, the built-in benchmark, and CLI smoke paths.

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

## Thesis Experiment Results

The main thesis results are collected in
[`experiments/thesis_results`](experiments/thesis_results). This directory
contains paper-facing PNG figures and summary CSV tables for the full
evaluation chain, not only the EMBER-Harness prototype.

| Experiment block | Main artifacts | Summary |
| --- | --- | --- |
| Dynamic bias evolution | `fig_01_dynamic_bias_round_lines.png` | Three tested open-source models show different risk trajectories under dual-agent and multi-agent debate. |
| Bias dimension contribution | `fig_02_bias_dimension_contribution.png` | Political and ethnic/cultural dimensions contribute most under the current topic distribution. |
| Attack strength | `fig_03_attack_strength_lines.png` | Strong adversarial prompts produce clearer late-round risk growth than neutral or mild prompts. |
| EMBER-Agent mitigation | `fig_05` to `fig_07` | EMBER-Agent lowers average composite bias in all six model-scenario combinations. |
| Risk-aware PEFT mitigation | `fig_08` to `fig_09` | SFT and SFT+RL give the largest evaluator-score reductions in the training-time mitigation study. |
| Evaluator-human validation | `fig_10_evaluator_human_alignment.png` | BiasExpert, GPT-5.5, and human samples show broadly similar aggregate tendencies, while disagreement remains visible. |
| EMBER-Harness stage gates | `fig_11_harness_safety_cost_tradeoff.png` | Stage-gate checking improves detection rate and detection lag with lower check cost than per-call checking. |

![Dynamic bias evolution](experiments/thesis_results/figures/fig_01_dynamic_bias_round_lines.png)

![Qwen3-4B mitigation curves](experiments/thesis_results/figures/fig_05_qwen_strategy_lines.png)

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
ember-agent-demo --json
ember-harness-run --strategy stage_gate --json
ember-benchmark --output outputs/ember_benchmark.csv
python -m pytest -s
```

The examples use deterministic rule-based evaluators, so they run without
private models or credentials.

## CLI Commands

| Command | Purpose |
| --- | --- |
| `ember-agent-demo` | Run the EMBER-Agent self-check and rewrite loop |
| `ember-harness-run` | Run a stage plan with final-only, per-call, or stage-gate checking |
| `ember-benchmark` | Run the built-in benchmark and export CSV |

## Provider Modes

| Provider | Description |
| --- | --- |
| `rule` | Offline deterministic provider; works out of the box |
| `openai` | OpenAI-compatible Chat Completions provider configured by `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` |
| `transformers` | Local Hugging Face Transformers provider configured by `--model` or `TRANSFORMERS_MODEL` |

## Runnable Closed Loop

`ember-harness-run` executes the same stage plan with three check placements:

| Strategy | Meaning |
| --- | --- |
| `final_only` | Check only after the full trajectory, then rewrite all and re-check if risk is found |
| `per_call` | Check each modeled LLM call and re-check repaired stage artifacts |
| `stage_gate` | Save a pending snapshot at each semantic stage, commit clean snapshots, and roll back to the latest committed snapshot when a gate fails |

With `--state-dir`, stage-gate runs persist one snapshot JSON file per check and
an `audit.jsonl` decision log. `ember-benchmark` exports expected token costs
for the built-in controlled scenarios.

The built-in Harness scenario option `--risk-stage` accepts any semantic stage
in the default plan, or `none` for a clean trajectory with no injected risk.

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
