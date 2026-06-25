# Paper Summary

The thesis studies emergent bias in adversarial multi-turn interactions with
large language models. Its contributions can be organized into four layers.

## 1. EMBER: Dynamic Evaluation

Static bias benchmarks cannot fully capture risks that emerge through long,
adversarial, or socially pressured interactions. EMBER therefore evaluates
models through multi-agent debate trajectories and tracks how bias changes
across rounds.

## 2. EMBER-Agent: Inference-Time Mitigation

Prompt-based mitigation can fade as the conversation becomes longer and more
adversarial. EMBER-Agent adapts the ReAct idea into a bias mitigation loop:
Think, Reflect, Observe, and Act. It checks and rewrites candidate answers
before final output, but introduces additional token and latency overhead.

## 3. Risk-Aware Parameter-Efficient Mitigation

To reduce inference-time self-checking overhead, the thesis studies a
risk-aware mitigation model with a risk head, lightweight adapters, supervised
fine-tuning, and reinforcement learning. The goal is to shift part of the
mitigation ability into trainable parameters rather than repeated external
self-reflection calls.

## 4. EMBER-Harness: Engineering Control Layer

EMBER-Harness addresses deployment placement. Final-only checking is cheap but
late; per-call checking is precise but expensive. Stage-gate checking places
EMBER-Agent at semantic stage boundaries such as input parsing, memory,
retrieval, planning, tool results, draft generation, and final response.

The mechanism includes:

- stage decomposition
- stage snapshots
- EMBER Gate risk review
- rollback to the latest clean snapshot
- audit logging

## 5. Experimental Takeaway

The experiments show that adversarial multi-turn interaction can expose and
amplify bias, EMBER-Agent can mitigate bias more consistently than initial
prompting, risk-aware parameter-efficient mitigation can reduce runtime
overhead, and EMBER-Harness provides a practical safety-cost trade-off for
agent deployment.

## 6. Repository Mapping

| Thesis concept | Repository location |
| --- | --- |
| EMBER dynamic evaluation | `src/ember/arena.py`, `code/` |
| EMBER-Agent | `src/ember/agent.py` |
| EMBER-Harness | `src/ember/harness.py`, `experiments/ember_harness/` |
| Data processing | `src/ember/data.py`, `code/data_clean.py` |
| Experiment figures | `experiments/ember_harness/figures/` |
