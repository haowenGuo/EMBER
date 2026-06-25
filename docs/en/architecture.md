# EMBER Architecture

## 1. Dynamic Evaluation

EMBER evaluates bias as a trajectory-level behavior. A target model is placed in
an adversarial multi-agent dialogue arena, challenged by one or more
provocateur agents, and scored across selected bias dimensions over multiple
rounds.

## 2. EMBER-Agent

EMBER-Agent is an inference-time mitigation loop:

```text
Think -> Reflect -> Observe -> Act
```

It generates a candidate answer, reflects on potential bias, observes whether
the answer passes the safety check, and either returns the answer or rewrites
it from feedback.

## 3. EMBER-Harness

EMBER-Harness is a deployment control layer for complex agent systems. Instead
of checking only the final answer or checking every model call, it checks
semantic stage boundaries:

```text
input_parse
memory_read
retrieval
planning
tool_result
draft_response
final_response
```

Each stage is snapshotted before being committed. Failed stages trigger local
rollback to the latest committed snapshot.

## 4. Why Stage Gates?

| Strategy | Strength | Weakness |
| --- | --- | --- |
| Final-only | Minimal checks, easy integration | Late detection, high rework, weak source localization |
| Per-call | Fine-grained detection | Highest token and latency overhead |
| Stage-gate | Balances detection and cost, supports local rollback | Requires explicit stage design and state management |
