# Experiments

The thesis experiments cover five directions:

1. Dynamic bias discovery with the EMBER framework.
2. The effect of adversarial pressure.
3. EMBER-Agent versus prompt-based mitigation.
4. Risk-aware parameter-efficient mitigation.
5. EMBER-Harness stage-gate evaluation.

Legacy scripts live under `code/`; the cleaned public implementation lives
under `src/ember`.

## EMBER-Harness Controlled Benchmark

`experiments/ember_harness` contains processed paper-ready artifacts:

- `data/overall-results.csv`
- `data/by-risk-stage-results.csv`
- `data/sensitivity-results.csv`
- `figures/*.png`
- `figures/*.pdf`
- `tables/*.tex`

The benchmark compares:

- final-only checking
- per-call checking
- stage-gate checking

The 100 cases are controlled task trajectories rather than simple prompt-answer
pairs. Each trajectory records task type, stage sequence, injected risk stage,
risk level, risk dimensions, and token profile. This design evaluates stage
gating and local rollback rather than natural user-log frequency.
