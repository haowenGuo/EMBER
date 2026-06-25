# EMBER-Harness Experiment Artifacts

This directory contains paper-ready artifacts for the EMBER-Harness stage-gate
experiment.

## Contents

- `data/`: processed CSV summaries.
- `figures/`: PDF and PNG figures generated from the formal run.
- `tables/`: LaTeX table fragments.
- `ember_harness_experiment_chapter.tex`: Chinese thesis section draft.

## Formal Run Summary

The controlled benchmark contains 100 task trajectories and compares three
check placements:

| Strategy | Check calls | Check tokens | Expected total tokens | Detection rate | Avg. lag |
| --- | ---: | ---: | ---: | ---: | ---: |
| final-only | 100 | 91,758 | 538,670 | 78.75% | 3.24 |
| stage-gate | 504 | 391,824 | 690,608 | 97.50% | 0.18 |
| per-call | 952 | 804,263 | 1,064,828 | 100.00% | 0.04 |

Stage-gate should be interpreted as a safety-cost trade-off, not as an
unconditional token minimizer.
