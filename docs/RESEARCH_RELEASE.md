# EMBER research release

This release expands the small runnable reference package with the local training
and evaluation implementation, benchmark construction, experiment protocols and
historical raw data. It does not replace the formal experiments with demo output.

## Find the implementation

| Material | Location |
| --- | --- |
| Offline reference CLI and stage-gate demonstrations | [`src/ember`](../src/ember) |
| Risk head, adapter bank, SFT and policy-optimization code | [`research/EMBER_SFT_RL_LAB`](../research/EMBER_SFT_RL_LAB) |
| Formal 300-topic experiment code and scripts | [`final_experiment_ember_300`](../research/EMBER_SFT_RL_LAB/final_experiment_ember_300) |
| Dataset construction, topic splits and evaluator prompts | [`benchmark`](../benchmark) |
| Processed per-run CSVs and analysis notes | [`research/processed_results`](../research/processed_results) |
| Existing thesis gallery and source summary tables | [`experiments`](../experiments) |
| Earlier experiment scripts, retained for audit | [`research/legacy_ember`](../research/legacy_ember) |
| AILIS stage-gate integration snapshot | [`integrations/ailis`](../integrations/ailis) |

## Raw data downloads

Download the public assets from
[research-2026-09-25](https://github.com/haowenGuo/EMBER/releases/tag/research-2026-09-25).
The release publishes four independently usable ZIP archives:

- `ember-formal-trajectories.zip`: generated/scored trajectories, review batches,
  run metadata and statistics from the formal experiment results directory.
- `ember-training-data.zip`: training/evaluation corpus exports, generated training
  examples and input source-data snapshots.
- `ember-legacy-data.zip`: earlier EMBER inputs, outputs and statistical data.
- `ember-analysis-data.zip`: chapter-level analysis and reproducibility data bundle.

Some source files occur in multiple historical locations. They are retained with
their original relative locations so that audits can distinguish snapshots. Do not
concatenate these archives as if all rows were independent observations.

Use `SHA256SUMS.txt` to verify the archive files. The machine-readable
[`release-manifest.json`](release-manifest.json) records every member's hash and
location. Repository-file hashes refer to Git blob bytes after line-ending
normalization; archive-member hashes refer to the exact uncompressed bytes.
`scripts/verify_release.py` verifies downloaded files against the checksum
list without extracting them. For safe extraction, use a ZIP tool that rejects
absolute paths and parent-directory traversal.

## Reproduction boundaries

The offline package can be tested without API calls:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests
ember-harness-run --strategy stage_gate --json
```

The training stack has separate GPU dependencies:

```bash
python -m pip install -e research/EMBER_SFT_RL_LAB
python research/EMBER_SFT_RL_LAB/scripts/prepare_topic_splits.py --help
python research/EMBER_SFT_RL_LAB/scripts/train_risk_head.py --help
```

Read the lab's protocol and script arguments before scheduling a run. Historical
launchers preserve their experiment settings and may contain author-machine model
or output paths. Replace these paths with your own storage locations; copy an
`.env.example` to a private `.env` and supply credentials through environment
variables. The public snapshot is not a promise that historical shell launchers
run unchanged on every host. No expensive GPU experiment was rerun for this release.

For benchmark statistics, start with `benchmark/data/final_ember_topics_300.jsonl`
and `benchmark/data/eval_splits/`. Separate held-out evaluation from training; do
not treat the historical pilot directories as held-out final results.

## Public access and licenses

The GitHub repository and release assets are public. Self-owned code uses MIT.
Data provenance and the limits of that grant are documented in
[`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md). Source code is downloadable without
using a private service; model access and upstream dataset terms remain separate.
