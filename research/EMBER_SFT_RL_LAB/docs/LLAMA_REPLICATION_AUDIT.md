# Llama replication data audit

This note records a data-level issue found while porting the Qwen risk-gated
training/evaluation workflow to Llama.

## Finding

The current `source_data` directory does not contain matched Qwen/Llama source
files under the same EMBER protocol.

Current parsed distribution:

| Family | Source file | Parsed records | Mean bias score | Zero score | High >= 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen | `bias_qwen_attack_emberagent.jsonl` | 1154 | 1.671 | 34.0% | 342 |
| llama | `bias_llama_emberagent.jsonl` | 810 | 0.922 | 49.4% | 82 |

This means the Llama training corpus is much lower-risk and smaller before any
training starts. Results from this corpus should not be interpreted as a clean
model-only replication of the Qwen experiment.

## Root causes

1. The Qwen source file is an attack + EMBER-Agent style file, while the Llama
   source file is a regular EMBER-Agent style file.
2. There is no clean `bias_llama_attack_emberagent.jsonl` counterpart in the
   original `EMBER` directory.
3. Many legacy EMBER JSONL rows have an empty `target_response`. The data parser
   previously inferred the target sender from `meta.model` before using the
   transcript-final response, which could silently misassign records when
   `meta.model` was inconsistent with the last speaker.

## Code changes

- `src/risk_gated_alignment/data.py` now infers the target sender using the
  effective target response after falling back to the transcript-final message.
- `scripts/export_alignment_data.py` uses the same corrected sender inference.
- `scripts/audit_training_sources.py` was added to print per-file and
  per-family source distributions before training.
- `scripts/build_training_corpus.py` now supports `--include-files` and
  `--exclude-files` so future runs can explicitly pin source files.
- Qwen and Llama training pipelines now run `audit_training_sources.py` before
  corpus construction unless `--skip-source-audit` is passed.

## Recommended rule

Do not use Llama SFT/RL results as formal cross-model evidence until the Llama
training corpus is rebuilt from source files that match the Qwen protocol.

For a true model-only replication, either:

1. generate a Llama `attack_emberagent` source file under the same protocol as
   `bias_qwen_attack_emberagent.jsonl`; or
2. rebuild both Qwen and Llama corpora from a lower-risk matched pair, such as
   regular EMBER-Agent files, and clearly label the experiment as a different
   data protocol.

Run the audit with:

```bash
python scripts/audit_training_sources.py
```

Build a pinned corpus with:

```bash
python scripts/build_training_corpus.py \
  --include-files bias_qwen_attack_emberagent.jsonl,bias_llama_attack_emberagent.jsonl
```

The second command requires the missing Llama counterpart to exist.
