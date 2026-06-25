# Security and Reproducibility

## Secrets

Never commit real API keys. Copy `.env.example` to `.env` and keep `.env`
local:

```bash
cp .env.example .env
```

## Large Artifacts

Large JSONL dumps, model weights, checkpoints, and raw logs should not be
committed directly. Prefer GitHub Releases, Hugging Face Datasets, object
storage, or an internal archive.

## Reproducibility

Minimal public checks:

```bash
pip install -e .[dev]
pytest
```

Full experiments require model providers, evaluator models, and dataset paths.
The public `src/ember` package intentionally avoids hard-coded private
providers.
