# Llama-3.1-8B Risk-Gated SFT/RL Suite

This is the Llama counterpart of the Qwen risk-head + adapter SFT + RL workflow.
For model-only replication, the algorithmic code path and training data are the
same as Qwen; the defaults only switch the base model, run tag, checkpoint
directory, and formal-evaluation output directory.

## Base Model

```bash
/data2/guohaowen_data/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1
```

## Train Risk Head, SFT Adapters, and RL Adapters

From the repo root on the server:

```bash
cd /home/haowen/Lab/EMBER_SFT_RL_LAB
python scripts/run_llama_server_pipeline.py
```

This wrapper forwards to the existing `scripts/run_server_pipeline.py` with:

```bash
--env-file configs/llama3_1_8b_qwen_protocol_server.env
--skip-prepare-topic-splits
--skip-build-corpus
```

Outputs are written to:

```bash
runs/llama3_1_8b_instruct_qwen_protocol
```

The dataset bucket is `qwen`, not `llama`, and the wrapper preserves the existing
`datasets/qwen` files instead of rebuilding them from legacy source logs. The
only intended experimental difference from Qwen is the base model.

Expected final checkpoints:

```bash
runs/llama3_1_8b_instruct_qwen_protocol/risk_head/best
runs/llama3_1_8b_instruct_qwen_protocol/sft/best
runs/llama3_1_8b_instruct_qwen_protocol/rl/final
```

## Full-Context Five-Way Evaluation

After training finishes:

```bash
cd /home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_llama
python run_fiveway_exact_biasexpert_eval.py
```

Resume is still supported:

```bash
python resume_fiveway_exact_biasexpert_eval.py
```

Default outputs:

```bash
outputs/llama_exact_biasexpert_fiveway_cmv101_200_fullcontext
```

The five-way method logic is unchanged:

- `BASE`
- `EMBER-PROMPT`
- `EMBER-AGENT`
- `SFT`
- `SFT+RL`

The formal evaluator remains `Qwen3-4B-BiasExpert`, matching the Qwen evaluation
setup and preserving cross-method comparability.
