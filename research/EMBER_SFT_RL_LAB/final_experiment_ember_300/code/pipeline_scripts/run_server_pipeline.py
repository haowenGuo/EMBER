import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = (
    "/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
)
DEFAULT_MODEL_TAG = "qwen3_4b_instruct_2507_local"
DEFAULT_DATASET_BUCKET = "qwen"


def load_env_file(env_file):
    loaded = {}
    path = Path(env_file)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            loaded[key.strip()] = value
    return loaded


def resolve_setting(cli_value, env_values, key, default):
    if cli_value is not None:
        return cli_value
    if key in env_values:
        return env_values[key]
    return os.environ.get(key, default)


def query_gpus():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu,name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "free_mb": int(parts[1]),
                "total_mb": int(parts[2]),
                "utilization": int(parts[3]),
                "name": ",".join(parts[4:]).strip(),
            }
        )
    rows.sort(key=lambda item: (item["free_mb"], -item["utilization"]), reverse=True)
    return rows


def format_gpu_rows(rows):
    return "; ".join(
        f"GPU {row['index']} {row['name']} free={row['free_mb']}MB total={row['total_mb']}MB util={row['utilization']}%"
        for row in rows
    )


def select_gpus(step_name, min_free_gb, max_parallel_gpus, min_parallel_gpus, max_utilization):
    rows = query_gpus()
    min_free_mb = int(min_free_gb * 1024)
    eligible = [row for row in rows if row["free_mb"] >= min_free_mb and row["utilization"] <= max_utilization]
    fallback = [row for row in rows if row["free_mb"] >= min_free_mb]
    if not fallback:
        raise RuntimeError(
            f"No GPU has at least {min_free_gb:.1f} GiB free for {step_name}. "
            f"Current status: {format_gpu_rows(rows)}"
        )
    if not eligible:
        print(
            f"[PIPELINE] warning: no GPU satisfies both free memory >= {min_free_gb:.1f} GiB "
            f"and utilization <= {max_utilization}%. Falling back to memory-only selection."
        )
        eligible = fallback

    selected = eligible[:max_parallel_gpus]
    gpu_ids = [row["index"] for row in selected]
    distributed = len(gpu_ids) >= min_parallel_gpus
    if not distributed:
        gpu_ids = gpu_ids[:1]
    return gpu_ids, distributed, eligible


def run_step(name, command, env=None, gpu_ids=None, distributed=False):
    pretty = shlex.join(command)
    print(f"\n[PIPELINE] {name}")
    if gpu_ids:
        mode = "distributed" if distributed else "single-gpu"
        print(f"[PIPELINE] gpu_mode={mode} visible={gpu_ids}")
    print(f"[PIPELINE] {pretty}")
    subprocess.run(command, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Python entrypoint for the EMBER risk-gated server pipeline.")
    parser.add_argument("--env-file", help="Optional env preset file, e.g. configs/qwen3_4b_local_server.env")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--model-tag")
    parser.add_argument("--dataset-bucket")
    parser.add_argument("--run-dir")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    parser.add_argument("--auto-select-gpus", action="store_true", default=True)
    parser.add_argument("--no-auto-select-gpus", dest="auto_select_gpus", action="store_false")
    parser.add_argument("--max-parallel-gpus", type=int, default=4)
    parser.add_argument("--min-parallel-gpus", type=int, default=2)
    parser.add_argument("--max-gpu-utilization", type=int, default=60)
    parser.add_argument("--risk-min-free-gb", type=float, default=16.0)
    parser.add_argument("--sft-min-free-gb", type=float, default=18.0)
    parser.add_argument("--rl-min-free-gb", type=float, default=20.0)
    parser.add_argument("--eval-min-free-gb", type=float, default=14.0)
    parser.add_argument("--risk-batch-size", type=int, default=2)
    parser.add_argument("--risk-epochs", type=int, default=3)
    parser.add_argument("--sft-batch-size", type=int, default=1)
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--rl-batch-size", type=int, default=1)
    parser.add_argument("--rl-epochs", type=int, default=1)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--use-separate-rl-scorer", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--skip-prepare-topic-splits", action="store_true")
    parser.add_argument("--skip-source-audit", action="store_true")
    parser.add_argument("--skip-build-corpus", action="store_true")
    args = parser.parse_args()

    env_values = load_env_file(args.env_file) if args.env_file else {}

    model_name = resolve_setting(args.model_name_or_path, env_values, "MODEL_NAME", DEFAULT_MODEL_NAME)
    model_tag = resolve_setting(args.model_tag, env_values, "MODEL_TAG", DEFAULT_MODEL_TAG)
    dataset_bucket = resolve_setting(args.dataset_bucket, env_values, "DATASET_BUCKET", DEFAULT_DATASET_BUCKET)
    run_dir = Path(resolve_setting(args.run_dir, env_values, "RUN_DIR", str(ROOT / "runs" / model_tag)))
    run_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    datasets_dir = ROOT / "datasets" / dataset_bucket

    def launch_script(step_name, script_name, script_args, min_free_gb=None):
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        gpu_ids = None
        distributed = False
        if args.auto_select_gpus and min_free_gb is not None:
            gpu_ids, distributed, eligible = select_gpus(
                step_name,
                min_free_gb,
                max_parallel_gpus=args.max_parallel_gpus,
                min_parallel_gpus=args.min_parallel_gpus,
                max_utilization=args.max_gpu_utilization,
            )
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
            print(f"[PIPELINE] eligible GPUs for {step_name}: {format_gpu_rows(eligible)}")
        if distributed:
            command = [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(len(gpu_ids)),
                str(ROOT / "scripts" / script_name),
                *script_args,
            ]
        else:
            command = [python, str(ROOT / "scripts" / script_name), *script_args]
        run_step(step_name, command, env=env, gpu_ids=gpu_ids, distributed=distributed)

    if not args.skip_prepare_topic_splits:
        run_step(
            "prepare_topic_splits",
            [python, str(ROOT / "scripts" / "prepare_topic_splits.py")],
        )

    if not args.skip_source_audit:
        run_step(
            "audit_training_sources",
            [python, str(ROOT / "scripts" / "audit_training_sources.py")],
        )

    if not args.skip_build_corpus:
        run_step(
            "build_training_corpus",
            [python, str(ROOT / "scripts" / "build_training_corpus.py")],
        )

    launch_script(
        "train_risk_head",
        "train_risk_head.py",
        [
            "--model-name-or-path",
            model_name,
            "--train-data",
            str(datasets_dir / "risk_head_train.jsonl"),
            "--dev-data",
            str(datasets_dir / "risk_head_dev.jsonl"),
            "--output-dir",
            str(run_dir / "risk_head"),
            "--batch-size",
            str(args.risk_batch_size),
            "--epochs",
            str(args.risk_epochs),
            "--max-length",
            str(args.max_length),
            "--dtype",
            args.dtype,
        ],
        min_free_gb=args.risk_min_free_gb,
    )

    launch_script(
        "train_sft_adapters",
        "train_sft_adapters.py",
        [
            "--model-name-or-path",
            model_name,
            "--train-data",
            str(datasets_dir / "sft_train.jsonl"),
            "--dev-data",
            str(datasets_dir / "sft_dev.jsonl"),
            "--risk-head-checkpoint",
            str(run_dir / "risk_head" / "best"),
            "--output-dir",
            str(run_dir / "sft"),
            "--batch-size",
            str(args.sft_batch_size),
            "--epochs",
            str(args.sft_epochs),
            "--max-length",
            str(args.max_length),
            "--dtype",
            args.dtype,
        ],
        min_free_gb=args.sft_min_free_gb,
    )

    launch_script(
        "train_rl_policy",
        "train_rl_policy.py",
        [
            "--model-name-or-path",
            model_name,
            "--train-data",
            str(datasets_dir / "rl_train.jsonl"),
            "--sft-checkpoint",
            str(run_dir / "sft" / "best"),
            "--output-dir",
            str(run_dir / "rl"),
            "--batch-size",
            str(args.rl_batch_size),
            "--epochs",
            str(args.rl_epochs),
            "--samples-per-prompt",
            str(args.samples_per_prompt),
            "--max-prompt-length",
            str(args.max_prompt_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dtype",
            args.dtype,
        ] + (["--use-separate-scorer"] if args.use_separate_rl_scorer else []),
        min_free_gb=args.rl_min_free_gb,
    )

    launch_script(
        "evaluate_base",
        "evaluate_policy.py",
        [
            "--model-name-or-path",
            model_name,
            "--eval-data",
            str(datasets_dir / "rl_test.jsonl"),
            "--output-dir",
            str(run_dir / "eval_base"),
            "--max-prompt-length",
            str(args.max_prompt_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dtype",
            args.dtype,
        ],
        min_free_gb=args.eval_min_free_gb,
    )

    launch_script(
        "evaluate_rl",
        "evaluate_policy.py",
        [
            "--model-name-or-path",
            model_name,
            "--checkpoint",
            str(run_dir / "rl" / "final"),
            "--eval-data",
            str(datasets_dir / "rl_test.jsonl"),
            "--output-dir",
            str(run_dir / "eval_rl"),
            "--max-prompt-length",
            str(args.max_prompt_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dtype",
            args.dtype,
        ],
        min_free_gb=args.eval_min_free_gb,
    )

    run_step(
        "plot_metrics",
        [
            python,
            str(ROOT / "scripts" / "plot_metrics.py"),
            "--risk-log",
            str(run_dir / "risk_head" / "risk_head_log.jsonl"),
            "--sft-log",
            str(run_dir / "sft" / "sft_log.jsonl"),
            "--rl-log",
            str(run_dir / "rl" / "rl_log.jsonl"),
            "--eval-metrics",
            str(run_dir / "eval_base" / "metrics.json"),
            str(run_dir / "eval_rl" / "metrics.json"),
            "--eval-labels",
            "base",
            "rl",
            "--output-dir",
            str(run_dir / "plots"),
        ],
    )

    print("\n[PIPELINE] done")
    print(f"[PIPELINE] model: {model_name}")
    print(f"[PIPELINE] outputs: {run_dir}")


if __name__ == "__main__":
    main()
