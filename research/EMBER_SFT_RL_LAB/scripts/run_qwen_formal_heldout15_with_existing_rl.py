import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_ROOT = SCRIPT_DIR.parent
REQUIRED_RL_FILES = [
    "adapter_bank.pt",
    "risk_head.pt",
    "trainable_config.json",
]
DEFAULT_EXPERIMENT_NAME = "qwen_formal_heldout15_existing_rl_from_old_lab"
DEFAULT_OUTPUT_DIR = "./runs/qwen_formal_heldout15_existing_rl_from_old_lab"
REVISION_SUITE_CANDIDATES = [
    LAB_ROOT / "revision_suite",
    Path("/home/haowen/Lab/EMBER_SFT_RL_LAB/revision_suite"),
    Path("/home/haowen/Lab/LLM_Eval/EMBER_ARR_REVISION_LAB/revision_suite"),
    LAB_ROOT.parent / "LLM_Eval" / "EMBER_ARR_REVISION_LAB" / "revision_suite",
    LAB_ROOT.parent / "EMBER_ARR_REVISION_LAB" / "revision_suite",
]
OLD_RL_CHECKPOINT_CANDIDATES = [
    Path("/home/haowen/Lab/EMBER_SFT_RL_LAB/runs/qwen3_4b_instruct_2507_local/rl/final"),
    LAB_ROOT / "runs" / "qwen3_4b_instruct_2507_local" / "rl" / "final",
]
BLOCKED_FLAGS = {"--detach", "--status", "--stop"}


def path_has_files(path, filenames):
    path = Path(path)
    return path.is_dir() and all((path / name).exists() for name in filenames)


def resolve_revision_suite_root():
    required = [
        Path("scripts") / "run_revision_pipeline.py",
        Path("configs") / "server_qwen_formal_heldout15_comparison.json",
    ]
    checked = []
    for candidate in REVISION_SUITE_CANDIDATES:
        checked.append(str(candidate))
        if candidate.is_dir() and all((candidate / item).exists() for item in required):
            return candidate.resolve()
    raise RuntimeError(
        "Could not locate the revision_suite directory. "
        "Checked these locations:\n"
        + "\n".join(checked)
    )


def resolve_old_rl_checkpoint():
    for candidate in OLD_RL_CHECKPOINT_CANDIDATES:
        if path_has_files(candidate, REQUIRED_RL_FILES):
            return candidate.resolve()
    raise RuntimeError(
        "Could not locate the old RL checkpoint directory.\n"
        "Expected one of these locations to exist and contain adapter_bank.pt / risk_head.pt / trainable_config.json:\n"
        + "\n".join(str(path) for path in OLD_RL_CHECKPOINT_CANDIDATES)
    )


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_runtime_config(base_config_path, rl_checkpoint_dir):
    config = load_json(base_config_path)
    found = False
    for spec in config.get("generation", {}).get("target_models", []):
        if spec.get("name") == "qwen_rl_aligned" or spec.get("variant_label") == "RL-aligned":
            spec["checkpoint_dir"] = str(rl_checkpoint_dir)
            found = True
            break
    if not found:
        raise RuntimeError("Could not find the RL-aligned target entry in the base config.")

    config["experiment_name"] = DEFAULT_EXPERIMENT_NAME
    config["output_dir"] = DEFAULT_OUTPUT_DIR
    config.setdefault("runtime_notes", {})
    config["runtime_notes"]["selected_rl_checkpoint_dir"] = str(rl_checkpoint_dir)
    config["runtime_notes"]["runner"] = str(Path(__file__).resolve())
    return config


def main():
    passthrough = sys.argv[1:]
    blocked = [flag for flag in passthrough if flag in BLOCKED_FLAGS]
    if blocked:
        raise SystemExit(
            "This script is foreground-only and does not support: "
            + ", ".join(sorted(set(blocked)))
        )

    revision_suite_root = resolve_revision_suite_root()
    old_rl_checkpoint = resolve_old_rl_checkpoint()

    base_config_path = revision_suite_root / "configs" / "server_qwen_formal_heldout15_comparison.json"
    runtime_config_path = revision_suite_root / "configs" / "server_qwen_formal_heldout15_existing_rl_from_old_lab.json"
    pipeline_script = revision_suite_root / "scripts" / "run_revision_pipeline.py"

    runtime_config = build_runtime_config(
        base_config_path=base_config_path,
        rl_checkpoint_dir=old_rl_checkpoint,
    )
    dump_json(runtime_config_path, runtime_config)

    command = [
        sys.executable,
        str(pipeline_script),
        "--config",
        str(runtime_config_path),
        "--skip-judge-consistency",
        *passthrough,
    ]

    print("[FORMAL-EXISTING-RL] revision_suite_root=" + str(revision_suite_root))
    print("[FORMAL-EXISTING-RL] rl_checkpoint_dir=" + str(old_rl_checkpoint))
    print("[FORMAL-EXISTING-RL] runtime_config=" + str(runtime_config_path))
    print("[FORMAL-EXISTING-RL] output_dir=" + str(runtime_config["output_dir"]))
    print("[FORMAL-EXISTING-RL] " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
