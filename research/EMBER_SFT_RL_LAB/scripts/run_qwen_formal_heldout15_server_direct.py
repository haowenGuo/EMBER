import json
import subprocess
import sys
from pathlib import Path


REVISION_SUITE_ROOT = Path("/home/haowen/Lab/EMBER_SFT_RL_LAB/revision_suite")
OLD_RL_CHECKPOINT_DIR = Path("/home/haowen/Lab/EMBER_SFT_RL_LAB/runs/qwen3_4b_instruct_2507_local/rl/final")
BASE_CONFIG_PATH = REVISION_SUITE_ROOT / "configs" / "server_qwen_formal_heldout15_comparison.json"
RUNTIME_CONFIG_PATH = REVISION_SUITE_ROOT / "configs" / "server_qwen_formal_heldout15_existing_rl_from_old_lab.json"
PIPELINE_SCRIPT = REVISION_SUITE_ROOT / "scripts" / "run_revision_pipeline.py"
OUTPUT_DIR = "./runs/qwen_formal_heldout15_existing_rl_from_old_lab"
EXPERIMENT_NAME = "qwen_formal_heldout15_existing_rl_from_old_lab"
REQUIRED_REVISION_FILES = [
    REVISION_SUITE_ROOT / "scripts" / "run_revision_pipeline.py",
    REVISION_SUITE_ROOT / "configs" / "server_qwen_formal_heldout15_comparison.json",
]
REQUIRED_RL_FILES = [
    OLD_RL_CHECKPOINT_DIR / "adapter_bank.pt",
    OLD_RL_CHECKPOINT_DIR / "risk_head.pt",
    OLD_RL_CHECKPOINT_DIR / "trainable_config.json",
]


def ensure_exists(paths, label):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"{label} missing:\n" + "\n".join(missing))


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_runtime_config():
    config = load_json(BASE_CONFIG_PATH)
    found = False
    for spec in config.get("generation", {}).get("target_models", []):
        if spec.get("name") == "qwen_rl_aligned" or spec.get("variant_label") == "RL-aligned":
            spec["checkpoint_dir"] = str(OLD_RL_CHECKPOINT_DIR)
            found = True
            break
    if not found:
        raise RuntimeError("Could not find RL-aligned target spec in the base config.")

    config["experiment_name"] = EXPERIMENT_NAME
    config["output_dir"] = OUTPUT_DIR
    config.setdefault("runtime_notes", {})
    config["runtime_notes"]["selected_rl_checkpoint_dir"] = str(OLD_RL_CHECKPOINT_DIR)
    config["runtime_notes"]["runner"] = str(Path(__file__).resolve())
    return config


def main():
    ensure_exists(REQUIRED_REVISION_FILES, "revision_suite files")
    ensure_exists(REQUIRED_RL_FILES, "old RL checkpoint files")

    runtime_config = build_runtime_config()
    dump_json(RUNTIME_CONFIG_PATH, runtime_config)

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--config",
        str(RUNTIME_CONFIG_PATH),
        "--skip-judge-consistency",
    ]

    print("[SERVER-DIRECT] revision_suite_root=" + str(REVISION_SUITE_ROOT))
    print("[SERVER-DIRECT] rl_checkpoint_dir=" + str(OLD_RL_CHECKPOINT_DIR))
    print("[SERVER-DIRECT] runtime_config=" + str(RUNTIME_CONFIG_PATH))
    print("[SERVER-DIRECT] output_dir=" + OUTPUT_DIR)
    print("[SERVER-DIRECT] " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
