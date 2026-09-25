import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPT = ROOT / "scripts" / "run_exact_ember_biasexpert_fourway.py"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "qwen_exact_ember_biasexpert_fourway"


def main():
    extra_args = sys.argv[1:]
    command = [sys.executable, str(TARGET_SCRIPT)]
    if "--output-dir" not in extra_args:
        command.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])
    command.extend(extra_args)
    print(f"[EXACT-EMBER-RESUME] output_dir={DEFAULT_OUTPUT_DIR}")
    print("[EXACT-EMBER-RESUME] " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
