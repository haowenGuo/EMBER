import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / "configs" / "llama3_1_8b_qwen_protocol_server.env"
PIPELINE_SCRIPT = ROOT / "scripts" / "run_server_pipeline.py"


def has_env_file_arg(argv):
    return "--env-file" in argv or any(arg.startswith("--env-file=") for arg in argv)


if __name__ == "__main__":
    if not has_env_file_arg(sys.argv[1:]):
        sys.argv[1:1] = ["--env-file", str(DEFAULT_ENV_FILE)]
    for flag in ["--skip-prepare-topic-splits", "--skip-build-corpus"]:
        if flag not in sys.argv[1:]:
            sys.argv.append(flag)
    runpy.run_path(str(PIPELINE_SCRIPT), run_name="__main__")
