#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/home/haowen/Lab/enter/envs/ai/bin/python}"
VLLM_ENV="${VLLM_ENV:-/home/haowen/Lab/enter/envs/vllm}"

if [ ! -x "$VLLM_ENV/bin/python" ]; then
  "$BOOTSTRAP_PYTHON" -m venv "$VLLM_ENV"
fi

VLLM_PYTHON="$VLLM_ENV/bin/python"

"$VLLM_PYTHON" -m pip install -U pip

PIP_ARGS=(--timeout 120 --retries 3 --progress-bar off)

if ! "$VLLM_PYTHON" -m pip install -U vllm openai aiohttp "${PIP_ARGS[@]}" -i https://mirrors.cloud.tencent.com/pypi/simple; then
  echo "[install_vllm_runtime] Tencent mirror failed, retrying Tsinghua mirror."
  if ! "$VLLM_PYTHON" -m pip install -U vllm openai aiohttp "${PIP_ARGS[@]}" -i https://pypi.tuna.tsinghua.edu.cn/simple; then
    echo "[install_vllm_runtime] Tsinghua mirror failed, retrying PyPI."
    "$VLLM_PYTHON" -m pip install -U vllm openai aiohttp "${PIP_ARGS[@]}"
  fi
fi

"$VLLM_PYTHON" - <<'PY'
import vllm
print("vllm", getattr(vllm, "__version__", "unknown"))
PY

echo "VLLM_PYTHON=$VLLM_PYTHON"
