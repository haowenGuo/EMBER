import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPOS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-27B-FP8",
]
CACHE_DIR = "/data2/guohaowen_data/huggingface_cache/hub"


def main():
    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    print(f"[DOWNLOAD] cache_dir={CACHE_DIR}", flush=True)
    print(f"[DOWNLOAD] HF_ENDPOINT={endpoint}", flush=True)
    for repo_id in REPOS:
        info = api.model_info(repo_id)
        print(
            f"[DOWNLOAD] repo={repo_id} sha={info.sha} lastModified={info.lastModified}",
            flush=True,
        )
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=CACHE_DIR,
            revision="main",
            max_workers=8,
        )
        files = list(Path(path).iterdir())
        print(f"[DOWNLOAD] SNAPSHOT_PATH={path}", flush=True)
        print(f"[DOWNLOAD] file_count={len(files)}", flush=True)


if __name__ == "__main__":
    main()
