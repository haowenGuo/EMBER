from pathlib import Path

from modelscope import snapshot_download


MODELS = [
    ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
    ("Qwen/Qwen3.6-27B-FP8", "Qwen3.6-27B-FP8"),
]
BASE_DIR = Path("/data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen")


def main():
    print(f"[MODELSCOPE] base_dir={BASE_DIR}", flush=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, name in MODELS:
        local_dir = BASE_DIR / name
        print(f"[MODELSCOPE] model={model_id}", flush=True)
        print(f"[MODELSCOPE] local_dir={local_dir}", flush=True)
        path = snapshot_download(
            model_id=model_id,
            local_dir=str(local_dir),
            max_workers=8,
        )
        file_count = sum(1 for item in Path(path).rglob("*") if item.is_file())
        print(f"[MODELSCOPE] DONE model={model_id} path={path} files={file_count}", flush=True)


if __name__ == "__main__":
    main()
