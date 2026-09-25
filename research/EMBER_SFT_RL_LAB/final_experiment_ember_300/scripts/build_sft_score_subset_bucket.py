import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


COPY_FILES = [
    "risk_head_train.jsonl",
    "risk_head_dev.jsonl",
    "eval_train.jsonl",
    "eval_dev.jsonl",
    "negative_train.jsonl",
    "negative_dev.jsonl",
    "rl_train.jsonl",
    "rl_dev.jsonl",
]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_sft_rows(rows, max_score: int):
    kept = []
    score_counter = Counter()
    for row in rows:
        score = row.get("total_score")
        if score is None:
            continue
        score = int(score)
        if score <= max_score:
            kept.append(row)
            score_counter[score] += 1
    return kept, score_counter


def main():
    parser = argparse.ArgumentParser(
        description="Copy an existing dataset bucket and replace only SFT splits with a lower-score subset."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sft-max-score", required=True, type=int)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in COPY_FILES:
        shutil.copy2(source_dir / name, output_dir / name)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "sft_max_score": args.sft_max_score,
        "copied_files": COPY_FILES,
        "sft_counts": {},
    }

    for split in ["train", "dev"]:
        source_path = source_dir / f"sft_{split}.jsonl"
        rows = list(read_jsonl(source_path))
        kept_rows, score_counter = filter_sft_rows(rows, args.sft_max_score)
        write_jsonl(output_dir / f"sft_{split}.jsonl", kept_rows)
        summary["sft_counts"][split] = {
            "source_rows": len(rows),
            "kept_rows": len(kept_rows),
            "score_histogram": dict(sorted(score_counter.items())),
        }

    with (output_dir / "corpus_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
