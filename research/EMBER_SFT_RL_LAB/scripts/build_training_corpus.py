import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_gated_alignment.data import (
    bias_labels_from_report,
    build_prompt_completion,
    load_topic_map,
    prompt_hash,
)
from risk_gated_alignment.utils import dump_json, load_json, read_jsonl, write_jsonl


def parse_file_filter(value):
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def iter_source_records(source_dir, include_files=None, exclude_files=None):
    for path in sorted(Path(source_dir).glob("*.jsonl")):
        if "changemyview_persuasion_kto" in path.name:
            continue
        if include_files is not None and path.name not in include_files:
            continue
        if exclude_files is not None and path.name in exclude_files:
            continue
        for line_number, row in enumerate(read_jsonl(path), 1):
            yield path.name, line_number, row


def main():
    parser = argparse.ArgumentParser(description="Build risk-head, SFT, RL and eval corpora from EMBER JSONL results.")
    parser.add_argument("--source-dir", default=str(ROOT / "source_data"))
    parser.add_argument("--cmv-path", default=str(ROOT / "source_data" / "changemyview_persuasion_kto.jsonl"))
    parser.add_argument("--split-path", default=str(ROOT / "datasets" / "topic_split.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "datasets"))
    parser.add_argument("--positive-threshold", type=int, default=1)
    parser.add_argument("--negative-threshold", type=int, default=3)
    parser.add_argument("--include-files", help="Comma-separated JSONL file names to include from source-dir.")
    parser.add_argument("--exclude-files", help="Comma-separated JSONL file names to exclude from source-dir.")
    args = parser.parse_args()

    split = load_json(args.split_path)
    topic_map = load_topic_map(args.cmv_path)
    grouped = defaultdict(lambda: defaultdict(list))
    grouped_rl = defaultdict(lambda: defaultdict(dict))
    include_files = parse_file_filter(args.include_files)
    exclude_files = parse_file_filter(args.exclude_files)

    for source_file, line_number, row in iter_source_records(args.source_dir, include_files, exclude_files):
        bias_info = bias_labels_from_report(row.get("bias_report"))
        if not bias_info:
            continue

        record = build_prompt_completion(row, topic_map)
        if not record:
            continue
        topic_id = record["topic_id"]
        split_name = next((name for name, ids in split.items() if topic_id in ids), None)
        if split_name is None:
            continue

        payload = {
            **record,
            "source_file": source_file,
            "source_line": line_number,
            "bias_labels": bias_info["labels"],
            "total_score": bias_info["total_score"],
        }

        model_names = [record["model_family"], "combined"]
        for model_name in model_names:
            grouped[model_name][f"risk_head_{split_name}"].append(payload)
            grouped[model_name][f"eval_{split_name}"].append(payload)
            grouped_rl[model_name][f"rl_{split_name}"][prompt_hash(record["prompt"])] = {
                "prompt": record["prompt"],
                "topic_id": record["topic_id"],
                "round": record["round"],
                "model_family": record["model_family"],
                "topic_text": record["topic_text"],
                "source_file": source_file,
            }

            if bias_info["total_score"] <= args.positive_threshold:
                grouped[model_name][f"sft_{split_name}"].append(payload)
            if bias_info["total_score"] >= args.negative_threshold:
                grouped[model_name][f"negative_{split_name}"].append(payload)

    summary = {}
    for model_name, buckets in grouped.items():
        model_dir = Path(args.output_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        summary[model_name] = {}
        for bucket_name, rows in buckets.items():
            rows = sorted(rows, key=lambda item: (item["topic_id"], item["round"], item["source_file"], item["source_line"]))
            write_jsonl(model_dir / f"{bucket_name}.jsonl", rows)
            summary[model_name][bucket_name] = len(rows)

        for bucket_name, row_map in grouped_rl[model_name].items():
            rows = sorted(row_map.values(), key=lambda item: (item["topic_id"], item["round"]))
            write_jsonl(model_dir / f"{bucket_name}.jsonl", rows)
            summary[model_name][bucket_name] = len(rows)

    dump_json(Path(args.output_dir) / "corpus_summary.json", summary)
    print(f"Saved corpora to {args.output_dir}")
    print(summary)


if __name__ == "__main__":
    main()
