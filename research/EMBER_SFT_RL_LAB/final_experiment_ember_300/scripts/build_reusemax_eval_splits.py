#!/usr/bin/env python3
"""Build FIRST/SECOND splits that maximize reuse of already evaluated CMV topics."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
DEFAULT_CMV_PATH = REPO_ROOT / "source_data" / "changemyview_persuasion_kto.jsonl"
DEFAULT_DOMESTIC_PATH = ROOT / "data" / "domestic_topics_100.jsonl"
DEFAULT_PREVIOUS_TOPICS = ROOT / "data" / "final_ember_topics_300.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "eval_splits_reusemax"

DIMENSIONS = ["political", "gender", "ethnic_cultural", "age", "religion", "disability"]

FIRST_DOMESTIC_QUOTAS = {
    "political": 5,
    "gender": 4,
    "ethnic_cultural": 4,
    "age": 4,
    "religion": 4,
    "disability": 4,
}

SECOND_DOMESTIC_QUOTAS = {
    "political": 4,
    "gender": 5,
    "ethnic_cultural": 4,
    "age": 4,
    "religion": 4,
    "disability": 4,
}


def load_builder_module() -> Any:
    path = ROOT / "scripts" / "build_final_ember_topic_dataset.py"
    spec = importlib.util.spec_from_file_location("final_ember_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load builder module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def regroup(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {dimension: [] for dimension in DIMENSIONS}
    for row in rows:
        dimension = row.get("primary_dimension")
        if dimension in grouped:
            grouped[dimension].append(row)
    return grouped


def cmv_quotas_for(domestic_quotas: dict[str, int]) -> dict[str, int]:
    return {dimension: 25 - domestic_quotas[dimension] for dimension in DIMENSIONS}


def load_reused_source_ids(scored_path: Path | None, previous_topics_path: Path) -> set[str]:
    source_ids: set[str] = set()
    if scored_path and scored_path.exists():
        for row in read_jsonl(scored_path):
            topic = row.get("topic") or {}
            if topic.get("source") == "cmv" and topic.get("source_topic_id"):
                source_ids.add(topic["source_topic_id"])
    if source_ids:
        return source_ids

    for row in read_jsonl(previous_topics_path):
        if row.get("source") == "cmv" and int(row.get("final_index_1based") or 999999) <= 150:
            source_ids.add(row["source_topic_id"])
    return source_ids


def make_cmv_records(builder: Any, cmv_path: Path, total_cmv_quotas: dict[str, int], seed: int) -> list[dict[str, Any]]:
    cmv_rows = builder.read_jsonl(cmv_path)
    cmv_sample, _ = builder.stratified_sample_cmv(cmv_rows, total_cmv_quotas, seed)
    return [builder.make_cmv_record(item, index) for index, item in enumerate(cmv_sample, start=1)]


def interleave_by_dimension(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = regroup(rows)
    ordered = []
    for offset in range(25):
        for dimension in DIMENSIONS:
            bucket = grouped[dimension]
            if offset < len(bucket):
                ordered.append(bucket[offset])
    return ordered


def assign_split_ids(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows, start=1):
        new_row = dict(row)
        new_row["topic_id"] = f"{split_name.lower()}_{index:03d}"
        new_row["eval_split"] = split_name
        new_row["eval_index_1based"] = index
        new_row["topic_index_1based"] = index
        new_row["final_index_1based"] = index
        output.append(new_row)
    return output


def take_for_split(
    split_name: str,
    cmv_by_dimension: dict[str, list[dict[str, Any]]],
    domestic_by_dimension: dict[str, list[dict[str, Any]]],
    cmv_offsets: dict[str, int],
    domestic_offsets: dict[str, int],
    domestic_quotas: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    rows = []
    cmv_quotas = cmv_quotas_for(domestic_quotas)
    for dimension in DIMENSIONS:
        cmv_start = cmv_offsets[dimension]
        cmv_end = cmv_start + cmv_quotas[dimension]
        domestic_start = domestic_offsets[dimension]
        domestic_end = domestic_start + domestic_quotas[dimension]
        rows.extend(cmv_by_dimension[dimension][cmv_start:cmv_end])
        rows.extend(domestic_by_dimension[dimension][domestic_start:domestic_end])
        cmv_offsets[dimension] = cmv_end
        domestic_offsets[dimension] = domestic_end
    return assign_split_ids(interleave_by_dimension(rows), split_name), cmv_offsets, domestic_offsets


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "eval_split",
        "eval_index_1based",
        "topic_id",
        "source",
        "source_topic_id",
        "source_index_1based",
        "primary_dimension",
        "dimension_name_zh",
        "language",
        "title",
        "construction_method",
        "reused_previously_evaluated",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "source_counts": dict(Counter(row.get("source") for row in rows)),
        "primary_dimension_counts": {
            dimension: sum(1 for row in rows if row.get("primary_dimension") == dimension)
            for dimension in DIMENSIONS
        },
        "reused_cmv_topics": sum(1 for row in rows if row.get("reused_previously_evaluated")),
        "new_topics_to_run": sum(1 for row in rows if not row.get("reused_previously_evaluated")),
        "source_by_dimension": {
            source: {
                dimension: sum(
                    1
                    for row in rows
                    if row.get("source") == source and row.get("primary_dimension") == dimension
                )
                for dimension in DIMENSIONS
            }
            for source in sorted({str(row.get("source")) for row in rows})
        },
        "reused_by_dimension": {
            dimension: sum(
                1
                for row in rows
                if row.get("primary_dimension") == dimension and row.get("reused_previously_evaluated")
            )
            for dimension in DIMENSIONS
        },
    }


def write_doc(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Reuse-Max FIRST/SECOND Evaluation Splits",
        "",
        "This split keeps the corrected protocol while maximizing reuse of topics already evaluated in the previous vLLM run.",
        "",
        "## Protocol",
        "",
        "- FIRST_EVAL and SECOND_EVAL each contain 150 topics.",
        "- Each split contains 125 CMV topics and 25 domestic Chinese-context topics.",
        "- Each split contains exactly 25 primary topics per bias dimension.",
        "- Already evaluated CMV topics are prioritized and can be seeded into generated/scored outputs without rerunning.",
        "",
        "## Summary",
        "",
    ]
    for split_name in ["FIRST_EVAL", "SECOND_EVAL", "COMBINED"]:
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append(json.dumps(summary[split_name], ensure_ascii=False, indent=2))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmv-path", type=Path, default=DEFAULT_CMV_PATH)
    parser.add_argument("--domestic-path", type=Path, default=DEFAULT_DOMESTIC_PATH)
    parser.add_argument("--previous-topics-path", type=Path, default=DEFAULT_PREVIOUS_TOPICS)
    parser.add_argument("--previous-scored-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = load_builder_module()
    first_cmv_quotas = cmv_quotas_for(FIRST_DOMESTIC_QUOTAS)
    second_cmv_quotas = cmv_quotas_for(SECOND_DOMESTIC_QUOTAS)
    total_cmv_quotas = {
        dimension: first_cmv_quotas[dimension] + second_cmv_quotas[dimension]
        for dimension in DIMENSIONS
    }
    reused_source_ids = load_reused_source_ids(args.previous_scored_path, args.previous_topics_path)
    total_needed = sum(total_cmv_quotas.values())
    reusable_previous_rows = []
    previous_rows_by_id = {
        row.get("source_topic_id"): row
        for row in read_jsonl(args.previous_topics_path)
        if row.get("source") == "cmv"
    }
    for source_id in sorted(reused_source_ids, key=lambda value: int(value.split("_")[-1])):
        row = previous_rows_by_id.get(source_id)
        if row:
            item = dict(row)
            item["reused_previously_evaluated"] = True
            reusable_previous_rows.append(item)

    reusable_by_dimension = regroup(reusable_previous_rows)
    supplemental_quotas = dict(total_cmv_quotas)
    for dimension, rows in reusable_by_dimension.items():
        supplemental_quotas[dimension] = max(0, supplemental_quotas[dimension] - len(rows))

    # Oversample because the builder does not know which source_topic_id values
    # were already run; after filtering overlaps, we still need enough fresh CMV
    # rows in each dimension.
    oversample_quotas = {
        dimension: supplemental_quotas[dimension] + len(reusable_by_dimension[dimension]) + 10
        for dimension in DIMENSIONS
    }
    supplemental_rows = make_cmv_records(builder, args.cmv_path, oversample_quotas, args.seed)
    seen_ids = {row.get("source_topic_id") for row in reusable_previous_rows}
    supplemental_rows = [
        {**row, "reused_previously_evaluated": False}
        for row in supplemental_rows
        if row.get("source_topic_id") not in seen_ids
    ]

    cmv_by_dimension = {dimension: [] for dimension in DIMENSIONS}
    for row in reusable_previous_rows + supplemental_rows:
        dimension = row.get("primary_dimension")
        if dimension in cmv_by_dimension and len(cmv_by_dimension[dimension]) < total_cmv_quotas[dimension]:
            cmv_by_dimension[dimension].append(row)

    for dimension in DIMENSIONS:
        if len(cmv_by_dimension[dimension]) < total_cmv_quotas[dimension]:
            raise RuntimeError(
                f"Not enough CMV {dimension}: {len(cmv_by_dimension[dimension])} < {total_cmv_quotas[dimension]}"
            )

    domestic_records = read_jsonl(args.domestic_path)
    for row in domestic_records:
        row["reused_previously_evaluated"] = False
    domestic_by_dimension = regroup(domestic_records)

    cmv_offsets = {dimension: 0 for dimension in DIMENSIONS}
    domestic_offsets = {dimension: 0 for dimension in DIMENSIONS}
    first_rows, cmv_offsets, domestic_offsets = take_for_split(
        "FIRST_EVAL",
        cmv_by_dimension,
        domestic_by_dimension,
        cmv_offsets,
        domestic_offsets,
        FIRST_DOMESTIC_QUOTAS,
    )
    second_rows, cmv_offsets, domestic_offsets = take_for_split(
        "SECOND_EVAL",
        cmv_by_dimension,
        domestic_by_dimension,
        cmv_offsets,
        domestic_offsets,
        SECOND_DOMESTIC_QUOTAS,
    )
    combined_rows = []
    for index, row in enumerate(first_rows + second_rows, start=1):
        combined_rows.append({**row, "combined_index_1based": index})

    summary = {
        "protocol": "first_second_eval_reusemax_v1",
        "seed": args.seed,
        "total_needed": total_needed,
        "previous_reused_source_ids": len(reused_source_ids),
        "domestic_quotas": {
            "FIRST_EVAL": FIRST_DOMESTIC_QUOTAS,
            "SECOND_EVAL": SECOND_DOMESTIC_QUOTAS,
        },
        "cmv_quotas": {
            "FIRST_EVAL": first_cmv_quotas,
            "SECOND_EVAL": second_cmv_quotas,
        },
        "FIRST_EVAL": summarize(first_rows),
        "SECOND_EVAL": summarize(second_rows),
        "COMBINED": summarize(combined_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "first_eval_150.jsonl", first_rows)
    write_jsonl(args.output_dir / "second_eval_150.jsonl", second_rows)
    write_jsonl(args.output_dir / "first_second_eval_300.jsonl", combined_rows)
    write_metadata(args.output_dir / "first_eval_150_metadata.csv", first_rows)
    write_metadata(args.output_dir / "second_eval_150_metadata.csv", second_rows)
    write_metadata(args.output_dir / "first_second_eval_300_metadata.csv", combined_rows)
    (args.output_dir / "first_second_eval_reusemax_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_doc(args.output_dir / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
