#!/usr/bin/env python3
"""Build balanced FIRST_EVAL and SECOND_EVAL topic splits.

Protocol:
- FIRST_EVAL: 150 topics for first evaluation and later SFT/RL data construction.
- SECOND_EVAL: 150 disjoint topics for held-out testing.
- Each split contains 25 domestic topics and 125 CMV topics.
- Each split is balanced by primary bias dimension: 25 topics per dimension.

This intentionally creates a new split set instead of overwriting the older
`final_ember_topics_300.jsonl`, whose first 150 rows were all CMV because the
file was ordered as 200 CMV + 100 domestic.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
DEFAULT_CMV_PATH = REPO_ROOT / "source_data" / "changemyview_persuasion_kto.jsonl"
DEFAULT_DOMESTIC_PATH = ROOT / "data" / "domestic_topics_100.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "eval_splits"

DIMENSIONS = ["political", "gender", "ethnic_cultural", "age", "religion", "disability"]
DIMENSION_NAMES_ZH = {
    "political": "政治与公共治理",
    "gender": "性别与婚育平等",
    "ethnic_cultural": "民族、地域与文化差异",
    "age": "年龄与代际关系",
    "religion": "宗教信仰与民俗文化",
    "disability": "残疾、疾病与心理健康",
}

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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
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


def quota_sum(quotas: dict[str, int]) -> int:
    return sum(quotas[dimension] for dimension in DIMENSIONS)


def cmv_quotas_for(domestic_quotas: dict[str, int]) -> dict[str, int]:
    return {dimension: 25 - domestic_quotas[dimension] for dimension in DIMENSIONS}


def regroup(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {dimension: [] for dimension in DIMENSIONS}
    for row in rows:
        dimension = row.get("primary_dimension")
        if dimension in grouped:
            grouped[dimension].append(row)
    return grouped


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


def interleave_by_dimension(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = regroup(rows)
    ordered = []
    for offset in range(25):
        for dimension in DIMENSIONS:
            bucket = grouped[dimension]
            if offset < len(bucket):
                ordered.append(bucket[offset])
    return ordered


def make_split(
    split_name: str,
    cmv_by_dimension: dict[str, list[dict[str, Any]]],
    domestic_by_dimension: dict[str, list[dict[str, Any]]],
    cmv_offsets: dict[str, int],
    domestic_offsets: dict[str, int],
    domestic_quotas: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    cmv_quotas = cmv_quotas_for(domestic_quotas)
    rows = []
    for dimension in DIMENSIONS:
        cmv_start = cmv_offsets[dimension]
        cmv_end = cmv_start + cmv_quotas[dimension]
        domestic_start = domestic_offsets[dimension]
        domestic_end = domestic_start + domestic_quotas[dimension]
        rows.extend(cmv_by_dimension[dimension][cmv_start:cmv_end])
        rows.extend(domestic_by_dimension[dimension][domestic_start:domestic_end])
        cmv_offsets[dimension] = cmv_end
        domestic_offsets[dimension] = domestic_end
    rows = interleave_by_dimension(rows)
    return assign_split_ids(rows, split_name), cmv_offsets, domestic_offsets


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
        "primary_dimension_counts": {dimension: sum(1 for row in rows if row.get("primary_dimension") == dimension) for dimension in DIMENSIONS},
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
    }


def write_split_doc(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# FIRST/SECOND Evaluation Split Protocol",
        "",
        "This file documents the corrected split protocol after discovering that the old first-150 slice was ordered as CMV-only.",
        "",
        "## Protocol",
        "",
        "- FIRST_EVAL: 150 topics for first evaluation and downstream SFT/RL data construction.",
        "- SECOND_EVAL: 150 disjoint topics for held-out testing of BASE, EMBER-PROMPT, EMBER-AGENT, SFT, and SFT+RL.",
        "- Each split contains 125 CMV topics and 25 domestic Chinese-context topics.",
        "- Each split is balanced by six primary bias dimensions: 25 topics per dimension.",
        "- Domestic topics are allocated as evenly as possible because 25 is not divisible by 6.",
        "",
        "## Why This Correction Was Needed",
        "",
        "The earlier `final_ember_topics_300.jsonl` was globally balanced, but ordered as 200 CMV rows followed by 100 domestic rows. Running rows 1-150 therefore used only CMV topics and missed all domestic topics. It also missed disability as a primary topic because disability CMV topics were placed in rows 168-200.",
        "",
        "## Split Summary",
        "",
    ]
    for split_name in ["FIRST_EVAL", "SECOND_EVAL"]:
        split_summary = summary[split_name]
        lines.extend(
            [
                f"### {split_name}",
                "",
                f"- Total: {split_summary['total']}",
                f"- Source counts: {split_summary['source_counts']}",
                f"- Primary dimension counts: {split_summary['primary_dimension_counts']}",
                f"- Source by dimension: {split_summary['source_by_dimension']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmv-path", type=Path, default=DEFAULT_CMV_PATH)
    parser.add_argument("--domestic-path", type=Path, default=DEFAULT_DOMESTIC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = load_builder_module()
    if quota_sum(FIRST_DOMESTIC_QUOTAS) != 25 or quota_sum(SECOND_DOMESTIC_QUOTAS) != 25:
        raise RuntimeError("Domestic quotas must sum to 25 for each split.")

    first_cmv_quotas = cmv_quotas_for(FIRST_DOMESTIC_QUOTAS)
    second_cmv_quotas = cmv_quotas_for(SECOND_DOMESTIC_QUOTAS)
    total_cmv_quotas = {
        dimension: first_cmv_quotas[dimension] + second_cmv_quotas[dimension]
        for dimension in DIMENSIONS
    }

    cmv_rows = builder.read_jsonl(args.cmv_path)
    cmv_sample, cmv_diagnostics = builder.stratified_sample_cmv(cmv_rows, total_cmv_quotas, args.seed)
    cmv_records = [
        builder.make_cmv_record(item, index)
        for index, item in enumerate(cmv_sample, start=1)
    ]
    domestic_records = read_jsonl(args.domestic_path)

    cmv_by_dimension = regroup(cmv_records)
    domestic_by_dimension = regroup(domestic_records)
    for dimension in DIMENSIONS:
        needed_cmv = total_cmv_quotas[dimension]
        needed_domestic = FIRST_DOMESTIC_QUOTAS[dimension] + SECOND_DOMESTIC_QUOTAS[dimension]
        if len(cmv_by_dimension[dimension]) < needed_cmv:
            raise RuntimeError(f"Not enough CMV {dimension}: {len(cmv_by_dimension[dimension])} < {needed_cmv}")
        if len(domestic_by_dimension[dimension]) < needed_domestic:
            raise RuntimeError(f"Not enough domestic {dimension}: {len(domestic_by_dimension[dimension])} < {needed_domestic}")

    cmv_offsets = {dimension: 0 for dimension in DIMENSIONS}
    domestic_offsets = {dimension: 0 for dimension in DIMENSIONS}
    first_rows, cmv_offsets, domestic_offsets = make_split(
        "FIRST_EVAL",
        cmv_by_dimension,
        domestic_by_dimension,
        cmv_offsets,
        domestic_offsets,
        FIRST_DOMESTIC_QUOTAS,
    )
    second_rows, cmv_offsets, domestic_offsets = make_split(
        "SECOND_EVAL",
        cmv_by_dimension,
        domestic_by_dimension,
        cmv_offsets,
        domestic_offsets,
        SECOND_DOMESTIC_QUOTAS,
    )

    combined_rows = []
    for global_index, row in enumerate(first_rows + second_rows, start=1):
        new_row = dict(row)
        new_row["combined_index_1based"] = global_index
        combined_rows.append(new_row)

    summary = {
        "protocol": "first_second_eval_balanced_v1",
        "seed": args.seed,
        "cmv_path": str(args.cmv_path),
        "domestic_path": str(args.domestic_path),
        "domestic_quotas": {
            "FIRST_EVAL": FIRST_DOMESTIC_QUOTAS,
            "SECOND_EVAL": SECOND_DOMESTIC_QUOTAS,
        },
        "cmv_quotas": {
            "FIRST_EVAL": first_cmv_quotas,
            "SECOND_EVAL": second_cmv_quotas,
        },
        "cmv_sampling": cmv_diagnostics,
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
    (args.output_dir / "first_second_eval_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_split_doc(args.output_dir / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
