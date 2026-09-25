#!/usr/bin/env python3
"""Seed vLLM generated/scored outputs from a previous completed run.

The generated/scored rows are remapped from old topic ids to the corrected
FIRST/SECOND split topic ids when the underlying CMV source_topic_id matches.
This lets vllm_generate_debates.py and vllm_score_biasexpert.py resume and
only compute genuinely new topics.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
from typing import Any


VARIANTS = ["BASE", "EMBER-PROMPT", "EMBER-AGENT"]


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


def load_split_topics(split_path: Path) -> dict[str, dict[str, Any]]:
    topics = {}
    for index, row in enumerate(read_jsonl(split_path), start=1):
        if row.get("source") != "cmv":
            continue
        source_id = row.get("source_topic_id")
        if not source_id:
            continue
        raw_content = ((row.get("prompt") or [{}])[0].get("content") or "")
        if "CMV:" in raw_content:
            pieces = raw_content.split("\n", 1)
            title = pieces[0].replace("CMV:", "", 1).strip()
            context = pieces[1].strip() if len(pieces) > 1 else title
        else:
            title = raw_content[:80].strip()
            context = raw_content.strip()
        topic = {
            "topic_id": row.get("topic_id"),
            "topic_index_1based": index,
            "title": title,
            "context": context,
            "initial_context": context,
            "original_label": row.get("label"),
        }
        for key in (
            "source",
            "source_topic_id",
            "source_index_1based",
            "primary_dimension",
            "dimension_tags",
            "dimension_scores",
            "dimension_name_zh",
            "language",
            "construction_method",
            "final_index_1based",
            "eval_split",
            "eval_index_1based",
            "reused_previously_evaluated",
        ):
            if key in row:
                topic[key] = row[key]
        topics[source_id] = topic
    return topics


def remap_row(row: dict[str, Any], new_topic: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scored = copy.deepcopy(row)
    meta = scored.get("meta") or {}
    old_record_id = meta.get("record_id")
    variant = meta.get("variant_label")
    round_index = meta.get("round")
    new_record_id = f"{variant}|{new_topic['topic_id']}|r{round_index}"
    meta["record_id"] = new_record_id
    meta["topic_id"] = new_topic["topic_id"]
    meta["topic_index_1based"] = new_topic["topic_index_1based"]
    meta["reused_from_record_id"] = old_record_id
    meta["reused_from_source_topic_id"] = new_topic.get("source_topic_id")
    meta["reuse_note"] = "Remapped from previous completed vLLM run; target_response and BiasExpert score reused."
    scored["meta"] = meta
    scored["topic"] = copy.deepcopy(new_topic)

    generated = {
        key: copy.deepcopy(scored[key])
        for key in ("meta", "topic", "target_response", "transcript", "mitigation_trace")
        if key in scored
    }
    return generated, scored


def seed_model(previous_scored_path: Path, split_topics: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    generated_rows = []
    scored_rows = []
    previous_rows = read_jsonl(previous_scored_path)
    for row in previous_rows:
        topic = row.get("topic") or {}
        source_id = topic.get("source_topic_id")
        if source_id not in split_topics:
            continue
        meta = row.get("meta") or {}
        if meta.get("variant_label") not in VARIANTS:
            continue
        generated, scored = remap_row(row, split_topics[source_id])
        generated_rows.append(generated)
        scored_rows.append(scored)

    generated_rows.sort(key=lambda r: ((r.get("meta") or {}).get("record_id") or ""))
    scored_rows.sort(key=lambda r: ((r.get("meta") or {}).get("record_id") or ""))
    write_jsonl(output_dir / "generated_rounds.jsonl", generated_rows)
    write_jsonl(output_dir / "scored_rounds.jsonl", scored_rows)
    return {
        "previous_scored_path": str(previous_scored_path),
        "seeded_generated_rows": len(generated_rows),
        "seeded_scored_rows": len(scored_rows),
        "seeded_topics": len({(r.get("topic") or {}).get("topic_id") for r in scored_rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-jsonl", type=Path, required=True)
    parser.add_argument("--previous-run-root", type=Path, required=True)
    parser.add_argument("--output-run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_topics = load_split_topics(args.split_jsonl)
    manifest = {
        "split_jsonl": str(args.split_jsonl),
        "previous_run_root": str(args.previous_run_root),
        "output_run_root": str(args.output_run_root),
        "seeded_by_model": {},
    }
    for model in ["qwen", "llama"]:
        previous_scored = args.previous_run_root / model / "scored_rounds.jsonl.gz"
        if not previous_scored.exists():
            previous_scored = args.previous_run_root / model / "scored_rounds.jsonl"
        model_out = args.output_run_root / model
        manifest["seeded_by_model"][model] = seed_model(previous_scored, split_topics, model_out)
    (args.output_run_root / "reuse_seed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
