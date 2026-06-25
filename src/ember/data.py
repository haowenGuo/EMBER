"""Data helpers for EMBER JSONL experiment records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_cmv_range(
    rows: list[dict[str, Any]],
    start: int = 1,
    end: int = 100,
    topic_path: tuple[str, ...] = ("meta", "topic_id"),
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        topic_id: Any = row
        for key in topic_path:
            topic_id = topic_id.get(key) if isinstance(topic_id, dict) else None
        if not isinstance(topic_id, str) or not topic_id.startswith("cmv_"):
            continue
        try:
            topic_number = int(topic_id.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if start <= topic_number <= end:
            result.append(row)
    return result


def deduplicate_topic_round(
    rows: list[dict[str, Any]],
    keep: str = "last",
) -> list[dict[str, Any]]:
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")

    index: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in rows:
        meta = row.get("meta", {})
        key = (meta.get("topic_id"), meta.get("rounds"))
        if key not in index or keep == "last":
            index[key] = row
    return list(index.values())
