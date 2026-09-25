#!/usr/bin/env python3
"""Generate detailed statistics for FIRST_EVAL / SECOND_EVAL EMBER runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODELS = ["qwen", "llama"]
VARIANTS = ["BASE", "EMBER-PROMPT", "EMBER-AGENT"]
METHODS = ["EMBER-PROMPT", "EMBER-AGENT"]
DIMENSIONS = ["political", "gender", "ethnic_cultural", "age", "religion", "disability"]
ROUNDS = list(range(6))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def flatten_rows(run_root: Path) -> list[dict[str, Any]]:
    flat_rows = []
    for model in MODELS:
        for row in read_jsonl(run_root / model / "scored_rounds.jsonl"):
            meta = row.get("meta") or {}
            topic = row.get("topic") or {}
            score = row.get("total_bias_score")
            labels = row.get("bias_labels") or {}
            flat = {
                "model": model,
                "variant": meta.get("variant_label"),
                "round": meta.get("round"),
                "topic_id": meta.get("topic_id"),
                "topic_index_1based": meta.get("topic_index_1based") or topic.get("topic_index_1based"),
                "source": topic.get("source"),
                "source_topic_id": topic.get("source_topic_id"),
                "primary_dimension": topic.get("primary_dimension"),
                "dimension_name_zh": topic.get("dimension_name_zh"),
                "language": topic.get("language"),
                "construction_method": topic.get("construction_method"),
                "eval_split": topic.get("eval_split"),
                "reused": bool(meta.get("reused_from_record_id")),
                "score": float(score) if is_number(score) else None,
                "valid": is_number(score),
            }
            for dimension in DIMENSIONS:
                value = labels.get(dimension)
                flat[f"label_{dimension}"] = float(value) if is_number(value) else None
            flat_rows.append(flat)
    return flat_rows


def summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    output = []
    for key_tuple, items in sorted(group_rows(rows, keys).items(), key=lambda item: tuple(str(v) for v in item[0])):
        scores = [row["score"] for row in items if row.get("valid")]
        record = {key: value for key, value in zip(keys, key_tuple)}
        record.update(
            {
                "total_rows": len(items),
                "valid_rows": len(scores),
                "parse_failed_rows": len(items) - len(scores),
                "mean": mean(scores),
                "std": stdev(scores),
                "median": median(scores),
                "p25": quantile(scores, 0.25),
                "p75": quantile(scores, 0.75),
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
            }
        )
        output.append(record)
    return output


def score_bin(score: float) -> str:
    if score <= 0:
        return "0"
    if score <= 1:
        return "1"
    if score <= 2:
        return "2"
    if score <= 4:
        return "3-4"
    return "5+"


def distribution(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    output = []
    for key_tuple, items in sorted(group_rows(rows, keys).items(), key=lambda item: tuple(str(v) for v in item[0])):
        valid_scores = [int(row["score"]) for row in items if row.get("valid")]
        total = len(valid_scores)
        counts = Counter(valid_scores)
        for score in sorted(counts):
            record = {key: value for key, value in zip(keys, key_tuple)}
            record.update({"score": score, "count": counts[score], "pct": counts[score] / total if total else None})
            output.append(record)
    return output


def binned_distribution(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    output = []
    order = {"0": 0, "1": 1, "2": 2, "3-4": 3, "5+": 4}
    for key_tuple, items in sorted(group_rows(rows, keys).items(), key=lambda item: tuple(str(v) for v in item[0])):
        bins = Counter(score_bin(row["score"]) for row in items if row.get("valid"))
        total = sum(bins.values())
        for bin_name in sorted(bins, key=lambda value: order[value]):
            record = {key: value for key, value in zip(keys, key_tuple)}
            record.update({"score_bin": bin_name, "count": bins[bin_name], "pct": bins[bin_name] / total if total else None})
            output.append(record)
    return output


def label_dimension_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    output = []
    for key_tuple, items in sorted(group_rows(rows, keys).items(), key=lambda item: tuple(str(v) for v in item[0])):
        for dimension in DIMENSIONS:
            scores = [row[f"label_{dimension}"] for row in items if is_number(row.get(f"label_{dimension}"))]
            record = {key: value for key, value in zip(keys, key_tuple)}
            record.update(
                {
                    "bias_dimension": dimension,
                    "valid_rows": len(scores),
                    "mean": mean(scores),
                    "std": stdev(scores),
                    "median": median(scores),
                    "nonzero_rate": sum(1 for value in scores if value > 0) / len(scores) if scores else None,
                    "severe_rate_ge3": sum(1 for value in scores if value >= 3) / len(scores) if scores else None,
                    "max": max(scores) if scores else None,
                }
            )
            output.append(record)
    return output


def wide_primary_round(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        for dimension in DIMENSIONS:
            zh = next((row.get("dimension_name_zh") for row in rows if row.get("primary_dimension") == dimension), "")
            for round_index in ROUNDS:
                record = {"model": model, "primary_dimension": dimension, "dimension_name_zh": zh, "round": round_index}
                for variant in VARIANTS:
                    values = [
                        row["score"]
                        for row in rows
                        if row.get("model") == model
                        and row.get("primary_dimension") == dimension
                        and row.get("round") == round_index
                        and row.get("variant") == variant
                        and row.get("valid")
                    ]
                    record[f"{variant}_mean"] = mean(values)
                    record[f"{variant}_n"] = len(values)
                base = record.get("BASE_mean")
                if base is not None:
                    for method in METHODS:
                        method_mean = record.get(f"{method}_mean")
                        record[f"{method}_delta_vs_BASE"] = method_mean - base if method_mean is not None else None
                output.append(record)
    return output


def wide_label_round(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        for dimension in DIMENSIONS:
            for round_index in ROUNDS:
                record = {"model": model, "bias_dimension": dimension, "round": round_index}
                for variant in VARIANTS:
                    values = [
                        row[f"label_{dimension}"]
                        for row in rows
                        if row.get("model") == model
                        and row.get("round") == round_index
                        and row.get("variant") == variant
                        and is_number(row.get(f"label_{dimension}"))
                    ]
                    record[f"{variant}_mean"] = mean(values)
                    record[f"{variant}_n"] = len(values)
                base = record.get("BASE_mean")
                if base is not None:
                    for method in METHODS:
                        method_mean = record.get(f"{method}_mean")
                        record[f"{method}_delta_vs_BASE"] = method_mean - base if method_mean is not None else None
                output.append(record)
    return output


def pairwise(rows: list[dict[str, Any]], unit_keys: list[str], strata_keys: list[str]) -> list[dict[str, Any]]:
    pivot: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not row.get("valid") or row.get("variant") not in VARIANTS:
            continue
        pivot[tuple(row.get(key) for key in unit_keys)][row["variant"]] = row

    bucket: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    for variants in pivot.values():
        if "BASE" not in variants:
            continue
        base_score = variants["BASE"]["score"]
        for method in METHODS:
            if method not in variants:
                continue
            method_row = variants[method]
            stratum = tuple(method_row.get(key) for key in strata_keys)
            bucket[(method, *stratum)].append((base_score, method_row["score"]))

    output = []
    for key_tuple, pairs in sorted(bucket.items(), key=lambda item: tuple(str(v) for v in item[0])):
        method = key_tuple[0]
        strata = key_tuple[1:]
        deltas = [method_score - base_score for base_score, method_score in pairs]
        base_values = [base_score for base_score, _ in pairs]
        method_values = [method_score for _, method_score in pairs]
        record = {"method": method}
        record.update({key: value for key, value in zip(strata_keys, strata)})
        base_mean = mean(base_values)
        record.update(
            {
                "paired_n": len(pairs),
                "base_mean": base_mean,
                "method_mean": mean(method_values),
                "delta_method_minus_base": mean(deltas),
                "relative_reduction": (-mean(deltas) / base_mean) if base_mean else None,
                "win_rate_method_lower": sum(1 for base_score, method_score in pairs if method_score < base_score) / len(pairs),
                "tie_rate": sum(1 for base_score, method_score in pairs if method_score == base_score) / len(pairs),
                "worse_rate_method_higher": sum(1 for base_score, method_score in pairs if method_score > base_score) / len(pairs),
            }
        )
        output.append(record)
    return output


def topic_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {}
    for row in rows:
        key = (row.get("topic_id"), row.get("model"))
        if key not in seen:
            seen[key] = row
    # Use qwen only to avoid double-counting topics across models.
    topic_rows = [row for (topic_id, model), row in seen.items() if model == "qwen"]
    output = []
    for keys in (["source"], ["primary_dimension"], ["source", "primary_dimension"]):
        for key_tuple, items in sorted(group_rows(topic_rows, keys).items(), key=lambda item: tuple(str(v) for v in item[0])):
            record = {key: value for key, value in zip(keys, key_tuple)}
            record["topic_count"] = len(items)
            output.append(record)
    return output


def make_report(out_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    overall = tables["overall_by_model_variant"]
    pairwise_model = tables["pairwise_by_model"]
    lines = [
        "# FIRST_EVAL Reuse-Max Statistics Report",
        "",
        "## Overall Variant Means",
        "",
    ]
    for row in overall:
        lines.append(
            f"- {row['model']} {row['variant']}: mean={fmt(row['mean'])}, valid={row['valid_rows']}/{row['total_rows']}, std={fmt(row['std'])}."
        )
    lines.extend(["", "## Pairwise Reduction vs BASE", ""])
    for row in pairwise_model:
        lines.append(
            f"- {row['model']} {row['method']}: delta={fmt(row['delta_method_minus_base'])}, "
            f"relative_reduction={fmt(row['relative_reduction'] * 100 if row['relative_reduction'] is not None else None)}%, "
            f"win={fmt(row['win_rate_method_lower'] * 100)}%, worse={fmt(row['worse_rate_method_higher'] * 100)}%."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `overall_by_model_variant.csv`",
            "- `round_curve.csv`",
            "- `primary_dimension_overall.csv`",
            "- `primary_dimension_round_curve.csv`",
            "- `primary_dimension_round_curve_wide.csv`",
            "- `bias_label_dimension_overall.csv`",
            "- `bias_label_dimension_round_curve.csv`",
            "- `bias_label_dimension_round_curve_wide.csv`",
            "- `score_distribution_overall.csv`",
            "- `score_distribution_by_round.csv`",
            "- `score_bin_distribution_overall.csv`",
            "- `score_bin_distribution_by_dimension.csv`",
            "- `pairwise_by_model.csv`",
            "- `pairwise_by_round.csv`",
            "- `pairwise_by_primary_dimension.csv`",
            "- `pairwise_by_source.csv`",
            "- `topic_counts.csv`",
        ]
    )
    (out_dir / "statistics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(args.run_root)
    tables = {
        "overall_by_model_variant": summary(rows, ["model", "variant"]),
        "round_curve": summary(rows, ["model", "variant", "round"]),
        "source_overall": summary(rows, ["model", "variant", "source"]),
        "primary_dimension_overall": summary(rows, ["model", "variant", "primary_dimension"]),
        "primary_dimension_round_curve": summary(rows, ["model", "variant", "primary_dimension", "round"]),
        "bias_label_dimension_overall": label_dimension_summary(rows, ["model", "variant"]),
        "bias_label_dimension_round_curve": label_dimension_summary(rows, ["model", "variant", "round"]),
        "primary_dimension_round_curve_wide": wide_primary_round(rows),
        "bias_label_dimension_round_curve_wide": wide_label_round(rows),
        "score_distribution_overall": distribution(rows, ["model", "variant"]),
        "score_distribution_by_round": distribution(rows, ["model", "variant", "round"]),
        "score_distribution_by_primary_dimension": distribution(rows, ["model", "variant", "primary_dimension"]),
        "score_bin_distribution_overall": binned_distribution(rows, ["model", "variant"]),
        "score_bin_distribution_by_round": binned_distribution(rows, ["model", "variant", "round"]),
        "score_bin_distribution_by_dimension": binned_distribution(rows, ["model", "variant", "primary_dimension"]),
        "pairwise_by_model": pairwise(rows, ["model", "topic_id", "round"], ["model"]),
        "pairwise_by_round": pairwise(rows, ["model", "topic_id", "round"], ["model", "round"]),
        "pairwise_by_primary_dimension": pairwise(rows, ["model", "topic_id", "round"], ["model", "primary_dimension"]),
        "pairwise_by_source": pairwise(rows, ["model", "topic_id", "round"], ["model", "source"]),
        "topic_counts": topic_counts(rows),
    }
    fields = {
        "overall_by_model_variant": ["model", "variant", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "round_curve": ["model", "variant", "round", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "source_overall": ["model", "variant", "source", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "primary_dimension_overall": ["model", "variant", "primary_dimension", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "primary_dimension_round_curve": ["model", "variant", "primary_dimension", "round", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "bias_label_dimension_overall": ["model", "variant", "bias_dimension", "valid_rows", "mean", "std", "median", "nonzero_rate", "severe_rate_ge3", "max"],
        "bias_label_dimension_round_curve": ["model", "variant", "round", "bias_dimension", "valid_rows", "mean", "std", "median", "nonzero_rate", "severe_rate_ge3", "max"],
        "primary_dimension_round_curve_wide": ["model", "primary_dimension", "dimension_name_zh", "round", "BASE_mean", "BASE_n", "EMBER-PROMPT_mean", "EMBER-PROMPT_n", "EMBER-PROMPT_delta_vs_BASE", "EMBER-AGENT_mean", "EMBER-AGENT_n", "EMBER-AGENT_delta_vs_BASE"],
        "bias_label_dimension_round_curve_wide": ["model", "bias_dimension", "round", "BASE_mean", "BASE_n", "EMBER-PROMPT_mean", "EMBER-PROMPT_n", "EMBER-PROMPT_delta_vs_BASE", "EMBER-AGENT_mean", "EMBER-AGENT_n", "EMBER-AGENT_delta_vs_BASE"],
        "score_distribution_overall": ["model", "variant", "score", "count", "pct"],
        "score_distribution_by_round": ["model", "variant", "round", "score", "count", "pct"],
        "score_distribution_by_primary_dimension": ["model", "variant", "primary_dimension", "score", "count", "pct"],
        "score_bin_distribution_overall": ["model", "variant", "score_bin", "count", "pct"],
        "score_bin_distribution_by_round": ["model", "variant", "round", "score_bin", "count", "pct"],
        "score_bin_distribution_by_dimension": ["model", "variant", "primary_dimension", "score_bin", "count", "pct"],
        "pairwise_by_model": ["model", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "pairwise_by_round": ["model", "round", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "pairwise_by_primary_dimension": ["model", "primary_dimension", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "pairwise_by_source": ["model", "source", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "topic_counts": ["source", "primary_dimension", "topic_count"],
    }
    for name, table in tables.items():
        write_csv(args.output_dir / f"{name}.csv", table, fields[name])
    make_report(args.output_dir, tables)
    print(f"Wrote statistics to {args.output_dir}")


if __name__ == "__main__":
    main()
