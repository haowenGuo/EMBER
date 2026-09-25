#!/usr/bin/env python3
"""Analyze BASE / EMBER-PROMPT / EMBER-AGENT effects for the final EMBER run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIMENSIONS = ["political", "gender", "ethnic_cultural", "age", "religion", "disability"]
VARIANTS = ["BASE", "EMBER-PROMPT", "EMBER-AGENT"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
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


def bootstrap_ci(values: list[float], *, iters: int = 2000, seed: int = 13) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(iters):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        boot.append(total / n)
    return quantile(boot, 0.025), quantile(boot, 0.975)


def fmt(value: Any, ndigits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_topics(path: Path) -> dict[str, dict[str, Any]]:
    topics = {}
    for row in read_jsonl(path):
        topic_id = row.get("topic_id")
        if topic_id:
            topics[topic_id] = row
    return topics


def flatten_scored(model: str, run_root: Path, topics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    path = run_root / model / "scored_rounds.jsonl"
    for row in read_jsonl(path):
        meta = row.get("meta") or {}
        topic = row.get("topic") or topics.get(meta.get("topic_id"), {})
        score = row.get("total_bias_score")
        labels = row.get("bias_labels") or {}
        flat = {
            "model": model,
            "record_id": meta.get("record_id"),
            "topic_id": meta.get("topic_id"),
            "topic_index_1based": meta.get("topic_index_1based") or topic.get("final_index_1based"),
            "source": topic.get("source"),
            "source_topic_id": topic.get("source_topic_id"),
            "primary_dimension": topic.get("primary_dimension"),
            "dimension_name_zh": topic.get("dimension_name_zh"),
            "language": topic.get("language"),
            "construction_method": topic.get("construction_method"),
            "variant": meta.get("variant_label"),
            "round": meta.get("round"),
            "target_endpoint_index": meta.get("target_endpoint_index"),
            "score": float(score) if is_number(score) else None,
            "valid": is_number(score),
            "title": topic.get("title") or row.get("topic", {}).get("title"),
        }
        for dim in DIMENSIONS:
            flat[f"label_{dim}"] = labels.get(dim)
        rows.append(flat)
    return rows


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    return groups


def simple_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for group_key, group_rows in sorted(group_by(rows, keys).items(), key=lambda item: tuple(str(v) for v in item[0])):
        scores = [r["score"] for r in group_rows if r.get("valid")]
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(
            {
                "total_rows": len(group_rows),
                "valid_rows": len(scores),
                "parse_failed_rows": len(group_rows) - len(scores),
                "mean": mean(scores),
                "std": stdev(scores),
                "median": median(scores),
                "p25": quantile(scores, 0.25),
                "p75": quantile(scores, 0.75),
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
            }
        )
        out.append(row)
    return out


def pivot_scores(rows: list[dict[str, Any]], unit_keys: list[str]) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    pivot: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not row.get("valid") or row.get("variant") not in VARIANTS:
            continue
        key = tuple(row.get(k) for k in unit_keys)
        pivot[key][row["variant"]] = row
    return pivot


def bucket_base(value: float) -> str:
    if value <= 0:
        return "base=0"
    if value <= 1:
        return "base=1"
    if value <= 2:
        return "base=2"
    if value <= 4:
        return "base=3-4"
    return "base>=5"


def pairwise(rows: list[dict[str, Any]], unit_keys: list[str], strata_keys: list[str] | None = None) -> list[dict[str, Any]]:
    strata_keys = strata_keys or []
    pivot = pivot_scores(rows, unit_keys)
    buckets: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    for _, variants in pivot.items():
        if "BASE" not in variants:
            continue
        base = variants["BASE"]["score"]
        for method in ["EMBER-PROMPT", "EMBER-AGENT"]:
            if method not in variants:
                continue
            method_score = variants[method]["score"]
            first = variants[method]
            stratum = tuple(first.get(k) for k in strata_keys)
            buckets[(method, *stratum)].append((base, method_score))

    out = []
    for key, pairs in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0])):
        method = key[0]
        stratum_values = key[1:]
        bases = [p[0] for p in pairs]
        methods = [p[1] for p in pairs]
        deltas = [m - b for b, m in pairs]
        improvements = [b - m for b, m in pairs]
        ci_low, ci_high = bootstrap_ci(deltas)
        base_mean = mean(bases)
        method_mean = mean(methods)
        row = {"method": method}
        row.update({k: v for k, v in zip(strata_keys, stratum_values)})
        row.update(
            {
                "paired_n": len(pairs),
                "base_mean": base_mean,
                "method_mean": method_mean,
                "delta_method_minus_base": mean(deltas),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "relative_reduction": (mean(improvements) / base_mean) if base_mean and base_mean > 0 else None,
                "win_rate_method_lower": sum(1 for b, m in pairs if m < b) / len(pairs),
                "tie_rate": sum(1 for b, m in pairs if m == b) / len(pairs),
                "worse_rate_method_higher": sum(1 for b, m in pairs if m > b) / len(pairs),
            }
        )
        out.append(row)
    return out


def pairwise_by_base_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pivot = pivot_scores(rows, ["model", "topic_id", "round"])
    buckets: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for _, variants in pivot.items():
        if "BASE" not in variants:
            continue
        base = variants["BASE"]["score"]
        model = variants["BASE"]["model"]
        bucket = bucket_base(base)
        for method in ["EMBER-PROMPT", "EMBER-AGENT"]:
            if method in variants:
                buckets[(model, method, bucket)].append((base, variants[method]["score"]))
    out = []
    order = {"base=0": 0, "base=1": 1, "base=2": 2, "base=3-4": 3, "base>=5": 4}
    for (model, method, bucket), pairs in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1], order[item[0][2]])):
        bases = [p[0] for p in pairs]
        methods = [p[1] for p in pairs]
        deltas = [m - b for b, m in pairs]
        ci_low, ci_high = bootstrap_ci(deltas)
        out.append(
            {
                "model": model,
                "method": method,
                "base_bucket": bucket,
                "paired_n": len(pairs),
                "base_mean": mean(bases),
                "method_mean": mean(methods),
                "delta_method_minus_base": mean(deltas),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "win_rate_method_lower": sum(1 for b, m in pairs if m < b) / len(pairs),
                "tie_rate": sum(1 for b, m in pairs if m == b) / len(pairs),
                "worse_rate_method_higher": sum(1 for b, m in pairs if m > b) / len(pairs),
            }
        )
    return out


def topic_level(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_by([r for r in rows if r.get("valid")], ["model", "topic_id", "variant"])
    topic_variant_rows = []
    for key, group_rows in grouped.items():
        model, topic_id, variant = key
        scores = [r["score"] for r in group_rows]
        first = group_rows[0]
        topic_variant_rows.append(
            {
                "model": model,
                "topic_id": topic_id,
                "topic_index_1based": first.get("topic_index_1based"),
                "source": first.get("source"),
                "primary_dimension": first.get("primary_dimension"),
                "dimension_name_zh": first.get("dimension_name_zh"),
                "variant": variant,
                "valid_rounds": len(scores),
                "mean_score": mean(scores),
                "title": first.get("title"),
            }
        )

    pivot = pivot_scores(
        [
            {
                **row,
                "round": "topic_avg",
                "score": row["mean_score"],
                "valid": row["mean_score"] is not None,
            }
            for row in topic_variant_rows
        ],
        ["model", "topic_id", "round"],
    )
    deltas = []
    for _, variants in pivot.items():
        if "BASE" not in variants:
            continue
        base = variants["BASE"]
        for method in ["EMBER-PROMPT", "EMBER-AGENT"]:
            if method not in variants:
                continue
            method_row = variants[method]
            deltas.append(
                {
                    "model": base["model"],
                    "topic_id": base["topic_id"],
                    "topic_index_1based": base.get("topic_index_1based"),
                    "source": base.get("source"),
                    "primary_dimension": base.get("primary_dimension"),
                    "dimension_name_zh": base.get("dimension_name_zh"),
                    "method": method,
                    "base_topic_mean": base["score"],
                    "method_topic_mean": method_row["score"],
                    "delta_method_minus_base": method_row["score"] - base["score"],
                    "title": base.get("title"),
                }
            )
    top_improved = sorted(deltas, key=lambda r: r["delta_method_minus_base"])[:40]
    top_regressed = sorted(deltas, key=lambda r: r["delta_method_minus_base"], reverse=True)[:40]
    return topic_variant_rows, top_improved, top_regressed


def dimension_label_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (model, variant), group_rows in sorted(group_by(rows, ["model", "variant"]).items()):
        valid = [r for r in group_rows if r.get("valid")]
        for dim in DIMENSIONS:
            vals = [r.get(f"label_{dim}") for r in valid if is_number(r.get(f"label_{dim}"))]
            out.append(
                {
                    "model": model,
                    "variant": variant,
                    "bias_dimension": dim,
                    "valid_rows": len(vals),
                    "mean_score": mean([float(v) for v in vals]),
                    "nonzero_rate": sum(1 for v in vals if v > 0) / len(vals) if vals else None,
                    "severe_rate_ge3": sum(1 for v in vals if v >= 3) / len(vals) if vals else None,
                }
            )
    return out


def read_variant_summary(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            variant = row.get("variant") or row.get("variant_label")
            mean_value = row.get("mean_total_bias_score")
            rows.append(
                {
                    "run_label": label,
                    "variant": variant,
                    "rows": row.get("rows") or row.get("total_rows"),
                    "valid_rows": row.get("valid_rows"),
                    "mean_total_bias_score": float(mean_value) if mean_value not in (None, "") else None,
                }
            )
        return rows


def comparison_summary(final_variant_rows: list[dict[str, Any]], small_variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, float]] = defaultdict(dict)
    for row in final_variant_rows:
        by_run[f"final150_{row['model']}"][row["variant"]] = row["mean"]
    for row in small_variant_rows:
        by_run[row["run_label"]][row["variant"]] = row["mean_total_bias_score"]
    out = []
    for run_label, values in sorted(by_run.items()):
        base = values.get("BASE")
        for method in ["EMBER-PROMPT", "EMBER-AGENT"]:
            method_mean = values.get(method)
            out.append(
                {
                    "run_label": run_label,
                    "base_mean": base,
                    "method": method,
                    "method_mean": method_mean,
                    "absolute_delta": (method_mean - base) if base is not None and method_mean is not None else None,
                    "relative_reduction": ((base - method_mean) / base) if base and method_mean is not None else None,
                }
            )
    return out


def format_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append({k: fmt(row.get(k)) for k in fields})
    return out


def make_report(out_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    comparison = tables["small_vs_final"]
    paired = tables["pairwise_by_model"]
    bucket = tables["pairwise_by_base_bucket"]
    lines = [
        "# EMBER Prompt/Agent Effect Deep Analysis",
        "",
        "## Key Findings",
        "",
    ]
    for row in comparison:
        if row["run_label"].startswith("final150"):
            lines.append(
                f"- {row['run_label']} {row['method']}: mean {fmt(row['method_mean'])} vs BASE {fmt(row['base_mean'])}, "
                f"delta {fmt(row['absolute_delta'])}, relative reduction {fmt(row['relative_reduction'] * 100 if row['relative_reduction'] is not None else None)}%."
            )
    lines.extend(["", "## Why the 3-topic test looked stronger", ""])
    for row in comparison:
        if row["run_label"].startswith("small_"):
            lines.append(
                f"- {row['run_label']} {row['method']}: delta {fmt(row['absolute_delta'])}, "
                f"relative reduction {fmt(row['relative_reduction'] * 100 if row['relative_reduction'] is not None else None)}%."
            )
    lines.extend(
        [
            "",
            "The final run is much broader: 150 topics, rounds 0-5, mixed source/dimension strata, and a local Qwen3.5-9B attacker. "
            "The earlier small runs only had 33 rows and were therefore very sensitive to topic selection and high-risk examples.",
            "",
            "## Paired Bootstrap CI",
            "",
        ]
    )
    for row in paired:
        lines.append(
            f"- {row['model']} {row['method']}: paired n={row['paired_n']}, delta={fmt(row['delta_method_minus_base'])}, "
            f"95% CI [{fmt(row['delta_ci95_low'])}, {fmt(row['delta_ci95_high'])}], "
            f"win={fmt(row['win_rate_method_lower'] * 100)}%, worse={fmt(row['worse_rate_method_higher'] * 100)}%."
        )
    lines.extend(["", "## Baseline Ceiling/Floor Effect", ""])
    for row in bucket:
        if row["base_bucket"] in ("base=0", "base=1", "base=3-4", "base>=5"):
            lines.append(
                f"- {row['model']} {row['method']} {row['base_bucket']}: n={row['paired_n']}, "
                f"delta={fmt(row['delta_method_minus_base'])}, win={fmt(row['win_rate_method_lower'] * 100)}%, "
                f"worse={fmt(row['worse_rate_method_higher'] * 100)}%."
            )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `overall_summary.csv`",
            "- `round_summary_deep.csv`",
            "- `source_summary.csv`",
            "- `primary_dimension_summary.csv`",
            "- `pairwise_by_model.csv`",
            "- `pairwise_by_round.csv`",
            "- `pairwise_by_source.csv`",
            "- `pairwise_by_primary_dimension.csv`",
            "- `pairwise_by_base_bucket.csv`",
            "- `dimension_label_summary.csv`",
            "- `topic_variant_means.csv`",
            "- `top_topic_improvements.csv`",
            "- `top_topic_regressions.csv`",
            "- `small_vs_final.csv`",
        ]
    )
    (out_dir / "deep_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--topics-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--small-qwen-variant-summary", type=Path)
    parser.add_argument("--small-llama-variant-summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topics = load_topics(args.topics_jsonl)
    rows = []
    for model in ["qwen", "llama"]:
        rows.extend(flatten_scored(model, args.run_root, topics))

    small_rows = []
    if args.small_qwen_variant_summary:
        small_rows.extend(read_variant_summary(args.small_qwen_variant_summary, "small3_qwen_prompt_agent_v2"))
    if args.small_llama_variant_summary:
        small_rows.extend(read_variant_summary(args.small_llama_variant_summary, "small3_llama_prompt_agent_v2"))

    tables: dict[str, list[dict[str, Any]]] = {}
    tables["overall_summary"] = simple_summary(rows, ["model", "variant"])
    tables["round_summary_deep"] = simple_summary(rows, ["model", "variant", "round"])
    tables["source_summary"] = simple_summary(rows, ["model", "variant", "source"])
    tables["primary_dimension_summary"] = simple_summary(rows, ["model", "variant", "primary_dimension"])
    tables["pairwise_by_model"] = pairwise(rows, ["model", "topic_id", "round"], ["model"])
    tables["pairwise_by_round"] = pairwise(rows, ["model", "topic_id", "round"], ["model", "round"])
    tables["pairwise_by_source"] = pairwise(rows, ["model", "topic_id", "round"], ["model", "source"])
    tables["pairwise_by_primary_dimension"] = pairwise(rows, ["model", "topic_id", "round"], ["model", "primary_dimension"])
    tables["pairwise_by_base_bucket"] = pairwise_by_base_bucket(rows)
    tables["dimension_label_summary"] = dimension_label_summary(rows)
    topic_variant_means, top_improved, top_regressed = topic_level(rows)
    tables["topic_variant_means"] = topic_variant_means
    tables["top_topic_improvements"] = top_improved
    tables["top_topic_regressions"] = top_regressed
    tables["small_vs_final"] = comparison_summary(tables["overall_summary"], small_rows)

    fields = {
        "overall_summary": ["model", "variant", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "round_summary_deep": ["model", "variant", "round", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "source_summary": ["model", "variant", "source", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "primary_dimension_summary": ["model", "variant", "primary_dimension", "total_rows", "valid_rows", "parse_failed_rows", "mean", "std", "median", "p25", "p75", "min", "max"],
        "pairwise_by_model": ["model", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "delta_ci95_low", "delta_ci95_high", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "pairwise_by_round": ["model", "round", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "delta_ci95_low", "delta_ci95_high", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "pairwise_by_source": ["model", "source", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "delta_ci95_low", "delta_ci95_high", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "pairwise_by_primary_dimension": ["model", "primary_dimension", "method", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "delta_ci95_low", "delta_ci95_high", "relative_reduction", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "pairwise_by_base_bucket": ["model", "method", "base_bucket", "paired_n", "base_mean", "method_mean", "delta_method_minus_base", "delta_ci95_low", "delta_ci95_high", "win_rate_method_lower", "tie_rate", "worse_rate_method_higher"],
        "dimension_label_summary": ["model", "variant", "bias_dimension", "valid_rows", "mean_score", "nonzero_rate", "severe_rate_ge3"],
        "topic_variant_means": ["model", "topic_id", "topic_index_1based", "source", "primary_dimension", "dimension_name_zh", "variant", "valid_rounds", "mean_score", "title"],
        "top_topic_improvements": ["model", "topic_id", "topic_index_1based", "source", "primary_dimension", "dimension_name_zh", "method", "base_topic_mean", "method_topic_mean", "delta_method_minus_base", "title"],
        "top_topic_regressions": ["model", "topic_id", "topic_index_1based", "source", "primary_dimension", "dimension_name_zh", "method", "base_topic_mean", "method_topic_mean", "delta_method_minus_base", "title"],
        "small_vs_final": ["run_label", "base_mean", "method", "method_mean", "absolute_delta", "relative_reduction"],
    }
    for name, table in tables.items():
        write_csv(args.output_dir / f"{name}.csv", format_rows(table, fields[name]), fields[name])

    make_report(args.output_dir, tables)
    print(f"Wrote deep analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
