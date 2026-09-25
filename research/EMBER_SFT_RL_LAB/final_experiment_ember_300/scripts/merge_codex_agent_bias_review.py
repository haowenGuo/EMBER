import argparse
import json
from pathlib import Path

from vllm_common import bias_labels_from_report, ensure_dir, read_jsonl, write_analysis, write_jsonl

DEFAULT_MODELS = ("qwen", "llama")


def dump_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_scores(score_dir):
    scores = {}
    for path in sorted(Path(score_dir).glob("*.jsonl")):
        for row in read_jsonl(path) or []:
            review_id = row.get("review_id")
            report = row.get("bias_report")
            if review_id and isinstance(report, dict):
                scores[review_id] = report
    return scores


def parse_csv_list(raw):
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def merge_model_rows(input_jsonl, model_name, scores, output_dir):
    merged = []
    matched = 0
    for row in read_jsonl(input_jsonl) or []:
        meta = row.get("meta") or {}
        review_id = f"{model_name}|{meta.get('record_id')}"
        if review_id not in scores:
            continue
        report = dict(scores[review_id])
        report["parser_mode"] = report.get("parser_mode", "codex_agent_manual")
        info = bias_labels_from_report(report)
        new_row = dict(row)
        new_row["qwen_bias_report"] = row.get("bias_report")
        new_row["qwen_bias_labels"] = row.get("bias_labels")
        new_row["qwen_total_bias_score"] = row.get("total_bias_score")
        new_row["bias_report"] = report
        new_row["bias_labels"] = info["labels"] if info else None
        new_row["total_bias_score"] = info["total_score"] if info else None
        new_row["codex_agent_bias_evaluator"] = {
            "evaluator": "Codex-Agent",
            "prompt_equivalence": "Exact BiasExpert prompt semantics applied by the in-thread agent.",
        }
        merged.append(new_row)
        matched += 1
    model_dir = Path(output_dir) / model_name
    ensure_dir(model_dir)
    write_jsonl(model_dir / "scored_rounds.jsonl", merged)
    write_analysis(merged, model_dir / "analysis")
    return {"model": model_name, "matched_rows": matched, "output": str(model_dir / "scored_rounds.jsonl")}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Codex-Agent bias scores back into FIRST_EVAL rows.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="qwen,llama")
    return parser.parse_args()


def main():
    args = parse_args()
    scores = load_scores(args.score_dir)
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    summaries = []
    models = parse_csv_list(args.models) or list(DEFAULT_MODELS)
    for model_name in models:
        summaries.append(
            merge_model_rows(
                run_root / model_name / "scored_rounds.jsonl",
                model_name,
                scores,
                output_dir,
            )
        )
    dump_json(
        output_dir / "merge_manifest.json",
        {
            "score_dir": args.score_dir,
            "score_rows": len(scores),
            "models_requested": models,
            "models": summaries,
        },
    )


if __name__ == "__main__":
    main()
