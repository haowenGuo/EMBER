import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_gated_alignment.data import bias_labels_from_report, build_prompt_completion, load_topic_map, normalize_model_name
from risk_gated_alignment.utils import dump_json, read_jsonl


def short_counts(counter, limit=5):
    return dict(counter.most_common(limit))


def summarize_scores(scores):
    if not scores:
        return {
            "valid": 0,
            "mean": None,
            "zero_pct": None,
            "high_ge_3": 0,
            "high_ge_4": 0,
        }
    return {
        "valid": len(scores),
        "mean": sum(scores) / len(scores),
        "zero_pct": sum(1 for score in scores if score == 0) / len(scores),
        "high_ge_3": sum(1 for score in scores if score >= 3),
        "high_ge_4": sum(1 for score in scores if score >= 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit EMBER JSONL source files before building risk-head/SFT/RL corpora. "
            "This is meant to catch cross-model protocol mismatches early."
        )
    )
    parser.add_argument("--source-dir", default=str(ROOT / "source_data"))
    parser.add_argument("--cmv-path", default=str(ROOT / "source_data" / "changemyview_persuasion_kto.jsonl"))
    parser.add_argument("--output-json", default=str(ROOT / "datasets" / "source_audit.json"))
    args = parser.parse_args()

    topic_map = load_topic_map(args.cmv_path)
    per_file = {}
    per_family_scores = defaultdict(list)
    per_family_files = defaultdict(Counter)

    for path in sorted(Path(args.source_dir).glob("*.jsonl")):
        if "changemyview_persuasion_kto" in path.name:
            continue

        scores = []
        raw_rows = 0
        valid_bias = 0
        built_records = 0
        topics = set()
        meta_models = Counter()
        last_senders = Counter()
        model_families = Counter()
        empty_target_response = 0

        for row in read_jsonl(path):
            raw_rows += 1
            meta = row.get("meta") or {}
            if meta.get("topic_id"):
                topics.add(meta.get("topic_id"))
            if meta.get("model"):
                meta_models[meta.get("model")] += 1
            transcript = row.get("transcript") or []
            if transcript and isinstance(transcript[-1], dict):
                last_senders[transcript[-1].get("sender") or transcript[-1].get("role") or "unknown"] += 1
            if not str(row.get("target_response") or "").strip():
                empty_target_response += 1

            bias_info = bias_labels_from_report(row.get("bias_report"))
            if bias_info is None:
                continue
            valid_bias += 1
            score = bias_info["total_score"]
            scores.append(score)

            record = build_prompt_completion(row, topic_map)
            if record is None:
                continue
            built_records += 1
            family = record["model_family"]
            model_families[family] += 1
            per_family_scores[family].append(score)
            per_family_files[family][path.name] += 1

        summary = summarize_scores(scores)
        per_file[path.name] = {
            "raw_rows": raw_rows,
            "valid_bias": valid_bias,
            "built_records": built_records,
            "score_mean": summary["mean"],
            "zero_pct": summary["zero_pct"],
            "high_ge_3": summary["high_ge_3"],
            "high_ge_4": summary["high_ge_4"],
            "topic_count": len(topics),
            "empty_target_response": empty_target_response,
            "model_families_after_parse": short_counts(model_families),
            "meta_models": short_counts(meta_models),
            "last_senders": short_counts(last_senders),
        }

    per_family = {}
    for family, scores in sorted(per_family_scores.items()):
        summary = summarize_scores(scores)
        per_family[family] = {
            "records": len(scores),
            "score_mean": summary["mean"],
            "zero_pct": summary["zero_pct"],
            "high_ge_3": summary["high_ge_3"],
            "high_ge_4": summary["high_ge_4"],
            "source_files": short_counts(per_family_files[family], limit=20),
        }

    output = {
        "source_dir": str(Path(args.source_dir).resolve()),
        "cmv_path": str(Path(args.cmv_path).resolve()),
        "per_file": per_file,
        "per_family": per_family,
    }
    dump_json(args.output_json, output)

    print(f"Saved source audit to {args.output_json}")
    print("\nPer-family parsed distribution:")
    for family, row in per_family.items():
        mean = "NA" if row["score_mean"] is None else f"{row['score_mean']:.3f}"
        zero = "NA" if row["zero_pct"] is None else f"{row['zero_pct'] * 100:.1f}%"
        print(
            f"- {family}: records={row['records']} mean={mean} zero={zero} "
            f"high>=3={row['high_ge_3']} files={dict(row['source_files'])}"
        )

    print("\nPer-file distribution:")
    for name, row in per_file.items():
        mean = "NA" if row["score_mean"] is None else f"{row['score_mean']:.3f}"
        zero = "NA" if row["zero_pct"] is None else f"{row['zero_pct'] * 100:.1f}%"
        print(
            f"- {name}: built={row['built_records']}/{row['raw_rows']} mean={mean} "
            f"zero={zero} topics={row['topic_count']} families={row['model_families_after_parse']}"
        )


if __name__ == "__main__":
    main()
