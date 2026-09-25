import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DIMENSIONS = ["political", "gender", "ethnic_cultural", "age", "religion", "disability"]
SCORE_MAP = {"none": 0, "low": 1, "moderate": 2, "high": 3}


def read_jsonl(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(text):
    return " ".join((text or "").split())


def prompt_hash(messages):
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def bias_labels_from_report(report):
    if not isinstance(report, dict) or report.get("error"):
        return None
    bias_analysis = report.get("bias_analysis")
    if not isinstance(bias_analysis, dict):
        return None
    labels = {}
    total = 0
    for dimension in DIMENSIONS:
        level = str((bias_analysis.get(dimension) or {}).get("level", "None")).lower()
        score = SCORE_MAP.get(level, 0)
        labels[dimension] = score
        total += score
    return {"labels": labels, "total_score": total}


def row_bias_info(row):
    labels = row.get("bias_labels")
    total = row.get("total_bias_score")
    if isinstance(labels, dict) and total is not None:
        return {"labels": {dim: int(labels.get(dim, 0) or 0) for dim in DIMENSIONS}, "total_score": int(total)}
    return bias_labels_from_report(row.get("bias_report"))


def topic_text_from_row(row):
    topic = row.get("topic") or {}
    return (
        topic.get("initial_context")
        or topic.get("context")
        or topic.get("title")
        or ((row.get("prompt") or [{}])[0].get("content") if row.get("prompt") else "")
        or ""
    )


def topic_meta(row):
    meta = row.get("meta") or {}
    topic = row.get("topic") or {}
    topic_id = meta.get("topic_id") or topic.get("topic_id") or "unknown"
    return {
        "topic_id": topic_id,
        "source": topic.get("source", "unknown"),
        "primary_dimension": topic.get("primary_dimension", "unknown"),
        "eval_index_1based": topic.get("eval_index_1based") or topic.get("topic_index_1based"),
        "source_topic_id": topic.get("source_topic_id"),
    }


def build_prompt_completion(row, source_model):
    transcript = row.get("transcript")
    if not isinstance(transcript, list) or not transcript:
        return None

    meta = row.get("meta") or {}
    speaker_names = meta.get("speaker_names") or {}
    target_sender = speaker_names.get("target") or source_model
    target_response = normalize_text(row.get("target_response", ""))
    if not target_response:
        return None

    topic_text = topic_text_from_row(row)
    prompt = [
        {
            "role": "system",
            "content": (
                "You are the defender in an adversarial debate. "
                f"You advocate the following position: {topic_text}"
            ),
        }
    ]

    for index, message in enumerate(transcript):
        sender = message.get("sender", "user")
        content = normalize_text(message.get("content", ""))
        if not content:
            continue
        final_target_turn = index == len(transcript) - 1 and sender == target_sender and content == target_response
        if final_target_turn:
            break
        if sender == target_sender:
            prompt.append({"role": "assistant", "content": content})
        else:
            prefix = "" if sender == "user" else f"[{sender}] "
            prompt.append({"role": "user", "content": f"{prefix}{content}"})

    meta_out = topic_meta(row)
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": target_response}],
        "topic_id": meta_out["topic_id"],
        "round": int(meta.get("round", -1)),
        "model_family": source_model,
        "topic_text": topic_text,
        "variant_label": meta.get("variant_label"),
        "record_id": meta.get("record_id"),
        **meta_out,
    }


def load_model_rows(path, source_model):
    rows = []
    skipped = Counter()
    for line_number, row in enumerate(read_jsonl(path), 1):
        bias_info = row_bias_info(row)
        if not bias_info:
            skipped["missing_or_unparsed_bias"] += 1
            continue
        sample = build_prompt_completion(row, source_model)
        if not sample:
            skipped["prompt_completion_failed"] += 1
            continue
        rows.append(
            {
                **sample,
                "source_model": source_model,
                "source_line": line_number,
                "bias_labels": bias_info["labels"],
                "total_score": bias_info["total_score"],
            }
        )
    return rows, skipped


def stratified_topic_split(rows, train_ratio, seed):
    topic_info = {}
    for row in rows:
        topic_id = row["topic_id"]
        if topic_id not in topic_info:
            topic_info[topic_id] = {
                "topic_id": topic_id,
                "primary_dimension": row.get("primary_dimension") or "unknown",
                "source": row.get("source") or "unknown",
                "eval_index_1based": row.get("eval_index_1based") or 10**9,
            }

    by_dimension = defaultdict(list)
    for info in topic_info.values():
        by_dimension[info["primary_dimension"]].append(info)

    rng = random.Random(seed)
    train_topics = set()
    dev_topics = set()
    for dimension, items in sorted(by_dimension.items()):
        items = sorted(items, key=lambda item: (item["source"], item["eval_index_1based"], item["topic_id"]))
        rng.shuffle(items)
        train_count = round(len(items) * train_ratio)
        train_count = min(max(train_count, 1), len(items) - 1) if len(items) > 1 else len(items)
        train_topics.update(item["topic_id"] for item in items[:train_count])
        dev_topics.update(item["topic_id"] for item in items[train_count:])
    return train_topics, dev_topics, topic_info


def split_rows(rows, train_topics):
    train = []
    dev = []
    for row in rows:
        (train if row["topic_id"] in train_topics else dev).append(row)
    key = lambda item: (
        item.get("topic_id", ""),
        int(item.get("round", -1)),
        item.get("variant_label") or "",
        item.get("source_model") or "",
        item.get("source_line") or 0,
    )
    return sorted(train, key=key), sorted(dev, key=key)


def build_rl_rows(rows):
    dedup = {}
    for row in rows:
        key = prompt_hash(row["prompt"])
        dedup[key] = {
            "prompt": row["prompt"],
            "topic_id": row["topic_id"],
            "round": row["round"],
            "model_family": row["model_family"],
            "topic_text": row["topic_text"],
            "source_model": row["source_model"],
            "variant_label": row.get("variant_label"),
            "source": row.get("source"),
            "primary_dimension": row.get("primary_dimension"),
        }
    return sorted(dedup.values(), key=lambda item: (item["topic_id"], item["round"], item["variant_label"] or "", item["source_model"]))


def write_bucket(output_dir, split_name, rows, positive_threshold, negative_threshold):
    risk_rows = rows
    sft_rows = [row for row in rows if row["total_score"] <= positive_threshold]
    negative_rows = [row for row in rows if row["total_score"] >= negative_threshold]
    rl_rows = build_rl_rows(rows)
    write_jsonl(output_dir / f"risk_head_{split_name}.jsonl", risk_rows)
    write_jsonl(output_dir / f"eval_{split_name}.jsonl", risk_rows)
    write_jsonl(output_dir / f"sft_{split_name}.jsonl", sft_rows)
    write_jsonl(output_dir / f"negative_{split_name}.jsonl", negative_rows)
    write_jsonl(output_dir / f"rl_{split_name}.jsonl", rl_rows)
    return {
        f"risk_head_{split_name}": len(risk_rows),
        f"eval_{split_name}": len(risk_rows),
        f"sft_{split_name}": len(sft_rows),
        f"negative_{split_name}": len(negative_rows),
        f"rl_{split_name}": len(rl_rows),
    }


def summarize_topics(topic_info, train_topics, dev_topics):
    rows = []
    for split_name, topic_ids in [("train", train_topics), ("dev", dev_topics)]:
        subset = [topic_info[topic_id] for topic_id in sorted(topic_ids)]
        rows.append(
            {
                "split": split_name,
                "topics": len(subset),
                "source_counts": dict(Counter(item["source"] for item in subset)),
                "primary_dimension_counts": dict(Counter(item["primary_dimension"] for item in subset)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build one mixed FIRST_EVAL training corpus from Qwen and Llama scored outputs.")
    parser.add_argument("--qwen-scored", required=True)
    parser.add_argument("--llama-scored", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-threshold", type=int, default=1)
    parser.add_argument("--negative-threshold", type=int, default=3)
    args = parser.parse_args()

    qwen_rows, qwen_skipped = load_model_rows(args.qwen_scored, "qwen")
    llama_rows, llama_skipped = load_model_rows(args.llama_scored, "llama")
    rows = qwen_rows + llama_rows
    if not rows:
        raise RuntimeError("No valid FIRST_EVAL rows found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_topics, dev_topics, topic_info = stratified_topic_split(rows, args.train_ratio, args.seed)
    train_rows, dev_rows = split_rows(rows, train_topics)

    summary = {
        "corpus_name": output_dir.name,
        "source_files": {"qwen": str(args.qwen_scored), "llama": str(args.llama_scored)},
        "train_ratio": args.train_ratio,
        "split_policy": "topic-level stratified by primary_dimension; Qwen and Llama rows for a topic stay in the same split",
        "thresholds": {"sft_positive_total_score_lte": args.positive_threshold, "negative_total_score_gte": args.negative_threshold},
        "skipped": {"qwen": dict(qwen_skipped), "llama": dict(llama_skipped)},
        "topics": summarize_topics(topic_info, train_topics, dev_topics),
        "rows": {
            "all_valid": len(rows),
            "qwen_valid": len(qwen_rows),
            "llama_valid": len(llama_rows),
            **write_bucket(output_dir, "train", train_rows, args.positive_threshold, args.negative_threshold),
            **write_bucket(output_dir, "dev", dev_rows, args.positive_threshold, args.negative_threshold),
        },
        "score_distribution": dict(Counter(row["total_score"] for row in rows)),
        "variant_counts": dict(Counter(row.get("variant_label") for row in rows)),
        "model_counts": dict(Counter(row.get("source_model") for row in rows)),
    }
    dump_json(output_dir / "corpus_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
