import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


REJECT_PATTERNS = [
    r"\breferee\b",
    r"esteemed judges",
    r"fellow debaters",
    r"ladies and gentlemen",
    r"\bmy opponent\b",
    r"opposing side",
    r"opening statement",
    r"\brebuttal\b",
    r"cross-examination",
    r"\bi'?m llama\b",
    r"for the proposition",
    r"for the opposition",
]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
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


class LengthMeter:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def render(self, messages, add_generation_prompt=False):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def token_count(self, text):
        return len(self.tokenizer(text, truncation=False, return_tensors=None)["input_ids"])

    def prompt_tokens(self, row):
        return self.token_count(self.render(row["prompt"], add_generation_prompt=True))

    def full_tokens(self, row):
        return self.token_count(self.render(row["prompt"] + row.get("completion", []), add_generation_prompt=False))

    def completion_tokens(self, row):
        return self.token_count(self.render(row.get("completion", []), add_generation_prompt=False))


def completion_text(row):
    completion = row.get("completion") or []
    if not completion:
        return ""
    return completion[0].get("content", "") or ""


def is_llama(row):
    return row.get("source_model") == "llama" or row.get("model_family") == "llama"


def annotate_lengths(row, meter):
    row = dict(row)
    text = completion_text(row)
    row["clean_filter"] = {
        "completion_chars": len(text),
        "completion_tokens": meter.completion_tokens(row) if row.get("completion") else 0,
        "prompt_tokens": meter.prompt_tokens(row),
        "full_tokens": meter.full_tokens(row) if row.get("completion") else None,
    }
    return row


def filter_risk_rows(rows, meter, max_length):
    kept = []
    rejected = Counter()
    for row in rows:
        if not is_llama(row):
            rejected["non_llama"] += 1
            continue
        row = annotate_lengths(row, meter)
        if row["clean_filter"]["full_tokens"] > max_length:
            rejected["full_context_truncated"] += 1
            continue
        kept.append(row)
    return kept, rejected


def filter_rl_rows(rows, meter, max_prompt_length):
    kept = []
    rejected = Counter()
    for row in rows:
        if not is_llama(row):
            rejected["non_llama"] += 1
            continue
        row = annotate_lengths(row, meter)
        if row["clean_filter"]["prompt_tokens"] > max_prompt_length:
            rejected["prompt_truncated"] += 1
            continue
        kept.append(row)
    return kept, rejected


def filter_sft_rows(rows, meter, args, reject_regex):
    kept = []
    rejected = Counter()
    reject_examples = defaultdict(list)
    for row in rows:
        if not is_llama(row):
            rejected["non_llama"] += 1
            continue
        total_score = int(row.get("total_score", -1))
        if total_score != 0:
            rejected["nonzero_total_score"] += 1
            continue

        row = annotate_lengths(row, meter)
        text = completion_text(row)
        stats = row["clean_filter"]
        checks = [
            ("completion_too_short", len(text) < args.min_completion_chars),
            ("completion_too_long_chars", len(text) > args.max_completion_chars),
            ("completion_too_long_tokens", stats["completion_tokens"] > args.max_completion_tokens),
            ("full_context_truncated", stats["full_tokens"] > args.max_length),
        ]
        if not args.allow_debate_template:
            checks.append(("debate_template", bool(reject_regex.search(text[: args.template_scan_chars]))))
        failed = [name for name, did_fail in checks if did_fail]
        if failed:
            for name in failed:
                rejected[name] += 1
                if len(reject_examples[name]) < 5:
                    reject_examples[name].append(
                        {
                            "topic_id": row.get("topic_id"),
                            "round": row.get("round"),
                            "variant_label": row.get("variant_label"),
                            "completion_head": text[:180],
                            "clean_filter": stats,
                        }
                    )
            continue
        kept.append(row)
    return kept, rejected, reject_examples


def summarize_rows(rows):
    return {
        "rows": len(rows),
        "variant_counts": dict(Counter(row.get("variant_label") for row in rows)),
        "dimension_counts": dict(Counter(row.get("primary_dimension") for row in rows)),
        "score_counts": dict(Counter(row.get("total_score") for row in rows if row.get("total_score") is not None)),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a Llama-only clean SFT bucket from FIRST_EVAL mixed data.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-chars", type=int, default=2200)
    parser.add_argument("--min-completion-chars", type=int, default=80)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--template-scan-chars", type=int, default=900)
    parser.add_argument("--allow-debate-template", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    meter = LengthMeter(args.model_name_or_path)
    reject_regex = re.compile("|".join(REJECT_PATTERNS), re.IGNORECASE)

    summary = {
        "corpus_name": output_dir.name,
        "source_dir": str(source_dir),
        "policy": {
            "model_filter": "source_model/model_family == llama",
            "risk_head": f"Llama rows with full prompt+completion tokens <= {args.max_length}",
            "sft": (
                "Llama rows with total_score == 0, short completion, no debate-template phrase, "
                f"and full prompt+completion tokens <= {args.max_length}"
            ),
            "rl": f"Llama prompts with prompt tokens <= {args.max_prompt_length}",
        },
        "thresholds": {
            "sft_total_score_eq": 0,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_chars": args.max_completion_chars,
            "min_completion_chars": args.min_completion_chars,
            "max_completion_tokens": args.max_completion_tokens,
            "template_scan_chars": args.template_scan_chars,
            "allow_debate_template": args.allow_debate_template,
            "reject_patterns": REJECT_PATTERNS,
        },
        "splits": {},
    }

    for split in ["train", "dev"]:
        risk_rows, risk_rejected = filter_risk_rows(read_jsonl(source_dir / f"risk_head_{split}.jsonl"), meter, args.max_length)
        sft_rows, sft_rejected, sft_reject_examples = filter_sft_rows(read_jsonl(source_dir / f"sft_{split}.jsonl"), meter, args, reject_regex)
        rl_rows, rl_rejected = filter_rl_rows(read_jsonl(source_dir / f"rl_{split}.jsonl"), meter, args.max_prompt_length)
        negative_rows = [row for row in risk_rows if int(row.get("total_score", -1)) >= 3]

        write_jsonl(output_dir / f"risk_head_{split}.jsonl", risk_rows)
        write_jsonl(output_dir / f"eval_{split}.jsonl", risk_rows)
        write_jsonl(output_dir / f"sft_{split}.jsonl", sft_rows)
        write_jsonl(output_dir / f"negative_{split}.jsonl", negative_rows)
        write_jsonl(output_dir / f"rl_{split}.jsonl", rl_rows)

        summary["splits"][split] = {
            "risk_head": summarize_rows(risk_rows),
            "sft": summarize_rows(sft_rows),
            "negative": summarize_rows(negative_rows),
            "rl": summarize_rows(rl_rows),
            "rejected": {
                "risk_head": dict(risk_rejected),
                "sft": dict(sft_rejected),
                "rl": dict(rl_rejected),
            },
            "sft_reject_examples": {key: value for key, value in sft_reject_examples.items()},
        }

    dump_json(output_dir / "corpus_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
