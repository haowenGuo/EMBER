import argparse
import asyncio
import copy
import csv
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from vllm_common import (
    append_jsonl,
    bias_labels_from_report,
    dump_json,
    ensure_dir,
    extract_json_payload,
    load_bias_expert_prompt,
    write_analysis,
    write_jsonl,
)


VARIANT_ORDER = {"BASE": 0, "EMBER-PROMPT": 1, "EMBER-AGENT": 2, "SFT": 3, "SFT+RL": 4}


def mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fmt3(value):
    return "" if value is None else f"{value:.3f}"


def write_csv(path, fieldnames, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class OpenAIBiasEvaluator:
    def __init__(
        self,
        model,
        api_key,
        base_url=None,
        api_mode="responses",
        max_output_tokens=5000,
        temperature=0.3,
        top_p=0.95,
        request_timeout=240,
        max_retries=3,
        retry_sleep_seconds=3.0,
    ):
        self.client = None
        if api_mode in {"responses", "chat", "auto"}:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.api_mode = api_mode
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self._responses_without_sampling = False
        self._chat_uses_max_completion_tokens = False

    async def evaluate(self, prompt):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.api_mode == "chat":
                    return await self._chat_completion(prompt)
                if self.api_mode == "anthropic_messages":
                    return await self._anthropic_messages(prompt)
                if self.api_mode == "auto":
                    try:
                        return await self._responses(prompt)
                    except AttributeError:
                        return await self._chat_completion(prompt)
                return await self._responses(prompt)
            except Exception as exc:
                last_error = exc
                if self._should_retry_without_sampling(exc):
                    self._responses_without_sampling = True
                    continue
                if self._should_retry_chat_max_completion_tokens(exc):
                    self._chat_uses_max_completion_tokens = True
                    continue
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_sleep_seconds * attempt)
        raise RuntimeError(f"OpenAI bias evaluation failed after {self.max_retries} attempts: {last_error}")

    def _should_retry_without_sampling(self, exc):
        message = str(exc).lower()
        return (
            not self._responses_without_sampling
            and self.api_mode in {"responses", "auto"}
            and ("temperature" in message or "top_p" in message)
        )

    def _should_retry_chat_max_completion_tokens(self, exc):
        message = str(exc).lower()
        return (
            not self._chat_uses_max_completion_tokens
            and self.api_mode == "chat"
            and "max_tokens" in message
            and "max_completion_tokens" in message
        )

    async def _responses(self, prompt):
        params = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": self.max_output_tokens,
        }
        if not self._responses_without_sampling:
            params["temperature"] = self.temperature
            params["top_p"] = self.top_p
        response = await self.client.responses.create(**params)
        return extract_response_text(response)

    async def _chat_completion(self, prompt):
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self._chat_uses_max_completion_tokens:
            params["max_completion_tokens"] = self.max_output_tokens
        else:
            params["max_tokens"] = self.max_output_tokens
        response = await self.client.chat.completions.create(**params)
        return (response.choices[0].message.content or "").strip()

    async def _anthropic_messages(self, prompt):
        if not self.base_url:
            raise RuntimeError("--base-url is required for api-mode=anthropic_messages")
        url = normalize_messages_url(self.base_url)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = response.text[:1000]
                raise RuntimeError(f"anthropic_messages HTTP {response.status_code}: {body}") from exc
            return extract_anthropic_messages_text(response.json())


def obj_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_response_text(response):
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = obj_get(response, "output", []) or []
    for item in output_items:
        direct_text = obj_get(item, "text")
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()
        for content in obj_get(item, "content", []) or []:
            text = obj_get(content, "text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    raise ValueError("OpenAI response contained no text content")


def normalize_messages_url(base_url):
    url = base_url.rstrip("/")
    if url.endswith("/messages"):
        return url
    if url.endswith("/v1"):
        return f"{url}/messages"
    return f"{url}/v1/messages"


def extract_anthropic_messages_text(payload):
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()
    content = payload.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(part.strip() for part in parts if part.strip()).strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") or {}
        text = message.get("content")
        if isinstance(text, str) and text.strip():
            return text.strip()
    raise ValueError("Anthropic-compatible message response contained no text content")


def load_legacy_config(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Legacy config not found: {path}")
    spec = importlib.util.spec_from_file_location("ember_openai_legacy_config", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_completed(path):
    completed = {}
    for row in read_jsonl(path) or []:
        record_id = (row.get("meta") or {}).get("record_id")
        if record_id:
            completed[record_id] = row
    return completed


def row_matches_filters(row, args):
    meta = row.get("meta") or {}
    if args.variants:
        variant = meta.get("variant_label")
        if variant not in args.variants:
            return False
    topic_index = meta.get("topic_index_1based")
    if topic_index is not None:
        try:
            topic_index = int(topic_index)
        except Exception:
            topic_index = None
    if args.topic_start is not None and topic_index is not None and topic_index < args.topic_start:
        return False
    if args.topic_end is not None and topic_index is not None and topic_index > args.topic_end:
        return False
    return True


async def score_row(row, evaluator, bias_prompt, output_path, write_lock, completed, args):
    meta = row.get("meta") or {}
    record_id = meta.get("record_id")
    if record_id in completed:
        return

    response_text = row.get("target_response") or ""
    raw_text = await evaluator.evaluate(bias_prompt + response_text)
    payload, parser_mode = extract_json_payload(raw_text)
    if payload is None:
        bias_report = {"error": "JSON_PARSE_FAILED", "raw": raw_text, "parser_mode": parser_mode}
    else:
        bias_report = payload
        bias_report["raw"] = raw_text
        bias_report["parser_mode"] = parser_mode

    bias_info = bias_labels_from_report(bias_report)
    scored = copy.deepcopy(row)
    if "bias_report" in scored:
        scored["qwen_bias_report"] = scored.get("bias_report")
        scored["qwen_bias_labels"] = scored.get("bias_labels")
        scored["qwen_total_bias_score"] = scored.get("total_bias_score")
    scored["bias_report"] = bias_report
    scored["bias_labels"] = bias_info["labels"] if bias_info else None
    scored["total_bias_score"] = bias_info["total_score"] if bias_info else None
    scored["openai_bias_evaluator"] = {
        "model": args.model,
        "api_mode": args.api_mode,
        "model_label": args.model_label,
        "prompt_source": args.bias_prompt_source,
        "prompt_equivalence": "BIAS_EXPERT_PROMPT concatenated directly with target_response",
    }

    async with write_lock:
        append_jsonl(output_path, scored)
        if record_id:
            completed[record_id] = scored


def write_comparison(scored_rows, analysis_dir):
    groups_variant = defaultdict(list)
    groups_round = defaultdict(list)
    row_level = []
    for row in scored_rows:
        meta = row.get("meta") or {}
        qwen_score = row.get("qwen_total_bias_score")
        openai_score = row.get("total_bias_score")
        delta = None
        if qwen_score is not None and openai_score is not None:
            delta = float(openai_score) - float(qwen_score)
        variant = meta.get("variant_label", "")
        round_index = meta.get("round")
        item = {
            "record_id": meta.get("record_id", ""),
            "variant_label": variant,
            "topic_id": meta.get("topic_id", ""),
            "round": round_index,
            "qwen_total_bias_score": qwen_score,
            "openai_total_bias_score": openai_score,
            "openai_minus_qwen": delta,
        }
        row_level.append(item)
        groups_variant[variant].append(item)
        groups_round[(variant, round_index)].append(item)

    def aggregate(rows):
        qwen_values = [row["qwen_total_bias_score"] for row in rows if row["qwen_total_bias_score"] is not None]
        openai_values = [row["openai_total_bias_score"] for row in rows if row["openai_total_bias_score"] is not None]
        deltas = [row["openai_minus_qwen"] for row in rows if row["openai_minus_qwen"] is not None]
        return {
            "total_rows": len(rows),
            "paired_rows": len(deltas),
            "mean_qwen_total_bias_score": fmt3(mean(qwen_values)),
            "mean_openai_total_bias_score": fmt3(mean(openai_values)),
            "mean_openai_minus_qwen": fmt3(mean(deltas)),
        }

    variant_rows = []
    for variant, rows in sorted(groups_variant.items(), key=lambda item: VARIANT_ORDER.get(item[0], 999)):
        agg = aggregate(rows)
        agg["variant_label"] = variant
        variant_rows.append(agg)

    round_rows = []
    for (variant, round_index), rows in sorted(
        groups_round.items(), key=lambda item: (VARIANT_ORDER.get(item[0][0], 999), item[0][1])
    ):
        agg = aggregate(rows)
        agg["variant_label"] = variant
        agg["round"] = round_index
        round_rows.append(agg)

    write_csv(
        Path(analysis_dir) / "openai_vs_qwen_by_variant.csv",
        [
            "variant_label",
            "total_rows",
            "paired_rows",
            "mean_qwen_total_bias_score",
            "mean_openai_total_bias_score",
            "mean_openai_minus_qwen",
        ],
        variant_rows,
    )
    write_csv(
        Path(analysis_dir) / "openai_vs_qwen_by_round.csv",
        [
            "variant_label",
            "round",
            "total_rows",
            "paired_rows",
            "mean_qwen_total_bias_score",
            "mean_openai_total_bias_score",
            "mean_openai_minus_qwen",
        ],
        round_rows,
    )
    write_csv(
        Path(analysis_dir) / "openai_vs_qwen_rows.csv",
        [
            "record_id",
            "variant_label",
            "topic_id",
            "round",
            "qwen_total_bias_score",
            "openai_total_bias_score",
            "openai_minus_qwen",
        ],
        row_level,
    )


async def run_scoring(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    output_path = output_dir / "scored_rounds.jsonl"
    completed = load_completed(output_path)
    rows = [row for row in (read_jsonl(args.input_jsonl) or []) if row_matches_filters(row, args)]
    if args.limit is not None:
        rows = rows[: args.limit]

    legacy_config = load_legacy_config(args.legacy_config)
    api_key = os.environ.get(args.api_key_env) or getattr(legacy_config, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} or OPENAI_API_KEY in --legacy-config is required.")
    args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or getattr(
        legacy_config, "OPENAI_BASE_URL", None
    )
    args.model = args.model or os.environ.get("OPENAI_EVAL_MODEL") or getattr(
        legacy_config, "openai_model_name", None
    ) or "gpt-5.5"

    bias_prompt = load_bias_expert_prompt(args.bias_prompt_source)
    evaluator = OpenAIBiasEvaluator(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        api_mode=args.api_mode,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    dump_json(
        output_dir / "scoring_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_jsonl": args.input_jsonl,
            "output_jsonl": str(output_path),
            "model": args.model,
            "model_label": args.model_label,
            "api_mode": args.api_mode,
            "base_url_present": bool(args.base_url),
            "legacy_config_present": bool(args.legacy_config),
            "bias_prompt_source": args.bias_prompt_source,
            "prompt_equivalence": "exact BiasExpert prompt loaded from BIAS_EXPERT_PROMPT and concatenated with target_response",
            "rows_after_filters": len(rows),
            "concurrency": args.concurrency,
            "limit": args.limit,
            "variants": sorted(args.variants) if args.variants else None,
            "topic_start": args.topic_start,
            "topic_end": args.topic_end,
        },
    )

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(row):
        async with semaphore:
            await score_row(row, evaluator, bias_prompt, output_path, write_lock, completed, args)

    await asyncio.gather(*(guarded(row) for row in rows))

    consolidated = {((row.get("meta") or {}).get("record_id")): row for row in read_jsonl(output_path) or []}
    ordered = []
    for row in rows:
        record_id = (row.get("meta") or {}).get("record_id")
        if record_id in consolidated:
            ordered.append(consolidated[record_id])
    write_jsonl(output_path, ordered)
    write_analysis(ordered, output_dir / "analysis")
    write_comparison(ordered, output_dir / "analysis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score EMBER generated/scored rounds with an OpenAI model using the exact BiasExpert prompt."
    )
    parser.add_argument("--input-jsonl", required=True, help="Existing generated_rounds.jsonl or scored_rounds.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bias-prompt-source", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-label", default="gpt55")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--legacy-config", default=os.environ.get("OPENAI_LEGACY_CONFIG"))
    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat", "auto", "anthropic_messages"],
        default="responses",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--topic-start", type=int)
    parser.add_argument("--topic-end", type=int)
    parser.add_argument("--variants", default="")
    args = parser.parse_args()
    args.variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    return args


def main():
    asyncio.run(run_scoring(parse_args()))


if __name__ == "__main__":
    main()
