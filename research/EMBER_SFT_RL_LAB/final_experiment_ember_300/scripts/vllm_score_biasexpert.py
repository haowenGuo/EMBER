import argparse
import asyncio
import copy
import time
from pathlib import Path

from openai import AsyncOpenAI

from vllm_common import (
    append_jsonl,
    bias_labels_from_report,
    dump_json,
    ensure_dir,
    extract_json_payload,
    load_bias_expert_prompt,
    read_jsonl,
    write_analysis,
    write_jsonl,
)


class BiasEndpointPool:
    def __init__(self, endpoints, model_name, max_concurrency=24, max_tokens=5000, request_timeout=240):
        self.clients = [
            AsyncOpenAI(base_url=endpoint.rstrip("/"), api_key="EMPTY", timeout=request_timeout)
            for endpoint in endpoints
        ]
        self.semaphores = [asyncio.Semaphore(max_concurrency) for _ in endpoints]
        self.model_name = model_name
        self.max_tokens = max_tokens

    async def evaluate(self, endpoint_index, prompt):
        client = self.clients[endpoint_index]
        semaphore = self.semaphores[endpoint_index]
        messages = [{"role": "user", "content": prompt}]
        async with semaphore:
            for attempt in range(1, 4):
                try:
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.3,
                        top_p=0.95,
                        max_tokens=self.max_tokens,
                    )
                    return (response.choices[0].message.content or "").strip()
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 * attempt)


def load_completed(path):
    completed = {}
    for row in read_jsonl(path) or []:
        record_id = (row.get("meta") or {}).get("record_id")
        if record_id:
            completed[record_id] = row
    return completed


async def score_row(row, endpoint_index, pool, bias_prompt, output_path, write_lock, completed):
    record_id = (row.get("meta") or {}).get("record_id")
    if record_id in completed:
        return
    raw_text = await pool.evaluate(endpoint_index, bias_prompt + (row.get("target_response") or ""))
    payload, parser_mode = extract_json_payload(raw_text)
    if payload is None:
        bias_report = {"error": "JSON_PARSE_FAILED", "raw": raw_text, "parser_mode": parser_mode}
    else:
        bias_report = payload
        bias_report["raw"] = raw_text
        bias_report["parser_mode"] = parser_mode
    bias_info = bias_labels_from_report(bias_report)
    scored = copy.deepcopy(row)
    scored["bias_report"] = bias_report
    scored["bias_labels"] = bias_info["labels"] if bias_info else None
    scored["total_bias_score"] = bias_info["total_score"] if bias_info else None
    async with write_lock:
        append_jsonl(output_path, scored)
        completed[record_id] = scored


async def run_scoring(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    generated_rows = list(read_jsonl(args.generated_jsonl) or [])
    output_path = output_dir / "scored_rounds.jsonl"
    completed = load_completed(output_path)
    bias_prompt = load_bias_expert_prompt(args.bias_prompt_source)
    pool = BiasEndpointPool(
        args.bias_endpoints,
        args.bias_model,
        max_concurrency=args.endpoint_concurrency,
        max_tokens=args.bias_max_tokens,
        request_timeout=args.request_timeout,
    )
    dump_json(
        output_dir / "scoring_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_jsonl": args.generated_jsonl,
            "bias_prompt_source": args.bias_prompt_source,
            "bias_model": args.bias_model,
            "bias_endpoints": args.bias_endpoints,
            "rows": len(generated_rows),
            "architecture": "Offline vLLM BiasExpert scoring after debate generation is complete.",
        },
    )
    shards = [[] for _ in args.bias_endpoints]
    for index, row in enumerate(generated_rows):
        shards[index % len(shards)].append(row)
    write_lock = asyncio.Lock()

    async def run_shard(endpoint_index, rows):
        semaphore = asyncio.Semaphore(args.rows_per_endpoint)

        async def guarded(row):
            async with semaphore:
                await score_row(row, endpoint_index, pool, bias_prompt, output_path, write_lock, completed)

        await asyncio.gather(*(guarded(row) for row in rows))

    await asyncio.gather(*(run_shard(index, rows) for index, rows in enumerate(shards)))
    consolidated = {((row.get("meta") or {}).get("record_id")): row for row in read_jsonl(output_path) or []}
    ordered = [consolidated[key] for key in sorted(consolidated)]
    write_jsonl(output_path, ordered)
    write_analysis(ordered, output_dir / "analysis")


def parse_args():
    parser = argparse.ArgumentParser(description="Score generated EMBER debate rounds with vLLM BiasExpert endpoints.")
    parser.add_argument("--generated-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bias-endpoints", required=True, help="Comma-separated OpenAI-compatible base URLs.")
    parser.add_argument("--bias-model", default="biasexpert")
    parser.add_argument("--bias-prompt-source", required=True)
    parser.add_argument("--endpoint-concurrency", type=int, default=24)
    parser.add_argument("--rows-per-endpoint", type=int, default=48)
    parser.add_argument("--bias-max-tokens", type=int, default=5000)
    parser.add_argument("--request-timeout", type=int, default=240)
    args = parser.parse_args()
    args.bias_endpoints = [item.strip().rstrip("/") for item in args.bias_endpoints.split(",") if item.strip()]
    if not args.bias_endpoints:
        raise ValueError("--bias-endpoints must contain at least one endpoint.")
    return args


def main():
    asyncio.run(run_scoring(parse_args()))


if __name__ == "__main__":
    main()
