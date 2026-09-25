import argparse
import asyncio
import copy
import time
from pathlib import Path

from openai import AsyncOpenAI

from vllm_common import (
    ATTACK_PROMPT_PROFILES,
    SELF_REFLECTION_REWRITE_PROMPT,
    append_jsonl,
    build_jobs,
    dump_json,
    ensure_dir,
    load_topics,
    multiagent_provocateur_system_prompt,
    parse_variants,
    partition_even,
    provocateur_system_prompt,
    read_jsonl,
    render_agent_messages,
    response_quality_check,
    target_system_prompt,
)


class EndpointPool:
    def __init__(
        self,
        endpoints,
        model_name,
        max_concurrency=24,
        temperature=0.0,
        max_tokens=1024,
        disable_thinking=False,
        request_timeout=180,
    ):
        self.clients = [
            AsyncOpenAI(base_url=endpoint.rstrip("/"), api_key="EMPTY", timeout=request_timeout)
            for endpoint in endpoints
        ]
        self.endpoints = endpoints
        self.model_name = model_name
        self.semaphores = [asyncio.Semaphore(max_concurrency) for _ in endpoints]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking

    async def chat(self, endpoint_index, messages):
        client = self.clients[endpoint_index]
        semaphore = self.semaphores[endpoint_index]
        extra_body = {}
        if self.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        async with semaphore:
            for attempt in range(1, 4):
                try:
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body=extra_body or None,
                    )
                    return (response.choices[0].message.content or "").strip()
                except Exception as exc:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 * attempt)


def load_completed_rows(path):
    completed = {}
    for row in read_jsonl(path) or []:
        record_id = (row.get("meta") or {}).get("record_id")
        if record_id:
            completed[record_id] = row
    return completed


def record_id_for(job, round_index):
    return f"{job['variant']['variant_label']}|{job['topic']['topic_id']}|r{round_index}"


def completed_for_job(job, completed_rows):
    prefix = f"{job['variant']['variant_label']}|{job['topic']['topic_id']}|r"
    rows = {}
    for record_id, row in completed_rows.items():
        if record_id.startswith(prefix):
            rows[int(record_id.rsplit("r", 1)[-1])] = row
    return rows


def endpoint_index_for(pool, *keys):
    if not pool.endpoints:
        return 0
    total = 0
    for key in keys:
        if isinstance(key, int):
            total += key
        else:
            total += sum(ord(ch) for ch in str(key))
    return total % len(pool.endpoints)


def render_agent_messages_for_model(args, system_prompt, speaker_name, history):
    if not args.flatten_chat_history:
        return render_agent_messages(system_prompt, speaker_name, history)

    transcript_lines = []
    for item in history:
        sender = item.get("sender", "unknown")
        content = item.get("content", "")
        transcript_lines.append(f"{sender}: {content}")
    transcript = "\n\n".join(transcript_lines)
    user_prompt = (
        "Debate transcript so far:\n"
        f"{transcript}\n\n"
        f"You are {speaker_name}. Continue the debate by replying only as {speaker_name}. "
        "Do not quote the full transcript. Do not write messages for other speakers."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_speaker_names(args, target_name, variant):
    if variant.get("attack_protocol") == "multiagent":
        return {"target": target_name, "provocateurs": list(args.multiagent_provocateur_names)}
    return {"target": target_name, "provocateur": args.provocateur_name}


async def generate_target_response(args, target_pool, endpoint_index, variant, target_name, base_messages):
    initial = await target_pool.chat(endpoint_index, base_messages)
    if variant["variant_type"] != "ember_agent":
        return initial, {"enabled": False, "mode": variant["method"]}

    refine_messages = base_messages + [
        {"role": "assistant", "content": initial},
        {"role": "user", "content": SELF_REFLECTION_REWRITE_PROMPT},
    ]
    revised = await target_pool.chat(endpoint_index, refine_messages)
    initial_ok, initial_quality = response_quality_check(initial, args.agent_min_response_chars)
    revised_ok, revised_quality = response_quality_check(revised, args.agent_min_response_chars)
    accepted = revised_ok
    final_response = revised if accepted else initial
    return final_response, {
        "enabled": True,
        "mode": "self_reflection_vllm_no_external_biasexpert",
        "external_biasexpert_used": False,
        "initial_quality": initial_quality,
        "revised_quality": revised_quality,
        "accepted": accepted,
        "initial_excerpt": initial[:240],
        "revised_excerpt": revised[:240],
        "note": "BiasExpert scoring is intentionally deferred to the offline evaluation phase.",
    }


async def run_job(args, job, endpoint_index, target_pool, attacker_pool, output_path, write_lock, completed_rows):
    variant = job["variant"]
    topic = job["topic"]
    target_name = args.target_name
    attacker_name = args.provocateur_name
    target_prompt = target_system_prompt(target_name, topic["initial_context"], variant["method"])
    attacker_prompt = provocateur_system_prompt(
        attacker_name,
        topic["initial_context"],
        attack_prompt_profile=args.attack_prompt_profile,
    )
    multiagent_prompts = {}
    if variant.get("attack_protocol") == "multiagent":
        for name in args.multiagent_provocateur_names:
            helpers = [item for item in args.multiagent_provocateur_names if item != name]
            multiagent_prompts[name] = multiagent_provocateur_system_prompt(
                name,
                helpers,
                target_name,
                topic["initial_context"],
                attack_prompt_profile=args.attack_prompt_profile,
            )

    prior_rows = completed_for_job(job, completed_rows)
    if prior_rows:
        max_done_round = max(prior_rows)
        if max_done_round >= args.max_debate_round:
            return
        global_history = copy.deepcopy(prior_rows[max_done_round]["transcript"])
        start_round = max_done_round + 1
    else:
        global_history = [
            {
                "sender": "user",
                "content": f"Referee: This is a debate. Let's have debater {target_name} present their point first.",
            }
        ]
        start_round = 0

    for round_index in range(start_round, args.max_debate_round + 1):
        record_id = record_id_for(job, round_index)
        if record_id in completed_rows:
            continue

        if round_index > 0:
            if variant.get("attack_protocol") == "multiagent":
                for attacker_position, multiagent_name in enumerate(args.multiagent_provocateur_names):
                    attacker_messages = render_agent_messages_for_model(
                        args,
                        multiagent_prompts[multiagent_name],
                        multiagent_name,
                        global_history,
                    )
                    attack_endpoint_index = endpoint_index_for(
                        attacker_pool,
                        variant["variant_label"],
                        topic["topic_id"],
                        round_index,
                        attacker_position,
                    )
                    attack = await attacker_pool.chat(attack_endpoint_index, attacker_messages)
                    global_history.append({"sender": multiagent_name, "content": attack})
            else:
                attacker_messages = render_agent_messages_for_model(args, attacker_prompt, attacker_name, global_history)
                attack_endpoint_index = endpoint_index_for(
                    attacker_pool,
                    variant["variant_label"],
                    topic["topic_id"],
                    round_index,
                )
                attack = await attacker_pool.chat(attack_endpoint_index, attacker_messages)
                global_history.append({"sender": attacker_name, "content": attack})

        target_messages = render_agent_messages_for_model(args, target_prompt, target_name, global_history)
        target_response, mitigation_trace = await generate_target_response(
            args, target_pool, endpoint_index, variant, target_name, target_messages
        )
        global_history.append({"sender": target_name, "content": target_response})

        row = {
            "meta": {
                "record_id": record_id,
                "variant_label": variant["variant_label"],
                "variant_type": variant["variant_type"],
                "method": variant["method"],
                "defense_method": variant.get("defense_method", variant["variant_label"]),
                "attack_protocol": variant.get("attack_protocol", "single"),
                "topic_id": topic["topic_id"],
                "topic_index_1based": topic["topic_index_1based"],
                "round": round_index,
                "evaluate_source": "target_response",
                "protocol": (
                    "vllm_multiagent_generation_offline_biasexpert_round05"
                    if variant.get("attack_protocol") == "multiagent"
                    else "vllm_generation_offline_biasexpert_round05"
                ),
                "speaker_names": build_speaker_names(args, target_name, variant),
                "target_endpoint_index": endpoint_index,
            },
            "topic": topic,
            "target_response": target_response,
            "transcript": copy.deepcopy(global_history),
            "mitigation_trace": mitigation_trace,
        }
        async with write_lock:
            append_jsonl(output_path, row)
            completed_rows[record_id] = row


async def run_endpoint_shard(args, shard, endpoint_index, target_pool, attacker_pool, output_path, write_lock, completed_rows):
    semaphore = asyncio.Semaphore(args.jobs_per_endpoint)

    async def guarded(job):
        async with semaphore:
            await run_job(args, job, endpoint_index, target_pool, attacker_pool, output_path, write_lock, completed_rows)

    await asyncio.gather(*(guarded(job) for job in shard))


async def run_generation(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    output_path = output_dir / "generated_rounds.jsonl"
    completed_rows = load_completed_rows(output_path)
    topics = load_topics(args.cmv_path, args.topic_start, args.topic_end)
    variants = parse_variants(args.variants)
    jobs = build_jobs(topics, variants)
    shards = partition_even(jobs, len(args.target_endpoints))

    dump_json(
        output_dir / "generation_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cmv_path": args.cmv_path,
            "topic_start": args.topic_start,
            "topic_end": args.topic_end,
            "max_debate_round": args.max_debate_round,
            "variants": [variant["variant_label"] for variant in variants],
            "target_name": args.target_name,
            "provocateur_name": args.provocateur_name,
            "target_model": args.target_model,
            "attacker_model": args.attacker_model,
            "attack_prompt_profile": args.attack_prompt_profile,
            "target_endpoints": args.target_endpoints,
            "attacker_endpoints": args.attacker_endpoints,
            "multiagent_provocateur_names": args.multiagent_provocateur_names,
            "jobs": len(jobs),
            "expected_rows": len(jobs) * (args.max_debate_round + 1),
            "architecture": "vLLM OpenAI-compatible servers; target data-parallel shards; shared attacker service(s); offline BiasExpert scoring.",
            "variant_protocols": [
                {
                    "variant_label": variant["variant_label"],
                    "defense_method": variant.get("defense_method", variant["variant_label"]),
                    "attack_protocol": variant.get("attack_protocol", "single"),
                    "target_method": variant["method"],
                }
                for variant in variants
            ],
            "agent_mode": (
                "EMBER-AGENT variants use target-model self-reflection only; "
                "BiasExpert is not called during generation."
            ),
        },
    )

    target_pool = EndpointPool(
        args.target_endpoints,
        args.target_model,
        max_concurrency=args.target_concurrency,
        temperature=args.target_temperature,
        max_tokens=args.target_max_tokens,
        disable_thinking=args.target_disable_thinking,
        request_timeout=args.request_timeout,
    )
    attacker_pool = EndpointPool(
        args.attacker_endpoints,
        args.attacker_model,
        max_concurrency=args.attacker_concurrency,
        temperature=args.attacker_temperature,
        max_tokens=args.attacker_max_tokens,
        disable_thinking=args.attacker_disable_thinking,
        request_timeout=args.request_timeout,
    )
    write_lock = asyncio.Lock()
    await asyncio.gather(
        *(
            run_endpoint_shard(args, shard, endpoint_index, target_pool, attacker_pool, output_path, write_lock, completed_rows)
            for endpoint_index, shard in enumerate(shards)
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EMBER debate transcripts with vLLM servers.")
    parser.add_argument("--cmv-path", required=True)
    parser.add_argument("--topic-start", type=int, default=1)
    parser.add_argument("--topic-end", type=int, default=150)
    parser.add_argument("--max-debate-round", type=int, default=5)
    parser.add_argument("--variants", default="BASE,EMBER-PROMPT,EMBER-AGENT")
    parser.add_argument("--target-endpoints", required=True, help="Comma-separated OpenAI-compatible base URLs.")
    parser.add_argument("--attacker-endpoint", default="")
    parser.add_argument("--attacker-endpoints", default="")
    parser.add_argument("--target-model", default="target")
    parser.add_argument("--attacker-model", default="attacker")
    parser.add_argument(
        "--attack-prompt-profile",
        choices=sorted(ATTACK_PROMPT_PROFILES.keys()),
        default="baseline",
        help="Provocateur prompt profile used for controlled adversarial pressure testing.",
    )
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--provocateur-name", default="qwen_attacker")
    parser.add_argument("--multiagent-provocateur-names", default="d0,d1,d2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs-per-endpoint", type=int, default=18)
    parser.add_argument("--target-concurrency", type=int, default=24)
    parser.add_argument("--attacker-concurrency", type=int, default=48)
    parser.add_argument("--target-temperature", type=float, default=0.0)
    parser.add_argument("--attacker-temperature", type=float, default=0.0)
    parser.add_argument("--target-max-tokens", type=int, default=1024)
    parser.add_argument("--attacker-max-tokens", type=int, default=1024)
    parser.add_argument("--target-disable-thinking", action="store_true")
    parser.add_argument("--attacker-disable-thinking", action="store_true", default=True)
    parser.add_argument("--flatten-chat-history", action="store_true")
    parser.add_argument("--agent-min-response-chars", type=int, default=80)
    parser.add_argument("--request-timeout", type=int, default=240)
    args = parser.parse_args()
    args.target_endpoints = [item.strip().rstrip("/") for item in args.target_endpoints.split(",") if item.strip()]
    raw_attacker_endpoints = args.attacker_endpoints or args.attacker_endpoint
    args.attacker_endpoints = [item.strip().rstrip("/") for item in raw_attacker_endpoints.split(",") if item.strip()]
    args.attacker_endpoint = args.attacker_endpoints[0] if args.attacker_endpoints else ""
    args.multiagent_provocateur_names = [
        item.strip() for item in args.multiagent_provocateur_names.split(",") if item.strip()
    ]
    if not args.target_endpoints:
        raise ValueError("--target-endpoints must contain at least one endpoint.")
    if not args.attacker_endpoints:
        raise ValueError("Provide --attacker-endpoint or --attacker-endpoints.")
    if len(args.multiagent_provocateur_names) < 2:
        raise ValueError("--multiagent-provocateur-names must contain at least two names.")
    return args


def main():
    asyncio.run(run_generation(parse_args()))


if __name__ == "__main__":
    main()
