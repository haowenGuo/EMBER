import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from openai import OpenAI
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from risk_gated_alignment.data import render_chat
from risk_gated_alignment.modeling import RiskGatedCausalLM, load_trainable_state
from vllm_common import (
    append_jsonl,
    build_jobs,
    load_topics,
    provocateur_system_prompt,
    read_jsonl,
    render_agent_messages,
    target_system_prompt,
)


RISK_GATED_VARIANTS = {
    "SFT": {"variant_label": "SFT", "variant_type": "sft", "method": "baseline"},
    "SFT+RL": {"variant_label": "SFT+RL", "variant_type": "sft_rl", "method": "baseline"},
}


def parse_variants(raw):
    requested = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not requested:
        requested = ["SFT", "SFT+RL"]
    unknown = sorted(set(requested) - set(RISK_GATED_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown risk-gated variants: {unknown}")
    return [RISK_GATED_VARIANTS[item] for item in requested]


def load_completed(path):
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


def chat_attacker(client, args, messages):
    extra_body = {}
    if args.attacker_disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=args.attacker_model,
                messages=messages,
                temperature=args.attacker_temperature,
                max_tokens=args.attacker_max_tokens,
                extra_body=extra_body or None,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * attempt)


def decode_new_tokens(tokenizer, full_ids, prompt_length):
    return tokenizer.decode(full_ids[prompt_length:], skip_special_tokens=True).strip()


def generate_target(model, tokenizer, args, messages):
    prompt_text = render_chat(tokenizer, messages, add_generation_prompt=True)
    encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=args.max_prompt_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_length = int(attention_mask.sum().item())
    with torch.no_grad():
        full_ids, _ = model.sample(
            input_ids,
            attention_mask,
            max_new_tokens=args.target_max_tokens,
            temperature=args.target_temperature,
            top_p=args.target_top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
    return decode_new_tokens(tokenizer, full_ids[0], prompt_length)


class RiskGatedModelCache:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.loaded_variant = None
        self.model = None

    def checkpoint_for(self, variant_label):
        if variant_label == "SFT":
            return self.args.sft_checkpoint
        if variant_label == "SFT+RL":
            return self.args.rl_checkpoint
        raise ValueError(f"Unsupported variant: {variant_label}")

    def get(self, variant_label):
        if self.loaded_variant == variant_label and self.model is not None:
            return self.model, self.tokenizer
        if self.model is not None:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        checkpoint = self.checkpoint_for(variant_label)
        model = RiskGatedCausalLM(self.args.model_name_or_path, dtype=self.args.dtype)
        load_trainable_state(model, checkpoint)
        model.freeze_base()
        model.freeze_risk_head()
        model.freeze_adapters()
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self.model = model
        self.loaded_variant = variant_label
        return self.model, self.tokenizer


def run_job(args, job, model_cache, attacker_client, output_path, completed_rows):
    variant = job["variant"]
    topic = job["topic"]
    target_name = args.target_name
    attacker_name = args.provocateur_name
    target_prompt = target_system_prompt(target_name, topic["initial_context"], variant["method"])
    attacker_prompt = provocateur_system_prompt(attacker_name, topic["initial_context"])

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

    model, tokenizer = model_cache.get(variant["variant_label"])
    for round_index in range(start_round, args.max_debate_round + 1):
        record_id = record_id_for(job, round_index)
        if record_id in completed_rows:
            continue

        if round_index > 0:
            attacker_messages = render_agent_messages(attacker_prompt, attacker_name, global_history)
            attack = chat_attacker(attacker_client, args, attacker_messages)
            global_history.append({"sender": attacker_name, "content": attack})

        target_messages = render_agent_messages(target_prompt, target_name, global_history)
        target_response = generate_target(model, tokenizer, args, target_messages)
        global_history.append({"sender": target_name, "content": target_response})

        row = {
            "meta": {
                "record_id": record_id,
                "variant_label": variant["variant_label"],
                "variant_type": variant["variant_type"],
                "method": variant["method"],
                "topic_id": topic["topic_id"],
                "topic_index_1based": topic["topic_index_1based"],
                "round": round_index,
                "evaluate_source": "target_response",
                "protocol": "risk_gated_generation_offline_biasexpert_round05",
                "speaker_names": {"target": target_name, "provocateur": attacker_name},
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "checkpoint": model_cache.checkpoint_for(variant["variant_label"]),
            },
            "topic": topic,
            "target_response": target_response,
            "transcript": copy.deepcopy(global_history),
            "mitigation_trace": {
                "enabled": True,
                "mode": variant["variant_label"],
                "risk_gated_checkpoint": model_cache.checkpoint_for(variant["variant_label"]),
                "external_biasexpert_used": False,
                "note": "BiasExpert scoring is deferred to the offline evaluation phase.",
            },
        }
        append_jsonl(output_path, row)
        completed_rows[record_id] = row


def main():
    parser = argparse.ArgumentParser(description="Generate SFT/SFT+RL EMBER debates with custom risk-gated models.")
    parser.add_argument("--cmv-path", required=True)
    parser.add_argument("--topic-start", type=int, default=1)
    parser.add_argument("--topic-end", type=int, default=150)
    parser.add_argument("--max-debate-round", type=int, default=5)
    parser.add_argument("--variants", default="SFT,SFT+RL")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    parser.add_argument("--attacker-endpoint", required=True)
    parser.add_argument("--attacker-model", default="attacker")
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--provocateur-name", default="qwen_attacker")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--target-max-tokens", type=int, default=1024)
    parser.add_argument("--attacker-max-tokens", type=int, default=1024)
    parser.add_argument("--target-temperature", type=float, default=0.0)
    parser.add_argument("--target-top-p", type=float, default=0.95)
    parser.add_argument("--attacker-temperature", type=float, default=0.0)
    parser.add_argument("--attacker-disable-thinking", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"generated_rounds.worker{args.worker_index}.jsonl"
    completed_rows = load_completed(output_path)

    topics = load_topics(args.cmv_path, args.topic_start, args.topic_end)
    jobs = build_jobs(topics, parse_variants(args.variants))
    shard = [job for index, job in enumerate(jobs) if index % args.worker_count == args.worker_index]
    model_cache = RiskGatedModelCache(args)
    attacker_client = OpenAI(base_url=args.attacker_endpoint.rstrip("/"), api_key="EMPTY")

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmv_path": args.cmv_path,
        "topic_start": args.topic_start,
        "topic_end": args.topic_end,
        "max_debate_round": args.max_debate_round,
        "variants": [variant["variant_label"] for variant in parse_variants(args.variants)],
        "target_name": args.target_name,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "jobs_total": len(jobs),
        "jobs_this_worker": len(shard),
        "expected_rows_this_worker": len(shard) * (args.max_debate_round + 1),
        "model_name_or_path": args.model_name_or_path,
        "sft_checkpoint": args.sft_checkpoint,
        "rl_checkpoint": args.rl_checkpoint,
        "architecture": "RiskGatedCausalLM target workers data-parallel by topic/variant; shared vLLM attacker; offline BiasExpert scoring.",
    }
    with (output_dir / f"generation_manifest.worker{args.worker_index}.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    for job in shard:
        run_job(args, job, model_cache, attacker_client, output_path, completed_rows)


if __name__ == "__main__":
    main()
