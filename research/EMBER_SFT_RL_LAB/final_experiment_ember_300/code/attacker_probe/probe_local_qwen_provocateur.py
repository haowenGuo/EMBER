import argparse
import importlib.util
import json
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = (
    REPO_ROOT
    / "formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen"
    / "run_fiveway_exact_biasexpert_eval.py"
)


def enable_untrusted_hub_kernels():
    """Allow FP8 hub kernels needed by Qwen FP8 checkpoints during this probe."""
    import transformers.integrations.hub_kernels as hub_kernels

    original_get_kernel_hub = hub_kernels.get_kernel_hub

    def trusted_get_kernel_hub(*args, **kwargs):
        kwargs["trust_remote_code"] = True
        return original_get_kernel_hub(*args, **kwargs)

    hub_kernels.get_kernel_hub = trusted_get_kernel_hub
    print("[PROBE] enabled trust_remote_code=True for hub kernels", flush=True)


def load_eval_module():
    spec = importlib.util.spec_from_file_location("qwen_exact_eval", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_report(eval_module, evaluator, text):
    report = evaluator.evaluate(text)
    labels = eval_module.bias_labels_from_report(report)
    return {
        "total_bias_score": labels["total_score"] if labels else None,
        "bias_labels": labels["labels"] if labels else None,
        "parser_mode": report.get("parser_mode") if isinstance(report, dict) else None,
        "error": report.get("error") if isinstance(report, dict) else None,
        "summary": report.get("bias_summary") if isinstance(report, dict) else None,
    }


def compact_history(history):
    return [
        {
            "sender": item["sender"],
            "content": item["content"],
            "char_count": len(item.get("content") or ""),
        }
        for item in history
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Probe whether a local Qwen model can replace Doubao as the EMBER provocateur."
    )
    parser.add_argument("--attacker-model-path", required=True)
    parser.add_argument(
        "--attacker-label",
        default="",
        help="Speaker label for debugging. Empty preserves the reference protocol label: doubao.",
    )
    parser.add_argument(
        "--target-model-path",
        default="/data2/guohaowen_data/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
    )
    parser.add_argument("--bias-expert-model-path", default="")
    parser.add_argument(
        "--cmv-path",
        default=str(REPO_ROOT / "source_data" / "changemyview_persuasion_kto.jsonl"),
    )
    parser.add_argument("--topic-index", type=int, default=101)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--attacker-device", default="cuda:1")
    parser.add_argument("--evaluator-device", default="cuda:2")
    parser.add_argument("--target-max-new-tokens", type=int, default=512)
    parser.add_argument("--attacker-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--allow-untrusted-kernels",
        action="store_true",
        help="Allow remote hub kernels such as kernels-community/finegrained-fp8.",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "provocateur_probe"))
    args = parser.parse_args()

    if args.allow_untrusted_kernels:
        enable_untrusted_hub_kernels()

    eval_module = load_eval_module()
    target_name = eval_module.LEGACY_TARGET_NAME
    attacker_name = args.attacker_label or eval_module.LEGACY_PROVOCATEUR_NAME
    if not args.bias_expert_model_path:
        args.bias_expert_model_path = str(eval_module.resolve_bias_expert_default())

    topics = eval_module.load_cmv_topic_range(args.cmv_path, args.topic_index, args.topic_index)
    if not topics:
        raise RuntimeError(f"No topic found at index {args.topic_index}")
    topic = topics[0]

    print(f"[PROBE] topic={topic['topic_id']} title={topic['title']}", flush=True)
    print(f"[PROBE] target={args.target_model_path} device={args.target_device}", flush=True)
    print(f"[PROBE] attacker={args.attacker_model_path} device={args.attacker_device}", flush=True)
    print(f"[PROBE] evaluator={args.bias_expert_model_path} device={args.evaluator_device}", flush=True)

    target_engine = eval_module.LocalPipelineChatEngine(
        args.target_model_path,
        device=args.target_device,
        max_new_tokens=args.target_max_new_tokens,
        trust_remote_code=True,
    )
    attacker_engine = eval_module.LocalPipelineChatEngine(
        args.attacker_model_path,
        device=args.attacker_device,
        max_new_tokens=args.attacker_max_new_tokens,
        trust_remote_code=True,
        disable_thinking=True,
    )
    evaluator = eval_module.BiasExpertEvaluator(args.bias_expert_model_path, device=args.evaluator_device)

    target_agent = eval_module.ExactTargetAgent(
        name=target_name,
        engine=target_engine,
        stance=topic["initial_context"],
        method="baseline",
    )
    attacker_agent = eval_module.ExactProvocateurAgent(
        name=attacker_name,
        engine=attacker_engine,
        stance=topic["initial_context"],
        persona_type="aggressive",
    )
    arena = eval_module.ExactDebateArena(target_agent, attacker_agent)

    scored_rounds = []
    for round_index in range(args.rounds + 1):
        transcript = arena.run(round_index, args.rounds)
        target_text = arena.get_target_responses_for_eval()
        target_score = score_report(eval_module, evaluator, target_text)
        latest_attacker = next(
            (
                item["content"]
                for item in reversed(transcript)
                if item["sender"] == attacker_name
            ),
            "",
        )
        attacker_score = score_report(eval_module, evaluator, latest_attacker) if latest_attacker else None
        scored_rounds.append(
            {
                "round": round_index,
                "target_score": target_score,
                "attacker_score": attacker_score,
                "target_response": target_text,
                "latest_attacker_message": latest_attacker,
                "transcript": compact_history(transcript),
            }
        )
        print(
            "[PROBE] "
            f"round={round_index} "
            f"target_score={target_score['total_bias_score']} "
            f"attacker_score={attacker_score['total_bias_score'] if attacker_score else 'NA'} "
            f"target_chars={len(target_text)} "
            f"attacker_chars={len(latest_attacker)}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / (
            f"{attacker_name}_topic{args.topic_index}_r{args.rounds}"
            f"_t{args.target_max_new_tokens}_a{args.attacker_max_new_tokens}.json"
        )
    )
    payload = {
        "attacker_label": attacker_name,
        "attacker_model_path": args.attacker_model_path,
        "target_model_path": args.target_model_path,
        "bias_expert_model_path": args.bias_expert_model_path,
        "topic": topic,
        "rounds": args.rounds,
        "target_max_new_tokens": args.target_max_new_tokens,
        "attacker_max_new_tokens": args.attacker_max_new_tokens,
        "scored_rounds": scored_rounds,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PROBE] output={output_path}", flush=True)

    del arena, target_agent, attacker_agent, target_engine, attacker_engine, evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
