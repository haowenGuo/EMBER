import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_gated_alignment.data import EvalPromptDataset, pad_collate
from risk_gated_alignment.modeling import RiskGatedCausalLM, load_trainable_state
from risk_gated_alignment.rewards import total_reward
from risk_gated_alignment.utils import (
    all_gather_objects,
    barrier,
    cleanup_distributed,
    dump_json,
    ensure_dir,
    is_main_process,
    reduce_scalar,
    SequentialShardSampler,
    setup_distributed,
)


def decode_new_tokens(tokenizer, full_ids, prompt_length):
    new_ids = full_ids[prompt_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate on held-out prompts and score risk/stance/refusal metrics.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    args = parser.parse_args()

    dist_state = setup_distributed()
    ensure_dir(args.output_dir)
    device = dist_state["device"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = EvalPromptDataset(args.eval_data, tokenizer, args.max_prompt_length)
    collate = lambda batch: pad_collate(batch, tokenizer.pad_token_id)
    sampler = SequentialShardSampler(dataset) if dist_state["distributed"] else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, collate_fn=collate)

    model = RiskGatedCausalLM(args.model_name_or_path, dtype=args.dtype)
    if args.checkpoint:
        load_trainable_state(model, args.checkpoint)
    model.freeze_base()
    model.freeze_risk_head()
    model.freeze_adapters()
    model.to(device)
    model.eval()

    weights = {"risk": 1.0, "stance": 0.4, "length": 0.1, "refusal": 0.4}
    predictions = []
    totals = {"risk_sum": 0.0, "stance": 0.0, "length": 0.0, "refusal": 0.0, "reward": 0.0}

    try:
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                for row_idx in range(input_ids.size(0)):
                    prompt_ids = input_ids[row_idx : row_idx + 1]
                    prompt_mask = attention_mask[row_idx : row_idx + 1]
                    prompt_len = int(prompt_mask.sum().item())
                    prompt_ids = prompt_ids[:, :prompt_len]
                    prompt_mask = prompt_mask[:, :prompt_len]
                    generated_ids, _ = model.sample(
                        prompt_ids,
                        prompt_mask,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    completion_text = decode_new_tokens(tokenizer, generated_ids[0], prompt_len)
                    risk_scores = model.score_messages(
                        tokenizer,
                        batch["prompt_messages"][row_idx],
                        completion_text,
                        max_length=args.max_prompt_length + args.max_new_tokens,
                    )
                    reward = total_reward(risk_scores, batch["topic_text"][row_idx], completion_text, weights)
                    for key in totals:
                        totals[key] += reward[key]
                    predictions.append(
                        {
                            "topic_id": batch["topic_id"][row_idx],
                            "round": batch["round"][row_idx],
                            "completion": completion_text,
                            "risk_scores": risk_scores.tolist(),
                            "metrics": reward,
                        }
                    )

        count = reduce_scalar(len(predictions), device, average=False)
        metrics = {f"mean_{key}": reduce_scalar(value, device, average=False) / max(count, 1.0) for key, value in totals.items()}
        gathered_predictions = all_gather_objects(predictions)
        if is_main_process():
            merged_predictions = []
            for shard in gathered_predictions:
                merged_predictions.extend(shard)
            dump_json(Path(args.output_dir) / "metrics.json", metrics)
            with open(Path(args.output_dir) / "predictions.jsonl", "w", encoding="utf-8") as f:
                for row in merged_predictions:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(metrics)
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
