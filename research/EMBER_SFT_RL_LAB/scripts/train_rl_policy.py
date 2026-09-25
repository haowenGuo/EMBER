import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_gated_alignment.data import RLPromptDataset, pad_collate
from risk_gated_alignment.modeling import RiskGatedCausalLM, load_trainable_state, save_trainable_state, sequence_logprob
from risk_gated_alignment.rewards import total_reward
from risk_gated_alignment.utils import (
    barrier,
    cleanup_distributed,
    count_parameters,
    ensure_dir,
    is_main_process,
    reduce_scalar,
    seed_everything,
    setup_distributed,
    unwrap_model,
)


def decode_new_tokens(tokenizer, full_ids, prompt_length):
    new_ids = full_ids[prompt_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_labels(full_ids, prompt_length):
    labels = full_ids.clone()
    labels[:prompt_length] = -100
    return labels


def main():
    parser = argparse.ArgumentParser(description="Lightweight group policy optimization for adapter bank.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--dev-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--stance-weight", type=float, default=0.4)
    parser.add_argument("--length-weight", type=float, default=0.1)
    parser.add_argument("--refusal-weight", type=float, default=0.4)
    parser.add_argument("--repetition-weight", type=float, default=0.4)
    parser.add_argument("--overlength-weight", type=float, default=0.4)
    parser.add_argument("--overlength-chars", type=int, default=2200)
    parser.add_argument("--overlength-hard-chars", type=int, default=3600)
    parser.add_argument("--kl-weight", type=float, default=0.03)
    parser.add_argument("--use-separate-scorer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    args = parser.parse_args()

    dist_state = setup_distributed()
    seed_everything(args.seed + dist_state["rank"])
    ensure_dir(args.output_dir)
    device = dist_state["device"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = RLPromptDataset(args.train_data, tokenizer, args.max_prompt_length)
    collate = lambda batch: pad_collate(batch, tokenizer.pad_token_id)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist_state["distributed"] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
    )

    model = RiskGatedCausalLM(args.model_name_or_path, dtype=args.dtype)
    load_trainable_state(model, args.sft_checkpoint)
    model.freeze_base()
    model.freeze_risk_head()
    model.unfreeze_adapters()
    model.to(device)
    if dist_state["distributed"]:
        model = DDP(model, device_ids=[dist_state["local_rank"]], output_device=dist_state["local_rank"])
    policy = unwrap_model(model)

    scorer_model = None
    if args.use_separate_scorer or args.kl_weight > 0:
        scorer_model = RiskGatedCausalLM(args.model_name_or_path, dtype=args.dtype)
        load_trainable_state(scorer_model, args.sft_checkpoint)
        scorer_model.freeze_base()
        scorer_model.freeze_risk_head()
        scorer_model.freeze_adapters()
        scorer_model.to(device)
        scorer_model.eval()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    logs_path = Path(args.output_dir) / "rl_log.jsonl"
    reward_weights = {
        "risk": args.risk_weight,
        "stance": args.stance_weight,
        "length": args.length_weight,
        "refusal": args.refusal_weight,
        "repetition": args.repetition_weight,
        "overlength": args.overlength_weight,
        "overlength_chars": args.overlength_chars,
        "overlength_hard_chars": args.overlength_hard_chars,
    }

    if is_main_process():
        print("Parameter counts:", count_parameters(model))
    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for step, batch in enumerate(train_loader, 1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                prompt_messages = batch["prompt_messages"]
                topic_texts = batch["topic_text"]

                optimizer.zero_grad(set_to_none=True)
                group_records = []
                all_losses = []

                for row_idx in range(input_ids.size(0)):
                    prompt_ids = input_ids[row_idx : row_idx + 1]
                    prompt_mask = attention_mask[row_idx : row_idx + 1]
                    prompt_len = int(prompt_mask.sum().item())
                    prompt_ids = prompt_ids[:, :prompt_len]
                    prompt_mask = prompt_mask[:, :prompt_len]
                    row_samples = []

                    for _ in range(args.samples_per_prompt):
                        generated_ids, _ = policy.sample(
                            prompt_ids,
                            prompt_mask,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.8,
                            top_p=0.95,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        completion_text = decode_new_tokens(tokenizer, generated_ids[0], prompt_len)
                        scorer = scorer_model if scorer_model is not None else policy
                        scorer_risk = scorer.score_messages(
                            tokenizer,
                            prompt_messages[row_idx],
                            completion_text,
                            max_length=args.max_prompt_length + args.max_new_tokens,
                        )
                        reward_info = total_reward(scorer_risk, topic_texts[row_idx], completion_text, reward_weights)

                        full_ids = generated_ids.to(device)
                        full_mask = torch.ones_like(full_ids, device=device)
                        labels = build_labels(full_ids[0], prompt_len).unsqueeze(0).to(device)
                        outputs = model(input_ids=full_ids, attention_mask=full_mask)
                        seq_logprob = sequence_logprob(outputs["logits"], labels)
                        kl_penalty = 0.0
                        if args.kl_weight > 0 and scorer_model is not None:
                            with torch.no_grad():
                                ref_outputs = scorer_model(input_ids=full_ids, attention_mask=full_mask)
                                ref_logprob = sequence_logprob(ref_outputs["logits"], labels)
                                kl_penalty = torch.abs(seq_logprob.detach() - ref_logprob).mean().item()
                            reward_info["reward"] -= args.kl_weight * kl_penalty
                            reward_info["kl"] = kl_penalty
                        else:
                            reward_info["kl"] = 0.0

                        row_samples.append(
                            {
                                "reward": reward_info["reward"],
                                "risk_sum": reward_info["risk_sum"],
                                "stance": reward_info["stance"],
                                "length": reward_info["length"],
                                "refusal": reward_info["refusal"],
                                "repetition": reward_info["repetition"],
                                "overlength": reward_info["overlength"],
                                "kl": reward_info["kl"],
                                "seq_logprob": seq_logprob,
                            }
                        )

                    rewards = torch.tensor([sample["reward"] for sample in row_samples], device=device, dtype=torch.float32)
                    advantages = rewards - rewards.mean()
                    row_loss = 0.0
                    for sample, advantage in zip(row_samples, advantages):
                        row_loss = row_loss - advantage.detach() * sample["seq_logprob"].mean()
                    all_losses.append(row_loss / len(row_samples))
                    group_records.extend(row_samples)

                loss = torch.stack(all_losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

                reward_count = max(len(group_records), 1)
                reward_sum = sum(item["reward"] for item in group_records)
                risk_sum = sum(item["risk_sum"] for item in group_records)
                stance_sum = sum(item["stance"] for item in group_records)
                repetition_sum = sum(item["repetition"] for item in group_records)
                overlength_sum = sum(item["overlength"] for item in group_records)
                kl_sum = sum(item["kl"] for item in group_records)

                loss_value = reduce_scalar(loss.item(), device, average=True)
                reward_sum = reduce_scalar(reward_sum, device, average=False)
                risk_sum = reduce_scalar(risk_sum, device, average=False)
                stance_sum = reduce_scalar(stance_sum, device, average=False)
                repetition_sum = reduce_scalar(repetition_sum, device, average=False)
                overlength_sum = reduce_scalar(overlength_sum, device, average=False)
                kl_sum = reduce_scalar(kl_sum, device, average=False)
                reward_count = reduce_scalar(reward_count, device, average=False)
                record = {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss_value),
                    "mean_reward": reward_sum / max(reward_count, 1.0),
                    "mean_risk_sum": risk_sum / max(reward_count, 1.0),
                    "mean_stance": stance_sum / max(reward_count, 1.0),
                    "mean_repetition": repetition_sum / max(reward_count, 1.0),
                    "mean_overlength": overlength_sum / max(reward_count, 1.0),
                    "mean_kl": kl_sum / max(reward_count, 1.0),
                }
                if is_main_process():
                    with open(logs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    print(record)

            if is_main_process():
                save_trainable_state(Path(args.output_dir) / f"epoch_{epoch}", model, {"stage": "rl", "epoch": epoch})
            barrier()

        if is_main_process():
            save_trainable_state(Path(args.output_dir) / "final", model, {"stage": "rl"})
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
