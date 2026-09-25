import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_gated_alignment.data import SFTDataset, pad_collate
from risk_gated_alignment.modeling import RiskGatedCausalLM, load_trainable_state, save_trainable_state
from risk_gated_alignment.utils import (
    barrier,
    cleanup_distributed,
    count_parameters,
    ensure_dir,
    is_main_process,
    reduce_scalar,
    SequentialShardSampler,
    seed_everything,
    setup_distributed,
)


def evaluate(model, loader, device, aux_weight):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            risk_penalty = outputs["risk_scores"].mean()
            loss = outputs["loss"] + aux_weight * risk_penalty
            total_loss += loss.item()
            batches += 1
    total_loss = reduce_scalar(total_loss, device, average=False)
    batches = reduce_scalar(batches, device, average=False)
    return total_loss / max(batches, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Train debiasing adapter bank with SFT.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--risk-head-checkpoint")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--risk-head-aux-weight", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    args = parser.parse_args()

    dist_state = setup_distributed()
    seed_everything(args.seed + dist_state["rank"])
    ensure_dir(args.output_dir)
    device = dist_state["device"]
    autocast_kwargs = {"device_type": "cuda", "enabled": device.type == "cuda"}
    if args.dtype == "bf16":
        autocast_kwargs["dtype"] = torch.bfloat16
    elif args.dtype == "fp16":
        autocast_kwargs["dtype"] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SFTDataset(args.train_data, tokenizer, args.max_length)
    dev_dataset = SFTDataset(args.dev_data, tokenizer, args.max_length)
    collate = lambda batch: pad_collate(batch, tokenizer.pad_token_id)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist_state["distributed"] else None
    dev_sampler = SequentialShardSampler(dev_dataset) if dist_state["distributed"] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=dev_sampler,
        collate_fn=collate,
    )

    model = RiskGatedCausalLM(args.model_name_or_path, dtype=args.dtype)
    if args.risk_head_checkpoint:
        load_trainable_state(model, args.risk_head_checkpoint)
    model.freeze_base()
    model.freeze_risk_head()
    model.unfreeze_adapters()
    model.to(device)
    if dist_state["distributed"]:
        model = DDP(model, device_ids=[dist_state["local_rank"]], output_device=dist_state["local_rank"])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = GradScaler(enabled=device.type == "cuda" and args.dtype == "fp16")
    logs_path = Path(args.output_dir) / "sft_log.jsonl"
    best_dev = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if is_main_process():
        print(
            "SFT dataset stats:",
            {
                "train_samples": len(train_dataset),
                "train_skipped": getattr(train_dataset, "skipped_rows", 0),
                "dev_samples": len(dev_dataset),
                "dev_skipped": getattr(dev_dataset, "skipped_rows", 0),
            },
        )
        print("Parameter counts:", count_parameters(model))
    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            total = 0.0
            steps = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(**autocast_kwargs):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    risk_penalty = outputs["risk_scores"].mean()
                    loss = outputs["loss"] + args.risk_head_aux_weight * risk_penalty
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
                steps += 1

            total = reduce_scalar(total, device, average=False)
            steps = reduce_scalar(steps, device, average=False)
            train_loss = total / max(steps, 1.0)
            dev_loss = evaluate(model, dev_loader, device, args.risk_head_aux_weight)
            improved = dev_loss < (best_dev - args.early_stopping_min_delta)
            if improved:
                best_dev = dev_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            early_stop = (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "best_dev_loss": best_dev,
                "best_epoch": best_epoch,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
                "early_stop": early_stop,
            }
            if is_main_process():
                with open(logs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(record)

                if improved:
                    save_trainable_state(Path(args.output_dir) / "best", model, {"stage": "sft"})
            barrier()
            if early_stop:
                if is_main_process():
                    print(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_dev_loss": best_dev,
                            "patience": args.early_stopping_patience,
                            "min_delta": args.early_stopping_min_delta,
                        }
                    )
                break

        if is_main_process():
            save_trainable_state(Path(args.output_dir) / "final", model, {"stage": "sft"})
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
