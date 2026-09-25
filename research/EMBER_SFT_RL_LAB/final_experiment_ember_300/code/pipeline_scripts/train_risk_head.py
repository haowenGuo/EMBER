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

from risk_gated_alignment.data import RiskHeadDataset, pad_collate
from risk_gated_alignment.modeling import RiskGatedCausalLM, save_trainable_state
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


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["risk_targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.mse_loss(outputs["risk_scores"], targets)
            total_loss += loss.item()
            batches += 1
    total_loss = reduce_scalar(total_loss, device, average=False)
    batches = reduce_scalar(batches, device, average=False)
    return total_loss / max(batches, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Train the 6-dim risk head.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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

    train_dataset = RiskHeadDataset(args.train_data, tokenizer, args.max_length)
    dev_dataset = RiskHeadDataset(args.dev_data, tokenizer, args.max_length)
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
    model.freeze_base()
    model.freeze_adapters()
    model.unfreeze_risk_head()
    model.to(device)
    if dist_state["distributed"]:
        model = DDP(model, device_ids=[dist_state["local_rank"]], output_device=dist_state["local_rank"])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = GradScaler(enabled=device.type == "cuda" and args.dtype == "fp16")
    logs_path = Path(args.output_dir) / "risk_head_log.jsonl"
    best_dev = float("inf")

    if is_main_process():
        print("Parameter counts:", count_parameters(model))
    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            steps = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["risk_targets"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(**autocast_kwargs):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = torch.nn.functional.mse_loss(outputs["risk_scores"], targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()
                steps += 1

            running = reduce_scalar(running, device, average=False)
            steps = reduce_scalar(steps, device, average=False)
            train_loss = running / max(steps, 1.0)
            dev_loss = evaluate(model, dev_loader, device)
            record = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss}
            if is_main_process():
                with open(logs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(record)

                if dev_loss < best_dev:
                    best_dev = dev_loss
                    save_trainable_state(Path(args.output_dir) / "best", model, {"stage": "risk_head"})
            barrier()

        if is_main_process():
            save_trainable_state(Path(args.output_dir) / "final", model, {"stage": "risk_head"})
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
