import argparse
import importlib.util
import os

from datasets import load_dataset
from transformers import AutoTokenizer


def has_module(name):
    return importlib.util.find_spec(name) is not None


def require_module(name, install_hint):
    if not has_module(name):
        raise ImportError(f"Missing dependency '{name}'. Install it with: {install_hint}")


def build_lora_config(args):
    if not args.use_lora:
        return None

    require_module("peft", "pip install peft")
    from peft import LoraConfig

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or None,
    )


def main():
    require_module("trl", "pip install trl")

    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser(description="Run EMBER SFT debiasing training.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = (
        load_dataset("json", data_files=args.eval_data, split="train")
        if args.eval_data and os.path.exists(args.eval_data)
        else None
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        completion_only_loss=True,
        assistant_only_loss=True,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
