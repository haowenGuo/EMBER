import argparse
import importlib.util
import os
import re
from functools import lru_cache

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SCORE_MAP = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}

DIMENSION_KEYS = [
    "political",
    "gender",
    "ethnic_cultural",
    "age",
    "religion",
    "disability",
]


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


def extract_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(item.get("content", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def tokenize_words(text):
    return set(re.findall(r"[a-zA-Z]{4,}", (text or "").lower()))


class BiasExpertScorer:
    def __init__(self, model_path, max_input_length=5000, max_new_tokens=1024):
        self.model_path = model_path
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.model.device

    def _extract_json_object(self, raw_text):
        import json

        decoder = json.JSONDecoder()
        for index, char in enumerate(raw_text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(raw_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "bias_analysis" in parsed:
                return parsed
        return None

    def _score_payload(self, payload):
        total = 0
        analysis = payload.get("bias_analysis") or {}
        for key in DIMENSION_KEYS:
            level = ((analysis.get(key) or {}).get("level", "None"))
            total += SCORE_MAP.get(str(level).lower(), 0)
        return total

    def score_text(self, text):
        prompt = (
            "Analyze the assistant response for bias across political, gender, ethnic/cultural, "
            "age, religion, and disability dimensions. Return JSON with a bias_analysis object.\n\n"
            f"Assistant response:\n{text}"
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        raw_text = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        payload = self._extract_json_object(raw_text)
        if not payload:
            return 6.0
        return float(self._score_payload(payload))


def main():
    require_module("trl", "pip install trl")

    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser(description="Run EMBER GRPO debiasing training.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--bias-expert-model-path")
    parser.add_argument("--bias-weight", type=float, default=1.0)
    parser.add_argument("--stance-weight", type=float, default=0.5)
    parser.add_argument("--format-weight", type=float, default=0.25)
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
    tokenizer.padding_side = "left"

    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = (
        load_dataset("json", data_files=args.eval_data, split="train")
        if args.eval_data and os.path.exists(args.eval_data)
        else None
    )

    scorer = (
        BiasExpertScorer(args.bias_expert_model_path)
        if args.bias_expert_model_path
        else None
    )

    @lru_cache(maxsize=10000)
    def cached_bias_score(text):
        if scorer is None:
            return 0.0
        return scorer.score_text(text)

    def bias_reward(prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            completion_text = extract_completion_text(completion)
            bias_score = cached_bias_score(completion_text)
            rewards.append(-args.bias_weight * bias_score)
        return rewards

    def stance_reward(prompts, completions, topic_text=None, **kwargs):
        topic_text = topic_text or [""] * len(completions)
        rewards = []
        for completion, topic in zip(completions, topic_text):
            completion_text = extract_completion_text(completion)
            prompt_tokens = tokenize_words(topic)
            completion_tokens = tokenize_words(completion_text)
            if not prompt_tokens:
                rewards.append(0.0)
                continue
            overlap = len(prompt_tokens & completion_tokens) / max(1, min(len(prompt_tokens), 12))
            rewards.append(args.stance_weight * min(overlap, 1.0))
        return rewards

    def format_reward(prompts, completions, **kwargs):
        rewards = []
        refusal_markers = [
            "i can't",
            "i cannot",
            "i won't",
            "i refuse",
            "as an ai",
            "cannot help",
        ]
        for completion in completions:
            completion_text = extract_completion_text(completion)
            text_lower = completion_text.lower()
            reward = 0.0
            if 80 <= len(completion_text) <= 1200:
                reward += args.format_weight
            if any(marker in text_lower for marker in refusal_markers):
                reward -= args.format_weight
            rewards.append(reward)
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=[bias_reward, stance_reward, format_reward],
        peft_config=build_lora_config(args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
