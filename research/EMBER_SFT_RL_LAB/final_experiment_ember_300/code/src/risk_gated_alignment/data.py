import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .utils import format_messages_fallback, read_jsonl


DIMENSIONS = [
    "political",
    "gender",
    "ethnic_cultural",
    "age",
    "religion",
    "disability",
]

SCORE_MAP = {"none": 0, "low": 1, "moderate": 2, "high": 3}


def load_topic_map(cmv_path):
    topic_map = {}
    for index, row in enumerate(read_jsonl(cmv_path), 1):
        prompt_list = row.get("prompt") or []
        if not prompt_list:
            continue
        topic_map[f"cmv_{index:03d}"] = prompt_list[0].get("content", "")
    return topic_map


def normalize_model_name(raw_name):
    name = (raw_name or "").lower()
    if "gpt" in name:
        return "gpt"
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    if "doubao" in name:
        return "doubao"
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "unknown"


def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def extract_bias_payload(report):
    if not isinstance(report, dict):
        return None
    if isinstance(report.get("bias_analysis"), dict):
        return report
    raw = report.get("raw")
    if not isinstance(raw, str):
        return None
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "bias_analysis" in payload:
            return payload
    return None


def bias_labels_from_report(report):
    payload = extract_bias_payload(report)
    if not payload:
        return None

    bias_analysis = payload.get("bias_analysis") or {}
    labels = {}
    total = 0
    for dim in DIMENSIONS:
        level = ((bias_analysis.get(dim) or {}).get("level", "None"))
        score = SCORE_MAP.get(str(level).lower(), 0)
        labels[dim] = score
        total += score
    return {"labels": labels, "total_score": total}


def infer_target_sender(record, target_response=None):
    transcript = record.get("transcript") or []
    target_response = normalize_text(target_response or record.get("target_response", ""))
    if transcript and target_response:
        last_message = transcript[-1]
        if normalize_text(last_message.get("content", "")) == target_response:
            return last_message.get("sender")

    meta_model = (record.get("meta") or {}).get("model", "")
    normalized = normalize_model_name(meta_model)
    if normalized != "unknown":
        return normalized
    return None


def build_prompt_completion(record, topic_map):
    transcript = record.get("transcript")
    if not isinstance(transcript, list) or not transcript:
        return None

    target_response = normalize_text(record.get("target_response", ""))
    if not target_response:
        target_response = normalize_text((transcript[-1] or {}).get("content", ""))
    if not target_response:
        return None

    topic_id = (record.get("meta") or {}).get("topic_id", "unknown")
    topic_text = topic_map.get(topic_id, "")
    target_sender = infer_target_sender(record, target_response=target_response)
    model_family = normalize_model_name(target_sender)
    if model_family == "unknown":
        model_family = normalize_model_name((record.get("meta") or {}).get("model"))

    prompt = [
        {
            "role": "system",
            "content": (
                "You are the defender in an adversarial debate. "
                f"You advocate the following position: {topic_text}"
            ),
        }
    ]

    for index, message in enumerate(transcript):
        sender = message.get("sender", "user")
        content = normalize_text(message.get("content", ""))
        if not content:
            continue

        final_target_turn = (
            index == len(transcript) - 1
            and sender == target_sender
            and content == target_response
        )
        if final_target_turn:
            break

        if sender == target_sender:
            prompt.append({"role": "assistant", "content": content})
        else:
            prefix = "" if sender == "user" else f"[{sender}] "
            prompt.append({"role": "user", "content": f"{prefix}{content}"})

    completion = [{"role": "assistant", "content": target_response}]
    return {
        "prompt": prompt,
        "completion": completion,
        "topic_id": topic_id,
        "round": int((record.get("meta") or {}).get("rounds", (record.get("meta") or {}).get("round", -1))),
        "model_family": model_family,
        "topic_text": topic_text,
    }


def prompt_hash(messages):
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def build_topic_split(cmv_path, output_path, train_ratio=0.7, dev_ratio=0.15, seed=42, max_topics=100):
    topic_map = load_topic_map(cmv_path)
    topic_ids = sorted(topic_map)
    if max_topics is not None:
        topic_ids = topic_ids[:max_topics]
    rng = random.Random(seed)
    rng.shuffle(topic_ids)

    train_cut = int(len(topic_ids) * train_ratio)
    dev_cut = int(len(topic_ids) * (train_ratio + dev_ratio))

    split = {
        "train": sorted(topic_ids[:train_cut]),
        "dev": sorted(topic_ids[train_cut:dev_cut]),
        "test": sorted(topic_ids[dev_cut:]),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    return split


def render_chat(tokenizer, messages, add_generation_prompt=False):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return format_messages_fallback(messages, add_generation_prompt=add_generation_prompt)


class RiskHeadDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = list(read_jsonl(path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        full_messages = row["prompt"] + row["completion"]
        text = render_chat(self.tokenizer, full_messages, add_generation_prompt=False)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [row["bias_labels"][dim] / 3.0 for dim in DIMENSIONS],
            dtype=torch.float32,
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "risk_targets": labels,
        }


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.skipped_rows = 0
        for row in read_jsonl(path):
            sample = self._build_sample(row)
            if sample is None:
                self.skipped_rows += 1
                continue
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def _build_sample(self, row):
        prompt_text = render_chat(self.tokenizer, row["prompt"], add_generation_prompt=True)
        full_text = render_chat(
            self.tokenizer,
            row["prompt"] + row["completion"],
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        full = self.tokenizer(
            full_text,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]

        # Keep the completion tokens whenever the dialogue is too long.
        start = max(0, input_ids.size(0) - self.max_length)
        if start > 0:
            input_ids = input_ids[start:]
            attention_mask = attention_mask[start:]

        labels = input_ids.clone()
        prompt_length = min(prompt_ids.size(0), full["input_ids"][0].size(0))
        prompt_tokens_in_window = max(0, prompt_length - start)
        if prompt_tokens_in_window > 0:
            labels[:prompt_tokens_in_window] = -100

        valid_label_count = int((labels != -100).sum().item())
        if valid_label_count == 0:
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "topic_text": row.get("topic_text", ""),
            "valid_label_count": valid_label_count,
        }

    def __getitem__(self, index):
        return self.samples[index]


class RLPromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = list(read_jsonl(path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt_text = render_chat(self.tokenizer, row["prompt"], add_generation_prompt=True)
        encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "prompt_messages": row["prompt"],
            "topic_text": row.get("topic_text", ""),
            "topic_id": row.get("topic_id", "unknown"),
            "round": row.get("round", -1),
        }


class EvalPromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = list(read_jsonl(path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt_text = render_chat(self.tokenizer, row["prompt"], add_generation_prompt=True)
        encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "prompt_messages": row["prompt"],
            "topic_text": row.get("topic_text", ""),
            "topic_id": row.get("topic_id", "unknown"),
            "round": row.get("round", -1),
            "gold_total_score": row.get("total_score"),
            "gold_bias_labels": row.get("bias_labels"),
        }


def pad_collate(features, pad_token_id):
    max_len = max(item["input_ids"].size(0) for item in features)
    batch = {}
    for key in ["input_ids", "attention_mask", "labels", "risk_targets"]:
        if key not in features[0]:
            continue
        tensors = []
        for item in features:
            value = item[key]
            if key == "risk_targets":
                tensors.append(value)
                continue
            pad_value = 0 if key == "attention_mask" else (-100 if key == "labels" else pad_token_id)
            if value.size(0) < max_len:
                value = torch.cat(
                    [value, torch.full((max_len - value.size(0),), pad_value, dtype=value.dtype)],
                    dim=0,
                )
            tensors.append(value)
        batch[key] = torch.stack(tensors)

    for key in features[0]:
        if key in batch:
            continue
        batch[key] = [item[key] for item in features]
    return batch
