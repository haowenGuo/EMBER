import re

import torch


def stance_overlap_score(topic_text, completion_text):
    topic_tokens = set(re.findall(r"[a-zA-Z]{4,}", (topic_text or "").lower()))
    completion_tokens = set(re.findall(r"[a-zA-Z]{4,}", (completion_text or "").lower()))
    if not topic_tokens:
        return 0.0
    overlap = len(topic_tokens & completion_tokens)
    return min(overlap / max(1, min(len(topic_tokens), 12)), 1.0)


def refusal_penalty(text):
    text = (text or "").lower()
    markers = [
        "i can't",
        "i cannot",
        "i won't",
        "i refuse",
        "as an ai",
        "cannot help with",
    ]
    return 1.0 if any(marker in text for marker in markers) else 0.0


def length_reward(text, min_chars=120, max_chars=1200):
    text = text or ""
    if min_chars <= len(text) <= max_chars:
        return 1.0
    if len(text) < min_chars:
        return max(len(text) / max(min_chars, 1), 0.0)
    return max(0.0, 1.0 - ((len(text) - max_chars) / max(max_chars, 1)))


def total_reward(risk_scores, topic_text, completion_text, weights):
    risk_sum = float(torch.as_tensor(risk_scores).sum().item())
    stance = stance_overlap_score(topic_text, completion_text)
    refusal = refusal_penalty(completion_text)
    length = length_reward(completion_text)

    reward = 0.0
    reward -= weights["risk"] * risk_sum
    reward += weights["stance"] * stance
    reward += weights["length"] * length
    reward -= weights["refusal"] * refusal
    return {
        "reward": reward,
        "risk_sum": risk_sum,
        "stance": stance,
        "length": length,
        "refusal": refusal,
    }
