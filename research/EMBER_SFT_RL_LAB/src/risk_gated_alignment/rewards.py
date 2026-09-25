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


def repetition_penalty(text):
    words = re.findall(r"[a-zA-Z0-9']+", (text or "").lower())
    if len(words) < 40:
        return 0.0

    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    max_unigram_ratio = max(counts.values()) / max(len(words), 1)
    unigram_penalty = max(0.0, min((max_unigram_ratio - 0.12) / 0.23, 1.0))

    if len(words) < 80:
        return unigram_penalty

    trigrams = {}
    for index in range(len(words) - 2):
        key = tuple(words[index : index + 3])
        trigrams[key] = trigrams.get(key, 0) + 1
    max_trigram_count = max(trigrams.values()) if trigrams else 1
    trigram_penalty = max(0.0, min((max_trigram_count - 4) / 12, 1.0))
    return max(unigram_penalty, trigram_penalty)


def overlength_penalty(text, max_chars=2200, hard_chars=3600):
    length = len(text or "")
    if length <= max_chars:
        return 0.0
    return max(0.0, min((length - max_chars) / max(hard_chars - max_chars, 1), 1.0))


def total_reward(risk_scores, topic_text, completion_text, weights):
    risk_sum = float(torch.as_tensor(risk_scores).sum().item())
    stance = stance_overlap_score(topic_text, completion_text)
    refusal = refusal_penalty(completion_text)
    length = length_reward(completion_text)
    repetition = repetition_penalty(completion_text)
    overlength = overlength_penalty(
        completion_text,
        max_chars=weights.get("overlength_chars", 2200),
        hard_chars=weights.get("overlength_hard_chars", 3600),
    )

    reward = 0.0
    reward -= weights.get("risk", 1.0) * risk_sum
    reward += weights.get("stance", 0.4) * stance
    reward += weights.get("length", 0.1) * length
    reward -= weights.get("refusal", 0.4) * refusal
    reward -= weights.get("repetition", 0.0) * repetition
    reward -= weights.get("overlength", 0.0) * overlength
    return {
        "reward": reward,
        "risk_sum": risk_sum,
        "stance": stance,
        "length": length,
        "refusal": refusal,
        "repetition": repetition,
        "overlength": overlength,
    }
