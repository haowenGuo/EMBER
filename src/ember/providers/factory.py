"""Provider and evaluator factory helpers."""

from __future__ import annotations

import os

from .base import TextProvider
from .evaluators import JsonLLMEvaluator
from .local import TransformersProvider
from .openai_compatible import OpenAICompatibleProvider
from .rule import RuleBasedEvaluator, RuleBasedTextProvider


def build_text_provider(
    provider: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    unsafe_first_pass: bool = True,
) -> TextProvider:
    if provider == "rule":
        return RuleBasedTextProvider(unsafe_first_pass=unsafe_first_pass)
    if provider == "openai":
        return OpenAICompatibleProvider(model=model, base_url=base_url, api_key=api_key)
    if provider == "transformers":
        model_name = model or os.getenv("TRANSFORMERS_MODEL")
        if not model_name:
            raise RuntimeError("--model or TRANSFORMERS_MODEL is required for transformers provider.")
        return TransformersProvider(model=model_name)
    raise ValueError(f"Unsupported provider: {provider}")


def build_stage_evaluator(kind: str, provider: TextProvider | None = None):
    if kind == "rule":
        return RuleBasedEvaluator().check_stage
    if kind == "llm":
        if provider is None:
            raise RuntimeError("LLM evaluator requires a text provider.")
        return JsonLLMEvaluator(provider).check_stage
    raise ValueError(f"Unsupported evaluator: {kind}")
