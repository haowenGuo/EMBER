"""Provider adapters for EMBER.

The core package does not require any cloud SDK at import time. Provider
dependencies are imported lazily only when the corresponding adapter is used.
"""

from .base import TextProvider
from .evaluators import JsonLLMEvaluator, loads_json_object
from .factory import build_stage_evaluator, build_text_provider
from .local import TransformersProvider
from .openai_compatible import OpenAICompatibleProvider
from .rule import RuleBasedEvaluator, RuleBasedTextProvider

__all__ = [
    "JsonLLMEvaluator",
    "OpenAICompatibleProvider",
    "RuleBasedEvaluator",
    "RuleBasedTextProvider",
    "TextProvider",
    "TransformersProvider",
    "build_stage_evaluator",
    "build_text_provider",
    "loads_json_object",
]
