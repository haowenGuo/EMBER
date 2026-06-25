"""EMBER public reference implementation."""

from .agent import EmberAgent, RuleBasedBiasEvaluator
from .arena import DialogueTurn, MultiAgentDebateArena
from .harness import (
    GateDecision,
    RiskReport,
    StageGateHarness,
    StageSnapshot,
    compare_token_strategies,
)
from .providers import (
    JsonLLMEvaluator,
    OpenAICompatibleProvider,
    RuleBasedEvaluator,
    RuleBasedTextProvider,
    TextProvider,
    TransformersProvider,
    build_stage_evaluator,
    build_text_provider,
)
from .runner import HarnessRunner, HarnessRunResult, StageTask, default_stage_plan, run_builtin_benchmark

__all__ = [
    "DialogueTurn",
    "EmberAgent",
    "GateDecision",
    "JsonLLMEvaluator",
    "MultiAgentDebateArena",
    "OpenAICompatibleProvider",
    "RuleBasedEvaluator",
    "RiskReport",
    "RuleBasedBiasEvaluator",
    "RuleBasedTextProvider",
    "StageGateHarness",
    "StageTask",
    "StageSnapshot",
    "TextProvider",
    "TransformersProvider",
    "HarnessRunner",
    "HarnessRunResult",
    "build_stage_evaluator",
    "build_text_provider",
    "compare_token_strategies",
    "default_stage_plan",
    "run_builtin_benchmark",
]
