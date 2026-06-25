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

__all__ = [
    "DialogueTurn",
    "EmberAgent",
    "GateDecision",
    "MultiAgentDebateArena",
    "RiskReport",
    "RuleBasedBiasEvaluator",
    "StageGateHarness",
    "StageSnapshot",
    "compare_token_strategies",
]
