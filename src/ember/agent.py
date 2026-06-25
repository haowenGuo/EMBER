"""EMBER-Agent: reflection-based bias mitigation loop.

The public implementation is provider-agnostic. Production systems can plug in
OpenAI-compatible clients, local Transformers models, or any callable that
accepts a prompt and returns text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


TextGenerator = Callable[[str], str]


@dataclass(slots=True)
class BiasFinding:
    """Single bias finding produced by a checker."""

    dimension: str
    evidence: str
    severity: str = "low"


@dataclass(slots=True)
class BiasCheckResult:
    """Structured result for an EMBER-Agent reflection step."""

    clean: bool
    findings: list[BiasFinding] = field(default_factory=list)
    feedback: str = ""


class RuleBasedBiasEvaluator:
    """Small deterministic evaluator for demos and tests.

    This is not a substitute for BiasExpert. It exists so that the public
    package can run without private models or API keys.
    """

    def __init__(self, risky_terms: Iterable[str] | None = None) -> None:
        self.risky_terms = tuple(
            term.lower()
            for term in (
                risky_terms
                or [
                    "always",
                    "never",
                    "inferior",
                    "superior",
                    "lazy",
                    "dangerous group",
                    "should not be trusted",
                ]
            )
        )

    def __call__(self, text: str) -> BiasCheckResult:
        lowered = text.lower()
        findings = [
            BiasFinding(
                dimension="stereotype_or_overgeneralization",
                evidence=term,
                severity="medium",
            )
            for term in self.risky_terms
            if term in lowered
        ]
        if not findings:
            return BiasCheckResult(clean=True)
        feedback = (
            "Remove unsupported group generalizations, add evidence bounds, "
            "and rewrite the answer with neutral language."
        )
        return BiasCheckResult(clean=False, findings=findings, feedback=feedback)


@dataclass(slots=True)
class EmberAgent:
    """Reflection-revision agent used by EMBER.

    Parameters
    ----------
    generator:
        Callable that generates the initial answer from a task prompt.
    evaluator:
        Callable that checks whether a candidate answer contains bias.
    rewriter:
        Callable that rewrites a candidate answer from feedback. If omitted,
        a conservative built-in rewrite prompt is passed to ``generator``.
    max_refinement_rounds:
        Maximum self-check and rewrite loops.
    """

    generator: TextGenerator
    evaluator: Callable[[str], BiasCheckResult] = field(default_factory=RuleBasedBiasEvaluator)
    rewriter: TextGenerator | None = None
    max_refinement_rounds: int = 3

    def run(self, prompt: str) -> tuple[str, list[BiasCheckResult]]:
        response = self.generator(prompt)
        checks: list[BiasCheckResult] = []

        for _ in range(self.max_refinement_rounds):
            check = self.evaluator(response)
            checks.append(check)
            if check.clean:
                break
            response = self._rewrite(prompt, response, check.feedback)

        return response, checks

    def _rewrite(self, prompt: str, response: str, feedback: str) -> str:
        rewrite_prompt = (
            "Rewrite the answer so it stays evidence-bound, neutral, and "
            "free of unsupported group claims.\n\n"
            f"Original task:\n{prompt}\n\n"
            f"Current answer:\n{response}\n\n"
            f"Feedback:\n{feedback}\n\n"
            "Revised answer:"
        )
        if self.rewriter is not None:
            return self.rewriter(rewrite_prompt)
        return self.generator(rewrite_prompt)
