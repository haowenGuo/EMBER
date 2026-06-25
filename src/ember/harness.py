"""EMBER-Harness stage gates, snapshots, and rollback."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


RiskEvaluator = Callable[[str, str, dict[str, Any]], "RiskReport"]


@dataclass(slots=True)
class RiskReport:
    level: str
    score: float
    reason: str
    dimensions: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.level.lower() in {"medium", "high"} or self.score >= 0.5


@dataclass(slots=True)
class GateDecision:
    stage_id: str
    status: str
    action: str
    reason: str
    snapshot_id: str
    rollback_to: str | None = None
    risk: RiskReport | None = None


@dataclass(slots=True)
class StageSnapshot:
    snapshot_id: str
    stage_id: str
    artifact: str
    metadata: dict[str, Any]
    status: str
    parent_id: str | None
    created_at: float


class RuleBasedGate:
    """Deterministic gate used for examples and tests."""

    def __init__(self, risky_terms: list[str] | None = None) -> None:
        self.risky_terms = risky_terms or [
            "inferior",
            "superior",
            "always",
            "never",
            "dangerous group",
            "should not be trusted",
        ]

    def __call__(self, stage_id: str, artifact: str, metadata: dict[str, Any]) -> RiskReport:
        lowered = artifact.lower()
        hits = [term for term in self.risky_terms if term in lowered]
        if not hits:
            return RiskReport(level="none", score=0.0, reason="No demo risk term matched.")
        return RiskReport(
            level="high",
            score=0.9,
            reason=f"Demo gate matched risky terms at stage '{stage_id}': {', '.join(hits)}",
            dimensions=["stereotype_or_overgeneralization"],
        )


class StageGateHarness:
    """Stage-level control layer for EMBER-Agent.

    A stage artifact is first saved as a pending snapshot. Only artifacts that
    pass the EMBER Gate are committed and allowed to enter later context. On
    failure, the harness points the runner back to the latest committed
    snapshot so the current stage can be repaired locally.
    """

    def __init__(
        self,
        evaluator: RiskEvaluator | None = None,
        state_dir: str | Path | None = None,
    ) -> None:
        self.evaluator = evaluator or RuleBasedGate()
        self.state_dir = Path(state_dir) if state_dir else None
        self.snapshots: list[StageSnapshot] = []
        self.audit_log: list[GateDecision] = []
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_committed(self) -> StageSnapshot | None:
        for snapshot in reversed(self.snapshots):
            if snapshot.status == "committed":
                return snapshot
        return None

    def run_stage(
        self,
        stage_id: str,
        artifact: str,
        metadata: dict[str, Any] | None = None,
    ) -> GateDecision:
        metadata = metadata or {}
        parent = self.last_committed
        snapshot = StageSnapshot(
            snapshot_id=f"ember-{uuid.uuid4().hex[:12]}",
            stage_id=stage_id,
            artifact=artifact,
            metadata=metadata,
            status="pending",
            parent_id=parent.snapshot_id if parent else None,
            created_at=time.time(),
        )
        self.snapshots.append(snapshot)
        self._persist_snapshot(snapshot)

        risk = self.evaluator(stage_id, artifact, metadata)
        if risk.blocked:
            snapshot.status = "failed"
            rollback_to = parent.snapshot_id if parent else None
            decision = GateDecision(
                stage_id=stage_id,
                status="failed",
                action="rollback",
                reason=risk.reason,
                snapshot_id=snapshot.snapshot_id,
                rollback_to=rollback_to,
                risk=risk,
            )
        else:
            snapshot.status = "committed"
            decision = GateDecision(
                stage_id=stage_id,
                status="passed",
                action="commit",
                reason=risk.reason,
                snapshot_id=snapshot.snapshot_id,
                risk=risk,
            )

        self.audit_log.append(decision)
        self._persist_snapshot(snapshot)
        self._persist_decision(decision)
        return decision

    def rollback_context(self) -> dict[str, Any] | None:
        snapshot = self.last_committed
        if snapshot is None:
            return None
        return {
            "snapshot_id": snapshot.snapshot_id,
            "stage_id": snapshot.stage_id,
            "artifact": snapshot.artifact,
            "metadata": snapshot.metadata,
        }

    def export_audit(self) -> list[dict[str, Any]]:
        return [asdict(decision) for decision in self.audit_log]

    def _persist_snapshot(self, snapshot: StageSnapshot) -> None:
        if not self.state_dir:
            return
        path = self.state_dir / f"{snapshot.snapshot_id}.json"
        path.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_decision(self, decision: GateDecision) -> None:
        if not self.state_dir:
            return
        path = self.state_dir / "audit.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")


def compare_token_strategies(
    work_tokens: int,
    model_call_count: int,
    stage_gate_count: int,
    average_check_tokens: int,
    bias_probability: float,
    rework_all_tokens: int,
    rework_stage_tokens: int,
) -> dict[str, dict[str, float]]:
    """Compute the Chapter 4 token formulas for three check placements."""

    return {
        "final_only": {
            "work_tokens": work_tokens,
            "check_tokens": average_check_tokens,
            "expected_rework_tokens": bias_probability * rework_all_tokens,
            "expected_total_tokens": work_tokens
            + average_check_tokens
            + bias_probability * rework_all_tokens,
        },
        "per_call": {
            "work_tokens": work_tokens,
            "check_tokens": model_call_count * average_check_tokens,
            "expected_rework_tokens": 0,
            "expected_total_tokens": work_tokens + model_call_count * average_check_tokens,
        },
        "stage_gate": {
            "work_tokens": work_tokens,
            "check_tokens": stage_gate_count * average_check_tokens,
            "expected_rework_tokens": bias_probability * rework_stage_tokens,
            "expected_total_tokens": work_tokens
            + stage_gate_count * average_check_tokens
            + bias_probability * rework_stage_tokens,
        },
    }
