import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from ember import HarnessRunner, default_stage_plan, run_builtin_benchmark


def test_harness_runner_repairs_stage_gate_risk():
    runner = HarnessRunner()
    result = runner.run(default_stage_plan("retrieval"), strategy="stage_gate")

    assert result.detected_risks == 1
    assert result.rollback_count == 1
    assert result.check_calls == 8
    assert result.records[2].repaired is True
    assert "neutral" in result.records[2].artifact


def test_default_stage_plan_supports_explicit_no_risk():
    runner = HarnessRunner()
    result = runner.run(default_stage_plan("none"), strategy="stage_gate")

    assert result.detected_risks == 0
    assert result.rollback_count == 0


def test_default_stage_plan_rejects_unknown_risk_stage():
    with pytest.raises(ValueError, match="risk_stage must be one of"):
        default_stage_plan("retreival")


def test_builtin_benchmark_has_all_three_strategies():
    results = run_builtin_benchmark()
    strategies = {result.strategy for result in results}

    assert strategies == {"final_only", "stage_gate", "per_call"}
    assert len(results) == 18


def test_builtin_benchmark_orders_check_costs():
    results = run_builtin_benchmark()
    check_calls = {}
    expected_tokens = {}
    for result in results:
        check_calls[result.strategy] = check_calls.get(result.strategy, 0) + result.check_calls
        expected_tokens[result.strategy] = (
            expected_tokens.get(result.strategy, 0) + result.expected_total_tokens
        )

    assert check_calls["final_only"] < check_calls["stage_gate"] < check_calls["per_call"]
    assert expected_tokens["final_only"] < expected_tokens["stage_gate"] < expected_tokens["per_call"]


def test_stage_gate_persists_snapshots_and_audit():
    state_dir = Path(".pytest_tmp") / f"stage-state-{uuid.uuid4().hex}"
    try:
        runner = HarnessRunner(state_dir=state_dir)
        result = runner.run(default_stage_plan("retrieval"), strategy="stage_gate")
        audit_path = state_dir / "audit.jsonl"
        audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        snapshot_files = list(state_dir.glob("ember-*.json"))

        assert audit_path.exists()
        assert len(audit_rows) == result.check_calls
        assert len(snapshot_files) == result.check_calls
        assert any(row["status"] == "failed" and row["action"] == "rollback" for row in audit_rows)
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


def test_cli_agent_demo_json_runs():
    output = subprocess.check_output(
        [sys.executable, "-m", "ember.cli", "agent-demo", "--json"],
        text=True,
    )
    payload = json.loads(output)

    assert payload["check_count"] >= 1
    assert payload["clean"] is True


def test_cli_harness_run_json_runs():
    output = subprocess.check_output(
        [sys.executable, "-m", "ember.cli", "harness-run", "--json"],
        text=True,
    )
    payload = json.loads(output)

    assert payload["strategy"] == "stage_gate"
    assert payload["detected_risks"] == 1
