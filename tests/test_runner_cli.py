import json
import subprocess
import sys

from ember import HarnessRunner, default_stage_plan, run_builtin_benchmark


def test_harness_runner_repairs_stage_gate_risk():
    runner = HarnessRunner()
    result = runner.run(default_stage_plan("retrieval"), strategy="stage_gate")

    assert result.detected_risks == 1
    assert result.rollback_count == 1
    assert result.check_calls == 8
    assert result.records[2].repaired is True
    assert "neutral" in result.records[2].artifact


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


def test_stage_gate_persists_snapshots_and_audit(tmp_path):
    runner = HarnessRunner(state_dir=tmp_path)
    result = runner.run(default_stage_plan("retrieval"), strategy="stage_gate")
    audit_path = tmp_path / "audit.jsonl"
    audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    snapshot_files = list(tmp_path.glob("ember-*.json"))

    assert audit_path.exists()
    assert len(audit_rows) == result.check_calls
    assert len(snapshot_files) == result.check_calls
    assert any(row["status"] == "failed" and row["action"] == "rollback" for row in audit_rows)


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
