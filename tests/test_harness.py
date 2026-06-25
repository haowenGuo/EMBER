from ember import StageGateHarness, compare_token_strategies


def test_stage_gate_rolls_back_to_last_committed_snapshot():
    harness = StageGateHarness()

    first = harness.run_stage("input_parse", "neutral request")
    assert first.status == "passed"

    second = harness.run_stage("retrieval", "one group is always inferior")
    assert second.status == "failed"
    assert second.action == "rollback"

    context = harness.rollback_context()
    assert context is not None
    assert context["stage_id"] == "input_parse"


def test_token_strategy_comparison():
    result = compare_token_strategies(
        work_tokens=1000,
        model_call_count=8,
        stage_gate_count=4,
        average_check_tokens=100,
        bias_probability=0.5,
        rework_all_tokens=800,
        rework_stage_tokens=200,
    )

    assert result["final_only"]["expected_total_tokens"] == 1500
    assert result["per_call"]["expected_total_tokens"] == 1800
    assert result["stage_gate"]["expected_total_tokens"] == 1500
