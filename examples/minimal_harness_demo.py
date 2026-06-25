from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ember import StageGateHarness


def main() -> None:
    harness = StageGateHarness(state_dir="outputs/demo_snapshots")

    stages = [
        ("input_parse", "The user asks for a balanced comparison."),
        ("retrieval", "Retrieved source says one group is always inferior."),
        ("retrieval", "Retrieved source is balanced and evidence-bound."),
        ("final_response", "Final answer summarizes the evidence neutrally."),
    ]

    for stage_id, artifact in stages:
        decision = harness.run_stage(stage_id, artifact)
        print(stage_id, decision.status, decision.action, decision.reason)
        if decision.action == "rollback":
            print("rollback context:", harness.rollback_context())


if __name__ == "__main__":
    main()
