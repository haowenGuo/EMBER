from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ember import EmberAgent, RuleBasedBiasEvaluator


def generator(prompt: str) -> str:
    if "Revised answer" in prompt:
        return "The evidence is mixed, and no group-level conclusion should be made without context."
    return "This group is always inferior at the task."


def main() -> None:
    agent = EmberAgent(
        generator=generator,
        evaluator=RuleBasedBiasEvaluator(),
        max_refinement_rounds=2,
    )
    answer, checks = agent.run("Explain differences in task outcomes.")
    print("answer:", answer)
    print("checks:", len(checks))


if __name__ == "__main__":
    main()
