"""Command-line entry points for EMBER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import EmberAgent
from .providers import RuleBasedEvaluator, build_stage_evaluator, build_text_provider
from .runner import (
    HarnessRunner,
    STAGE_ORDER,
    default_stage_plan,
    load_stage_plan,
    run_builtin_benchmark,
    save_stage_plan,
    write_benchmark_csv,
)


def _provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["rule", "openai", "transformers"], default="rule")
    parser.add_argument("--model", help="Provider model name or local model path.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Prefer environment variables.")


def agent_demo(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EMBER-Agent self-check and rewrite demo.")
    _provider_args(parser)
    parser.add_argument("--prompt", default="Explain differences in task outcomes.")
    parser.add_argument("--safe-first-pass", action="store_true", help="Rule provider starts with safe text.")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    provider = build_text_provider(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        unsafe_first_pass=not args.safe_first_pass,
    )
    evaluator = RuleBasedEvaluator()
    agent = EmberAgent(
        generator=lambda prompt: provider.generate(prompt),
        evaluator=evaluator.check_text,
        max_refinement_rounds=args.max_rounds,
    )
    answer, checks = agent.run(args.prompt)
    payload = {
        "answer": answer,
        "check_count": len(checks),
        "clean": checks[-1].clean if checks else True,
        "findings": [
            {
                "dimension": finding.dimension,
                "evidence": finding.evidence,
                "severity": finding.severity,
            }
            for check in checks
            for finding in check.findings
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Answer: {answer}")
        print(f"Checks: {payload['check_count']} | Clean: {payload['clean']}")
    return 0


def harness_run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a stage-gated EMBER-Harness plan.")
    _provider_args(parser)
    parser.add_argument("--evaluator", choices=["rule", "llm"], default="rule")
    parser.add_argument("--strategy", choices=["final_only", "stage_gate", "per_call"], default="stage_gate")
    parser.add_argument(
        "--risk-stage",
        choices=[*STAGE_ORDER, "none"],
        default="retrieval",
        help="Built-in scenario risk stage. Use 'none' for a clean trajectory.",
    )
    parser.add_argument("--plan", help="Path to a JSON stage plan.")
    parser.add_argument("--write-plan", help="Write the built-in plan to JSON and exit.")
    parser.add_argument("--state-dir", default="outputs/harness_state")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    provider = build_text_provider(args.provider, model=args.model, base_url=args.base_url, api_key=args.api_key)
    evaluator = build_stage_evaluator(args.evaluator, provider=provider)
    stages = load_stage_plan(args.plan) if args.plan else default_stage_plan(args.risk_stage)
    if args.write_plan:
        save_stage_plan(args.write_plan, stages)
        print(f"Wrote plan: {args.write_plan}")
        return 0

    runner = HarnessRunner(provider=provider, evaluator=evaluator, state_dir=args.state_dir)
    result = runner.run(stages, strategy=args.strategy)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Strategy: {result.strategy}")
        print(f"Check calls: {result.check_calls}")
        print(f"Detected risks: {result.detected_risks}")
        print(f"Rollbacks: {result.rollback_count}")
        print(f"Expected total tokens: {result.expected_total_tokens:.0f}")
        print(f"Final artifact: {result.final_artifact}")
    return 0


def benchmark(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the built-in EMBER-Harness benchmark.")
    parser.add_argument("--output", default="outputs/ember_benchmark.csv")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    results = run_builtin_benchmark()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_benchmark_csv(output, results)
    payload = [result.to_dict() for result in results]
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.print_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        by_strategy: dict[str, dict[str, float]] = {}
        for result in results:
            row = by_strategy.setdefault(
                result.strategy,
                {"runs": 0, "check_calls": 0, "detected": 0, "tokens": 0.0},
            )
            row["runs"] += 1
            row["check_calls"] += result.check_calls
            row["detected"] += result.detected_risks
            row["tokens"] += result.expected_total_tokens
        print(f"Wrote benchmark CSV: {output}")
        for strategy, row in by_strategy.items():
            print(
                f"{strategy}: runs={row['runs']:.0f}, checks={row['check_calls']:.0f}, "
                f"detected={row['detected']:.0f}, expected_tokens={row['tokens']:.0f}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EMBER command group")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("agent-demo")
    subparsers.add_parser("harness-run")
    subparsers.add_parser("benchmark")
    args, remaining = parser.parse_known_args(argv)
    if args.command == "agent-demo":
        return agent_demo(remaining)
    if args.command == "harness-run":
        return harness_run(remaining)
    if args.command == "benchmark":
        return benchmark(remaining)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
