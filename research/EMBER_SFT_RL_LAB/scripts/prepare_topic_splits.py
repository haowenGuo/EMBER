import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_gated_alignment.data import build_topic_split


def main():
    parser = argparse.ArgumentParser(description="Create topic-level train/dev/test split for EMBER alignment.")
    parser.add_argument(
        "--cmv-path",
        default=str(ROOT / "source_data" / "changemyview_persuasion_kto.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "datasets" / "topic_split.json"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-topics", type=int, default=100)
    args = parser.parse_args()

    split = build_topic_split(
        args.cmv_path,
        args.output_path,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        max_topics=args.max_topics,
    )
    print(f"Saved topic split to {args.output_path}")
    print({key: len(value) for key, value in split.items()})


if __name__ == "__main__":
    main()
