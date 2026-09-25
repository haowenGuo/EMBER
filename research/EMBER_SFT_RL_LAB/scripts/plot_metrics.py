import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_gated_alignment.utils import ensure_dir, read_jsonl


def plot_training_curve(log_path, output_path, title, y_keys):
    rows = list(read_jsonl(log_path))
    if not rows:
        return
    x_key = "step" if "step" in rows[0] else "epoch"
    xs = [row[x_key] for row in rows]
    plt.figure(figsize=(8, 4))
    for key in y_keys:
        ys = [row[key] for row in rows if key in row]
        if ys:
            plt.plot(xs[: len(ys)], ys, label=key)
    plt.title(title)
    plt.xlabel(x_key.capitalize())
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_eval_bars(metrics_paths, labels, output_path):
    metric_names = []
    loaded = []
    for path in metrics_paths:
        with open(path, "r", encoding="utf-8") as f:
            loaded.append(json.load(f))
    if loaded:
        metric_names = list(loaded[0].keys())

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]
    for axis, metric_name in zip(axes, metric_names):
        values = [metrics[metric_name] for metrics in loaded]
        axis.bar(labels, values)
        axis.set_title(metric_name)
        axis.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves and evaluation bar charts.")
    parser.add_argument("--risk-log")
    parser.add_argument("--sft-log")
    parser.add_argument("--rl-log")
    parser.add_argument("--eval-metrics", nargs="*")
    parser.add_argument("--eval-labels", nargs="*")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    if args.risk_log:
        plot_training_curve(args.risk_log, Path(args.output_dir) / "risk_head_curve.png", "Risk Head", ["train_loss", "dev_loss"])
    if args.sft_log:
        plot_training_curve(args.sft_log, Path(args.output_dir) / "sft_curve.png", "SFT", ["train_loss", "dev_loss"])
    if args.rl_log:
        plot_training_curve(args.rl_log, Path(args.output_dir) / "rl_curve.png", "RL", ["loss", "mean_reward", "mean_risk_sum"])
    if args.eval_metrics and args.eval_labels and len(args.eval_metrics) == len(args.eval_labels):
        plot_eval_bars(args.eval_metrics, args.eval_labels, Path(args.output_dir) / "eval_compare.png")


if __name__ == "__main__":
    main()
