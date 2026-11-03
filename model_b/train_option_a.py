"""CLI entry point for the repository's exact Option A fine-tune."""

import argparse
import json
import subprocess
from pathlib import Path

from .compare import compare_evaluation, compare_training
from .data import evaluate_model, load_pairs
from .experiment import load_reference_checkpoint, run_option_a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", help="Resolved Git SHA; inferred from this checkout if omitted")
    args = parser.parse_args()
    commit = args.repo_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[1]).strip()
    train_pairs, val_pairs, train_names, val_names = load_pairs(
        args.data_root, args.split_file)
    run_dir = run_option_a(args.source, args.run_dir, train_pairs, val_pairs,
                           train_names, val_names, args.data_root, commit)
    model, checkpoint, _ = load_reference_checkpoint(
        run_dir / "best.pth", train_names, val_names, kind="option_a")
    result, rows = evaluate_model(model.cuda(), val_pairs)
    evaluation = compare_evaluation(result, rows, reference="best", output_dir=run_dir)
    training = compare_training(run_dir, output_dir=run_dir)
    summary = dict(repo_commit=commit, best_local_step=int(checkpoint["local_step"]),
                   final_evaluation=evaluation, training_comparison=training)
    (run_dir / "reproducibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
