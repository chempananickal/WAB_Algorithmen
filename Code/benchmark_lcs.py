"""Benchmark suite for Longest Common Substring (LCS) algorithms.

This script compares:
1) Suffix Automaton (SAM)
2) Enhanced Suffix Array (Suffix Array + LCP)

It generates reproducible test cases across multiple scenarios and string lengths,
runs each test case multiple times, and exports raw measurements, summary tables,
LaTeX tables, and plots for report usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from helpers.benchmarking import print_console_summary, run_benchmarks
from helpers.plotting import export_plots
from helpers.statistics_io import (
    aggregate_results,
    export_latex_tables,
    load_raw_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Suffix Automaton vs Enhanced Suffix Array for LCS."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "plot", "both"],
        default="both",
        help="Run benchmarks, generate plots from existing results, or both.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="100,500,1000,5000,10000",
        help="Comma-separated string lengths, e.g. 100,250,500",
    )
    parser.add_argument(
        "--cases-per-length",
        type=int,
        default=25,
        help="How many generated cases per scenario and length.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Benchmark repetitions per case and algorithm.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Code/Results",
        help="Directory for CSV, LaTeX tables, and plots.",
    )
    parser.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        default=True,
        help="Show single-line in-place progress output.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable progress output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    if any(x <= 0 for x in lengths):
        raise ValueError("All lengths must be positive integers.")
    if args.cases_per_length <= 0:
        raise ValueError("--cases-per-length must be > 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"run", "both"}:
        raw_df, lcs_cases_df = run_benchmarks(
            lengths=lengths,
            cases_per_length=args.cases_per_length,
            runs=args.runs,
            seed=args.seed,
            show_progress=args.show_progress,
        )
        raw_df.to_csv(output_dir / "raw_runs.csv", index=False)
        lcs_cases_df.to_csv(output_dir / "case_lcs_values.csv", index=False)

        config = {
            "lengths": lengths,
            "cases_per_length": args.cases_per_length,
            "runs": args.runs,
            "seed": args.seed,
            "output_dir": str(output_dir),
        }
        (output_dir / "benchmark_config.txt").write_text(
            "\n".join(f"{k}: {v}" for k, v in config.items()),
            encoding="utf-8",
        )

    if args.mode in {"plot", "both"}:
        raw_df = load_raw_results(output_dir)
        summary_df = aggregate_results(raw_df)
        summary_df.to_csv(output_dir / "summary_stats.csv", index=False)
        export_latex_tables(summary_df, output_dir)
        export_plots(summary_df, output_dir)
        print_console_summary(summary_df)

        print(f"\nSaved summary to: {output_dir / 'summary_stats.csv'}")
        print(f"Saved LaTeX tables in: {output_dir / 'tables'}")
        print(f"Saved plots in: {output_dir / 'plots'}")

    if args.mode in {"run", "both"}:
        print(f"Saved raw data to: {output_dir / 'raw_runs.csv'}")
        print(f"Saved case strings + LCS values to: {output_dir / 'case_lcs_values.csv'}")


if __name__ == "__main__":
    main()
