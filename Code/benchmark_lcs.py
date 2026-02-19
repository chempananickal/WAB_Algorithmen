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
import sys
import time
import tracemalloc
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import random


@dataclass(frozen=True)
class LCSResult:
    substring: str
    length: int


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    scenario: str
    length: int
    s: str
    t: str
    metadata: dict[str, Any]


def lcs_suffix_automaton(s: str, t: str) -> LCSResult:
    """Longest common substring using a Suffix Automaton built from s."""
    if not s or not t:
        return LCSResult("", 0)

    transitions: list[dict[str, int]] = [{}]
    suffix_link: list[int] = [-1]
    state_length: list[int] = [0]
    last = 0

    def extend(ch: str) -> None:
        nonlocal last
        cur = len(transitions)
        transitions.append({})
        suffix_link.append(0)
        state_length.append(state_length[last] + 1)

        p = last
        while p != -1 and ch not in transitions[p]:
            transitions[p][ch] = cur
            p = suffix_link[p]

        if p == -1:
            suffix_link[cur] = 0
        else:
            q = transitions[p][ch]
            if state_length[p] + 1 == state_length[q]:
                suffix_link[cur] = q
            else:
                clone = len(transitions)
                transitions.append(dict(transitions[q]))
                suffix_link.append(suffix_link[q])
                state_length.append(state_length[p] + 1)

                while p != -1 and transitions[p].get(ch) == q:
                    transitions[p][ch] = clone
                    p = suffix_link[p]

                suffix_link[q] = clone
                suffix_link[cur] = clone

        last = cur

    for char in s:
        extend(char)

    state = 0
    current_length = 0
    best_length = 0
    best_end_position = -1

    for index, char in enumerate(t):
        if char in transitions[state]:
            state = transitions[state][char]
            current_length += 1
        else:
            while state != -1 and char not in transitions[state]:
                state = suffix_link[state]

            if state == -1:
                state = 0
                current_length = 0
                continue

            current_length = state_length[state] + 1
            state = transitions[state][char]

        if current_length > best_length:
            best_length = current_length
            best_end_position = index

    if best_length == 0:
        return LCSResult("", 0)

    start = best_end_position - best_length + 1
    return LCSResult(t[start : best_end_position + 1], best_length) # best end pos + 1 becuase slice end is exclusive


def _pick_separator(s: str, t: str) -> str: # Added because I'm paranoid and because I might reuse this with a near full ASCII search space
    """Pick a separator character that does not occur in either input string."""
    used = set(s) | set(t)
    for codepoint in range(1, 1024):
        candidate = chr(codepoint)
        if candidate not in used:
            return candidate
    raise ValueError("Could not find a separator character not present in input strings.")


def build_suffix_array(text: str) -> list[int]:
    """Build suffix array with prefix-doubling (O(n log n))."""
    n = len(text)
    if n == 0:
        return []

    suffix_array = list(range(n))
    rank = [ord(ch) for ch in text]
    k = 1

    while True:
        suffix_array.sort(key=lambda idx: (rank[idx], rank[idx + k] if idx + k < n else -1))

        new_rank = [0] * n
        for i in range(1, n):
            previous = suffix_array[i - 1]
            current = suffix_array[i]
            prev_key = (rank[previous], rank[previous + k] if previous + k < n else -1)
            curr_key = (rank[current], rank[current + k] if current + k < n else -1)
            new_rank[current] = new_rank[previous] + (curr_key != prev_key)

        rank = new_rank
        if rank[suffix_array[-1]] == n - 1:
            break
        k <<= 1

    return suffix_array


def build_lcp_array(text: str, suffix_array: list[int]) -> list[int]:
    """Build LCP array using Kasai's algorithm (O(n))."""
    n = len(text)
    if n == 0:
        return []

    rank = [0] * n
    for i, suffix_start in enumerate(suffix_array):
        rank[suffix_start] = i

    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            continue
        j = suffix_array[r - 1]
        while i + h < n and j + h < n and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        if h > 0:
            h -= 1

    return lcp


def lcs_enhanced_suffix_array(s: str, t: str) -> LCSResult:
    """Longest common substring using Enhanced Suffix Array (SA + LCP)."""
    if not s or not t:
        return LCSResult("", 0)

    separator = _pick_separator(s, t)
    joined = s + separator + t
    split_index = len(s)

    suffix_array = build_suffix_array(joined)
    lcp = build_lcp_array(joined, suffix_array)

    best_length = 0
    best_start = 0

    for i in range(1, len(suffix_array)):
        left = suffix_array[i - 1]
        right = suffix_array[i]

        left_from_s = left < split_index
        right_from_s = right < split_index
        if left_from_s == right_from_s:
            continue

        candidate = lcp[i]
        if candidate > best_length:
            best_length = candidate
            best_start = right

    if best_length == 0:
        return LCSResult("", 0)

    substring = joined[best_start : best_start + best_length]
    if separator in substring:
        substring = substring.split(separator, maxsplit=1)[0]

    return LCSResult(substring, len(substring))


def random_string(rng: random.Random, length: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def mutate_string(rng: random.Random, source: str, mutation_rate: float, alphabet: str) -> str:
    if not source:
        return source
    chars = list(source)
    mutations = max(1, int(round(len(chars) * mutation_rate)))
    indices = rng.sample(range(len(chars)), k=min(mutations, len(chars)))
    for idx in indices:
        original = chars[idx]
        alternatives = [c for c in alphabet if c != original]
        chars[idx] = rng.choice(alternatives) if alternatives else original
    return "".join(chars)


def insert_substring(rng: random.Random, host: str, fragment: str) -> str:
    position = rng.randint(0, len(host))
    return host[:position] + fragment + host[position:]


def generate_cases(lengths: list[int], cases_per_length: int, seed: int) -> list[BenchmarkCase]:
    """Generate reproducible test scenarios across all requested lengths."""
    rng = random.Random(seed)
    dna_alphabet = "ACGT"
    full_alphabet = "abcdefghijklmnopqrstuvwxyz"
    disjoint_a = "abcdefghijklm"
    disjoint_b = "nopqrstuvwxyz"

    scenarios = [
        "random_uniform",
        "mutated_implant",
        "repetitive_with_noise",
        "near_identical",
        "disjoint_alphabet",
    ]

    cases: list[BenchmarkCase] = []
    for length in lengths:
        for case_no in range(1, cases_per_length + 1):
            for scenario in scenarios:
                case_id = f"{scenario}_n{length}_c{case_no}"

                if scenario == "random_uniform":
                    s = random_string(rng, length, full_alphabet)
                    t = random_string(rng, length, full_alphabet)
                    metadata = {}

                elif scenario == "mutated_implant":
                    motif_length = max(8, length // 4)
                    motif = random_string(rng, motif_length, dna_alphabet)
                    mutated = mutate_string(rng, motif, mutation_rate=0.12, alphabet=dna_alphabet)

                    base_s = random_string(rng, length, dna_alphabet)
                    base_t = random_string(rng, length, dna_alphabet)
                    s = insert_substring(rng, base_s, motif)
                    t = insert_substring(rng, base_t, mutated)
                    metadata = {
                        "motif_length": motif_length,
                        "mutation_rate": 0.12,
                    }

                elif scenario == "repetitive_with_noise":
                    pattern_length = max(3, length // 20)
                    pattern = random_string(rng, pattern_length, dna_alphabet)
                    repeats = max(1, length // pattern_length)
                    s_base = (pattern * repeats)[:length]
                    t_base = (pattern[::-1] * repeats)[:length]
                    s = mutate_string(rng, s_base, mutation_rate=0.06, alphabet=dna_alphabet)
                    t = mutate_string(rng, t_base, mutation_rate=0.08, alphabet=dna_alphabet)
                    metadata = {
                        "pattern_length": pattern_length,
                        "mutation_rates": "0.06/0.08",
                    }

                elif scenario == "near_identical":
                    s = random_string(rng, length, full_alphabet)
                    t = mutate_string(rng, s, mutation_rate=0.02, alphabet=full_alphabet)
                    metadata = {"mutation_rate": 0.02}

                elif scenario == "disjoint_alphabet":
                    s = random_string(rng, length, disjoint_a)
                    t = random_string(rng, length, disjoint_b)
                    metadata = {}

                else:
                    raise ValueError(f"Unknown scenario: {scenario}")

                cases.append(
                    BenchmarkCase(
                        case_id=case_id,
                        scenario=scenario,
                        length=length,
                        s=s,
                        t=t,
                        metadata=metadata,
                    )
                )
    return cases


def measure_algorithm(
    algorithm: Callable[[str, str], LCSResult],
    s: str,
    t: str,
) -> tuple[float, float, int]:
    """Return runtime (ms), peak memory (KiB), and LCS length for one call."""
    tracemalloc.start()
    start = time.perf_counter_ns()
    result = algorithm(s, t)
    end = time.perf_counter_ns()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_ms = (end - start) / 1_000_000
    peak_kib = peak / 1024
    return elapsed_ms, peak_kib, result.length


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["scenario", "length", "algorithm"], as_index=False).agg(
        time_mean_ms=("time_ms", "mean"),
        time_median_ms=("time_ms", "median"),
        time_std_ms=("time_ms", "std"),
        time_iqr_ms=("time_ms", iqr),
        time_min_ms=("time_ms", "min"),
        time_max_ms=("time_ms", "max"),
        memory_mean_kib=("peak_memory_kib", "mean"),
        memory_median_kib=("peak_memory_kib", "median"),
        memory_std_kib=("peak_memory_kib", "std"),
        memory_iqr_kib=("peak_memory_kib", iqr),
        memory_min_kib=("peak_memory_kib", "min"),
        memory_max_kib=("peak_memory_kib", "max"),
        lcs_mean_length=("lcs_length", "mean"),
        runs=("run", "count"),
    )
    return grouped.sort_values(by=["scenario", "length", "algorithm"]).reset_index(drop=True)


def export_latex_tables(summary_df: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for scenario, scenario_df in summary_df.groupby("scenario"):
        compact = scenario_df[
            [
                "length",
                "algorithm",
                "time_mean_ms",
                "time_median_ms",
                "time_std_ms",
                "time_iqr_ms",
                "memory_mean_kib",
                "memory_median_kib",
                "memory_std_kib",
                "memory_iqr_kib",
                "runs",
            ]
        ].copy()

        latex = compact.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
        (tables_dir / f"summary_{scenario}.tex").write_text(latex, encoding="utf-8")


def export_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for scenario, scenario_df in summary_df.groupby("scenario"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        for algorithm, algorithm_df in scenario_df.groupby("algorithm"):
            sorted_df = algorithm_df.sort_values("length")
            axes[0].plot(
                sorted_df["length"],
                sorted_df["time_mean_ms"],
                marker="o",
                label=algorithm,
            )
            axes[1].plot(
                sorted_df["length"],
                sorted_df["memory_mean_kib"],
                marker="o",
                label=algorithm,
            )

        axes[0].set_title(f"Runtime vs Length ({scenario})")
        axes[0].set_xlabel("String length")
        axes[0].set_ylabel("Mean runtime [ms]")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Peak memory vs Length ({scenario})")
        axes[1].set_xlabel("String length")
        axes[1].set_ylabel("Mean peak memory [KiB]")
        axes[1].grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.savefig(plots_dir / f"benchmark_{scenario}.png", dpi=220)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Suffix Automaton vs Enhanced Suffix Array for LCS."
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="100,250,500,1000,10000",
        help="Comma-separated string lengths, e.g. 100,250,500",
    )
    parser.add_argument(
        "--cases-per-length",
        type=int,
        default=5,
        help="How many generated cases per scenario and length.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
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


def _format_seconds_as_timedelta(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, round(seconds)))))


def _render_progress(current: int, total: int, elapsed_seconds: float, eta_seconds: float, width: int = 30) -> str:
    fraction = current / total if total else 1.0
    filled = int(width * fraction)
    bar = "=" * filled + "-" * (width - filled)

    percent = fraction * 100
    return (
        f"[{bar}] {percent:6.2f}% "
        f"({current}/{total}) "
        f"elapsed={_format_seconds_as_timedelta(elapsed_seconds)} "
        f"eta={_format_seconds_as_timedelta(eta_seconds)}"
    )


def run_benchmarks(
    lengths: list[int],
    cases_per_length: int,
    runs: int,
    seed: int,
    show_progress: bool = True,
) -> pd.DataFrame:
    cases = generate_cases(lengths=lengths, cases_per_length=cases_per_length, seed=seed)

    algorithms: dict[str, Callable[[str, str], LCSResult]] = {
        "Suffix Automaton": lcs_suffix_automaton,
        "Enhanced Suffix Array": lcs_enhanced_suffix_array,
    }

    rows: list[dict[str, Any]] = []
    total_steps = len(cases) * runs * len(algorithms)
    completed_steps = 0
    started_at = time.perf_counter()

    plan: list[tuple[str, int]] = []
    for case in cases:
        for _ in range(runs):
            for algorithm_name in algorithms:
                plan.append((algorithm_name, case.length))

    observed_sums: dict[tuple[str, int], float] = {}
    observed_counts: dict[tuple[str, int], int] = {}
    global_observed_time = 0.0
    global_observed_count = 0

    def estimate_remaining_seconds(completed_index: int) -> float:
        if completed_index >= len(plan):
            return 0.0

        global_mean = (
            global_observed_time / global_observed_count if global_observed_count > 0 else 0.0
        )
        total_remaining = 0.0
        for algorithm_name, length in plan[completed_index:]:
            key = (algorithm_name, length)
            if key in observed_counts and observed_counts[key] > 0:
                total_remaining += observed_sums[key] / observed_counts[key]
            elif global_mean > 0:
                total_remaining += global_mean
        return total_remaining

    if show_progress:
        elapsed_seconds = max(0.0, time.perf_counter() - started_at)
        eta_seconds = estimate_remaining_seconds(0)
        sys.stdout.write("\r" + _render_progress(0, total_steps, elapsed_seconds, eta_seconds))
        sys.stdout.flush()

    for case in cases:
        for run_index in range(1, runs + 1):
            run_outcomes: dict[str, int] = {}

            for algorithm_name, algorithm in algorithms.items():
                elapsed_ms, peak_kib, lcs_length = measure_algorithm(algorithm, case.s, case.t)
                run_outcomes[algorithm_name] = lcs_length

                elapsed_seconds = elapsed_ms / 1000.0
                bucket = (algorithm_name, case.length)
                observed_sums[bucket] = observed_sums.get(bucket, 0.0) + elapsed_seconds
                observed_counts[bucket] = observed_counts.get(bucket, 0) + 1
                global_observed_time += elapsed_seconds
                global_observed_count += 1

                rows.append(
                    {
                        "case_id": case.case_id,
                        "scenario": case.scenario,
                        "length": case.length,
                        "run": run_index,
                        "algorithm": algorithm_name,
                        "time_ms": elapsed_ms,
                        "peak_memory_kib": peak_kib,
                        "lcs_length": lcs_length,
                    }
                )

                completed_steps += 1
                if show_progress:
                    elapsed_wall = max(0.0, time.perf_counter() - started_at)
                    eta_seconds = estimate_remaining_seconds(completed_steps)
                    sys.stdout.write(
                        "\r" + _render_progress(completed_steps, total_steps, elapsed_wall, eta_seconds)
                    )
                    sys.stdout.flush()

            sam_length = run_outcomes["Suffix Automaton"]
            esa_length = run_outcomes["Enhanced Suffix Array"]
            if sam_length != esa_length:
                raise ValueError(
                    "Correctness mismatch detected: "
                    f"case={case.case_id}, run={run_index}, SAM={sam_length}, ESA={esa_length}"
                )

    if show_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return pd.DataFrame(rows)


def print_console_summary(summary_df: pd.DataFrame) -> None:
    for scenario, scenario_df in summary_df.groupby("scenario"):
        print(f"\n=== Scenario: {scenario} ===")
        pivot = scenario_df.pivot_table(
            index="length",
            columns="algorithm",
            values="time_mean_ms",
        ).sort_index()
        print("Mean runtime [ms]:")
        print(pivot.round(4).to_string())


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

    raw_df = run_benchmarks(
        lengths=lengths,
        cases_per_length=args.cases_per_length,
        runs=args.runs,
        seed=args.seed,
        show_progress=args.show_progress,
    )
    summary_df = aggregate_results(raw_df)

    raw_df.to_csv(output_dir / "raw_runs.csv", index=False)
    summary_df.to_csv(output_dir / "summary_stats.csv", index=False)
    export_latex_tables(summary_df, output_dir)
    export_plots(summary_df, output_dir)
    print_console_summary(summary_df)

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

    print(f"\nSaved raw data to: {output_dir / 'raw_runs.csv'}")
    print(f"Saved summary to: {output_dir / 'summary_stats.csv'}")
    print(f"Saved LaTeX tables in: {output_dir / 'tables'}")
    print(f"Saved plots in: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()