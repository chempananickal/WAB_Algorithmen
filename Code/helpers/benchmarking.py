from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import fields, is_dataclass
from datetime import timedelta
from typing import Any, Callable

import pandas as pd

from helpers.algorithms_esa import build_enhanced_suffix_array, query_enhanced_suffix_array
from helpers.algorithms_sam import build_suffix_automaton, query_suffix_automaton
from helpers.case_generation import generate_cases
from helpers.models import LCSResult


def measure_algorithm(
    build_fn: Callable[[str, str], Any],
    query_fn: Callable[[Any, str], LCSResult],
    s: str,
    t: str,
) -> tuple[float, float, float, float, float, float, LCSResult]:
    """Return runtime and memory metrics plus LCS result.

    Metrics are:
    - build runtime [ms]
    - query runtime [ms]
    - build phase peak memory [KiB]
    - query phase peak memory [KiB]
    - query extra peak over post-build baseline [KiB]
    - deep size of built index structure [KiB]
    """

    def deep_sizeof(obj: Any, seen: set[int] | None = None) -> int:
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                size += deep_sizeof(key, seen)
                size += deep_sizeof(value, seen)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += deep_sizeof(item, seen)
        elif is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                size += deep_sizeof(getattr(obj, field.name), seen)

        return size

    tracemalloc.start()

    build_start = time.perf_counter_ns()
    built = build_fn(s, t)
    build_end = time.perf_counter_ns()

    build_current_bytes, build_peak_bytes = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()

    query_start = time.perf_counter_ns()
    result = query_fn(built, t)
    query_end = time.perf_counter_ns()

    _, query_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    build_ms = (build_end - build_start) / 1_000_000
    query_ms = (query_end - query_start) / 1_000_000
    build_peak_kib = build_peak_bytes / 1024
    query_peak_kib = query_peak_bytes / 1024
    query_extra_kib = max(0.0, (query_peak_bytes - build_current_bytes) / 1024)
    index_size_kib = deep_sizeof(built) / 1024

    return (
        build_ms,
        query_ms,
        build_peak_kib,
        query_peak_kib,
        query_extra_kib,
        index_size_kib,
        result,
    )


def _format_seconds_as_timedelta(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, round(seconds)))))


def _render_progress(
    current: int,
    total: int,
    elapsed_seconds: float,
    eta_seconds: float,
    width: int = 30,
) -> str:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = generate_cases(lengths=lengths, cases_per_length=cases_per_length, seed=seed)

    algorithms: dict[str, tuple[Callable[[str, str], Any], Callable[[Any, str], LCSResult]]] = {
        "Suffix Automaton": (
            lambda s, _t: build_suffix_automaton(s),
            lambda sam, t: query_suffix_automaton(sam, t),
        ),
        "Enhanced Suffix Array": (
            build_enhanced_suffix_array,
            lambda esa, _t: query_enhanced_suffix_array(esa),
        ),
    }

    rows: list[dict[str, Any]] = []
    case_lcs_rows: list[dict[str, Any]] = []
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
            run_outcomes: dict[str, LCSResult] = {}

            for algorithm_name, (build_fn, query_fn) in algorithms.items():
                (
                    build_ms,
                    query_ms,
                    build_peak_kib,
                    query_peak_kib,
                    query_extra_kib,
                    index_size_kib,
                    lcs_result,
                ) = measure_algorithm(build_fn, query_fn, case.s, case.t)
                run_outcomes[algorithm_name] = lcs_result

                elapsed_seconds = (build_ms + query_ms) / 1000.0
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
                        "build_time_ms": build_ms,
                        "query_time_ms": query_ms,
                        "total_time_ms": build_ms + query_ms,
                        "peak_memory_kib": max(build_peak_kib, query_peak_kib),
                        "build_peak_memory_kib": build_peak_kib,
                        "query_peak_memory_kib": query_peak_kib,
                        "query_extra_memory_kib": query_extra_kib,
                        "index_size_kib": index_size_kib,
                        "lcs_length": lcs_result.length,
                    }
                )

                completed_steps += 1
                if show_progress:
                    elapsed_wall = max(0.0, time.perf_counter() - started_at)
                    eta_seconds = estimate_remaining_seconds(completed_steps)
                    sys.stdout.write(
                        "\r"
                        + _render_progress(
                            completed_steps, total_steps, elapsed_wall, eta_seconds
                        )
                    )
                    sys.stdout.flush()

            sam_result = run_outcomes["Suffix Automaton"]
            esa_result = run_outcomes["Enhanced Suffix Array"]
            if (
                sam_result.length != esa_result.length
                or sam_result.substrings != esa_result.substrings
            ):
                raise ValueError(
                    f"Correctness mismatch detected: case={case.case_id}, run={run_index}, "
                    f"SAM=({sam_result.length}, {sorted(sam_result.substrings)}), "
                    f"ESA=({esa_result.length}, {sorted(esa_result.substrings)})"
                )

            if run_index == 1:
                case_lcs_rows.append(
                    {
                        "case_id": case.case_id,
                        "scenario": case.scenario,
                        "length": case.length,
                        "s": case.s,
                        "t": case.t,
                        "lcs_length": sam_result.length,
                        "lcs_values": json.dumps(sorted(sam_result.substrings), ensure_ascii=False),
                    }
                )

    if show_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return pd.DataFrame(rows), pd.DataFrame(case_lcs_rows)


def print_console_summary(summary_df: pd.DataFrame) -> None:
    for scenario, scenario_df in summary_df.groupby("scenario"):
        print(f"\n=== Scenario: {scenario} ===")
        pivot = scenario_df.pivot_table(
            index="length",
            columns="algorithm",
            values="total_mean_ms",
        ).sort_index()
        print("Mean total runtime [ms]:")
        print(pivot.round(4).to_string())
