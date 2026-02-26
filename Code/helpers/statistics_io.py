from __future__ import annotations

from pathlib import Path

import pandas as pd


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def q1(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def q3(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["scenario", "length", "algorithm"], as_index=False).agg(
        build_mean_ms=("build_time_ms", "mean"),
        build_median_ms=("build_time_ms", "median"),
        build_std_ms=("build_time_ms", "std"),
        build_iqr_ms=("build_time_ms", iqr),
        build_q1_ms=("build_time_ms", q1),
        build_q3_ms=("build_time_ms", q3),
        build_min_ms=("build_time_ms", "min"),
        build_max_ms=("build_time_ms", "max"),
        query_mean_ms=("query_time_ms", "mean"),
        query_median_ms=("query_time_ms", "median"),
        query_std_ms=("query_time_ms", "std"),
        query_iqr_ms=("query_time_ms", iqr),
        query_q1_ms=("query_time_ms", q1),
        query_q3_ms=("query_time_ms", q3),
        query_min_ms=("query_time_ms", "min"),
        query_max_ms=("query_time_ms", "max"),
        total_mean_ms=("total_time_ms", "mean"),
        total_median_ms=("total_time_ms", "median"),
        total_std_ms=("total_time_ms", "std"),
        total_iqr_ms=("total_time_ms", iqr),
        total_q1_ms=("total_time_ms", q1),
        total_q3_ms=("total_time_ms", q3),
        memory_mean_kib=("peak_memory_kib", "mean"),
        memory_median_kib=("peak_memory_kib", "median"),
        memory_std_kib=("peak_memory_kib", "std"),
        memory_iqr_kib=("peak_memory_kib", iqr),
        memory_q1_kib=("peak_memory_kib", q1),
        memory_q3_kib=("peak_memory_kib", q3),
        memory_min_kib=("peak_memory_kib", "min"),
        memory_max_kib=("peak_memory_kib", "max"),
        build_peak_memory_mean_kib=("build_peak_memory_kib", "mean"),
        build_peak_memory_median_kib=("build_peak_memory_kib", "median"),
        build_peak_memory_std_kib=("build_peak_memory_kib", "std"),
        build_peak_memory_iqr_kib=("build_peak_memory_kib", iqr),
        build_peak_memory_q1_kib=("build_peak_memory_kib", q1),
        build_peak_memory_q3_kib=("build_peak_memory_kib", q3),
        query_peak_memory_mean_kib=("query_peak_memory_kib", "mean"),
        query_peak_memory_median_kib=("query_peak_memory_kib", "median"),
        query_peak_memory_std_kib=("query_peak_memory_kib", "std"),
        query_peak_memory_iqr_kib=("query_peak_memory_kib", iqr),
        query_peak_memory_q1_kib=("query_peak_memory_kib", q1),
        query_peak_memory_q3_kib=("query_peak_memory_kib", q3),
        query_extra_memory_mean_kib=("query_extra_memory_kib", "mean"),
        query_extra_memory_median_kib=("query_extra_memory_kib", "median"),
        query_extra_memory_std_kib=("query_extra_memory_kib", "std"),
        query_extra_memory_iqr_kib=("query_extra_memory_kib", iqr),
        query_extra_memory_q1_kib=("query_extra_memory_kib", q1),
        query_extra_memory_q3_kib=("query_extra_memory_kib", q3),
        index_size_mean_kib=("index_size_kib", "mean"),
        index_size_median_kib=("index_size_kib", "median"),
        index_size_std_kib=("index_size_kib", "std"),
        index_size_iqr_kib=("index_size_kib", iqr),
        index_size_q1_kib=("index_size_kib", q1),
        index_size_q3_kib=("index_size_kib", q3),
        lcs_mean_length=("lcs_length", "mean"),
        lcs_median_length=("lcs_length", "median"),
        runs=("run", "count"),
    )
    return grouped.sort_values(by=["scenario", "length", "algorithm"]).reset_index(drop=True)


def export_latex_tables(summary_df: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    # Split tables into smaller, sensible groups so they fit in LaTeX documents.
    # Further subdivide into narrow tables (build/query/total times, memory groups, index).
    groups = {
        "build_time": (
            [
                "length",
                "build_mean_ms",
                "build_std_ms",
                "build_median_ms",
                "build_iqr_ms",
                "runs",
            ],
            [
                "Length",
                "Build Mean (ms)",
                "Build SD (ms)",
                "Build Median (ms)",
                "Build IQR (ms)",
                "Runs",
            ],
        ),
        "query_time": (
            [
                "length",
                "query_mean_ms",
                "query_std_ms",
                "query_median_ms",
                "query_iqr_ms",
                "runs",
            ],
            [
                "Length",
                "Query Mean (ms)",
                "Query SD (ms)",
                "Query Median (ms)",
                "Query IQR (ms)",
                "Runs",
            ],
        ),
        "total_time": (
            [
                "length",
                "total_mean_ms",
                "total_std_ms",
                "total_median_ms",
                "total_iqr_ms",
                "runs",
            ],
            [
                "Length",
                "Total Mean (ms)",
                "Total SD (ms)",
                "Total Median (ms)",
                "Total IQR (ms)",
                "Runs",
            ],
        ),
        "memory_overall": (
            [
                "length",
                "memory_mean_kib",
                "memory_std_kib",
                "memory_median_kib",
                "memory_iqr_kib",
            ],
            [
                "Length",
                "Memory Mean (KiB)",
                "Memory SD (KiB)",
                "Memory Median (KiB)",
                "Memory IQR (KiB)",
            ],
        ),
        "build_peak_memory": (
            [
                "length",
                "build_peak_memory_mean_kib",
                "build_peak_memory_std_kib",
                "build_peak_memory_median_kib",
                "build_peak_memory_iqr_kib",
            ],
            [
                "Length",
                "BPM Mean (KiB)",
                "BPM SD (KiB)",
                "BPM Median (KiB)",
                "BPM IQR (KiB)",
            ],
        ),
        "query_peak_memory": (
            [
                "length",
                "query_peak_memory_mean_kib",
                "query_peak_memory_std_kib",
                "query_peak_memory_median_kib",
                "query_peak_memory_iqr_kib",
            ],
            [
                "Length",
                "QPM Mean (KiB)",
                "QPM SD (KiB)",
                "QPM Median (KiB)",
                "QPM IQR (KiB)",
            ],
        ),
        "query_extra_memory": (
            [
                "length",
                "query_extra_memory_mean_kib",
                "query_extra_memory_std_kib",
                "query_extra_memory_median_kib",
                "query_extra_memory_iqr_kib",
            ],
            [
                "Length",
                "QEM Mean (KiB)",
                "QEM SD (KiB)",
                "QEM Median (KiB)",
                "QEM IQR (KiB)",
            ],
        ),
        "index": (
            [
                "length",
                "index_size_mean_kib",
                "index_size_std_kib",
                "index_size_median_kib",
                "index_size_iqr_kib",
            ],
            [
                "Length",
                "IDX Mean (KiB)",
                "IDX SD (KiB)",
                "IDX Median (KiB)",
                "IDX IQR (KiB)",
            ],
        ),
    }

    for scenario, scenario_df in summary_df.groupby("scenario"):
        # Build one combined file per scenario containing all groups (times, memory, index).
        per_scenario_parts: list[str] = []
        for group_name, (columns, col_titles) in groups.items():
            subtables: list[str] = []
            for algorithm, algo_df in scenario_df.groupby("algorithm"):
                compact = algo_df[columns].copy()
                compact.columns = col_titles
                safe_algo = algorithm.replace(" ", "_").lower()
                tabular = compact.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
                alg_esc = algorithm.replace("_", "\\_")
                scen_esc = scenario.replace("_", "\\_")
                title = f"{alg_esc} ({group_name.replace('_', ' ').title()})"
                description = f"Condensed results for {alg_esc} in scenario {scen_esc}."
                latex = (
                    "\\begin{table}[H]\\centering\\small\n"
                    + f"\\textbf{{{title}}}\\\\\n"
                    + f"\\textit{{{description}}}\\\\[4pt]\n"
                    + tabular
                    + "\\end{table}"
                )
                subtables.append(latex)

            group_full = "\n\n".join(subtables)
            # Add a small LaTeX comment as a separator so it's easy to split/input later.
            per_scenario_parts.append(f"% ===== group: {group_name.replace('_', ' ').title()} =====\n" + group_full)

        combined = "\n\n".join(per_scenario_parts)
        (tables_dir / f"summary_{scenario}.tex").write_text(combined, encoding="utf-8")


def load_raw_results(output_dir: Path) -> pd.DataFrame:
    raw_path = output_dir / "raw_runs.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw results at {raw_path}. Run with --mode run or --mode both first."
        )
    return pd.read_csv(raw_path)
