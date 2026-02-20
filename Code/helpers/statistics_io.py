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
        query_peak_memory_mean_kib=("query_peak_memory_kib", "mean"),
        query_peak_memory_median_kib=("query_peak_memory_kib", "median"),
        query_peak_memory_std_kib=("query_peak_memory_kib", "std"),
        query_peak_memory_iqr_kib=("query_peak_memory_kib", iqr),
        query_extra_memory_mean_kib=("query_extra_memory_kib", "mean"),
        query_extra_memory_median_kib=("query_extra_memory_kib", "median"),
        query_extra_memory_std_kib=("query_extra_memory_kib", "std"),
        query_extra_memory_iqr_kib=("query_extra_memory_kib", iqr),
        index_size_mean_kib=("index_size_kib", "mean"),
        index_size_median_kib=("index_size_kib", "median"),
        index_size_std_kib=("index_size_kib", "std"),
        index_size_iqr_kib=("index_size_kib", iqr),
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
                "build_mean_ms",
                "build_median_ms",
                "build_std_ms",
                "build_iqr_ms",
                "query_mean_ms",
                "query_median_ms",
                "query_std_ms",
                "query_iqr_ms",
                "total_mean_ms",
                "total_median_ms",
                "total_std_ms",
                "total_iqr_ms",
                "total_q1_ms",
                "total_q3_ms",
                "memory_mean_kib",
                "memory_median_kib",
                "memory_std_kib",
                "memory_iqr_kib",
                "memory_q1_kib",
                "memory_q3_kib",
                "build_peak_memory_mean_kib",
                "query_peak_memory_mean_kib",
                "query_extra_memory_mean_kib",
                "index_size_mean_kib",
                "runs",
            ]
        ].copy()

        latex = compact.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
        (tables_dir / f"summary_{scenario}.tex").write_text(latex, encoding="utf-8")


def load_raw_results(output_dir: Path) -> pd.DataFrame:
    raw_path = output_dir / "raw_runs.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw results at {raw_path}. Run with --mode run or --mode both first."
        )
    return pd.read_csv(raw_path)
