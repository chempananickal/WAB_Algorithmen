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
    """Export clean tabular tables. LaTeX handles formatting via custom commands."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    groups = {
        "build_time": (
            ["length", "build_mean_ms", "build_std_ms", "build_median_ms", "build_iqr_ms"],
            ["Length", "Build Time Mean (ms)", "Build Time SD (ms)", "Build Time Median (ms)", "Build Time IQR (ms)"],
        ),
        "query_time": (
            ["length", "query_mean_ms", "query_std_ms", "query_median_ms", "query_iqr_ms"],
            ["Length", "Query Time Mean (ms)", "Query Time SD (ms)", "Query Time Median (ms)", "Query Time IQR (ms)"],
        ),
        "total_time": (
            ["length", "total_mean_ms", "total_std_ms", "total_median_ms", "total_iqr_ms"],
            ["Length", "Total Time Mean (ms)", "Total Time SD (ms)", "Total Time Median (ms)", "Total Time IQR (ms)"],
        ),
        "build_peak_memory": (
            ["length", "build_peak_memory_mean_kib", "build_peak_memory_std_kib", "build_peak_memory_median_kib", "build_peak_memory_iqr_kib"],
            ["Length", "Build Peak Memory Mean (KiB)", "Build Peak Memory SD (KiB)", "Build Peak Memory Median (KiB)", "Build Peak Memory IQR (KiB)"],
        ),
        "query_peak_memory": (
            ["length", "query_peak_memory_mean_kib", "query_peak_memory_std_kib", "query_peak_memory_median_kib", "query_peak_memory_iqr_kib"],
            ["Length", "Query Peak Memory Mean (KiB)", "Query Peak Memory SD (KiB)", "Query Peak Memory Median (KiB)", "Query Peak Memory IQR (KiB)"],
        ),
        "query_extra_memory": (
            ["length", "query_extra_memory_mean_kib", "query_extra_memory_std_kib", "query_extra_memory_median_kib", "query_extra_memory_iqr_kib"],
            ["Length", "Query Extra Memory Mean (KiB)", "Query Extra Memory SD (KiB)", "Query Extra Memory Median (KiB)", "Query Extra Memory IQR (KiB)"],
        ),
        "index_size": (
            ["length", "index_size_mean_kib", "index_size_std_kib", "index_size_median_kib", "index_size_iqr_kib"],
            ["Length", "Index Size Mean (KiB)", "Index Size SD (KiB)", "Index Size Median (KiB)", "Index Size IQR (KiB)"],
        ),
    }

    for scenario, scenario_df in summary_df.groupby("scenario"):
        per_scenario_parts: list[str] = []
        
        for group_name, (columns, col_titles) in groups.items():
            # Combine both algorithms into one table for direct comparison
            # Add algorithm column for clarity
            compact = scenario_df[["algorithm"] + columns].copy()
            compact.columns = ["Algorithm"] + col_titles
            
            # Abbreviate algorithm names to save space
            compact["Algorithm"] = compact["Algorithm"].replace({
                "Enhanced Suffix Array": "ESA",
                "Suffix Automaton": "SAM"
            })
            
            ncols = len(compact.columns)
            
            # Sort by length then algorithm for easy comparison
            compact = compact.sort_values(by=["Length", "Algorithm"]).reset_index(drop=True)
            
            # Export table with pandas, then convert to tabularx
            table_tex = compact.to_latex(
                index=False,
                float_format=lambda x: f"{x:.4f}",
                escape=False
            )
            
            # Convert tabular to tabularx with column dividers
            import re
            col_spec = '|' + '|'.join(['X'] * ncols) + '|'
            table_tex = re.sub(
                r'\\begin\{tabular\}\{[^}]+\}',
                r'\\begin{tabularx}{\\textwidth}{' + col_spec + r'}',
                table_tex
            )
            table_tex = table_tex.replace(r'\end{tabular}', r'\end{tabularx}')
            
            # Add horizontal line after each SAM row (but not before \bottomrule)
            table_tex = re.sub(
                r'(SAM & .+? \\\\)\n(?!\\bottomrule)',
                r'\1\n\\hline\n',
                table_tex
            )
            
            # Metadata
            scen_esc = scenario.replace("_", "\\_")
            title = f"{group_name.replace('_', ' ').title()} Comparison"
            
            # Useful description: sample size + interpretation hint
            is_timing = "time" in group_name
            is_memory = "memory" in group_name or "index" in group_name
            if is_timing:
                description = "Based on 500 benchmark runs per input length. Times in milliseconds; lower is better."
            elif is_memory:
                description = "Based on 500 benchmark runs per input length. Memory in KiB; lower is better."
            else:
                description = "Based on 500 benchmark runs per input length."
            
            # Use custom LaTeX environment (defined in settings.tex)
            latex = (
                f"\\begin{{benchmarktable}}{{{title}}}{{{description}}}\n"
                + table_tex
                + "\\end{benchmarktable}"
            )
            
            per_scenario_parts.append(f"% ===== group: {group_name.replace('_', ' ').title()} =====\n" + latex)

        combined = "\n\n".join(per_scenario_parts)
        (tables_dir / f"summary_{scenario}.tex").write_text(combined, encoding="utf-8")


def load_raw_results(output_dir: Path) -> pd.DataFrame:
    raw_path = output_dir / "raw_runs.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw results at {raw_path}. Run with --mode run or --mode both first."
        )
    return pd.read_csv(raw_path)
