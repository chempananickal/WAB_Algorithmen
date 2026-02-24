from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def export_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_metric_by_scenario(summary_df, metric, yerr=None, title=None, ylabel=None, filename=None, kind="errorbar", fill_between=None):
        import math
        scenarios = list(summary_df['scenario'].unique())
        n = len(scenarios)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        axes = axes.flatten() if n > 1 else [axes]
        legend_handles, legend_labels = None, None
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            scenario_df = summary_df[summary_df['scenario'] == scenario]
            for algorithm, algorithm_df in scenario_df.groupby("algorithm"):
                sorted_df = algorithm_df.sort_values("length")
                label = f"{algorithm}"
                if kind == "errorbar":
                    line = ax.errorbar(
                        sorted_df["length"],
                        sorted_df[metric],
                        yerr=sorted_df[yerr] if yerr else None,
                        marker="o",
                        capsize=3,
                        label=label,
                    )
                elif kind == "plot":
                    line = ax.plot(
                        sorted_df["length"],
                        sorted_df[metric],
                        marker="o",
                        label=label,
                    )
                    if fill_between:
                        ax.fill_between(
                            sorted_df["length"],
                            sorted_df[fill_between[0]],
                            sorted_df[fill_between[1]],
                            alpha=0.2,
                        )
            ax.set_title(f"{scenario}")
            ax.set_xlabel("String length")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            # Ensure y-axis tick labels are visible for all subplots
            ax.yaxis.set_tick_params(labelleft=True)
            # Save legend handles/labels from the first subplot with data
            if legend_handles is None or legend_labels is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        # Hide unused subplots and use the last vacant one for the legend
        last_used = i
        for j in range(i+1, nrows*ncols):
            ax = axes[j]
            ax.axis('off')
        # Place legend in the last vacant subplot, or in the last subplot if all are used
        legend_ax = axes[last_used+1] if (last_used+1) < len(axes) else axes[-1]
        legend_ax.axis('off')
        legend_ax.legend(legend_handles, legend_labels, loc='center')
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(plots_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Build time mean + SD
    # plot_metric_by_scenario(
    #     summary_df,
    #     metric="build_mean_ms",
    #     yerr="build_std_ms",
    #     title="Build time mean + SD",
    #     ylabel="Mean build time [ms]",
    #     filename="build_time_mean_sd.png",
    #     kind="errorbar"
    # )

    # Build time median + IQR
    plot_metric_by_scenario(
        summary_df,
        metric="build_median_ms",
        title="Build time median + IQR",
        ylabel="Median build time [ms]",
        filename="build_time_median_iqr.png",
        kind="plot",
        fill_between=("build_q1_ms", "build_q3_ms")
    )

    # Query time mean + SD
    # plot_metric_by_scenario(
    #     summary_df,
    #     metric="query_mean_ms",
    #     yerr="query_std_ms",
    #     title="Query time mean + SD",
    #     ylabel="Mean query time [ms]",
    #     filename="query_time_mean_sd.png",
    #     kind="errorbar"
    # )

    # Query time median + IQR
    plot_metric_by_scenario(
        summary_df,
        metric="query_median_ms",
        title="Query time median + IQR",
        ylabel="Median query time [ms]",
        filename="query_time_median_iqr.png",
        kind="plot",
        fill_between=("query_q1_ms", "query_q3_ms")
    )

    # Total time mean + SD
    # plot_metric_by_scenario(
    #     summary_df,
    #     metric="total_mean_ms",
    #     yerr="total_std_ms",
    #     title="Total time mean + SD",
    #     ylabel="Mean total time [ms]",
    #     filename="total_time_mean_sd.png",
    #     kind="errorbar"
    # )

    # Total time median + IQR
    plot_metric_by_scenario(
        summary_df,
        metric="total_median_ms",
        title="Total time median + IQR",
        ylabel="Median total time [ms]",
        filename="total_time_median_iqr.png",
        kind="plot",
        fill_between=("total_q1_ms", "total_q3_ms")
    )

    # Peak memory mean + SD
    # plot_metric_by_scenario(
    #     summary_df,
    #     metric="memory_mean_kib",
    #     yerr="memory_std_kib",
    #     title="Peak memory mean + SD",
    #     ylabel="Mean peak memory [KiB]",
    #     filename="peak_memory_mean_sd.png",
    #     kind="errorbar"
    # )

    # Peak memory median + IQR
    plot_metric_by_scenario(
        summary_df,
        metric="memory_median_kib",
        title="Peak memory median + IQR",
        ylabel="Median peak memory [KiB]",
        filename="peak_memory_median_iqr.png",
        kind="plot",
        fill_between=("memory_q1_kib", "memory_q3_kib")
    )

    # Index size mean + SD
    # plot_metric_by_scenario(
    #     summary_df,
    #     metric="index_size_mean_kib",
    #     yerr="index_size_std_kib",
    #     title="Index size mean + SD",
    #     ylabel="Mean index size [KiB]",
    #     filename="index_size_mean_sd.png",
    #     kind="errorbar"
    # )
    
    # Index size median + IQR
    plot_metric_by_scenario(
        summary_df,
        metric="index_size_median_kib",
        title="Index size median + IQR",
        ylabel="Median index size [KiB]",
        filename="index_size_median_iqr.png",
        kind="plot",
        fill_between=("index_size_q1_kib", "index_size_q3_kib")
    )

    # Build peak memory mean + SD
    # plot_metric_by_scenario(
    #     summary_df,
    #     metric="build_peak_memory_mean_kib",
    #     yerr="build_peak_memory_std_kib",
    #     title="Build peak memory mean + SD",
    #     ylabel="Mean build peak memory [KiB]",
    #     filename="build_peak_memory_mean_sd.png",
    #     kind="errorbar"
    # )

    # Build peak memory median + IQR
    plot_metric_by_scenario(
        summary_df,
        metric="build_peak_memory_median_kib",
        title="Build peak memory median + IQR",
        ylabel="Median build peak memory [KiB]",
        filename="build_peak_memory_median_iqr.png",
        kind="plot",
        fill_between=("build_peak_memory_iqr_kib", "build_peak_memory_iqr_kib")
    )

    # Query extra memory mean + SD
    # plot_metric_by_scenario(
    #     summary_df,
    #     metric="query_extra_memory_mean_kib",
    #     yerr="query_extra_memory_std_kib",
    #     title="Query extra memory mean + SD",
    #     ylabel="Mean query extra memory [KiB]",
    #     filename="query_extra_memory_mean_sd.png",
    #     kind="errorbar"
    # )

    # Query extra memory median + IQR
    plot_metric_by_scenario(
        summary_df,
        metric="query_extra_memory_median_kib",
        title="Query extra memory median + IQR",
        ylabel="Median query extra memory [KiB]",
        filename="query_extra_memory_median_iqr.png",
        kind="plot",
        fill_between=("query_extra_memory_iqr_kib", "query_extra_memory_iqr_kib")
    )
