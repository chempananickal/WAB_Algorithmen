from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def export_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for scenario, scenario_df in summary_df.groupby("scenario"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for algorithm, algorithm_df in scenario_df.groupby("algorithm"):
            sorted_df = algorithm_df.sort_values("length")
            axes[0].errorbar(
                sorted_df["length"],
                sorted_df["build_mean_ms"],
                yerr=sorted_df["build_std_ms"],
                marker="o",
                capsize=3,
                label=algorithm,
            )
            axes[1].errorbar(
                sorted_df["length"],
                sorted_df["query_mean_ms"],
                yerr=sorted_df["query_std_ms"],
                marker="o",
                capsize=3,
                label=algorithm,
            )

        axes[0].set_title(f"Build time vs Length ({scenario})")
        axes[0].set_xlabel("String length")
        axes[0].set_ylabel("Mean build time [ms]")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Query time vs Length ({scenario})")
        axes[1].set_xlabel("String length")
        axes[1].set_ylabel("Mean query time [ms]")
        axes[1].grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        fig.savefig(plots_dir / f"mean_{scenario}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for algorithm, algorithm_df in scenario_df.groupby("algorithm"):
            sorted_df = algorithm_df.sort_values("length")
            axes[0].plot(
                sorted_df["length"],
                sorted_df["build_median_ms"],
                marker="o",
                label=algorithm,
            )
            axes[0].fill_between(
                sorted_df["length"],
                sorted_df["build_q1_ms"],
                sorted_df["build_q3_ms"],
                alpha=0.2,
            )
            axes[1].plot(
                sorted_df["length"],
                sorted_df["query_median_ms"],
                marker="o",
                label=algorithm,
            )
            axes[1].fill_between(
                sorted_df["length"],
                sorted_df["query_q1_ms"],
                sorted_df["query_q3_ms"],
                alpha=0.2,
            )

        axes[0].set_title(f"Build median + IQR ({scenario})")
        axes[0].set_xlabel("String length")
        axes[0].set_ylabel("Median build time [ms]")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Query median + IQR ({scenario})")
        axes[1].set_xlabel("String length")
        axes[1].set_ylabel("Median query time [ms]")
        axes[1].grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        fig.savefig(plots_dir / f"median_iqr_{scenario}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for algorithm, algorithm_df in scenario_df.groupby("algorithm"):
            sorted_df = algorithm_df.sort_values("length")
            axes[0].errorbar(
                sorted_df["length"],
                sorted_df["total_mean_ms"],
                yerr=sorted_df["total_std_ms"],
                marker="o",
                capsize=3,
                label=algorithm,
            )
            axes[1].plot(
                sorted_df["length"],
                sorted_df["total_median_ms"],
                marker="o",
                label=algorithm,
            )
            axes[1].fill_between(
                sorted_df["length"],
                sorted_df["total_q1_ms"],
                sorted_df["total_q3_ms"],
                alpha=0.2,
            )

        axes[0].set_title(f"Total time mean + SD ({scenario})")
        axes[0].set_xlabel("String length")
        axes[0].set_ylabel("Mean total time [ms]")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Total time median + IQR ({scenario})")
        axes[1].set_xlabel("String length")
        axes[1].set_ylabel("Median total time [ms]")
        axes[1].grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        fig.savefig(plots_dir / f"total_stats_{scenario}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for algorithm, algorithm_df in scenario_df.groupby("algorithm"):
            sorted_df = algorithm_df.sort_values("length")
            axes[0].errorbar(
                sorted_df["length"],
                sorted_df["memory_mean_kib"],
                yerr=sorted_df["memory_std_kib"],
                marker="o",
                capsize=3,
                label=algorithm,
            )
            axes[1].plot(
                sorted_df["length"],
                sorted_df["memory_median_kib"],
                marker="o",
                label=algorithm,
            )
            axes[1].fill_between(
                sorted_df["length"],
                sorted_df["memory_q1_kib"],
                sorted_df["memory_q3_kib"],
                alpha=0.2,
            )

        axes[0].set_title(f"Peak memory mean + SD ({scenario})")
        axes[0].set_xlabel("String length")
        axes[0].set_ylabel("Mean peak memory [KiB]")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Peak memory median + IQR ({scenario})")
        axes[1].set_xlabel("String length")
        axes[1].set_ylabel("Median peak memory [KiB]")
        axes[1].grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        fig.savefig(plots_dir / f"memory_stats_{scenario}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for algorithm, algorithm_df in scenario_df.groupby("algorithm"):
            sorted_df = algorithm_df.sort_values("length")
            axes[0].errorbar(
                sorted_df["length"],
                sorted_df["index_size_mean_kib"],
                yerr=sorted_df["index_size_std_kib"],
                marker="o",
                capsize=3,
                label=algorithm,
            )
            axes[1].errorbar(
                sorted_df["length"],
                sorted_df["build_peak_memory_mean_kib"],
                yerr=sorted_df["build_peak_memory_std_kib"],
                marker="o",
                capsize=3,
                label=f"{algorithm} (build peak)",
            )
            axes[1].errorbar(
                sorted_df["length"],
                sorted_df["query_extra_memory_mean_kib"],
                yerr=sorted_df["query_extra_memory_std_kib"],
                marker="s",
                capsize=3,
                label=f"{algorithm} (query extra)",
            )

        axes[0].set_title(f"Index size vs Length ({scenario})")
        axes[0].set_xlabel("String length")
        axes[0].set_ylabel("Mean index size [KiB]")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Build/query memory vs Length ({scenario})")
        axes[1].set_xlabel("String length")
        axes[1].set_ylabel("Mean memory [KiB]")
        axes[1].grid(True, alpha=0.3)

        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        fig.savefig(plots_dir / f"space_practical_{scenario}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
