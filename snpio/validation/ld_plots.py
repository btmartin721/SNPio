"""Static plots for SNPio linkage-disequilibrium validation results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

_ALLOWED_FORMATS: Final[frozenset[str]] = frozenset({"pdf", "png", "svg"})
_STATISTIC_ORDER: Final[tuple[str, ...]] = ("D", "D2", "Dz", "pi2")


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, plot_name: str
) -> None:
    """Fail early when a validation table cannot support a requested plot."""

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{plot_name} is missing columns: {sorted(missing)}")


def _save_figure(
    figure: Figure,
    output_directory: str | Path,
    stem: str,
    *,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, Path]:
    """Save one figure in each requested format and close it."""

    normalized = tuple(dict.fromkeys(str(value).lower() for value in formats))
    invalid = set(normalized).difference(_ALLOWED_FORMATS)
    if invalid:
        raise ValueError(f"Unsupported plot formats: {sorted(invalid)}")
    if not normalized:
        raise ValueError("At least one plot format is required.")
    if dpi < 72:
        raise ValueError("dpi must be at least 72.")

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    try:
        for plot_format in normalized:
            path = directory / f"{stem}.{plot_format}"
            save_kwargs = {"bbox_inches": "tight"}

            if plot_format == "png":
                save_kwargs["dpi"] = dpi

            figure.savefig(path, **save_kwargs)
            files[plot_format] = path
    finally:
        plt.close(figure)
    return files


def _positive_floor(values: np.ndarray) -> float:
    """Return a plotting floor below the smallest positive finite value."""

    finite = values[np.isfinite(values) & (values > 0.0)]
    return float(finite.min() / 10.0) if finite.size else 1e-18


def plot_exact_expectation_errors(
    results: pd.DataFrame,
    output_directory: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, Path]:
    """Plot maximum exact-enumeration error by statistic and sample size."""

    _require_columns(
        results,
        {"Statistic", "Sample_Size", "Absolute_Error", "Passed"},
        plot_name="Exact-expectation results",
    )
    grouped = (
        results.groupby(["Statistic", "Sample_Size"], observed=True)["Absolute_Error"]
        .max()
        .reset_index()
    )
    sample_sizes = sorted(grouped["Sample_Size"].unique())
    statistic_order = [
        statistic
        for statistic in _STATISTIC_ORDER
        if statistic in set(grouped["Statistic"])
    ]
    all_values = grouped["Absolute_Error"].to_numpy(dtype=float)
    tolerance = float(results["Tolerance"].max()) if "Tolerance" in results else 1e-12
    floor = _positive_floor(np.append(all_values, tolerance))

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(statistic_order), dtype=float)
    offsets = np.linspace(-0.18, 0.18, max(1, len(sample_sizes)))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.7, max(1, len(sample_sizes))))
    for offset, color, sample_size in zip(offsets, colors, sample_sizes):
        subset = grouped.loc[grouped["Sample_Size"] == sample_size].set_index(
            "Statistic"
        )
        values = np.asarray(
            [subset.loc[statistic, "Absolute_Error"] for statistic in statistic_order],
            dtype=float,
        )
        axis.scatter(
            x + offset,
            np.maximum(values, floor),
            s=55,
            color=color,
            label=f"n = {sample_size}",
            zorder=3,
        )
    axis.axhline(
        tolerance,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=f"Tolerance = {tolerance:.0e}",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, statistic_order)
    axis.set_ylabel("Maximum absolute error")
    axis.set_xlabel("LD statistic")
    axis.set_title("Exact multinomial expectation validation")
    axis.legend(
        frameon=False,
        loc="center right",
        ncol=min(3, len(sample_sizes) + 1),
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        output_directory,
        "exact_expectation_errors",
        formats=formats,
        dpi=dpi,
    )


def plot_golden_reference_errors(
    results: pd.DataFrame,
    output_directory: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, Path]:
    """Plot maximum SNPio-versus-reference errors against allowed tolerances."""

    _require_columns(
        results,
        {"Statistic", "Absolute_Error", "Tolerance", "Passed"},
        plot_name="Golden-reference results",
    )
    grouped = (
        results.groupby("Statistic", observed=True)
        .agg(
            Maximum_Error=("Absolute_Error", "max"),
            Maximum_Tolerance=("Tolerance", "max"),
            Cases=("Passed", "size"),
            Passed=("Passed", "sum"),
        )
        .reindex(_STATISTIC_ORDER)
        .dropna(how="all")
    )
    values = grouped[["Maximum_Error", "Maximum_Tolerance"]].to_numpy(dtype=float)
    floor = _positive_floor(values.ravel())
    x = np.arange(len(grouped), dtype=float)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.scatter(
        x - 0.08,
        np.maximum(grouped["Maximum_Error"], floor),
        color=plt.get_cmap("tab10")(0),
        marker="o",
        s=65,
        label="Maximum observed error",
        zorder=3,
    )
    axis.scatter(
        x + 0.08,
        np.maximum(grouped["Maximum_Tolerance"], floor),
        color=plt.get_cmap("tab10")(1),
        marker="_",
        s=220,
        linewidth=2.2,
        label="Maximum allowed tolerance",
        zorder=3,
    )
    for index, row in enumerate(grouped.itertuples()):
        axis.vlines(
            index,
            max(row.Maximum_Error, floor),
            max(row.Maximum_Tolerance, floor),
            color="0.75",
            linewidth=1.0,
            zorder=1,
        )
        axis.annotate(
            f"{int(row.Passed)}/{int(row.Cases)}",
            (index, max(row.Maximum_Tolerance, floor)),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, grouped.index)
    axis.set_ylabel("Absolute error or tolerance")
    axis.set_xlabel("LD statistic")
    axis.set_title("moments-popgen 1.6.0 golden-reference validation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_ylim(
        max(floor * 0.6, np.nanmin(values) * 0.6),
        np.nanmax(values) * 3.0,
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        output_directory,
        "golden_reference_errors",
        formats=formats,
        dpi=dpi,
    )


def plot_published_island_fox_comparison(
    comparison: pd.DataFrame,
    output_directory: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, Path]:
    """Plot SNPio and published island-fox effective-size intervals."""

    required = {
        "Population",
        "Published_Ne",
        "Published_Ne_CI_Lower",
        "Published_Ne_CI_Upper",
        "Ne",
        "Ne_CI_Lower",
        "Ne_CI_Upper",
    }
    _require_columns(comparison, required, plot_name="Published island-fox comparison")
    frame = comparison.reset_index(drop=True)
    y = np.arange(len(frame), dtype=float)

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    published_error = np.vstack(
        [
            frame["Published_Ne"] - frame["Published_Ne_CI_Lower"],
            frame["Published_Ne_CI_Upper"] - frame["Published_Ne"],
        ]
    )
    snpio_error = np.vstack(
        [
            frame["Ne"] - frame["Ne_CI_Lower"],
            frame["Ne_CI_Upper"] - frame["Ne"],
        ]
    )
    axis.errorbar(
        frame["Published_Ne"],
        y + 0.12,
        xerr=published_error,
        fmt="o",
        capsize=3,
        color=plt.get_cmap("tab10")(0),
        label="Published (90% CI)",
    )
    axis.errorbar(
        frame["Ne"],
        y - 0.12,
        xerr=snpio_error,
        fmt="D",
        capsize=3,
        color=plt.get_cmap("tab10")(1),
        label="SNPio (90% CI)",
    )
    axis.set_yticks(y, frame["Population"])
    axis.invert_yaxis()
    axis.set_xlabel(r"Recent effective population size ($N_e$)")
    axis.set_title("Published island-fox benchmark")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save_figure(
        figure,
        output_directory,
        "published_island_fox_comparison",
        formats=formats,
        dpi=dpi,
    )


def plot_simulation_calibration(
    summary: pd.DataFrame,
    output_directory: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, Path]:
    """Plot forward-time bias, effective-size recovery, coverage, and rDz null."""

    required = {
        "Population_Size",
        "Sample_Size",
        "Relative_r2D_Bias",
        "Pooled_Ne",
        "Matched_Census_r2D_CI_Coverage",
        "Mean_rDz",
        "SE_rDz",
    }
    _require_columns(summary, required, plot_name="Simulation calibration summary")
    sample_sizes = sorted(summary["Sample_Size"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, max(1, len(sample_sizes))))
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), sharex="col")
    has_census = {
        "Census_Relative_r2D_Bias",
        "Census_Pooled_Ne",
    }.issubset(summary.columns)
    has_census_rdz = {"Census_rDz", "Census_SE_rDz"}.issubset(summary.columns)
    has_census_ne_coverage = "Census_Ne_CI_Coverage" in summary.columns
    tolerance_values = (
        summary["Model_r2D_Relative_Bias_Tolerance"].dropna().unique()
        if "Model_r2D_Relative_Bias_Tolerance" in summary.columns
        else np.asarray([])
    )
    tolerance = None
    if tolerance_values.size:
        tolerance = float(tolerance_values[0])
        axes[0, 0].axhspan(
            -tolerance,
            tolerance,
            color="0.85",
            alpha=0.5,
        )

    for color, sample_size in zip(colors, sample_sizes):
        subset = summary.loc[summary["Sample_Size"] == sample_size].sort_values(
            "Population_Size"
        )
        population_size = subset["Population_Size"].to_numpy(dtype=float)
        axes[0, 0].plot(
            population_size,
            subset["Relative_r2D_Bias"],
            marker="o",
            color=color,
            label="_nolegend_",
        )
        axes[0, 1].plot(
            population_size,
            subset["Pooled_Ne"],
            marker="o",
            color=color,
        )
        if has_census:
            axes[0, 0].plot(
                population_size,
                subset["Census_Relative_r2D_Bias"],
                marker="x",
                linestyle=":",
                color=color,
                label="_nolegend_",
            )
            axes[0, 1].plot(
                population_size,
                subset["Census_Pooled_Ne"],
                marker="x",
                linestyle=":",
                color=color,
            )
        axes[1, 0].plot(
            population_size,
            subset["Matched_Census_r2D_CI_Coverage"],
            marker="o",
            color=color,
            label=(
                r"matched-census $r_D^2$"
                if sample_size == sample_sizes[0]
                else "_nolegend_"
            ),
        )
        if has_census_ne_coverage:
            axes[1, 0].plot(
                population_size,
                subset["Census_Ne_CI_Coverage"],
                marker="x",
                linestyle=":",
                color=color,
                label=(
                    r"fixed census $N$ diagnostic"
                    if sample_size == sample_sizes[0]
                    else "_nolegend_"
                ),
            )
        axes[1, 1].errorbar(
            population_size,
            subset["Mean_rDz"],
            yerr=subset["SE_rDz"].fillna(0.0),
            marker="o",
            capsize=3,
            color=color,
        )
        if has_census_rdz:
            axes[1, 1].errorbar(
                population_size,
                subset["Census_rDz"],
                yerr=subset["Census_SE_rDz"].fillna(0.0),
                marker="x",
                linestyle=":",
                capsize=3,
                color=color,
            )

    population_values = summary["Population_Size"].to_numpy(dtype=float)

    positive_sizes = np.unique(population_values[population_values > 0.0])

    if positive_sizes.size:
        axes[0, 1].plot(
            positive_sizes,
            positive_sizes,
            linestyle="--",
            color="0.35",
            linewidth=1.2,
        )

    axes[0, 0].axhline(0.0, linestyle="--", color="0.35", linewidth=1.2)
    axes[1, 0].axhline(0.95, linestyle="--", color="0.35", linewidth=1.2)
    axes[1, 1].axhline(0.0, linestyle="--", color="0.35", linewidth=1.2)

    axes[0, 0].set_ylabel(r"Relative bias in $r_D^2$")
    axes[0, 0].set_title(r"$r_D^2$ practical-bias calibration")
    axes[0, 1].set_ylabel(r"Pooled estimated $N_e$")
    axes[0, 1].set_title(r"$N_e$ recovery")
    axes[1, 0].set_ylabel("95% CI coverage")

    coverage_is_formal = bool(
        "Coverage_Checked" in summary.columns and summary["Coverage_Checked"].any()
    )

    axes[1, 0].set_title(
        r"Matched-census $r_D^2$ coverage"
        + (" (formal)" if coverage_is_formal else " (diagnostic)")
    )

    axes[1, 1].set_ylabel(r"Mean $r_{Dz}$ ± SE")
    axes[1, 1].set_title(r"Sample and census $r_{Dz}=0$ null")

    for axis in axes.ravel():
        axis.set_xscale("linear")
        axis.set_xticks(positive_sizes)
        axis.set_xticklabels([f"{value:g}" for value in positive_sizes])
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.grid(alpha=0.25)

    axes[1, 0].set_xlabel(r"Simulated $N_e$")
    axes[1, 1].set_xlabel(r"Simulated $N_e$")
    axes[1, 0].legend(frameon=False, fontsize="small")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linestyle="none",
            label=f"n = {sample_size}",
        )
        for color, sample_size in zip(colors, sample_sizes)
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="0.3",
                marker="o",
                linestyle="-",
                label="sample",
            ),
            Line2D(
                [0],
                [0],
                color="0.3",
                marker="x",
                linestyle=":",
                label="independent census",
            ),
        ]
    )
    if tolerance is not None:
        legend_handles.append(
            Patch(
                facecolor="0.85",
                alpha=0.5,
                label=f"±{tolerance:.0%} practical margin",
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("Forward-time LD calibration", y=0.995)

    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.97))

    return _save_figure(
        figure,
        output_directory,
        "simulation_calibration",
        formats=formats,
        dpi=dpi,
    )


def plot_pair_convergence(
    summary: pd.DataFrame,
    output_directory: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, Path]:
    """Plot between-seed r2D and Ne stability across locus-pair budgets."""

    required = {
        "Population",
        "max_pairs",
        "r2D_mean",
        "r2D_std",
        "Ne_mean",
        "Ne_std",
    }
    _require_columns(summary, required, plot_name="Pair-convergence summary")
    populations = summary["Population"].drop_duplicates().tolist()
    pair_budgets = np.sort(summary["max_pairs"].unique())
    figure, axes = plt.subplots(
        len(populations),
        2,
        figsize=(10.0, max(4.0, 2.3 * len(populations))),
        squeeze=False,
    )
    for row_index, population in enumerate(populations):
        subset = summary.loc[summary["Population"] == population].sort_values(
            "max_pairs"
        )
        pair_budget = subset["max_pairs"].to_numpy(dtype=float)
        axes[row_index, 0].errorbar(
            pair_budget,
            subset["r2D_mean"],
            yerr=subset["r2D_std"].fillna(0.0),
            marker="o",
            capsize=3,
            color=plt.get_cmap("tab10")(0),
        )
        axes[row_index, 0].axhline(0.0, linestyle="--", color="0.5", linewidth=0.9)
        axes[row_index, 1].errorbar(
            pair_budget,
            subset["Ne_mean"],
            yerr=subset["Ne_std"].fillna(0.0),
            marker="D",
            capsize=3,
            color=plt.get_cmap("tab10")(1),
        )
        axes[row_index, 0].set_ylabel(str(population))
        for axis in axes[row_index]:
            axis.set_xscale("log")
            axis.set_xticks(pair_budgets)
            axis.set_xticklabels([f"{value:,.0f}" for value in pair_budgets])
            axis.xaxis.set_minor_locator(NullLocator())
            axis.set_xlim(pair_budgets.min() * 0.9, pair_budgets.max() * 1.1)
            axis.grid(alpha=0.25)
    axes[0, 0].set_title(r"$r_D^2$ mean ± SD")
    axes[0, 1].set_title(r"$N_e$ mean ± SD")
    axes[-1, 0].set_xlabel("Maximum locus pairs")
    axes[-1, 1].set_xlabel("Maximum locus pairs")
    figure.suptitle("Locus-pair subsampling convergence", y=1.005)
    figure.tight_layout()
    return _save_figure(
        figure,
        output_directory,
        "pair_convergence",
        formats=formats,
        dpi=dpi,
    )


__all__ = [
    "plot_exact_expectation_errors",
    "plot_golden_reference_errors",
    "plot_pair_convergence",
    "plot_published_island_fox_comparison",
    "plot_simulation_calibration",
]
