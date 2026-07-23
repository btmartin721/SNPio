"""Visualizations for unbiased linkage-disequilibrium analyses."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from logging import Logger


class LinkageDisequilibriumPlotter:
    """Render LD outputs into SNPio's analysis plot hierarchy."""

    # Okabe-Ito colors, paired with redundant marker shapes so diagnostic
    # meaning remains visible in grayscale and under color-vision deficiency.
    PRIMARY_COLOR = "#0072B2"
    PRIMARY_FILL_COLOR = "#56B4E9"
    WARNING_COLOR = "#D55E00"
    NEUTRAL_COLOR = "#4D4D4D"
    ESTIMATE_MARKER = "o"
    WARNING_MARKER = "D"

    def __init__(
        self,
        output_dir: Path,
        *,
        plot_format: str,
        dpi: int,
        fontsize: int,
        title_fontsize: int,
        despine: bool,
        show: bool,
        logger: "Logger | None" = None,
    ) -> None:
        """Initialize a plotter for linkage-disequilibrium outputs.

        This plotter generates three types of plots from the outputs of the ``LinkageDisequilibriumAnalyzer`` class: (1) a population summary of normalized LD and the ``rDz`` diagnostic, (2) a plot of recent effective population size estimates, and (3) distributions of non-zero pairwise LD estimates and the fraction of informative pairs.

        Args:
            output_dir (Path): Directory to save plots.
            plot_format (str): File format for saved plots (e.g., "png", "pdf").
            dpi (int): Resolution for saved plots.
            fontsize (int): Base font size for plot text.
            title_fontsize (int): Font size for plot titles.
            despine (bool): Whether to remove top and right spines from plots.
            show (bool): Whether to display plots interactively.
            logger (Logger | None): Optional logger for informational messages.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_format = plot_format
        self.dpi = dpi
        self.fontsize = fontsize
        self.title_fontsize = title_fontsize
        self.despine = despine
        self.show = show
        self.logger = logger

    def plot_all(
        self, summary: pd.DataFrame, pairwise: pd.DataFrame
    ) -> dict[str, Path]:
        """Generate all applicable LD plots and return their paths."""

        files = {
            "ld_population_plot": self._plot_population_summary(summary),
        }

        ne_path = self._plot_effective_population_size(summary)

        if ne_path is not None:
            files["ld_ne_plot"] = ne_path

        distribution_path = self._plot_pairwise_distributions(pairwise, summary)
        if distribution_path is not None:
            files["ld_distribution_plot"] = distribution_path

        return files

    def _finalize(self, figure, path: Path) -> Path:
        """Save, optionally display, and close a Matplotlib figure."""

        figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(figure)

        if self.logger is not None:
            self.logger.info(f"Saved linkage-disequilibrium plot to {path}")

        return path

    @staticmethod
    def _asymmetric_error(
        point: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Convert interval endpoints into non-negative Matplotlib errors."""

        low_error = np.where(np.isfinite(lower), np.maximum(point - lower, 0), 0)
        high_error = np.where(np.isfinite(upper), np.maximum(upper - point, 0), 0)
        return np.vstack([low_error, high_error])

    @staticmethod
    def _population_labels(summary: pd.DataFrame) -> dict[str, str]:
        """Build population labels that expose the analyzed sample size."""

        labels: dict[str, str] = {}
        for row in summary.itertuples(index=False):
            population = str(row.Population)
            samples = getattr(row, "Samples", None)
            if samples is None or not np.isfinite(float(samples)):
                labels[population] = population
            else:
                labels[population] = f"{population}\n(n={int(samples)})"
        return labels

    @staticmethod
    def _diagnostic_flags(summary: pd.DataFrame) -> np.ndarray:
        """Flag populations whose bootstrap ``rDz`` interval excludes zero."""

        lower = summary["rDz_CI_Lower"].to_numpy(dtype=float)
        upper = summary["rDz_CI_Upper"].to_numpy(dtype=float)
        return np.isfinite(lower) & np.isfinite(upper) & ((lower > 0) | (upper < 0))

    @staticmethod
    def _use_log_scale(values: np.ndarray, threshold: float = 100.0) -> bool:
        """Return whether strictly positive values warrant a logarithmic axis."""

        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2 or np.any(finite <= 0):
            return False
        return bool(finite.max() / finite.min() >= threshold)

    @classmethod
    def _point_style(cls, diagnostic_warning: bool) -> dict[str, str]:
        """Return redundant color and marker encodings for an estimate."""

        if diagnostic_warning:
            return {"color": cls.WARNING_COLOR, "marker": cls.WARNING_MARKER}

        return {"color": cls.PRIMARY_COLOR, "marker": cls.ESTIMATE_MARKER}

    @classmethod
    def _pairwise_uses_log_scale(cls, column: str, values: np.ndarray) -> bool:
        """Use log scaling only for wide-ranging, strictly positive ``Pi2``."""

        return column == "Pi2" and cls._use_log_scale(values)

    def _plot_population_summary(self, summary: pd.DataFrame) -> Path:
        """Plot normalized LD and the ``rDz`` model diagnostic by population."""

        populations = summary["Population"].astype(str).to_numpy()

        label_map = self._population_labels(summary)

        population_labels = [label_map[population] for population in populations]

        diagnostic_flags = self._diagnostic_flags(summary)

        x = np.arange(populations.size)

        figure, axes = plt.subplots(
            1, 2, figsize=(max(12, populations.size * 1.8), 7.4)
        )

        specifications = [
            ("r2D", r"$r_D^2$", "Unbiased normalized LD"),
            ("rDz", r"$r_{Dz}$", "Model-assumption diagnostic"),
        ]

        for axis, (column, ylabel, title) in zip(axes, specifications):
            points = summary[column].to_numpy(dtype=float)
            lower = summary[f"{column}_CI_Lower"].to_numpy(dtype=float)
            upper = summary[f"{column}_CI_Upper"].to_numpy(dtype=float)

            for index, point in enumerate(points):
                flagged = column == "rDz" and diagnostic_flags[index]
                style = self._point_style(flagged)

                axis.errorbar(
                    x[index],
                    point,
                    yerr=self._asymmetric_error(
                        points[index : index + 1],
                        lower[index : index + 1],
                        upper[index : index + 1],
                    ),
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=2.1,
                    capsize=5,
                    markersize=9,
                )

            axis.axhline(0, color=self.NEUTRAL_COLOR, linewidth=1, linestyle="--")

            axis.set_title(f"{title} (linear scale)", fontsize=self.title_fontsize)

            axis.set_ylabel(ylabel, fontsize=self.fontsize)
            axis.set_xticks(x, population_labels, rotation=45, ha="right")
            axis.tick_params(labelsize=max(8, self.fontsize))
            axis.margins(y=0.12)

            if self.despine:
                sns.despine(ax=axis)

        figure.suptitle(
            "Linkage disequilibrium from unlinked SNPs",
            fontsize=self.title_fontsize + 1,
        )

        figure.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker=self.ESTIMATE_MARKER,
                    linestyle="none",
                    color=self.PRIMARY_COLOR,
                    label="Estimate; error bars are 95% bootstrap CIs",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=self.WARNING_MARKER,
                    linestyle="none",
                    color=self.WARNING_COLOR,
                    label=r"Diagnostic warning: $r_{Dz}$ CI excludes zero",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.NEUTRAL_COLOR,
                    linestyle="--",
                    label="Zero reference",
                ),
            ],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.115),
            ncol=3,
            fontsize=max(8, self.fontsize - 4),
        )
        figure.text(
            0.5,
            0.02,
            r"Orange diamonds mark populations whose 95% bootstrap $r_{Dz}$ "
            "interval excludes zero. Because unlinked "
            r"$r_{Dz}$ is expected near zero under the simple population model, "
            "this may indicate recent migration or population structure.\n"
            r"It is not a failed calculation, but the corresponding $N_e$ "
            "estimate should be interpreted cautiously.",
            ha="center",
            va="bottom",
            fontsize=max(8, self.fontsize - 4),
            color=self.NEUTRAL_COLOR,
        )

        figure.tight_layout(rect=(0, 0.22, 1, 0.94))

        path = self.output_dir / f"linkage_disequilibrium_summary.{self.plot_format}"

        return self._finalize(figure, path)

    def _plot_effective_population_size(self, summary: pd.DataFrame) -> Path | None:
        """Plot recent effective population-size estimates and intervals."""

        finite = summary.loc[
            np.isfinite(summary["Ne"].to_numpy(dtype=float))
            & (summary["Ne"].to_numpy(dtype=float) > 0)
        ].copy()

        if finite.empty:
            return None

        finite = finite.sort_values("Ne", ascending=True).reset_index(drop=True)
        y = np.arange(finite.shape[0])
        points = finite["Ne"].to_numpy(dtype=float)
        lower = finite["Ne_CI_Lower"].to_numpy(dtype=float)
        upper = finite["Ne_CI_Upper"].to_numpy(dtype=float)
        lower = np.where(lower > 0, lower, np.nan)
        unbounded_upper = np.isposinf(upper)
        bounded_upper = np.where(np.isfinite(upper), upper, points)
        label_map = self._population_labels(finite)
        diagnostic_flags = self._diagnostic_flags(finite)

        figure, axis = plt.subplots(figsize=(9, max(5.5, finite.shape[0] * 0.7 + 3.0)))

        for index, point in enumerate(points):
            style = self._point_style(bool(diagnostic_flags[index]))
            axis.errorbar(
                point,
                y[index],
                xerr=self._asymmetric_error(
                    points[index : index + 1],
                    lower[index : index + 1],
                    bounded_upper[index : index + 1],
                ),
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=2.1,
                capsize=5,
                markersize=9,
            )

            if unbounded_upper[index]:
                axis.annotate(
                    r"$+\infty$",
                    xy=(point, y[index]),
                    xytext=(14, 0),
                    textcoords="offset points",
                    va="center",
                    color=style["color"],
                    fontsize=max(8, self.fontsize),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": style["color"],
                        "lw": 1.2,
                    },
                )

        populations = finite["Population"].astype(str).tolist()

        axis.set_yticks(y, [label_map[population] for population in populations])

        scale_values = np.concatenate([points, lower, upper])
        use_log_scale = self._use_log_scale(scale_values)

        if use_log_scale:
            axis.set_xscale("log")

        scale_label = (
            "log scale; values are untransformed" if use_log_scale else "linear scale"
        )

        axis.set_xlabel(rf"Recent effective population size ($N_e$; {scale_label})")

        axis.set_title(
            r"Effective population size from unlinked $r_D^2$",
            fontsize=self.title_fontsize,
        )

        axis.margins(x=0.08)

        axis.tick_params(labelsize=max(8, self.fontsize))
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=self.ESTIMATE_MARKER,
                linestyle="none",
                color=self.PRIMARY_COLOR,
                label="Estimate; error bars are 95% bootstrap CIs",
            )
        ]

        if diagnostic_flags.any():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=self.WARNING_MARKER,
                    linestyle="none",
                    color=self.WARNING_COLOR,
                    label=r"Caution: $r_{Dz}$ CI excludes zero",
                )
            )
        if unbounded_upper.any():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=r"$\rightarrow$",
                    linestyle="none",
                    color=self.NEUTRAL_COLOR,
                    label=r"upper $N_e$ interval is unbounded",
                )
            )

        figure.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.075 if diagnostic_flags.any() else 0.02),
            ncol=min(3, len(legend_handles)),
            fontsize=max(8, self.fontsize - 4),
        )

        if diagnostic_flags.any():
            figure.text(
                0.5,
                0.015,
                r"A non-zero $r_{Dz}$ signal may reflect recent migration or "
                "population structure, so the simple model used to convert "
                r"$r_D^2$ into $N_e$ may not fit that population well.",
                ha="center",
                va="bottom",
                fontsize=max(8, self.fontsize - 4),
                color=self.NEUTRAL_COLOR,
            )

        if self.despine:
            sns.despine(ax=axis)

        bottom = 0.16 if diagnostic_flags.any() else 0.08
        figure.tight_layout(rect=(0, bottom, 1, 1))

        path = (
            self.output_dir
            / f"linkage_disequilibrium_effective_size.{self.plot_format}"
        )

        return self._finalize(figure, path)

    def _plot_pairwise_distributions(
        self,
        pairwise: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> Path | None:
        """Plot non-zero pairwise estimates and the informative-pair fraction."""

        if pairwise.empty:
            return None

        statistics = [
            ("D", r"$\widehat{D}$"),
            ("D2", r"$\widehat{D^2}$"),
            ("Dz", r"$\widehat{Dz}$"),
            ("Pi2", r"$\widehat{\pi_2}$"),
            ("r2_star", r"Pairwise $r^{2*}$"),
        ]
        working = pairwise.copy()
        working["Population"] = working["Population"].astype(str)
        populations = working["Population"].drop_duplicates().tolist()
        label_map = self._population_labels(summary)

        population_labels = [
            label_map.get(population, population) for population in populations
        ]

        figure, axes = plt.subplots(2, 3, figsize=(18, 11))

        for axis, (column, title) in zip(axes.flat, statistics):
            data = (
                working[["Population", column]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            data = data.loc[data[column] != 0]

            if data.empty:
                axis.set_visible(False)
                continue

            sns.boxplot(
                data=data,
                x="Population",
                y=column,
                order=populations,
                color=self.PRIMARY_FILL_COLOR,
                showfliers=False,
                width=0.65,
                boxprops={
                    "facecolor": self.PRIMARY_FILL_COLOR,
                    "edgecolor": self.PRIMARY_COLOR,
                    "alpha": 0.8,
                },
                medianprops={
                    "color": self.NEUTRAL_COLOR,
                    "linewidth": 1.8,
                },
                whiskerprops={
                    "color": self.PRIMARY_COLOR,
                    "linewidth": 1.3,
                },
                capprops={
                    "color": self.PRIMARY_COLOR,
                    "linewidth": 1.3,
                },
                ax=axis,
            )

            axis.set_title(title, fontsize=self.title_fontsize)
            axis.set_xlabel("")
            axis.set_xticks(
                np.arange(len(populations)),
                population_labels,
                rotation=45,
                ha="right",
            )

            axis.tick_params(labelsize=max(8, self.fontsize - 4))

            values = data[column].to_numpy(dtype=float)
            use_log_scale = self._pairwise_uses_log_scale(column, values)
            if use_log_scale:
                axis.set_yscale("log")
                axis.set_ylabel(
                    "Estimate (log scale; values are untransformed)",
                    fontsize=max(8, self.fontsize - 2),
                )
            else:
                axis.set_ylabel(
                    "Estimate (linear scale)",
                    fontsize=max(8, self.fontsize - 2),
                )
                axis.axhline(
                    0,
                    color=self.NEUTRAL_COLOR,
                    linestyle="--",
                    linewidth=0.9,
                )

            if column in {"D", "D2", "Dz", "r2_star"}:
                y_lower, y_upper = axis.get_ylim()
                extent = max(abs(y_lower), abs(y_upper))
                axis.set_ylim(-extent, extent)

            axis.margins(y=0.1)

            if self.despine:
                sns.despine(ax=axis)

        informative_axis = axes.flat[-1]
        pi2 = working["Pi2"].replace([np.inf, -np.inf], np.nan)
        informative = pi2.notna() & (pi2 != 0)

        informative_percent = (
            informative.groupby(working["Population"], sort=False)
            .mean()
            .reindex(populations)
            .mul(100)
        )

        bars = informative_axis.bar(
            np.arange(len(populations)),
            informative_percent.to_numpy(dtype=float),
            color=self.PRIMARY_FILL_COLOR,
            edgecolor=self.PRIMARY_COLOR,
            linewidth=1.0,
        )

        informative_axis.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in informative_percent],
            padding=3,
            fontsize=max(7, self.fontsize),
            rotation=90,
        )

        informative_axis.set_title(
            "Informative sampled pairs", fontsize=self.title_fontsize
        )

        informative_axis.set_ylabel(r"Pairs with non-zero $\widehat{\pi_2}$ (%)")

        informative_axis.set_xticks(
            np.arange(len(populations)), population_labels, rotation=45, ha="right"
        )

        informative_axis.set_ylim(0, 105)
        informative_axis.tick_params(labelsize=max(8, self.fontsize - 4))

        if self.despine:
            sns.despine(ax=informative_axis)

        figure.suptitle(
            "Sampled pairwise unbiased LD estimators",
            fontsize=self.title_fontsize + 1,
        )

        figure.text(
            0.5,
            0.935,
            "Blue boxes show the interquartile range, dark center lines show "
            "medians, and whiskers extend to 1.5 × IQR; population is encoded "
            "by the x-axis, not by color.\n"
            "Exact-zero estimates are omitted from the boxplots and summarized "
            "by the blue informative-pair bars.",
            ha="center",
            va="top",
            fontsize=max(8, self.fontsize - 3),
            color=self.NEUTRAL_COLOR,
        )

        figure.tight_layout(rect=(0, 0, 1, 0.89))

        path = (
            self.output_dir
            / f"linkage_disequilibrium_pairwise_distributions.{self.plot_format}"
        )

        return self._finalize(figure, path)


__all__ = ["LinkageDisequilibriumPlotter"]
