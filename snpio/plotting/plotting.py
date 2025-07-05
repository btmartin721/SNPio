import json
import warnings
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

import holoviews as hv
import matplotlib as mpl
from pysam import index

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from holoviews import opts
from mpl_toolkits.mplot3d import Axes3D  # Don't remove this import.

hv.extension("bokeh")

from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils import misc
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC, build_dataframe
from snpio.utils.multiqc_reporter import SNPioMultiQC

if TYPE_CHECKING:
    from snpio.utils.missing_stats import MissingStats


class Plotting:
    """Class containing various methods for generating plots based on genotype data.

    This class is initialized with a GenotypeData object containing necessary data. The class attributes are set based on the provided values, the GenotypeData object, or default values.

    Attributes:
        genotype_data (GenotypeData): Initialized GenotypeData object containing necessary data.
        prefix (str): Prefix string for output directories and files.
        output_dir (Path): Output directory for saving plots.
        show (bool): Whether to display the plots.
        plot_format (str): Format in which to save the plots.
        dpi (int): Resolution of the saved plots.
        plot_fontsize (int): Font size for the plot labels.
        plot_title_fontsize (int): Font size for the plot titles.
        despine (bool): Whether to remove the top and right plot axis spines.
        verbose (bool): Whether to enable verbose logging.
        debug (bool): Whether to enable debug logging.
        logger (logging.Logger): Logger object for logging messages.
        boolean_filter_methods (list): List of boolean filter methods.
        missing_filter_methods (list): List of missing data filter methods.
        maf_filter_methods (list): List of MAF filter methods.
        mpl_params (dict): Default Matplotlib parameters for the plots.

    Methods:
        plot_sankey_filtering_report: Plot a Sankey diagram for the filtering report.
        plot_pca: Plot a PCA scatter plot with 2 or 3 dimensions, colored by missing data proportions, and labeled by population with symbols for each sample.
        plot_summary_statistics: Plot summary statistics per sample and per population on the same figure. The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst).
        plot_dapc: Plot a DAPC scatter plot. with 2 or 3 dimensions, colored by population, and labeled by population with symbols for each sample.
        plot_fst_heatmap: Plot a heatmap of Fst values between populations, sorted by highest Fst and displaying only the lower triangle.
        plot_fst_outliers: Plot a heatmap of Fst values for outlier SNPs, highlighting contributing population pairs.
        plot_d_statistics: Create plots for D-statistics with multiple test corrections.
        _set_logger: Set the logger object based on the debug attribute. If debug is True, the logger will log debug messages.
        _get_attribute_value: Determine the value for an attribute based on the provided argument, genotype_data attribute, or default value. If a value is provided during initialization, it is used. Otherwise, the genotype_data attribute is used if available. If neither is available, the default value is used.
        _plot_summary_statistics_per_sample: Plot summary statistics per sample. If an axis is provided, the plot is drawn on that axis.
        _plot_summary_statistics_per_population: Plot summary statistics per population. If an axis is provided, the plot is drawn on that axis.
    """

    def __init__(
        self,
        genotype_data: Any,
        show: bool | None = None,
        plot_format: str | None = None,
        dpi: int | None = None,
        plot_fontsize: int | None = None,
        plot_title_fontsize: int | None = None,
        despine: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> None:
        """Initialize the Plotting class.

        This class contains various methods for generating plots based on genotype data. The class is initialized with a GenotypeData object containing necessary data. The class attributes are set based on the provided values, the GenotypeData object, or default values.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object containing necessary data.
            show (bool | None): Whether to display the plots. Defaults to `genotype_data.show` if available, otherwise `False`.
            plot_format (str | None): The format in which to save the plots (e.g., 'png', 'svg'). Defaults to `genotype_data.plot_format` if available, otherwise `'png'`.
            dpi (int | None): The resolution of the saved plots. Unused for vector `plot_format` types. Defaults to `genotype_data.dpi` if available, otherwise `300`.
            plot_fontsize (int | None): The font size for the plot labels. Defaults to `genotype_data.plot_fontsize` if available, otherwise `18`.
            plot_title_fontsize (int | None): The font size for the plot titles. Defaults to `genotype_data.plot_title_fontsize` if available, otherwise `22`.
            despine (bool | None): Whether to remove the top and right plot axis spines. Defaults to `genotype_data.despine` if available, otherwise `True`.
            verbose (bool | None): Whether to enable verbose logging. Defaults to `genotype_data.verbose` if available, otherwise `False`.
            debug (bool | None): Whether to enable debug logging. Defaults to `genotype_data.debug` if available, otherwise `False`.

        Note:
            - The `genotype_data` attribute must be provided during initialization.

            - The `show`, `plot_format`, `dpi`, `plot_fontsize`, `plot_title_fontsize`, `despine`, `verbose`, and `debug` attributes are set based on the provided values, the `genotype_data` object, or default values.

            - The `output_dir` attribute is set to the `prefix_output/nremover/plots` directory.

            - The `logger` attribute is set based on the `debug` attribute.

            - The `boolean_filter_methods`, `missing_filter_methods`, and `maf_filter_methods` attributes are set to lists of filter methods.

            - The `mpl_params` dictionary contains default Matplotlib parameters for the plots.

            - The Matplotlib parameters are updated with the `mpl_params` dictionary.

            - The `plotting` object is used to set the attributes based on the provided values, the `genotype_data` object, or default values.
        """
        self.genotype_data = genotype_data
        self.prefix: str = getattr(genotype_data, "prefix", "plot")

        if self.genotype_data.was_filtered:
            self.output_dir: Path = Path(f"{self.prefix}_output") / "nremover"
        else:
            self.output_dir: Path = Path(f"{self.prefix}_output")

        self.output_dir_gd: Path = self.output_dir / "plots" / "gtdata"
        self.output_dir_analysis: Path = self.output_dir / "plots" / "analysis"
        self.report_dir_gd: Path = self.output_dir / "reports" / "gtdata"
        self.report_dir_analysis: Path = self.output_dir / "reports" / "analysis"

        self.output_dir_gd.mkdir(parents=True, exist_ok=True)
        self.output_dir_analysis.mkdir(parents=True, exist_ok=True)
        self.report_dir_gd.mkdir(parents=True, exist_ok=True)
        self.report_dir_analysis.mkdir(parents=True, exist_ok=True)

        self.verbose: bool | None = verbose
        self.debug: bool | None = debug

        prefix: str = genotype_data.prefix

        logman = LoggerManager(__name__, prefix=prefix, debug=debug, verbose=verbose)

        self.logger: Logger = logman.get_logger()

        self.iupac = IUPAC(logger=self.logger)

        # Define default values for attributes
        self._defaults: Dict[str, bool | str | int | float] = {
            "show": False,
            "plot_format": "png",
            "dpi": 300,
            "plot_fontsize": 18,
            "plot_title_fontsize": 22,
            "despine": True,
            "verbose": verbose,
            "debug": debug,
        }

        # Mapping of attributes to their provided values
        self._provided_values = {
            "show": show,
            "plot_format": plot_format,
            "dpi": dpi,
            "plot_fontsize": plot_fontsize,
            "plot_title_fontsize": plot_title_fontsize,
            "despine": despine,
            "verbose": verbose,
            "debug": debug,
        }

        # List of attributes to set
        self._attributes: List[str] = [
            "show",
            "plot_format",
            "dpi",
            "plot_fontsize",
            "plot_title_fontsize",
            "despine",
            "verbose",
            "debug",
        ]

        # Set attributes using the helper method
        for attr in self._attributes:
            value = self._get_attribute_value(attr)
            setattr(self, attr, value)

        self.boolean_filter_methods: List[str] = [
            "filter_singletons",
            "filter_biallelic",
            "filter_monomorphic",
            "thin_loci",
            "filter_linked",
        ]

        self.missing_filter_methods: List[str] = [
            "filter_missing",
            "filter_missing_sample",
            "filter_missing_pop",
        ]

        self.maf_filter_methods: List[str] = ["filter_maf", "filter_mac"]

        self.mpl_params = {
            "xtick.labelsize": self.plot_fontsize,
            "ytick.labelsize": self.plot_fontsize,
            "legend.fontsize": self.plot_fontsize,
            "legend.fancybox": True,
            "legend.shadow": True,
            "figure.titlesize": self.plot_title_fontsize,
            "figure.facecolor": "white",
            "figure.dpi": self.dpi,
            "font.size": self.plot_fontsize,
            "axes.titlesize": self.plot_title_fontsize,
            "axes.labelsize": self.plot_fontsize,
            "axes.grid": False,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False if self.despine else True,
            "axes.spines.right": False if self.despine else True,
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }

        mpl.rcParams.update(self.mpl_params)

        self.snpio_mqc = SNPioMultiQC

    def _get_attribute_value(self, attr: str) -> Any:
        """Determine the value for an attribute based on the provided argument.

        This method checks if a value was provided during initialization, if the genotype_data object has the attribute, or if a default value is available. It returns the determined value for the attribute.

        Args:
            attr (str): The name of the attribute.

        Returns:
            Any: The determined value for the attribute.
        """
        # Check if a value was provided during initialization
        provided = self._provided_values.get(attr)
        if provided is not None:
            self.logger.debug(f"Using provided value for '{attr}': {provided}")
            return provided

        # Check if genotype_data has the attribute
        genotype_val: Any | None = getattr(self.genotype_data, attr, None)
        if genotype_val is not None:
            self.logger.debug(f"Using genotype_data value for '{attr}': {genotype_val}")
            return genotype_val

        # Use the default value
        default: bool | str | int | float | None = self._defaults.get(attr)
        self.logger.debug(f"Using default value for '{attr}': {default}")
        return default

    def _plot_summary_statistics_per_sample(
        self,
        summary_stats: pd.DataFrame,
        ax: plt.Axes | None = None,
        window: int = 5,
        subsample_rate: int = 50,
    ) -> None:
        """Plot summary statistics per sample.

        This method plots the summary statistics per sample as lines for each statistic (Ho, He, Pi, Fst) on the same figure. The summary statistics are smoothed using a rolling average with a window size of 5 and subsampled to reduce plot density.

        Args:
            summary_stats (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.
            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.
            window (int): Window size for rolling average smoothing (optional).
            subsample_rate (int): Rate of subsampling to reduce density on plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Drop NaNs to ensure no unwanted line artifacts
        summary_stats = summary_stats.dropna()

        # Interpolate remaining NaNs (if any) for smoother lines
        interpolated_stats: pd.DataFrame = summary_stats[
            ["Ho", "He", "Pi"]
        ].interpolate(method="linear")

        # Apply rolling average for smoothing
        smoothed_stats = interpolated_stats.rolling(window=window, center=True).mean()

        # Subsample data to reduce plot density
        sampled_indices = summary_stats.index[::subsample_rate]
        sampled_smoothed_stats = smoothed_stats.loc[sampled_indices]

        # Plot each statistic with distinct colors
        colors: List[str] = ["blue", "green", "red"]
        for stat, color in zip(["Ho", "He", "Pi"], colors):
            ax.plot(
                sampled_indices,
                sampled_smoothed_stats[stat],
                label=stat,
                color=color,
                linewidth=1.5,
            )

        # Labeling and grid improvements
        ax.set_xlabel("Locus")
        ax.set_ylabel("Value")
        ax.set_title("Summary Statistics per Locus (Overall)")
        ax.legend(
            title="Statistics",
            loc="upper left",
            bbox_to_anchor=(1, 1.02),
            fancybox=True,
            shadow=True,
        )
        plt.tight_layout()

    def _plot_summary_statistics_per_population(
        self, per_population_stats: dict, ax: plt.Axes | None = None
    ) -> None:
        """Plot mean summary statistics per population as grouped bar chart.

        This method plots the mean summary statistics per population as a grouped bar chart with different colors for each statistic (Ho, He, Pi).

        Args:
            per_population_stats (dict): Dictionary containing summary statistics per population.
            ax (matplotlib.axes.Axes | None, optional): The matplotlib axis on which to plot the summary statistics.
        """
        if ax is None:
            _, ax = plt.subplots()

        pop_stats = pd.DataFrame(columns=["Ho", "He", "Pi"])
        for pop_id, stats_df in per_population_stats.items():
            stats_df = stats_df.copy()

            if not stats_df.empty:
                stats_df = stats_df.dropna(how="any", axis=0)
                stats_df = stats_df.apply(lambda x: x.astype(float))
                stats_df["PopulationID"] = pop_id

                with warnings.catch_warnings():
                    # TODO: pandas >= 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    pop_stats = pd.concat([pop_stats, stats_df], ignore_index=True)

            else:
                self.logger.warning(
                    f"Empty or NaN summary statistics for population {pop_id}"
                )

        pop_stats: pd.DataFrame = pd.melt(
            pop_stats, id_vars="PopulationID", var_name="Statistic", value_name="Value"
        )

        if len(per_population_stats) <= 9:
            pal = "Set1"
        elif len(per_population_stats) <= 10:
            pal = "tab10"
        elif len(per_population_stats) <= 12:
            pal = "Set3"
        elif len(per_population_stats) <= 20:
            pal = "tab20"
        else:
            pal = "viridis"  # Fallback to viridis for many populations

        ax = sns.barplot(
            data=pop_stats,
            x="PopulationID",
            y="Value",
            hue="Statistic",
            ax=ax,
            palette=pal,
            edgecolor="black",
        )
        ax.set_xlabel("Population")
        ax.set_ylabel("Mean Value")
        ax.set_title("Mean Summary Statistics per Population")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.legend(title="Statistics", loc="best", bbox_to_anchor=(1.04, 1))

    def plot_permutation_dist(
        self,
        obs_fst: float,
        dist: np.ndarray,
        pop1_label: str,
        pop2_label: str,
        dist_type: str = "fst",
    ):
        """Plot the permutation distribution of Fst values.

        This method creates a histogram of the permutation distribution of Fst values, with a kernel density estimate (KDE) overlay. It also adds vertical lines for the observed Fst value and the mean of the permutation distribution.

        Args:
            obs_fst (float): The observed Fst value.
            dist (np.ndarray): The permutation distribution of Fst values.
            pop1_label (str): Label for the first population.
            pop2_label (str): Label for the second population.
            dist_type (str): Type of distance metric used (default: "fst"). Other option: "nei".
        """

        sns.set_style("white")
        sns.despine()

        if dist.size == 0:
            self.logger.warning(
                f"No permutation distribution data available for {pop1_label} vs {pop2_label}. Skipping plot."
            )
            return

        try:
            sns.histplot(
                dist,
                bins="auto",
                kde=True,
                color="darkorchid",
                alpha=0.7,
                label="Permutation Distribution",
                fill=True,
                legend=True,
            )
        except Exception as e:
            self.logger.warning(
                f"Error plotting permutation histogram for populations {pop1_label} and {pop2_label}: {e}"
            )
            return

        plt.axvline(obs_fst, color="orange", linestyle="--", label="Observed Fst")

        plt.axvline(
            dist.mean(),
            color="limegreen",
            linestyle="--",
            label="Mean Permuted Fst",
        )

        plt.title(f"Permutation Dist: {pop1_label} vs {pop2_label}")
        plt.xlabel("Fst")
        plt.ylabel("Frequency")
        plt.legend(loc="center", bbox_to_anchor=(0.5, -0.4), shadow=True, fancybox=True)

        out_file = (
            f"{dist_type}_permutation_dist_{pop1_label}_{pop2_label}.{self.plot_format}"
        )
        plt.savefig(self.output_dir_analysis / out_file)
        if self.show:
            plt.show()
        plt.close()

        dist_data = {
            "Permutation Fst": dist.tolist(),
            "Observed Fst": obs_fst,
            "Mean Permuted Fst": dist.mean(),
            "Population 1": pop1_label,
            "Population 2": pop2_label,
            "Distance Type": dist_type,
        }

        with open(
            self.report_dir_analysis
            / f"{dist_type}_permutation_dist_{pop1_label}_{pop2_label}.json",
            "w",
        ) as f:
            json.dump(dist_data, f, indent=4)

    def _plot_fst_heatmap(
        self,
        df_fst_mean: Dict[tuple, Any] | pd.DataFrame,
        *,
        df_fst_lower: pd.DataFrame | None = None,
        df_fst_upper: pd.DataFrame | None = None,
        df_fst_pvals: pd.DataFrame | None = None,
        use_pvalues: bool = False,
        palette: str = "magma",
        title: str = "Mean Fst Between Populations",
        dist_type: str = "fst",
    ) -> None:
        """Plot a heatmap of Fst values (or p-values).

        Args:
            df_fst_mean (dict | pd.DataFrame): Dictionary or DataFrame containing mean Fst values.
            df_fst_lower (pd.DataFrame, optional): DataFrame containing lower confidence intervals (optional).
            df_fst_upper (pd.DataFrame, optional): DataFrame containing upper confidence intervals (optional).
            df_fst_pvals (pd.DataFrame, optional): DataFrame containing p-values for Fst values (optional).
            use_pvalues (bool): Whether to use p-values instead of Fst values for the heatmap.
            palette (str): Color palette for the heatmap.
            title (str): Title for the heatmap.
            dist_type (str): Type of distance metric used (default: "fst"). Other option: "nei".
        """
        fig_size = (48, 48)
        title_fontsize = 72
        tick_fontsize = 72
        annot_fontsize = 38
        cbar_fontsize = 72

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(df_fst_mean, dtype=bool))
        np.fill_diagonal(df_fst_mean.values, np.nan)
        df_fst_mean = df_fst_mean.mask(mask)  # Mask upper triangle
        df_fst_mean = df_fst_mean.round(3)

        mask_upper = np.triu(np.ones_like(df_fst_mean, dtype=bool), k=0)
        mask_lower = np.tril(np.ones_like(df_fst_mean, dtype=bool), k=0)

        mode = "fst"

        if df_fst_lower is not None and df_fst_upper is not None:
            df_fst_ci = pd.DataFrame(
                np.full(df_fst_lower.shape, np.nan),
                index=df_fst_lower.index,
                columns=df_fst_lower.columns,
            )

            # Create a mask for the lower triangle
            df_fst_ci.values[mask_lower] = (
                df_fst_lower.values[mask_lower] if df_fst_lower is not None else None
            )

            df_fst_ci.values[mask_upper] = (
                df_fst_upper.values[mask_upper] if df_fst_upper is not None else None
            )

            # Set the diagonal to NaN to avoid displaying self-comparisons
            np.fill_diagonal(df_fst_lower.values, np.nan)
            np.fill_diagonal(df_fst_upper.values, np.nan)
            df_fst_lower = df_fst_lower.mask(mask)  # Mask upper triangle
            df_fst_upper = df_fst_upper.mask(mask)  # Mask upper triangle
            mode = "bootstrap"
        if df_fst_pvals is not None:
            # Set the diagonal to NaN to avoid displaying self-comparisons
            np.fill_diagonal(df_fst_pvals.values, np.nan)
            df_fst_pvals = df_fst_pvals.mask(mask)
            mode = "bootstrap_with_p"

        df_annot = df_fst_mean.copy()
        df_annot = df_annot.mask(mask)  # Mask upper triangle
        df_annot = df_annot.astype(str)

        if df_fst_pvals is not None and use_pvalues:
            df_annot = df_annot + "\n" + df_fst_pvals.round(3).astype(str)

        elif df_fst_lower is not None and df_fst_upper is not None:
            df_annot = (
                df_annot
                + "\n["
                + df_fst_lower.round(3).astype(str)
                + ", "
                + df_fst_upper.round(3).astype(str)
                + "]"
            )

        annotation_matrix = df_annot.to_numpy()

        plt.figure(figsize=fig_size)
        ax = sns.heatmap(
            df_fst_mean,
            annot=annotation_matrix,
            annot_kws={"fontsize": annot_fontsize},
            fmt="",
            cmap=palette,
            mask=mask,
            linewidths=0,
            vmin=0,
            cbar=True,
            square=True,
        )
        ax.invert_xaxis()

        if mode == "fst":
            title_str = f"Multi-locus {dist_type.capitalize()}"
            cbar_label = f"Multi-locus {dist_type.capitalize()}"
        elif mode == "bootstrap_with_p" and use_pvalues:
            title_str = f"Mean {dist_type.capitalize()} and Permuted P-values"
            cbar_label = f"{dist_type.capitalize()}; P-values shown in cell"
        else:
            title_str = f"{title} (95% CI)"
            cbar_label = f"Mean {dist_type.capitalize()}"

        ax.set_title(title_str, fontsize=title_fontsize)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fontsize
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fontsize)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        cbar.set_label(cbar_label, fontsize=cbar_fontsize)

        out_file = f"{dist_type}_between_populations_heatmap.{self.plot_format}"
        plt.savefig(self.output_dir_analysis / out_file)
        if self.show:
            plt.show()
        plt.close()

    def plot_d_statistics(
        self, df: pd.DataFrame, method: Literal["patterson", "partitioned", "dfoil"]
    ) -> None:
        """Create plots for D-statistics with multiple‐test corrections."""
        method = method.lower()
        if method not in {"patterson", "partitioned", "dfoil"}:
            raise ValueError(f"Unsupported method: {method}")

        # Ensure we have a “Sample Combo” column
        combo_cols = [c for c in df.columns if c.startswith("P")]
        df["Sample Combo"] = df[combo_cols].astype(str).agg("-".join, axis=1)

        sns.set_theme(style="white")
        mpl.rcParams.update(self.mpl_params)

        # 1) Bar Plot of Significance Counts
        fig, ax = plt.subplots(figsize=(14, 6))
        if method != "dfoil":
            # Single-vector: raw, bonf, fdr
            sig = df["Significant (Raw)"].sum()
            ns = (~df["Significant (Raw)"]).sum()
            bsig = df["Significant (Bonferroni)"].sum()
            bns = (~df["Significant (Bonferroni)"]).sum()
            fsig = df["Significant (FDR-BH)"].sum()
            fns = (~df["Significant (FDR-BH)"]).sum()
            counts = pd.DataFrame(
                {
                    "Correction": [
                        "Raw",
                        "Raw",
                        "Bonferri",
                        "Bonferri",
                        "FDR-BH",
                        "FDR-BH",
                    ],
                    "Significance": ["Significant", "Not Significant"] * 3,
                    "Count": [sig, ns, bsig, bns, fsig, fns],
                }
            )
            sns.barplot(
                data=counts, x="Correction", y="Count", hue="Significance", ax=ax
            )
            ax.set_title(f"{method.capitalize()} D-Statistics Significance Counts")
        else:
            # DFOIL: 4 stats × 3 corrections
            dnames = ["DFO", "DFI", "DOL", "DIL"]
            rows = []
            for stat in dnames:
                for corr, col in [
                    ("Raw", f"P_{stat}"),
                    ("Bonf", f"P_{stat}_bonf"),
                    ("FDR", f"P_{stat}_fdr"),
                ]:
                    sig = (df[col] < 0.05).sum()
                    ns = (df[col] >= 0.05).sum()
                    rows.append(
                        {
                            "Stat": stat,
                            "Correction": corr,
                            "Significant": sig,
                            "Not Significant": ns,
                        }
                    )
            counts = pd.DataFrame(rows).melt(
                id_vars=["Stat", "Correction"],
                value_vars=["Significant", "Not Significant"],
                var_name="Sig",
                value_name="Count",
            )
            sns.catplot(
                data=counts,
                x="Stat",
                y="Count",
                hue="Sig",
                col="Correction",
                kind="bar",
                height=6,
                aspect=1,
                sharey=False,
            )
            ax = plt.gca()
            ax.set_title("D-FOIL Significance Counts")
        ax.legend(loc="best", bbox_to_anchor=(1.05, 1))
        out = f"{method}_significance_counts.{self.plot_format}"
        plt.tight_layout()
        plt.savefig(self.output_dir_analysis / out)
        if self.show:
            plt.show()
        plt.close()

        # 2) Distribution of Z-Scores
        fig, ax = plt.subplots(figsize=(10, 6))
        if method != "dfoil":
            sns.histplot(df["Z-Score"], kde=True, ax=ax)
            ax.axvline(df["Z-Score"].mean(), linestyle="--")
            ax.set_title(f"{method.capitalize()} Z-Score Distribution")
            ax.set_xlabel("Z-Score")
        else:
            # melt Z_DFO, Z_DFI, etc.
            dnames = ["DFO", "DFI", "DOL", "DIL"]
            melt = df[[f"Z_{s}" for s in dnames]].melt(
                var_name="Statistic", value_name="Z-score"
            )
            sns.histplot(
                data=melt,
                x="Z-score",
                hue="Statistic",
                multiple="stack",
                kde=True,
                ax=ax,
            )
            ax.set_title("D-FOIL Z-Score Distribution")
        out = f"{method}_z_distribution.{self.plot_format}"
        plt.tight_layout()
        fig.savefig(self.output_dir_analysis / out)
        if self.show:
            plt.show()
        plt.close()

        # 3) Barplot of Z-Score by Sample
        # dynamic sizing
        n = df["Sample Combo"].nunique()
        fig_height = max(8, n * 0.4)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        if method != "dfoil":
            sns.barplot(
                data=df,
                x="Z-Score",
                y="Sample Combo",
                hue="Significant (FDR-BH)",
                ax=ax,
            )
            ax.set_title(f"{method.capitalize()} Z-Score per Sample Combo")
        else:
            # for DFOIL we stack 4 bars per combo
            dnames = ["DFO", "DFI", "DOL", "DIL"]
            melt = df.melt(
                id_vars=["Sample Combo"],
                value_vars=[f"Z_{s}" for s in dnames],
                var_name="Statistic",
                value_name="Z-score",
            )
            sns.barplot(
                data=melt,
                x="Z-score",
                y="Sample Combo",
                hue="Statistic",
                orient="h",
                ax=ax,
            )
            ax.set_title("D-FOIL Z-Score per Sample Combo")
        out = f"{method}_z_by_combo.{self.plot_format}"

        fig.savefig(self.output_dir_analysis / out)

        if self.show:
            plt.show()
        plt.close()

        if method != "dfoil":
            # Patterson or Partitioned
            # Format significance counts into a two‐column table
            significance_df = pd.DataFrame(
                {
                    "Correction": ["Uncorrected", "Bonferroni", "FDR-BH"],
                    "Significant": [
                        df["Significant (Raw)"].sum(),
                        df["Significant (Bonferroni)"].sum(),
                        df["Significant (FDR-BH)"].sum(),
                    ],
                    "Not Significant": [
                        (~df["Significant (Raw)"]).sum(),
                        (~df["Significant (Bonferroni)"]).sum(),
                        (~df["Significant (FDR-BH)"]).sum(),
                    ],
                }
            ).melt(id_vars="Correction", var_name="Significance", value_name="Count")
            significance_df = self._format_significance_labels(significance_df)

            self.snpio_mqc.queue_table(
                df=significance_df,
                panel_id=f"{method}_d_statistics_significance_counts",
                section="introgression",
                title=f"SNPio: {method.capitalize()} D-Statistics Significance Counts",
                index_label="Significance (Correction)",
                description="Counts of significant and not significant D-statistics under each multiple‐test correction.",
                pconfig={
                    "id": f"{method}_d_statistics_significance_counts",
                    "title": f"SNPio: {method.capitalize()} D-Statistics Significance Counts",
                    "col1_header": "Significance (Correction)",
                    "scale": "YlOrBr",
                    "namespace": "introgression",
                },
                headers={
                    "Significance (Correction)": {
                        "title": "Significance Status (with Correction)",
                        "description": "Whether the D-statistic passed P < 0.05 under each correction.",
                    },
                    "Count": {
                        "title": "Number of D-statistics",
                        "description": "How many sample‐combinations were significant or not.",
                        "scale": "YlOrBr",
                    },
                },
            )

            self.snpio_mqc.queue_custom_lineplot(
                df=df["Z-Score"],
                panel_id=f"{method}_d_statistics_distribution",
                section="introgression",
                title=f"SNPio: {method.capitalize()} D-Statistics Z-Score Distribution",
                description="Distribution of Z-scores across all sample combinations.",
                index_label="Z-Score Bins",
                pconfig={
                    "id": f"{method}_d_statistics_distribution",
                    "title": f"SNPio: {method.capitalize()} Z-Score Distribution",
                    "xlab": "Z-Score",
                    "ylab": "Estimated Density",
                    "ymin": 0,
                    "xmin": -4.5,
                    "xmax": 4.5,
                },
            )

        else:
            # NOTE: DFOIL is different — build 4×3 table of significance counts
            dnames = ["DFO", "DFI", "DOL", "DIL"]
            rows = []
            for stat in dnames:
                for corr, col in [
                    ("Uncorrected", f"P_{stat}"),
                    ("Bonferroni", f"P_{stat}_bonf"),
                    ("FDR-BH", f"P_{stat}_fdr"),
                ]:
                    sig = (df[col] < 0.05).sum()
                    ns = (df[col] >= 0.05).sum()
                    rows.append(
                        {
                            "Statistic": stat,
                            "Correction": corr,
                            "Significant": sig,
                            "Not Significant": ns,
                        }
                    )
            significance_df = pd.DataFrame(rows).melt(
                id_vars=["Statistic", "Correction"],
                value_vars=["Significant", "Not Significant"],
                var_name="Significance",
                value_name="Count",
            )

            self.snpio_mqc.queue_table(
                df=significance_df,
                panel_id=f"{method}_d_statistics_significance_counts",
                section="introgression",
                title="SNPio: D-FOIL D-Statistics Significance Counts",
                index_label="Statistic / Correction",
                description="Per‐statistic and per‐correction counts of significant vs non‐significant tests for D-FOIL.",
                pconfig={
                    "id": f"{method}_d_statistics_significance_counts",
                    "title": "SNPio: D-FOIL Significance Counts",
                    "col1_header": "Statistic, Correction",
                    "scale": "YlOrBr",
                    "namespace": "introgression",
                },
                headers={
                    "Statistic": {
                        "title": "DFOIL Statistic",
                        "description": "Which sub‐statistic (DFO, DFI, DOL, DIL).",
                    },
                    "Correction": {
                        "title": "P-Value Correction",
                        "description": "Uncorrected, Bonferroni-adjusted, or FDR-BH-adjusted.",
                    },
                    "Count": {
                        "title": "Number of Tests",
                        "description": "Count of sample‐combinations in each category.",
                        "scale": "YlOrBr",
                    },
                },
            )

            # For Z‐score distribution: queue one lineplot per sub‐statistic
            for stat in dnames:
                self.snpio_mqc.queue_custom_lineplot(
                    df=df[f"Z_{stat}"],
                    panel_id=f"{method}_{stat}_z_distribution",
                    section="introgression",
                    title=f"SNPio: D-FOIL {stat} Z-Score Distribution",
                    description=f"Distribution of Z-scores for D-FOIL {stat}.",
                    index_label="Z-Score Bins",
                    pconfig={
                        "id": f"{method}_{stat}_z_distribution",
                        "title": f"D-FOIL {stat} Z-Score Distribution",
                        "xlab": "Z-Score",
                        "ylab": "Estimated Density",
                        "ymin": 0,
                        "xmin": -4.5,
                        "xmax": 4.5,
                    },
                )

    def _format_significance_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Significance (Correction)"] = (
            df["Significance"] + " (" + df["Correction"] + ")"
        )

        df = df.drop(columns=["Significance", "Correction"])

        df["Significance (Correction)"] = df["Significance (Correction)"].str.replace(
            "Significant (Raw)", "Significant (Uncorrected)"
        )

        df["Significance (Correction)"] = df["Significance (Correction)"].str.replace(
            "Not Significant (Raw)", "Not Significant (Uncorrected)"
        )

        return df.set_index("Significance (Correction)")

    def plot_fst_outliers(
        self,
        outlier_snps: pd.DataFrame,
        method: Literal["dbscan", "permutation"],
        max_outliers_to_plot: int | None = None,
    ) -> None:
        """Create a heatmap of Fst values for outlier SNPs, highlighting contributing population pairs.

        Args:
            outlier_snps (pd.DataFrame): DataFrame containing outlier SNPs and their Fst values.
            method (str): Method used for outlier detection ("dbscan" or "permutation").
        """

        # Copy the DataFrame to avoid modifying the original data
        data = outlier_snps.copy()
        data = data.rename(columns={"Locus": "SNP"})

        data["Contributing_Pairs"] = data["Population_Pair"]

        # Create a column for fast lookup
        data["Contributing_Pairs_Set"] = data["Contributing_Pairs"].apply(set)

        # Add boolean column for highlighting
        data["Contributing"] = data.apply(
            lambda row: row["Population_Pair"] in row["Contributing_Pairs_Set"], axis=1
        )

        # Check for duplicates before pivoting
        dupes = data.duplicated(subset=["SNP", "Population_Pair"])
        if dupes.any():
            self.logger.warning(
                f"Found {dupes.sum()} duplicate SNP-PopPair entries. Keeping first only."
            )
            data = data[~dupes]

        # Create mask after deduplicating
        fst_pivot = data.pivot(index="SNP", columns="Population_Pair", values="Fst")

        cols = []
        for col in fst_pivot.columns:
            if col != "Population_Pair":
                cols.append(col.replace("_", "-"))
        fst_pivot.columns = cols

        self.logger.debug(f"Fst pivot table:\n{fst_pivot}")

        # Plot the heatmap
        fig, ax = plt.subplots(1, 1, figsize=(15, max(8, len(fst_pivot) // 2)))

        sns.set_style("white")

        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        if max_outliers_to_plot is not None:
            if len(fst_pivot) > max_outliers_to_plot:
                self.logger.warning(
                    f"More than {max_outliers_to_plot} outlier SNPs detected. Plotting only the first {max_outliers_to_plot} SNPs."
                )
                fst_pivot = fst_pivot.head(max_outliers_to_plot)

        sns.heatmap(
            fst_pivot,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="grey",
            cbar_kws={"label": "Fst"},
            square=False,
            xticklabels=True,
            yticklabels=True,
            ax=ax,
        )

        # Plot title and axis labels
        ax.set_title(
            "Fst Values for Outlier SNPs\nContributing Populations Highlighted"
        )
        ax.set_xlabel("Population Pairs")
        ax.set_ylabel("SNPs")

        # Rotate x-axis labels
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        method = method.lower()

        if method not in {"dbscan", "permutation"}:
            msg = f"Method must be either 'dbscan' or 'permutation', but got: {method}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Save the plot
        of = f"outlier_snps_heatmap_{method}.{self.plot_format}"
        outpath = self.output_dir_analysis / of
        fig.savefig(outpath, bbox_inches="tight")

        if self.show:
            plt.show()
        plt.close()

        if max_outliers_to_plot is None:
            self.snpio_mqc.queue_heatmap(
                df=fst_pivot,
                panel_id=f"fst_outliers_{method}_method",
                section="outliers",
                title=f"SNPio: Fst Outliers ({method.capitalize()} Method)",
                description=f"Fst outliers detected using the {method} method. The table contains the locus names for Fst outliers, adjusted P-values (q_value), contributing population pairs to each outlier, and the observed Weir and Cockerham (1984) Fst values. All outliers are shown. Q-values are adjusted P-values based on the specified multiple test correction method.",
                index_label="Contributing Population Pair",
                pconfig={
                    "id": f"fst_outliers_{method}_method",
                    "title": f"SNPio: Fst Outliers ({method.capitalize()} Method)",
                    "xlab": "Population 1",
                    "ylab": "Population 2",
                    "zlab": "Q-value (Adjusted P-value)",
                    "min": 0.0,
                    "max": 1.0,
                    "display_values": False,
                    "height": 1000,
                },
            )
        elif (
            max_outliers_to_plot is not None
            and isinstance(max_outliers_to_plot, int)
            and max_outliers_to_plot > 0
        ):
            self.snpio_mqc.queue_heatmap(
                df=fst_pivot[["Locus", "q_value", "Population_Pair"]]
                .set_index(["Locus", "Population_Pair"])
                .head(max_outliers_to_plot),
                panel_id=f"fst_outliers_{method}_method",
                section="outliers",
                title=f"SNPio: Fst Outliers ({method.capitalize()} Method)",
                description=f"Fst outliers detected using the {method} method. The table contains the locus names for Fst outliers, adjusted P-values (q_value), contributing population pairs to each outlier, and the observed Weir and Cockerham (1984) Fst values. Due to space limitations, only the top {max_outliers_to_plot} outliers are shown. Q-values are adjusted P-values based on the specified multiple test correction method.",
                index_label="Contributing Population Pair",
                pconfig={
                    "id": f"fst_outliers_{method}_method",
                    "title": f"SNPio: Fst Outliers ({method.capitalize()} Method)",
                    "xlab": "Population 1",
                    "ylab": "Population 2",
                    "zlab": "Q-value (Adjusted P-value)",
                    "min": 0.0,
                    "max": 1.0,
                    "display_values": False,
                    "height": max(max_outliers_to_plot, 1000),
                },
            )
        else:
            msg = f"Invalid max_outliers_to_plot value: {max_outliers_to_plot}. Must be a positive integer or None."
            self.logger.error(msg)
            raise ValueError(msg)

    def plot_summary_statistics(
        self, summary_statistics: dict, use_pvalues: bool = False
    ) -> None:
        """Plot summary statistics per sample and per population.

        This method plots summary statistics per sample and per population on the same figure. The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst). The method also plots summary statistics per sample and per population using Seaborn PairGrid plots. The method saves the plots to the output directory and displays them if ``show`` is True.

        Args:
            summary_statistics (dict): Dictionary containing summary statistics for plotting.
            use_pvalues (bool, optional): If True, display p-values for Fst values. Defaults to False.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

        self._plot_summary_statistics_per_sample(
            summary_statistics["overall"], ax=axes[0]
        )
        self._plot_summary_statistics_per_population(
            summary_statistics["per_population"], ax=axes[1]
        )

        fig.suptitle("Summary Statistics Overview", fontsize=16, y=1.05)
        fig.tight_layout()
        of: str = f"summary_statistics.{self.plot_format}"
        outpath: Path = self.output_dir_analysis / of
        fig.savefig(outpath)

        if self.show:
            plt.show()

        plt.close()

        # Save summary statistics per sample and per population in JSON format
        json_overall: dict = self._make_json_serializable(summary_statistics)

        if not hasattr(self.genotype_data, "marker_names"):
            marker_names = [f"locus_{i}" for i in range(self.genotype_data.num_snps)]
        else:
            marker_names = self.genotype_data.marker_names

        df_sumstats = build_dataframe(
            data=json_overall["overall"],
            index_labels=marker_names,
            index_name="Locus (CHROM:POS)",
            columns_name=" Summary Statistic",
            logger=self.logger,
        )

        df_sumstats = df_sumstats.rename(
            columns={
                "Ho": "Observed Heterozygosity (Ho)",
                "He": "Expected Heterozygosity (He)",
                "Pi": "Nucleotide Diversity (Pi)",
            }
        )

        d = df_sumstats[
            [
                "Observed Heterozygosity (Ho)",
                "Expected Heterozygosity (He)",
                "Nucleotide Diversity (Pi)",
            ]
        ].to_dict(orient="list")

        # 2) Pass them into pconfig when queueing the box plot
        self.snpio_mqc.queue_custom_boxplot(
            df=df_sumstats,
            panel_id="summary_statistics_overall",
            section="detailed_statistics",
            title="SNPio: Summary Statistics",
            index_label="Summary Statistic",
            description=(
                "Box plot of genetic differentiation summary statistics: "
                "(Nucleotide Diversity (Pi), Expected Heterozygosity (He), "
                "Observed Heterozygosity (Ho)). Each overlaid point corresponds to an outlier locus outside the Interquartile Range (IQR)."
            ),
            pconfig={
                "id": "summary_statistics_overall",
                "title": "SNPio: Summary Statistics",
                "series_label": "summary statistics",
            },
        )

        if hasattr(self.genotype_data, "marker_names"):
            marker_names = self.genotype_data.marker_names
        else:
            marker_names = [f"locus_{i}" for i in range(self.genotype_data.num_snps)]

        dflist = []
        for pop_id, pop_data in summary_statistics["per_population"].items():
            dftmp = pop_data.copy()
            dftmp["Population ID"] = pop_id
            dftmp = dftmp.rename_axis("Locus (CHROM:POS)")
            dftmp.index = marker_names
            dftmp = dftmp.set_index(["Population ID", dftmp.index])
            dflist.append(dftmp)

        df_per_pop = pd.concat(dflist)

        df_per_pop = df_per_pop.reset_index().rename(
            columns={"level_1": "Locus (CHROM:POS)"}
        )

        df_per_pop_pivot_ho = df_per_pop[
            ["Locus (CHROM:POS)", "Population ID", "Ho"]
        ].pivot(
            index="Locus (CHROM:POS)",
            columns="Population ID",
            values="Ho",
        )

        df_per_pop_pivot_he = df_per_pop[
            ["Locus (CHROM:POS)", "Population ID", "He"]
        ].pivot(
            index="Locus (CHROM:POS)",
            columns="Population ID",
            values="He",
        )

        df_per_pop_pivot_pi = df_per_pop[
            ["Locus (CHROM:POS)", "Population ID", "Pi"]
        ].pivot(
            index="Locus (CHROM:POS)",
            columns="Population ID",
            values="Pi",
        )

        self.snpio_mqc.queue_custom_boxplot(
            df=df_per_pop_pivot_pi,
            panel_id="summary_statistics_per_population_pi",
            section="detailed_statistics",
            title="SNPio: Per-locus Nucleotide Diversity (Pi) for each Population",
            index_label="Locus (CHROM:POS)",
            description="Box plot of per-locus Nucleotide Diversity (Pi) for each population. Points represent outlier loci outside the box whiskers.",
        )

        self.snpio_mqc.queue_custom_boxplot(
            df=df_per_pop_pivot_he,
            panel_id="summary_statistics_per_population_he",
            section="detailed_statistics",
            title="SNPio: Per-locus Expected Heterozygosity (He) for each Population",
            index_label="Locus (CHROM:POS)",
            description="Box plot of per-locus Expected Heterozygosity (He) for each population. Points represent outlier loci outside the box whiskers.",
        )

        self.snpio_mqc.queue_custom_boxplot(
            df=df_per_pop_pivot_ho,
            panel_id="summary_statistics_per_population_ho",
            section="detailed_statistics",
            title="SNPio: Per-locus Observed Heterozygosity (Ho) for each Population",
            index_label="Locus (CHROM:POS)",
            description="Box plot of per-locus Observed Heterozygosity (Ho) for each population. Points represent outlier loci outside the box whiskers.",
        )

        # Plot Fst heatmap
        self._plot_fst_heatmap(
            summary_statistics["Fst_between_populations_obs"],
            df_fst_lower=summary_statistics["Fst_between_populations_lower"],
            df_fst_upper=summary_statistics["Fst_between_populations_upper"],
            df_fst_pvals=summary_statistics["Fst_between_populations_pvalues"],
            use_pvalues=use_pvalues,
        )

    def _flatten_fst_data(self, fst_dict, stat_name):
        df = pd.DataFrame(fst_dict).T
        df.index.name = "Population"
        df.reset_index(inplace=True)
        return {
            "id": f"fst_{stat_name}",
            "description": f"Pairwise Fst ({stat_name})",
            "headers": df.columns.tolist(),
            "data": df.to_dict(orient="records"),
        }

    def _flatten_overall(self, overall_dict):
        df = pd.DataFrame(overall_dict)
        df = df.reset_index()
        df = df.rename(columns={"index": "Index"})
        return {
            "id": "overall_summary",
            "description": "Overall diversity statistics",
            "headers": df.columns.tolist(),
            "data": df.to_dict(orient="records"),
        }

    def _flatten_per_population(self, per_pop_dict: Dict[str, Any]) -> pd.DataFrame:
        """Turn the nested 'per_population' block into a tidy DataFrame.

        The function is resilient to either dicts OR pandas objects at the
        second level. It will convert DataFrames to dicts, Series to lists,
        and handle other types accordingly.

        Returns:
            pd.DataFrame: A tidy DataFrame with the following columns:
            - population: Population identifier
            - locus: Locus index
            - He: Expected heterozygosity
            - Ho: Observed heterozygosity
        """
        rows: Dict[Dict[str, Any]] = {}

        for pop, stats in per_pop_dict.items():

            # ✦ 1 — Normalise `stats` to a pure dict-of-lists ✦
            if isinstance(stats, pd.DataFrame):
                stats = stats.to_dict(orient="list")
            elif isinstance(stats, pd.Series):
                # series of lists → dict with a single key
                stats = {stats.name: list(stats.values)}
            elif not isinstance(stats, dict):
                msg = f"Expected dict / DataFrame / Series for stats, got {type(stats)}"
                self.logger.error(msg)
                raise TypeError(msg)

            if (
                hasattr(self.genotype_data, "marker_names")
                and self.genotype_data.marker_names is not None
            ):
                n_loci = len(self.genotype_data.marker_names)
                # If marker names are available, use them as locus identifiers
                for locus in range(n_loci):
                    row = {
                        "population": pop,
                        "locus_index": locus,
                    }
                    for stat_name, values in stats.items():
                        row[stat_name] = values[locus]
                    rows[self.genotype_data.marker_names[locus]] = row
            else:
                # ✦ 2 — Use the first entry to determine locus count ✦
                n_loci = len(next(iter(stats.values())))

                # ✦ 3 — Build long-format rows ✦
                for locus in range(n_loci):
                    row = {"population": pop, "locus_index": locus}
                    for stat_name, values in stats.items():
                        row[stat_name] = values[locus]
                    rows[f"locus_{locus}"] = row
        return rows

    def _make_json_serializable(self, obj: dict) -> dict:
        """Convert an object to a JSON-serializable format.

        This method converts an object to a JSON-serializable format by converting DataFrames to dictionaries, Series to lists, and handling other types accordingly.

        Args:
            obj (dict): The object to convert.

        Returns:
            dict: The converted object in a JSON-serializable format.
        """
        d = {}
        for key, value in obj.items():
            if isinstance(value, pd.DataFrame):
                d[key] = value.to_dict(orient="list")
            elif isinstance(value, dict):
                d[key] = {
                    k: (
                        v.to_dict(orient="list")
                        if isinstance(v, (pd.DataFrame, pd.Series))
                        else v
                    )
                    for k, v in value.items()
                }
            else:
                d[key] = value

        return d

    def _plot_sankey_filtering_report(
        self, df: pd.DataFrame, search_mode: bool = False, fn: str | None = None
    ) -> None:
        """Plot a Sankey diagram for the filtering report.

        This method plots a Sankey diagram for the filtering report. The Sankey diagram shows the flow of loci through the filtering steps. The loci are filtered based on the missing data proportion, MAF, MAC, and other filtering thresholds. The Sankey diagram shows the number of loci kept and removed at each step. The Sankey diagram is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/nremover/plots/sankey_plot_{thresholds}.{plot_format}``. The plot is saved in the format specified by the ``plot_format`` attribute.

        Args:
            df (pd.DataFrame): The input DataFrame containing the filtering report.
            search_mode (bool, optional): Whether the Sankey diagram is being plotted in search mode. Defaults to False.
            fn (str | None, optional): The filename to save the plot. If None, the default filename is used. Defaults to None.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input DataFrame is empty.
            ValueError: Raised if multiple threshold combinations are detected when attempting to plot the Sankey diagram.

        Example:
            >>> from snpio import VCFReader, NRemover
            >>> gd = VCFReader(filename="example.vcf", popmapfile="popmap.txt")
            >>> nrm = NRemover(gd)
            >>> nrm.filter_missing(0.75).filter_mac(2).filter_missing_pop(0.5).filter_singletons(exclude_heterozygous=True).filter_monomorphic().filter_biallelic().resolve()
            >>> nrm.plot_sankey_filtering_report()
        """
        with warnings.catch_warnings(action="ignore"):
            # Import Holoviews and Bokeh
            hv.extension("bokeh")

        plot_dir: Path = self.output_dir_gd / "sankey_plots"
        plot_dir.mkdir(exist_ok=True, parents=True)

        # Copy the DataFrame to avoid modifying the original
        df = df.copy()

        # Filter out the missing sample filter method
        df = df[df["Filter_Method"] != "filter_missing_sample"]

        if df.empty:
            msg = "No data to plot. Please check the filtering thresholds."
            self.logger.error(msg)
            raise ValueError(msg)

        # Ensure correct data types
        # Round float columns to 2 or 3 decimal places.
        df["Missing_Threshold"] = df["Missing_Threshold"].astype(float).round(3)
        df["MAF_Threshold"] = df["MAF_Threshold"].astype(float).round(3)
        df["Kept_Prop"] = df["Kept_Prop"].astype(float).round(2)
        df["Removed_Prop"] = df["Removed_Prop"].astype(float).round(2)

        df["MAC_Threshold"] = df["MAC_Threshold"].astype(int)
        df["Bool_Threshold"] = df["Bool_Threshold"].astype(int)
        df["Removed_Count"] = df["Removed_Count"].astype(int)
        df["Kept_Count"] = df["Kept_Count"].astype(int)
        df["Total_Loci"] = df["Total_Loci"].astype(int)
        df["Step"] = df["Step"].astype(int)

        # Create a new column for the threshold combination
        df["Threshold"] = (
            df["Missing_Threshold"].astype(str)
            + "_"
            + df["MAF_Threshold"].astype(str)
            + "_"
            + df["Bool_Threshold"].astype(str)
            + "_"
            + df["MAC_Threshold"].astype(str)
        )

        # Sort the DataFrame
        # Get thresholds as list.
        thresholds = df["Threshold"].tolist()

        if len(thresholds) == 0:
            raise ValueError("No data to plot. Please check the filtering thresholds.")
        elif len(thresholds) > 1:
            if search_mode:
                msg = "Multiple threshold combinations detected when attempting to plot the Sankey diagram."
                self.logger.error(msg)
                raise ValueError(msg)

        if search_mode:
            thresholds = "_".join([str(value) for value in thresholds[0].split("_")])
        else:
            thresholds = "_".join(
                [
                    str(value)
                    for threshold in thresholds
                    for value in threshold.split("_")
                ]
            )

        self.logger.debug(f"Thresholds: {thresholds}")

        if search_mode:
            # Sort the DataFrame
            df = df.sort_values(by=["Threshold", "Step"]).reset_index(drop=True)

        # Filter DataFrame for the current combination of thresholds
        dftmp: pd.DataFrame = df[df["Filter_Method"] != "filter_missing_sample"]

        self.logger.debug(f"Filtering report for thresholds: {thresholds}")

        # Assign colors
        dftmp["LinkColor_Kept"] = "#2ca02c"  # Green for kept loci
        dftmp["LinkColor_Removed"] = "#d62728"  # Red for removed loci

        # Sort the DataFrame by step.
        dftmp = dftmp.sort_values(by="Step").reset_index(drop=True)

        # Build the flows with a common "Removed" node and edge labels
        flows = []

        for i in dftmp.index:
            # Use a common "Unfiltered" node for the initial unfiltered loci
            source = "Unfiltered" if i == 0 else dftmp.loc[i - 1, "Filter_Method"]

            # Use a common "Kept" node for the final kept loci
            target = dftmp.loc[i, "Filter_Method"]
            kept_count = dftmp.loc[i, "Kept_Count"]
            removed_count = dftmp.loc[i, "Removed_Count"]
            link_color_kept = dftmp.loc[i, "LinkColor_Kept"]
            link_color_removed = dftmp.loc[i, "LinkColor_Removed"]

            # Use a common "Removed" node
            removed_target = "Removed"

            # Flow for removed loci at this step
            flows.append(
                {
                    "Source": source,
                    "Target": removed_target,
                    "Count": removed_count,
                    "LinkColor": link_color_removed,
                    "EdgeLabel": f"{target.replace('_', ' ').title()} Removed",
                }
            )

            # Flow for kept loci moving to the next filter
            flows.append(
                {
                    "Source": source,
                    "Target": target,
                    "Count": kept_count,
                    "LinkColor": link_color_kept,
                    "EdgeLabel": f"{target.replace('_', ' ').title()} Kept",
                }
            )

        # Ensure the last step flows into "Kept"
        final_source = dftmp.iloc[-1]["Filter_Method"]
        final_kept_count = dftmp.iloc[-1]["Kept_Count"]
        final_link_color_kept = dftmp.iloc[-1]["LinkColor_Kept"]

        flows.append(
            {
                "Source": final_source,
                "Target": "Kept",
                "Count": final_kept_count,
                "LinkColor": final_link_color_kept,
                "EdgeLabel": "Kept",
            }
        )

        # Create DataFrame for flows
        dftmp_combined = pd.DataFrame(flows)

        self.logger.debug(f"Sankey plot data: {dftmp_combined}")

        try:
            with warnings.catch_warnings(action="ignore"):
                warnings.filterwarnings(action="ignore")

                # Create the Sankey plot with edge labels and colors.
                sankey_plot = hv.Sankey(
                    dftmp_combined,
                    kdims=[
                        hv.Dimension("Source", label="EdgeLabel"),
                        hv.Dimension("Target"),
                    ],
                    vdims=["Count", "LinkColor", "EdgeLabel"],
                ).opts(
                    opts.Sankey(
                        width=800,
                        height=600,
                        edge_color="LinkColor",
                        node_color="blue",
                        node_padding=20,
                        label_position="left",
                        fontsize={"labels": "8pt", "title": "12pt"},
                    )
                )

                if fn is not None and not isinstance(fn, str):
                    msg = f"Filename must be a string, but got: {type(fn)}"
                    self.logger.error(msg)
                    raise TypeError(msg)

                # Save the plot to an HTML file
                of: str = "filtering_results_sankey_mqc.html" if fn is None else fn
                fname: Path = plot_dir / of

                if not fname.suffix == ".html":
                    fname = fname.with_suffix(".html")

                if not fname.stem.endswith("_mqc"):
                    fname = fname.with_name(f"{fname.stem}_mqc{fname.suffix}")

                hv.save(sankey_plot, fname, fmt="html")

        except Exception as e:
            self.logger.warning(
                f"Failed to generate Sankey plot with thresholds: {','.join(thresholds)}: error: {e}"
            )

            # Save the plot to an HTML file
        of: str = "filtering_results_sankey_mqc.html" if fn is None else fn
        fname: Path = plot_dir / of

        if not fname.suffix == ".html":
            fname = fname.with_suffix(".html")

        if not fname.stem.endswith("_mqc"):
            fname = fname.with_name(f"{fname.stem}_mqc{fname.suffix}")

        fname.parent.mkdir(exist_ok=True, parents=True)

        if not fname.exists() or not fname.is_file():
            msg = f"Failed to save Sankey plot to {fname}. Please check the output directory."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.snpio_mqc.queue_html(
            html=fname,
            panel_id="sankey_html",
            section="filtering",
            title="SNPio: Sankey Filtering Report",
            index_label="Filter Method",
            description=(
                "Sankey filtering report showing the flow of loci through the filtering steps. Retained loci are shown in green, while removed loci are shown in red. The Sankey diagram shows the number of loci kept and removed at each step."
            ),
        )

        self.logger.info(f"Sankey filtering report diagram saved to: {fname}")

    def plot_gt_distribution(self, df: pd.DataFrame, annotation_size: int = 15) -> None:
        """Plot the distribution of genotype counts.

        This method plots the distribution of genotype counts as a bar plot. The bar plot shows the genotype counts for each genotype. The plot is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/gtdata/plots/genotype_distribution.{plot_format}``. The plot is saved in the format specified by the `plot_format` attribute.

        Args:
            df (pd.DataFrame): The input dataframe containing the genotype counts.

            annotation_size (int, optional): The font size for count annotations. Defaults to 15.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.

        Note:
            The input dataframe must contain the genotype counts.
        """
        # Validate the input dataframe
        df = misc.validate_input_type(df, return_type="df")

        df_melt: pd.DataFrame = pd.melt(df, value_name="Count")
        cnts = df_melt["Count"].value_counts()
        cnts.index.names = ["Genotype Int"]
        cnts = pd.DataFrame(cnts).reset_index()
        cnts = cnts.sort_values(by="Genotype Int")
        cnts["Genotype Int"] = cnts["Genotype Int"].astype(str)

        int_iupac_dict = self.iupac.int_iupac_dict
        int_iupac_dict = {str(v): k for k, v in int_iupac_dict.items()}
        cnts["Genotype"] = cnts["Genotype Int"].map(int_iupac_dict)
        cnts.columns = [col[0].upper() + col[1:] for col in cnts.columns]

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        g: plt.Axes = sns.barplot(
            x="Genotype", y="Count", data=cnts, ax=ax, color="orange"
        )
        g.set_xlabel("Genotype")
        g.set_ylabel("Count")
        g.set_title("Genotype Counts")
        g.tick_params(axis="both", labelsize=self.ticksize)
        for p in g.patches:
            g.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + 0.075, p.get_height() + 0.01),
                xytext=(0, 1),
                textcoords="offset points",
                va="bottom",
                fontsize=annotation_size,
            )

        of: str = f"genotype_distribution.{self.plot_format}"
        outpath: Path = self.output_dir_gd / of
        fig.savefig(outpath)

        if self.show:
            plt.show()
        plt.close()

        self.snpio_mqc.queue_barplot(
            df=cnts,
            panel_id="genotype_distribution",
            section="genotype_distribution",
            title="SNPio: Genotype Distribution",
            description=(
                "Bar plot showing the distribution of genotype counts across all samples."
            ),
            index_label="Genotype Int",
        )

        self.logger.info(f"Genotype distribution plot saved to: {outpath}")

    def plot_search_results(self, df_combined: pd.DataFrame) -> None:
        """Plot and save the filtering results based on the available data.

        This method plots the filtering results based on the available data. The filtering results are plotted for the per-sample and per-locus missing data proportions, MAF, and boolean filtering thresholds. The plots are saved to files in the output directory. If the `show` attribute is True, the plots are displayed. The plots are saved in the format specified by the `plot_format` attribute into the `output_dir` directory in the format: ``<prefix>_output/gtdata/plots/filtering_results_{method}.{plot_format}``.

        Args:
            df_combined (pd.DataFrame): The input dataframe containing the filtering results.

        Raises:
            ValueError: Raised if the input dataframe is empty.

        Note:
            - The input dataframe must contain the filtering results.
            - The input dataframe must contain the filtering results for the per-sample and per-locus missing data proportions.
            - The input dataframe must contain the filtering results for the MAF and boolean filtering thresholds.
            - The input dataframe must contain the filtering results for the removed and kept loci proportions.
            - The input dataframe must contain the filtering results for the removed and kept loci counts.
            - The input dataframe must contain the filtering results for the filtering method.
            - The input dataframe must contain the filtering results for the filtering step.
            - The input dataframe must contain the filtering results for the filtering thresholds.
            - The input dataframe must contain the filtering results for the removed and kept loci counts.
            - The input dataframe must contain the filtering results for the removed and kept loci proportions.
        """
        if df_combined.empty:
            msg = "No data to plot. Please check the filtering thresholds."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Plotting search results.")
        self.logger.debug(f"Combined data: {df_combined}")

        df_combined["Missing_Threshold"] = df_combined["Missing_Threshold"].round(2)
        df_combined["MAF_Threshold"] = df_combined["MAF_Threshold"].round(2)
        df_combined["Bool_Threshold"] = df_combined["Bool_Threshold"].round(2)
        df_combined["Removed_Prop"] = df_combined["Removed_Prop"].round(2)
        df_combined["Kept_Prop"] = df_combined["Kept_Prop"].round(2)

        # Existing plotting methods
        self._plot_combined(df_combined)
        self._plot_pops(df_combined)
        self._plot_maf(df_combined)
        self._plot_boolean(df_combined)

        msg: str = f"Plotting complete. Plots saved to directory {self.output_dir_gd}."
        self.logger.info(msg)

        self.snpio_mqc.queue_table(
            df=df_combined.reset_index(drop=True),
            panel_id="filtering_results_combined",
            section="filtering",
            title="SNPio: Combined NRemover2 Filtering Results",
            index_label="Filter Method",
            description=(
                "Combined filtering results showing the proportion of loci removed and kept for each filtering method."
            ),
            pconfig={
                "id": "filtering_results_combined",
                "title": "SNPio: Combined NRemover2 Filtering Results",
                "series_label": "Filter Method",
                "scale": "YlOrBr",
            },
        )

    def _plot_combined(self, df: pd.DataFrame) -> None:
        """Plot missing data proportions for Sample and Global data.

        This method plots the missing data proportions for Sample and locus-level missing data. The plot shows the proportion of loci removed and kept for each missing data threshold. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The input dataframe containing the missing data proportions.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.

        Note:
            - The input dataframe must contain the missing data proportions.
            - The input dataframe must contain the missing data proportions for the Sample and Global data.
            - The input dataframe must contain the missing data proportions for the removed and kept loci.
            - The input dataframe must contain the missing data proportions for the missing data threshold.
            - The input dataframe must contain the missing data proportions for the filtering method.
            - The input dataframe must contain the missing data proportions for the filtering step.
            - The input dataframe must contain the missing data proportions for the removed and kept loci counts.
            - The input dataframe must contain the missing data proportions for the removed and kept loci proportions.
            - The input dataframe must contain the missing data proportions for the filtering thresholds.
        """
        df = df[
            df["Filter_Method"].isin(["filter_missing", "filter_missing_sample"])
        ].copy()

        if not df.empty:
            self.logger.info("Plotting global per-locus filtering results.")
            self.logger.debug(f"Missing data: {df}")

            fig, axs = plt.subplots(1, 2, figsize=(10, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax: plt.Axes = sns.lineplot(
                    x="Missing_Threshold",
                    y=ycol,
                    hue="Filter_Method",
                    palette="Dark2",
                    markers=False,
                    data=df,
                    ax=ax,
                )

                ylab: str = ycol.split("_")[0].capitalize()

                ax.set_xlabel("Filtering Threshold")
                ax.set_ylabel(f"{ylab} Proportion")
                ax.set_title(f"{ylab} Data")
                ax.legend(title="Filter Method")
                ax.set_ylim(-0.05, 1.12)
                ax.set_xlim(0, 1)

                ax.set_xticks(
                    df["Missing_Threshold"].astype(float).unique(), minor=False
                )

                ax.legend(
                    title="Filter Method", bbox_to_anchor=(0.5, 1.2), loc="center"
                )

            of: str = f"filtering_results_missing_loci_samples.{self.plot_format}"
            fig.savefig(self.output_dir_gd / of)

            if self.show:
                plt.show()

            plt.close()

        else:
            self.logger.info("Missing data filtering results ares empty.")

    def _plot_pops(self, df: pd.DataFrame) -> None:
        """Plot population-level missing data proportions.

        This method plots the proportion of loci removed and kept for each population-level missing data threshold. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The input dataframe containing the population-level missing data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.

        Note:
            - The input dataframe must contain the population-level missing data proportions.
            - The input dataframe must contain the population-level missing data proportions for the removed and kept loci.
            - The input dataframe must contain the population-level missing data proportions for the missing data threshold.
            - The input dataframe must contain the population-level missing data proportions for the filtering method.
            - The input dataframe must contain the population-level missing data proportions for the filtering step.
            - The input dataframe must contain the population-level missing data proportions for the removed and kept loci counts.
            - The input dataframe must contain the population-level missing data proportions for the removed and kept loci proportions.
            - The input dataframe must contain the population-level missing data proportions for the filtering thresholds.
        """
        df = df[df["Filter_Method"] == "filter_missing_pop"].copy()

        self.logger.debug(f"Population-level missing data: {df}")

        if not df.empty:
            self.logger.info("Plotting population-level missing data.")
            self.logger.debug(f"Population-level missing data: {df}")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax: plt.Axes = sns.lineplot(
                    x="Missing_Threshold",
                    y=ycol,
                    data=df,
                    ax=ax,
                    color=sns.color_palette("Dark2")[0],
                    markers=False,
                    linewidth=2,
                    linestyle="-",
                    legend=False,
                )

                ylab: str = ycol.split("_")[0].capitalize()

                ax.set_xlabel("Filtering Threshold")
                ax.set_ylabel(f"{ylab} Proportion")
                ax.set_title(f"{ylab} Data")
                ax.set_ylim(0, 1.12)
                ax.set_xticks(
                    df["Missing_Threshold"].astype(float).unique(), minor=False
                )

            of: str = f"filtering_results_missing_population.{self.plot_format}"
            outpath: Path = self.output_dir_gd / of
            fig.savefig(outpath)

            if self.show:
                plt.show()

            plt.close()

        else:
            self.logger.info("Population-level missing data is empty.")

    def _plot_maf(self, df: pd.DataFrame) -> None:
        """Plot MAF filtering data.

        This method plots the MAF filtering data. The MAF filtering data includes the proportion of loci removed and kept for each MAF threshold. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The input dataframe containing the MAF data.

        Raises:
            ValueError: Raised if the input dataframe is empty.

        Note:
            - The input dataframe must contain the MAF filtering data.
            - The input dataframe must contain the MAF filtering data for the removed and kept loci.
            - The input dataframe must contain the MAF filtering data for the MAF threshold.
            - The input dataframe must contain the MAF filtering data for the filtering method.
            - The input dataframe must contain the MAF filtering data for the filtering step.
            - The input dataframe must contain the MAF filtering data for the removed and kept loci counts.
            - The input dataframe must contain the MAF filtering data for the removed and kept loci proportions.
            - The input dataframe must contain the MAF filtering data for the filtering thresholds
        """
        df_mac: pd.DataFrame = df[df["Filter_Method"] == "filter_mac"].copy()
        df = df[df["Filter_Method"] == "filter_maf"].copy()

        self.logger.debug(f"MAF data: {df}")
        self.logger.debug(f"MAC data: {df_mac}")

        if not df.empty:
            self.logger.info("Plotting minor allele frequency data.")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax: plt.Axes = sns.lineplot(
                    x="MAF_Threshold",
                    y=ycol,
                    data=df,
                    color=sns.color_palette("Dark2")[0],
                    markers=False,
                    linewidth=2,
                    linestyle="-",
                    legend=False,
                    ax=ax,
                )

                ylab: str = ycol.split("_")[0].capitalize()

                ax.set_xlabel("Filtering Threshold")
                ax.set_ylabel(f"{ylab} Proportion")
                ax.set_title(f"{ylab} Data")
                ax.set_ylim(-0.05, 1.12)
                ax.set_xticks(df["MAF_Threshold"].astype(float).unique(), minor=False)

            of: str = f"filtering_results_maf.{self.plot_format}"
            outpath: Path = self.output_dir_gd / of
            fig.savefig(outpath)

            if self.show:
                plt.show()

            plt.close()

        else:
            self.logger.info("MAF data is empty.")

        if not df_mac.empty:
            self.logger.info("Plotting minor allele count data.")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax = sns.lineplot(
                    x="MAC_Threshold",
                    y=ycol,
                    data=df_mac,
                    color=sns.color_palette("Dark2")[0],
                    markers=False,
                    linewidth=2,
                    linestyle="-",
                    legend=False,
                    ax=ax,
                )

                ylab = ycol.split("_")[0].capitalize()

                ax.set_xlabel("Filtering Threshold")
                ax.set_ylabel(f"{ylab} Count")
                ax.set_title(f"{ylab} Data")
                ax.set_ylim(-0.05, 1.12)
                ax.set_xticks(df_mac["MAC_Threshold"].astype(int).unique(), minor=False)

            of: str = f"filtering_results_mac.{self.plot_format}"
            outpath: Path = self.output_dir_gd / of
            fig.savefig(outpath)

            if self.show:
                plt.show()

            plt.close()
        else:
            self.logger.info("MAC data is empty.")

    def _plot_boolean(self, df: pd.DataFrame) -> None:
        """Plot boolean datasets, including: Monomorphic, Biallelic, Thin Loci, Singleton, and Linked.

        This method plots the boolean filtering data. The boolean filtering data includes the proportion of loci removed and kept for each boolean filtering threshold. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The input dataframe containing the boolean data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.

        Note:
            - The input dataframe must contain the boolean filtering data.
            - The input dataframe must contain the boolean filtering data for the removed and kept loci.
            - The input dataframe must contain the boolean filtering data for the boolean filtering threshold.
            - The input dataframe must contain the boolean filtering data for the filtering method.
            - The input dataframe must contain the boolean filtering data for the filtering step.
            - The input dataframe must contain the boolean filtering data for the removed and kept loci counts.
            - The input dataframe must contain the boolean filtering data for the removed and kept loci proportions.
            - The input dataframe must contain the boolean filtering data for the filtering thresholds.
        """
        df = df[df["Filter_Method"].isin(self.boolean_filter_methods)].copy()

        if not df.empty:
            self.logger.info("Plotting boolean filtering data.")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax: plt.Axes = sns.lineplot(
                    x="Bool_Threshold",
                    y=ycol,
                    data=df,
                    hue="Filter_Method",
                    palette="Dark2",
                    markers=False,
                    linewidth=2,
                    linestyle="-",
                    ax=ax,
                )

                ylab: str = ycol.split("_")[0].capitalize()

                ax.set_xlabel("Heterozygous Genotypes")
                ax.set_ylabel(f"{ylab} Proportion")
                ax.set_title(f"{ylab} Data")
                ax.set_ylim(-0.05, 1.12)
                ax.set_xlim(0, 1)
                ax.set_xticks([0.0, 1.0], minor=False)
                ax.set_xticklabels(
                    labels=["Included", "Excluded"], rotation=45, minor=False
                )

                ax.legend(
                    title="Filter Method", loc="center", bbox_to_anchor=(0.5, 1.2)
                )

            of: str = f"filtering_results_bool.{self.plot_format}"
            outpath: Path = self.output_dir_gd / of
            fig.savefig(outpath)

            if self.show:
                plt.show()

            plt.close()

        else:
            self.logger.info("Boolean data is empty.")

    def plot_pop_counts(self, populations: pd.Series) -> None:
        """Plot the population counts.

        This function takes a series of population data and plots the counts and proportions of each population ID. The resulting plot is saved to a file of the specified format. The plot shows the counts and proportions of each population ID. The plot is colored based on the median count and proportion.

        Args:
            populations (pd.Series): The series containing population data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input data is not a pandas Series.

        Note:
            - The population data should be in the format of a pandas Series.
            - The plot will be saved in the '<prefix>_output/gtdata/plots' directory.
            - Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").
            - The plot will be colored based on the median count and proportion.
            - The plot will show the counts and proportions of each population ID.
            - The plot will show the counts and proportions of each population ID.
        """
        # Create the countplot
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))

        if not isinstance(populations, pd.Series):
            populations = pd.Series(populations)

        # Calculate the counts and proportions
        counts = populations.value_counts()
        proportions = counts / len(populations)

        # Calculate the median count and proportion
        median_count = np.median(counts)
        median_proportion = np.median(proportions)

        colors = sns.color_palette("colorblind")

        df = pd.DataFrame({"Population ID": counts.index, "Count": counts})
        df_perc = pd.DataFrame(
            {"Population ID": proportions.index, "Proportion": proportions}
        )

        # Convert to percentage
        df_perc = df_perc.set_index("Population ID").mul(100).reset_index()
        df_perc.columns = ["Population ID", "Percentage"]

        color = "#8551A8"
        unique_y = df["Population ID"].unique().tolist()
        cats = {
            "Count": {"name": "Count", "color": color},
            "Proportion": {"name": "Proportion", "color": color},
        }

        self.snpio_mqc.queue_barplot(
            df=[df, df_perc],
            panel_id="population_counts",
            section="overview",
            title="SNPio: Population Counts",
            description="Counts samples belonging to each population.",
            index_label="Population ID",
            pconfig={
                "id": "population_counts",
                "title": "SNPio: Population Counts",
                "cpswitch": False,
                "cpswitch_c_active": False,
                "data_labels": [
                    {
                        "name": "Count",
                        "ylab": "Count",
                        "title": "Counts of Samples in each Population",
                    },
                    {
                        "name": "Percentage",
                        "ylab": "Percentage",
                        "ymax": 100,
                        "title": "Percent of Samples in each Population",
                    },
                ],
            },
        )

        for ax, data, ylabel, median, color, median_color in zip(
            axs,
            [counts, proportions],
            ["Count", "Proportion"],
            [median_count, median_proportion],
            [colors[1], colors[0]],
            [colors[0], colors[1]],
        ):
            ax: plt.Axes = sns.barplot(x=data.index, y=data.values, color=color, ax=ax)
            median_line: plt.Line2D = ax.axhline(
                median, color=median_color, linestyle="--"
            )

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(labels=ax.get_xticklabels(), minor=False, rotation=90)
            ax.set_title("Population Counts")
            ax.set_xlabel("Population ID")
            ax.set_ylabel(ylabel)
            ax.legend([median_line], ["Median"], loc="upper right")

        of: Path = self.output_dir_gd / f"population_counts.{self.plot_format}"
        fig.savefig(of)

        if self.show:
            plt.show()

        plt.close()

    def visualize_missingness(
        self,
        df: pd.DataFrame,
        prefix: str | None = None,
        zoom: bool = False,
        bar_color: str = "gray",
        heatmap_palette: str = "magma",
    ) -> "MissingStats":
        """Visualize missing data across loci, individuals, and populations.

        This method generates a series of bar plots and heatmaps to visualize the missing data statistics across individuals, loci, and populations. It calculates the missing proportions and creates visualizations to help identify patterns of missingness in the genotype data.

        Returns:
            MissingStats: The dataclass bundle containing all missing value statistics. It contains the following attributes:
            - per_individual: Series with missing proportions per individual.
            - per_locus: Series with missing proportions per locus.
            - per_population: Series with missing proportions per population.
            - per_individual_population: DataFrame with missing proportions per individual and population.
            - per_population_locus: DataFrame with missing proportions per population and locus.
        """
        self.logger.info("Generating missingness report...")

        prefix = self.prefix if prefix is None else prefix

        if not isinstance(df, pd.DataFrame):
            df = misc.validate_input_type(df, return_type="df")

        use_pops = False if self.genotype_data.popmapfile is None else True
        stats = self.genotype_data.calc_missing(df, use_pops=use_pops)

        ncol = 3
        nrow = 1 if stats.per_population is None else 2

        mpl.rcParams.update({"savefig.bbox": "standard"})

        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(12, 10 if nrow == 1 else 14),
            constrained_layout=True,
        )
        fig.suptitle("Missingness Report")

        # Plot 1: Per-Individual barplot
        ax = axes[0, 0]
        ax.barh(stats.per_individual.index, stats.per_individual, color=bar_color)
        ax.set_xlim([0, 1] if not zoom else None)
        ax.set_xlabel("Missing Prop.")
        ax.set_ylabel("Sample")
        ax.tick_params(axis="y", labelleft=False)

        # Plot 2: Per-Locus barplot
        ax = axes[0, 1]
        ax.barh(stats.per_locus.index, stats.per_locus, color=bar_color)
        ax.set_xlim([0, 1] if not zoom else None)
        ax.set_xlabel("Missing Prop.")
        ax.set_ylabel("Locus")
        ax.tick_params(axis="y", labelleft=False)

        # Plot 3: Per-Population totals (if available)
        if stats.per_population is not None:
            ax = axes[0, 2]
            ax.barh(stats.per_population.index, stats.per_population, color=bar_color)
            ax.set_xlim([0, 1] if not zoom else None)
            ax.set_xlabel("Missing Prop.")
            ax.set_ylabel("Population")

            # Plot 4: Heatmap of Pop x Locus missingness
            ax = axes[1, 0]
            vmax = None if zoom else 1.0
            sns.heatmap(
                stats.per_population_locus,
                vmin=0.0,
                vmax=vmax,
                cmap=sns.color_palette(heatmap_palette, as_cmap=True),
                yticklabels=False,
                cbar_kws={"label": "Missing Prop."},
                ax=ax,
            )
            ax.set_xlabel("Population")
            ax.set_ylabel("Locus")

            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.label.set_va("bottom")  # Align properly

        # Plot 5: Per-Individual heatmap (reshaped)
        ax = axes[0, 2] if stats.per_population is None else axes[1, 1]

        melt_df = stats.per_individual_population.reset_index()
        melt_df = melt_df.melt(
            id_vars=melt_df.columns[:2], var_name="Locus", value_name="Missing"
        )
        melt_df["Missing"] = melt_df["Missing"].replace(
            {False: "Present", True: "Missing"}
        )

        sns.histplot(
            data=melt_df,
            y="Locus",
            hue="Missing",
            hue_order=["Present", "Missing"],
            multiple="fill",
            ax=ax,
        )
        ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Proportion")
        ax.set_ylabel("Locus")
        ax.get_legend().set_title(None)

        # Plot 6: Per-Population heatmap (stacked)
        if stats.per_population is not None:
            ax = axes[1, 2]
            sns.histplot(
                data=melt_df,
                y="population",
                hue="Missing",
                hue_order=["Present", "Missing"],
                multiple="fill",
                ax=ax,
            )

            ax.set_xlabel("Proportion")
            ax.get_legend().set_title(None)

        # Save figure
        out_path = self.output_dir_gd / f"missingness_report.{self.plot_format}"
        fig.savefig(out_path)

        if self.show:
            plt.show()
        plt.close()

        mpl.rcParams.update({"savefig.bbox": "tight"})

        return stats

    def plot_dist_matrix(
        self,
        df: pd.DataFrame,
        *,
        pvals: pd.DataFrame | None = None,
        palette: str = "coolwarm",
        title: str = "Distance Matrix",
        dist_type: str = "fst",
    ) -> None:
        """Plot distance matrix.

        This method plots a distance matrix using seaborn's heatmap function. The distance matrix is calculated from the input DataFrame. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The input DataFrame containing the distance matrix data.
            pvals (pd.DataFrame, optional): The p-values for the distance matrix. Defaults to None.
            palette (str, optional): The color palette to use for the heatmap. Defaults to "coolwarm".
            title (str, optional): The title of the plot. Defaults to "Distance Matrix".
        """
        self._plot_fst_heatmap(
            df,
            df_fst_pvals=pvals,
            use_pvalues=True,
            palette=palette,
            title=title,
            dist_type=dist_type,
        )
