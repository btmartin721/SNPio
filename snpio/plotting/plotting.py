import json
import warnings
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

import holoviews as hv
import matplotlib as mpl
import plotly.express as px

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from holoviews import opts
from mpl_toolkits.mplot3d import Axes3D  # Don't remove this import.

hv.extension("bokeh")

from snpio.utils import misc
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC, build_dataframe
from snpio.utils.multiqc_reporter import SNPioMultiQC

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData
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
        genotype_data: "GenotypeData",
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
            - The `show`, `plot_format`, `dpi`, `plot_fontsize`, `plot_title_fontsize`, `despine`, `verbose`, and `debug` attributes are set based on the provided values, the `genotype_data` object, or default values.

            - The `output_dir` attribute is set to the `prefix_output/nremover/plots` directory or the `prefix_output/plots` directory if the genotype data was not filtered when initializing the `Plotting` class.

            - The `mpl_params` dictionary contains default Matplotlib parameters for the plots and are updated with the `mpl_params` dictionary.

            - The `plotting` object is used to set the attributes based on the provided values, the `genotype_data` object, or default values.
        """
        self.genotype_data = genotype_data
        self.prefix: str = getattr(genotype_data, "prefix", "plot")

        self.output_dir: Path = Path(f"{self.prefix}_output")
        if self.genotype_data.was_filtered:
            self.output_dir: Path = self.output_dir / "nremover"

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
        if isinstance(df_fst_mean, pd.DataFrame):
            df_fst_mean = df_fst_mean.copy()

        if df_fst_lower is not None and isinstance(df_fst_lower, pd.DataFrame):
            df_fst_lower = df_fst_lower.copy()

        if df_fst_upper is not None and isinstance(df_fst_upper, pd.DataFrame):
            df_fst_upper = df_fst_upper.copy()

        if df_fst_pvals is not None and isinstance(df_fst_pvals, pd.DataFrame):
            df_fst_pvals = df_fst_pvals.copy()

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

    def _prepare_d_stats_plotting_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the D-stats DataFrame for plotting: ensure 'Sample Combo' exists, drop non-finite values in key columns, and de-duplicate by (Method, Sample Combo)."""
        # Normalize index -> column if needed
        if df.index.name is not None and df.index.name.lower() in [
            "quartet",
            "quintet",
            "sample combo",
        ]:
            df = df.reset_index().rename(columns={df.index.name: "Sample Combo"})

        # Key stat columns (present ones only)
        key_stat_cols = [c for c in df.columns if c.startswith(("Z_", "P_", "D", "X2"))]

        # Remove infs/nans from key columns
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=key_stat_cols)
        if len(df_clean) < len(df):
            self.logger.warning(
                f"Removed {len(df) - len(df_clean)} rows with NaN/Inf values before plotting."
            )

        # De-duplicate (common source of "plotted twice" symptoms)
        if "Method" in df_clean.columns:
            sample_key = (
                "Sample Combo"
                if "Sample Combo" in df_clean.columns
                else ("Quartet" if "Quartet" in df_clean.columns else None)
            )
            if sample_key is not None:
                before = len(df_clean)
                df_clean = df_clean.drop_duplicates(subset=["Method", sample_key])
                if len(df_clean) < before:
                    self.logger.warning(
                        f"Removed {before - len(df_clean)} duplicate rows."
                    )

        return df_clean

    def _get_significance_counts_df(
        self,
        df: pd.DataFrame,
        d_stats: list[str],
        method_name: Literal["patterson", "partitioned", "dfoil"],
    ) -> pd.DataFrame:
        """Calculates significance counts for different correction levels."""
        rows = []
        for stat in d_stats:
            stat_key = {"D-statistic": "D"}.get(stat, stat)
            sig_col_raw = f"Significant (Raw){'' if method_name == 'patterson' else f' {stat_key}'}"
            sig_col_bonf = f"Significant (Bonferroni){'' if method_name == 'patterson' else f' {stat_key}'}"
            sig_col_fdr = f"Significant (FDR-BH){'' if method_name == 'patterson' else f' {stat_key}'}"

            for corr_name, sig_col_name in [
                ("Uncorrected", sig_col_raw),
                ("Bonferroni", sig_col_bonf),
                ("FDR-BH", sig_col_fdr),
            ]:
                if sig_col_name in df.columns:
                    sig = df[sig_col_name].sum()
                    ns = len(df) - sig
                    rows.append(
                        {
                            "Statistic": stat,
                            "Correction": corr_name,
                            "Significant": sig,
                            "Not Significant": ns,
                        }
                    )

        return pd.DataFrame(rows).melt(
            id_vars=["Statistic", "Correction"],
            value_vars=["Significant", "Not Significant"],
            var_name="Significance",
            value_name="Count",
        )

    def plot_d_statistics(
        self, df: pd.DataFrame, method: Literal["patterson", "partitioned", "dfoil"]
    ) -> None:
        """
        Main controller for creating and saving a suite of plots and reports for D-statistics results.
        """
        if df.empty:
            self.logger.warning(
                "Input DataFrame for plotting is empty. Skipping all D-statistic plots."
            )
            return

        # 1. Clean data ONCE at the beginning.
        df_clean = self._prepare_d_stats_plotting_df(df)

        if df_clean.empty:
            self.logger.warning(
                "No finite data remains after cleaning. Skipping all D-statistic plots."
            )
            return

        # 2. Call individual plotting and reporting methods with the CLEAN data
        self.plot_d_statistics_heatmap(df_clean, method_name=method)
        self.plot_dstat_significance_counts(df_clean, method_name=method)
        self.plot_dstat_chi_square_distribution(df_clean, method_name=method)
        self.plot_dstat_pvalue_distribution(df_clean, method_name=method)
        self.plot_stacked_significance_barplot(df_clean, method_name=method)
        self._queue_d_stats_multiqc_reports(df_clean, method)

    def plot_d_statistics_heatmap(
        self,
        df: pd.DataFrame,
        method_name: Literal["patterson", "partitioned", "dfoil"] = "patterson",
    ):
        """Plots a heatmap of D-statistics colored by -log10(P-value)."""
        df = df.copy()
        df = df[df["Method"] == method_name]
        if df.empty:
            return

        map_info = {
            "patterson": {"d_cols": ["D-statistic"], "pval_map": {"D-statistic": "P"}},
            "partitioned": {
                "d_cols": ["D1", "D2", "D12"],
                "pval_map": {d: f"P_{d}" for d in ["D1", "D2", "D12"]},
            },
            "dfoil": {
                "d_cols": ["DFO", "DFI", "DOL", "DIL"],
                "pval_map": {d: f"P_{d}" for d in ["DFO", "DFI", "DOL", "DIL"]},
            },
        }
        all_d = map_info[method_name]["d_cols"]
        pmap = map_info[method_name]["pval_map"]

        # Robust alignment
        present_d = [d for d in all_d if pmap.get(d) in df.columns]
        if not present_d:
            self.logger.warning(
                f"No P-value columns found for {method_name}; skipping heatmap."
            )
            return
        d_cols = present_d
        pval_cols = [pmap[d] for d in d_cols]

        # Get labels (index or column)
        if "Sample Combo" in df.columns:
            combo_source = df["Sample Combo"]
        elif "Quartet" in df.columns:
            combo_source = df["Quartet"]
        elif df.index.name in ["Quartet", "Sample Combo"]:
            combo_source = df.index.to_series()
        else:
            raise KeyError(
                "Could not find sample-combo labels ('Sample Combo'/'Quartet')."
            )

        def format_quartet_label(label: str) -> str:
            parts = str(label).split("-")
            if len(parts) == 4:
                return f"P1:{parts[0]}, P2:{parts[1]}, P3:{parts[2]}, Out:{parts[3]}"
            if len(parts) == 5:
                return f"P1:{parts[0]}, P2:{parts[1]}, P3:{parts[2]}, P4:{parts[3]}, Out:{parts[4]}"
            return label

        quartet_labels = combo_source.apply(format_quartet_label).values

        # Build Z matrix
        pval_matrix = df[pval_cols].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pval_matrix = -np.log10(pval_matrix)

        finite_max = (
            np.nanmax(log_pval_matrix[np.isfinite(log_pval_matrix)])
            if np.isfinite(log_pval_matrix).any()
            else -np.log10(0.001)
        )
        log_pval_matrix[np.isinf(log_pval_matrix)] = finite_max + 1
        log_pval_matrix = np.nan_to_num(log_pval_matrix, nan=0.0)

        # Annotations
        annotations = np.full((len(df), len(d_cols)), "", dtype=object)
        for j, stat in enumerate(d_cols):
            stat_key = {"D-statistic": "D"}.get(stat, stat)
            col_raw = f"Significant (Raw){'' if method_name == 'patterson' else f' {stat_key}'}"
            col_bonf = f"Significant (Bonferroni){'' if method_name == 'patterson' else f' {stat_key}'}"
            col_fdr = f"Significant (FDR-BH){'' if method_name == 'patterson' else f' {stat_key}'}"
            for i in range(len(df)):
                if col_bonf in df and df[col_bonf].iloc[i]:
                    annotations[i, j] = "**"
                elif col_fdr in df and df[col_fdr].iloc[i]:
                    annotations[i, j] = "†"
                elif col_raw in df and df[col_raw].iloc[i]:
                    annotations[i, j] = "*"

        fig = px.imshow(
            log_pval_matrix,
            x=d_cols,
            y=quartet_labels,
            color_continuous_scale="Reds",
            aspect="auto",
            labels={"color": "-log10(P-value)"},
            zmin=0,
            zmax=finite_max,
        )
        for i, j in np.ndindex(annotations.shape):
            if annotations[i, j]:
                fig.add_annotation(
                    text=annotations[i, j],
                    x=j,
                    y=i,
                    showarrow=False,
                    font=dict(color="black", size=14),
                    xanchor="center",
                    yanchor="middle",
                )

        title = f"{method_name.title()} D-statistics Heatmap"
        fig.update_layout(
            title=title,
            xaxis_title=f"{method_name.title()} D-statistic(s)",
            yaxis_title="Sample Combination",
            xaxis_side="top",
            template="plotly_white",
            height=min(max(len(df) * 30, 300), 1200),
        )
        fig.update_yaxes(showticklabels=False)

        output_path = (
            self.output_dir_analysis / f"d_statistics_heatmap_{method_name}.html"
        )
        fig.write_html(output_path, full_html=False, include_plotlyjs="cdn")

        method_name_pretty = {
            "patterson": "Patterson's 4-taxon D-statistic",
            "partitioned": "Partitioned D-statistic",
            "dfoil": "DFOIL D-statistic",
        }[method_name]

        # De-dupe queueing
        panel_id = f"d_statistics_heatmap_{method_name}"
        if not hasattr(self, "_queued_panels"):
            self._queued_panels = set()
        if panel_id not in self._queued_panels:
            self.snpio_mqc.queue_html(
                output_path,
                panel_id=panel_id,
                section="introgression",
                title=f"SNPio: D-statistics Heatmap ({method_name_pretty})",
                index_label="Quartet",
                description=(
                    f"This is a heatmap of {method_name_pretty}, with cells colored by the -log10(P-value). "
                    "Significance is marked with '\\*', '\\*\\*', and '†' for uncorrected, Bonferroni, and FDR-BH, respectively."
                ),
            )
            self._queued_panels.add(panel_id)

    def plot_dstat_significance_counts(self, df: pd.DataFrame, method_name: str):
        """Plots the number of significant results per D-statistic."""
        d_stats = {
            "patterson": ["D-statistic"],
            "partitioned": ["D1", "D2", "D12"],
            "dfoil": ["DFO", "DFI", "DOL", "DIL"],
        }[method_name]
        counts_df = self._get_significance_counts_df(df, d_stats, method_name)

        method_title = method_name.title()
        if method_name == "patterson":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=counts_df, x="Correction", y="Count", hue="Significance", ax=ax
            )
            ax.set_title(f"{method_title} D-Statistics Significance Counts")
            ax.legend(loc="best")
            plt.tight_layout()
        else:
            g = sns.catplot(
                data=counts_df,
                x="Statistic",
                y="Count",
                hue="Significance",
                col="Correction",
                kind="bar",
                height=6,
                aspect=0.8,
                sharey=False,
            )
            g.fig.suptitle(f"{method_title} D-Statistics Significance Counts", y=1.03)
            g.set_axis_labels("Statistic", "Count").tight_layout(rect=[0, 0, 1, 0.97])
            fig = g.fig  # unify handle

        output_path = (
            self.output_dir_analysis
            / f"d_statistics_significance_counts_{method_name}.html"
        )
        img_path = Path(str(output_path).replace(".html", f".{self.plot_format}"))
        fig.savefig(str(img_path), dpi=150, bbox_inches="tight")  # save static image
        if self.show:
            plt.show()
        plt.close(fig)

        # queue to MultiQC
        panel_id = f"d_statistics_significance_counts_plot_{method_name}"
        title = f"SNPio: {method_title} D-Statistics Significance Counts"
        desc = "Counts across Uncorrected, Bonferroni, and FDR-BH corrections."
        try:
            if hasattr(self, "snpio_mqc") and hasattr(self.snpio_mqc, "queue_html"):
                # create a tiny wrapper html alongside the image
                self.snpio_mqc.queue_html(
                    output_path,
                    panel_id=panel_id,
                    section="introgression",
                    title=title,
                    index_label=None,
                    description=desc,
                )
            else:
                self.logger.warning("MultiQC interface not available; skipping queue.")
        except Exception as e:
            self.logger.warning(f"Could not queue significance-counts plot: {e}")

    def plot_dstat_chi_square_distribution(
        self,
        df: pd.DataFrame,
        method_name: Literal["patterson", "partitioned", "dfoil"],
    ):
        """Plots the distribution of Chi-square values for D-statistics."""
        df = df.copy()

        if df.index.name in ["Quartet", "Sample Combo"]:
            df = df.reset_index().rename(columns={df.index.name: "Sample Combo"})

        d_cols = {
            "patterson": ["D"],
            "partitioned": ["D1", "D2", "D12"],
            "dfoil": ["DFO", "DFI", "DOL", "DIL"],
        }[method_name]

        chi_cols = [f"X2_{d}" if method_name != "patterson" else "X2" for d in d_cols]
        hover_data = ["Sample Combo"] + [
            f"P_X2_{d}" if method_name != "patterson" else "P_X2" for d in d_cols
        ]

        long_df = df.melt(
            id_vars=hover_data,
            value_vars=chi_cols,
            var_name="D-statistic",
            value_name="Chi2",
        )

        fig = px.violin(
            long_df,
            x="D-statistic",
            y="Chi2",
            box=True,
            points="all",
            title="Chi-square Value Distributions by D-statistic",
            template="plotly_white",
            hover_data=hover_data,
        )

        output_path = (
            self.output_dir_analysis
            / f"dstat_chi_square_distribution_{method_name}.html"
        )
        fig.write_html(output_path)
        if self.show:
            fig.show()

        # queue to MultiQC
        try:
            if hasattr(self, "snpio_mqc") and hasattr(self.snpio_mqc, "queue_html"):
                self.snpio_mqc.queue_html(
                    output_path,
                    panel_id=f"dstat_chi_square_distribution_{method_name}",
                    section="introgression",
                    title=f"SNPio: {method_name.title()} Chi-square Value Distributions",
                    index_label=None,
                    description="Violin/box distribution of X² across combinations.",
                )
            else:
                self.logger.warning("MultiQC interface not available; skipping queue.")
        except Exception as e:
            self.logger.warning(f"Could not queue chi-square distribution plot: {e}")

    def plot_dstat_pvalue_distribution(self, df: pd.DataFrame, method_name: str):
        """Plots the distribution of -log10(P-values) for D-statistics."""
        pval_cols = {
            "patterson": ["P"],
            "partitioned": ["P_D1", "P_D2", "P_D12"],
            "dfoil": ["P_DFO", "P_DFI", "P_DOL", "P_DIL"],
        }[method_name]
        with np.errstate(divide="ignore", invalid="ignore"):
            long_df = (
                df[pval_cols]
                .apply(lambda col: -np.log10(pd.to_numeric(col, errors="coerce")))
                .melt(var_name="P-value", value_name="-log10(P)")
                .dropna()
            )

        fig = px.histogram(
            long_df,
            x="-log10(P)",
            color="P-value",
            nbins=50,
            title="-log10(P-values) Distribution",
            template="plotly_white",
        )

        output_path = (
            self.output_dir_analysis / f"dstat_pvalue_distribution_{method_name}.html"
        )
        fig.write_html(output_path)
        if self.show:
            fig.show()

        # NEW: queue to MultiQC
        try:
            if hasattr(self, "snpio_mqc") and hasattr(self.snpio_mqc, "queue_html"):
                self.snpio_mqc.queue_html(
                    output_path,
                    panel_id=f"dstat_pvalue_distribution_{method_name}",
                    section="introgression",
                    title=f"SNPio: {method_name.title()} -log10(P) Distribution",
                    index_label=None,
                    description="Histogram of -log10(P) across D-statistics.",
                )
            else:
                self.logger.warning("MultiQC interface not available; skipping queue.")
        except Exception as e:
            self.logger.warning(f"Could not queue p-value distribution plot: {e}")

    def plot_stacked_significance_barplot(self, df: pd.DataFrame, method_name: str):
        """Creates a stacked bar plot of significance categories."""
        d_stats = {
            "patterson": ["D"],
            "partitioned": ["D1", "D2", "D12"],
            "dfoil": ["DFO", "DFI", "DOL", "DIL"],
        }[method_name]

        records = []
        for stat in d_stats:

            def sig_category(row):
                if row.get(
                    f"Significant (Bonferroni){'' if method_name == 'patterson' else f' {stat}'}",
                    False,
                ):
                    return "Bonferroni"
                elif row.get(
                    f"Significant (FDR-BH){'' if method_name == 'patterson' else f' {stat}'}",
                    False,
                ):
                    return "FDR-BH"
                elif row.get(
                    f"Significant (Raw){'' if method_name == 'patterson' else f' {stat}'}",
                    False,
                ):
                    return "Raw"
                return "Not significant"

            counts = df.apply(sig_category, axis=1).value_counts()
            for cat, count in counts.items():
                records.append({"D-statistic": stat, "Category": cat, "Count": count})

        stacked_df = pd.DataFrame(records)
        fig = px.bar(
            stacked_df,
            x="D-statistic",
            y="Count",
            color="Category",
            title="Significance Category Distribution",
            category_orders={
                "Category": ["Bonferroni", "FDR-BH", "Raw", "Not significant"]
            },
            template="plotly_white",
            barmode="stack",
        )

        output_path = (
            self.output_dir_analysis
            / f"dstat_stacked_significance_barplot_{method_name}.html"
        )
        fig.write_html(output_path)
        if self.show:
            fig.show()

        # NEW: queue to MultiQC
        try:
            if hasattr(self, "snpio_mqc") and hasattr(self.snpio_mqc, "queue_html"):
                self.snpio_mqc.queue_html(
                    output_path,
                    panel_id=f"dstat_stacked_significance_barplot_{method_name}",
                    section="introgression",
                    title="SNPio: Significance Category Distribution",
                    index_label=None,
                    description="Stacked counts: Bonferroni, FDR-BH, Raw, Not significant.",
                )
            else:
                self.logger.warning("MultiQC interface not available; skipping queue.")
        except Exception as e:
            self.logger.warning(f"Could not queue stacked significance barplot: {e}")

    def _format_significance_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to format labels for MultiQC tables."""
        df["Significance (Correction)"] = (
            df["Significance"] + " (" + df["Correction"] + ")"
        )
        df = df.drop(columns=["Significance", "Correction"])
        df["Significance (Correction)"] = df["Significance (Correction)"].str.replace(
            "(Raw)", "(Uncorrected)", regex=False
        )
        return df.set_index("Significance (Correction)")

    def _queue_d_stats_multiqc_reports(self, df: pd.DataFrame, method: str):
        """Queues all D-statistics reports for MultiQC using a clean DataFrame."""
        if method == "patterson":
            method_name_pretty = "Patterson's 4-taxon D-statistic"
            significance_df = self._get_significance_counts_df(
                df, ["D-statistic"], method
            )
            significance_df = self._format_significance_labels(significance_df)

            self.snpio_mqc.queue_table(
                df=significance_df,
                panel_id=f"d_statistics_significance_counts_table_{method}",
                section="introgression",
                title=f"SNPio: {method_name_pretty} D-Statistics Significance Counts",
                index_label="Significance (Correction)",
                description=f"This table summarizes the number of significant and non-significant {method_name_pretty} results. It provides counts for uncorrected p-values, as well as for values adjusted using Bonferroni and FDR-BH multiple-test corrections.",
                pconfig={
                    "id": f"d_statistics_significance_counts_table_{method}",
                    "title": f"SNPio: {method_name_pretty} D-Statistics Significance Counts",
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
                        "description": "How many sample-combinations were significant or not.",
                        "scale": "YlOrBr",
                    },
                },
            )
            self.snpio_mqc.queue_custom_lineplot(
                df=df["Z"],
                panel_id=f"d_statistics_distribution_{method}_d",
                section="introgression",
                title=f"SNPio: {method_name_pretty} D-Statistics Z-Score Distribution",
                description="This plot illustrates the distribution of Z-scores for all sample combinations, providing a visual assessment of the overall trend and variance in the D-statistic results.",
                index_label="Z-Score Bins",
                pconfig={
                    "id": f"d_statistics_distribution_{method}",
                    "title": f"SNPio: {method_name_pretty} Z-Score Distribution",
                    "xlab": "Z-Score",
                    "ylab": "Estimated Density",
                    "ymin": 0,
                    "xmin": -4.5,
                    "xmax": 4.5,
                },
            )

        elif method in {"partitioned", "dfoil"}:
            method_name_pretty = (
                "Partitioned D-statistic"
                if method == "partitioned"
                else "DFOIL D-statistic"
            )
            d_stats = (
                ["D1", "D2", "D12"]
                if method == "partitioned"
                else ["DFO", "DFI", "DOL", "DIL"]
            )

            significance_df = self._get_significance_counts_df(df, d_stats, method)

            self.snpio_mqc.queue_table(
                df=significance_df,
                panel_id=f"d_statistics_significance_counts_table_{method}",
                section="introgression",
                title=f"SNPio: {method_name_pretty} Significance Counts",
                index_label="Statistic / Correction",
                description=f"This table breaks down the counts of significant versus non-significant tests for each {method_name_pretty} D-statistic ({', '.join(d_stats)}), categorized by the type of multiple-test correction applied (Uncorrected, Bonferroni, FDR-BH).",
                pconfig={
                    "id": f"d_statistics_significance_counts_table_{method}",
                    "title": f"SNPio: {method_name_pretty} Significance Counts",
                    "col1_header": "Statistic, Correction",
                    "scale": "YlOrBr",
                    "namespace": "introgression",
                },
                headers={
                    "Statistic": {
                        "title": f"{method.capitalize()} Statistic",
                        "description": f"Which sub-statistic ({', '.join(d_stats)}).",
                    },
                    "Correction": {
                        "title": "P-Value Correction",
                        "description": "Uncorrected, Bonferroni-adjusted, or FDR-BH-adjusted.",
                    },
                    "Count": {
                        "title": "Number of Tests",
                        "description": "Count of sample-combinations in each category.",
                        "scale": "YlOrBr",
                    },
                },
            )

            for stat in d_stats:
                self.snpio_mqc.queue_custom_lineplot(
                    df=df[f"Z_{stat}"],
                    panel_id=f"d_statistics_z_distribution_{method}_{stat}",
                    section="introgression",
                    title=f"SNPio: {method_name_pretty} {stat} Z-Score Distribution",
                    description=f"This plot shows the distribution of Z-scores specifically for the {stat} component of the {method_name_pretty} D-statistic, allowing for a detailed look at its behavior across all sample combinations.",
                    index_label="Z-Score Bins",
                    pconfig={
                        "id": f"d_statistics_z_distribution_{method}_{stat}",
                        "title": f"SNPio: {method_name_pretty} {stat} Z-Score Distribution",
                        "xlab": "Z-Score",
                        "ylab": "Estimated Density",
                        "ymin": 0,
                        "xmin": -4.5,
                        "xmax": 4.5,
                    },
                )

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
            max_outliers_to_plot (int | None): Maximum number of outliers to plot. If None, all outliers are plotted.

        Raises:
            ValueError: If the method is not "dbscan" or "permutation".
            ValueError: If max_outliers_to_plot is not positive.
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
            method_pretty = "DBSCAN" if method == "dbscan" else "Permutation"

            self.snpio_mqc.queue_heatmap(
                df=fst_pivot,
                panel_id=f"fst_outliers_{method}_method",
                section="outliers",
                title=f"SNPio: Fst Outliers ({method_pretty} Method)",
                description=f"This heatmap displays Fst outliers detected using the {method_pretty} method. The table includes locus names for Fst outliers, their adjusted P-values (q-values), the population pairs contributing to each outlier, and the observed Weir and Cockerham (1984) Fst values. All outliers are shown. Q-values represent adjusted P-values based on the chosen multiple test correction method.",
                index_label="Contributing Population Pair",
                pconfig={
                    "id": f"fst_outliers_{method}_method",
                    "title": f"SNPio: Fst Outliers ({method_pretty} Method)",
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
            method_pretty = "DBSCAN" if method == "dbscan" else "Permutation"

            self.snpio_mqc.queue_heatmap(
                df=fst_pivot[["Locus", "q_value", "Population_Pair"]]
                .set_index(["Locus", "Population_Pair"])
                .head(max_outliers_to_plot),
                panel_id=f"fst_outliers_{method}_method",
                section="outliers",
                title=f"SNPio: Fst Outliers ({method_pretty} Method)",
                description=f"This heatmap displays Fst outliers detected using the {method_pretty} method. The table includes locus names for Fst outliers, their adjusted P-values (q-values), the population pairs contributing to each outlier, and the observed Weir and Cockerham (1984) Fst values. For space reasons, only the top {max_outliers_to_plot} outliers are displayed. Q-values represent adjusted P-values based on the chosen multiple test correction method.",
                index_label="Contributing Population Pair",
                pconfig={
                    "id": f"fst_outliers_{method}_method",
                    "title": f"SNPio: Fst Outliers ({method_pretty} Method)",
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
        self,
        summary_statistics: Dict[str, pd.DataFrame | pd.Series | dict],
        use_pvalues: bool = False,
    ) -> None:
        """Plot summary statistics per sample and per population.

        This method plots summary statistics per sample and per population on the same figure. The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst). The method also plots summary statistics per sample and per population using Seaborn PairGrid plots. The method saves the plots to the output directory and displays them if ``show`` is True.

        Args:
            summary_statistics (Dict[str, pd.DataFrame | pd.Series | dict]): Dictionary containing summary statistics for plotting.
            use_pvalues (bool): If True, display p-values for Fst values. Defaults to False.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

        self._plot_summary_statistics_per_sample(
            summary_statistics["overall"], ax=axes[0]
        )

        if (
            self.genotype_data.has_popmap
            and summary_statistics.get("per_population", None) is not None
        ):
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
            description="This box plot visualizes key genetic differentiation summary statistics: Nucleotide Diversity (Pi), Expected Heterozygosity (He), and Observed Heterozygosity (Ho). Each point overlaid on the plot represents an outlier locus falling outside the Interquartile Range (IQR).",
            pconfig={
                "id": "summary_statistics_overall",
                "title": "SNPio: Summary Statistics",
                "series_label": "summary statistics",
            },
        )

        if (
            hasattr(self.genotype_data, "marker_names")
            and self.genotype_data.marker_names is not None
        ):
            marker_names = self.genotype_data.marker_names
        else:
            marker_names = [f"locus_{i}" for i in range(self.genotype_data.num_snps)]

        if self.genotype_data.has_popmap:
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
            ].pivot(index="Locus (CHROM:POS)", columns="Population ID", values="Ho")

            df_per_pop_pivot_he = df_per_pop[
                ["Locus (CHROM:POS)", "Population ID", "He"]
            ].pivot(index="Locus (CHROM:POS)", columns="Population ID", values="He")

            df_per_pop_pivot_pi = df_per_pop[
                ["Locus (CHROM:POS)", "Population ID", "Pi"]
            ].pivot(index="Locus (CHROM:POS)", columns="Population ID", values="Pi")

            self.snpio_mqc.queue_custom_boxplot(
                df=df_per_pop_pivot_pi,
                panel_id="summary_statistics_per_population_pi",
                section="detailed_statistics",
                title="SNPio: Per-locus Nucleotide Diversity (Pi) for each Population",
                index_label="Locus (CHROM:POS)",
                description="This box plot displays the per-locus Nucleotide Diversity (Pi) for each population. Points indicate outlier loci that fall outside the box whiskers, highlighting loci with unusual diversity levels within a population.",
            )

            self.snpio_mqc.queue_custom_boxplot(
                df=df_per_pop_pivot_he,
                panel_id="summary_statistics_per_population_he",
                section="detailed_statistics",
                title="SNPio: Per-locus Expected Heterozygosity (He) for each Population",
                index_label="Locus (CHROM:POS)",
                description="This box plot shows the per-locus Expected Heterozygosity (He) for each population. Points indicate outlier loci that fall outside the box whiskers, highlighting loci with unusual expected heterozygosity levels within a population.",
            )

            self.snpio_mqc.queue_custom_boxplot(
                df=df_per_pop_pivot_ho,
                panel_id="summary_statistics_per_population_ho",
                section="detailed_statistics",
                title="SNPio: Per-locus Observed Heterozygosity (Ho) for each Population",
                index_label="Locus (CHROM:POS)",
                description="This box plot presents the per-locus Observed Heterozygosity (Ho) for each population. Points indicate outlier loci that fall outside the box whiskers, highlighting loci with unusual observed heterozygosity levels within a population.",
            )

            # Plot Fst heatmap
            self._plot_fst_heatmap(
                summary_statistics["Fst_between_populations_obs"],
                df_fst_lower=summary_statistics["Fst_between_populations_lower"],
                df_fst_upper=summary_statistics["Fst_between_populations_upper"],
                df_fst_pvals=summary_statistics["Fst_between_populations_pvalues"],
                use_pvalues=use_pvalues,
            )
        else:
            self.logger.info(
                "No population map provided; skipping per-population summary statistics plots."
            )

    def _flatten_fst_data(self, fst_dict, stat_name):
        """Flatten the Fst data into a tidy DataFrame."""
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
        """Flatten the overall statistics into a tidy DataFrame."""
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

        Args:
            per_pop_dict (Dict[str, Any]): A dictionary where keys are population identifiers and values are either dicts, DataFrames, or Series containing statistics.

        Returns:
            pd.DataFrame: A tidy DataFrame with the following columns: population: Population identifier, locus: Locus index, He: Expected heterozygosity, Ho: Observed heterozygosity
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
            search_mode (bool): Whether the Sankey diagram is being plotted in search mode. Defaults to False.
            fn (str | None): The filename to save the plot. If None, the default filename is used. Defaults to None.

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
        df["Missing_Threshold"] = (
            df["Missing_Threshold"].fillna(1.0).astype(float).round(3)
        )
        df["MAF_Threshold"] = df["MAF_Threshold"].fillna(0.0).astype(float).round(3)
        df["Kept_Prop"] = df["Kept_Prop"].astype(float).round(2)
        df["Removed_Prop"] = df["Removed_Prop"].astype(float).round(2)
        df["MAC_Threshold"] = df["MAC_Threshold"].fillna(0).astype(int)
        df["Bool_Threshold"] = df["Bool_Threshold"].fillna(False).astype(int)
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
            description="This Sankey diagram visualizes the flow of loci through various filtering steps. Loci that are retained are shown in green, while those removed are shown in red. The diagram interactively displays the number of loci kept and removed at each stage of the filtering process.",
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
            description="This bar plot shows the total counts of different genotypes (e.g., A/A, A/T, T/T) across all samples and all loci in the dataset, providing a summary of the allelic variation.",
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
        """
        if df_combined.empty:
            msg = "No data to plot. Please check the filtering thresholds."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Plotting search results.")
        self.logger.debug(f"Combined data: {df_combined}")

        df_combined["Missing_Threshold"] = df_combined["Missing_Threshold"].round(3)
        df_combined["MAF_Threshold"] = df_combined["MAF_Threshold"].round(3)
        df_combined["Bool_Threshold"] = df_combined["Bool_Threshold"].round(3)
        df_combined["Removed_Prop"] = df_combined["Removed_Prop"].round(3)
        df_combined["Kept_Prop"] = df_combined["Kept_Prop"].round(3)

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
            description="This table presents a comprehensive summary of the filtering results from the NRemover2 search, detailing the proportion of loci that were removed and kept for each filtering method and its corresponding threshold.",
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

            outpath = self._plot_combined_missing_plotly(df)

            self.snpio_mqc.queue_html(
                html=outpath,
                panel_id="filtering_results_missing_loci_samples_plotly",
                section="filtering",
                title="SNPio: Missing Data Filtering Results (Per-locus and Sample)",
                index_label="Filter Method",
                description="This interactive plot displays the results of missing data filtering, showing the proportion of loci removed and kept for various missing data thresholds, applied at both a per-locus and per-sample level.",
            )

        else:
            self.logger.info("Missing data filtering results are empty.")

    def _plot_combined_missing_plotly(self, df: pd.DataFrame) -> Path:
        df = df[
            df["Filter_Method"].isin(["filter_missing", "filter_missing_sample"])
        ].copy()

        if df.empty:
            self.logger.info("Missing data filtering results are empty.")
            raise ValueError("No sample or global missing data to plot.")

        self.logger.info(
            "Plotting global and sample-level missing data filtering results."
        )
        self.logger.debug(f"Missing data: {df}")

        # Melt and build legend labels
        plot_df = self._prepare_plot_dataframe(
            df,
            id_vars=[
                "Missing_Threshold",
                "Filter_Method",
                "Kept_Count",
                "Removed_Count",
            ],
        )

        fig = px.line(
            plot_df,
            x="Missing_Threshold",
            y="Proportion",
            color="Legend_Label",
            line_group="Legend_Label",
            markers=True,
            hover_data={
                "Missing_Threshold": True,
                "Proportion": ":.3f",
                "Filter_Method": True,
                "Filter_Type": True,
            },
            title="Global and Sample-Level Missing Data Filtering",
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Missing Data Threshold",
            yaxis_title="Proportion",
            height=500,
            width=1000,
            margin=dict(t=60, b=50),
        )
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.05, 1.12])

        outpath: Path = (
            self.output_dir_gd / "filtering_results_missing_loci_samples.html"
        )
        fig.write_html(str(outpath), include_plotlyjs="cdn")
        return outpath

    def _prepare_plot_dataframe(
        self, df: pd.DataFrame, id_vars: List[str]
    ) -> pd.DataFrame:
        """Prepare the DataFrame for plotting combined missing data results.

        This method prepares the DataFrame for plotting combined missing data results. It melts the DataFrame to create a long format suitable for plotting with Plotly. The resulting DataFrame contains the missing data thresholds, filter methods, removed and kept counts, and proportions for each filter type.

        Args:
            df (pd.DataFrame): The input DataFrame containing the missing data results.
            id_vars (List[str]): The columns to melt the DataFrame on.
        Returns:
            pd.DataFrame: The prepared DataFrame for plotting.
        """
        plot_df = df.melt(
            id_vars=id_vars,
            value_vars=["Removed_Prop", "Kept_Prop"],
            var_name="Filter_Type",
            value_name="Proportion",
        )

        group_vars = [x for x in id_vars if not x.endswith("Count")]
        group_vars += ["Filter_Type"]
        plot_df = plot_df.groupby(group_vars, as_index=False).agg(
            {"Proportion": "mean"}
        )
        plot_df = plot_df.drop_duplicates(subset=group_vars)

        if "Filter_Method" in plot_df.columns:
            plot_df["Legend_Label"] = (
                plot_df["Filter_Method"] + " - " + plot_df["Filter_Type"]
            )
        else:
            plot_df["Legend_Label"] = (
                "Population-wise Missing - " + plot_df["Filter_Type"]
            )

        # Sort for clean lines
        return plot_df.sort_values(by=[id_vars[0], "Legend_Label"])

    def _plot_pops(self, df: pd.DataFrame) -> None:
        """Plot population-level missing data proportions.

        This method plots the proportion of loci removed and kept for each population-level missing data threshold. The plot is saved to a file. The plot shows the proportion of loci removed and kept for each missing data threshold at the population level. The plot is saved to a file in the output directory with the filename: ``<prefix>_output/gtdata/plots/filtering_results_missing_population.{plot_format}``.

        Args:
            df (pd.DataFrame): The input dataframe containing the population-level missing data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.
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

            outfile = self._plot_missing_pop_plotly(df)

            self.snpio_mqc.queue_html(
                html=outfile,
                panel_id="filtering_results_missing_population_plotly",
                section="filtering",
                title="SNPio: Population-Level Missing Data Filtering",
                index_label="Missing Threshold",
                description="This interactive plot illustrates how the proportion of loci changes with different missing data thresholds applied at the population level. Loci are retained only if they meet the specified missing data threshold across all populations; otherwise, they are removed.",
            )

        else:
            self.logger.info("Population-level missing data is empty.")

    def _plot_missing_pop_plotly(self, df: pd.DataFrame) -> Path:
        """Plot per-population missing data thresholds using Plotly Express.

        This method generates an interactive line plot for the population-level missing data thresholds. The plot shows the proportion of loci removed and kept for each missing data threshold. The plot is saved to an HTML file.

        Args:
            df (pd.DataFrame): The input dataframe containing the population-level missing data.

        Returns:
            Path: The path to the generated HTML file.

        Raises:
            ValueError: Raised if the input dataframe is empty.
        """
        df = df[df["Filter_Method"] == "filter_missing_pop"].copy()

        if df.empty:
            self.logger.info("Population-level missing data is empty.")
            raise ValueError("No population-level missing data to plot.")

        self.logger.info("Plotting population-level missing data.")

        plot_df = self._prepare_plot_dataframe(
            df, id_vars=["Missing_Threshold", "Kept_Count", "Removed_Count"]
        )

        fig = px.line(
            plot_df,
            x="Missing_Threshold",
            y="Proportion",
            color="Legend_Label",
            line_group="Legend_Label",
            markers=True,
            hover_data={"Missing_Threshold": True, "Proportion": ":.3f"},
            title="Population-Level Missing Data Filtering",
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Missing Data Threshold",
            yaxis_title="Proportion",
            height=500,
            width=900,
        )
        fig.update_yaxes(range=[-0.05, 1.12])

        outpath = self.output_dir_gd / "filtering_results_missing_population.html"
        fig.write_html(str(outpath), include_plotlyjs="cdn")
        return outpath

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

        if not df.empty or not df_mac.empty:
            # Plot MAF and MAC data using Plotly
            maf_outpath, mac_outpath = self._plot_maf_plotly(df)

            if maf_outpath is not None:
                self.snpio_mqc.queue_html(
                    html=maf_outpath,
                    panel_id="maf_thresholds",
                    section="filtering",
                    title="SNPio: MAF Threshold Search Results",
                    index_label="Index",
                    description="This interactive plot displays how the proportion of loci changes with different Minor Allele Frequency (MAF) thresholds. It helps in selecting an appropriate MAF filter by showing the trade-off between loci retained and removed.",
                )

            if mac_outpath is not None:
                self.snpio_mqc.queue_html(
                    html=mac_outpath,
                    panel_id="mac_thresholds",
                    section="filtering",
                    title="SNPio: MAC Threshold Search Results",
                    index_label="Index",
                    description="This interactive plot shows the effect of different Minor Allele Count (MAC) thresholds on the dataset. It visualizes the proportion of loci that are kept or discarded, aiding in the selection of an optimal MAC filter.",
                )

    def _plot_maf_plotly(self, df: pd.DataFrame) -> Tuple[Path | None, Path | None]:
        """Plot MAF and MAC filtering data using Plotly Express and save as HTML.

        This method plots the MAF and MAC filtering data using Plotly Express. It generates interactive line plots for both MAF and MAC thresholds, showing the proportion of loci removed and kept. The plots are saved as HTML files.

        Args:
            df (pd.DataFrame): The input dataframe containing the MAF and MAC data.

        Returns:
            Tuple[Path | None, Path | None]: Paths to the saved HTML files for MAF and MAC plots.
        """
        df_mac = df[df["Filter_Method"] == "filter_mac"].copy()
        df_maf = df[df["Filter_Method"] == "filter_maf"].copy()

        if df_mac.empty and df_maf.empty:
            self.logger.warning(
                "Both MAF and MAC data are empty. Try checking the 'filter_mac' and/or 'filter_maf' filtering thresholds."
            )
            return None, None

        if df_mac.empty or df_maf.empty:
            self.logger.warning(
                "One of MAF or MAC data is empty. Only plotting the available data."
            )

        maf_outpath = None
        mac_outpath = None

        # --- MAF Plot ---
        if not df_maf.empty:
            self.logger.info("Plotting minor allele frequency (MAF) data.")

            plot_df = self._prepare_plot_dataframe(df_maf, id_vars=["MAF_Threshold"])

            fig_maf = px.line(
                plot_df,
                x="MAF_Threshold",
                y="Proportion",
                color="Legend_Label",
                line_group="Legend_Label",
                markers=True,
                hover_data={"MAF_Threshold": True, "Proportion": ":.3f"},
                title="MAF Filtering Summary",
            )

            fig_maf.update_layout(
                template="plotly_white",
                xaxis_title="MAF Threshold",
                yaxis_title="Proportion",
                height=500,
                width=900,
            )
            fig_maf.update_yaxes(range=[-0.05, 1.12])

            maf_outpath = self.output_dir_gd / "filtering_results_maf.html"
            fig_maf.write_html(str(maf_outpath), include_plotlyjs="cdn")

        # --- MAC Plot (unchanged) ---
        if not df_mac.empty:
            self.logger.info("Plotting minor allele count (MAC) data.")

            plot_df = self._prepare_plot_dataframe(df_mac, id_vars=["MAC_Threshold"])

            fig_mac = px.line(
                plot_df,
                x="MAC_Threshold",
                y="Proportion",
                color="Legend_Label",
                line_group="Legend_Label",
                markers=True,
                hover_data={"MAC_Threshold": True, "Proportion": ":.3f"},
                title="MAC Filtering Summary",
            )

            fig_mac.update_layout(
                template="plotly_white",
                xaxis_title="MAC Threshold",
                yaxis_title="Proportion",
                height=500,
                width=900,
            )
            fig_mac.update_yaxes(range=[-0.05, 1.12])

            mac_outpath = self.output_dir_gd / "filtering_results_mac.html"
            fig_mac.write_html(str(mac_outpath), include_plotlyjs="cdn")

        return maf_outpath, mac_outpath

    def _plot_boolean(self, df: pd.DataFrame) -> None:
        """Plot boolean datasets, including: Monomorphic, Biallelic, Thin Loci, Singleton, and Linked.

        This method plots the boolean filtering data. The boolean filtering data includes the proportion of loci removed and kept for each boolean filtering threshold. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The input dataframe containing the boolean data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.
        """
        df = df[df["Filter_Method"].isin(self.boolean_filter_methods)].copy()

        if not df.empty:
            self.logger.info("Plotting boolean filtering data.")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Kept_Prop", "Removed_Prop"]):
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

            # Melt the DataFrame for combined Removed_Prop and Kept_Prop plots
            outpath = self._bool_filter_summary_plotly(df)

            self.snpio_mqc.queue_html(
                html=outpath,
                panel_id="boolean_thresholds",
                section="filtering",
                title="SNPio: Boolean Filtering Results",
                index_label="Index",
                description="These line graphs show the impact of various boolean filters (e.g., Biallelic, Singletons, Monomorphic) on the dataset. The plots visualize the proportion of loci that are removed and kept, helping to understand the effect of each filtering criterion.",
            )

        else:
            self.logger.warning(
                "Filtered boolean dataset was empty. This likely means that either 'filter_biallelic', 'filter_monomorphic', or 'filter_singletons' removed all loci."
            )

    def _bool_filter_summary_plotly(self, df: pd.DataFrame) -> Path:
        """Plot boolean filtering results using Plotly Express.

        This method creates an interactive line plot summarizing the boolean filtering results. It visualizes the proportion of loci removed and kept for each boolean filtering threshold, grouped by filter method and type.

        Args:
            df (pd.DataFrame): The input DataFrame containing boolean filtering data.

        Returns:
            Path: The path to the saved HTML file.
        """
        plot_df = self._prepare_plot_dataframe(
            df,
            id_vars=["Bool_Threshold", "Filter_Method", "Kept_Count", "Removed_Count"],
        )

        # Create line plot with hover data
        fig = px.line(
            plot_df,
            x="Bool_Threshold",
            y="Proportion",
            color="Filter_Method",
            line_dash="Filter_Type",
            markers=True,
            facet_col="Filter_Type",
            facet_col_spacing=0.1,
            hover_data={
                "Bool_Threshold": True,
                "Proportion": ":.3f",
                "Filter_Method": True,
                "Filter_Type": False,
            },
            title="Boolean Filtering Method Summary",
        )

        fig.update_layout(
            template="plotly_white",
            height=500,
            width=950,
            legend_title="Filter Method",
            margin=dict(t=60, b=50),
        )

        fig.update_xaxes(
            title_text="Heterozygous Genotypes",
            tickvals=[0.0, 1.0],
            ticktext=["Included", "Excluded"],
            range=[-0.05, 1.05],
        )

        fig.update_yaxes(title_text="Proportion", range=[-0.05, 1.12])

        # Save to file
        outpath: Path = self.output_dir_gd / "filtering_results_bool.html"
        fig.write_html(str(outpath), include_plotlyjs="cdn")
        return outpath

    def plot_pop_counts(self, populations: pd.Series) -> None:
        """Plot the population counts.

        This function takes a series of population data and plots the counts and proportions of each population ID. The resulting plot is saved to a file of the specified format. The plot shows the counts and proportions of each population ID. The plot is colored based on the median count and proportion.

        Args:
            populations (pd.Series): The series containing population data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input data is not a pandas Series.
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

        self.snpio_mqc.queue_barplot(
            df=[df, df_perc],
            panel_id="population_counts",
            section="overview",
            title="SNPio: Population Counts",
            description="This bar plot displays the number of samples belonging to each population, providing a clear overview of the sample distribution across the defined populations.",
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

        Args:
            df (pd.DataFrame): The input DataFrame containing genotype data.
            prefix (str, optional): Prefix for the output file names. Defaults to None.
            zoom (bool, optional): If True, zooms in on the missing proportions (0-1). Defaults to False.
            bar_color (str, optional): Color for the bar plots. Defaults to "gray".
            heatmap_palette (str, optional): Color palette for the heatmaps. Defaults to "magma".

        Returns:
            MissingStats: The dataclass bundle containing all missing value statistics. It contains the following attributes: Per_individual: Series with missing proportions per individual, Per_locus: Series with missing proportions per locus, Per_population: Series with missing proportions per population, Per_individual_population: DataFrame with missing proportions per individual and population, Per_population_locus: DataFrame with missing proportions per population and locus.
        """
        self.logger.info("Generating missingness report...")

        prefix = self.prefix if prefix is None else prefix

        if not isinstance(df, pd.DataFrame):
            df = misc.validate_input_type(df, return_type="df")

        has_popmap = self.genotype_data.has_popmap
        stats = self.genotype_data.calc_missing(df, use_pops=has_popmap)

        ncol = 3
        nrow = 2 if has_popmap else 1

        mpl.rcParams.update({"savefig.bbox": "standard"})

        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(12, 10 if nrow == 1 else 14),
            constrained_layout=True,
        )
        fig.suptitle("Missingness Report")

        # Plot 1: Per-Individual barplot
        ax = axes[0, 0] if has_popmap else axes[0]
        ax.barh(stats.per_individual.index, stats.per_individual, color=bar_color)
        ax.set_xlim([0, 1] if not zoom else None)
        ax.set_xlabel("Missing Prop.")
        ax.set_ylabel("Sample")
        ax.tick_params(axis="y", labelleft=False)

        # Plot 2: Per-Locus barplot
        ax = axes[0, 1] if has_popmap else axes[1]
        ax.barh(stats.per_locus.index, stats.per_locus, color=bar_color)
        ax.set_xlim([0, 1] if not zoom else None)
        ax.set_xlabel("Missing Prop.")
        ax.set_ylabel("Locus")
        ax.tick_params(axis="y", labelleft=False)

        # Plot 3: Per-Population totals (if available)
        if has_popmap and stats.per_population is not None:
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
        ax = axes[1, 1] if has_popmap else axes[2]

        if has_popmap:
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
            if stats.per_population_locus is not None:
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
            pvals (pd.DataFrame): The p-values for the distance matrix. Defaults to None.
            palette (str): The color palette to use for the heatmap. Defaults to "coolwarm".
            title (str): The title of the plot. Defaults to "Distance Matrix".
            dist_type (str): The type of distance to plot. Defaults to "fst".
        """
        self._plot_fst_heatmap(
            df,
            df_fst_pvals=pvals,
            use_pvalues=True,
            palette=palette,
            title=title,
            dist_type=dist_type,
        )

    def plot_allele_summary(
        self, summary: pd.Series, figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Plot allele summary statistics from summarize_alleles output.

        This method visualizes: Missingness (overall, median, percent with any, quartiles); Heterozygosity (overall, mean per-sample/locus); Allelic spectrum & HWE (mono→quad, mean/effective # alleles, expected het & F_IS); MAF summary & spectrum (singleton, mean/median, rare, 5 frequency bins)

        Args:
            summary (pd.Series): Output from summarize_alleles.
            figsize (Tuple[int, int]): Base size (w,h); height is doubled for a 2 x 2 grid.

        Raises:
            ValueError: If any of the expected keys are missing.
        """
        # ensure all needed keys exist
        needed = [
            # Missingness
            "Overall Missing Prop.",
            "Median Sample Missing",
            "Median Locus Missing",
            "Pct Samples with Missing",
            "Pct Loci with Missing",
            "Sample Miss Q1",
            "Sample Miss Q3",
            "Locus Miss Q1",
            "Locus Miss Q3",
            # Heterozygosity
            "Overall Heterozygosity Prop.",
            "Mean Sample Heterozygosity Prop.",
            "Mean Locus Heterozygosity Prop.",
            # Allelic spectrum & HWE
            "Prop. Monomorphic",
            "Prop. Biallelic",
            "Prop. Triallelic",
            "Prop. Quadallelic",
            "Mean Alleles per Locus",
            "Mean Effective Alleles",
            "Mean Expected Heterozygosity",
            "Mean F_IS",
            # MAF summary & bins
            "Prop. Singleton Loci",
            "MAF Mean",
            "MAF Median",
            "Prop. Rare Variants",
            "MAF < 0.01",
            "0.01 ≤ MAF < 0.05",
            "0.05 ≤ MAF < 0.10",
            "0.10 ≤ MAF < 0.20",
            "MAF ≥ 0.20",
        ]
        missing = set(needed) - set(summary.index)
        if missing:
            raise ValueError(f"plot_allele_summary: missing keys {missing}")

        sns.set_theme(style="whitegrid", font_scale=1.2)
        # keep width, double the height for a 2×2 grid
        fig, axs = plt.subplots(
            2, 2, figsize=(figsize[0], figsize[1] * 2), constrained_layout=True
        )

        # 1) Missingness
        miss_keys = [
            "Overall Missing Prop.",
            "Median Sample Missing",
            "Median Locus Missing",
            "Pct Samples with Missing",
            "Pct Loci with Missing",
        ]
        sns.barplot(
            y=miss_keys,
            x=summary[miss_keys].values,
            ax=axs[0, 0],
            palette="Blues_d",
            hue=miss_keys,
            legend=False,
        )
        axs[0, 0].set(title="Missingness", xlim=(0, 1))

        # 2) Heterozygosity
        het_keys = [
            "Overall Heterozygosity Prop.",
            "Mean Sample Heterozygosity Prop.",
            "Mean Locus Heterozygosity Prop.",
        ]
        sns.barplot(
            y=het_keys,
            x=summary[het_keys].values,
            ax=axs[0, 1],
            hue=het_keys,
            palette="Greens_d",
            legend=False,
        )
        axs[0, 1].set(title="Heterozygosity", xlim=(0, 1))

        # 3) Allelic spectrum & HWE
        spec_keys = [
            "Prop. Monomorphic",
            "Prop. Biallelic",
            "Prop. Triallelic",
            "Prop. Quadallelic",
            "Mean Alleles per Locus",
            "Mean Effective Alleles",
            "Mean Expected Heterozygosity",
            "Mean F_IS",
        ]
        sns.barplot(
            y=spec_keys,
            x=summary[spec_keys].values,
            ax=axs[1, 0],
            palette="Purples_d",
            hue=spec_keys,
            legend=False,
        )
        axs[1, 0].set(title="Allelic Spectrum & HWE", xlim=(0, 1))

        # 4) MAF summary & spectrum
        maf_keys = [
            "Prop. Singleton Loci",
            "MAF Mean",
            "MAF Median",
            "Prop. Rare Variants",
            "MAF < 0.01",
            "0.01 ≤ MAF < 0.05",
            "0.05 ≤ MAF < 0.10",
            "0.10 ≤ MAF < 0.20",
            "MAF ≥ 0.20",
        ]
        sns.barplot(
            y=maf_keys,
            x=summary[maf_keys].values,
            ax=axs[1, 1],
            palette="Oranges_d",
            hue=maf_keys,
            legend=False,
        )
        axs[1, 1].set(title="MAF Summary & Spectrum", xlim=(0, 1))

        # save & (optionally) show
        savepath: Path = self.output_dir_gd / f"allele_summary.{self.plot_format}"
        fig.savefig(savepath)
        if self.show:
            plt.show()
        plt.close(fig)

        # 1. Mapping from original keys → prettier, more intuitive labels
        pretty_names = {
            # Missingness
            "Overall Missing Prop.": "Overall Missingness",
            "Median Sample Missing": "Median Missingness per Sample",
            "Median Locus Missing": "Median Missingness per Locus",
            "Sample Miss Q1": "Sample Missingness (1st Quartile)",
            "Sample Miss Q3": "Sample Missingness (3rd Quartile)",
            "Locus Miss Q1": "Locus Missingness (1st Quartile)",
            "Locus Miss Q3": "Locus Missingness (3rd Quartile)",
            # Heterozygosity
            "Overall Heterozygosity Prop.": "Overall Heterozygosity",
            "Mean Sample Heterozygosity Prop.": "Mean Sample Heterozygosity",
            "Mean Locus Heterozygosity Prop.": "Mean Locus Heterozygosity",
            "Sample Heterozygosity Q1": "Sample Heterozygosity (1st Quartile)",
            "Sample Heterozygosity Q3": "Sample Heterozygosity (3rd Quartile)",
            "Locus Heterozygosity Q1": "Locus Heterozygosity (1st Quartile)",
            "Locus Heterozygosity Q3": "Locus Heterozygosity (3rd Quartile)",
            # Allelic spectrum & HWE
            "Prop. Monomorphic": "Monomorphic Loci (%)",
            "Prop. Biallelic": "Biallelic Loci (%)",
            "Prop. Triallelic": "Triallelic Loci (%)",
            "Prop. Quadallelic": "Quadallelic Loci (%)",
            "Mean Expected Heterozygosity": "Mean Expected Heterozygosity",
            "Mean F_IS": "Mean Inbreeding Coefficient (F_IS)",
            # MAF & singletons
            "Prop. Singleton Loci": "Singleton Loci (%)",
            "MAF Mean": "Mean Minor Allele Frequency",
            "MAF Median": "Median Minor Allele Frequency",
            "Prop. Rare Variants": "Rare Variant Loci (%)",
            "MAF < 0.01": "MAF < 1%",
            "0.01 ≤ MAF < 0.05": "1% ≤ MAF < 5%",
            "0.05 ≤ MAF < 0.10": "5% ≤ MAF < 10%",
            "0.10 ≤ MAF < 0.20": "10% ≤ MAF < 20%",
            "MAF ≥ 0.20": "MAF ≥ 20%",
        }

        # 2. Rename the summary Series
        pretty_summary = summary.rename(pretty_names)

        # 3. Original key groups
        missing_orig = [
            "Overall Missing Prop.",
            "Median Sample Missing",
            "Median Locus Missing",
            "Sample Miss Q1",
            "Sample Miss Q3",
            "Locus Miss Q1",
            "Locus Miss Q3",
        ]
        het_orig = [
            "Overall Heterozygosity Prop.",
            "Mean Sample Heterozygosity Prop.",
            "Mean Locus Heterozygosity Prop.",
            "Sample Heterozygosity Q1",
            "Sample Heterozygosity Q3",
            "Locus Heterozygosity Q1",
            "Locus Heterozygosity Q3",
        ]
        spec_orig = [
            "Prop. Monomorphic",
            "Prop. Biallelic",
            "Prop. Triallelic",
            "Prop. Quadallelic",
            "Mean Expected Heterozygosity",
            "Mean F_IS",
        ]
        maf_orig = [
            "Prop. Singleton Loci",
            "MAF Mean",
            "MAF Median",
            "Prop. Rare Variants",
            "MAF < 0.01",
            "0.01 ≤ MAF < 0.05",
            "0.05 ≤ MAF < 0.10",
            "0.10 ≤ MAF < 0.20",
            "MAF ≥ 0.20",
        ]

        # 4. Build lists of the new, pretty keys
        missing_keys = [pretty_names[k] for k in missing_orig]
        het_keys = [pretty_names[k] for k in het_orig]
        spec_keys = [pretty_names[k] for k in spec_orig]
        maf_keys = [pretty_names[k] for k in maf_orig]

        # 5. Slice into four Series (one per tab)
        data_missing = pretty_summary[missing_keys]
        data_het = pretty_summary[het_keys]
        data_spec = pretty_summary[spec_keys]
        data_maf = pretty_summary[maf_keys]

        # 6. Define the tab buttons' labels and y-axis labels
        data_labels = [
            {"name": "Missingness", "ylab": "Proportion", "ymin": 0, "ymax": 1},
            {"name": "Heterozygosity", "ylab": "Proportion", "ymin": 0, "ymax": 1},
            {
                "name": "Allelic Spectrum & HWE",
                "ylab": "Proportion",
                "ymin": 0,
                "ymax": 1,
            },
            {
                "name": "MAF Summary & Spectrum",
                "ylab": "Proportion",
                "ymin": 0,
                "ymax": 1,
            },
        ]

        # 7. Queue the MultiQC barplot with four switchable datasets
        self.snpio_mqc.queue_barplot(
            df=[data_missing, data_het, data_spec, data_maf],
            panel_id="allele_summary",
            section="allele_summary",
            title="SNPio: Allele Summary Statistics",
            description="This interactive bar plot provides a comprehensive overview of allele summary statistics, organized into four switchable panels: Missingness, Heterozygosity, Allelic Spectrum & HWE, and MAF & Singletons. Each panel visualizes different aspects of the genetic data, allowing for a thorough assessment of data quality and characteristics.",
            index_label="Statistic",
            pconfig={
                "id": "allele_summary",
                "title": "SNPio: Allele Summary Statistics",
                "series_label": "Statistic",
                "data_labels": data_labels,
            },
        )
