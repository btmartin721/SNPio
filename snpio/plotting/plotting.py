import string
import warnings
from ctypes import Union
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import holoviews as hv
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from holoviews import opts
from mpl_toolkits.mplot3d import Axes3D  # Don't remove this import.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

hv.extension("bokeh")

from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils import misc
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


class Plotting:
    """Class containing various methods for generating plots based on genotype data.

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
        _plot_summary_statistics_per_population_grid: Plot summary statistics per population using a Seaborn PairGrid plot. Not yet implemented.
        _plot_summary_statistics_per_sample_grid: Plot summary statistics per sample using a Seaborn PairGrid plot. Not yet implemented.
    """

    def __init__(
        self,
        genotype_data: Any,
        show: Optional[bool] = None,
        plot_format: Optional[str] = None,
        dpi: Optional[int] = None,
        plot_fontsize: Optional[int] = None,
        plot_title_fontsize: Optional[int] = None,
        despine: Optional[bool] = None,
        verbose: Optional[bool] = None,
        debug: Optional[bool] = None,
    ) -> None:
        """Initialize the Plotting class.

        This class contains various methods for generating plots based on genotype data. The class is initialized with a GenotypeData object containing necessary data. The class attributes are set based on the provided values, the GenotypeData object, or default values.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object containing necessary data.
            show (bool, optional): Whether to display the plots. Defaults to `genotype_data.show` if available, otherwise `False`.
            plot_format (str, optional): The format in which to save the plots (e.g., 'png', 'svg'). Defaults to `genotype_data.plot_format` if available, otherwise `'png'`.
            dpi (int, optional): The resolution of the saved plots. Unused for vector `plot_format` types. Defaults to `genotype_data.dpi` if available, otherwise `300`.
            plot_fontsize (int, optional): The font size for the plot labels. Defaults to `genotype_data.plot_fontsize` if available, otherwise `18`.
            plot_title_fontsize (int, optional): The font size for the plot titles. Defaults to `genotype_data.plot_title_fontsize` if available, otherwise `22`.
            despine (bool, optional): Whether to remove the top and right plot axis spines. Defaults to `genotype_data.despine` if available, otherwise `True`.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to `genotype_data.verbose` if available, otherwise `False`.
            debug (bool, optional): Whether to enable debug logging. Defaults to `genotype_data.debug` if available, otherwise `False`.

        Raises:
            ValueError: Raised if the input genotype_data object is not provided.

        Note:
            The `genotype_data` attribute must be provided during initialization.

            The `show`, `plot_format`, `dpi`, `plot_fontsize`, `plot_title_fontsize`, `despine`, `verbose`, and `debug` attributes are set based on the provided values, the `genotype_data` object, or default values.

            The `output_dir` attribute is set to the `prefix_output/nremover/plots` directory.

            The `logger` attribute is set based on the `debug` attribute.

            The `boolean_filter_methods`, `missing_filter_methods`, and `maf_filter_methods` attributes are set to lists of filter methods.

            The `mpl_params` dictionary contains default Matplotlib parameters for the plots.

            The Matplotlib parameters are updated with the `mpl_params` dictionary.

            The `plotting` object is used to set the attributes based on the provided values, the `genotype_data` object, or default values.
        """
        self.genotype_data = genotype_data
        self.prefix: str = getattr(genotype_data, "prefix", "plot")

        self.output_dir: Path = Path(f"{self.prefix}_output")
        self.output_dir_gd: Path = self.output_dir / "gtdata" / "plots"
        self.output_dir_analysis: Path = self.output_dir / "analysis" / "plots"
        self.output_dir_nrm: Path = self.output_dir / "nremover" / "plots"

        self.output_dir_gd.mkdir(parents=True, exist_ok=True)
        self.output_dir_analysis.mkdir(parents=True, exist_ok=True)
        self.output_dir_nrm.mkdir(parents=True, exist_ok=True)

        self.verbose: bool | None = verbose
        self.debug: bool | None = debug

        prefix: str = genotype_data.prefix
        kwargs: Dict[str, str | bool] = {
            "prefix": prefix,
            "verbose": verbose,
            "debug": debug,
        }
        logman = LoggerManager(__name__, **kwargs)
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
            "verbose": False,
            "debug": False,
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

    def _get_attribute_value(self, attr: str) -> Any:
        """Determine the value for an attribute based on the provided argument,
        genotype_data attribute, or default value.

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
        ax: Optional[plt.Axes] = None,
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
        self, per_population_stats: dict, ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot mean summary statistics per population as grouped bar chart.

        This method plots the mean summary statistics per population as a grouped bar chart with different colors for each statistic (Ho, He, Pi).

        Args:
            per_population_stats (dict): Dictionary containing summary statistics per population.
            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.
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

    def _plot_summary_statistics_per_sample_grid(
        self, summary_stats: pd.DataFrame
    ) -> None:
        """Plot summary statistics per locus using a Seaborn PairGrid plot.

        This method plots the summary statistics per locus using a Seaborn PairGrid plot with scatter plots on the upper diagonal, kernel density plots on the lower diagonal, and kernel density plots on the diagonal.

        Args:
            summary_stats (pd.DataFrame): DataFrame containing summary statistics per locus.
        """
        g = sns.PairGrid(summary_stats)
        g.map_upper(sns.scatterplot, color="steelblue")
        g.map_lower(sns.kdeplot, cmap="Blues", fill=True, alpha=0.5)
        g.map_diag(sns.kdeplot, lw=2, color="darkblue", fill=True)

        g.figure.suptitle("Summary Statistics per Locus", y=1.02)
        of: str = f"summary_statistics_per_locus.{self.plot_format}"
        g.savefig(self.output_dir_analysis / of)

        if self.show:
            plt.show()

        plt.close()

    def _plot_summary_statistics_per_population_grid(
        self, per_population_stats: dict, ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot summary statistics per population using violin plots.

        This method plots the summary statistics per population using violin plots for each statistic (Ho, He, Pi).

        Args:
            per_population_stats (dict): Dictionary containing summary statistics per population.
            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.
        """

        if ax is None:
            _, ax = plt.subplots()

        # Combine population data into a single DataFrame
        combined_df = pd.DataFrame()
        for pop_id, stats_df in per_population_stats.items():
            stats_df = stats_df.copy()
            stats_df["Population"] = pop_id
            combined_df: pd.DataFrame = pd.concat(
                [combined_df, stats_df], ignore_index=True
            )

        # Melt DataFrame for easier plotting with seaborn
        melted_df: pd.DataFrame = combined_df.melt(
            id_vars="Population",
            value_vars=["Ho", "He", "Pi"],
            var_name="Statistic",
            value_name="Value",
        )

        pal: str = "Paired" if len(per_population_stats) <= 12 else "tab20"

        # Set up the figure
        plt.figure(figsize=(16, 9))
        sns.violinplot(
            x="Statistic",
            y="Value",
            hue="Population",
            data=melted_df,
            split=True,
            inner="quart",
            palette=pal,
        )

        # Customize the plot for clarity
        plt.title("Summary Statistics per Population")
        plt.xlabel("Statistic")
        plt.ylabel("Value")
        plt.legend(title="Population", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save the plot
        of: str = f"summary_statistics_per_population.{self.plot_format}"
        outpath: Path = self.output_dir_analysis / of
        plt.savefig(outpath)

        if self.show:
            plt.show()

        plt.close()

    def _plot_fst_heatmap(self, fst_between_pops: Dict[tuple, pd.Series]) -> None:
        """Plot a heatmap of Fst values between populations.

        This method calculates the mean Fst for each population pair and plots a heatmap of the mean Fst values between populations. The heatmap is sorted by the mean Fst values across rows and columns, and only the lower triangle is displayed.

        Args:
            fst_between_pops (Dict[tuple, pd.Series]): Dictionary containing Fst values between populations. The keys are population pairs, and the values are Series of Fst values.
        """

        # Calculate mean Fst and 95% CI for each population pair
        if isinstance(fst_between_pops.values(), pd.Series):
            # values are Series of Fst values of shape (n_loci,)
            fst_stats = {
                pop_pair: (
                    fst_series.mean(),
                    fst_series.mean()
                    - 1.96 * fst_series.std() / np.sqrt(len(fst_series)),
                    fst_series.mean()
                    + 1.96 * fst_series.std() / np.sqrt(len(fst_series)),
                )
                for pop_pair, fst_series in fst_between_pops.items()
            }
        else:  # numpy array of shape (n_loci, n_bootstraps)
            fst_stats = {
                pop_pair: (
                    np.nanmean(
                        bootstrap_means
                    ),  # Mean of the bootstrap means ignoring NaNs
                    np.nanpercentile(
                        bootstrap_means, 2.5
                    ),  # Lower 95% CI bound ignoring NaNs
                    np.nanpercentile(
                        bootstrap_means, 97.5
                    ),  # Upper 95% CI bound ignoring NaNs
                )
                for pop_pair, fst_array in fst_between_pops.items()
                for bootstrap_means in [
                    np.nanmean(fst_array, axis=0)
                ]  # Mean across loci for each bootstrap replicate
            }

        # Extract unique population names and initialize an empty Fst matrix
        populations = sorted({pop for pair in fst_stats.keys() for pop in pair})
        fst_matrix = pd.DataFrame(index=populations, columns=populations, data=np.nan)
        annotation_matrix = pd.DataFrame(
            index=populations, columns=populations, data=""
        )

        # Populate the matrices with mean Fst and formatted CI annotations
        for (pop1, pop2), (mean_fst, lower_ci, upper_ci) in fst_stats.items():
            fst_matrix.loc[pop1, pop2] = mean_fst
            fst_matrix.loc[pop2, pop1] = mean_fst
            annotation_matrix.loc[pop1, pop2] = (
                f"{mean_fst:.2f}\n[{lower_ci:.2f}, {upper_ci:.2f}]"
            )
            annotation_matrix.loc[pop2, pop1] = (
                f"{mean_fst:.2f}\n[{lower_ci:.2f}, {upper_ci:.2f}]"
            )

        # Calculate the mean Fst for sorting across rows and columns
        mean_fst_per_population = fst_matrix.mean(axis=1).sort_values(ascending=False)
        sorted_populations = mean_fst_per_population.index

        # Reorder the matrix to apply consistent sorting on both rows and columns
        fst_matrix = fst_matrix.loc[sorted_populations, sorted_populations]
        annotation_matrix = annotation_matrix.loc[
            sorted_populations, sorted_populations
        ]

        # Apply a mask to show only the lower triangle, including the diagonal
        mask = np.triu(np.ones_like(fst_matrix, dtype=bool))

        # Plot the heatmap
        plt.figure(figsize=(48, 48))
        ax = sns.heatmap(
            fst_matrix,
            annot=annotation_matrix,  # Use custom annotations for mean ± CI
            annot_kws={"fontsize": 38},
            fmt="",  # String format for annotations
            cmap="magma",
            mask=mask,  # Mask upper triangle
            linewidths=0,
            vmin=0,
            cbar=True,
            square=True,
        )

        # Reverse the X-axis to ensure proper alignment
        ax.invert_xaxis()  # Reverse X-axis

        # Add title and adjust labels
        ax.set_title("Mean Fst Between Populations (95% CI)", fontsize=72)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=72)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=72)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=72)
        cbar.set_label("Mean Fst", fontsize=72)

        # Save the plot
        of: str = f"fst_between_populations_heatmap.{self.plot_format}"
        plt.savefig(self.output_dir_analysis / of)

        if self.show:
            plt.show()

        plt.close()

    def plot_d_statistics(self, df: pd.DataFrame) -> None:
        """Create plots for D-statistics with multiple test corrections.

        This method creates three plots:
        1. Bar plot of significant D-statistic counts (raw, Bonferroni, FDR-BH).
        2. Distribution plot of D-statistic values using raw p-values for comparison.
        3. Box plot of D-statistics by population combination.

        The method saves the plots to the output directory and displays them if ``show`` is True.

        Args:
            df (pd.DataFrame): DataFrame containing D-statistics and p-values.
        """
        df = df.copy()

        sns.set_theme(style="white")

        # NOTE: Not sure why I re-updated it here.
        # TODO: Check if this is necessary.
        # Update Matplotlib parameters
        mpl.rcParams.update(self.mpl_params)

        # 1. Bar Plot of Significant D-Statistic Counts (Raw, Bonferroni,
        # FDR-BH)
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        significance_df: pd.DataFrame = pd.DataFrame(
            {
                "Correction": ["Raw", "Bonferroni", "FDR-BH"],
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

        ax: plt.Axes = sns.barplot(
            data=significance_df, x="Correction", y="Count", hue="Significance", ax=ax
        )
        ax.set_title(
            "Significant D-Statistics (p < 0.05) per Multiple Test Correction Method"
        )
        ax.set_xlabel("Correction Method")
        ax.set_ylabel("Count")
        ax.legend(title="Significance", loc="best", bbox_to_anchor=(1.2, 1))

        of: str = f"d_statistics_significance_counts.{self.plot_format}"
        fn: Path = self.output_dir_analysis / of
        fig.savefig(fn)

        if self.show:
            plt.show()

        plt.close()

        # 2. Distribution Plot of D-Statistic Values (Using Raw p-values for
        # comparison)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax = sns.histplot(df["Z-Score"], kde=True, ax=ax)
        ax.axvline(
            df["Z-Score"].mean(),
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Mean Z-Score",
        )
        ax.set_title("Distribution of Observed Z-Scores for D-Statistics")
        ax.set_xlabel("Z-Score")
        ax.set_ylabel("Frequency")
        ax.legend()

        of = f"d_statistics_distribution.{self.plot_format}"
        fn = self.output_dir_analysis / of
        fig.savefig(fn)

        if self.show:
            plt.show()

        plt.close()

        # Step 1: Generate "Sample Combo" column
        combo_cols: List[str | int] = [
            col for col in df.columns if col.startswith("Sample")
        ]
        df["Sample Combo"] = df[combo_cols].apply(
            lambda x: "-".join(x.dropna()), axis=1
        )

        # Step 2: Subset the relevant data
        df2: pd.DataFrame = df[
            [
                "Sample Combo",
                "Z-Score",
                "Significant (Raw)",
                "Significant (Bonferroni)",
                "Significant (FDR-BH)",
            ]
        ]

        # Step 3: Determine dynamic figsize and fontsize based on number of unique combinations
        num_combos = df2["Sample Combo"].nunique()

        # Adjust figsize: base width is fixed, height scales with the number of combinations
        fig_width = 14
        fig_height = max(8, num_combos * 0.5)  # 0.5 height per combo, minimum height 8

        # Adjust font size based on the number of combinations
        base_fontsize = 12  # starting font size

        # Adjust down slightly as combos increase
        dynamic_fontsize = max(8, min(base_fontsize, 12 - 0.1 * num_combos))

        # Step 4: Plot with dynamic parameters
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.barplot(
            data=df2,
            x="Z-Score",
            y="Sample Combo",
            hue="Significant (FDR-BH)",
            ax=ax,
            palette={True: "#66c2a5", False: "#fc8d62"},
        )

        # Update plot labels with dynamic fontsize
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=dynamic_fontsize + 15)
        ax.set_title(
            "Z-Score per Sample Combination (with FDR-BH Correction)",
            fontsize=dynamic_fontsize + 30,
        )
        ax.set_xlabel("Z-Score", fontsize=dynamic_fontsize + 20)
        ax.set_ylabel("Sample Combination", fontsize=dynamic_fontsize + 20)
        ax.legend(
            title="Significant (FDR-BH)",
            fontsize=dynamic_fontsize + 20,
            title_fontsize=dynamic_fontsize + 20,
            loc="best",
            bbox_to_anchor=(1.2, 1),
        )

        # Save the plot.
        # Cannot use 'png' here, and must use a vector format (e.g., 'pdf')
        # due to the frequently large number of combinations.
        of: str = f"d_statistics_sample_combos_barplot.pdf"
        outpath: Path = self.output_dir_analysis / of
        fig.savefig(outpath)

        # Show plot if specified
        if self.show:
            plt.show()

        plt.close()

    def plot_fst_outliers(self, outlier_snps: pd.DataFrame) -> None:
        """Create a heatmap of Fst values for outlier SNPs.

        This method creates a heatmap of Fst values for outlier SNPs, highlighting contributing population pairs. The heatmap is sorted by SNP index and population pair, and only the lower triangle is displayed.

        Args:
            outlier_snps (pd.DataFrame): DataFrame containing Fst values and Contributing_Pairs for outlier SNPs.
        """
        # Copy the DataFrame to avoid modifying the original data
        data: pd.DataFrame = outlier_snps.copy()

        # Ensure SNP identifiers are in the index
        if data.index.name != "SNP":
            data.index.name = "SNP"

        data = data.reset_index()

        self.logger.debug(f"data_index: {data.index}")
        self.logger.debug(f"data_columns: {data.columns}")

        # Extract the Fst values (excluding the Contributing_Pairs column)
        fst_values: pd.DataFrame = data.sort_index(axis=1).drop(
            columns=["Contributing_Pairs"]
        )

        if isinstance(data.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            data.columns = data.columns.to_flat_index()

            data.columns = [
                col[0] if "Contributing_Pairs" in col or "SNP" in col else col
                for col in data.columns
            ]

        if isinstance(fst_values.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            fst_values.columns = fst_values.columns.to_flat_index()

            fst_values.columns = [
                col[0] if "Contributing_Pairs" in col or "SNP" in col else col
                for col in fst_values.columns
            ]

        # Convert the Contributing_Pairs from list to set for faster lookup
        data["Contributing_Pairs_Set"] = data["Contributing_Pairs"].apply(set)
        data = data.set_index("SNP")

        # Melt the DataFrame to long format for easier plotting
        fst_long: pd.DataFrame = fst_values.melt(
            id_vars="SNP", var_name="Population_Pair", value_name="Fst"
        )

        # Add a column to indicate whether each cell is a contributing pair
        def is_contributing_pair(row):
            snp = row["SNP"]
            pair = row["Population_Pair"]
            contributing_pairs = data.loc[snp, "Contributing_Pairs_Set"]
            return pair in contributing_pairs

        fst_long["Contributing"] = fst_long.apply(is_contributing_pair, axis=1)

        # Pivot back to wide format with MultiIndex (SNP, Contributing)
        fst_pivot: pd.DataFrame = fst_long.pivot(
            index="SNP", columns="Population_Pair", values="Fst"
        )

        contributing_mask = fst_long.pivot(
            index="SNP", columns="Population_Pair", values="Contributing"
        )

        # Create a mask for highlighting contributing pairs (masks out
        # non-contributing pairs)
        mask = ~contributing_mask.astype(bool)

        # Ensure SNP indices are numeric
        fst_pivot.index = fst_pivot.index.astype(int)
        mask.index = mask.index.astype(int)

        # Sort SNPs by their indices in numeric order
        fst_pivot.sort_index(inplace=True)
        mask: pd.DataFrame = mask.loc[fst_pivot.index]

        # Plot the heatmap
        # Adjust height based on number of SNPs
        fig, ax = plt.subplots(1, 1, figsize=(15, max(8, len(fst_pivot) // 2)))
        sns.set_theme(font_scale=1.2)
        cmap: mpl_colors.LinearSegmentedColormap = sns.diverging_palette(
            220, 20, as_cmap=True
        )

        # Plot all cells with masked non-contributing pairs
        ax: plt.Axes = sns.heatmap(
            fst_pivot,
            cmap=cmap,
            linewidths=0.5,
            linecolor="grey",
            cbar_kws={"label": "Fst"},
            mask=mask,
            square=False,
            xticklabels=True,
            yticklabels=True,
            ax=ax,
        )

        # Overlay the contributing pairs without masking
        sns.heatmap(
            fst_pivot,
            cmap=cmap,
            linewidths=0.5,
            linecolor="grey",
            cbar=False,
            mask=~mask,
            square=False,
            xticklabels=True,
            yticklabels=True,
            ax=ax,
        )

        # Customize the plot
        ax.set_title(
            "Fst Values for Outlier SNPs\nContributing Populations Highlighted"
        )
        ax.set_xlabel("Population Pairs", fontsize=14)
        ax.set_ylabel("SNPs", fontsize=14)

        # Rotate x-axis labels for better readability
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Set y-axis labels to be the SNP indices
        ax.set_yticklabels(fst_pivot.index)

        # Save the plot
        of: str = f"outlier_snps_heatmap.{self.plot_format}"
        outpath: Path = self.output_dir_analysis / of
        fig.savefig(outpath)

        if self.show:
            plt.show()

        plt.close()

    def plot_summary_statistics(self, summary_statistics: dict) -> None:
        """Plot summary statistics per sample and per population.

        This method plots summary statistics per sample and per population on the same figure. The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst). The method also plots summary statistics per sample and per population using Seaborn PairGrid plots. The method saves the plots to the output directory and displays them if ``show`` is True.

        Args:
            summary_statistics (dict): Dictionary containing summary statistics for plotting.
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

        # Plot Fst heatmap
        self._plot_fst_heatmap(summary_statistics["Fst_between_populations"])

    def plot_pca(
        self, pca: PCA, alignment: np.ndarray, popmap: pd.DataFrame, dimensions: int = 2
    ) -> None:
        """Plot a PCA scatter plot.

        This method plots a PCA scatter plot with 2 or 3 dimensions, colored by population, and labeled by population with symbols for each sample. The plot is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/gtdata/plots/pca_plot.{plot_format}``. The plot is saved in the format specified by the ``plot_format`` attribute.

        Note:
            - The PCA object must be fitted before calling this method.
            - The PCA object must be fitted using the genotype data provided in the `alignment` argument.
            - The `popmap` DataFrame must contain the population mapping information with columns "SampleID" and "PopulationID".
            - The `dimensions` argument must be either 2 or 3.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/pca_plot.{plot_format}``.
            - The plot is colored by population and labeled by population with symbols for each sample.

        Args:
            pca (sklearn.decomposition.PCA): The fitted PCA object.
                The fitted PCA object used for dimensionality reduction and transformation.

            alignment (numpy.ndarray): The genotype data used for PCA.
                The genotype data in the form of a numpy array.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            dimensions (int, optional): Number of dimensions to plot (2 or 3). Defaults to 2.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.

        Returns:
            None: The PCA scatter plot is saved to a file.
        """
        pca_transformed = pd.DataFrame(
            pca.transform(alignment),
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        )

        popmap = pd.DataFrame(
            list(popmap.items()), columns=["SampleID", "PopulationID"]
        )

        popmap.columns = ["SampleID", "PopulationID"]
        pca_transformed["PopulationID"] = popmap["PopulationID"]

        if dimensions == 2:
            sns.scatterplot(data=pca_transformed, x="PC1", y="PC2", hue="PopulationID")
        elif dimensions == 3:
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.add_subplot(111, projection="3d")
            sns.scatterplot(
                data=pca_transformed,
                x="PC1",
                y="PC2",
                z="PC3",
                hue="PopulationID",
                ax=ax,
            )
        else:
            raise ValueError("dimensions must be 2 or 3")

        of: str = f"pca_plot.{self.plot_format}"
        outpath: Path = self.output_dir_analysis / of
        plt.savefig(outpath)

        if self.show:
            plt.show()

        plt.close()

    def plot_dapc(
        self,
        dapc: LinearDiscriminantAnalysis,
        alignment: np.ndarray,
        popmap: pd.DataFrame,
        dimensions: int = 2,
    ):
        """Plot a DAPC scatter plot.

        This method plots a DAPC scatter plot with 2 or 3 dimensions, colored by population, and labeled by population with symbols for each sample. The plot is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/gtdata/plots/dapc_plot.{plot_format}``. The plot is saved in the format specified by the ``plot_format`` attribute.

        Args:
            dapc (sklearn.discriminant_analysis.LinearDiscriminantAnalysis):  The fitted DAPC object used for dimensionality reduction and transformation.

            alignment (numpy.ndarray): The genotype data in the form of a numpy array.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            dimensions (int, optional): Number of dimensions to plot (2 or 3). Defaults to 2.

        Returns:
            None: The DAPC scatter plot is saved to a file.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.

        Note:
            - The DAPC object must be fitted before calling this method.
            - The DAPC object must be fitted using the genotype data provided in the `alignment` argument.
            - The `popmap` DataFrame must contain the population mapping information with columns "SampleID" and "PopulationID".
            - The `dimensions` argument must be either 2 or 3.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/dapc_plot.{plot_format}``.
            - The plot is colored by population and labeled by population with symbols for each sample.
        """
        dapc_transformed = pd.DataFrame(
            dapc.transform(alignment),
            columns=[f"DA{i+1}" for i in range(dapc.n_components_)],
        )
        dapc_transformed["PopulationID"] = popmap["PopulationID"]

        if dimensions == 2:
            sns.scatterplot(data=dapc_transformed, x="DA1", y="DA2", hue="PopulationID")
        elif dimensions == 3:
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.add_subplot(111, projection="3d")
            sns.scatterplot(
                data=dapc_transformed,
                x="DA1",
                y="DA2",
                z="DA3",
                hue="PopulationID",
                ax=ax,
            )
        else:
            raise ValueError("dimensions must be 2 or 3")

        of: str = f"dapc_plot.{self.plot_format}"
        outpath: Path = self.output_dir_analysis / of
        plt.savefig(outpath)

        if self.show:
            plt.show()

        plt.close()

    def plot_sankey_filtering_report(
        self, df: pd.DataFrame, search_mode: bool = False
    ) -> None:
        """Plot a Sankey diagram for the filtering report.

        This method plots a Sankey diagram for the filtering report. The Sankey diagram shows the flow of loci through the filtering steps. The loci are filtered based on the missing data proportion, MAF, MAC, and other filtering thresholds. The Sankey diagram shows the number of loci kept and removed at each step. The Sankey diagram is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/nremover/plots/sankey_plot_{thresholds}.{plot_format}``. The plot is saved in the format specified by the ``plot_format`` attribute.

        Args:
            df (pd.DataFrame): The input DataFrame containing the filtering report.
            search_mode (bool, optional): Whether the Sankey diagram is being plotted in search mode. Defaults to False.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input DataFrame is empty.
            ValueError: Raised if multiple threshold combinations are detected when attempting to plot the Sankey diagram.

        Note:
            The Sankey diagram shows the flow of loci through the filtering steps.

            The loci are filtered based on the missing data proportion, MAF, MAC, and other filtering thresholds.

            The Sankey diagram shows the number of loci kept and removed at each step.

            The Sankey diagram is saved to a file in the `output_dir` directory.

            The Sankey diagram is displayed if the `show` attribute is True.

            The Sankey diagram is saved in the format specified by the `plot_format` attribute.

            The Sankey diagram is saved with the filename: ``<prefix>_output/nremover/plots/sankey_plot_{thresholds}.{plot_format}``.

            The Sankey diagram is colored based on the kept and removed loci.

            The kept loci are colored green, and the removed loci are colored red.

            The Sankey diagram is plotted using the Bokeh plotting library.

            The Sankey diagram is plotted using the Holoviews library.

            The Sankey diagram is plotted using the Bokeh extension for Holoviews.

            The Sankey diagram is plotted using the `hv.Sankey` method from Holoviews.

            The Sankey diagram is plotted with edge labels showing the number of loci kept and removed at each step.

            The Sankey diagram is plotted with a common "Removed" node for the removed loci at each step.

            The Sankey diagram is plotted with a common "Kept" node for the kept loci moving to the next filter.

            The Sankey diagram is plotted with a common "Unfiltered" node for the initial unfiltered loci.

            The Sankey diagram is plotted with a common "Kept" node for the final kept loci.

            The Sankey diagram is plotted with a common "Removed" node for the final removed loci.

        Example:
            >>> from snpio import VCFReader, NRemover
            >>> gd = VCFReader(filename="example.vcf", popmapfile="popmap.txt")
            >>> nrm = NRemover(gd)
            >>> nrm.filter_missing(0.75).filter_mac(2).filter_missing_pop(0.5).filter_singletons(exclude_heterozygous=True).filter_monomorphic().filter_biallelic().resolve()
            >>> nrm.plot_sankey_filtering_report()
        """
        # Import Holoviews and Bokeh
        hv.extension("bokeh")

        plot_dir: Path = self.output_dir_nrm / "sankey_plots"
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

            if isinstance(thresholds, list):
                thresholds = "_".join([str(threshold) for threshold in thresholds])

            # Save the plot to an HTML file
            of: str = f"filtering_results_sankey_thresholds{thresholds}.html"
            fname: Path = plot_dir / of
            hv.save(sankey_plot, fname, fmt="html")

        except ValueError as e:
            self.logger.warning(
                f"Failed to generate Sankey plot with thresholds: {thresholds}: error: {e}"
            )

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
            - The input dataframe must contain the genotype counts.
            - The input dataframe must have the genotype counts as columns.
            - The input dataframe must have the genotype counts as rows.
            - The input dataframe must have the genotype counts as values.
            - The input dataframe must have the genotype counts as integers.
            - The input dataframe must have the genotype counts as strings.
            - The input dataframe must have the genotype counts as IUPAC codes.
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

        msg: str = f"Plotting complete. Plots saved to directory {self.output_dir_nrm}."
        self.logger.info(msg)

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
            outpath: Path = self.output_dir_nrm / of
            fig.savefig(outpath)

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
            outpath: Path = self.output_dir_nrm / of
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
            outpath: Path = self.output_dir_nrm / of
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
            outpath: Path = self.output_dir_nrm / of
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
            outpath: Path = self.output_dir_nrm / of
            fig.savefig(outpath)

            if self.show:
                plt.show()

            plt.close()

        else:
            self.logger.info("Boolean data is empty.")

    def plot_filter_report(self, df: pd.DataFrame) -> None:
        """Plot the filter report.

        This method plots the filter report data. The filter report data contains the proportion of loci removed and kept for each filtering threshold. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The dataframe containing the filter report data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.

        Note:
            - The input dataframe must contain the filter report data.
            - The input dataframe must contain the filter report data for the removed and kept loci.
            - The input dataframe must contain the filter report data for the filtering threshold.
            - The input dataframe must contain the filter report data for the filtering method.
            - The input dataframe must contain the filter report data for the filtering step.
            - The input dataframe must contain the filter report data for the removed and kept loci counts.
            - The input dataframe must contain the filter report data for the removed and kept loci proportions.
            - The input dataframe must contain the filter report data for the filtering thresholds.
            - The input dataframe must contain the filter report data for the missing data threshold.
            - The input dataframe must contain the filter report data for the MAF threshold.
            - The input dataframe must contain the filter report data for the MAC threshold.
            - The input dataframe must contain the filter report data for the boolean threshold.
        """
        self.logger.info("Generating filter report plots...")
        self.logger.debug(f"Filter report data: {df}")

        df["Missing_Threshold"] = df["Missing_Threshold"].astype(float)
        df["MAF_Threshold"] = df["MAF_Threshold"].astype(float)
        df["MAC_Threshold"] = df["MAC_Threshold"].astype(int)
        df["Bool_Threshold"] = df["Bool_Threshold"].astype(float)
        df = df.sort_values(
            by=["Missing_Threshold", "MAF_Threshold", "Bool_Threshold", "MAC_Threshold"]
        )
        df["Removed_Prop"] = df["Removed_Prop"].astype(float)
        df["Kept_Prop"] = df["Kept_Prop"].astype(float)
        df["Filter_Method"] = df["Filter_Method"].str.replace("_", " ").str.title()
        df["Removed_Prop"] = df["Removed_Prop"].round(2)
        df["Kept_Prop"] = df["Kept_Prop"].round(2)

        # plot the boxplots
        fig, axs = plt.subplots(5, 2, figsize=(24, 12))

        kwargs = {"y": "Removed_Prop", "hue": "Filter_Method", "data": df}

        for i, (ax, sns_method, xval) in enumerate(
            zip(
                axs.flatten(),
                [
                    sns.boxplot,
                    sns.histplot,
                    sns.lineplot,
                    sns.lineplot,
                    sns.violinplot,
                    sns.violinplot,
                    sns.histplot,
                    sns.ecdfplot,
                    sns.histplot,
                    sns.ecdfplot,
                ],
                [
                    "Missing_Threshold",
                    "Missing_Threshold",
                    "MAF_Threshold",
                    "MAC_Threshold",
                    "Bool_Threshold",
                    "Bool_Threshold",
                    "MAF_Threshold",
                    "MAF_Threshold",
                    "MAC_Threshold",
                    "MAC_Threshold",
                ],
            )
        ):
            if sns_method == sns.violinplot and i == 2:
                kwargs["inner"] = "box"

            elif sns_method == sns.violinplot and i == 3:
                kwargs["inner"] = "quartile"

            kwargs["x"] = xval

            ax: plt.Axes = sns_method(**kwargs, ax=ax)

        plot_format = plot_format.lower()
        of: Path = self.output_dir_nrm / f"filter_report.{self.plot_format}"
        of.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(of)

        if self.show:
            plt.show()

        plt.close()

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

    def plot_performance(
        self,
        resource_data: Dict[str, List[Union[float, int]]],
        color: str = "#8C56E3",
        figsize: Tuple[int, int] = (18, 10),
    ) -> None:
        """Plots the performance metrics: CPU Load, Memory Footprint, and Execution Time using boxplots.

        This function takes a dictionary of performance data and plots the metrics for each method using boxplots to show variability. The resulting plots are saved in a file of the specified format. The plot shows the CPU Load, Memory Footprint, and Execution Time for each method. The plot is colored based on the specified color.

        Args:
            resource_data (Dict[str, List[Union[int, float]]]): Dictionary with performance data. Keys are method names, and values are lists of dictionaries with keys 'cpu_load', 'memory_footprint', and 'execution_time'.
            color (str, optional): Color to be used in the plot. Should be a valid color string. Defaults to "#8C56E3".
            figsize (Tuple[int, int], optional): Size of the figure. Should be a tuple of 2 integers. Defaults to (18, 10).

        Returns:
            None. The function saves the plot to a file.

        Raises:
            ValueError: Raised if the input data is not a dictionary.

        Note:
            - The performance data should be in the format of a dictionary with method names as keys and lists of dictionaries as values. Each dictionary should have keys 'cpu_load', 'memory_footprint', and 'execution_time'.

            - The plot will be saved in the '<prefix>_output/gtdata/plots/performance' directory.

            - Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").
        """
        plot_dir = Path(f"{self.prefix}_output", "gtdata", "plots", "performance")
        plot_dir.mkdir(exist_ok=True, parents=True)

        methods = list(resource_data.keys())

        # Prepare data for boxplots
        cpu_loads = {
            method: [data["cpu_load"] for data in resource_data[method]]
            for method in methods
        }
        memory_footprints = {
            method: [data["memory_footprint"] for data in resource_data[method]]
            for method in methods
        }
        execution_times = {
            method: [data["execution_time"] for data in resource_data[method]]
            for method in methods
        }

        # Convert to a format suitable for seaborn boxplots
        cpu_data = [(method, val) for method, vals in cpu_loads.items() for val in vals]
        memory_data = [
            (method, val) for method, vals in memory_footprints.items() for val in vals
        ]
        execution_data = [
            (method, val) for method, vals in execution_times.items() for val in vals
        ]

        # Separate the data for plotting
        cpu_methods, cpu_values = zip(*cpu_data)
        memory_methods, memory_values = zip(*memory_data)
        exec_methods, exec_values = zip(*execution_data)

        # Set up the figure and axes
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # Data for plotting
        plot_data = [
            (cpu_methods, cpu_values, "CPU Load (%)", "CPU Load Performance"),
            (
                memory_methods,
                memory_values,
                "Memory Footprint (MB)",
                "Memory Footprint Performance",
            ),
            (
                exec_methods,
                exec_values,
                "Execution Time (seconds)",
                "Execution Time Performance",
            ),
        ]

        # Plot each metric
        for ax, (methods, values, ylabel, title) in zip(axs, plot_data):
            sns.barplot(x=methods, y=values, color=color, ax=ax)
            ax.set_xlabel("Methods")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(ax.get_xticks(), minor=False)

            ticklabs = [
                x.get_text().replace("_", " ").title() for x in ax.get_xticklabels()
            ]
            ticklabs = [x.replace("Filter Maf", "Filter MAF") for x in ticklabs]
            ticklabs = [x.replace("Filter Mac", "Filter MAC") for x in ticklabs]
            ax.set_xticklabels(ticklabs, rotation=90)
            ax.set_ylim(bottom=-0.05)

        # Save the plot to a file
        of = plot_dir / f"benchmarking_barplot.{self.plot_format}"
        fig.savefig(of)

        if self.show:
            plt.show()
        plt.close()

    def run_pca(
        self,
        n_components: Optional[int] = None,
        center: bool = True,
        scale: bool = False,
        n_axes: int = 2,
        point_size: int = 15,
        bottom_margin: float = 0,
        top_margin: float = 0,
        left_margin: float = 0,
        right_margin: float = 0,
        width: int = 1088,
        height: int = 700,
    ) -> Tuple[np.ndarray, PCA]:
        """Runs PCA and makes scatterplot with colors showing missingness.

        Genotypes are plotted as separate shapes per population and colored according to missingness per individual.

        This function is run at the end of each imputation method, but can be run independently to change plot and PCA parameters such as ``n_axes=3`` or ``scale=True`` for full customization. Setting ``n_axes=3`` will make a 3D PCA plot.

        PCA (principal component analysis) scatterplot can have either two or three axes, set with the n_axes parameter.

        The plot is saved as both an interactive HTML file and as a static image. Each population is represented by point shapes. The interactive plot has associated metadata when hovering over the points.

        Files are saved to a reports directory as <prefix>_output/imputed_pca.<plot_format|html>. Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").

        Args:
            n_components (int, optional): Number of principal components to include in the PCA. Defaults to None (all components).

            center (bool, optional): If True, centers the genotypes to the mean before doing the PCA. If False, no centering is done. Defaults to True.

            scale (bool, optional): If True, scales the genotypes to unit variance before doing the PCA. If False, no scaling is done. Defaults to False.

            n_axes (int, optional): Number of principal component axes to plot. Must be set to either 2 or 3. If set to 3, a 3-dimensional plot will be made. Defaults to 2.

            point_size (int, optional): Point size for scatterplot points. Defaults to 15.

            bottom_margin (int, optional): Adjust bottom margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            top_margin (int, optional): Adjust top margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            left_margin (int, optional): Adjust left margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            right_margin (int, optional): Adjust right margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            width (int, optional): Width of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 1088.

            height (int, optional): Height of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 700.

        Note:
            The PCA is run on the genotype data. Missing data is imputed using K-nearest-neighbors (per-sample) before running the PCA. The PCA is run using the sklearn.decomposition.PCA class.

            The PCA data is saved as a numpy array with shape (n_samples, n_components).

            The PCA object is saved as a sklearn.decomposition.PCA object. Any of the sklearn.decomposition.PCA attributes can be accessed from this object. See the sklearn documentation.

            The explained variance ratio can be calculated from the PCA object.

            The plot is saved as both an interactive HTML file and as a static image. Each population is represented by point shapes. The interactive plot has associated metadata when hovering over the points.

            Files are saved to a reports directory as <prefix>_output/imputed_pca.<plot_format|html>. Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg).

        Returns:
            numpy.ndarray: PCA data as a numpy array with shape (n_samples, n_components).

            sklearn.decomposision.PCA: Scikit-learn PCA object from sklearn.decomposision.PCA. Any of the sklearn.decomposition.PCA attributes can be accessed from this object. See sklearn documentation.

        Raises:
            ValueError: If n_axes is not set to 2 or 3.

        Example:
            >>> from snpio import Plotting, VCFReader
            >>>
            >>> gd = VCFReader("snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz", popmap="snpio/example_data/popmaps/phylogen_nomx.popmap", force_popmap=True, verbose=True)
            >>>
            >>> # Define the plotting object
            >>> plotting = Plotting(gd)
            >>>
            >>> # Run the PCA and get the components and PCA object
            >>> components, pca = plotting.run_pca()
            >>>
            >>> # Calculate and print explained variance ratio
            >>> explvar = pca.explained_variance_ratio_
            >>> print(explvar)
            >>> # Output: [0.123, 0.098, 0.087, ...]
        """
        plot_dir = Path(f"{self.prefix}_output", "gtdata", "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        if n_axes not in {2, 3}:
            msg = f"{n_axes} axes are not supported; n_axes must be either 2 or 3."
            self.logger.error(msg)
            raise ValueError(msg)

        # Encode the genotype data.
        ge = GenotypeEncoder(self.genotype_data)
        df = misc.validate_input_type(ge.genotypes_012, return_type="df")
        df = df.astype(int).replace([-9], np.nan)

        pca_df = df.copy()
        if center or scale:
            # Center data to mean. Scaling to unit variance is off.
            ss = StandardScaler(with_mean=center, with_std=scale)
            pca_df = ss.fit_transform(pca_df)

        # Run PCA.
        model = PCA(n_components=n_components)

        # PCA can't handle missing data. So impute it here using the
        # K-nearest-neighbors (per-sample).
        imputer = KNNImputer(weights="distance")
        pca_df = imputer.fit_transform(pca_df)
        components = model.fit_transform(pca_df)

        cols, idx = ["Axis1", "Axis2", "Axis3"], list(range(3))
        df_pca = pd.DataFrame(components[:, idx], columns=cols)
        df_pca["SampleID"] = self.genotype_data.samples
        df_pca["Population"] = self.genotype_data.populations
        df_pca["Size"] = point_size

        _, ind, __, ___, ____ = self.genotype_data.calc_missing(df, use_pops=False)

        df_pca["missPerc"] = ind

        # ggplot default
        my_scale = [("rgb(19, 43, 67)"), ("rgb(86,177,247)")]

        z = "Axis3" if n_axes == 3 else None
        pc1 = model.explained_variance_ratio_[0] * 100
        pc2 = model.explained_variance_ratio_[1] * 100
        labs = {
            "Axis1": f"PC1 ({pc1:.2f}% Explained Variance)",
            "Axis2": f"PC2 ({pc2:.2f}% Explained Variance)",
            "missPerc": "Missing Prop.",
            "Population": "Population",
        }

        if z is not None:
            pc3 = model.explained_variance_ratio_[2] * 100
            labs["Axis3"] = f"PC3 ({pc3:.2f%}% Explained Variance)"
            kwargs = dict(zip(["x", "y", "z"], cols))
            func = px.scatter_3d
        else:
            kwargs = dict(zip(["x", "y"], cols[0:2]))
            func = px.scatter

        fig = func(
            df_pca,
            **kwargs,
            color="missPerc",
            symbol="Population",
            color_continuous_scale=my_scale,
            custom_data=["Axis3", "SampleID", "Population", "missPerc"],
            size="Size",
            size_max=point_size,
            labels=labs,
            range_color=[0.0, 1.0],
            title="PCA Per-Population Missingness Scatterplot",
        )

        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "Axis 1: %{x}",
                    "Axis 2: %{y}",
                    "Axis 3: %{customdata[0]}",
                    "Sample ID: %{customdata[1]}",
                    "Population: %{customdata[2]}",
                    "Missing Prop.: %{customdata[3]}",
                ]
            ),
        )
        fig.update_layout(
            showlegend=True,
            margin=dict(
                b=bottom_margin,
                t=top_margin + 100,
                l=left_margin,
                r=right_margin,
            ),
            width=width,
            height=height,
            legend_orientation="h",
            legend_title="Population",
            legend_title_side="top",
            font=dict(size=24),
        )

        of = plot_dir / f"pca_missingness.{self.plot_format}"
        fig.write_html(of.with_suffix(".html"))
        fig.write_image(of, format=self.plot_format)
        return components, model

    def visualize_missingness(
        self,
        df: pd.DataFrame,
        prefix: Optional[str] = None,
        zoom: bool = False,
        horizontal_space: float = 0.6,
        vertical_space: float = 0.6,
        bar_color: str = "gray",
        heatmap_palette: str = "magma",
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        """Make multiple plots to visualize missing data.

        This method makes multiple plots to visualize missing data. The plots include per-individual and per-locus missing data proportions, per-population + per-locus missing data proportions, per-population missing data proportions, and per-individual and per-population missing data proportions.

        Note:
            - The plots are saved in the '<prefix>_output/gtdata/plots' directory.

            - Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").

            - The heatmap plot uses the seaborn library. The heatmap palette can be set using the heatmap_palette parameter. The default palette is 'magma'.

            - The barplots use the matplotlib library. The color of the bars can be set using the bar_color parameter. The default color is 'gray'.

        Args:
            df (pandas.DataFrame): DataFrame with snps to visualize.

            prefix (str, optional): Prefix to use for the output files. If None, the prefix is set to the input filename. Defaults to None.

            zoom (bool, optional): If True, zooms in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to False.


            horizontal_space (float, optional): Set width spacing between subplots. If your plot are overlapping horizontally, increase horizontal_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            vertical_space (float, optioanl): Set height spacing between subplots. If your plots are overlapping vertically, increase vertical_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            bar_color (str, optional): Color of the bars on the non-stacked barplots. Can be any color supported by matplotlib. See matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, optional): Palette to use for heatmap plot. Can be any palette supported by seaborn. See seaborn documentation. Defaults to 'magma'.

        Returns:
            Tuple: Returns the missing data proportions for per-individual, per-locus, per-population + per-locus, per-population, and per-individual + per-population.

        Raises:
            ValueError: If the input data is not a pandas DataFrame.
        """
        # For missingness report filename.
        prefix = self.prefix if prefix is None else prefix

        plot_dir = Path(f"{self.prefix}_output", "gtdata", "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(df, pd.DataFrame):
            df = misc.validate_input_type(df, return_type="df")

        loc, ind, poploc, poptotal, indpop = self.genotype_data.calc_missing(df)

        ncol = 3
        nrow = 1 if self.genotype_data.populations is None else 2

        fig, axes = plt.subplots(nrow, ncol, figsize=(8, 11))
        plt.subplots_adjust(wspace=horizontal_space, hspace=vertical_space)
        fig.suptitle("Missingness Report")

        ax = axes[0, 0]

        ax.set_title("Per-Individual")
        ax.barh(self.genotype_data.samples, ind, color=bar_color, height=1.0)

        if not zoom:
            ax.set_xlim([0, 1])

        ax.set_ylabel("Sample")
        ax.set_xlabel("Missing Prop.")
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        ax = axes[0, 1]

        ax.set_title("Per-Locus")
        ax.barh(range(self.genotype_data.num_snps), loc, color=bar_color, height=1.0)
        if not zoom:
            ax.set_xlim([0, 1])
        ax.set_ylabel("Locus")
        ax.set_xlabel("Missing Prop.")
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        id_vars = ["SampleID"]
        if poptotal is not None:
            ax = axes[0, 2]

            ax.set_title("Per-Population Total")
            ax.barh(poptotal.index, poptotal, color=bar_color, height=1.0)
            if not zoom:
                ax.set_xlim([0, 1])
            ax.set_xlabel("Missing Prop.")
            ax.set_ylabel("Population")

            ax = axes[1, 0]

            ax.set_title("Per-Population +\nPer-Locus", loc="center")

            vmax = None if zoom else 1.0

            sns.heatmap(
                poploc,
                vmin=0.0,
                vmax=vmax,
                cmap=sns.color_palette(heatmap_palette, as_cmap=True),
                yticklabels=False,
                cbar_kws={"label": "Missing Prop."},
                ax=ax,
            )
            ax.set_xlabel("Population")
            ax.set_ylabel("Locus")

            id_vars.append("Population")

        melt_df = indpop.isna()
        melt_df["SampleID"] = self.genotype_data.samples
        indpop["SampleID"] = self.genotype_data.samples

        if poptotal is not None:
            melt_df["Population"] = self.genotype_data.populations
            indpop["Population"] = self.genotype_data.populations

        melt_df = melt_df.melt(value_name="Missing", id_vars=id_vars)
        melt_df = melt_df.sort_values(by=id_vars[::-1])
        melt_df["Missing"] = melt_df["Missing"].replace(
            [False, True], ["Present", "Missing"]
        )

        ax = axes[0, 2] if poptotal is None else axes[1, 1]

        ax.set_title("Per-Individual")

        ax = sns.histplot(
            data=melt_df, y="variable", hue="Missing", multiple="fill", ax=ax
        )

        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        ax.get_legend().set_title(None)

        if poptotal is not None:
            ax = axes[1, 2]

            ax.set_title("Per-Population")
            ax = sns.histplot(
                data=melt_df, y="Population", hue="Missing", multiple="fill", ax=ax
            )
            ax.get_legend().set_title(None)

        of = plot_dir / f"{prefix}_missingness_report.{self.plot_format}"
        fig.savefig(of)

        if self.show:
            plt.show()
        plt.close()

        return loc, ind, poploc, poptotal, indpop

    def plot_dist_matrix(
        self,
        df: pd.DataFrame,
        pvals: pd.DataFrame | None = None,
        palette: str = "coolwarm",
        title: str = "Distance Matrix",
    ) -> None:
        """Plot a distance matrix.

        This method plots a distance matrix. A distance matrix is a heatmap plot that shows the pairwise distances between populations. The plot is saved to a file.

        Args:
            df (pd.DataFrame): The input dataframe containing the distance matrix data.
            pvals (pd.DataFrame | None): The input dataframe containing the p-values for the distance matrix data. If not provided, the p-values will not be plotted. Defaults to None.
            palette (str): The color palette to use for the heatmap plot. Defaults to "coolwarm".
            title (str): The title of the plot. Defaults to "Distance Matrix".

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframes are empty.

        Note:
            - The input dataframe must contain the distance matrix data.
            - The input dataframe must contain the p-values for the distance matrix data.
            - The plot will be saved in the '<prefix>_output/gtdata/plots' directory.
            - Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").
            - The distance matrix is a heatmap plot that shows the pairwise distances between populations.
        """
        plot_dir = Path(f"{self.prefix}_output", "analysis", "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            msg = "The input distance matrix is empty."
            self.logger.error(msg)
            raise ValueError(msg)

        if pvals is not None and pvals.empty:
            msg = "The input p-values dataframe is empty."
            self.logger.error(msg)
            raise ValueError(msg)

        # Compute row sums to determine sorting order
        sort_order = df.sum(axis=1).sort_values(ascending=False).index

        # Reorder both rows and columns using the same order
        df = df.loc[sort_order, sort_order]

        if pvals is not None:
            pvals = pvals.loc[sort_order, sort_order]

        # Create a mask for the upper triangle (excluding diagonal)
        mask = np.tril(np.ones(df.shape, dtype=bool), k=-1)

        sns.set_style(style="white")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap=palette,
            cbar_kws={"label": "Distance"},
            robust=True,
            mask=mask,
            ax=ax,
        )

        ax.invert_yaxis()

        if pvals is not None:
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    if pvals.iloc[i, j] < 0.05:
                        ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center")

        ax.set_title(string.capwords(title))
        ax.set_xlabel("Population")
        ax.set_ylabel("Population")

        fn = "_".join(title.lower().split())
        fig.savefig(plot_dir / f"{fn}.{self.plot_format}")

        if self.show:
            plt.show()
        plt.close()
