import itertools
import math
import warnings
from pathlib import Path
from typing import Optional

warnings.simplefilter(action="ignore", category=FutureWarning)

import holoviews as hv
import matplotlib as mpl
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from holoviews import opts
from mpl_toolkits.mplot3d import Axes3D  # Don't remove this import.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

hv.extension("bokeh")

from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from snpio.utils import misc
from snpio.utils.logging import setup_logger


class Plotting:
    """Class with various methods for plotting."""

    def __init__(
        self,
        genotype_data,
        show: bool = False,
        plot_format: str = "png",
        dpi: int = 300,
        plot_fontsize: int = 18,
        plot_title_fontsize: int = 22,
        plot_ticksize: int = 16,
        despine: bool = True,
        verbose: bool = False,
        debug: bool = False,
        prefix: Optional[str] = None,
    ):
        """Initialize the Plotting class.

        Args:
            genotype_data (GenotypeData): Initialized GentoypeData object.

            show (bool, optional): Whether to display the plots. Defaults to False.

            plot_format (str, optional): The format in which to save the plots. Defaults to "png".

            dpi (int, optional): The resolution of the saved plots. Unused for vector `plot_format` types. Defaults to 300.

            plot_fontsize (int, optional): The font size for the plot labels. Defaults to 18.

            plot_title_fontsize (int, optional): The font size for the plot titles. Defaults to 22.

            plot_ticksize (int, optional): The font size for the plot ticks. Defaults to 16.

            despine (bool, optional): Whether to remove the top and right plot axis spines. Defaults to True.

            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.

            debug (bool, optional): Whether to enable debug logging. Defaults to False.

            prefix (str, optional): The prefix to use for the output files. If not provided, then the prefix set in `genotype_data` will be used. Defaults to None.
        """
        self.genotype_data = genotype_data
        self.alignment = genotype_data.snp_data
        self.popmap = genotype_data.populations
        self.populations = genotype_data.populations
        self.prefix = genotype_data.prefix
        self.show = show
        self.plot_format = plot_format
        self.dpi = dpi
        self.verbose = verbose

        log_file = Path(f"{self.prefix}_output", "logs", "plotting.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)

        level = "DEBUG" if debug else "INFO"
        self.logger = setup_logger(__name__, log_file=log_file, level=level)

        self.boolean_filter_methods = [
            "filter_singletons",
            "filter_biallelic",
            "filter_monomorphic",
            "thin_loci",
            "filter_linked",
        ]

        self.missing_filter_methods = [
            "filter_missing",
            "filter_missing_sample",
            "filter_missing_pop",
        ]

        self.maf_filter_methods = ["filter_maf", "filter_mac"]

        self.output_dir = Path(f"{self.prefix}_output", "nremover", "plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        mpl_params = {
            "axes.titlesize": plot_title_fontsize,
            "axes.labelsize": plot_fontsize,
            "xtick.labelsize": plot_ticksize,
            "ytick.labelsize": plot_ticksize,
            "legend.fontsize": plot_fontsize,
            "figure.titlesize": plot_title_fontsize,
            "font.size": plot_fontsize,
            "font.family": "sans-serif",
            "axes.grid": False,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False if despine else True,
            "axes.spines.right": False if despine else True,
        }

        mpl.rcParams.update(mpl_params)

    def _plot_summary_statistics_per_sample(self, summary_stats, ax=None):
        """Plot summary statistics per sample.

        Args:
            summary_stats (pandas.DataFrame): The DataFrame containing the summary statistics per sample to be plotted.

            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(summary_stats["Ho"], label="Ho")
        ax.plot(summary_stats["He"], label="He")
        ax.plot(summary_stats["Pi"], label="Pi")
        ax.plot(summary_stats["Fst"], label="Fst")

        ax.set_xlabel("Locus")
        ax.set_ylabel("Value")
        ax.set_title("Summary Statistics per Sample")
        ax.legend()

    def _plot_summary_statistics_per_population(self, summary_stats, popmap, ax=None):
        """Plot summary statistics per population.

        Args:
            summary_stats (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

            popmap (pd.DataFrame): The DataFrame containing the population mapping used to group the summary statistics.

            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.

        """
        if ax is None:
            _, ax = plt.subplots()

        # Group the summary statistics by population.
        pop_summary_stats = summary_stats.groupby(popmap["PopulationID"]).mean()

        ax.plot(pop_summary_stats["Ho"], label="Ho")
        ax.plot(pop_summary_stats["He"], label="He")
        ax.plot(pop_summary_stats["Pi"], label="Pi")
        ax.plot(pop_summary_stats["Fst"], label="Fst")

        ax.set_xlabel("Population")
        ax.set_ylabel("Value")
        ax.set_title("Summary Statistics per Population")
        ax.legend()

    def _plot_summary_statistics_per_population_grid(self, summary_statistics_df):
        """Plot summary statistics per population using a Seaborn PairGrid plot.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.
        """
        g = sns.PairGrid(summary_statistics_df)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=3, legend=False)

        of = f"summary_statistics_per_population_grid.{self.plot_format}"
        of = self.output_dir / of
        g.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()
        plt.close()

    def _plot_summary_statistics_per_sample_grid(self, summary_statistics_df):
        """Plot summary statistics per sample using a Seaborn PairGrid plot.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.
        """
        g = sns.PairGrid(summary_statistics_df)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=3, legend=False)

        of = f"summary_statistics_per_sample_grid.{self.plot_format}"
        of = self.output_dir / of
        g.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()
        plt.close()

    def plot_summary_statistics(self, summary_statistics_df):
        """Plot summary statistics per sample and per population on the same figure.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

        self._plot_summary_statistics_per_sample(summary_statistics_df, ax=axes[0])
        self._plot_summary_statistics_per_population(summary_statistics_df, ax=axes[1])

        of = f"summary_statistics.{self.plot_format}"
        of = self.output_dir / of
        fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")

        if self.show:
            plt.show()
        plt.close()

        self._plot_summary_statistics_per_sample_grid(summary_statistics_df)
        self._plot_summary_statistics_per_population_grid(summary_statistics_df)

    def plot_pca(self, pca, alignment, popmap, dimensions=2):
        """Plot a PCA scatter plot.

        Args:
            pca (sklearn.decomposition.PCA): The fitted PCA object.
                The fitted PCA object used for dimensionality reduction and transformation.

            alignment (numpy.ndarray): The genotype data used for PCA.
                The genotype data in the form of a numpy array.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            dimensions (int, optional): Number of dimensions to plot (2 or 3). Defaults to 2.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.
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
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
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

        of = f"pca_plot.{self.plot_format}"
        of = self.output_dir / of

        plt.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()

    def plot_dapc(self, dapc, alignment, popmap, dimensions=2):
        """Plot a DAPC scatter plot.

        Args:
            dapc (sklearn.discriminant_analysis.LinearDiscriminantAnalysis):  The fitted DAPC object used for dimensionality reduction and transformation.

            alignment (numpy.ndarray): The genotype data in the form of a numpy array.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            dimensions (int, optional): Number of dimensions to plot (2 or 3). Defaults to 2.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.
        """
        dapc_transformed = pd.DataFrame(
            dapc.transform(alignment),
            columns=[f"DA{i+1}" for i in range(dapc.n_components_)],
        )
        dapc_transformed["PopulationID"] = popmap["PopulationID"]

        if dimensions == 2:
            sns.scatterplot(data=dapc_transformed, x="DA1", y="DA2", hue="PopulationID")
        elif dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
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

        of = f"dapc_plot.{self.plot_format}"
        of = self.output_dir / of
        plt.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()
        plt.close()

    def _plot_dapc_cv(self, df, popmap, n_components_range):
        """Plot the DAPC cross-validation results.

        Args:
            df (Union[numpy.ndarray, pandas.DataFrame): The input DataFrame or array with the genotypes.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            n_components_range (range): The range of principal components to use for cross-validation.

        Returns:
            None: A plot is saved to a .png file.

        """
        msg = "The _plot_dapc_cv method is not yet implemented."
        self.logger.error(msg)
        raise NotImplementedError(msg)

        components = []
        scores = []

        for n in range(2, n_components_range):
            lda = LinearDiscriminantAnalysis(n_components=n)
            score = cross_val_score(lda, df, popmap["PopulationID"].values, cv=5).mean()
            components.append(n)
            scores.append(score)

        of = f"dapc_cv_results.{self.plot_format}"
        of = self.output_dir / of

        plt.figure(figsize=(16, 9))
        sns.lineplot(x=components, y=scores, marker="o")
        plt.xlabel("Number of Components")
        plt.ylabel("Mean Cross-validation Score")
        plt.title("DAPC Cross-Validation Scores")
        plt.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()
        plt.close()

        best_idx = pd.Series(scores).idxmin()
        best_score = scores[best_idx]
        best_component = components[best_idx]

        print(f"\n\nOptimal DAPC Components: {best_component}")
        print(f"Best DAPC CV Score: {best_score}")

        return best_component

    def plot_sfs(self, pop_gen_stats, population1, population2, savefig=True):
        """Plot a heatmap for the 2D SFS between two given populations and
        bar plots for the 1D SFS of each population.

        Note:
            This method is not yet implemented.

        Args:
            pop_gen_stats (PopGenStatistics): An instance of the PopGenStatistics class.

            population1 (str): The name of the first population.

            population2 (str): The name of the second population.

            savefig (bool, optional): Whether to save the figure to a file. Defaults to True. If True, the figure will be saved to a file.
        """
        msg = "The plot_sfs method is not yet implemented."
        self.logger.error(msg)
        raise NotImplementedError(msg)

        sfs1 = pop_gen_stats.calculate_1d_sfs(population1)
        sfs2 = pop_gen_stats.calculate_1d_sfs(population2)
        sfs2d = pop_gen_stats.calculate_2d_sfs(population1, population2)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        sns.barplot(x=np.arange(1, len(sfs1) + 1), y=sfs1, ax=axs[0])
        axs[0].plot(np.arange(1, len(sfs1) + 1), sfs1, "k-")
        axs[0].set_title(f"1D SFS for {population1}")
        axs[0].xaxis.set_ticks(np.arange(0, len(sfs1) + 1, 5))
        axs[0].set_xticklabels(axs[0].get_xticks(), rotation=45)

        sns.barplot(x=np.arange(1, len(sfs2) + 1), y=sfs2, ax=axs[1])
        axs[1].plot(np.arange(1, len(sfs2) + 1), sfs2, "k-")
        axs[1].set_title(f"1D SFS for {population2}")
        axs[1].xaxis.set_ticks(np.arange(0, len(sfs2) + 1, 5))
        axs[1].set_xticklabels(axs[1].get_xticks(), rotation=45)

        colors = ["white", "green", "yellow", "orange", "red"]
        n_colors = len(colors)
        cmap = mpl_colors.LinearSegmentedColormap.from_list(
            "my_colormap", colors, N=n_colors * 10
        )

        sns.heatmap(sfs2d, cmap=cmap, ax=axs[2])
        axs[2].set_title(f"2D SFS for {population1} and {population2}")

        if savefig:
            plt.savefig(f"sfs_{population1}_{population2}.png")

        if self.show:
            plt.show()

    def plot_joint_sfs_grid(self, pop_gen_stats, populations, savefig=True):
        """Plot the joint SFS between all possible pairs of populations in the popmap file in a grid layout.

        Note:
            This method is not yet implemented.

        Args:
            pop_gen_stats (PopGenStatistics): An instance of the PopGenStatistics class.

            populations (list): A list of population names.

            savefig (bool, optional): Whether to save the figure to a file. Defaults to True. If True, the figure will be saved to a file.

        """
        msg = "The plot_joint_sfs_grid method is not yet implemented."
        self.logger.error(msg)
        raise NotImplementedError(msg)

        n_populations = len(populations)
        n_cols = math.ceil(math.sqrt(n_populations))
        n_rows = math.ceil(n_populations / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        for i, (pop1, pop2) in enumerate(
            itertools.combinations_with_replacement(populations, 2)
        ):
            row, col = divmod(i, n_cols)
            sfs2d = pop_gen_stats.calculate_2d_sfs(pop1, pop2)
            sns.heatmap(sfs2d, cmap="coolwarm", ax=axs[row, col], cbar=False)
            axs[row, col].set_title(f"Joint SFS for {pop1} and {pop2}")

        # Remove unused axes
        for j in range(i + 1, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            fig.delaxes(axs[row, col])

        fig.tight_layout()

        if savefig:
            plt.savefig("joint_sfs_grid.png")

        if self.show:
            plt.show()

    def plot_sankey_filtering_report(self, df, search_mode=False):
        """Plot a Sankey diagram for the filtering report.

        Args:
            df (pd.DataFrame): The input DataFrame containing the filtering report.
            search_mode (bool, optional): Whether the Sankey diagram is being plotted in search mode. Defaults to False.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input DataFrame is empty.
        """
        hv.extension("bokeh")

        plot_dir = self.output_dir / "sankey_plots"
        plot_dir.mkdir(exist_ok=True, parents=True)

        df = df.copy()
        df = df[df["Filter_Method"] != "filter_missing_sample"]

        if df.empty:
            msg = "No data to plot. Please check the filtering thresholds."
            self.logger.error(msg)
            raise ValueError(msg)

        # Ensure correct data types
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

        df["Threshold"] = (
            df["Missing_Threshold"].astype(str)
            + "_"
            + df["MAF_Threshold"].astype(str)
            + "_"
            + df["Bool_Threshold"].astype(str)
            + "_"
            + df["MAC_Threshold"].astype(str)
        )

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
        dftmp = df[df["Filter_Method"] != "filter_missing_sample"]

        self.logger.debug(f"Filtering report for thresholds: {thresholds}")

        # Assign colors
        dftmp["LinkColor_Kept"] = "#2ca02c"  # Green for kept loci
        dftmp["LinkColor_Removed"] = "#d62728"  # Red for removed loci

        dftmp = dftmp.sort_values(by="Step").reset_index(drop=True)

        # Build the flows with a common "Removed" node and edge labels
        flows = []

        for i in dftmp.index:

            source = "Unfiltered" if i == 0 else dftmp.loc[i - 1, "Filter_Method"]

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
            # Create the Sankey plot with edge labels
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
            of = f"filtering_results_sankey_thresholds{thresholds}.html"
            fname = plot_dir / of
            hv.save(sankey_plot, fname, fmt="html")

        except ValueError as e:
            self.logger.warning(
                f"Failed to generate Sankey plot with thresholds: {thresholds}: error: {e}"
            )

    def plot_gt_distribution(self, df, annotation_size=15):
        """Plot the distribution of genotype counts.

        Args:
            df (pd.DataFrame): The input dataframe containing the genotype counts.

            annotation_size (int, optional): The font size for count annotations. Defaults to 15.

        Returns:
            None: A plot is saved to a file.

        Raise
        """
        df = misc.validate_input_type(df, return_type="df")
        df_melt = pd.melt(df, value_name="Count")
        cnts = df_melt["Count"].value_counts()
        cnts.index.names = ["Genotype Int"]
        cnts = pd.DataFrame(cnts).reset_index()
        cnts.sort_values(by="Genotype Int", inplace=True)
        cnts["Genotype Int"] = cnts["Genotype Int"].astype(str)

        int_iupac_dict = misc.get_int_iupac_dict()
        int_iupac_dict = {str(v): k for k, v in int_iupac_dict.items()}
        cnts["Genotype"] = cnts["Genotype Int"].map(int_iupac_dict)
        cnts.columns = [col[0].upper() + col[1:] for col in cnts.columns]

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        g = sns.barplot(x="Genotype", y="Count", data=cnts, ax=ax, color="orange")
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

        of = f"genotype_distribution.{self.plot_format}"
        of = self.output_dir / of

        fig.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()
        plt.close()

    def plot_search_results(self, df_combined):
        """Plot and save the filtering results based on the available data.

        Args:
            df_combined (pd.DataFrame): The input dataframe containing the filtering results.

        Returns:
            None: Plots are saved to files.

        Raises:
            ValueError: Raised if the input dataframe is empty.
        """
        if df_combined.empty:
            msg = "No data to plot. Please check the filtering thresholds."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.verbose:
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

        if self.verbose:
            self.logger.info(
                f"Plotting complete. Plots saved to directory {self.output_dir}."
            )

    def _plot_combined(self, df):
        """Plot missing data proportions for Sample and Global data.

        Args:
            df (pd.DataFrame): The input dataframe containing the missing data proportions.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.
        """
        df = df[
            df["Filter_Method"].isin(["filter_missing", "filter_missing_sample"])
        ].copy()

        if not df.empty:
            if self.verbose:
                self.logger.info("Plotting global per-locus filtering results.")
            self.logger.debug(f"Missing data: {df}")

            fig, axs = plt.subplots(1, 2, figsize=(10, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax = sns.lineplot(
                    x="Missing_Threshold",
                    y=ycol,
                    hue="Filter_Method",
                    palette="Dark2",
                    markers=False,
                    data=df,
                    ax=ax,
                )

                ylab = ycol.split("_")[0].capitalize()

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

            of = f"filtering_results_missing_loci_samples.{self.plot_format}"
            of = self.output_dir / of
            fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")

            if self.show:
                plt.show()
            plt.close()

        else:
            if self.verbose:
                self.logger.info("Missing data filtering results ares empty.")

    def _plot_pops(self, df):
        """Plot population-level missing data proportions."""
        df = df[df["Filter_Method"] == "filter_missing_pop"].copy()

        self.logger.debug(f"Population-level missing data: {df}")

        if not df.empty:
            if self.verbose:
                self.logger.info("Plotting population-level missing data.")
            self.logger.debug(f"Population-level missing data: {df}")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax = sns.lineplot(
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

                ylab = ycol.split("_")[0].capitalize()

                ax.set_xlabel("Filtering Threshold")
                ax.set_ylabel(f"{ylab} Proportion")
                ax.set_title(f"{ylab} Data")
                ax.set_ylim(0, 1.12)
                ax.set_xticks(
                    df["Missing_Threshold"].astype(float).unique(), minor=False
                )

            of = f"filtering_results_missing_population.{self.plot_format}"
            of = self.output_dir / of
            fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")

            if self.show:
                plt.show()
            plt.close()

        else:
            if self.verbose:
                self.logger.info("Population-level missing data is empty.")

    def _plot_maf(self, df):
        """Plot MAF filtering data.

        Args:
            df (pd.DataFrame): The input dataframe containing the MAF data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.
        """
        df_mac = df[df["Filter_Method"] == "filter_mac"].copy()
        df = df[df["Filter_Method"] == "filter_maf"].copy()

        self.logger.debug(f"MAF data: {df}")
        self.logger.debug(f"MAC data: {df_mac}")

        if not df.empty:
            if self.verbose:
                self.logger.info("Plotting minor allele frequency data.")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax = sns.lineplot(
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

                ylab = ycol.split("_")[0].capitalize()

                ax.set_xlabel("Filtering Threshold")
                ax.set_ylabel(f"{ylab} Proportion")
                ax.set_title(f"{ylab} Data")
                ax.set_ylim(-0.05, 1.12)
                ax.set_xticks(df["MAF_Threshold"].astype(float).unique(), minor=False)

            of = self.output_dir / f"filtering_results_maf.{self.plot_format}"
            fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            if self.show:
                plt.show()
            plt.close()

        else:
            if self.verbose:
                self.logger.info("MAF data is empty.")

        if not df_mac.empty:
            if self.verbose:
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

            of = self.output_dir / f"filtering_results_mac.{self.plot_format}"
            fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            if self.show:
                plt.show()
            plt.close()
        else:
            if self.verbose:
                self.logger.info("MAC data is empty.")

    def _plot_boolean(self, df):
        """Plot boolean datasets like Monomorphic, Biallelic, Thin Loci, Singleton, Linked.

        Args:
            df (pd.DataFrame): The input dataframe containing the boolean data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.
        """
        df = df[df["Filter_Method"].isin(self.boolean_filter_methods)].copy()

        if not df.empty:
            if self.verbose:
                self.logger.info("Plotting boolean filtering data.")

            fig, axs = plt.subplots(1, 2, figsize=(8, 6))

            for ax, ycol in zip(axs, ["Removed_Prop", "Kept_Prop"]):
                ax = sns.lineplot(
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

                ylab = ycol.split("_")[0].capitalize()

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

            of = self.output_dir / f"filtering_results_bool.{self.plot_format}"
            fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")

            if self.show:
                plt.show()
            plt.close()

        else:
            if self.verbose:
                self.logger.info("Boolean data is empty.")

    def plot_filter_report(self, df):
        """
        Plot the filter report.

        Args:
            df (pd.DataFrame): The dataframe containing the filter report data.

        Returns:
            None: A plot is saved to a file.

        Raises:
            ValueError: Raised if the input dataframe is empty.
        """
        if self.verbose:
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

            ax = sns_method(**kwargs, ax=ax)

        plot_format = plot_format.lower()
        of = self.output_dir / f"filter_report.{self.plot_format}"
        of.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")

        if self.show:
            plt.show()
        plt.close()

    def plot_pop_counts(self, populations):
        """
        Plot the population counts.

        Args:
            populations (pd.Series): The series containing population data.

        Returns:
            None: A plot is saved to a file.
        """
        # Create the countplot
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))

        # Calculate the counts and proportions
        counts = pd.value_counts(populations)
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
            ax = sns.barplot(x=data.index, y=data.values, color=color, ax=ax)
            median_line = ax.axhline(median, color=median_color, linestyle="--")

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(labels=ax.get_xticklabels(), minor=False, rotation=90)
            ax.set_title("Population Counts")
            ax.set_xlabel("Population ID")
            ax.set_ylabel(ylabel)
            ax.legend([median_line], ["Median"], loc="upper right")

        of = self.output_dir / f"population_counts.{self.plot_format}"
        fig.savefig(of, dpi=self.dpi, bbox_inches="tight", facecolor="white")

        if self.show:
            plt.show()
        plt.close()

    def plot_performance(self, resource_data, color="#8C56E3", figsize=(18, 10)):
        """Plots the performance metrics: CPU Load, Memory Footprint, and Execution Time using boxplots.

        This function takes a dictionary of performance data and plots the metrics for each method using boxplots to show variability. The resulting plots are saved in a file of the specified format.

        Args:
            resource_data (dict): Dictionary with performance data.
                                Keys are method names, and values are lists of dictionaries with
                                keys 'cpu_load', 'memory_footprint', and 'execution_time'.
            color (str, optional): Color to be used in the plot. Should be a valid color string.
                                Defaults to "#8C56E3".
            figsize (tuple, optional): Size of the figure. Should be a tuple of 2 integers. Defaults to (18, 10).

        Returns:
            None. The function saves the plot to a file.
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
                x.get_text().replace("_", " ").title()
                for x in ax.get_xticklabels()
            ]
            ticklabs = [x.replace("Filter Maf", "Filter MAF") for x in ticklabs]
            ticklabs = [x.replace("Filter Mac", "Filter MAC") for x in ticklabs]
            ax.set_xticklabels(ticklabs, rotation=90)
            ax.set_ylim(bottom=-0.05)

        # Save the plot to a file
        of = plot_dir / f"benchmarking_barplot.{self.plot_format}"
        fig.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()
        plt.close()

    def run_pca(
        self,
        n_components=None,
        center=True,
        scale=False,
        n_axes=2,
        point_size=15,
        bottom_margin=0,
        top_margin=0,
        left_margin=0,
        right_margin=0,
        width=1088,
        height=700,
    ):
        """Runs PCA and makes scatterplot with colors showing missingness.

        Genotypes are plotted as separate shapes per population and colored according to missingness per individual.

        This function is run at the end of each imputation method, but can be run independently to change plot and PCA parameters such as ``n_axes=3`` or ``scale=True``\. Setting ``n_axes=3`` will make a 3D PCA plot.

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

        Returns:
            numpy.ndarray: PCA data as a numpy array with shape (n_samples, n_components).

            sklearn.decomposision.PCA: Scikit-learn PCA object from sklearn.decomposision.PCA. Any of the sklearn.decomposition.PCA attributes can be accessed from this object. See sklearn documentation.

        Raises:
            ValueError: If n_axes is not set to 2 or 3.
            ValueError: If n_axes is set to <2.

        Examples:
            >>> data = GenotypeData(
            >>>     filename="snps.str",
            >>>     filetype="auto",
            >>>     popmapfile="popmap.txt",
            >>> )
            >>>
            >>> components, pca = run_pca(
            >>>     data,
            >>>     scale=True,
            >>>     center=True,
            >>>     plot_format="png"
            >>> )
            >>>
            >>> # Calculate and print explained variance ratio
            >>> explvar = pca.explained_variance_ratio_
            >>> print(explvar)

        """
        plot_dir = f"{self.prefix}_output"
        plot_dir = Path(plot_dir, "gtdata", "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        if n_axes > 3:
            msg = ">3 axes is not supported; n_axes must be either 2 or 3."
            self.logger.error(msg)
            raise ValueError(msg)
        if n_axes < 2:
            msg = "<2 axes is not supported; n_axes must be either 2 or 3."
            self.logger.error(msg)
            raise ValueError(msg)

        df = misc.validate_input_type(
            self.genotype_data.genotypes_012(fmt="pandas"), return_type="df"
        )

        df.replace(-9, np.nan, inplace=True)

        if center or scale:
            # Center data to mean. Scaling to unit variance is off.
            scaler = StandardScaler(with_mean=center, with_std=scale)
            pca_df = scaler.fit_transform(df)
        else:
            pca_df = df.copy()

        # Run PCA.
        model = PCA(n_components=n_components)

        # PCA can't handle missing data. So impute it here using the K
        # nearest neighbors (samples).
        imputer = KNNImputer(weights="distance")
        pca_df = imputer.fit_transform(pca_df)
        components = model.fit_transform(pca_df)

        df_pca = pd.DataFrame(
            components[:, [0, 1, 2]], columns=["Axis1", "Axis2", "Axis3"]
        )

        df_pca["SampleID"] = self.genotype_data.samples
        df_pca["Population"] = self.genotype_data.populations
        df_pca["Size"] = point_size

        _, ind, _, _, _ = self.genotype_data.calc_missing(df, use_pops=False)
        df_pca["missPerc"] = ind

        my_scale = [("rgb(19, 43, 67)"), ("rgb(86,177,247)")]  # ggplot default

        z = "Axis3" if n_axes == 3 else None
        labs = {
            "Axis1": f"PC1 ({round(model.explained_variance_ratio_[0] * 100, 2)}%)",
            "Axis2": f"PC2 ({round(model.explained_variance_ratio_[1] * 100, 2)}%)",
            "missPerc": "Missing Prop.",
            "Population": "Population",
        }

        if z is not None:
            labs["Axis3"] = (
                f"PC3 ({round(model.explained_variance_ratio_[2] * 100, 2)}%)"
            )
            fig = px.scatter_3d(
                df_pca,
                x="Axis1",
                y="Axis2",
                z="Axis3",
                color="missPerc",
                symbol="Population",
                color_continuous_scale=my_scale,
                custom_data=["Axis3", "SampleID", "Population", "missPerc"],
                size="Size",
                size_max=point_size,
                labels=labs,
            )
        else:
            fig = px.scatter(
                df_pca,
                x="Axis1",
                y="Axis2",
                color="missPerc",
                symbol="Population",
                color_continuous_scale=my_scale,
                custom_data=["Axis3", "SampleID", "Population", "missPerc"],
                size="Size",
                size_max=point_size,
                labels=labs,
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
                t=top_margin,
                l=left_margin,
                r=right_margin,
            ),
            width=width,
            height=height,
            legend_orientation="h",
            legend_title="Population",
            legend_title_side="top",
        )

        of = plot_dir / f"pca.{self.plot_format}"
        fig.write_html(of.with_suffix(".html"))
        fig.write_image(of, format=self.plot_format)
        return components, model

    def visualize_missingness(
        self,
        df,
        prefix=None,
        zoom=True,
        horizontal_space=0.6,
        vertical_space=0.6,
        bar_color="gray",
        heatmap_palette="magma",
    ):
        """Make multiple plots to visualize missing data.

        Args:
            df (pandas.DataFrame): DataFrame with snps to visualize.

            prefix (str, optional): Prefix to use for the output files. If None, the prefix is set to the input filename. Defaults to None.

            zoom (bool, optional): If True, zooms in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.


            horizontal_space (float, optional): Set width spacing between subplots. If your plot are overlapping horizontally, increase horizontal_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            vertical_space (float, optioanl): Set height spacing between subplots. If your plots are overlapping vertically, increase vertical_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            bar_color (str, optional): Color of the bars on the non-stacked barplots. Can be any color supported by matplotlib. See matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, optional): Palette to use for heatmap plot. Can be any palette supported by seaborn. See seaborn documentation. Defaults to 'magma'.

        Returns:
            pandas.DataFrame: Per-locus missing data proportions.
            pandas.DataFrame: Per-individual missing data proportions.
            pandas.DataFrame: Per-population + per-locus missing data proportions.

            pandas.DataFrame: Per-population missing data proportions.
            pandas.DataFrame: Per-individual and per-population missing data proportions.
        """
        prefix = prefix if prefix is not None else self.prefix

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
        melt_df.sort_values(by=id_vars[::-1], inplace=True)
        melt_df["Missing"].replace(False, "Present", inplace=True)
        melt_df["Missing"].replace(True, "Missing", inplace=True)

        ax = axes[0, 2] if poptotal is None else axes[1, 1]

        ax.set_title("Per-Individual")
        g = sns.histplot(
            data=melt_df, y="variable", hue="Missing", multiple="fill", ax=ax
        )
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        g.get_legend().set_title(None)

        if poptotal is not None:
            ax = axes[1, 2]

            ax.set_title("Per-Population")
            g = sns.histplot(
                data=melt_df, y="Population", hue="Missing", multiple="fill", ax=ax
            )
            g.get_legend().set_title(None)

        of = plot_dir / f"{prefix}_missingness_report.{self.plot_format}"
        fig.savefig(of, bbox_inches="tight", facecolor="white", dpi=self.dpi)

        if self.show:
            plt.show()
        plt.close()

        return loc, ind, poploc, poptotal, indpop
