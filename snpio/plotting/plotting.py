import sys
import os
from pathlib import Path
from functools import reduce
import seaborn as sns
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import itertools
import warnings

from typing import Tuple

import holoviews as hv
import panel as pn
import plotly.express as px


hv.extension("bokeh")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from snpio.utils import misc


class Plotting:
    """Class with various static methods for plotting."""

    def __init__(self, popgenio):
        """Class Constructor.

        Args:
            genotype_data (GenotypeData): Initialized GentoypeData object.
        """

        self.alignment = popgenio.alignment
        self.popmap = popgenio.populations
        self.populations = popgenio.populations

    @staticmethod
    def _plot_summary_statistics_per_sample(summary_stats, ax=None):
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

    @staticmethod
    def _plot_summary_statistics_per_population(
        summary_stats, popmap, ax=None
    ):
        """Plot summary statistics per population.

        Args:
            summary_stats (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

            popmap (pd.DataFrame): The DataFrame containing the population mapping used to group the summary statistics.

            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.

        """
        if ax is None:
            _, ax = plt.subplots()

        # Group the summary statistics by population.
        pop_summary_stats = summary_stats.groupby(
            popmap["PopulationID"]
        ).mean()

        ax.plot(pop_summary_stats["Ho"], label="Ho")
        ax.plot(pop_summary_stats["He"], label="He")
        ax.plot(pop_summary_stats["Pi"], label="Pi")
        ax.plot(pop_summary_stats["Fst"], label="Fst")

        ax.set_xlabel("Population")
        ax.set_ylabel("Value")
        ax.set_title("Summary Statistics per Population")
        ax.legend()

    @staticmethod
    def _plot_summary_statistics_per_population_grid(
        summary_statistics_df, show=False
    ):
        """Plot summary statistics per population using a Seaborn PairGrid plot.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

            show (bool, optional): Whether to display the plot. Defaults to False. If True, the plot will be displayed.

        """
        g = sns.PairGrid(summary_statistics_df)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=3, legend=False)

        g.savefig(
            "summary_statistics_per_population_grid.png", bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close()

    @staticmethod
    def _plot_summary_statistics_per_sample_grid(
        summary_statistics_df, show=False
    ):
        """Plot summary statistics per sample using a Seaborn PairGrid plot.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

            show (bool, optional): Whether to display the plot. Defaults to False. If True, the plot will be displayed.

        """
        g = sns.PairGrid(summary_statistics_df)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=3, legend=False)

        g.savefig(
            "summary_statistics_per_sample_grid.png", bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close()

    @classmethod
    def plot_summary_statistics(cls, summary_statistics_df, show=False):
        """Plot summary statistics per sample and per population on the same figure.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

            show (bool, optional): Whether to display the plot. Defaults to False. If True, the plot will be displayed.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

        cls._plot_summary_statistics_per_sample(
            summary_statistics_df, ax=axes[0]
        )
        cls._plot_summary_statistics_per_population(
            summary_statistics_df, ax=axes[1]
        )

        plt.tight_layout()
        plt.savefig("summary_statistics.png")

        if show:
            plt.show()

        plt.close()

        cls._plot_summary_statistics_per_sample_grid(summary_statistics_df)
        cls._plot_summary_statistics_per_population_grid(summary_statistics_df)

    @staticmethod
    def plot_pca(pca, alignment, popmap, dimensions=2, show=False):
        """Plot a PCA scatter plot.

        Args:
            pca (sklearn.decomposition.PCA): The fitted PCA object.
                The fitted PCA object used for dimensionality reduction and transformation.

            alignment (numpy.ndarray): The genotype data used for PCA.
                The genotype data in the form of a numpy array.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            dimensions (int, optional): Number of dimensions to plot (2 or 3). Defaults to 2.

            show (bool, optional): Whether to display the plot. Defaults to False. If True, the plot will be displayed.

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
            sns.scatterplot(
                data=pca_transformed, x="PC1", y="PC2", hue="PopulationID"
            )
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

        plt.savefig("pca_plot.png")

        if show:
            plt.show()

    @staticmethod
    def plot_dapc(dapc, alignment, popmap, dimensions=2, show=False):
        """Plot the DAPC scatter plot.

        Args:
            dapc (sklearn.discriminant_analysis.LinearDiscriminantAnalysis):  The fitted DAPC object used for dimensionality reduction and transformation.

            alignment (numpy.ndarray): The genotype data in the form of a numpy array.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            dimensions (int, optional): Number of dimensions to plot (2 or 3). Defaults to 2.

            show (bool, optional): Whether to display the plot. Defaults to False. If True, the plot will be displayed.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.
        """
        dapc_transformed = pd.DataFrame(
            dapc.transform(alignment),
            columns=[f"DA{i+1}" for i in range(dapc.n_components_)],
        )
        dapc_transformed["PopulationID"] = popmap["PopulationID"]

        if dimensions == 2:
            sns.scatterplot(
                data=dapc_transformed, x="DA1", y="DA2", hue="PopulationID"
            )
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

        plt.savefig("dapc_plot.png")

        if show:
            plt.show()

    @staticmethod
    def _plot_dapc_cv(
        df,
        popmap,
        n_components_range,
        prefix=None,
        plot_dir="plots",
    ):
        """Plot the DAPC cross-validation results.

        Args:
            df (Union[numpy.ndarray, pandas.DataFrame): The input DataFrame or array with the genotypes.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            n_components_range (range): The range of principal components to use for cross-validation.

            prefix (str): Prefix to prepend to output filename.

            plot_dir (str): Directory to save plot to.

        Returns:
            None: A plot is saved to a .png file.

        """
        components = []
        scores = []

        for n in range(2, n_components_range):
            lda = LinearDiscriminantAnalysis(n_components=n)
            score = cross_val_score(
                lda, df, popmap["PopulationID"].values, cv=5
            ).mean()
            components.append(n)
            scores.append(score)

        fname = (
            "dapc_cv_results.png"
            if prefix is None
            else f"{prefix}_dapc_cv_results.png"
        )

        plt.figure(figsize=(16, 9))
        sns.lineplot(x=components, y=scores, marker="o")
        plt.xlabel("Number of Components")
        plt.ylabel("Mean Cross-validation Score")
        plt.title("DAPC Cross-Validation Scores")
        plt.savefig(os.path.join(plot_dir, fname), bbox_inches="tight")
        plt.close()

        best_idx = pd.Series(scores).idxmin()
        best_score = scores[best_idx]
        best_component = components[best_idx]

        print(f"\n\nOptimal DAPC Components: {best_component}")
        print(f"Best DAPC CV Score: {best_score}")

        return best_component

    @staticmethod
    def plot_sfs(
        pop_gen_stats,
        population1,
        population2,
        savefig=True,
        show=False,
    ):
        """Plot a heatmap for the 2D SFS between two given populations and
        bar plots for the 1D SFS of each population.

        Args:
            pop_gen_stats (PopGenStatistics): An instance of the PopGenStatistics class.

            population1 (str): The name of the first population.

            population2 (str): The name of the second population.

            savefig (bool, optional): Whether to save the figure to a file. Defaults to True. If True, the figure will be saved to a file.

            show (bool, optional): Whether to show the figure inline. Defaults to True. If True, the figure will be displayed inline.

        """

        sfs1 = pop_gen_stats.calculate_1d_sfs(population1)
        sfs2 = pop_gen_stats.calculate_1d_sfs(population2)
        sfs2d = pop_gen_stats.calculate_2d_sfs(population1, population2)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        sns.barplot(x=np.arange(1, len(sfs1) + 1), y=sfs1, ax=axs[0])
        axs[0].plot(np.arange(1, len(sfs1) + 1), sfs1, "k-")
        axs[0].set_title(f"1D SFS for {population1}")
        axs[0].xaxis.set_ticks(
            np.arange(0, len(sfs1) + 1, 5)
        )  # Set a step for displaying the tick labels
        axs[0].set_xticklabels(
            axs[0].get_xticks(), rotation=45
        )  # Rotate the tick labels

        sns.barplot(x=np.arange(1, len(sfs2) + 1), y=sfs2, ax=axs[1])
        axs[1].plot(np.arange(1, len(sfs2) + 1), sfs2, "k-")
        axs[1].set_title(f"1D SFS for {population2}")
        axs[1].xaxis.set_ticks(
            np.arange(0, len(sfs2) + 1, 5)
        )  # Set a step for displaying the tick labels
        axs[1].set_xticklabels(
            axs[1].get_xticks(), rotation=45
        )  # Rotate the tick labels

        colors = ["white", "green", "yellow", "orange", "red"]
        n_colors = len(colors)
        cmap = mpl_colors.LinearSegmentedColormap.from_list(
            "my_colormap", colors, N=n_colors * 10
        )

        sns.heatmap(sfs2d, cmap=cmap, ax=axs[2])
        axs[2].set_title(f"2D SFS for {population1} and {population2}")

        if savefig:
            plt.savefig(f"sfs_{population1}_{population2}.png")

        if show:
            plt.show()

    @staticmethod
    def plot_joint_sfs_grid(
        pop_gen_stats, populations, savefig=True, show=True
    ):
        """Plot the joint SFS between all possible pairs of populations in the popmap file in a grid layout.

        Args:
            pop_gen_stats (PopGenStatistics): An instance of the PopGenStatistics class.

            populations (list): A list of population names.

            savefig (bool, optional): Whether to save the figure to a file. Defaults to True. If True, the figure will be saved to a file.

            show (bool, optional): Whether to show the figure inline. Defaults to True.

        """
        n_populations = len(populations)
        n_cols = math.ceil(math.sqrt(n_populations))
        n_rows = math.ceil(n_populations / n_cols)

        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows)
        )

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

        if show:
            plt.show()

    @staticmethod
    def _plotly_sankey(nodes, links, outfile):
        """Generate a Sankey diagram using Plotly.

        Args:
            nodes (list): A list of dictionaries representing the nodes in the diagram.
                Each dictionary should contain the following keys:
                - 'pad' (int): Padding around the node.
                - 'thickness' (int): Thickness of the node.
                - 'line' (dict): Dictionary specifying the line properties of the node.
                It should contain the following keys:
                - 'color' (str): Color of the node's outline.
                - 'width' (float): Width of the node's outline.
                - 'label' (str): Label for the node.

            links (list): A list of dictionaries representing the links between nodes.
                Each dictionary should contain the following keys:
                - 'source' (int): Index of the source node.
                - 'target' (int): Index of the target node.
                - 'value' (float): Value or flow of the link.
                - 'color' (str, optional): Color of the link. If not provided, a default color will be used.

            outfile (str): The path to save the generated image file.

        """
        # Prepare the data for the Sankey diagram
        link_colors = [
            "rgba(31, 119, 180, 0.8)",
            "rgba(255, 127, 14, 0.8)",
            "rgba(44, 160, 44, 0.8)",
            "rgba(214, 39, 40, 0.8)",
            "rgba(148, 103, 189, 0.8)",
            "rgba(140, 86, 75, 0.8)",
        ]

        for i, color in enumerate(link_colors):
            if i < len(links):
                links[i]["color"] = color

        # Create the Sankey diagram
        fig = go.Figure(
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[n["label"] for n in nodes],
                ),
                link=dict(
                    source=[l["source"] for l in links],
                    target=[l["target"] for l in links],
                    value=[l["value"] for l in links],
                    color=[
                        l.get("color", "rgba(0, 0, 0, 0.8)") for l in links
                    ],
                ),
            )
        )

        # Set the dimensions of the figure
        fig.update_layout(
            width=1600,  # Adjust the width
            height=1200,  # Adjust the height
            font=dict(size=24),
            title=dict(text="Filtering Report", x=0.5, y=0.95),
        )

        # Save the image to a file
        fig.write_image(outfile, scale=1.3, engine="kaleido")

    @classmethod
    def _calc_node_positions(
        cls,
        nodes: list,
        node_groups: list,
        level_arrangement: list,
        final_group_extra_gap: float = 0.05,
        y_sep: float = 0.3,
    ) -> Tuple[list, list]:
        """Calculate the positions of the nodes in the Sankey diagram.

        Args:
            nodes (list): A list of all nodes in the diagram.

            node_groups (list): A list of lists representing the groups of nodes.

            level_arrangement (list): A list of lists representing the arrangement of nodes within each level.

            final_group_extra_gap (float, optional): The extra gap for the last group. Defaults to 0.05.

            y_sep (float, optional): The separation between different levels. Defaults to 0.3.

        Returns:
            Tuple[list, list]: Two lists representing the x and y positions of the nodes.

        """
        final_group_extra_gap = 0.05
        normal_gap = (1 - final_group_extra_gap) / (len(node_groups) - 1)

        x_pos = [
            round(group_idx * normal_gap, 2)
            if group_idx < len(node_groups) - 1
            else round(group_idx * normal_gap + final_group_extra_gap, 2)
            for group_idx, group in enumerate(node_groups)
            for node_idx, node in enumerate(group)
        ]

        y_pos = []
        for group_idx, group in enumerate(node_groups):
            for node_idx, node in enumerate(group):
                if node in level_arrangement[0]:
                    y_pos.append(1)
                elif node in level_arrangement[1]:
                    y_pos.append(0.5)
                else:
                    y_pos.append(0)

        return x_pos, y_pos

    @staticmethod
    def plot_sankey_filtering_report(
        loci_removed_per_step,
        loci_before,
        loci_after,
        outfile,
        plot_dir="plots",
        included_steps=None,
    ):
        """Plot a Sankey diagram representing the filtering steps and the number of loci removed at each step.

        Args:
            loci_removed_per_step (List[Tuple[str, int]]): A list of tuples representing the filtering steps and the number of loci removed at each step.

            loci_before (int): The number of loci before filtering.

            loci_after (int): The number of loci after filtering.

            outfile (str): The output filename for the plot.

            plot_dir (str, optional): The directory to save the plot. Defaults to "plots".

            included_steps (List[int], optional): The indices of the filtering steps to include in the plot. Defaults to None.
        """

        if loci_before == loci_after:
            warnings.warn(
                "No loci were removed. Please ensure that at least one of the "
                "filtering options is changed from default."
            )

        else:
            included_steps = range(12)

            if included_steps is None:
                included_steps = list(range(6))

            steps = [
                ["Unfiltered", "Monomorphic", loci_removed_per_step[0][1]]
                if 0 in included_steps
                else None,
                [
                    "Unfiltered",
                    "Filter Singletons",
                    loci_before - loci_removed_per_step[0][1],
                ]
                if 1 in included_steps
                else None,
                [
                    "Filter Singletons",
                    "Singletons",
                    loci_removed_per_step[1][1],
                ]
                if 2 in included_steps
                else None,
                [
                    "Filter Singletons",
                    "Filter Non-Biallelic",
                    loci_before
                    - sum([x[1] for x in loci_removed_per_step[0:2]]),
                ]
                if 3 in included_steps
                else None,
                [
                    "Filter Non-Biallelic",
                    "Non-Biallelic",
                    loci_removed_per_step[2][1],
                ]
                if 4 in included_steps
                else None,
                [
                    "Filter Non-Biallelic",
                    "Filter Missing (Global)",
                    loci_before
                    - sum([x[1] for x in loci_removed_per_step[0:3]]),
                ]
                if 5 in included_steps
                else None,
                [
                    "Filter Missing (Global)",
                    "Missing (Global)",
                    loci_removed_per_step[3][1],
                ]
                if 6 in included_steps
                else None,
                [
                    "Filter Missing (Global)",
                    "Filter Missing (Populations)",
                    loci_before
                    - sum([x[1] for x in loci_removed_per_step[0:4]]),
                ]
                if 7 in included_steps
                else None,
                [
                    "Filter Missing (Populations)",
                    "Missing (Populations)",
                    loci_removed_per_step[4][1],
                ]
                if 8 in included_steps
                else None,
                [
                    "Filter Missing (Populations)",
                    "Filter MAF",
                    loci_before
                    - sum([x[1] for x in loci_removed_per_step[0:5]]),
                ]
                if 9 in included_steps
                else None,
                ["Filter MAF", "MAF", loci_removed_per_step[5][1]]
                if 10 in included_steps
                else None,
                ["Filter MAF", "Filtered", loci_after]
                if 11 in included_steps
                else None,
            ]

            steps = [step for step in steps if step is not None]

            l = []
            zeros = []
            node_labels = ["Unfiltered"]
            for step in steps:
                if step[2] > 0:
                    l.append(step)
                else:
                    zeros.append(step[2])

            df = pd.DataFrame(l, columns=["Source", "Target", "Count"])
            # Convert integer labels to strings
            df["Source"] = df["Source"].astype(str)
            df["Target"] = df["Target"].astype(str)

            node_labels = [
                "Unfiltered",
                "Monomorphic",
                "Filter Singletons",
                "Singletons",
                "Filter Non-Biallelic",
                "Non-Biallelic",
                "Filter Missing (Global)",
                "MAF",
                "Filter Missing (Populations)",
                "Missing (Global)",
                "Filter MAF",
                "Missing (Populations)",
                "Filtered",
            ]

            node_labels = [x for x in node_labels if x not in zeros]

            cmap = {
                "Unfiltered": "#66c2a5",
                "Filter Singletons": "#66c2a5",
                "Filter Non-Biallelic": "#66c2a5",
                "Filter Missing (Global)": "#66c2a5",
                "Filter Missing (Populations)": "#66c2a5",
                "Filter MAF": "#66c2a5",
                "Filtered": "#66c2a5",
                "Non-Biallelic": "#fc8d62",
                "Monomorphic": "#fc8d62",
                "Singletons": "#fc8d62",
                "Missing (Global)": "#fc8d62",
                "Missing (Populations)": "#fc8d62",
                "Missing (Sample)": "#fc8d62",
                "MAF": "#fc8d62",
            }

            # Add a new column 'LinkColor' to the dataframe
            df["LinkColor"] = df["Target"].apply(lambda x: cmap.get(x, "red"))

            sankey_plot = hv.Sankey(
                df, label="Sankey Filtering Report"
            ).options(
                node_color="blue",
                cmap=cmap,
                width=1000,
                height=500,
                edge_color="LinkColor",
                node_padding=40,
            )

            # Apply custom node labels
            label_array = np.array(node_labels)
            sankey_plot = sankey_plot.redim.values(Node=label_array)

            # Create custom legend
            legend = """
            <div style="position:absolute;right:20px;top:20px;border:1px solid black;padding:10px;background-color:white">
                <div style="display:flex;align-items:center;">
                    <div style="width:20px;height:20px;background-color:#66c2a5;margin-right:5px;"></div>
                    <div>Loci Remaining</div>
                </div>
                <div style="display:flex;align-items:center;">
                    <div style="width:20px;height:20px;background-color:#fc8d62;margin-right:5px;"></div>
                    <div>Loci Removed</div>
                </div>
            </div>
            """

            # Create the custom legend using hv.Div
            legend_plot = hv.Div(legend)

            # Convert the HoloViews objects to Bokeh models
            bokeh_sankey_plot = hv.render(sankey_plot)
            bokeh_legend_plot = hv.render(legend_plot)

            # Combine the Bokeh plots using Panel
            combined = pn.Row(bokeh_sankey_plot, bokeh_legend_plot)

            outfile_final = os.path.join(plot_dir, outfile)
            Path(plot_dir).mkdir(parents=True, exist_ok=True)

            # Save the plot to an HTML file
            combined.save(outfile_final)

    @staticmethod
    def plot_gt_distribution(
        df, plot_dir="plots", fontsize=28, ticksize=20, annotation_size=15
    ):
        """Plot the distribution of genotype counts.

        Args:
            df (pd.DataFrame): The input dataframe containing the genotype counts.

            plot_dir (str, optional): The directory to save the plot. Defaults to "plots".

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            ticksize (int, optional): The font size for tick labels. Defaults to 20.

            annotation_size (int, optional): The font size for count annotations. Defaults to 15.
        """
        df = misc.validate_input_type(df, return_type="df")
        df_melt = pd.melt(df, value_name="Count")
        cnts = df_melt["Count"].value_counts()
        cnts.index.names = ["Genotype Int"]
        cnts = pd.DataFrame(cnts).reset_index()
        cnts.sort_values(by="Genotype Int", inplace=True)
        cnts["Genotype Int"] = cnts["Genotype Int"].astype(str)

        onehot_dict = {
            "A": 0,
            "T": 1,
            "G": 2,
            "C": 3,
            "W": 4,
            "R": 5,
            "M": 6,
            "K": 7,
            "Y": 8,
            "S": 9,
            "-": -9,
            "N": -9,
        }
        onehot_dict = {str(v): k for k, v in onehot_dict.items()}
        cnts["Genotype"] = cnts["Genotype Int"].map(onehot_dict)
        cnts.columns = [col[0].upper() + col[1:] for col in cnts.columns]

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        g = sns.barplot(
            x="Genotype", y="Count", data=cnts, ax=ax, color="orange"
        )
        g.set_xlabel("Genotype", fontsize=fontsize)
        g.set_ylabel("Count", fontsize=fontsize)
        g.set_title("Genotype Counts", fontsize=fontsize)
        g.tick_params(axis="both", labelsize=ticksize)
        for p in g.patches:
            g.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + 0.075, p.get_height() + 0.01),
                xytext=(0, 1),
                textcoords="offset points",
                va="bottom",
                fontsize=annotation_size,
            )

        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        fig.savefig(
            os.path.join(plot_dir, "genotype_distributions.png"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    @staticmethod
    def make_labs(
        xlab,
        ylab,
        title,
        labelsize=20,
        fontsize=28,
        ymin=0.0,
        ymax=1.0,
        legend=False,
        legend_loc="upper left",
    ):
        """Set the labels and formatting for the plot.

        Args:
            xlab (str): The label for the x-axis.

            ylab (str): The label for the y-axis.

            title (str): The title of the plot.

            labelsize (int, optional): The font size for tick labels. Defaults to 20.

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            ymin (float, optional): The minimum y-axis value. Defaults to 0.0.

            ymax (float, optional): The maximum y-axis value. Defaults to 1.0.

            legend (bool, optional): Whether to display the legend. Defaults to False.

            legend_loc (str, optional): The location of the legend. Defaults to "upper left".
        """
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)
        plt.ylim(ymin, ymax)
        plt.title(title, fontsize=fontsize)
        plt.tick_params(axis="both", labelsize=labelsize)

        if legend:
            plt.legend(fontsize=fontsize, loc=legend_loc)

    @staticmethod
    def lineplot_maf(df, fontsize=28, labelsize=20, ymin=0.0, ymax=1.0):
        """
        Create a line plot to visualize the relationship between minimum MAF threshold and missing data proportion.

        Args:
            df (pd.DataFrame): The input dataframe containing the data.

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            labelsize (int, optional): The font size for tick labels. Defaults to 20.

            ymin (float, optional): The minimum y-axis value. Defaults to 0.0.

            ymax (float, optional): The maximum y-axis value. Defaults to 1.0.
        """
        sns.lineplot(
            x="Threshold",
            y="Proportion",
            hue="Type",
            data=df,
        )

        Plotting.make_labs(
            "Minimum MAF Threshold",
            "Proportion of Missing Data",
            "MAF vs. Missing Data Proportion",
            fontsize=fontsize,
            labelsize=labelsize,
            ymin=ymin,
            ymax=ymax,
        )

    @staticmethod
    def histogram_maf(maf, fontsize=28, labelsize=20, ymin=0.0, ymax=1.0):
        """
        Create a histogram to visualize the distribution of minor allele frequency (MAF).

        Args:
            maf (pd.Series or np.array): The input MAF data.

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            labelsize (int, optional): The font size for tick labels. Defaults to 20.

            ymin (float, optional): The minimum y-axis value. Defaults to 0.0.

            ymax (float, optional): The maximum y-axis value. Defaults to 1.0.
        """
        sns.histplot(maf, kde=False, bins=30)
        Plotting.make_labs(
            "Minor Allele Frequency",
            "Minor Allele Count",
            "Minor Allele Frequency Histogram",
            fontsize=fontsize,
            labelsize=labelsize,
            ymin=0.0,
            ymax=None,
        )

    @staticmethod
    def cdf_maf(
        maf,
        title="Cumulative Distribution of Minor Alleles",
        ylab="Cumulative Distribution",
        fontsize=28,
        labelsize=20,
        ymin=0.0,
        ymax=1.0,
    ):
        """
        Create a cumulative distribution function (CDF) plot to visualize the distribution of minor allele frequency (MAF).

        Args:
            maf (pd.Series or np.array): The input MAF data.

            title (str, optional): The title of the plot. Defaults to "Cumulative Distribution of Minor Alleles".

            ylab (str, optional): The label for the y-axis. Defaults to "Cumulative Distribution".

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            labelsize (int, optional): The font size for tick labels. Defaults to 20.

            ymin (float, optional): The minimum y-axis value. Defaults to 0.0.

            ymax (float, optional): The maximum y-axis value. Defaults to 1.0.
        """
        sns.ecdfplot(maf)
        Plotting.make_labs(
            "Minor Allele Frequency",
            ylab,
            title,
            fontsize=fontsize,
            labelsize=labelsize,
            ymin=ymin,
            ymax=ymax,
        )

    def violinplot(
        x,
        y,
        data,
        xlab,
        ylab,
        title,
        hue=None,
        fontsize=28,
        labelsize=20,
        ymin=0.0,
        ymax=1.0,
        legend=False,
        legend_loc="upper left",
        split=False,
    ):
        """
        Create a violin plot to visualize the distribution of a continuous variable across different categories.

        Args:
            x (str): The column name in the data frame to use as the x-axis variable.

            y (str): The column name in the data frame to use as the y-axis variable.

            data (pd.DataFrame): The input data frame.

            xlab (str): The label for the x-axis.

            ylab (str): The label for the y-axis.

            title (str): The title of the plot.

            hue (str, optional): The column name in the data frame to use for grouping the data.

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            labelsize (int, optional): The font size for tick labels. Defaults to 20.

            ymin (float, optional): The minimum y-axis value. Defaults to 0.0.

            ymax (float, optional): The maximum y-axis value. Defaults to 1.0.

            legend (bool, optional): Whether to show the legend. Defaults to False.

            legend_loc (str, optional): The location of the legend. Defaults to "upper left".

            split (bool, optional): Whether to split the violins by the hue variable. Defaults to False.
        """
        sns.violinplot(x=x, y=y, hue=hue, data=data, split=split)
        Plotting.make_labs(
            xlab,
            ylab,
            title,
            fontsize=fontsize,
            labelsize=labelsize,
            ymin=ymin,
            ymax=ymax,
            legend=legend,
            legend_loc=legend_loc,
        )

    def boxplot(
        x,
        y,
        data,
        xlab,
        ylab,
        title,
        hue=None,
        fontsize=28,
        labelsize=20,
        ymin=0.0,
        ymax=1.0,
        legend=False,
        legend_loc="upper left",
    ):
        """
        Create a box plot to visualize the distribution of a continuous variable across different categories.

        Args:
            x (str): The column name in the data frame to use as the x-axis variable.

            y (str): The column name in the data frame to use as the y-axis variable.

            data (pd.DataFrame): The input data frame.

            xlab (str): The label for the x-axis.

            ylab (str): The label for the y-axis.

            title (str): The title of the plot.

            hue (str, optional): The column name in the data frame to use for grouping the data.

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            labelsize (int, optional): The font size for tick labels. Defaults to 20.

            ymin (float, optional): The minimum y-axis value. Defaults to 0.0.

            ymax (float, optional): The maximum y-axis value. Defaults to 1.0.

            legend (bool, optional): Whether to show the legend. Defaults to False.

            legend_loc (str, optional): The location of the legend. Defaults to "upper left".
        """
        sns.boxplot(x=x, y=y, hue=hue, data=data)
        Plotting.make_labs(
            xlab,
            ylab,
            title,
            fontsize=fontsize,
            labelsize=labelsize,
            ymin=ymin,
            ymax=ymax,
            legend=legend,
            legend_loc=legend_loc,
        )

    @staticmethod
    def scatterplot(
        x,
        y,
        data,
        xlab,
        ylab,
        title,
        fontsize=28,
        labelsize=20,
        ymin=0.0,
        ymax=1.0,
        legend=False,
        legend_loc="upper left",
    ):
        """
        Create a scatter plot to visualize the relationship between two continuous variables.

        Args:
            x (str): The column name in the data frame to use as the x-axis variable.

            y (str): The column name in the data frame to use as the y-axis variable.

            data (pd.DataFrame): The input data frame.

            xlab (str): The label for the x-axis.

            ylab (str): The label for the y-axis.

            title (str): The title of the plot.

            fontsize (int, optional): The font size for labels and titles. Defaults to 28.

            labelsize (int, optional): The font size for tick labels. Defaults to 20.

            ymin (float, optional): The minimum y-axis value. Defaults to 0.0.

            ymax (float, optional): The maximum y-axis value. Defaults to 1.0.

            legend (bool, optional): Whether to show the legend. Defaults to False.

            legend_loc (str, optional): The location of the legend. Defaults to "upper left".
        """
        sns.scatterplot(x=x, y=y, data=data)
        Plotting.make_labs(
            xlab,
            ylab,
            title,
            fontsize=fontsize,
            labelsize=labelsize,
            ymin=ymin,
            ymax=ymax,
        )

    @staticmethod
    def plot_filter_report(
        df,
        df2,
        df_populations,
        df_maf,
        maf_per_threshold,
        maf_props_per_threshold,
        plot_dir,
        output_file,
        plot_fontsize,
        plot_ticksize,
        plot_ymin,
        plot_ymax,
        plot_legend_loc,
        show,
    ):
        """
        Plot the filter report.

        Args:
            df (pd.DataFrame): The dataframe containing the filter report data.

            df2 (pd.DataFrame): Another dataframe containing the filter report data.

            df_populations (pd.DataFrame): The dataframe containing population data for filtering.

            df_maf (pd.DataFrame): The dataframe containing MAF data.

            maf_per_threshold (list): A list of MAF values per threshold.

            maf_props_per_threshold (list): A list of MAF proportions per threshold.

            plot_dir (str): The directory to save the plots.

            output_file (str): The output file name for the main filter report plot.

            plot_fontsize (int): The font size for labels and titles in the plots.

            plot_ticksize (int): The font size for tick labels in the plots.

            plot_ymin (float): The minimum value for the y-axis in the plots.

            plot_ymax (float): The maximum value for the y-axis in the plots.

            plot_legend_loc (str): The location of the legend in the plots.

            show (bool): Whether to show the plots or not.
        """
        # plot the boxplots
        fig, axs = plt.subplots(3, 2, figsize=(48, 27))
        ax1 = sns.boxplot(
            x="Threshold", y="Proportion", hue="Type", data=df, ax=axs[0, 0]
        )

        ax2 = sns.boxplot(
            x="Threshold",
            y="Proportion",
            hue="Type",
            data=df_populations,
            ax=axs[0, 1],
        )

        ax3 = sns.lineplot(
            x="Threshold", y="Proportion", hue="Type", data=df, ax=axs[1, 0]
        )

        ax4 = sns.lineplot(
            x="Threshold",
            y="Proportion",
            hue="Type",
            data=df_populations,
            ax=axs[1, 1],
        )

        ax5 = sns.violinplot(
            x="Threshold",
            y="Proportion",
            hue="Type",
            data=df,
            inner="box",
            ax=axs[2, 0],
        )

        ax6 = sns.violinplot(
            x="Threshold",
            y="Proportion",
            hue="Type",
            data=df_populations,
            inner="quartile",
            ax=axs[2, 1],
        )

        titles = [
            "Global and Sample Filtering",
            "Per-population Filtering",
        ]

        titles.extend([""] * 4)

        for title, ax in zip(titles, [ax1, ax2, ax3, ax4, ax5, ax6]):
            plt.sca(ax)
            Plotting.make_labs(
                "Missing Data Threshold",
                "Proportion of Missing Data",
                title,
                legend=True,
                fontsize=plot_fontsize,
                labelsize=plot_ticksize,
                ymin=plot_ymin,
                ymax=plot_ymax,
                legend_loc=plot_legend_loc,
            )

        plt.tight_layout()

        outfile = os.path.join(plot_dir, output_file)
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        fig.savefig(outfile, facecolor="white")

        if show:
            plt.show()
        else:
            plt.close()

        # Plot the MAF visualizations in a separate figure
        fig_maf, axs_maf = plt.subplots(4, 2, figsize=(24, 32))

        for maf, props in zip(maf_per_threshold, maf_props_per_threshold):
            plt.sca(axs_maf[0, 0])
            Plotting.histogram_maf(maf)

            plt.sca(axs_maf[0, 1])
            Plotting.cdf_maf(maf)

            plt.sca(axs_maf[1, 0])
            Plotting.boxplot(
                "Threshold",
                "Proportion",
                df_maf,
                "Minimum MAF Threshold",
                "Proportion of Missing Data",
                "MAF vs. Missing Data",
                fontsize=plot_fontsize,
                labelsize=plot_ticksize,
                ymin=plot_ymin,
                ymax=plot_ymax,
            )

            plt.sca(axs_maf[1, 1])
            Plotting.scatterplot(
                "Threshold",
                "Proportion",
                df_maf,
                "Minimum MAF Threshold",
                "Minimum MAF Threshold",
                f"MAF vs. Missing Data",
                fontsize=plot_fontsize,
                labelsize=plot_ticksize,
                ymin=plot_ymin,
                ymax=plot_ymax,
            )

        plt.sca(axs_maf[2, 0])
        Plotting.lineplot_maf(df_maf)

        plt.sca(axs_maf[2, 1])
        Plotting.cdf_maf(
            props,
            title="Cumulative Missing Data (MAF)",
            ylab="Cumulative Missing Proportion",
            fontsize=plot_fontsize,
            labelsize=plot_ticksize,
            ymin=plot_ymin,
            ymax=plot_ymax,
        )

        plt.sca(axs_maf[3, 0])
        Plotting.violinplot(
            "Type",
            "Proportion",
            df2,
            "Filter Type",
            "Proportion of Missing Data",
            "Allele Count Filters",
            hue="Filtered",
            legend=True,
            legend_loc=plot_legend_loc,
            fontsize=plot_fontsize,
            labelsize=plot_ticksize,
            ymin=plot_ymin,
            ymax=plot_ymax,
            split=True,
        )

        plt.sca(axs_maf[3, 1])
        Plotting.boxplot(
            "Type",
            "Proportion",
            df2,
            "Filter Type",
            "Proportion of Missing Data",
            "Allele Count Filters",
            hue="Filtered",
            legend=True,
            legend_loc=plot_legend_loc,
            fontsize=plot_fontsize,
            labelsize=plot_ticksize,
            ymin=plot_ymin,
            ymax=plot_ymax,
        )

        plt.tight_layout()

        outfile_maf = os.path.join(plot_dir, f"maf_{output_file}")
        fig_maf.savefig(outfile_maf, facecolor="white")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_pop_counts(
        populations, plot_dir, fontsize=28, ticksize=20, show=False
    ):
        """
        Plot the population counts.

        Args:
            populations (pd.Series): The series containing population data.

            plot_dir (str): The directory to save the plot.

            fontsize (int): The font size for labels and titles in the plot.

            ticksize (int): The font size for tick labels in the plot.

            show (bool): Whether to show the plot or not.
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
            plt.sca(ax)
            sns.barplot(x=data.index, y=data.values, color=color)
            median_line = plt.axhline(
                median, color=median_color, linestyle="--"
            )  # Add a horizontal line for the median
            plt.xticks(rotation=90)  # Rotate the x-axis labels if they're long
            plt.title("Population Counts", fontsize=fontsize)
            plt.xlabel("Population ID", fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.tick_params(axis="both", labelsize=ticksize)
            plt.legend(
                [median_line], ["Median"], loc="upper right", fontsize=ticksize
            )

        plt.tight_layout()

        fig.savefig(
            os.path.join(plot_dir, "population_counts.png"),
            facecolor="white",
        )

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_performance(
        resource_data, fontsize=14, color="#8C56E3", figsize=(16, 9)
    ):
        """Plots the performance metrics: CPU Load, Memory Footprint, and Execution Time.

        This static method takes a dictionary of performance data and plots the metrics for each of the methods. The resulting plots are saved in a .png file.

        Args:
            resource_data (dict): Dictionary with performance data. Keys are method names, and values are dictionaries with keys 'cpu_load', 'memory_footprint', and 'execution_time'.

            fontsize (int, optional): Font size to be used in the plot. Defaults to 14.

            color (str, optional): Color to be used in the plot. Should be a valid color string. Defaults to "#8C56E3".

            figsize (tuple, optional): Size of the figure. Should be a tuple of 2 integers. Defaults to (16, 9).

        Returns:
            None. The function saves the plot as a .png file.
        """
        methods = list(resource_data.keys())

        cpu_loads = [data["cpu_load"] for data in resource_data.values()]
        memory_footprints = [
            data["memory_footprint"] for data in resource_data.values()
        ]
        execution_times = [
            data["execution_time"] for data in resource_data.values()
        ]

        # Plot CPU Load
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        plt.sca(axs[0])

        sns.barplot(
            x=methods,
            y=cpu_loads,
            errorbar=None,
            color=color,
        )
        plt.xlabel("Methods", fontsize=fontsize)
        plt.ylabel("CPU Load (%)", fontsize=fontsize)
        plt.title(f"CPU Load Performance", fontsize=fontsize)
        plt.xticks(rotation=90, fontsize=fontsize)
        plt.ylim(bottom=0)
        plt.tight_layout()

        plt.sca(axs[1])

        # Plot Memory Footprint
        sns.lineplot(
            x=methods,
            y=memory_footprints,
            errorbar=None,
            color=color,
        )
        plt.xlabel("Method Execution/ Property Access", fontsize=fontsize)
        plt.ylabel("Memory Footprint (MB)", fontsize=fontsize)
        plt.title(f"Memory Footprint Performance", fontsize=fontsize)
        plt.xticks(rotation=90, fontsize=fontsize)
        plt.tight_layout()

        plt.sca(axs[2])

        # Plot Execution Time
        sns.barplot(
            x=methods,
            y=execution_times,
            errorbar=None,
            color=color,
        )
        plt.xlabel("Methods", fontsize=fontsize)
        plt.ylabel("Execution Time (seconds)", fontsize=fontsize)
        plt.title(f"Execution Time Performance", fontsize=fontsize)
        plt.xticks(rotation=90, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()

        fig.savefig(f"tests/benchmarking_plot.png", facecolor="white")

    @staticmethod
    def run_pca(
        genotype_data,
        plot_dir="plots",
        prefix=None,
        n_components=None,
        center=True,
        scale=False,
        n_axes=2,
        point_size=15,
        font_size=15,
        plot_format="png",
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

        A GenotypeData object needs to be passed to the function as a positional argument.

        PCA (principal component analysis) scatterplot can have either two or three axes, set with the n_axes parameter.

        The plot is saved as both an interactive HTML file and as a static image. Each population is represented by point shapes. The interactive plot has associated metadata when hovering over the points.

        Files are saved to a reports directory as <prefix>_output/imputed_pca.<plot_format|html>. Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").

        Args:
            genotype_data (GenotypeData): Original GenotypeData object.

            plot_dir (str, optional): Path to plot directory. Report directory will be created if it does not already exist. Defaults to "plots".

            prefix (str, optional): Prefix for plot filename. Will be saved in ``plot_dir``\. If ``prefix`` is None, then no prefix will be prepended to the filename. Defaults to None.

            n_components (int, optional): Number of principal components to include in the PCA. Defaults to None (all components).

            center (bool, optional): If True, centers the genotypes to the mean before doing the PCA. If False, no centering is done. Defaults to True.

            scale (bool, optional): If True, scales the genotypes to unit variance before doing the PCA. If False, no scaling is done. Defaults to False.

            n_axes (int, optional): Number of principal component axes to plot. Must be set to either 2 or 3. If set to 3, a 3-dimensional plot will be made. Defaults to 2.

            point_size (int, optional): Point size for scatterplot points. Defaults to 15.

            font_size (int, optional): Font size for scatterplot points. Defaults to 15.

            plot_format (str, optional): Plot file format to use. Supported formats include: "pdf", "svg", "png", and "jpeg" (or "jpg"). An interactive HTML file is also created regardless of this setting. Defaults to "pdf".

            bottom_margin (int, optional): Adjust bottom margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            top_margin (int, optional): Adjust top margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            left_margin (int, optional): Adjust left margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            right_margin (int, optional): Adjust right margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            width (int, optional): Width of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 1088.

            height (int, optional): Height of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 700.

        Returns:
            numpy.ndarray: PCA data as a numpy array with shape (n_samples, n_components).

            sklearn.decomposision.PCA: Scikit-learn PCA object from sklearn.decomposision.PCA. Any of the sklearn.decomposition.PCA attributes can be accessed from this object. See sklearn documentation.

        Examples:
            >>> data = GenotypeData(
            >>>     filename="snps.str",
            >>>     filetype="structure2row",
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
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        if n_axes > 3:
            raise ValueError(
                ">3 axes is not supported; n_axes must be either 2 or 3."
            )
        if n_axes < 2:
            raise ValueError(
                "<2 axes is not supported; n_axes must be either 2 or 3."
            )

        df = misc.validate_input_type(
            genotype_data.genotypes_012(fmt="pandas"), return_type="df"
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

        df_pca["SampleID"] = genotype_data.samples
        df_pca["Population"] = genotype_data.populations
        df_pca["Size"] = point_size

        _, ind, _, _, _ = genotype_data.calc_missing(df, use_pops=False)
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
            labs[
                "Axis3"
            ] = f"PC3 ({round(model.explained_variance_ratio_[2] * 100, 2)}%)"
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
            legend_title_font=dict(size=font_size),
            legend_title_side="top",
            font=dict(size=font_size),
        )

        fname = "pca" if prefix is None else f"{prefix}_pca"

        fig.write_html(os.path.join(plot_dir, f"{fname}.html"))
        fig.write_image(
            os.path.join(plot_dir, f"{fname}.{plot_format}"),
        )

        return components, model

    @staticmethod
    def visualize_missingness(
        genotype_data,
        df,
        zoom=True,
        prefix=None,
        horizontal_space=0.6,
        vertical_space=0.6,
        bar_color="gray",
        heatmap_palette="magma",
        plot_format="png",
        plot_dir="plots",
        dpi=300,
    ):
        """Make multiple plots to visualize missing data.

        Args:
            genotype_data (GenotypeData): Initialized GentoypeData object.

            df (pandas.DataFrame): DataFrame with snps to visualize.

            zoom (bool, optional): If True, zooms in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.

            prefix (str, optional): Prefix for output directory and files. Plots and files will be written to a directory called <prefix>_reports. The report directory will be created if it does not already exist. If prefix is None, then the reports directory will not have a prefix. Defaults to None.

            horizontal_space (float, optional): Set width spacing between subplots. If your plot are overlapping horizontally, increase horizontal_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            vertical_space (float, optioanl): Set height spacing between subplots. If your plots are overlapping vertically, increase vertical_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            bar_color (str, optional): Color of the bars on the non-stacked barplots. Can be any color supported by matplotlib. See matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, optional): Palette to use for heatmap plot. Can be any palette supported by seaborn. See seaborn documentation. Defaults to 'magma'.

            plot_format (str, optional): Format to save plots. Can be any of the following: "pdf", "png", "svg", "ps", "eps". Defaults to "png".

            plot_dir (str, optional): Directory to save plots in. Defaults to "plots".

            dpi (int): The resolution in dots per inch. Defaults to 300.

        Returns:
            pandas.DataFrame: Per-locus missing data proportions.

            pandas.DataFrame: Per-individual missing data proportions.

            pandas.DataFrame: Per-population + per-locus missing data proportions.

            pandas.DataFrame: Per-population missing data proportions.

            pandas.DataFrame: Per-individual and per-population missing data proportions.
        """

        if not isinstance(df, pd.DataFrame):
            df = misc.validate_input_type(df, return_type="df")

        loc, ind, poploc, poptotal, indpop = genotype_data.calc_missing(df)

        ncol = 3
        nrow = 1 if genotype_data.populations is None else 2

        fig, axes = plt.subplots(nrow, ncol, figsize=(8, 11))
        plt.subplots_adjust(wspace=horizontal_space, hspace=vertical_space)
        fig.suptitle("Missingness Report")

        ax = axes[0, 0]

        ax.set_title("Per-Individual")
        ax.barh(genotype_data.samples, ind, color=bar_color, height=1.0)
        if not zoom:
            ax.set_xlim([0, 1])
        ax.set_ylabel("Sample")
        ax.set_xlabel("Missing Prop.")
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

        ax = axes[0, 1]

        ax.set_title("Per-Locus")
        ax.barh(
            range(genotype_data.num_snps), loc, color=bar_color, height=1.0
        )
        if not zoom:
            ax.set_xlim([0, 1])
        ax.set_ylabel("Locus")
        ax.set_xlabel("Missing Prop.")
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

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

            ax.set_title("Per-Population + Per-Locus")
            npops = len(poploc.columns)

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
        melt_df["SampleID"] = genotype_data.samples
        indpop["SampleID"] = genotype_data.samples

        if poptotal is not None:
            melt_df["Population"] = genotype_data.populations
            indpop["Population"] = genotype_data.populations

        melt_df = melt_df.melt(value_name="Missing", id_vars=id_vars)
        melt_df.sort_values(by=id_vars[::-1], inplace=True)
        melt_df["Missing"].replace(False, "Present", inplace=True)
        melt_df["Missing"].replace(True, "Missing", inplace=True)

        ax = axes[0, 2] if poptotal is None else axes[1, 1]

        ax.set_title("Per-Individual")
        g = sns.histplot(
            data=melt_df,
            y="variable",
            hue="Missing",
            multiple="fill",
            ax=ax,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )
        g.get_legend().set_title(None)

        if poptotal is not None:
            ax = axes[1, 2]

            ax.set_title("Per-Population")
            g = sns.histplot(
                data=melt_df,
                y="Population",
                hue="Missing",
                multiple="fill",
                ax=ax,
            )
            g.get_legend().set_title(None)

        fname = "missingness" if prefix is None else f"{prefix}_missingness"

        fig.savefig(
            os.path.join(plot_dir, f"{fname}.{plot_format}"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

        return loc, ind, poploc, poptotal, indpop

    @staticmethod
    def run_dapc(
        genotype_data,
        plot_dir="plots",
        prefix=None,
        n_components=None,
        center=True,
        scale=False,
        point_size=15,
        font_size=15,
        plot_format="pdf",
        bottom_margin=0,
        top_margin=0,
        left_margin=0,
        right_margin=0,
        width=1088,
        height=700,
    ):
        """Runs DAPC and makes scatterplot with colors showing missingness.

        NOTE: Not yet implemented and currently non-functional.

        Genotypes are plotted as separate shapes per population and colored according to missingness per individual.

        A GenotypeData object needs to be passed to the function as a positional argument.

        The plot is saved as both an interactive HTML file and as a static image. Each population is represented by point shapes. The interactive plot has associated metadata when hovering over the points.

        Files are saved to a reports directory as plot_dir/prefix_imputed_pca.<plot_format|html>. An underscore will be appended to the prefix, so you don't need to add one. Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").

        Args:
            genotype_data (GenotypeData): Original GenotypeData object.

            plot_dir (str, optional): Path to plot directory. Report directory will be created if it does not already exist. Defaults to "plots".

            prefix (str, optional): Prefix for plot filename. Will be saved in ``plot_dir``\. If ``prefix`` is None, then no prefix will be prepended to the filename. Defaults to None.

            n_components (int, optional): Number of principal components to include in the DA. NOTE: n_components cannot be larger than ``min(n_sites, n_populations - 1)``\. Defaults to None (n_populations - 1).

            center (bool, optional): If True, centers the genotypes to the mean before doing the DA. If False, no centering is done. Defaults to True.

            scale (bool, optional): If True, scales the genotypes to unit variance before doing the DA. If False, no scaling is done. Defaults to False.

            point_size (int, optional): Point size for scatterplot points. Defaults to 15.

            font_size (int, optional): Font size for scatterplot points. Defaults to 15.

            plot_format (str, optional): Plot file format to use. Supported formats include: "pdf", "svg", "png", and "jpeg" (or "jpg"). An interactive HTML file is also created regardless of this setting. Defaults to "pdf".

            bottom_margin (int, optional): Adjust bottom margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            top_margin (int, optional): Adjust top margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            left_margin (int, optional): Adjust left margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            right_margin (int, optional): Adjust right margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            width (int, optional): Width of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 1088.

            height (int, optional): Height of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 700.

        Returns:
            numpy.ndarray: DA data as a numpy array with shape (n_samples, n_components).

            sklearn.discriminant_analysis.LinearDiscriminantAnalysis: Scikit-learn LinearDiscriminateAnalysis object from sklearn.discriminant_analysis.LinearDiscriminantAnalysis. Any of the sklearn.discriminant_analysis.LinearDiscriminantAnalysis attributes can be accessed from this object. See sklearn documentation.

        Examples:
            >>> data = GenotypeData(
            >>>     filename="snps.str",
            >>>     filetype="structure2row",
            >>>     popmapfile="popmap.txt",
            >>> )
            >>>
            >>> components, dapc = run_dapc(
            >>>     data,
            >>>     scale=True,
            >>>     center=True,
            >>>     plot_format="png"
            >>> )
            >>>
            >>> # Calculate and print explained variance ratio
            >>> explvar = dapc.explained_variance_ratio_
            >>> print(explvar)

        """
        raise NotImplementedError("run_dapc has not yet been implemented.")

        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        df = misc.validate_input_type(
            genotype_data.genotypes_012(fmt="pandas"), return_type="df"
        )

        df.replace(-9, np.nan, inplace=True)

        if center or scale:
            # Center data to mean. Scaling to unit variance is off.
            scaler = StandardScaler(with_mean=center, with_std=scale)
            pca_df = scaler.fit_transform(df)
        else:
            pca_df = df.copy()

        if n_components is None:
            n_components = len(list(set(genotype_data.populations))) - 1

        # DA can't handle missing data. So impute it here using the K
        # nearest neighbors (samples).
        imputer = KNNImputer(weights="distance")
        pca_df = imputer.fit_transform(pca_df)
        popmap = pd.DataFrame(
            {
                "SampleID": genotype_data.samples,
                "PopulationID": genotype_data.populations,
            }
        )

        best_components = Plotting._plot_dapc_cv(
            pca_df, popmap, n_components, prefix=prefix, plot_dir=plot_dir
        )

        model = LinearDiscriminantAnalysis(n_components=best_components)
        components = model.fit_transform(pca_df, y=genotype_data.populations)

        if n_components is None:
            n_components = len(list(set(genotype_data.populations))) - 1

        df_pca = pd.DataFrame(
            components[:, [0, 1]], columns=["Axis1", "Axis2"]
        )

        df_pca["SampleID"] = genotype_data.samples
        df_pca["Population"] = genotype_data.populations
        df_pca["Size"] = point_size

        _, ind, _, _, _ = genotype_data.calc_missing(df, use_pops=False)
        df_pca["missPerc"] = ind

        my_scale = [("rgb(19, 43, 67)"), ("rgb(86,177,247)")]  # ggplot default

        labs = {
            "Axis1": f"DA1 ({round(model.explained_variance_ratio_[0] * 100, 2)}%)",
            "Axis2": f"DA2 ({round(model.explained_variance_ratio_[1] * 100, 2)}%)",
            "missPerc": "Missing Prop.",
            "Population": "Population",
        }

        fig = px.scatter(
            df_pca,
            x="Axis1",
            y="Axis2",
            color="missPerc",
            symbol="Population",
            color_continuous_scale=my_scale,
            custom_data=["SampleID", "Population", "missPerc"],
            size="Size",
            size_max=point_size,
            labels=labs,
        )
        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "Axis 1: %{x}",
                    "Axis 2: %{y}",
                    "Sample ID: %{customdata[0]}",
                    "Population: %{customdata[1]}",
                    "Missing Prop.: %{customdata[2]}",
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
            legend_title_font=dict(size=font_size),
            legend_title_side="top",
            font=dict(size=font_size),
        )

        fname = "da" if prefix is None else f"{prefix}_da"

        fig.write_html(os.path.join(plot_dir, f"{fname}.html"))
        fig.write_image(
            os.path.join(plot_dir, f"{fname}.{plot_format}"),
        )

        return components, model
