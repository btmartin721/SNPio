import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from . import misc
except (ModuleNotFoundError, ValueError):
    from utils import misc


class Plotting:
    """Functions for plotting imputer scoring and results."""

    @staticmethod
    def visualize_missingness(
        genotype_data,
        df,
        zoom=True,
        prefix="imputer",
        horizontal_space=0.6,
        vertical_space=0.6,
        bar_color="gray",
        heatmap_palette="magma",
        plot_format="pdf",
        dpi=300,
    ):
        """Make multiple plots to visualize missing data.

        Args:
            genotype_data (GenotypeData): Initialized GentoypeData object.

            df (pandas.DataFrame): DataFrame with snps to visualize.

            zoom (bool, optional): If True, zooms in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.

            prefix (str, optional): Prefix for output directory and files. Plots and files will be written to a directory called <prefix>_reports. The report directory will be created if it does not already exist. If prefix is None, then the reports directory will not have a prefix. Defaults to 'imputer'.

            horizontal_space (float, optional): Set width spacing between subplots. If your plot are overlapping horizontally, increase horizontal_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            vertical_space (float, optioanl): Set height spacing between subplots. If your plots are overlapping vertically, increase vertical_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            bar_color (str, optional): Color of the bars on the non-stacked barplots. Can be any color supported by matplotlib. See matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, optional): Palette to use for heatmap plot. Can be any palette supported by seaborn. See seaborn documentation. Defaults to 'magma'.

            plot_format (str, optional): Format to save plots. Can be any of the following: "pdf", "png", "svg", "ps", "eps". Defaults to "pdf".

            dpi (int): The resolution in dots per inch. Defaults to 300.

        Returns:
            pandas.DataFrame: Per-locus missing data proportions.
            pandas.DataFrame: Per-individual missing data proportions.
            pandas.DataFrame: Per-population + per-locus missing data proportions.
            pandas.DataFrame: Per-population missing data proportions.
            pandas.DataFrame: Per-individual and per-population missing data proportions.
        """

        loc, ind, poploc, poptotal, indpop = genotype_data.calc_missing(df)

        ncol = 3
        nrow = 1 if genotype_data.pops is None else 2

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
            melt_df["Population"] = genotype_data.pops
            indpop["Population"] = genotype_data.pops

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

        fig.savefig(
            os.path.join(
                f"{prefix}_output", "plots", f"missingness.{plot_format}"
            ),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.cla()
        plt.clf()
        plt.close()

        return loc, ind, poploc, poptotal, indpop

    @staticmethod
    def run_and_plot_pca(
        original_genotype_data,
        imputer_object,
        prefix="imputer",
        n_components=3,
        center=True,
        scale=False,
        n_axes=2,
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
        """Runs PCA and makes scatterplot with colors showing missingness.

        Genotypes are plotted as separate shapes per population and colored according to missingness per individual.

        This function is run at the end of each imputation method, but can be run independently to change plot and PCA parameters such as ``n_axes=3`` or ``scale=True``.

        The imputed and original GenotypeData objects need to be passed to the function as positional arguments.

        PCA (principal component analysis) scatterplot can have either two or three axes, set with the n_axes parameter.

        The plot is saved as both an interactive HTML file and as a static image. Each population is represented by point shapes. The interactive plot has associated metadata when hovering over the points.

        Files are saved to a reports directory as <prefix>_output/imputed_pca.<plot_format|html>. Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").

        Args:
            original_genotype_data (GenotypeData): Original GenotypeData object that was input into the imputer.

            imputer_object (Any imputer instance): Imputer object created when imputing. Can be any of the imputers, such as: ``ImputePhylo()``, ``ImputeUBP()``, and ``ImputeRandomForest()``.

            original_012 (pandas.DataFrame, numpy.ndarray, or List[List[int]], optional): Original 012-encoded genotypes (before imputing). Missing values are encoded as -9. This object can be obtained as ``df = GenotypeData.genotypes012_df``.

            prefix (str, optional): Prefix for report directory. Plots will be save to a directory called <prefix>_output/imputed_pca<html|plot_format>. Report directory will be created if it does not already exist. Defaults to "imputer".

            n_components (int, optional): Number of principal components to include in the PCA. Defaults to 3.

            center (bool, optional): If True, centers the genotypes to the mean before doing the PCA. If False, no centering is done. Defaults to True.

            scale (bool, optional): If True, scales the genotypes to unit variance before doing the PCA. If False, no scaling is done. Defaults to False.

            n_axes (int, optional): Number of principal component axes to plot. Must be set to either 2 or 3. If set to 3, a 3-dimensional plot will be made. Defaults to 2.

            point_size (int, optional): Point size for scatterplot points. Defaults to 15.

            plot_format (str, optional): Plot file format to use. Supported formats include: "pdf", "svg", "png", and "jpeg" (or "jpg"). An interactive HTML file is also created regardless of this setting. Defaults to "pdf".

            bottom_margin (int, optional): Adjust bottom margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            top (int, optional): Adjust top margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

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
            >>> components, pca = Plotting.run_and_plot_pca(
            >>>     data,
            >>>     ubp,
            >>>     scale=True,
            >>>     center=True,
            >>>     plot_format="png"
            >>> )
            >>>
            >>> # Calculate and print explained variance ratio
            >>> explvar = pca.explained_variance_ratio_
            >>> print(explvar)

        """
        report_path = os.path.join(f"{prefix}_output", "plots")
        Path(report_path).mkdir(parents=True, exist_ok=True)

        if n_axes > 3:
            raise ValueError(
                ">3 axes is not supported; n_axes must be either 2 or 3."
            )
        if n_axes < 2:
            raise ValueError(
                "<2 axes is not supported; n_axes must be either 2 or 3."
            )

        imputer = imputer_object.imputed

        df = misc.validate_input_type(
            imputer.genotypes012_df, return_type="df"
        )

        original_df = misc.validate_input_type(
            original_genotype_data.genotypes012_df, return_type="df"
        )

        original_df.replace(-9, np.nan, inplace=True)

        if center or scale:
            # Center data to mean. Scaling to unit variance is off.
            scaler = StandardScaler(with_mean=center, with_std=scale)
            pca_df = scaler.fit_transform(df)
        else:
            pca_df = df.copy()

        # Run PCA.
        model = PCA(n_components=n_components)
        components = model.fit_transform(pca_df)

        df_pca = pd.DataFrame(
            components[:, [0, 1, 2]], columns=["Axis1", "Axis2", "Axis3"]
        )

        df_pca["SampleID"] = original_genotype_data.samples
        df_pca["Population"] = original_genotype_data.pops
        df_pca["Size"] = point_size

        _, ind, _, _, _ = imputer.calc_missing(original_df, use_pops=False)
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
        fig.write_html(os.path.join(report_path, "imputed_pca.html"))
        fig.write_image(
            os.path.join(report_path, f"imputed_pca.{plot_format}"),
        )

        return components, model

    @staticmethod
    def plot_certainty_heatmap(y_certainty, sample_ids=None, prefix="imputer"):
        fig = plt.figure()
        hm = sns.heatmap(
            data=y_certainty,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Prob."},
        )
        hm.set_xlabel("Site")
        hm.set_ylabel("Sample")
        hm.set_title("Probabilities of Uncertain Sites")
        fig.tight_layout()
        fig.savefig(
            os.path.join(f"{prefix}_output", "plots", f"uncertainty_plot.png"),
            bbox_inches="tight",
            facecolor="white",
        )
