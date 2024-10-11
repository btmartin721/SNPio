import itertools
import math
from ctypes import Union
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils import misc
from snpio.utils.logging import LoggerManager


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
        plot_sfs: Plot a heatmap for the 2D SFS between two given populations and bar plots for the 1D SFS of each population. Not yet implemented.
        plot_joint_sfs_grid: Plot the joint SFS between all possible pairs of populations in the popmap file in a grid layout. Not yet implemented.
        _set_logger: Set the logger object based on the debug attribute. If debug is True, the logger will log debug messages.
        _get_attribute_value: Determine the value for an attribute based on the provided argument, genotype_data attribute, or default value. If a value is provided during initialization, it is used. Otherwise, the genotype_data attribute is used if available. If neither is available, the default value is used.
        _plot_summary_statistics_per_sample: Plot summary statistics per sample. If an axis is provided, the plot is drawn on that axis.
        _plot_summary_statistics_per_population: Plot summary statistics per population. If an axis is provided, the plot is drawn on that axis.
        _plot_summary_statistics_per_population_grid: Plot summary statistics per population using a Seaborn PairGrid plot. Not yet implemented.
        _plot_summary_statistics_per_sample_grid: Plot summary statistics per sample using a Seaborn PairGrid plot. Not yet implemented.
        _plot_dapc_cv: Plot the DAPC cross-validation results. Not yet implemented.
    """

    def __init__(
        self,
        genotype_data: Any,
        show: bool = None,
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

            show (bool, optional):
                Whether to display the plots.
                Defaults to `genotype_data.show` if available, otherwise `False`.

            plot_format (str, optional):
                The format in which to save the plots (e.g., 'png', 'svg').
                Defaults to `genotype_data.plot_format` if available, otherwise `'png'`.

            dpi (int, optional):
                The resolution of the saved plots. Unused for vector `plot_format` types.
                Defaults to `genotype_data.dpi` if available, otherwise `300`.

            plot_fontsize (int, optional):
                The font size for the plot labels.
                Defaults to `genotype_data.plot_fontsize` if available, otherwise `18`.

            plot_title_fontsize (int, optional):
                The font size for the plot titles.
                Defaults to `genotype_data.plot_title_fontsize` if available, otherwise `22`.

            despine (bool, optional):
                Whether to remove the top and right plot axis spines.
                Defaults to `genotype_data.despine` if available, otherwise `True`.

            verbose (bool, optional):
                Whether to enable verbose logging.
                Defaults to `genotype_data.verbose` if available, otherwise `False`.

            debug (bool, optional):
                Whether to enable debug logging.
                Defaults to `genotype_data.debug` if available, otherwise `False`.

        Raises:
            ValueError: Raised if the input genotype_data object is not provided.

        Returns:
            None: The Plotting object is initialized.

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
        self.prefix = getattr(genotype_data, "prefix", "plot")
        self.output_dir = Path(f"{self.prefix}_output", "nremover", "plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.debug = debug

        prefix = genotype_data.prefix
        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

        # Define default values for attributes
        self._defaults = {
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
        self._attributes = [
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

        mpl_params = {
            "xtick.labelsize": self.plot_fontsize,
            "ytick.labelsize": self.plot_fontsize,
            "legend.fontsize": self.plot_fontsize,
            "legend.fancybox": True,
            "legend.shadow": True,
            "figure.titlesize": self.plot_title_fontsize,
            "figure.facecolor": "white",
            "figure.dpi": self.dpi,
            "font.size": self.plot_fontsize,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Verdana",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
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
        }

        mpl.rcParams.update(mpl_params)

    def _get_attribute_value(self, attr: str) -> Any:
        """
        Determine the value for an attribute based on the provided argument,
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
        genotype_val = getattr(self.genotype_data, attr, None)
        if genotype_val is not None:
            self.logger.debug(f"Using genotype_data value for '{attr}': {genotype_val}")
            return genotype_val

        # Use the default value
        default = self._defaults.get(attr)
        self.logger.debug(f"Using default value for '{attr}': {default}")
        return default

    def _plot_summary_statistics_per_sample(
        self, summary_stats: pd.DataFrame, ax: Optional[mpl.axes.Axes] = None
    ) -> None:
        """Plot summary statistics per sample.

        This method plots the summary statistics per sample on the same figure. The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst). The x-axis represents the locus, and the y-axis represents the value of the summary statistic. The plot is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/gtdata/plots/summary_statistics_per_sample.{plot_format}``. The plot is saved in the format specified by the `plot_format` attribute.

        Args:
            summary_stats (pandas.DataFrame): The DataFrame containing the summary statistics per sample to be plotted.

            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.

        Returns:
            None: The summary statistics per sample plot is saved to a file.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.

        Note:
            - The summary statistics per sample are plotted as lines for each statistic (Ho, He, Pi, Fst).
            - The x-axis represents the locus, and the y-axis represents the value of the summary statistic.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/summary_statistics_per_sample.{plot_format}``.
            - The plot is colored by population and labeled by population with symbols for each sample.
            - The summary statistics per sample are grouped by population.
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

    def _plot_summary_statistics_per_population(
        self,
        summary_stats: pd.DataFrame,
        popmap: pd.DataFrame,
        ax: Optional[mpl.axes.Axes] = None,
    ) -> None:
        """Plot summary statistics per population.

        This method plots the summary statistics per population on the same figure. The summary statistics are grouped by population and plotted as lines for each statistic (Ho, He, Pi, Fst). The x-axis represents the population, and the y-axis represents the value of the summary statistic. The plot is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/gtdata/plots/summary_statistics_per_population.{plot_format}``. The plot is saved in the format specified by the `plot_format` attribute.

        Args:
            summary_stats (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

            popmap (pd.DataFrame): The DataFrame containing the population mapping used to group the summary statistics.

            ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot the summary statistics.

        Returns:
            None: The summary statistics per population plot is saved to a file.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.

        Note:
            - The summary statistics per population are grouped by population.
            - The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst).
            - The x-axis represents the population, and the y-axis represents the value of the summary statistic.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/summary_statistics_per_population.{plot_format}``.
            - The summary statistics per population are grouped by population.
            - The summary statistics are grouped by population.
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

    def _plot_summary_statistics_per_population_grid(
        self, summary_statistics_df: pd.DataFrame
    ) -> None:
        """Plot summary statistics per population using a Seaborn PairGrid plot.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

        Returns:
            None: The summary statistics per population grid plot is saved to a file.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.

        Note:
            - The summary statistics per population are plotted using a Seaborn PairGrid plot.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/summary_statistics_per_population_grid.{plot_format}``.
            - The summary statistics per population are grouped by population.
            - The summary statistics are grouped by population.
            - The summary statistics are plotted as a grid of scatter plots and kernel density plots.
            - The grid plot is colored by population and labeled by population with symbols for each sample.
            - The grid plot is saved to a file.
            - The grid plot is displayed if the `show` attribute is True.
        """
        g = sns.PairGrid(summary_statistics_df)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=3, legend=False)

        of = f"summary_statistics_per_population_grid.{self.plot_format}"
        of = self.output_dir / of
        g.savefig(of)

        if self.show:
            plt.show()
        plt.close()

    def _plot_summary_statistics_per_sample_grid(
        self, summary_statistics_df: pd.DataFrame
    ) -> None:
        """Plot summary statistics per sample using a Seaborn PairGrid plot.

        This method plots the summary statistics per sample using a Seaborn PairGrid plot. The summary statistics are plotted as a grid of scatter plots and kernel density plots. The grid plot is colored by population and labeled by population with symbols for each sample. The plot is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/gtdata/plots/summary_statistics_per_sample_grid.{plot_format}``. The plot is saved in the format specified by the `plot_format` attribute.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

        Returns:
            None: The summary statistics per sample grid plot is saved to a file.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.

        Note:
            - The summary statistics per sample are plotted using a Seaborn PairGrid plot.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/summary_statistics_per_sample_grid.{plot_format}``.
            - The summary statistics per sample are grouped by population.
            - The summary statistics are grouped by population.
            - The summary statistics are plotted as a grid of scatter plots and kernel density plots.
            - The grid plot is colored by population and labeled by population with symbols for each sample.
            - The grid plot is saved to a file.
        """
        g = sns.PairGrid(summary_statistics_df)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=3, legend=False)

        of = f"summary_statistics_per_sample_grid.{self.plot_format}"
        of = self.output_dir / of
        g.savefig(of)

        if self.show:
            plt.show()
        plt.close()

    def plot_summary_statistics(self, summary_statistics_df: pd.DataFrame) -> None:
        """Plot summary statistics per sample and per population on the same figure.

        This method plots the summary statistics per sample and per population on the same figure. The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst). The x-axis represents the locus, and the y-axis represents the value of the summary statistic. The plot is saved to a file. If the `show` attribute is True, the plot is displayed. The plot is saved to the `output_dir` directory with the filename: ``<prefix>_output/gtdata/plots/summary_statistics.{plot_format}``. The plot is saved in the format specified by the `plot_format` attribute.

        Args:
            summary_statistics_df (pd.DataFrame): The DataFrame containing the summary statistics to be plotted.

        Returns:
            None: The summary statistics plot is saved to a file.

        Raises:
            ValueError: Raised if the `dimensions` argument is neither 2 nor 3.

        Note:
            - The summary statistics per sample and per population are plotted on the same figure.
            - The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst).
            - The x-axis represents the locus, and the y-axis represents the value of the summary statistic.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/summary_statistics.{plot_format}``.
            - The summary statistics per sample and per population are grouped by population.
            - The summary statistics are grouped by population.
            - The summary statistics are plotted as lines for each statistic (Ho, He, Pi, Fst).
            - The plot is colored by population and labeled by population with symbols for each sample.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

        self._plot_summary_statistics_per_sample(summary_statistics_df, ax=axes[0])
        self._plot_summary_statistics_per_population(summary_statistics_df, ax=axes[1])

        of = f"summary_statistics.{self.plot_format}"
        of = self.output_dir / of
        fig.savefig(of)

        if self.show:
            plt.show()
        plt.close()

        self._plot_summary_statistics_per_sample_grid(summary_statistics_df)
        self._plot_summary_statistics_per_population_grid(summary_statistics_df)

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

        plt.savefig(of)

        if self.show:
            plt.show()

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
        plt.savefig(of)

        if self.show:
            plt.show()
        plt.close()

    def _plot_dapc_cv(
        self, df: pd.DataFrame, popmap: pd.DataFrame, n_components_range: range
    ) -> None:
        """Plot the DAPC cross-validation results.

        Args:
            df (Union[numpy.ndarray, pandas.DataFrame): The input DataFrame or array with the genotypes.

            popmap (pd.DataFrame): The DataFrame containing the population mapping information, with columns "SampleID" and "PopulationID".

            n_components_range (range): The range of principal components to use for cross-validation.

        Returns:
            None: A plot is saved to a .png file.

        Raises:
            NotImplementedError: This method is not yet implemented.

        Note:
            - This method is not yet implemented.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/dapc_cv_results.{plot_format}``.
            - The plot shows the mean cross-validation score for each number of components in the `n_components_range` range.
            - The optimal number of components is determined based on the minimum cross-validation score.
            - The optimal number of components and the best cross-validation score are printed to the console.

        Raises:
            NotImplementedError: This method is not yet implemented.
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
        plt.savefig(of)

        if self.show:
            plt.show()
        plt.close()

        best_idx = pd.Series(scores).idxmin()
        best_score = scores[best_idx]
        best_component = components[best_idx]

        print(f"\n\nOptimal DAPC Components: {best_component}")
        print(f"Best DAPC CV Score: {best_score}")

        return best_component

    def plot_sfs(
        self,
        pop_gen_stats: Any,
        population1: str,
        population2: str,
        savefig: bool = True,
    ):
        """Plot a heatmap for the 2D SFS between two given populations and
        bar plots for the 1D SFS of each population.

        Note:
            - This method is not yet implemented.
            - The plot is saved to a file in the `output_dir` directory.
            - The plot is displayed if the `show` attribute is True.
            - The plot is saved in the format specified by the `plot_format` attribute.
            - The plot is saved with the filename: ``<prefix>_output/gtdata/plots/sfs_{population1}_{population2}.{plot_format}``.
            - The 1D SFS for each population is calculated using the `calculate_1d_sfs` method.
            - The 2D SFS between the two populations is calculated using the `calculate_2d_sfs` method.
            - The 1D SFS for each population is plotted as a bar plot.
            - The 2D SFS between the two populations is plotted as a heatmap.
            - The 2D SFS heatmap is colored using a custom colormap.
            - The custom colormap is created using the colors "white", "green", "yellow", "orange", and "red".
            - The number of colors in the colormap is set to 50.
            - The custom colormap is used to color the 2D SFS heatmap.
            - The custom colormap is displayed in the colorbar.
            - The custom colormap is saved to a file in the `output_dir` directory.
            - The custom colormap is displayed if the `show` attribute is True.
            - The custom colormap is saved in the format specified by the `plot_format` attribute.

        Args:
            pop_gen_stats (PopGenStatistics): An instance of the PopGenStatistics class.

            population1 (str): The name of the first population.

            population2 (str): The name of the second population.

            savefig (bool, optional): Whether to save the figure to a file. Defaults to True. If True, the figure will be saved to a file.

        Returns:
            None: A plot is saved to a file.

        Raises:
            NotImplementedError: Raised if the method is not yet implemented.

        Note:
            - The 1D SFS for each population is calculated using the `calculate_1d_sfs` method.
            - The 2D SFS between the two populations is calculated using the `calculate_2d_sfs` method.
            - The 1D SFS for each population is plotted as a bar plot.
            - The 2D SFS between the two populations is plotted as a heatmap.
            - The 2D SFS heatmap is colored using a custom colormap.
            - The custom colormap is created using the colors "white", "green", "yellow", "orange", and "red".
            - The number of colors in the colormap is set to 50.
            - The custom colormap is used to color the 2D SFS heatmap.
            - The custom colormap is displayed in the colorbar.
            - The custom colormap is saved to a file in the `output_dir` directory.
            - The custom colormap is displayed if the `show` attribute is True.
            - The custom colormap is saved in the format specified by the `plot_format` attribute.
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

    def plot_joint_sfs_grid(
        self,
        pop_gen_stats: Any,
        populations: List[Union[str, int]],
        savefig: bool = True,
    ) -> None:
        """Plot the joint SFS between all possible pairs of populations in the popmap file in a grid layout.

        This method plots the joint SFS between all possible pairs of populations in the popmap file in a grid layout. The joint SFS is calculated using the `calculate_2d_sfs` method. The joint SFS is plotted as a heatmap for each pair of populations. The heatmap is colored using a custom colormap. The custom colormap is created using the colors "white", "green", "yellow", "orange", and "red". The number of colors in the colormap is set to 50. The custom colormap is used to color the joint SFS heatmap. The custom colormap is displayed in the colorbar. The custom colormap is saved to a file in the `output_dir` directory. The custom colormap is displayed if the `show` attribute is True. The custom colormap is saved in the format specified by the `plot_format` attribute. The plot is saved to a file in the `output_dir` directory. If the `show` attribute is True, the plot is displayed. The plot is saved in the format specified by the `plot_format` attribute. The plot is saved with the filename: ``<prefix>_output/gtdata/plots/joint_sfs_grid.{plot_format}``.

        Note:
            - This method is not yet implemented.

        Args:
            pop_gen_stats (PopGenStatistics): An instance of the PopGenStatistics class.

            populations (List[Union[str, int]]): A list of population names.

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

        plot_dir = self.output_dir / "sankey_plots"
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
        dftmp = df[df["Filter_Method"] != "filter_missing_sample"]

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
            of = f"filtering_results_sankey_thresholds{thresholds}.html"
            fname = plot_dir / of
            hv.save(sankey_plot, fname, fmt="html")

        except ValueError as e:
            self.logger.warning(
                f"Failed to generate Sankey plot with thresholds: {thresholds}: error: {e}"
            )

    def plot_gt_distribution(self, df: pd.DataFrame, annotation_size: int = 15) -> None:
        """Plot the distribution of genotype counts.

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

        df_melt = pd.melt(df, value_name="Count")
        cnts = df_melt["Count"].value_counts()
        cnts.index.names = ["Genotype Int"]
        cnts = pd.DataFrame(cnts).reset_index()
        cnts = cnts.sort_values(by="Genotype Int")
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
        fig.savefig(of)

        if self.show:
            plt.show()
        plt.close()

    def plot_search_results(self, df_combined: pd.DataFrame) -> None:
        """Plot and save the filtering results based on the available data.

        This method plots the filtering results based on the available data. The filtering results are plotted for the per-sample and per-locus missing data proportions, MAF, and boolean filtering thresholds. The plots are saved to files in the output directory. If the `show` attribute is True, the plots are displayed. The plots are saved in the format specified by the `plot_format` attribute into the `output_dir` directory in the format: ``<prefix>_output/gtdata/plots/filtering_results_{method}.{plot_format}``.

        Args:
            df_combined (pd.DataFrame): The input dataframe containing the filtering results.

        Returns:
            None: Plots are saved to files.

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

        msg = f"Plotting complete. Plots saved to directory {self.output_dir}."
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
            fig.savefig(of)

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
            fig.savefig(of)

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

        Returns:
            None: A plot is saved to a file.

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
        df_mac = df[df["Filter_Method"] == "filter_mac"].copy()
        df = df[df["Filter_Method"] == "filter_maf"].copy()

        self.logger.debug(f"MAF data: {df}")
        self.logger.debug(f"MAC data: {df_mac}")

        if not df.empty:
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
            fig.savefig(of)

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

            of = self.output_dir / f"filtering_results_mac.{self.plot_format}"
            fig.savefig(of)

            if self.show:
                plt.show()
            plt.close()
        else:
            self.logger.info("MAC data is empty.")

    def _plot_boolean(self, df: pd.DataFrame) -> None:
        """Plot boolean datasets like Monomorphic, Biallelic, Thin Loci, Singleton, Linked.

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
            fig.savefig(of)

            if self.show:
                plt.show()
            plt.close()

        else:
            self.logger.info("Boolean data is empty.")

    def plot_filter_report(self, df: pd.DataFrame) -> None:
        """
        Plot the filter report.

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

            ax = sns_method(**kwargs, ax=ax)

        plot_format = plot_format.lower()
        of = self.output_dir / f"filter_report.{self.plot_format}"
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
            ax = sns.barplot(x=data.index, y=data.values, color=color, ax=ax)
            median_line = ax.axhline(median, color=median_color, linestyle="--")

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(labels=ax.get_xticklabels(), minor=False, rotation=90)
            ax.set_title("Population Counts")
            ax.set_xlabel("Population ID")
            ax.set_ylabel(ylabel)
            ax.legend([median_line], ["Median"], loc="upper right")

        of = self.output_dir / f"population_counts.{self.plot_format}"
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
            >>> gd = VCFReader("example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz", popmap="example_data/popmaps/phylogen_nomx.popmap", force_popmap=True, verbose=True)
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
