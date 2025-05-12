from ctypes import ArgumentError
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import snpio.utils.custom_exceptions as exceptions
from snpio import GenotypeEncoder, Plotting
from snpio.popgenstats.amova import AMOVA
from snpio.popgenstats.d_statistics import DStatistics
from snpio.popgenstats.fst_outliers import FstOutliers
from snpio.popgenstats.genetic_distance import GeneticDistance
from snpio.popgenstats.summary_statistics import SummaryStatistics
from snpio.utils.logging import LoggerManager


class PopGenStatistics:
    """Class for calculating population genetics statistics from SNP data.

    This class provides methods for calculating population genetics statistics from SNP data. It is designed to work with GenotypeData objects. The PopGenStatistics class can calculate Patterson's D-statistic, partitioned D-statistic, D-foil statistic, summary statistics, and perform PCA and DAPC dimensionality reduction analysis.
    """

    def __init__(
        self, genotype_data: Any, verbose: bool = False, debug: bool = False
    ) -> None:
        """Initialize the PopGenStatistics object.

        This class provides methods for calculating population genetics statistics from SNP data. It is designed to work with GenotypeData objects. The PopGenStatistics class can calculate Patterson's D-statistic, partitioned D-statistic, D-foil statistic, summary statistics, and perform PCA and DAPC dimensionality reduction analysis.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing SNP data and metadata.
            verbose (bool): Whether to display verbose output. Defaults to False.
            debug (bool): Whether to display debug output. Defaults to False.
        """
        self.genotype_data: Any = genotype_data
        self.verbose: bool = verbose
        self.debug: bool = debug
        self.alignment: np.ndarray = genotype_data.snp_data
        self.popmap: Dict[str, str | int] = genotype_data.popmap
        self.populations: List[str | int] = genotype_data.populations

        plot_kwargs: Dict[str, Any] = genotype_data.plot_kwargs
        plot_kwargs["debug"] = debug
        plot_kwargs["verbose"] = verbose

        # Initialize plotting and dstats objects
        self.plotter: Any = Plotting(genotype_data, **plot_kwargs)

        # Initialize logger
        logman = LoggerManager(
            __name__, prefix=genotype_data.prefix, debug=debug, verbose=verbose
        )

        # Get logger object and set logging level
        level: str = "DEBUG" if debug else "INFO"
        logman.set_level(level)
        self.logger: Logger = logman.get_logger()
        self.logger.verbose = verbose

        self.d_stats = DStatistics(
            self.alignment, self.genotype_data.samples, self.logger
        )

        self.encoder = GenotypeEncoder(self.genotype_data)
        self.alignment_012: np.ndarray = self.encoder.genotypes_012.astype(np.float64)

        outdir = Path(f"{self.genotype_data.prefix}_output", "analysis")
        self.output_dir = outdir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def calculate_d_statistics(
        self,
        method: str,
        population1: str | List[str],
        population2: str | List[str],
        population3: str | List[str],
        outgroup: str | List[str],
        population4: str | List[str] | None = None,
        include_heterozygous: bool = False,
        num_bootstraps: int = 1000,
        n_jobs: int = -1,
        max_individuals_per_pop: int | None = None,
        individual_selection: str | dict = "random",
        output_file: str | None = None,
        save_plot: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate D-statistics, Z-scores, and P-values from D-statistics.

        This method calculates D-statistics, Z-scores, and P-values for all possible sample combinations based on the chosen ``method``. The results are saved to a CSV file and returned as a pandas DataFrame. The overall mean D-statistic, Z-score, and P-value are also returned as a dictionary.

        Args:
            method (str): The method to use for calculating D-statistics. Supported options include "patterson", "partitioned", "dfoil". "patterson" is the default and represents the traditional 4-taxon D-statistic. "partitioned" is a generalization of the D-statistic that allows for more than 4 populations. "dfoil" is a method that calculates D-statistics for all possible combinations of 4 populations (i.e., the FOIL method). Defaults to "patterson".
            population1 (str | List[str]): Population ID or list of IDs.
            population2 (str | List[str]): Population ID or list of IDs.
            population3 (str | List[str]): Population ID or list of IDs.
            outgroup (str | List[str]): Population ID or list of IDs.
            population4 (str | List[str] | None): Population ID or list of IDs. Required for "partitioned" and "dfoil" methods.
            include_heterozygous (bool): Whether to include heterozygous genotypes. Defaults to False.
            num_bootstraps (int): Number of bootstrap replicates.
            n_jobs (int): Number of parallel jobs. -1 uses all available cores. Defaults to -1.
            max_individuals_per_pop (int | None): Max individuals per population. Defaults to None. If specified, will select individuals from each population based on the criteria specified in ``individual_selection``.
            individual_selection (str | Dict[str, List[str]]): Method for selecting individuals. Defaults to "random". If max_individuals_per_pop is specified, can be a dictionary with population IDs as keys and lists of selected individuals as values.
            output_file (str | None): Path to save the results CSV file. If not specified, results will be saved to a default location. Defaults to None.
            save_plot (bool): Whether to save the plots D-statistic plots. Defaults to True.

        Returns:
            Tuple: A tuple containing the results of all sample combinations as a pandas DataFrame and the overall mean D-statistic, Z-score, and P-value as a dictionary.
        """
        self.logger.info(f"Calculating {method.capitalize()} D-statistics...")

        def get_population_indices(population: str | List[str]) -> List[int]:
            """Retrieve sample indices for a population or list of populations.

            Args:
                population (str | List[str]): Population ID or list of IDs.

            Returns:
                List[int]: List of sample indices.

            Raises:
                KeyError: If a population ID is not found.
                ValueError: If an invalid ``individual_selection`` method is specified.
            """
            populations: List[str] = (
                [population] if isinstance(population, str) else population
            )
            selected_samples: List[str] = []
            popmap_inverse: Dict[str, List[str | int]] = (
                self.genotype_data.popmap_inverse
            )

            for pop in populations:
                try:
                    samples: List[str | int] = popmap_inverse[pop]
                    # Limit individuals per population if specified
                    if (
                        max_individuals_per_pop
                        and len(samples) > max_individuals_per_pop
                    ):
                        if individual_selection == "random":
                            selected_samples.extend(
                                np.random.choice(
                                    samples, max_individuals_per_pop, replace=False
                                )
                            )
                        elif isinstance(individual_selection, dict):
                            selected_samples.extend(
                                individual_selection.get(
                                    pop, samples[:max_individuals_per_pop]
                                )
                            )
                        else:
                            msg: str = (
                                f"Invalid individual_selection: '{individual_selection}'."
                            )
                            self.logger.error(msg)
                            raise ValueError(msg)
                    else:
                        selected_samples.extend(samples)
                except KeyError:
                    msg = f"Population ID '{pop}' not found."
                    self.logger.error(msg)
                    raise exceptions.PopmapKeyError(msg)

            # Convert sample IDs to indices
            samples = self.genotype_data.samples
            sample_id_to_index: Dict[str | int, int] = {
                sample: idx for idx, sample in enumerate(samples)
            }
            return [
                sample_id_to_index[sample]
                for sample in selected_samples
                if sample in sample_id_to_index
            ]

        # Retrieve sample indices for each population
        d1_inds: List[int] = get_population_indices(population1)
        d2_inds: List[int] = get_population_indices(population2)
        d3_inds: List[int] = get_population_indices(population3)
        d4_inds: List[int] | None = (
            get_population_indices(population4) if population4 else None
        )
        outgroup_inds: List[int] = get_population_indices(outgroup)

        # Calculate Z-scores and P-values
        combo_z_p_values, overall_z_score, overall_p_value = (
            self.d_stats.calculate_z_and_p_values(
                d1_inds,
                d2_inds,
                d3_inds,
                outgroup_inds,
                d4_inds,
                include_heterozygous,
                num_bootstraps,
                method,
                n_jobs,
            )
        )

        # Directory and filename setup
        if output_file is None:
            outdir = Path(f"{self.genotype_data.prefix}_output", "analysis", "d_stats")
            outdir.mkdir(exist_ok=True, parents=True)
            output_file = outdir / f"{method}_dstat_results.csv"
        else:
            output_file = Path(output_file)
            output_file.parent.mkdir(exist_ok=True, parents=True)

        # Prepare columns for sample combinations
        max_combo_len: int = max(len(combo) for combo, _, _, _ in combo_z_p_values)
        combo_columns: List[str] = [f"Sample_{i+1}" for i in range(max_combo_len)]

        # Create data entries with columns for each sample in combination
        data: List[Dict[str, float]] = []
        for combo, observed_d, z, p in combo_z_p_values:
            combo_data: Dict[str, str] = {
                combo_columns[i]: str(combo[i]) if i < len(combo) else ""
                for i in range(max_combo_len)
            }
            combo_data.update(
                {
                    "Observed D-Statistic": float(observed_d),
                    "Z-Score": float(z),
                    "P-Value": float(p),
                }
            )
            data.append(combo_data)

        # Get overall mean results.
        overall_d_stat: np.floating[Any] = np.mean(
            [obs for _, obs, _, _ in combo_z_p_values]
        )
        overall_data: Dict[str, float] = {
            "Observed D-Statistic": float(overall_d_stat),
            "Z-Score": float(overall_z_score),
            "P-Value": float(overall_p_value),
        }

        # Create DataFrame from data
        df: pd.DataFrame = pd.DataFrame(data)

        # Bonferroni and FDR corrections
        df: pd.DataFrame = self._adjust_p_values(df)

        # Save results to CSV
        df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to {output_file}")

        if save_plot:
            # Plot D-statistic results
            self.plotter.plot_d_statistics(df)

        self.logger.info("D-statistics calculation complete!")

        return df, overall_data

    def _adjust_p_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust P-values using multiple testing correction methods.

        This method adjusts P-values using the Bonferroni and FDR B-H methods. The adjusted P-values are added to the DataFrame, along with significance columns for each correction.

        Args:
            df (pd.DataFrame): DataFrame containing P-values.

        Returns:
            pd.DataFrame: DataFrame with adjusted P-values and significance columns.
        """
        self.logger.info("Adjusting P-values...")

        # Make a copy of the DataFrame to avoid modifying the original
        df = df.copy()

        df["Bonferroni P-Value"] = multipletests(df["P-Value"], method="bonferroni")[1]
        df["FDR-BH P-Value"] = multipletests(df["P-Value"], method="fdr_bh")[1]

        # Add significance columns for each correction
        df["Significant (Raw)"] = df["P-Value"] < 0.05
        df["Significant (Bonferroni)"] = df["Bonferroni P-Value"] < 0.05
        df["Significant (FDR-BH)"] = df["FDR-BH P-Value"] < 0.05

        return df

    def detect_fst_outliers(
        self,
        correction_method: Literal["bonf", "fdr"] | None = None,
        alpha: float = 0.05,
        use_dbscan: bool = False,
        n_permutations: int = 1000,
        n_jobs: int = 1,
        alternative: Literal["upper", "lower", "both"] = "upper",
        seed: int = 42,
        kde_bandwidth: float | Literal["silverman", "scott"] | None = "scott",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect Fst outliers from SNP data using bootstrapping or DBSCAN.

        This method detects Fst outliers from SNP data using bootstrapping or DBSCAN clustering. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (str, optional): Multiple testing correction method that performs P-value adjustment. Shoould be either 'bonf' (Bonferroni) or 'fdr' (FDR B-H), or None. If not specified, no correction or P-value adjustment is applied. Defaults to None (no correction).
            alpha (float): Significance level for multiple test correction with adjusted P-values. Defaults to 0.05.
            use_dbscan (bool): Whether to use DBSCAN clustering to estimate Fst outliers per SNP. If False, bootstrap variance method is used instead. Defaults to False.
            n_permutations (int): Number of permutation replicates to use for estimating variance of Fst per SNP. Defaults to 1000.
            n_jobs (int): Number of parallel jobs. Only applies to DBSCAN method. -1 uses all available CPU threads. Defaults to 1.
            alternative (str): Alternative hypothesis for the Fst outlier test. Can be 'upper', 'lower', or 'both'. Defaults to 'both'.
            seed (int): Random seed for reproducibility. Defaults to 42.
            kde_bandwidth (float | Literal["scott", "silverman"] | None): Bandwidth for the Gaussian kernel density estimation used when estimating P-values. Can be a float, 'silverman' for Silverman's rule of thumb, 'scott' for Scott's rule of thumb, or None for no bandwidth. Defaults to 'scott'.

        Returns:
            pd.DataFrame: A DataFrame containing the locus indices for Fst outliers, the Fst values, contributing population pairs to each outlier, and the unadjusted and adjusted P-values (if applicable).
        """
        self.logger.info("Detecting Fst outliers...")

        fo = FstOutliers(
            self.genotype_data, self.plotter, verbose=self.verbose, debug=self.debug
        )

        # Validate inputs.
        alternative = alternative.lower()

        if alternative not in {"upper", "lower", "both"}:
            msg = f"Invalid alternative hypothesis. Supported options: 'upper', 'lower', 'both', but got: {alternative}"
            self.logger.error(msg)
            raise ValueError(msg)

        if isinstance(kde_bandwidth, str):
            kde_bandwidth = kde_bandwidth.lower()
        elif isinstance(kde_bandwidth, int):
            kde_bandwidth = float(kde_bandwidth)
            if kde_bandwidth < 0:
                msg = f"Invalid bandwidth. Must be a positive float, but got: {kde_bandwidth}"
                self.logger.error(msg)
                raise ValueError(msg)

        if (
            kde_bandwidth not in {"silverman", "scott"}
            and not isinstance(kde_bandwidth, float)
            and kde_bandwidth is not None
        ):
            msg = f"Invalid bandwidth. Supported options: 'silverman', 'scott', or float, but got: {kde_bandwidth} of type {type(kde_bandwidth)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if correction_method is not None:
            correction_method = correction_method.lower()
            if correction_method not in {"bonf", "fdr"}:
                msg = f"Invalid correction_method. Supported options: 'bonf', 'fdr', but got: {correction_method}"
                self.logger.error(msg)
                raise ValueError(msg)

            correction_method = "fdr_bh" if correction_method == "fdr" else "bonferroni"

        if alpha < 0 or alpha >= 1:
            msg = f"Invalid alpha value. Must be in the range [0, 1), but got: {alpha}"
            self.logger.error(msg)
            raise ValueError(msg)

        if n_permutations < 1 or not isinstance(n_permutations, int):
            msg = f"Invalid 'n_permutations' value. Must be an integer greater than 0, but got: {n_permutations}"
            self.logger.error(msg)
            raise exceptions.PermutationInferenceError(msg)

        method = "dbscan" if use_dbscan else "bootstrap"

        self.logger.info(
            f"Starting Fst outlier detection ({method.capitalize()} method)..."
        )

        # Returns outlier SNPs, Fst-values, and adjusted and unadjusted
        # p-values.
        if use_dbscan:
            # Use DBSCAN method for outlier detection
            df_fst_outliers = fo.detect_fst_outliers_dbscan(
                correction_method,
                alpha,
                n_jobs,
                n_bootstraps=n_permutations,
                alternative=alternative,
                seed=seed,
            )
        else:
            df_fst_outliers = fo.detect_outliers_permutation(
                n_perm=n_permutations,
                correction_method=correction_method,
                alpha=alpha,
                seed=seed,
                bandwidth=kde_bandwidth,
                alternative=alternative,
            )

        # Save the results to a CSV file
        pth = self.output_dir / f"fst_outliers_{method}_method.csv"
        df_fst_outliers.to_csv(pth, index=False, header=True)
        method = method.capitalize()
        self.logger.info(f"{method} Fst outliers saved to {pth}")
        self.logger.info(f"{method} Fst outlier detection complete!")

        return df_fst_outliers

    def summary_statistics(
        self,
        n_permutations=0,
        n_jobs=1,
        save_plots: bool = True,
        use_pvalues: bool = False,
    ):
        """Calculate a suite of summary statistics for SNP data.

        This method calculates a suite of summary statistics for SNP data, including observed heterozygosity (Ho), expected heterozygosity (He), nucleotide diversity (Pi), and Fst between populations. Summary statistics are calculated both overall and per population.

        Args:
            n_permutations (int): Number of bootstrap replicates to use for estimating variance of Fst per SNP. If 0, then bootstrapping is not used and confidence intervals are estimated from the data. Defaults to 0.
            n_jobs (int): Number of parallel jobs. If set to -1, all available CPU threads are used. Defaults to 1.
            save_plots (bool): Whether to save plots of the summary statistics. In any case, a dictionary of summary statistics is returned. Defaults to True.
            use_pvalues (bool): Whether to calculate p-values for Fst. Otherwise calculates 95% confidence intervals. Defaults to False.

        Returns:
            dict: A dictionary containing summary statistics per population and overall.
        """
        summary_stats = SummaryStatistics(
            self.genotype_data,
            self.alignment_012,
            self.plotter,
            verbose=self.verbose,
            debug=self.debug,
        )

        return summary_stats.calculate_summary_statistics(
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            save_plots=save_plots,
            use_pvalues=use_pvalues,
        )

    def amova(
        self,
        regionmap: Dict[str, str] | None = None,
        n_permutations: int = 0,
        n_jobs: int = 1,
        random_seed: int | None = None,
    ) -> Dict[str, float]:
        """Conduct AMOVA (Analysis Of Molecular Variance).

        AMOVA (Analysis of Molecular Variance) is a method for partitioning genetic variation into components due to differences among populations, among individuals within populations, and within individuals. This method calculates variance components and Phi statistics for a given number of hierarchical levels (1, 2, or 3). If bootstrapping is enabled, it also estimates p-values for the variance components. The number of hierarchical levels determines the structure of the AMOVA model: 1 => populations only, 2 => region -> populations, 3 => region -> population -> individuals. If regionmap is provided, it is used to map populations to regions.

        Notes:
            - Algorithm adapted from the R package 'poppr' (Kamvar et al., 2014).
            - The Phi statistic is a measure of genetic differentiation between populations.
            - Algorithm description: First, the total variance is calculated. Then, the variance components are calculated by summing the squared differences between the mean of each group and the global mean. The Phi statistic is calculated as the ratio of the among-group variance to the total variance. P-values are estimated by bootstrapping.

        Args:
            regionmap (dict, optional): Mapping from population_id -> region_id.
                If None but hierarchical_levels>1, raises ValueError.
            n_permutations (int): Number of bootstrap replicates across SNP loci.
            n_jobs (int): Number of parallel jobs. -1 uses all cores.
            random_seed (int | None): Random seed for reproducibility.

        Returns:
            dict: AMOVA results (variance components, Phi statistics, and possibly p-values).

        Raises:
            ValueError: If regionmap is required but not provided.
        """
        amova_instance = AMOVA(self.genotype_data, self.alignment, self.logger)
        return amova_instance.run(
            regionmap=regionmap,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )

    def neis_genetic_distance(
        self,
        n_permutations: int = 0,
        n_jobs: int = 1,
        use_pvalues: bool = False,
        palette: str = "magma",
        supress_plot: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate Nei's genetic distance between all pairs of populations.

        Optionally computes bootstrap-based p-values for each population pair if n_permutations > 0.

        Nei's genetic distance is defined as ``D = -ln( Ī )``, where Ī is the ratio of the average genetic identity to the geometric mean of the average homozygosities.

        Args:
            n_permutations (int): Number of bootstrap replicates to compute p-values. Defaults to 0 (only distances are returned).
            n_jobs (int): Number of parallel jobs. -1 uses all cores. Defaults to 1.
            use_pvalues (bool): If True, returns a tuple of (distance matrix, p-value matrix). Defaults to False.
            palette (str): Color palette for the distance matrix plot. Can use any matplotlib gradient-based palette. Some frequently used options include: "coolwarm", "viridis", "magma", and "inferno". Defaults to 'coolwarm'.
            supress_plot (bool): If True, suppresses the plotting of the distance matrix. Defaults to False.

        Returns:
            pd.DataFrame: If n_permutations == 0, returns a DataFrame of Nei's distances.
            Tuple[pd.DataFrame, pd.DataFrame]: If n_permutations > 0, returns a tuple of (distance matrix, p-value matrix).
        """
        gd = GeneticDistance(
            self.genotype_data, self.plotter, verbose=self.verbose, debug=self.debug
        )
        self.logger.info("Calculating Nei's genetic distance...")
        self.logger.info(f"Number of bootstraps: {n_permutations}")

        nei_results = gd.nei_distance(
            n_permutations=n_permutations, n_jobs=n_jobs, return_pvalues=use_pvalues
        )

        df_obs, df_lower, df_upper, df_pval = gd.parse_nei_result(nei_results)

        if not supress_plot:
            self.plotter.plot_dist_matrix(
                df_obs,
                pvals=df_pval if use_pvalues else None,
                palette=palette,
                title="Nei's Genetic Distance",
                dist_type="nei",
            )

        self.logger.info("Nei's genetic distance calculation complete!")
        return (df_obs, df_pval) if use_pvalues else df_obs
