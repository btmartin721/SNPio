import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from snpio import GenotypeEncoder, Plotting
from snpio.popgenstats.amova import AMOVA
from snpio.popgenstats.d_statistics import DStatistics
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

    def calculate_d_statistics(
        self,
        method: str,
        population1: str | List[str],
        population2: str | List[str],
        population3: str | List[str],
        outgroup: str | List[str],
        population4: Optional[str | List[str]] = None,
        include_heterozygous: bool = False,
        num_bootstraps: int = 1000,
        n_jobs: int = -1,
        max_individuals_per_pop: Optional[int] = None,
        individual_selection: str | dict = "random",
        output_file: Optional[str] = None,
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
            population4 (str | List[str], optional): Population ID or list of IDs. Required for "partitioned" and "dfoil" methods.
            include_heterozygous (bool): Whether to include heterozygous genotypes. Defaults to False.
            num_bootstraps (int): Number of bootstrap replicates.
            n_jobs (int): Number of parallel jobs. -1 uses all available cores. Defaults to -1.
            max_individuals_per_pop (Optional[int]): Max individuals per population. Defaults to None. If specified, will select individuals from each population based on the criteria specified in ``individual_selection``.
            individual_selection (str | Dict[str, List[str]]): Method for selecting individuals. Defaults to "random". If max_individuals_per_pop is specified, can be a dictionary with population IDs as keys and lists of selected individuals as values.
            output_file (Optional[str]): Path to save the results CSV file. If not specified, results will be saved to a default location. Defaults to None.
            save_plot (bool): Whether to save the plots D-statistic plots. Defaults to True.

        Returns:
            Tuple: A tuple containing the results of all sample combinations as a pandas DataFrame and the overall mean D-statistic, Z-score, and P-value as a dictionary.
        """
        self.logger.info(f"Calculating {method.capitalize()} D-statistics...")

        def get_population_indices(population: str | List[str]) -> List[int]:
            """Retrieve sample indices for a population or list of populations.

            Args:
                population (Union[str, List[str]]): Population ID or list of IDs.

            Returns:
                List[int]: List of sample indices.

            Raises:
                ValueError: If a population ID is not found.
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
                    self.logger.error(f"Population ID '{pop}' not found.")
                    raise ValueError(f"Population ID '{pop}' not found.")

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
        correction_method: Optional[str] = None,
        alpha: float = 0.05,
        use_bootstrap: bool = False,
        n_bootstraps: int = 1000,
        n_jobs: int = 1,
        save_plot: bool = True,
    ):
        """Detect Fst outliers from SNP data using bootstrapping or DBSCAN.

        This method detects Fst outliers from SNP data using bootstrapping or DBSCAN clustering. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (str, optional): Multiple testing correction method that performs P-value adjustment, 'bonf' (Bonferroni) or 'fdr' (FDR B-H). If not specified, no correction or P-value adjustment is applied. Defaults to None.
            alpha (float): Significance level for multiple test correction (with adjusted P-values). Defaults to 0.05.
            use_bootstrap (bool): Whether to use bootstrapping to estimate variance of Fst per SNP. If False, DBSCAN clustering is used instead. Defaults to False.
            n_bootstraps (int): Number of bootstrap replicates to use for estimating variance of Fst per SNP. Defaults to 1000.
            n_jobs (int): Number of CPU threads to use for parallelization. If set to -1, all available CPU threads are used. Defaults to 1.
            save_plot (bool): Whether to save the heatmap plot of Fst outliers. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A DataFrame containing the Fst outliers and contributing population pairs, and a DataFrame containing the adjusted P-values if ``correction_method`` was provided or the un-adjusted P-values otherwise.
        """
        self.logger.info("Detecting Fst outliers...")

        if correction_method:
            correction_method = correction_method.lower()
            if correction_method not in {"bonf", "fdr"}:
                msg: str = (
                    f"Invalid correction_method. Supported options: 'bonferroni', 'fdr', but got: {correction_method}"
                )
                self.logger.error(msg)
                raise ValueError(msg)

            correction_method = "fdr_bh" if correction_method == "fdr" else "bonferroni"

        if alpha < 0 or alpha >= 1:
            msg: str = (
                f"Invalid alpha value. Must be in the range [0, 1), but got: {alpha}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if n_bootstraps < 1 or not isinstance(n_bootstraps, int):
            msg: str = (
                f"Invalid n_bootstraps value. Must be an integer greater than 0, but got: {n_bootstraps}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        tail_direction = tail_direction.lower()
        if tail_direction not in {"both", "upper", "lower"}:
            msg: str = (
                f"Invalid tail_direction. Supported options: 'both', 'upper', 'lower', but got: {tail_direction}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        alternative = "two" if tail_direction == "both" else tail_direction

        # Returns tuple of outlier SNPs and adjusted p-values.
        if use_bootstrap:
            return fo.detect_fst_outliers_bootstrap(
                correction_method,
                alpha,
                n_bootstraps,
                n_jobs,
                alternative=alternative,
            )

        return fo.detect_fst_outliers_dbscan(correction_method, alpha, n_jobs)

    @exporter.capture_results
    def summary_statistics(
        self,
        n_bootstraps=0,
        n_jobs=1,
        save_plots: bool = True,
        use_pvalues: bool = False,
    ):
        """Calculate a suite of summary statistics for SNP data.

        This method calculates a suite of summary statistics for SNP data, including observed heterozygosity (Ho), expected heterozygosity (He), nucleotide diversity (Pi), and Fst between populations. Summary statistics are calculated both overall and per population.

        Args:
            n_bootstraps (int): Number of bootstrap replicates to use for estimating variance of Fst per SNP. If 0, then bootstrapping is not used and confidence intervals are estimated from the data. Defaults to 0.
            n_jobs (int): Number of parallel jobs. If set to -1, all available CPU threads are used. Defaults to 1.
            save_plots (bool): Whether to save plots of the summary statistics. In any case, a dictionary of summary statistics is returned. Defaults to True.
            use_pvalues (bool): Whether to calculate p-values for Fst. Otherwise calculates 95% confidence intervals. Defaults to False.

        Returns:
            dict: A dictionary containing summary statistics per population and overall.
        """
        summary_stats = SummaryStatistics(
            self.genotype_data, self.alignment_012, self.logger, self.plotter
        )
        return summary_stats.calculate_summary_statistics(
            n_bootstraps=n_bootstraps,
            n_jobs=n_jobs,
            save_plots=save_plots,
            use_pvalues=use_pvalues,
        )

    @exporter.capture_results
    def amova(
        self,
        regionmap: Optional[Dict[str, str]] = None,
        n_bootstraps: int = 0,
        n_jobs: int = 1,
        random_seed: Optional[int] = None,
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
            n_bootstraps (int): Number of bootstrap replicates across SNP loci.
            n_jobs (int): Number of parallel jobs. -1 uses all cores.
            random_seed (int, optional): Random seed for reproducibility.

        Returns:
            dict: AMOVA results (variance components, Phi statistics, and possibly p-values).

        Raises:
            ValueError: If regionmap is required but not provided.
        """
        amova_instance = AMOVA(self.genotype_data, self.alignment, self.logger)
        return amova_instance.run(
            regionmap=regionmap,
            n_bootstraps=n_bootstraps,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )

    def tajimas_d(self) -> pd.Series:
        """
        Calculate Tajima's D for each locus.

        Tajima's D is a measure of the difference between two estimates of genetic diversity: the average number of pairwise differences (π) and Watterson's θ (based on segregating sites). A significant deviation of Tajima's D from zero suggests non-neutral evolution.

        Returns:
            pd.Series: Tajima's D values for each locus.
        """
        self.logger.info("Calculating Tajima's D...")

        # Step 1: Set up alignment data
        alignment = self.alignment_012.astype(float).copy()
        alignment[alignment == -9] = np.nan  # Replace missing data with NaN

        # Step 2: Calculate nucleotide diversity (π)
        pi = self.nucleotide_diversity()

        # Step 3: Calculate the number of segregating sites (S)
        S = np.array(
            [np.sum(np.unique(col[~np.isnan(col)]) > 0) for col in alignment.T]
        )

        # Step 4: Calculate Watterson's theta
        n_samples = np.sum(~np.isnan(alignment), axis=0)

        # Filter out loci with all missing data
        a1 = np.array(
            [np.sum(1.0 / np.arange(1, n + 1)) if n > 0 else 0 for n in n_samples]
        )
        a2 = np.array(
            [
                np.sum(1.0 / (np.arange(1, n + 1) ** 2)) if n > 0 else 0
                for n in n_samples
            ]
        )
        a1 = np.array(
            [np.sum(1.0 / np.arange(1, n + 1)) if n > 0 else 0 for n in n_samples]
        )
        a2 = np.array(
            [
                np.sum(1.0 / (np.arange(1, n + 1) ** 2)) if n > 0 else 0
                for n in n_samples
            ]
        )

        theta = np.divide(S, a1, where=a1 > 0)
        theta[a1 == 0] = np.nan  # Handle invalid cases

        # Step 5: Calculate variance and constants for Tajima's D
        b1 = (n_samples - 1) / (2 * n_samples)
        b2 = (n_samples - 1) * (2 * n_samples + 1) / (6 * n_samples**2)
        c1 = b1 - (1 / a1)
        c2 = b2 - (n_samples + 2) / (a1 * n_samples) + a2 / (a1**2)

        e1 = np.divide(c1, a1, where=a1 > 0)
        e2 = np.divide(c2, a1**2 + a2, where=(a1**2 + a2) > 0)

        # Step 6: Calculate Tajima's D
        variance_term = e1 * S + e2 * S * (S - 1)
        variance = np.sqrt(variance_term)

        # Set variance to NaN where S is zero or one
        variance[(S == 0) | (S == 1)] = np.nan
        tajimas_d = np.divide(pi - theta, variance, where=variance > 0)

        # Handle invalid cases
        tajimas_d[(variance == 0) | (n_samples <= 1)] = np.nan
        # Handle invalid cases
        tajimas_d[(variance == 0) | (n_samples <= 1)] = np.nan

        tajimas_d = pd.Series(tajimas_d, name="Tajima's D")

        self.logger.info("Tajima's D calculation complete!")

        return tajimas_d

    def neis_genetic_distance(self) -> pd.DataFrame:
        """Calculate Nei's genetic distance between all pairs of populations.

        Nei's genetic distance is a measure of genetic divergence between populations. It is calculated based on the allele frequencies at each locus. A higher value indicates greater genetic distance, or differentiation, between populations.

        Returns:
            pd.DataFrame: A DataFrame where each cell (i, j) represents Nei's genetic distance between populations i and j.
        """
        # Step 1: Get allele frequencies per population
        allele_freqs_per_pop = self._calculate_allele_frequencies()

        # Step 2: Initialize DataFrame to store Nei's genetic distance for each population pair
        populations = list(allele_freqs_per_pop.keys())
        dist_matrix = pd.DataFrame(index=populations, columns=populations, dtype=float)

        # Step 3: Calculate Nei's genetic distance for each pair of populations
        for i, pop1 in enumerate(populations):
            freqs_pop1 = allele_freqs_per_pop[pop1]

            # Compare each population with every other population, including itself
            for j, pop2 in enumerate(populations):
                if i == j:
                    dist_matrix.loc[pop1, pop2] = (
                        0.0  # Genetic distance with itself is zero
                    )
                else:
                    freqs_pop2 = allele_freqs_per_pop[pop2]

                    # Calculate Nei's genetic distance across loci
                    neis_distance = self._calculate_neis_distance_between_pops(
                        freqs_pop1, freqs_pop2
                    )

                    # Set distance symmetrically
                    dist_matrix.loc[pop1, pop2] = neis_distance
                    dist_matrix.loc[pop2, pop1] = neis_distance

        # Ensure all expected indices are in the DataFrame
        dist_matrix = dist_matrix.reindex(index=populations, columns=populations)

        return dist_matrix

    def _calculate_allele_frequencies(self) -> dict:
        """
        Helper method to calculate allele frequencies for each population at each locus.

        Returns:
            dict: A dictionary where keys are population IDs and values are arrays of allele
            frequencies per locus.
        """
        pop_indices = self.genotype_data.get_population_indices()
        allele_freqs_per_pop = {}

        for pop_id, indices in pop_indices.items():
            # Subset alignment for population
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan  # Replace missing data

            # Calculate allele frequencies
            # Since diploid
            total_alleles = 2 * np.sum(~np.isnan(pop_alignment), axis=0)
            alt_allele_counts = np.nansum(pop_alignment, axis=0)

            # Avoid division by zero
            freqs = np.divide(alt_allele_counts, total_alleles, where=total_alleles > 0)

            allele_freqs_per_pop[pop_id] = freqs

        return allele_freqs_per_pop

    def _calculate_neis_distance_between_pops(
        self, freqs_pop1: np.ndarray, freqs_pop2: np.ndarray
    ) -> float:
        """
        Helper method to calculate Nei's genetic distance between two populations based on
        their allele frequencies.

        Args:
            freqs_pop1 (np.ndarray): Allele frequencies of population 1.
            freqs_pop2 (np.ndarray): Allele frequencies of population 2.

        Returns:
            float: Nei's genetic distance between the two populations.
        """
        # Step 1: Calculate mean allele frequencies (p-bar) across populations
        mean_freqs = (freqs_pop1 + freqs_pop2) / 2

        # Step 2: Calculate gene diversity within each population
        h_pop1 = 1 - np.nansum(freqs_pop1**2 + (1 - freqs_pop1) ** 2)
        h_pop2 = 1 - np.nansum(freqs_pop2**2 + (1 - freqs_pop2) ** 2)

        # Step 3: Calculate mean gene diversity across populations
        h_mean = 1 - np.nansum(mean_freqs**2 + (1 - mean_freqs) ** 2)

        # Step 4: Calculate Nei's genetic distance
        with np.errstate(divide="ignore", invalid="ignore"):
            nei_distance = -np.log((h_pop1 + h_pop2) / (2 * h_mean))

        return nei_distance
