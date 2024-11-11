import itertools
from logging import Logger
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from snpio import GenotypeEncoder, Plotting
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
        self.alignment_012: np.ndarray = self.encoder.genotypes_012

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

        func = self._bootstrap_fst if use_bootstrap else self._dbscan_fst
        args: List[Any] = [correction_method, alpha]

        if use_bootstrap:
            # Step 1: Calculate bootstrapped Fst values between population pairs
            args.extend([n_bootstraps, n_jobs])

        outlier_snps, contributing_pairs, adjusted_pvals_df = func(*args)

        # Add contributing pairs to the outlier_snps DataFrame
        outlier_snps["Contributing_Pairs"] = contributing_pairs

        self.logger.debug(f"{outlier_snps=}")

        if outlier_snps.empty:
            self.logger.info("No Fst outliers detected. Skipping correspoding plot.")
        else:
            self.logger.info(f"{len(outlier_snps)} Fst outliers detected.")

            # Plot the outlier SNPs
            self.plotter.plot_fst_outliers(outlier_snps)

        self.logger.info("Fst outlier detection complete!")

        # Return outlier SNPs
        return outlier_snps, adjusted_pvals_df

    def _dbscan_fst(
        self, correction_method: Optional[str], alpha: float
    ) -> Tuple[pd.DataFrame, List[Tuple[str]], pd.DataFrame]:
        """Detect Fst outliers using DBSCAN clustering.

        This method detects Fst outliers from SNP data using DBSCAN clustering. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (str): Multiple testing correction method.
            alpha (float): Significance level for multiple test correction.

        Returns:
            Tuple[pd.DataFrame, List[Tuple[str]], pd.DataFrame]: A DataFrame containing the Fst outliers, a list of contributing population pairs, and a DataFrame containing the adjusted P-values.
        """
        self.logger.info("Detecting Fst outliers using DBSCAN...")

        # Step 1: Calculate Fst values between population pairs
        fst_per_population_pair = self.weir_cockerham_fst_between_populations()

        # Step 2: Combine Fst values into a DataFrame
        fst_df: pd.DataFrame = pd.DataFrame.from_dict(
            {str(k): v.values for k, v in fst_per_population_pair.items()},
            orient="columns",
        )

        # Optionally, add SNP identifiers as index if available
        fst_df.index = np.arange(len(fst_df))

        # Handle missing values
        fst_df.dropna(inplace=True)

        # Step 3: Prepare data for DBSCAN
        fst_values = fst_df.to_numpy()
        n_population_pairs = fst_values.shape[1]
        n_data_points = fst_values.shape[0]
        min_samples: int = min(max(2, 2 * n_population_pairs), n_data_points - 1)

        # Scale the data
        scaler = StandardScaler()
        fst_values_scaled = scaler.fit_transform(fst_values)

        # Step 4: Estimate optimal eps
        eps: float = self._estimate_eps(fst_values_scaled, min_samples)

        # Step 5: Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(fst_values_scaled)

        # Step 6: Identify outliers
        outlier_indices = np.where(labels == -1)[0]
        outlier_snps: pd.DataFrame = fst_df.iloc[outlier_indices]

        # Step 7: Identify contributing population pairs with multiple testing correction
        # Compute mean and std of Fst values per population pair across all SNPs
        fst_means = fst_df.mean()
        fst_stds = fst_df.std()

        # Avoid division by zero
        fst_stds = fst_stds.replace(0, np.finfo(float).eps)

        # For each outlier SNP, compute z-scores
        z_scores = (outlier_snps - fst_means) / fst_stds

        # Compute two-tailed p-values for z-scores
        p_values = z_scores.map(
            lambda z: 2 * (1 - norm.cdf(abs(z))) if not pd.isnull(z) else np.nan
        )

        # Flatten all p-values into one array, excluding NaNs
        all_p_values: np.ndarray = p_values.to_numpy().flatten()
        valid_indices = ~np.isnan(all_p_values)
        all_p_values_valid = all_p_values[valid_indices]

        if correction_method:
            if not all_p_values_valid.size > 0:
                self.logger.warning(
                    "No valid p-values found. Skipping multiple testing correction."
                )
                self.logger.debug(f"All p-values: {all_p_values}")
                self.logger.debug(f"Valid indices: {valid_indices}")
                self.logger.debug(f"Valid p-values: {all_p_values_valid}")

                adjusted_pvals_valid = all_p_values_valid
            else:
                # Apply multiple testing correction
                adjusted_pvals_valid = multipletests(
                    all_p_values_valid, method=correction_method
                )[1]
        else:
            adjusted_pvals_valid = all_p_values_valid

            # Create adjusted p-values DataFrame
        adjusted_pvals = np.full_like(all_p_values, np.nan)
        adjusted_pvals[valid_indices] = adjusted_pvals_valid

        # Reshape adjusted_pvals to match p_values DataFrame
        adjusted_pvals_df = pd.DataFrame(
            adjusted_pvals.reshape(p_values.shape),
            index=p_values.index,
            columns=p_values.columns,
        )

        # Identify population pairs contributing to outlier status
        contributing_pairs = []
        for snp in adjusted_pvals_df.index:
            adjusted_pvals_row = adjusted_pvals_df.loc[snp]
            significant_pairs = adjusted_pvals_row[adjusted_pvals_row < alpha]
            significant_pairs = significant_pairs.index.tolist()
            contributing_pairs.append(significant_pairs)

        return outlier_snps, contributing_pairs, adjusted_pvals_df

    def _bootstrap_fst(
        self,
        correction_method: Optional[str],
        alpha: float,
        n_bootstraps: int,
        n_jobs: int,
    ) -> Tuple[pd.DataFrame, List[Tuple[str]], pd.DataFrame]:
        """Detect Fst outliers using bootstrapping approach.

        This method detects Fst outliers from SNP data using bootstrapping. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (str): Multiple testing correction method.
            alpha (float): Significance level for multiple test correction.
            n_bootstraps (int): Number of bootstrap replicates.
            n_jobs (int): Number of parallel jobs. If set to -1, all available CPU threads are used.

        Returns:
            Tuple[pd.DataFrame, List[Tuple[str]], pd.DataFrame]: A DataFrame containing the Fst outliers, a list of contributing population pairs, and a DataFrame containing the adjusted P-values.
        """
        self.logger.info("Detecting Fst outliers using bootstrapping...")

        # Step 1: Calculate bootstrapped Fst values between population pairs
        fst_bootstrap_per_population_pair = self.weir_cockerham_fst_between_populations(
            n_bootstraps=n_bootstraps, n_jobs=n_jobs
        )

        # Compute mean and std of Fst per SNP over bootstraps
        fst_means = {}
        fst_stds = {}
        for pop_pair, fst_bootstrap in fst_bootstrap_per_population_pair.items():
            fst_means[pop_pair] = np.nanmean(fst_bootstrap, axis=1)
            fst_stds[pop_pair] = np.nanstd(fst_bootstrap, axis=1)

        # Prepare DataFrames
        fst_mean_df = pd.DataFrame(fst_means)
        fst_std_df = pd.DataFrame(fst_stds)

        # Optionally, add SNP identifiers as index if available
        fst_mean_df.index = np.arange(len(fst_mean_df))
        fst_std_df.index = fst_mean_df.index

        # Determine the threshold for acceptable missing data
        missing_threshold = 0.2  # Allow up to 20% missing data per SNP

        # Calculate the proportion of missing data per SNP
        missing_proportion = fst_mean_df.isna().mean(axis=1)

        # Identify SNPs with acceptable missing data
        valid_snps = missing_proportion <= missing_threshold

        # Filter DataFrames to retain only valid SNPs
        fst_mean_df: pd.DataFrame = fst_mean_df[valid_snps]
        fst_std_df: pd.DataFrame = fst_std_df.loc[valid_snps]

        # Impute remaining NaNs with the mean of each column
        fst_mean_df = fst_mean_df.fillna(fst_mean_df.mean())
        fst_std_df = fst_std_df.fillna(fst_std_df.mean())

        self.logger.debug(f"fst_mean_df shape: {fst_mean_df}")
        self.logger.debug(f"fst_std_df shape: {fst_std_df}")

        # Handle missing values
        fst_mean_df = fst_mean_df.dropna()
        fst_std_df = fst_std_df.loc[fst_mean_df.index]

        self.logger.debug(f"fst_mean_df shape: {fst_mean_df}")
        self.logger.debug(f"fst_std_df shape: {fst_std_df}")

        # Use observed Fst values (from non-bootstrapped data)
        fst_per_population_pair = self.weir_cockerham_fst_between_populations()

        fst_df = pd.DataFrame.from_dict(
            {k: v.values for k, v in fst_per_population_pair.items()},
            orient="columns",
        )

        self.logger.debug(f"fst_mean_df shape: {fst_mean_df}")
        self.logger.debug(f"fst_std_df shape: {fst_std_df}")
        self.logger.debug(f"fst_df shape: {fst_df}")

        fst_df.index = np.arange(len(fst_df))
        fst_df: pd.DataFrame = fst_df.loc[fst_mean_df.index]  # Align indices

        self.logger.debug(f"fst_mean_df shape: {fst_mean_df}")
        self.logger.debug(f"fst_std_df shape: {fst_std_df}")
        self.logger.debug(f"fst_df shape: {fst_df}")

        # Ensure columns are aligned and in the same order
        common_columns = fst_df.columns.intersection(fst_mean_df.columns)

        # Reindex DataFrames to have the same columns
        fst_df = fst_df[common_columns]
        fst_mean_df = fst_mean_df[common_columns]
        fst_std_df = fst_std_df[common_columns]

        # Impute remaining NaNs with the mean of each column
        fst_df = fst_df.fillna(fst_df.mean())

        self.logger.debug(f"fst_mean_df shape: {fst_mean_df}")
        self.logger.debug(f"fst_std_df shape: {fst_std_df}")
        self.logger.debug(f"fst_df shape: {fst_df}")

        # Replace zero standard deviations with a small positive value
        # Avoids division by zero when computing Z-scores
        fst_std_df = fst_std_df.replace(0, np.finfo(float).eps)

        # Compute Z-scores
        z_scores = (fst_df - fst_mean_df) / fst_std_df

        # Compute two-tailed p-values for z-scores
        p_values = z_scores.map(
            lambda z: 2 * (1 - norm.cdf(abs(z))) if not pd.isnull(z) else np.nan
        )

        self.logger.debug(f"{p_values=}")
        self.logger.debug(f"{p_values.shape=}")
        self.logger.debug(f"{z_scores=}")
        self.logger.debug(f"{z_scores.shape=}")

        # Flatten all p-values into one array, excluding NaNs
        all_p_values = p_values.values.flatten()
        valid_indices = ~np.isnan(all_p_values)
        all_p_values_valid = all_p_values[valid_indices]

        if correction_method:
            # Apply multiple testing correction
            adjusted_pvals_valid = multipletests(
                all_p_values_valid, method=correction_method
            )[1]
        else:
            adjusted_pvals_valid = all_p_values_valid

        # Create adjusted p-values DataFrame
        adjusted_pvals = np.full_like(all_p_values, np.nan)
        adjusted_pvals[valid_indices] = adjusted_pvals_valid

        # Reshape adjusted_pvals to match p_values DataFrame
        adjusted_pvals_df = pd.DataFrame(
            adjusted_pvals.reshape(p_values.shape),
            index=p_values.index,
            columns=p_values.columns,
        )

        # Identify significant SNPs and contributing pairs
        significant: pd.DataFrame = adjusted_pvals_df < alpha

        # Get outlier SNPs
        outlier_indices = significant.any(axis=1)
        outlier_snps: pd.DataFrame = fst_df.loc[outlier_indices]

        # Identify contributing population pairs
        contributing_pairs = []
        for snp in adjusted_pvals_df.index[outlier_indices]:
            adjusted_pvals_row = adjusted_pvals_df.loc[snp]
            significant_pairs = adjusted_pvals_row[
                adjusted_pvals_row < alpha
            ].index.tolist()
            contributing_pairs.append(significant_pairs)

        # Return outlier SNPs
        return outlier_snps, contributing_pairs, adjusted_pvals_df

    def _estimate_eps(self, fst_values: pd.DataFrame, min_samples: int) -> float:
        """Estimate the optimal eps value for DBSCAN clustering.

        This method estimates the optimal eps value for DBSCAN clustering using the k-distance graph method. The optimal eps value is determined based on the knee point of the k-distance graph.

        Args:
            fst_values (pd.DataFrame): Fst values between population pairs.
            min_samples (int): Minimum number of samples required for DBSCAN clustering.

        Returns:
            float: The optimal eps value for DBSCAN clustering.
        """
        # Step 4: Compute the k-distance graph to find optimal eps
        neighbor = NearestNeighbors(n_neighbors=min_samples)
        nbrs: NearestNeighbors = neighbor.fit(fst_values)
        distances, _ = nbrs.kneighbors(fst_values)

        # Sort the distances to the min_samples-th nearest neighbor
        distances_k = np.sort(distances[:, -1])

        # Check if distances are all zeros
        if np.all(distances_k == 0):
            self.logger.warning(
                "Distances all zeros. Setting eps to a small positive value."
            )
            eps = 0.1  # Adjust as appropriate
        else:
            # Use KneeLocator to find the knee point
            kneedle = KneeLocator(
                range(len(distances_k)),
                distances_k,
                S=1.0,
                curve="convex",
                direction="increasing",
            )

            if kneedle.knee is not None:
                eps = distances_k[kneedle.knee]
            else:
                # Fallback: Use a percentile of distances
                eps: np.floating[Any] = np.percentile(distances_k, 10)
                self.logger.warning(
                    f"KneeLocator did not find a knee. Using eps from 10th percentile: {eps}"
                )

        if isinstance(eps, (np.float32, np.float64)):
            eps = float(eps)

        return eps

    def observed_heterozygosity(self):
        """Calculate observed heterozygosity (Ho) for each locus.

        This method calculates observed heterozygosity (Ho) for each locus in the SNP data. Ho is defined as the proportion of heterozygous individuals at a given locus.

        Returns:
            pd.Series: A pandas Series containing the observed heterozygosity values.
        """
        alignment = self.alignment_012.astype(float).copy()
        alignment[alignment == -9] = np.nan  # Replace missing data with NaN

        # Number of non-missing individuals per locus
        n_individuals = np.sum(~np.isnan(alignment), axis=0)

        ho = (alignment == 1).sum(axis=0) / n_individuals
        return ho

    def expected_heterozygosity(self):
        """Calculate expected heterozygosity (He) for each locus.

        This method calculates expected heterozygosity (He) for each locus in the SNP data. He is defined as the expected proportion of heterozygous individuals at a given locus under Hardy-Weinberg equilibrium.

        Returns:
            pd.Series: A pandas Series containing the expected heterozygosity values.
        """
        alignment = self.alignment_012.astype(float).copy()
        alignment[alignment == -9] = np.nan  # Replace missing data with NaN

        # Number of non-missing individuals per locus
        n_individuals = np.sum(~np.isnan(alignment), axis=0)

        # Sum of alternate alleles at each locus
        alt_allele_counts = np.nansum(alignment, axis=0)
        total_alleles = 2 * n_individuals  # Since diploidy

        # Frequency of the alternate allele (p)
        p = alt_allele_counts / total_alleles

        # Frequency of the reference allele (q)
        q = 1 - p

        # Expected heterozygosity (He = 2pq)
        he = 2 * p * q

        return he

    def nucleotide_diversity(self):
        """Calculate nucleotide diversity (Pi) for each locus.

        This method calculates nucleotide diversity (Pi) for each locus in the SNP data. Pi is defined as the average number of nucleotide differences per site between two sequences.

        Returns:
            pd.Series: A pandas Series containing the nucleotide diversity values.
        """
        alignment = self.alignment_012.astype(float).copy()
        alignment[alignment == -9] = np.nan  # Replace missing data with NaN

        # Number of non-missing individuals per locus
        n = np.sum(~np.isnan(alignment), axis=0)
        he = self.expected_heterozygosity()

        with np.errstate(divide="ignore", invalid="ignore"):
            pi = (n / (n - 1)) * he  # Bias correction for sample size
            pi[n <= 1] = np.nan  # Handle where n <= 1; avoid division by zero

        return pi

    def wrights_fst(self):
        """Calculate Wright's fixation index (Fst) for each locus.

        This method calculates Wright's fixation index (Fst) for each locus in the SNP data. Fst is a measure of population differentiation due to genetic structure. Fst values range from 0 to 1, where 0 indicates no genetic differentiation and 1 indicates complete differentiation.

        Returns:
            pd.Series: A pandas Series containing the Fst values.
        """
        ho = self.observed_heterozygosity()
        he = self.expected_heterozygosity()

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            fst = (he - ho) / he
            fst[he == 0] = np.nan  # Set Fst to NaN where He is zero

        return fst

    def summary_statistics(self, save_plots: bool = True):
        """Calculate a suite of summary statistics for SNP data.

        This method calculates a suite of summary statistics for SNP data, including observed heterozygosity (Ho), expected heterozygosity (He), nucleotide diversity (Pi), and Fst between populations. Summary statistics are calculated both overall and per population.

        Args:
            save_plots (bool): Whether to save plots of the summary statistics. In any case, a dictionary of summary statistics is returned. Defaults to True.

        Returns:
            dict: A dictionary containing summary statistics per population and overall.
        """
        self.logger.info("Calculating summary statistics...")

        # Overall statistics
        ho_overall = pd.Series(self.observed_heterozygosity())
        he_overall = pd.Series(self.expected_heterozygosity())
        pi_overall = pd.Series(self.nucleotide_diversity())

        # Per-population statistics
        ho_per_population = self.observed_heterozygosity_per_population()
        he_per_population = self.expected_heterozygosity_per_population()
        pi_per_population = self.nucleotide_diversity_per_population()

        summary_stats = {
            "overall": pd.DataFrame(
                {"Ho": ho_overall, "He": he_overall, "Pi": pi_overall}
            ),
            "per_population": {},
        }

        for pop_id in ho_per_population.keys():
            summary_stats["per_population"][pop_id] = pd.DataFrame(
                {
                    "Ho": ho_per_population[pop_id],
                    "He": he_per_population[pop_id],
                    "Pi": pi_per_population[pop_id],
                }
            )

        # Fst between populations
        fst_between_pops = self.weir_cockerham_fst_between_populations()
        summary_stats["Fst_between_populations"] = fst_between_pops

        if save_plots:
            self.plotter.plot_summary_statistics(summary_stats)

        self.logger.info("Summary statistics calculation complete!")

        return summary_stats

    def observed_heterozygosity_per_population(self):
        """Calculate observed heterozygosity (Ho) for each locus per population.

        This method calculates observed heterozygosity (Ho) for each locus in the SNP data per population. Ho is defined as the proportion of heterozygous individuals at a given locus within a population.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the observed heterozygosity values per locus for that population.
        """
        pop_indices = self.genotype_data.get_population_indices()
        ho_per_population = {}

        for pop_id, indices in pop_indices.items():
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan  # Replace missing data

            if pop_alignment.shape[0] == 0 or np.all(np.isnan(pop_alignment)):
                continue  # Skip populations with no data

            # Number of non-missing individuals per locus
            n_individuals = np.sum(~np.isnan(pop_alignment), axis=0)

            # Number of heterozygotes per locus
            num_heterozygotes = np.nansum(pop_alignment == 1, axis=0)

            with np.errstate(divide="ignore", invalid="ignore"):
                ho = num_heterozygotes / n_individuals

            ho_per_population[pop_id] = pd.Series(ho)

        return ho_per_population

    def expected_heterozygosity_per_population(self, return_n: bool = False):
        """Calculate expected heterozygosity (He) for each locus per population.

        This method calculates expected heterozygosity (He) for each locus in the SNP data per population. He is defined as the expected proportion of heterozygous individuals at a given locus under Hardy-Weinberg equilibrium.

        Args:
            return_n (bool): If True, also return the number of non-missing individuals per locus.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the expected heterozygosity values per locus for that population.
            If return_n is True, returns a tuple (he, n_individuals) per population.
        """
        pop_indices = self.genotype_data.get_population_indices()
        he_per_population = {}

        for pop_id, indices in pop_indices.items():
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan  # Replace missing data

            if pop_alignment.shape[0] == 0 or np.all(np.isnan(pop_alignment)):
                continue  # Skip populations with no data

            # Number of non-missing individuals per locus
            n_individuals = np.sum(~np.isnan(pop_alignment), axis=0)
            total_alleles = 2 * n_individuals  # Adjust for missing data

            # Sum of alternate alleles at each locus
            alt_allele_counts = np.nansum(pop_alignment, axis=0)

            with np.errstate(divide="ignore", invalid="ignore"):
                # Frequency of the alternate allele (p)
                p = alt_allele_counts / total_alleles

            # Frequency of the reference allele (q)
            q = 1 - p

            # Expected heterozygosity (He = 2pq)
            he = 2 * p * q

            if return_n:
                he_per_population[pop_id] = (pd.Series(he), n_individuals)
            else:
                he_per_population[pop_id] = pd.Series(he)

        return he_per_population

    def nucleotide_diversity_per_population(self):
        """Calculate nucleotide diversity (Pi) for each locus per population.

        This method calculates nucleotide diversity (Pi) for each locus in the SNP data per population. Pi is defined as the average number of nucleotide differences per site between two sequences.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the nucleotide diversity values per locus for that population.
        """
        he_and_n_per_population = self.expected_heterozygosity_per_population(
            return_n=True
        )
        pi_per_population = {}

        for pop_id, (he, n_individuals) in he_and_n_per_population.items():
            n = n_individuals.astype(float)

            with np.errstate(divide="ignore", invalid="ignore"):
                pi = (n / (n - 1)) * he  # Bias correction for sample size
                pi[n <= 1] = np.nan  # Handle cases where n <=1

            pi_per_population[pop_id] = pd.Series(pi)

        return pi_per_population

    def weir_cockerham_fst_between_populations(
        self, n_bootstraps: int = 0, n_jobs: int = -1
    ):
        """Calculate Weir and Cockerham's Fst for each locus between all pairs of populations.

        This method calculates Weir and Cockerham's Fst for each locus between all pairs of populations. Fst is a measure of population differentiation due to genetic structure. Fst values range from 0 to 1, where 0 indicates no genetic differentiation and 1 indicates complete differentiation.

        Args:
            n_bootstraps (int): Number of bootstrap replicates. If 0, no bootstrapping is performed. Defaults to 0.
            n_jobs (int): Number of threads to use for parallelization. If -1, use all available cores. Defaults to -1.

        Returns:
            dict: If n_bootstraps=0, returns a dictionary where keys are tuples of population pairs,
                and values are pandas Series containing Fst values per locus.
                If n_bootstraps>0, returns a dictionary where keys are tuples of population pairs,
                and values are arrays of shape (n_loci, n_bootstraps) containing Fst values per locus per bootstrap replicate.
        """
        pop_indices = self.genotype_data.get_population_indices()
        populations = list(pop_indices.keys())
        n_loci = self.alignment_012.shape[1]

        # Prepare data for each population
        pop_alignments = {}
        for pop in populations:
            indices = pop_indices[pop]
            alignment = self.alignment_012[indices, :].astype(float).copy()
            alignment[alignment == -9] = np.nan  # Replace missing data
            pop_alignments[pop] = alignment

        # Function to compute Fst per SNP between two populations
        def compute_fst_pair(alignment1, alignment2):
            # Number of non-missing individuals per locus
            n1_per_locus = np.sum(~np.isnan(alignment1), axis=0)
            n2_per_locus = np.sum(~np.isnan(alignment2), axis=0)
            n_per_locus = n1_per_locus + n2_per_locus

            # Sum of alternate alleles at each locus
            alt_allele_counts1 = np.nansum(alignment1, axis=0)
            alt_allele_counts2 = np.nansum(alignment2, axis=0)
            alt_allele_counts_total = alt_allele_counts1 + alt_allele_counts2

            # Total alleles per locus
            total_alleles1 = 2 * n1_per_locus
            total_alleles2 = 2 * n2_per_locus
            total_alleles = total_alleles1 + total_alleles2

            with np.errstate(divide="ignore", invalid="ignore"):
                # Allele frequencies
                p1 = alt_allele_counts1 / total_alleles1
                p2 = alt_allele_counts2 / total_alleles2
                p_total = alt_allele_counts_total / total_alleles

                # Compute components of Fst
                s2 = ((p1 - p_total) ** 2) * n1_per_locus + (
                    (p2 - p_total) ** 2
                ) * n2_per_locus
                s2 = s2 / (n_per_locus - 1)

                he_total = 2 * p_total * (1 - p_total)

                # Fst per locus
                fst = s2 / he_total

                # Avoid invalid values
                fst[(he_total == 0) | (n_per_locus <= 1)] = np.nan

            return fst

        if n_bootstraps == 0:
            # No bootstrapping, compute Fst per SNP
            fst_per_population_pair = {}

            # Iterate over all unique pairs of populations
            for pop1, pop2 in itertools.combinations(populations, 2):
                alignment1 = pop_alignments[pop1]
                alignment2 = pop_alignments[pop2]

                fst = compute_fst_pair(alignment1, alignment2)
                fst_per_population_pair[(pop1, pop2)] = pd.Series(fst)

            return fst_per_population_pair
        else:
            fst_bootstrap_per_population_pair = {
                (pop1, pop2): np.zeros((n_loci, n_bootstraps))
                for pop1, pop2 in itertools.combinations(populations, 2)
            }

            # Function to perform one bootstrap replicate by resampling SNPs with replacement
            def bootstrap_replicate(seed):
                np.random.seed(seed)

                # Resample SNPs with replacement
                resample_indices = np.random.choice(n_loci, size=n_loci, replace=True)

                # Compute Fst per population pair for resampled SNPs
                fst_replicate = {}
                for pop1, pop2 in itertools.combinations(populations, 2):
                    alignment1 = pop_alignments[pop1][:, resample_indices]
                    alignment2 = pop_alignments[pop2][:, resample_indices]

                    fst = compute_fst_pair(alignment1, alignment2)
                    fst_replicate[(pop1, pop2)] = fst

                return fst_replicate

            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            elif n_jobs > 1:
                n_jobs = min(n_jobs, multiprocessing.cpu_count())
            else:
                if n_jobs < -1:
                    msg = f"Invalid value for n_jobs: {n_jobs}. Setting n_jobs to 1."
                    self.logger.warning(msg)
                n_jobs = 1

            # Perform bootstrapping in parallel
            seeds = np.random.randint(0, 1e9, size=n_bootstraps)
            if n_jobs > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    results = list(executor.map(bootstrap_replicate, seeds))
            else:
                results = [bootstrap_replicate(seed) for seed in seeds]

            # Collect results
            for b, fst_replicate in enumerate(results):
                for pop_pair in fst_bootstrap_per_population_pair.keys():
                    fst_bootstrap_per_population_pair[pop_pair][:, b] = fst_replicate[
                        pop_pair
                    ]

            return fst_bootstrap_per_population_pair

    def amova(self):
        """Performs an Analysis of Molecular Variance (AMOVA) on the input data.

        This method performs an Analysis of Molecular Variance (AMOVA) on the input data. AMOVA is a statistical method used to assess the partitioning of genetic variation within and among populations. The method calculates variance components and the Phi_ST statistic, which quantifies the proportion of genetic variation due to population structure.

        Returns:
            Dict[str, float]: A dictionary containing within- and among-population variance components and the Phi_ST statistic.
        """
        self.logger.info("Performing Analysis of Molecular Variance (AMOVA)...")

        # Get the genotype data (samples x loci)
        # Replace missing data with NaN and convert to float
        genotypes = self.alignment_012.astype(float)
        genotypes[genotypes == -9] = np.nan  # Missing values encoded as -9

        # Get the population mapping
        popmap_inverse = (
            self.genotype_data.popmap_inverse
        )  # {population_id: [sample_ids]}

        # Create a mapping from sample IDs to indices
        sample_ids = self.genotype_data.samples
        sample_id_to_index = {
            sample_id: idx for idx, sample_id in enumerate(sample_ids)
        }

        # Group genotypes by populations
        populations = {}
        for pop_id, samples in popmap_inverse.items():
            indices = [sample_id_to_index[sample_id] for sample_id in samples]
            populations[pop_id] = genotypes[
                indices, :
            ]  # Extract genotypes for this population

        # Total number of populations
        G = len(populations)

        # Initialize overall sums
        SSP_total = 0.0  # Sum of squares among populations
        SSW_total = 0.0  # Sum of squares within populations
        df_among_total = 0
        df_within_total = 0

        L = genotypes.shape[1]  # Total number of loci

        for l in range(L):
            # Extract genotypes for locus l
            locus_genotypes = genotypes[:, l]

            # Skip locus if all data is missing
            if np.all(np.isnan(locus_genotypes)):
                continue

            # Overall mean genotype at locus l
            locus_mean = np.nanmean(locus_genotypes)

            # Initialize sums for this locus
            SSP_l = 0.0  # Sum of squares among populations at locus l
            SSW_l = 0.0  # Sum of squares within populations at locus l

            # Total number of individuals with data at locus l
            N_l = np.count_nonzero(~np.isnan(locus_genotypes))

            # Degrees of freedom
            df_total = N_l - 1
            df_among = G - 1
            df_within = df_total - df_among

            if df_within <= 0 or df_among <= 0:
                # Skip this locus if degrees of freedom are invalid
                continue

            # Population means and sums of squares
            for pop_id, pop_genotypes in populations.items():
                pop_locus_genotypes = pop_genotypes[:, l]
                n_i = np.count_nonzero(~np.isnan(pop_locus_genotypes))

                # Skip if no data for this population at locus l
                if n_i == 0:
                    continue

                pop_mean = np.nanmean(pop_locus_genotypes)

                # Sum of squares within population i at locus l
                SSW_l += np.nansum((pop_locus_genotypes - pop_mean) ** 2)

                # Sum of squares among populations at locus l
                SSP_l += n_i * (pop_mean - locus_mean) ** 2

            # Accumulate sums and degrees of freedom
            SSP_total += SSP_l
            SSW_total += SSW_l
            df_among_total += df_among
            df_within_total += df_within

        # Check that degrees of freedom are valid
        if df_among_total <= 0 or df_within_total <= 0:
            # Cannot compute variance components
            results = {
                "Among_population_variance": np.nan,
                "Within_population_variance": np.nan,
                "Phi_ST": np.nan,
            }
            return results

        # Calculate overall mean squares
        MSP = SSP_total / df_among_total
        MSW = SSW_total / df_within_total

        # Sample sizes per population
        n_i_array = np.array([len(populations[pop_id]) for pop_id in populations])
        N = np.sum(n_i_array)

        # Calculate n_c (average sample size corrected for sample size variance)
        n_c = (N - np.sum(n_i_array**2) / N) / (G - 1)

        # Variance components
        sigma_squared_among = (MSP - MSW) / n_c
        sigma_squared_within = MSW

        # Phi_ST statistic
        phi_st = sigma_squared_among / (sigma_squared_among + sigma_squared_within)

        # Compile results
        results = {
            "Among_population_variance": sigma_squared_among,
            "Within_population_variance": sigma_squared_within,
            "Phi_ST": phi_st,
        }

        results = {k: float(v) for k, v in results.items()}  # Convert to float

        self.logger.info("AMOVA calculation complete!")

        return results
