import warnings
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from snpio.popgenstats.summary_statistics import SummaryStatistics
from snpio.utils.results_exporter import ResultsExporter

exporter = ResultsExporter()


class FstOutliers:
    """Class for detecting Fst outliers between populations."""

    def __init__(self, genotype_data, alignment_012, logger, plotter):
        """Initialize the FstOutliers object.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing genotype data.
            logger (logging.Logger): Logger object.
        """
        self.genotype_data = genotype_data
        self.alignment_012 = alignment_012
        self.logger = logger
        self.plotter = plotter

        self.sum_stats = SummaryStatistics(
            genotype_data, alignment_012, logger, plotter
        )

        exporter.output_dir = Path(f"{self.genotype_data.prefix}_output", "analysis")

    @exporter.capture_results
    def detect_fst_outliers_dbscan(
        self,
        correction_method: Literal["bonferroni", "fdr_bh"] | None = None,
        alpha: float = 0.05,
        n_jobs: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect Fst outliers from SNP data using DBSCAN.

        This method detects Fst outliers from SNP data using DBSCAN clustering. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (str, optional): Multiple testing correction method that performs P-value adjustment, 'bonf' (Bonferroni) or 'fdr' (FDR B-H). If not specified, no correction or P-value adjustment is applied. Defaults to None.
            alpha (float): Significance level for multiple test correction (with adjusted P-values). Defaults to 0.05.
            n_jobs (int): Number of CPU threads to use for parallelization. If set to -1, all available CPU threads are used. Defaults to 1.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A DataFrame containing the Fst outliers and contributing population pairs, and a DataFrame containing the adjusted P-values if ``correction_method`` was provided or the un-adjusted P-values otherwise.
        """
        outlier_snps, contributing_pairs, adjusted_pvals_df = self._dbscan_fst(
            correction_method, alpha, n_jobs
        )

        # Add contributing pairs to the outlier_snps DataFrame
        outlier_snps["Contributing_Pairs"] = contributing_pairs

        self.logger.debug(f"{outlier_snps=}")

        if outlier_snps.empty:
            self.logger.warning("No Fst outliers detected. Skipping correspoding plot.")
        else:
            self.logger.info(f"{len(outlier_snps)} Fst outliers detected.")

            try:
                # Plot the outlier SNPs
                self.plotter.plot_fst_outliers(outlier_snps)
            except ValueError as e:
                self.logger.warning(f"Error plotting Fst outliers: {e}. Skipping plot.")

        self.logger.info("Fst outlier detection complete!")
        return outlier_snps, adjusted_pvals_df

    @exporter.capture_results
    def detect_fst_outliers_bootstrap(
        self,
        correction_method: Literal["bonferroni", "fdr_bh"] | None = None,
        alpha: float = 0.05,
        n_bootstraps: int = 1000,
        n_jobs: int = 1,
        alternative: Literal["two", "upper", "lower"] = "two",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect Fst outliers from SNP data using traditional bootstrapping.

        Outliers are identified based on the distribution of Fst values between population pairs. This method now supports one-tailed tests: specifying 'upper' identifies SNPs with significantly higher Fst than expected, and 'lower' identifies SNPs with significantly lower Fst than expected. The default 'two' option performs a two-tailed test.

        Args:
            correction_method (Literal["bonferroni", "fdr_bh"] | None): Multiple testing correction method.
                If None, no adjustment is applied.
            alpha (float): Significance level for adjusted P-values. Defaults to 0.05.
            n_bootstraps (int): Number of bootstrap replicates. Defaults to 1000.
            n_jobs (int): Number of CPU threads for parallelization. Defaults to 1.
            alternative (Literal["two", "upper", "lower"]): Type of test to perform.
                "two" for two-tailed, "upper" for testing higher than expected, and "lower" for lower than expected.
                Defaults to "two".

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing a DataFrame of Fst outliers (with contributing
            population pairs as a column) and a DataFrame of adjusted (or unadjusted) P-values.
        """
        outlier_snps, contributing_pairs, adjusted_pvals_df = self._boot_fst(
            correction_method, alpha, n_bootstraps, n_jobs, alternative=alternative
        )

        outlier_snps["Contributing_Pairs"] = contributing_pairs

        self.logger.debug(f"{outlier_snps=}")

        if outlier_snps.empty:
            self.logger.warning(
                "No Fst outliers detected. Skipping corresponding plot."
            )
        else:
            self.logger.info(f"{len(outlier_snps)} Fst outliers detected.")
            try:
                self.plotter.plot_fst_outliers(outlier_snps)
            except ValueError as e:
                self.logger.warning(f"Error plotting Fst outliers: {e}. Skipping plot.")

        self.logger.info("Fst outlier detection complete!")
        return outlier_snps, adjusted_pvals_df

    def _boot_fst(
        self,
        correction_method: Optional[str],
        alpha: float,
        n_bootstraps: int,
        n_jobs: int,
        alternative: Literal["two", "upper", "lower"],
    ) -> Tuple[pd.DataFrame, List[Tuple[str]], pd.DataFrame]:
        """Detect Fst outliers using a bootstrapping approach.

        Bootstrap replicates of pairwise Weir and Cockerham's Fst estimates are used to generate a null distribution for each SNP. Observed Fst values are compared against the bootstrap distribution using Z-scores and associated one- or two-tailed P-values. Multiple testing correction is applied if specified. Outliers are identified based on the adjusted P-values.

        Args:
            correction_method (Optional[str]): Correction method for multiple tests.
            alpha (float): Significance level.
            n_bootstraps (int): Number of bootstrap replicates.
            n_jobs (int): Number of parallel jobs (if -1, all cores are used).
            alternative (Literal["two", "upper", "lower"]): Type of test to perform.
                "two" for two-tailed, "upper" for testing higher than expected, and "lower" for lower than expected.

        Returns:
            Tuple[pd.DataFrame, List[Tuple[str]], pd.DataFrame]:
                - DataFrame of outlier SNPs (rows) with observed Fst values (columns),
                - List of contributing population pairs for each outlier SNP,
                - DataFrame of adjusted (or unadjusted) P-values.
        """
        self.logger.info("Detecting Fst outliers using bootstrapping...")

        # Compute bootstrapped Fst estimates.
        fst_bootstrap_per_population_pair = self.sum_stats.weir_cockerham_fst(
            n_bootstraps=n_bootstraps, n_jobs=n_jobs
        )

        fst_means = {}
        fst_stds = {}
        for pop_pair, fst_bootstrap in fst_bootstrap_per_population_pair.items():
            valid_vals = ~np.isnan(fst_bootstrap)
            if np.count_nonzero(valid_vals) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_vals = np.nanmean(fst_bootstrap, axis=-1)
                    std_vals = np.nanstd(fst_bootstrap, axis=-1)
                fst_means[pop_pair] = mean_vals
                fst_stds[pop_pair] = np.maximum(std_vals, 1e-3)
            else:
                self.logger.info(f"No valid bootstrap Fst values for pair {pop_pair}.")
                n_loci = fst_bootstrap.shape[0]
                fst_means[pop_pair] = np.full(n_loci, np.nan)
                fst_stds[pop_pair] = np.full(n_loci, np.nan)

        fst_mean_df = pd.DataFrame(fst_means)
        fst_std_df = pd.DataFrame(fst_stds)

        valid_rows = fst_mean_df.dropna().index
        fst_mean_df = fst_mean_df.loc[valid_rows]
        fst_std_df = fst_std_df.loc[valid_rows]

        self.logger.debug(f"Bootstrap mean DataFrame: {fst_mean_df=}")
        self.logger.debug(f"Bootstrap std DataFrame: {fst_std_df=}")

        fst_per_population_pair = self.sum_stats.weir_cockerham_fst()
        fst_df = pd.DataFrame.from_dict(
            {k: v.values for k, v in fst_per_population_pair.items()},
            orient="columns",
        )

        fst_df.index = np.arange(len(fst_df))
        fst_df = fst_df.loc[valid_rows]

        common_columns = fst_df.columns.intersection(fst_mean_df.columns)
        fst_df = fst_df[common_columns]
        fst_mean_df = fst_mean_df[common_columns]
        fst_std_df = fst_std_df[common_columns]

        fst_df = fst_df.fillna(0.0)

        self.logger.debug(f"Observed Fst DataFrame: {fst_df=}")
        self.logger.debug(f"Aligned bootstrap mean DataFrame: {fst_mean_df=}")
        self.logger.debug(f"Aligned bootstrap std DataFrame: {fst_std_df=}")

        # Compute Z-scores: (observed Fst - bootstrap mean) / bootstrap std
        z_scores = (fst_df - fst_mean_df) / fst_std_df

        # Compute one- or two-tailed P-values based on the alternative hypothesis
        if alternative == "two":
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        elif alternative == "upper":
            p_values = 1 - norm.cdf(z_scores)
        elif alternative == "lower":
            p_values = norm.cdf(z_scores)
        else:
            raise ValueError(
                "Invalid value for 'alternative'. Choose from 'two', 'upper', or 'lower'."
            )
        p_values = pd.DataFrame(
            p_values, index=z_scores.index, columns=z_scores.columns
        )

        self.logger.debug(f"Z-scores DataFrame: {z_scores=}")
        self.logger.debug(f"P-values DataFrame: {p_values=}")

        all_p_values = p_values.values.flatten()
        valid_idx = ~np.isnan(all_p_values)
        all_p_values_valid = all_p_values[valid_idx]
        if correction_method:
            adjusted_pvals_valid = multipletests(
                all_p_values_valid, method=correction_method
            )[1]
        else:
            adjusted_pvals_valid = all_p_values_valid
        adjusted_pvals = np.full(all_p_values.shape, np.nan)
        adjusted_pvals[valid_idx] = adjusted_pvals_valid
        adjusted_pvals_df = pd.DataFrame(
            adjusted_pvals.reshape(p_values.shape),
            index=p_values.index,
            columns=p_values.columns,
        )

        significant = adjusted_pvals_df < alpha
        outlier_indices = significant.any(axis=1)
        outlier_snps = fst_df.loc[outlier_indices]

        # Compute contributing pairs only for the outlier SNPs
        contributing_pairs = self._identify_significant_pairs(
            alpha, adjusted_pvals_df.loc[outlier_snps.index]
        )

        return outlier_snps, contributing_pairs, adjusted_pvals_df

    def _dbscan_fst(
        self, correction_method: str | None, alpha: float, n_jobs: int
    ) -> Tuple[pd.DataFrame, list, pd.DataFrame]:
        """Detect Fst outliers using DBSCAN clustering.

        Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame of the outlier SNPs, a list of contributing population pairs, and a DataFrame of adjusted (or unadjusted) p-values. The DBSCAN algorithm is used to cluster the Fst values and identify outliers. The optimal eps value is estimated using the k-distance graph method. Contributing population pairs are identified using a z-score approach on the outlier SNPs.

        Args:
            correction_method (Optional[str]): Correction method for multiple tests.
            alpha (float): Significance level for adjusted p-values.
            n_jobs (int): Number of CPU threads to use for parallelization.

        Returns:
            Tuple[pd.DataFrame, list, pd.DataFrame]:
                - DataFrame of Fst outliers,
                - List of contributing population pairs for each outlier SNP,
                - DataFrame of adjusted (or unadjusted) p-values.
        """
        self.logger.info("Detecting Fst outliers using DBSCAN...")

        # Step 1: Calculate Fst values between population pairs
        fst_per_population_pair = self.sum_stats.weir_cockerham_fst()
        fst_df = pd.DataFrame.from_dict(
            {str(k): v.values for k, v in fst_per_population_pair.items()},
            orient="columns",
        )

        fst_df.index = np.arange(len(fst_df))
        fst_df.dropna(inplace=True)

        # Step 2: Prepare data for DBSCAN
        fst_values = fst_df.to_numpy()
        n_population_pairs = fst_values.shape[1]
        n_data_points = fst_values.shape[0]
        min_samples = min(max(2, 2 * n_population_pairs), n_data_points - 1)
        scaler = StandardScaler()
        fst_values_scaled = scaler.fit_transform(fst_values)

        # Step 3: Estimate eps and run DBSCAN
        eps = self._estimate_eps(fst_values_scaled, min_samples)
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        labels = db.fit_predict(fst_values_scaled)
        outlier_indices = np.where(labels == -1)[0]
        outlier_snps = fst_df.iloc[outlier_indices]

        # Step 4: Identify contributing pairs using a z-score approach on the outlier SNPs
        fst_means = fst_df.mean()
        fst_stds = fst_df.std().replace(0, np.finfo(float).eps)
        z_scores = (outlier_snps - fst_means) / fst_stds

        # Compute two-tailed p-values using vectorized operations
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        p_values = pd.DataFrame(
            p_values, index=z_scores.index, columns=z_scores.columns
        )
        all_p_values = p_values.to_numpy().flatten()
        valid_idx = ~np.isnan(all_p_values)
        all_p_values_valid = all_p_values[valid_idx]
        if correction_method:
            if all_p_values_valid.size > 0:
                adjusted_pvals_valid = multipletests(
                    all_p_values_valid, method=correction_method
                )[1]
            else:
                self.logger.warning("No valid p-values found for DBSCAN outliers.")
                adjusted_pvals_valid = all_p_values_valid
        else:
            adjusted_pvals_valid = all_p_values_valid
        adjusted_pvals = np.full(all_p_values.shape, np.nan)
        adjusted_pvals[valid_idx] = adjusted_pvals_valid
        adjusted_pvals_df = pd.DataFrame(
            adjusted_pvals.reshape(p_values.shape),
            index=p_values.index,
            columns=p_values.columns,
        )

        # IMPORTANT: Compute contributing pairs only for the outlier SNPs
        contributing_pairs = self._identify_significant_pairs(
            alpha, adjusted_pvals_df.loc[outlier_snps.index]
        )

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
                eps = np.percentile(distances_k, 10)
                self.logger.warning(
                    f"Automated eps could not identify a knee. Using eps from 10th percentile: {eps}"
                )

        if isinstance(eps, (np.float32, np.float64)):
            eps = float(eps)

        return eps

    def _identify_significant_pairs(
        self, alpha: float, adjusted_pvals_df: pd.DataFrame
    ) -> list:
        """Identify significant population pairs for each SNP in a subset.

        Args:
            alpha (float): Significance level for identifying outliers.
            adjusted_pvals_subset_df (pd.DataFrame): A DataFrame of adjusted p-values for a subset of SNPs.

        Returns:
            list: A list of lists, where each inner list contains the population pairs (column names) that are significant (i.e. adjusted p-value < alpha) for the corresponding SNP.
        """
        adjusted_pvals_subset_df = adjusted_pvals_df.copy()
        contributing_pairs = []
        for snp in adjusted_pvals_subset_df.index:
            row = adjusted_pvals_subset_df.loc[snp]
            sig_pairs = row[row < alpha].index.tolist()
            contributing_pairs.append(sig_pairs)
        return contributing_pairs
