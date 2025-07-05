import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from snpio.popgenstats.fst_distance import FstDistance
from snpio.popgenstats.fst_outliers_dbscan import DBSCANOutlierDetector
from snpio.popgenstats.fst_outliers_perm import PermutationOutlierDetector
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


class FstOutliers:
    """Class for detecting Fst outliers between populations."""

    def __init__(
        self,
        genotype_data: Any,
        plotter: Any,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the FstOutliers object.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing genotype data.
            logger (logging.Logger): Logger object.
        """
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.verbose = verbose
        self.debug = debug

        self.fst_dist = FstDistance(
            genotype_data, plotter, verbose=verbose, debug=debug
        )

        if genotype_data.was_filtered:
            outdir = Path(f"{self.genotype_data.prefix}_output", "nremover", "analysis")
        else:
            outdir = Path(f"{self.genotype_data.prefix}_output", "analysis")
        outdir.mkdir(parents=True, exist_ok=True)

        logman = LoggerManager(
            __name__, prefix=self.genotype_data.prefix, verbose=verbose, debug=debug
        )
        self.logger = logman.get_logger()

        self.full_matrix = self.genotype_data.snp_data.astype(str)

        self.logger.debug(f"{self.full_matrix.shape=}")

        # Build an index for sample -> row
        self.sample_to_idx = {s: i for i, s in enumerate(self.genotype_data.samples)}

        # Convert population -> integer indices
        self.pop_indices = {
            pop: np.array([self.sample_to_idx[s] for s in sample_list], dtype=int)
            for pop, sample_list in self.genotype_data.popmap_inverse.items()
        }

        self.pop_names = sorted(self.pop_indices.keys())

        self.iupac = IUPAC(logger=self.logger)
        self.phased_encoding = self.iupac.get_phased_encoding()

    def detect_fst_outliers_dbscan(
        self,
        correction_method: Literal["bonferroni", "fdr_bh"] | None = None,
        alpha: float = 0.05,
        n_permutations: int = 1000,
        n_jobs: int = 1,
        seed: int | None = None,
        min_samples: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect Fst outliers from SNP data using DBSCAN.

        This method detects Fst outliers from SNP data using DBSCAN clustering. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (str, optional): Multiple testing correction method that performs P-value adjustment, 'bonf' (Bonferroni) or 'fdr' (FDR B-H). If not specified, no correction or P-value adjustment is applied. Defaults to None.
            alpha (float): Significance level for multiple test correction (with adjusted P-values). Defaults to 0.05.
            n_permutations (int): Number of permutations for significance testing. Defaults to 1000.
            n_jobs (int): Number of CPU threads to use for parallelization. If set to -1, all available CPU threads are used. Defaults to 1.
            seed (int): Random seed for reproducibility. Defaults to 42.
            min_samples (int): Minimum number of samples for DBSCAN clustering. Defaults to 5.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A DataFrame containing the Fst outliers and contributing population pairs, and a DataFrame containing the adjusted P-values if ``correction_method`` was provided or the un-adjusted P-values otherwise.
        """
        outlier_snps, contrib_pairs_dict = self._dbscan_fst(
            correction_method,
            alpha,
            n_permutations,
            n_jobs,
            seed=seed,
            min_samples=min_samples,
        )

        # Add contributing pairs to the outlier_snps DataFrame

        self.logger.debug(f"outlier_snps DataFrame:\n{outlier_snps}")

        if outlier_snps.empty:
            self.logger.warning("No Fst outliers detected. Skipping Fst outlier plot.")
        else:
            self.logger.info(
                f"{len(outlier_snps)} Fst outliers detected using DBSCAN method."
            )

        self.logger.info("DBSCAN Fst outlier detection complete!")

        return outlier_snps

    def _dbscan_fst(
        self,
        correction_method: (
            Literal[
                "bonferroni",
                "fdr_bh",
                "fdr_by",
                "holm",
                "sidak",
                "holm-sidak",
                "hommel",
            ]
            | None
        ),
        alpha: float,
        n_permutations: int,
        n_jobs: int,
        seed: int = None,
        min_samples: int = 5,
        z_score_threshold: float = 2.5,  # New parameter for contribution
    ) -> Tuple[pd.DataFrame, dict]:
        """Detect Fst outliers using DBSCAN clustering method.

        This method applies the DBSCAN clustering algorithm to identify outlier loci based on their Fst values. It returns a DataFrame containing the detected outliers and their associated metadata.

        Args:
            correction_method (str | None): Multiple testing correction method, e.g., 'bonferroni', 'fdr_bh', or None for no correction.
            alpha (float): Significance level for multiple testing correction.
            n_permutations (int): Number of permutations for significance testing.
            n_jobs (int): Number of CPU threads to use for parallelization.
            seed (int | None): Random seed for reproducibility. Defaults to None.
            min_samples (int): Minimum number of samples for DBSCAN clustering. Defaults to 5.
            z_score_threshold (float): Threshold for z-scores to determine contributing pairs. Defaults to 2.5.

        Returns:
            Tuple[pd.DataFrame, dict]: A DataFrame containing the Fst outliers and contributing population pairs, and a dictionary mapping loci to contributing pairs.
        """
        self.logger.info("Detecting Fst outliers with DBSCAN method…")

        # ------------------------------------------------------------------
        # 1. Prepare the (loci x pairs) Fst matrix (your code is good)
        # ------------------------------------------------------------------
        fst_dict = self._weir_cockerham_fst_permutation_per_locus()
        df_list = []
        for (pop1, pop2), df in fst_dict.items():
            dftmp = df[["Locus", "Fst"]]
            dftmp["Population_Pair"] = f"{pop1}_{pop2}"
            df_list.append(dftmp)
        fst_long = pd.concat(df_list, ignore_index=True)
        fst_wide = fst_long.pivot(
            index="Locus", columns="Population_Pair", values="Fst"
        ).sort_index()

        if fst_wide.shape[1] < 2:
            msg = f"DBSCAN requires at least 2 population pairs, but got: {fst_wide.shape[1]}"
            self.logger.error(msg)
            raise ValueError(msg)

        # ------------------------------------------------------------------
        # 2. Impute and prepare data
        # ------------------------------------------------------------------
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(fst_wide)
        df_imputed = pd.DataFrame(X, index=fst_wide.index, columns=fst_wide.columns)
        df_imputed = df_imputed.clip(lower=0.0)

        # ------------------------------------------------------------------
        # 3. Identify Global Outliers using the corrected multivariate method
        # ------------------------------------------------------------------
        est = DBSCANOutlierDetector(
            eps="auto",
            min_samples=min_samples,
            n_jobs=n_jobs,
            standardize="standardize",
            null_model="shuffle",
            n_simulations=n_permutations,
            random_state=seed,
            verbose=True,
        )

        est.fit(df_imputed)
        p_values_1d = est.estimate_pvalues()  # 1D array of p-values per locus

        # Returns two 1D Series: a boolean mask and q-values
        is_outlier_series, q_values_series = est.identify_outliers(
            p_values_1d, alpha=alpha, correction_method=correction_method
        )

        # Convert the boolean mask and q-values to Series with the same index
        # as df_imputed
        is_outlier_series.index = df_imputed.index
        q_values_series.index = df_imputed.index

        # Get the loci that are significant outliers
        significant_loci = is_outlier_series[is_outlier_series].index

        if significant_loci.empty:
            self.logger.warning(
                "No significant outliers were detected by DBSCAN method."
            )
            return pd.DataFrame(), {}

        # -------------------------------------------------------------------
        # 4. Post-Hoc: Analyze which pairs contributed to the outlier status
        # -------------------------------------------------------------------
        # Get the z-scores by transforming the data with the fitted scaler
        z_scores_df = pd.DataFrame(
            est._scaler.transform(df_imputed),
            index=df_imputed.index,
            columns=df_imputed.columns,
        )

        # Filter z-scores for our significant loci
        significant_z_scores = z_scores_df.loc[significant_loci]

        # Find which pairs in these loci have extreme z-scores
        contrib_mask = significant_z_scores.abs() >= z_score_threshold

        # Create the final results
        final_results = []
        contrib_pairs_dict = {}
        for locus in significant_loci:
            locus_data = fst_wide.loc[locus]
            locus_q_value = q_values_series.loc[locus]
            contributing_pairs = locus_data[contrib_mask.loc[locus]].index.tolist()
            contrib_pairs_dict[locus] = contributing_pairs

            # Add a row for each contributing pair for this locus
            for pair in contributing_pairs:
                final_results.append(
                    {
                        "Locus": locus,
                        "Population_Pair": pair,
                        "Fst": locus_data[pair],
                        "q_value": locus_q_value,
                    }
                )

        outliers_df = pd.DataFrame(final_results)

        # -------------------------------------------------------------------
        # 5. Validate and Return
        # -------------------------------------------------------------------
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            validation_metrics = est.validate(df_imputed)

        self.logger.info(f"DBSCAN validation metrics: {validation_metrics}")

        return outliers_df, contrib_pairs_dict

    def detect_outliers_permutation(
        self,
        n_perm: int = 1000,
        correction_method: Literal["fdr_bh", "bonferroni"] | None = "fdr_bh",
        alpha: float = 0.05,
        seed: int = 42,
        alternative: Literal["upper", "lower", "both"] = "both",
    ) -> pd.DataFrame:
        """Convenience method: detect all per-locus outliers in all pairs.

        This method runs the per-locus Fst + permutations for all population pairs and detects outliers based on the specified significance threshold. It returns a DataFrame containing only the outlier SNPs. The DataFrame includes columns for the locus, population pair, Fst value, p-value, and adjusted p-value. The method also includes a column for contributing pairs, which lists all population pairs that contributed to the outlier status of the SNP. The outlier status is determined based on the adjusted p-value and the specified significance threshold.

        Args:
            n_perm (int): Number of permutations per locus.
            correction_method (Literal["fdr_bh", "bonferroni"] | None): 'fdr_bh', 'bonferroni', or None.
            alpha (float): Significance threshold.
            seed (int): RNG seed for reproducibility. Defaults to 42.
            bandwidth (float): KDE bandwidth override (optional). Can be either 'silverman', 'scott', or float. Defaults to "scott".
            alternative (str): Type of test ('upper', 'lower', 'both'). Defaults to 'both' (two-tailed test).

        Returns:
            pd.DataFrame: Outlier Fst DataFrame with columns: Locus, Pop1, Pop2, Population_Pair, Fst, pval, pval_adj, Contributing_Pairs.
        """
        perm_detector = PermutationOutlierDetector(
            genotype_data=self.genotype_data, verbose=self.verbose, debug=self.debug
        )

        df_fst = perm_detector.run(
            n_perm=n_perm,
            correction_method=correction_method,
            alpha=alpha,
            seed=seed,
            alternative=alternative,
        )

        return df_fst

    def _weir_cockerham_fst_permutation_per_locus(
        self,
    ) -> dict[Tuple[str, str], pd.DataFrame]:
        """Compute per-locus Weir & Cockerham's Fst, empirical p-values, Z-scores, and bootstraps for each population pair.

        This method performs the necessary calculations to obtain Fst estimates and their associated statistics for each locus. It uses permutation testing to estimate the variability of the Fst estimates. The results are returned as a dictionary where keys are tuples of population names (pop1, pop2) and values are DataFrames containing the Fst values, empirical p-values, Z-scores, and bootstrap replicates for each locus.

        Returns:
            dict: A dictionary where keys are tuples of population names (pop1, pop2) and values are DataFrames with columns:
                - 'Locus': Locus names from genotype data or numeric indices if not available.
                - 'Fst': Observed Fst values for the population pair
        """
        full_matrix = self.genotype_data.snp_data.astype(str)
        popmap = self.genotype_data.popmap_inverse
        sample_names = self.genotype_data.samples
        sample_to_idx = {name: i for i, name in enumerate(sample_names)}
        pop_indices = {
            pop_name: np.array([sample_to_idx[s] for s in samples], dtype=int)
            for pop_name, samples in popmap.items()
        }

        n_loci = full_matrix.shape[1]

        result = {}
        for pop1, pop2 in combinations(popmap.keys(), 2):
            # Iterate over all population pairs
            inds1 = pop_indices[pop1]
            inds2 = pop_indices[pop2]

            a_vals, d_vals = self.fst_dist.fst_variance_components_per_locus(
                inds1, inds2, full_matrix
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                observed_fst = np.where(d_vals != 0, a_vals / d_vals, np.nan)

            if (
                hasattr(self.genotype_data, "marker_names")
                and self.genotype_data.marker_names is not None
            ):
                locus_names = self.genotype_data.marker_names
            else:
                locus_names = [f"locus_{i}" for i in range(n_loci)]

            df = pd.DataFrame({"Locus": locus_names, "Fst": observed_fst})
            result[(pop1, pop2)] = df

        return result
