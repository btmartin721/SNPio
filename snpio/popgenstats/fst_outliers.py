from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from snpio.popgenstats.fst_distance import FstDistance
from snpio.utils.logging import LoggerManager

# This dictionary is adapted from your original phased_encoding.
# It maps IUPAC or ambiguous nucleotides to "phased" diploid notation.
PHASED_ENCODING = {
    "A": "A/A",
    "T": "T/T",
    "C": "C/C",
    "G": "G/G",
    "N": "N/N",
    ".": "N/N",
    "?": "N/N",
    "-": "N/N",
    "W": "A/T",
    "S": "C/G",
    "Y": "C/T",
    "R": "A/G",
    "K": "G/T",
    "M": "A/C",
    "H": "A/C",
    "B": "A/G",
    "D": "C/T",
    "V": "A/G",
}


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

    def detect_fst_outliers_dbscan(
        self,
        correction_method: Literal["bonferroni", "fdr_bh"] | None = None,
        alpha: float = 0.05,
        n_jobs: int = 1,
        n_bootstraps: int = 1000,
        alternative: Literal["upper", "lower", "both"] = "upper",
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect Fst outliers from SNP data using DBSCAN.

        This method detects Fst outliers from SNP data using DBSCAN clustering. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (str, optional): Multiple testing correction method that performs P-value adjustment, 'bonf' (Bonferroni) or 'fdr' (FDR B-H). If not specified, no correction or P-value adjustment is applied. Defaults to None.
            alpha (float): Significance level for multiple test correction (with adjusted P-values). Defaults to 0.05.
            n_jobs (int): Number of CPU threads to use for parallelization. If set to -1, all available CPU threads are used. Defaults to 1.
            n_bootstraps (int): Number of bootstrap replicates for Fst calculation. Defaults to 1000.
            alternative (str): Type of test ('upper', 'lower', 'both'). Defaults to 'upper' (one-tailed test).
            seed (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A DataFrame containing the Fst outliers and contributing population pairs, and a DataFrame containing the adjusted P-values if ``correction_method`` was provided or the un-adjusted P-values otherwise.
        """
        outlier_snps, contributing_pairs = self._dbscan_fst(
            correction_method,
            alpha,
            n_jobs,
            n_bootstraps=n_bootstraps,
            alternative=alternative,
            seed=seed,
        )

        # Add contributing pairs to the outlier_snps DataFrame
        outlier_snps["Contributing_Pairs"] = outlier_snps["Locus"].map(
            lambda x: contributing_pairs.get(x, [])
        )

        outlier_snps = outlier_snps[outlier_snps["pval_adj"] <= alpha]

        self.logger.debug(f"outlier_snps DataFrame:\n{outlier_snps}")

        if outlier_snps.empty:
            self.logger.warning("No Fst outliers detected. Skipping Fst outlier plot.")
        else:
            self.logger.info(
                f"{len(outlier_snps)} Fst outliers detected using DBSCAN method."
            )

            try:
                # Plot the outlier SNPs
                self.plotter.plot_fst_outliers(outlier_snps, "dbscan")

            except ValueError as e:
                self.logger.warning(f"Error plotting Fst outliers: {e}")
                self.logger.warning("Skipping Fst outlier plot.")

        self.logger.info("DBSCAN Fst outlier detection complete!")
        return outlier_snps

    @staticmethod
    def _compute_empirical_pvals(
        fst_df_scaled: pd.DataFrame,
        outlier_snps: pd.DataFrame,
        alternative: str = "upper",
    ) -> pd.DataFrame:
        """
        Compute empirical p-values for outliers based on Fst distributions within each population pair.

        Args:
            fst_df_scaled (pd.DataFrame): Long-format DataFrame with Locus, Population_Pair, and Fst (standardized).
            outlier_snps (pd.DataFrame): Subset of fst_df_scaled with DBSCAN-labeled outliers.
            alternative (str): 'upper', 'lower', or 'both' for one/two-tailed test.

        Returns:
            pd.DataFrame: p-values for each row in `fst_df_scaled`, with NaNs for non-outliers.
        """
        pvals = pd.Series(index=fst_df_scaled.index, dtype=float)
        for idx, row in outlier_snps.iterrows():
            pop_pair = row["Population_Pair"]
            locus_fst = row["Fst"]

            # Subset all values for this population pair
            subset = fst_df_scaled[fst_df_scaled["Population_Pair"] == pop_pair]["Fst"]
            subset = subset.dropna()

            if alternative == "upper":
                pval = np.mean(subset >= locus_fst)
            elif alternative == "lower":
                pval = np.mean(subset <= locus_fst)
            elif alternative == "both":
                mean_fst = np.mean(subset)
                diff = np.abs(locus_fst - mean_fst)
                pval = np.mean(np.abs(subset - mean_fst) >= diff)
            else:
                raise ValueError(f"Unsupported alternative hypothesis: {alternative}")

            pvals.loc[idx] = pval

        return pvals.to_frame(name="pval")

    def _dbscan_fst(
        self,
        correction_method: str | None,
        alpha: float,
        n_jobs: int,
        n_bootstraps: int = 1000,
        alternative: Literal["upper", "lower", "both"] = "upper",
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, list, pd.DataFrame]:
        """Detect Fst outliers using DBSCAN clustering.

        Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame of the outlier SNPs, a list of contributing population pairs, and a DataFrame of adjusted (or unadjusted) p-values. The DBSCAN algorithm is used to cluster the Fst values and identify outliers. The optimal eps value is estimated using the k-distance graph method. Contributing population pairs are identified using a z-score approach on the outlier SNPs.

        Args:
            correction_method (str | None): Correction method for multiple tests.
            alpha (float): Significance level for adjusted p-values.
            n_jobs (int): Number of CPU threads to use for parallelization.
            n_bootstraps (int): Number of bootstrap replicates for Fst calculation.
            alternative (str): Type of test ('upper', 'lower', 'both').
            Defaults to 'upper' (one-tailed test).
            seed (int): Random seed for reproducibility.
            Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, list, pd.DataFrame]:
                - DataFrame of Fst outliers,
                - List of contributing population pairs for each outlier SNP,
                - DataFrame of adjusted (or unadjusted) p-values.
        """
        self.logger.info("Detecting Fst outliers using DBSCAN...")

        # Step 1: Calculate Fst values between population pairs
        fst_dict = self.fst_dist.weir_cockerham_fst_bootstrap_per_locus(
            n_bootstraps=n_bootstraps,
            seed=seed,
            alternative=alternative,
            outdir=None,
        )

        dflist = []
        for (pop1, pop2), df in fst_dict.items():
            df["Population_Pair"] = f"{pop1}_{pop2}"
            dflist.append(df)

        # Concatenate all DataFrames into a single DataFrame
        fst_df = pd.concat(dflist, axis=0, ignore_index=True)

        # Pivot the DataFrame to have population pairs as columns
        # and loci as rows
        fst_df = fst_df.pivot(index="Locus", columns="Population_Pair", values="Fst")

        # Step 2: Prepare data for DBSCAN
        n_population_pairs = len(fst_df.columns)
        if n_population_pairs < 2:
            self.logger.warning(
                "Not enough population pairs to perform DBSCAN clustering."
            )
            return pd.DataFrame(), [], pd.DataFrame()

        # NOTE: Set min_samples based on the number of population pairs
        # Ensure min_samples is at least 2 and not greater than
        # n_data_points - 1
        # This is to avoid issues with DBSCAN when the number of
        # samples is very small
        # and to ensure that we have at least one sample in each
        # cluster.
        # Drop loci with NaN values
        n_data_points = len(fst_df) - 1
        min_samples = min(max(2, 2 * n_population_pairs), n_data_points)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        # Impute missing values and scale the data
        # NOTE: This is important for DBSCAN to work properly
        # and to avoid issues with NaN values
        # and to ensure that the data is centered and scaled
        # before clustering.
        fst_values_imputed = imputer.fit_transform(fst_df)
        fst_values_scaled = scaler.fit_transform(fst_values_imputed)

        fst_df_scaled = pd.DataFrame(
            fst_values_scaled, columns=fst_df.columns, index=fst_df.index
        )

        fst_df_scaled = fst_df_scaled.reset_index()
        fst_df_scaled = fst_df_scaled.melt(
            id_vars=["Locus"], var_name="Population_Pair", value_name="Fst"
        )

        # Step 3: Estimate eps and run DBSCAN
        eps = self._estimate_eps(
            fst_df_scaled["Fst"].values.reshape(-1, 1), min_samples
        )
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        labels = db.fit_predict(fst_df_scaled["Fst"].values.reshape(-1, 1))
        outlier_indices = np.where(labels == -1)[0]
        outlier_snps = fst_df_scaled.iloc[outlier_indices]

        # Step 4: Compute empirical p-values for outliers
        pval_df = self._compute_empirical_pvals(
            fst_df_scaled=fst_df_scaled,
            outlier_snps=outlier_snps,
            alternative=alternative,
        )

        # Step 5: Multiple testing correction
        if correction_method:
            pvals_array = pval_df["pval"].dropna().to_numpy()
            adjusted_array = multipletests(pvals_array, method=correction_method)[1]
            pval_df.loc[pval_df["pval"].notna(), "pval_adj"] = adjusted_array
        else:
            pval_df["pval_adj"] = pval_df["pval"]

        # Merge corrected pvals back to original outlier_snps DataFrame
        outlier_snps = outlier_snps.copy()
        outlier_snps["pval"] = pval_df["pval"]
        outlier_snps["pval_adj"] = pval_df["pval_adj"]

        # Unscale Fst values back to original
        outlier_snps = self._unscale_fsts(
            fst_df, scaler, fst_values_scaled, outlier_snps
        )

        # Step 6: Identify contributing pairs
        contributing_pairs = self._identify_significant_pairs(alpha, outlier_snps)

        return outlier_snps, contributing_pairs

    def _unscale_fsts(self, fst_df, scaler, fst_values_scaled, outlier_snps):
        fst_values_unscaled = scaler.inverse_transform(fst_values_scaled)

        # Rebuild a long-form DataFrame of unscaled Fst values
        fst_df_unscaled = pd.DataFrame(
            fst_values_unscaled, columns=fst_df.columns, index=fst_df.index
        ).reset_index()

        fst_df_unscaled = fst_df_unscaled.melt(
            id_vars=["Locus"], var_name="Population_Pair", value_name="Fst_unscaled"
        )

        # Merge unscaled values into the outlier_snps DataFrame
        outlier_snps = outlier_snps.merge(
            fst_df_unscaled, on=["Locus", "Population_Pair"], how="left"
        )

        # Overwrite the scaled "Fst" with the unscaled one (and drop
        # temporary column)
        outlier_snps["Fst"] = outlier_snps["Fst_unscaled"]
        return outlier_snps.drop(columns=["Fst_unscaled"])

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
            eps = 0.1
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
    ) -> dict:
        """Return a dict mapping locus -> list of significant population pairs."""
        required_cols = {"Locus", "Population_Pair", "pval_adj"}
        if not required_cols.issubset(adjusted_pvals_df.columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        result = {}
        for locus, subset in adjusted_pvals_df.groupby("Locus"):
            sig_pairs = subset.loc[
                subset["pval_adj"] <= alpha, "Population_Pair"
            ].tolist()
            result[locus] = sig_pairs

        return result

    def compute_per_locus_fst(
        self, pop1_inds: np.ndarray, pop2_inds: np.ndarray
    ) -> np.ndarray:
        """Compute Weir & Cockerham Fst for each locus between two populations.

        Args:
            pop1_inds (np.ndarray): Indices of population 1 individuals.
            pop2_inds (np.ndarray): Indices of population 2 individuals.

        Returns:
            np.ndarray: Array of shape (n_loci,) with per-locus Fst values (NaN if invalid).
        """
        full_mat = self.full_matrix.copy()
        n_loci = full_mat.shape[1]
        fst_values = np.full(n_loci, np.nan)

        for loc in range(n_loci):
            g1 = [full_mat[i, loc] for i in pop1_inds]
            g2 = [full_mat[j, loc] for j in pop2_inds]

            # clean out unknowns
            g1 = FstDistance._clean_inds(g1)
            g2 = FstDistance._clean_inds(g2)
            if not g1 or not g2:
                continue

            # Convert to phased if needed
            g1 = [PHASED_ENCODING.get(x, x) for x in g1]
            g2 = [PHASED_ENCODING.get(x, x) for x in g2]

            # Check for monomorphism:
            unique_1 = set(g1)
            unique_2 = set(g2)
            combined_unique = unique_1.union(unique_2)
            if len(combined_unique) == 1:
                # Exactly one allele across both populations => monomorphic
                fst_values[loc] = np.nan
                continue

            # compute single-locus numerator, denominator
            try:
                a, d = self.fst_dist._two_pop_weir_cockerham_fst(g1, g2)
                fst_values[loc] = a / d if d > 0 else np.nan
            except ValueError:
                fst_values[loc] = np.nan

        return fst_values

    def per_locus_permutation_test(
        self,
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        n_perm: int = 1000,
        seed: int = 42,
        bandwidth: Literal["silverman", "scott"] | float = "scott",
        alternative: Literal["upper", "lower", "both"] = "both",
        mode: Literal["auto", "kde", "empirical"] = "auto",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Permute sample labels for each locus and compute p-values.

        Args:
            pop1_inds (np.ndarray): Indices of population 1 individuals.
            pop2_inds (np.ndarray): Indices of population 2 individuals.
            n_perm (int): Number of permutation replicates.
            seed (int): RNG seed.
            bandwidth (str | float): KDE bandwidth.
            alternative (str): 'upper', 'lower', or 'both'.
            mode (str): 'auto', 'kde', or 'empirical'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Observed Fst and p-values.
        """
        rng = np.random.default_rng(seed)
        full_mat = self.full_matrix.copy()
        n_loci = full_mat.shape[1]
        observed_fst = self.compute_per_locus_fst(pop1_inds, pop2_inds)

        all_inds = np.concatenate([pop1_inds, pop2_inds])
        n1 = len(pop1_inds)
        p_values = np.full(n_loci, np.nan)

        skipped_loci = {}
        flat_loci = []
        few_valids = []
        for loc in range(n_loci):
            obs = observed_fst[loc]
            if np.isnan(obs) or np.isinf(obs):
                p_values[loc] = 1.0
                skipped_loci[loc] = "NaN or Inf observed Fst"
                continue

            perm_dist = np.full(n_perm, np.nan)
            for i in range(n_perm):
                perm = rng.permutation(all_inds)
                new_p1, new_p2 = perm[:n1], perm[n1:]

                g1 = [full_mat[idx, loc] for idx in new_p1]
                g2 = [full_mat[idx, loc] for idx in new_p2]
                g1 = FstDistance._clean_inds(g1)
                g2 = FstDistance._clean_inds(g2)

                if not g1 or not g2:
                    continue

                g1 = [PHASED_ENCODING.get(x, x) for x in g1]
                g2 = [PHASED_ENCODING.get(x, x) for x in g2]

                try:
                    a, d = self.fst_dist._two_pop_weir_cockerham_fst(g1, g2)
                    perm_dist[i] = a / d if d > 0 else np.nan
                except ValueError:
                    continue

            valid_perm = perm_dist[~np.isnan(perm_dist)]
            if len(valid_perm) < n_perm * 0.5:
                few_valids.append(loc)
                continue

            if np.isclose(valid_perm.max(), valid_perm.min()):
                flat_loci.append(loc)
                continue

            try:
                if mode in {"auto", "kde"}:
                    # Extend KDE range by 20%
                    buffer = (valid_perm.max() - valid_perm.min()) * 0.2
                    x_eval = np.linspace(
                        valid_perm.min() - buffer, valid_perm.max() + buffer, 5000
                    )

                    kde = gaussian_kde(valid_perm, bw_method=bandwidth)
                    pdf_vals = kde.evaluate(x_eval)
                    cdf_vals = np.cumsum(pdf_vals)
                    cdf_vals /= cdf_vals[-1]

                    # Interpolate robustly
                    cdf_obs = np.interp(obs, x_eval, cdf_vals, left=0.0, right=1.0)

                    if alternative == "upper":
                        pval = 1.0 - cdf_obs
                    elif alternative == "lower":
                        pval = cdf_obs
                    elif alternative == "both":
                        pval = 2.0 * min(cdf_obs, 1.0 - cdf_obs)
                    else:
                        msg = f"Invalid alternative: {alternative}"
                        self.logger.error(msg)
                        raise ValueError(msg)

                    # Fallback if p is zero/one
                    if mode == "auto" and (pval < 0.0 or pval > 1.0):
                        msg = "KDE p-value out-of-bounds"
                        self.logger.error(msg)
                        raise ValueError(msg)

                if mode == "empirical" or (mode == "auto" and "pval" not in locals()):
                    if alternative == "upper":
                        count_extreme = np.sum(valid_perm >= obs)
                    elif alternative == "lower":
                        count_extreme = np.sum(valid_perm <= obs)
                    elif alternative == "both":
                        center = np.mean(valid_perm)
                        count_extreme = np.sum(
                            np.abs(valid_perm - center) >= np.abs(obs - center)
                        )
                    else:
                        msg = f"Invalid alternative: {alternative}"
                        self.logger.error(msg)
                        raise ValueError(msg)

                    pval = (count_extreme + 1) / (len(valid_perm) + 1)

                p_values[loc] = min(max(pval, 0.0), 1.0)

            except Exception as e:
                skipped_loci[loc] = e
                continue

        if skipped_loci:
            for loc in skipped_loci.keys():
                p_values[loc] = 1.0
            self.logger.warning(
                f"Skipped {len(skipped_loci)} loci due to errors. Setting P-values to 1.0."
            )
            self.logger.warning(
                f"The following errors were encountered: {list(set(skipped_loci.values()))}"
            )

        if flat_loci:
            for loc in flat_loci:
                p_values[loc] = 1.0
            self.logger.warning(
                f"{len(flat_loci)} loci have flat distributions: {list(set(flat_loci))}. Setting P-values to 1.0."
            )

        if few_valids:
            for loc in few_valids:
                p_values[loc] = 1.0
            self.logger.warning(
                f"{len(few_valids)} loci have too few valid permutations: {list(set(few_valids))}. Setting P-values to 1.0."
            )

        return observed_fst, p_values

    @staticmethod
    def correct_pvals(pvals: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
        """Apply multiple-testing correction to an array of p-values.

        Args:
            pvals (np.ndarray): 1D array of p-values.
            method (str): Correction method, e.g. 'bonferroni', 'fdr_bh', etc.

        Returns:
            np.ndarray: Corrected p-values (same shape as input).
        """
        mask = ~np.isnan(pvals)
        adjusted = np.full_like(pvals, np.nan)
        if np.any(mask):
            _, adjp, _, _ = multipletests(pvals[mask], method=method)
            adjusted[mask] = adjp
        return adjusted

    def detect_all_pairs(
        self,
        n_perm: int = 1000,
        correction_method: Literal["bonferroni", "fdr_bh"] | None = "fdr_bh",
        alpha: float = 0.05,
        seed: int = 42,
        bandwidth: Literal["silverman", "scott"] | float = "scott",
        alternative: Literal["upper", "lower", "both"] = "both",
    ) -> pd.DataFrame:
        """
        Run per-locus Fst + permutations for all population pairs; return outlier-formatted DataFrame.
        """
        results = []

        for pop1, pop2 in combinations(self.pop_names, 2):
            pop1_inds = self.pop_indices[pop1]
            pop2_inds = self.pop_indices[pop2]

            observed_fst, raw_pvals = self.per_locus_permutation_test(
                pop1_inds,
                pop2_inds,
                n_perm=n_perm,
                seed=seed,
                bandwidth=bandwidth,
                alternative=alternative,
                mode="auto",
            )

            if correction_method is not None:
                adj_pvals = self.correct_pvals(raw_pvals, method=correction_method)
            else:
                adj_pvals = raw_pvals

            for loc_idx, (fst, pval, padj) in enumerate(
                zip(observed_fst, raw_pvals, adj_pvals)
            ):
                results.append(
                    {
                        "Locus": loc_idx,
                        "Population_Pair": f"{pop1}_{pop2}",
                        "Fst": fst,
                        "pval": pval,
                        "pval_adj": padj,
                        "is_outlier": (not np.isnan(padj)) and padj <= alpha,
                    }
                )

        df = pd.DataFrame(results)

        # Identify contributing pairs
        contributing_map = (
            df[df["is_outlier"]]
            .groupby("Locus")["Population_Pair"]
            .apply(lambda x: list(sorted(set(x))))
            .to_dict()
        )

        df["Contributing_Pairs"] = df["Locus"].map(contributing_map)

        df = df[df["pval_adj"] <= alpha]

        # Final column ordering
        return df[
            [
                "Locus",
                "Population_Pair",
                "Fst",
                "pval",
                "pval_adj",
                "Contributing_Pairs",
            ]
        ].reset_index(drop=True)

    def detect_outliers_permutation(
        self,
        n_perm: int = 1000,
        correction_method: Literal["fdr_bh", "bonferroni"] | None = "fdr_bh",
        alpha: float = 0.05,
        seed: int = 42,
        bandwidth: Literal["silverman", "scott"] | float = "scott",
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
        df_fst = self.detect_all_pairs(
            n_perm=n_perm,
            correction_method=correction_method,
            alpha=alpha,
            seed=seed,
            bandwidth=bandwidth,
            alternative=alternative,
        )

        try:
            # Plot the outlier SNPs
            self.plotter.plot_fst_outliers(df_fst, "permutation")

        except Exception as e:
            self.logger.warning(f"Error plotting Fst outliers: {e}")
            self.logger.warning("Skipping Fst outlier plot.")

        return df_fst

    def _compute_empirical_pvals_per_pair(
        self,
        df: pd.DataFrame,
        alternative: Literal["upper", "lower", "both"] = "upper",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compute empirical p-values per population pair using observed Fst and permutation distributions.

        Args:
            df (pd.DataFrame): DataFrame with Fst, Population_Pair, and Locus.
            alternative (Literal["upper", "lower", "both"]): 'upper', 'lower', or 'both'.
            alpha (float): Significance threshold.

        Returns:
            pd.DataFrame: Input DataFrame with added 'pval', 'pval_adj', 'is_outlier'.
        """
        results = []

        for pair, group in df.groupby("Population_Pair"):
            mean_fst = group["Fst"].mean()

            for _, row in group.iterrows():
                locus_fst = row["Fst"]
                dist = group["Fst"].dropna().values

                if alternative == "upper":
                    pval = np.mean(dist >= locus_fst)
                elif alternative == "lower":
                    pval = np.mean(dist <= locus_fst)
                elif alternative == "both":
                    diff = np.abs(locus_fst - mean_fst)
                    pval = np.mean(np.abs(dist - mean_fst) >= diff)
                else:
                    raise ValueError(f"Invalid alternative: {alternative}")

                results.append((row["Locus"], pair, locus_fst, pval))

        # Build DataFrame
        df_pvals = pd.DataFrame(
            results, columns=["Locus", "Population_Pair", "Fst", "pval"]
        )

        # Multiple testing correction
        pvals_array = df_pvals["pval"].to_numpy()
        _, pvals_adj, _, _ = multipletests(pvals_array, method="fdr_bh")
        df_pvals["pval_adj"] = pvals_adj
        df_pvals["is_outlier"] = df_pvals["pval_adj"] <= alpha

        return df_pvals
