import itertools
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from snpio.utils.logging import LoggerManager


class GeneticDistance:
    """Class for computing pairwise Nei's genetic distance between populations with optional permutation and bootstrap inference."""

    def __init__(self, genotype_data, plotter, verbose=False, debug=False):
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.outdir = Path(f"{self.genotype_data.prefix}_output", "analysis")
        self.outdir.mkdir(parents=True, exist_ok=True)
        logman = LoggerManager(
            __name__, self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()

    @staticmethod
    def _clean_inds(inds):
        return [ind for ind in inds if all(x not in ind for x in ("-", "?", "n", "N"))]

    @staticmethod
    def _get_alleles(str_list):
        return sum([x.split("/") for x in str_list], [])

    def _calculate_allele_freqs(self, matrix, indices):
        n_loci = matrix.shape[1]
        allele_freqs = {}

        for locus in range(n_loci):
            genos = [matrix[i, locus] for i in indices]
            clean = self._clean_inds(genos)
            if not clean:
                allele_freqs[locus] = {}
                continue

            alleles = self._get_alleles(clean)
            total = len(alleles)
            counts = pd.Series(alleles).value_counts().to_dict()
            freqs = {a: c / total for a, c in counts.items()}
            allele_freqs[locus] = freqs

        return allele_freqs

    def _neis_distance(self, f1, f2):
        """Calculate Nei's genetic distance (1972) across all loci using global sums.

        Args:
            f1 (dict): Population 1 allele frequencies per locus: {locus: {allele: freq}}.
            f2 (dict): Population 2 allele frequencies per locus: {locus: {allele: freq}}.

        Returns:
            float: Nei's genetic distance D = -log(I), aggregated across loci.
        """
        total_num = 0.0
        total_denom1 = 0.0
        total_denom2 = 0.0

        for locus in f1:
            if locus not in f2:
                continue

            p1 = f1[locus]
            p2 = f2[locus]

            shared = set(p1) & set(p2)
            if not shared:
                continue

            total_num += sum(p1[a] * p2[a] for a in shared)
            total_denom1 += sum(freq**2 for freq in p1.values())
            total_denom2 += sum(freq**2 for freq in p2.values())

        if total_num == 0 or total_denom1 == 0 or total_denom2 == 0:
            return np.nan

        I = total_num / np.sqrt(total_denom1 * total_denom2)
        D = -np.log(I) if I > 0 else np.inf
        return D

    def _get_per_locus_nei(self, pop1_inds, pop2_inds, full_matrix):
        n_loci = full_matrix.shape[1]
        nei_vals = np.full(n_loci, np.nan)

        for loc in range(n_loci):
            genos1 = [full_matrix[i, loc] for i in pop1_inds]
            genos2 = [full_matrix[i, loc] for i in pop2_inds]
            genos1 = self._clean_inds(genos1)
            genos2 = self._clean_inds(genos2)
            if not genos1 or not genos2:
                continue
            f1 = self._calculate_allele_freqs(full_matrix[:, [loc]], list(pop1_inds))
            f2 = self._calculate_allele_freqs(full_matrix[:, [loc]], list(pop2_inds))
            nei_vals[loc] = self._neis_distance(f1, f2)

        return nei_vals

    def _nei_permutation_pval_locus_resampling(
        self, pop1_inds, pop2_inds, full_matrix, n_permutations=1000, seed=42
    ):
        # Precompute allele counts once
        counts1 = self._precompute_allele_counts(full_matrix, list(pop1_inds))
        counts2 = self._precompute_allele_counts(full_matrix, list(pop2_inds))

        loci = list(counts1.keys())

        # Observed Nei distance
        obs_nei = self._neis_distance_from_counts(counts1, counts2)

        rng = np.random.default_rng(seed)
        dist = np.full((n_permutations,), np.nan)

        for i in range(n_permutations):
            resample_idx = rng.choice(loci, size=len(loci), replace=True)

            resampled_counts1 = {loc: counts1[loc] for loc in resample_idx}
            resampled_counts2 = {loc: counts2[loc] for loc in resample_idx}

            dist[i] = self._neis_distance_from_counts(
                resampled_counts1, resampled_counts2
            )

        dist = dist[~np.isnan(dist)]
        pval = (np.sum(dist >= obs_nei) + 1) / (len(dist) + 1)

        return obs_nei, pval, dist

    def nei_distance(self, n_permutations=0, n_jobs=1, return_pvalues=False):
        """Calculate pairwise Nei's genetic distance between populations.

        This method can compute either the observed distance or perform permutation testing to calculate p-values for the distances. If `n_permutations` is set to 0, only the observed distance is calculated. If `n_permutations` is greater than 0, the method will perform the specified number of permutations to calculate p-values for the Nei distances. The method also supports parallel processing using the `n_jobs` parameter.

        Args:
            n_permutations (int): Number of permutations for p-value calculation.
                If 0, only the observed distance is calculated.
            n_jobs (int): Number of parallel jobs to run. Default is 1.
            return_pvalues (bool): If True, return p-values for Nei distances.
                If False, only the observed distances are returned.
        Returns:
            DataFrame: Pairwise Nei distances between populations.
                If n_permutations > 0, also includes p-values.
        """
        self.logger.info(
            f"Calculating Nei's genetic distance with {n_permutations} permutations and {n_jobs} jobs."
        )

        popmap = self.genotype_data.popmap_inverse
        sample_to_idx = {s: i for i, s in enumerate(self.genotype_data.samples)}
        pop_indices = {
            pop: np.array([sample_to_idx[s] for s in popmap[pop]]) for pop in popmap
        }
        pop_keys = list(popmap.keys())
        full_matrix = self.genotype_data.snp_data.astype(str)
        n_loci = full_matrix.shape[1]
        num_pops = len(pop_keys)

        if n_permutations == 0 and not return_pvalues:
            nei_mat = np.full((num_pops, num_pops), np.nan)
            for ia, ib in itertools.combinations(range(num_pops), 2):
                i1 = pop_indices[pop_keys[ia]]
                i2 = pop_indices[pop_keys[ib]]
                f1 = self._calculate_allele_freqs(full_matrix, list(i1))
                f2 = self._calculate_allele_freqs(full_matrix, list(i2))
                dist = self._neis_distance(f1, f2)
                nei_mat[ia, ib] = nei_mat[ib, ia] = dist
            np.fill_diagonal(nei_mat, 0.0)
            df = pd.DataFrame(nei_mat, index=pop_keys, columns=pop_keys)
            outpath = self.outdir / "pairwise_nei_distances.csv"
            df.to_csv(outpath, index=True, float_format="%.8f")
            self.logger.info(f"Nei distance caluclation complete!")
            return df

        elif return_pvalues and n_permutations > 0:
            result = {}
            for ia, ib in itertools.combinations(range(num_pops), 2):
                i1 = pop_indices[pop_keys[ia]]
                i2 = pop_indices[pop_keys[ib]]
                obs, pval, dist = self._nei_permutation_pval_locus_resampling(
                    i1, i2, full_matrix, n_permutations=n_permutations, seed=42
                )
                result[(pop_keys[ia], pop_keys[ib])] = {
                    "nei": obs,
                    "pvalue": pval,
                    "perm_dist": dist,
                }
                self.plotter.plot_permutation_dist(
                    obs, dist, pop_keys[ia], pop_keys[ib], dist_type="nei"
                )
            self.logger.info(f"Nei distance calculation complete!")
            return result

        else:
            result = {
                (pop_keys[ia], pop_keys[ib]): np.zeros(n_permutations, dtype=float)
                for ia, ib in itertools.combinations(range(num_pops), 2)
            }

            def bootstrap(seed):
                rng = np.random.default_rng(seed)
                resample = rng.choice(n_loci, size=n_loci, replace=True)
                replicate = {}
                for ia, ib in itertools.combinations(range(num_pops), 2):
                    i1 = pop_indices[pop_keys[ia]]
                    i2 = pop_indices[pop_keys[ib]]
                    f1 = self._calculate_allele_freqs(
                        full_matrix[:, resample], list(i1)
                    )
                    f2 = self._calculate_allele_freqs(
                        full_matrix[:, resample], list(i2)
                    )
                    replicate[(pop_keys[ia], pop_keys[ib])] = self._neis_distance(
                        f1, f2
                    )

                self.logger.info(f"Nei distance calculation complete!")

                return replicate

            seeds = np.random.default_rng().integers(0, 1e9, size=n_permutations)
            with ThreadPoolExecutor(
                max_workers=(mp.cpu_count() if n_jobs == -1 else n_jobs)
            ) as pool:
                for i, rep in enumerate(pool.map(bootstrap, seeds)):
                    for pair in rep:
                        result[pair][i] = rep[pair]

            self.logger.info(f"Nei distance calculation complete!")
            return result

    def parse_nei_result(self, result_dict, alpha: float = 0.05):
        """Convert the output of `nei_distance()` into DataFrames for:

        - The mean Nei distance among permutations or bootstraps,
        - The lower and upper confidence intervals,
        - And the p-values (if return_pvalues=True).

        This method auto-detects which case of result_dict it has:
        1) A direct DataFrame (case: no bootstrap, no p-values).
        2) A dict {(pop1, pop2): np.array([...])} for bootstrap replicates.
        3) A dict {(pop1, pop2): {"nei": float, "pvalue": pd.Series, "perm_dist": np.array([...])} for permutation results with an optional distribution array.

        Args:
            result_dict: The structure returned by `nei_distance()`.
            alpha (float): Significance level for CIs. Default 0.05 => 95% CIs.

        Returns:
            tuple: (df_mean, df_lower, df_upper, df_pval), where each is a
            pandas DataFrame (or None if not applicable).
            - df_mean: matrix of average Nei distances across replicates (or observed if no replicates).
            - df_lower: matrix of lower CI bounds.
            - df_upper: matrix of upper CI bounds.
            - df_pval: matrix of p-values if p-values exist; otherwise None.

        Notes:
            - For bootstrap results, df_mean, df_lower, and df_upper are computed from the replicate arrays in result_dict.
            - For permutation results, the method looks for "perm_dist" to compute a distribution-based mean and CIs. If "perm_dist" is missing, df_lower and df_upper will remain NaN.
            - If result_dict is just a DataFrame, it returns that as df_mean  and None for the others, since no replicates/p-values exist.
        """
        # 1) If the result is already a DataFrame => case 1
        if isinstance(result_dict, pd.DataFrame):
            # No distribution or p-values to parse, just return the matrix
            return result_dict, None, None, None

        # 2) Otherwise, we have a dictionary of some sort.
        # Inspect the first value to detect the structure.
        first_val = next(iter(result_dict.values()))

        # Helper function to get all populations in alphabetical order
        def all_populations_from_keys(dict_keys):
            """Given dict keys like (pop1, pop2), return a sorted list of unique pops."""
            pop_set = set()
            for k in dict_keys:
                pop_set.update(k)  # k is (pop1, pop2)
            return sorted(pop_set)

        # We'll create empty DataFrames for storing results.
        # We'll fill them if the dictionary structure allows it.
        df_mean, df_lower, df_upper, df_pval = None, None, None, None

        # ------------------------------------------
        # CASE 3 (Bootstrap): (pop1, pop2) -> np.array([...])
        # ------------------------------------------
        if isinstance(first_val, np.ndarray):
            # This indicates we likely have arrays of replicate
            # Nei values => bootstrap
            pop_pairs = list(result_dict.keys())
            pops = all_populations_from_keys(pop_pairs)

            # Initialize empty dataframes
            df_mean = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_lower = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_upper = pd.DataFrame(np.nan, index=pops, columns=pops)

            # Fill each pair from replicate distributions
            for (p1, p2), arr in result_dict.items():
                arr_nonan = arr[~np.isnan(arr)]
                if len(arr_nonan) == 0:
                    # If everything is NaN or there's no data
                    mean_val = np.nan
                    lower_val = np.nan
                    upper_val = np.nan
                else:
                    mean_val = np.mean(arr_nonan)
                    lower_val = np.percentile(arr_nonan, 100 * alpha / 2)
                    upper_val = np.percentile(arr_nonan, 100 * (1 - alpha / 2))

                df_mean.loc[p1, p2] = mean_val
                df_mean.loc[p2, p1] = mean_val
                df_lower.loc[p1, p2] = lower_val
                df_lower.loc[p2, p1] = lower_val
                df_upper.loc[p1, p2] = upper_val
                df_upper.loc[p2, p1] = upper_val

            # Diagonal is typically 0 for self-Nei distances
            np.fill_diagonal(df_mean.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            # p-values don't exist for a standard bootstrap approach
            df_pval = None

            # NOTE: For clarity, df_mean is the actual observed Nei matrix.

            # Write df_lower and df_upper to CSV
            self._combine_upper_lower_ci(df_upper, df_lower, diagonal="zero").to_csv(
                self.outdir / "pairwise_nei_distance_ci95.csv",
                index=True,
                header=True,
                float_format="%.8f",
            )

            return df_mean, df_lower, df_upper, df_pval

        # ------------------------------------------
        # CASE 2 (Permutation): (pop1, pop2) ->
        # {"nei": float, "pvalue": pd.Series, "perm_dist": optional ...}
        # ------------------------------------------
        if isinstance(first_val, dict) and "nei" in first_val and "pvalue" in first_val:
            pop_pairs = list(result_dict.keys())
            pops = all_populations_from_keys(pop_pairs)

            # Observed Nei distances
            df_obs = pd.DataFrame(np.nan, index=pops, columns=pops)

            # We store p-values in df_pval
            df_pval = pd.DataFrame(np.nan, index=pops, columns=pops)

            # If there's a distribution, we can compute mean & CI
            df_mean = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_lower = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_upper = pd.DataFrame(np.nan, index=pops, columns=pops)

            for (p1, p2), subdict in result_dict.items():
                obs_val = subdict["nei"]
                pval_series = pd.Series(subdict["pvalue"])

                # Check if we have a distribution of permuted Nei distances
                # If not, we can still compute mean & CIs
                dist = subdict.get("perm_dist", None)

                # Fill in the observed Nei distance
                df_obs.loc[p1, p2] = obs_val
                df_obs.loc[p2, p1] = obs_val

                # Extract the p-value (one-tailed)
                if not pval_series.empty:
                    p_value = pval_series.mean()
                    df_pval.loc[p1, p2] = p_value
                    df_pval.loc[p2, p1] = p_value

                # If we have a distribution of permuted Nei, compute its mean & CIs
                if dist is not None and len(dist) > 0:
                    dist_nonan = dist[~np.isnan(dist)]
                    if len(dist_nonan) == 0:
                        mean_val = np.nan
                        lower_val = np.nan
                        upper_val = np.nan
                    else:
                        mean_val = np.mean(dist_nonan)
                        lower_val = np.percentile(dist_nonan, 100 * alpha / 2)
                        upper_val = np.percentile(dist_nonan, 100 * (1 - alpha / 2))

                    df_mean.loc[p1, p2] = mean_val
                    df_mean.loc[p2, p1] = mean_val
                    df_lower.loc[p1, p2] = lower_val
                    df_lower.loc[p2, p1] = lower_val
                    df_upper.loc[p1, p2] = upper_val
                    df_upper.loc[p2, p1] = upper_val

            # Fill diagonals with typical defaults
            np.fill_diagonal(df_obs.values, 0.0)
            np.fill_diagonal(df_pval.values, 1.0)
            np.fill_diagonal(df_mean.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            # Write P-values to CSV
            df_pval.to_csv(
                self.outdir / "pairwise_nei_distance_pvalues.csv",
                index=True,
                header=True,
                float_format="%.8f",
            )

            df_obs.to_csv(
                self.outdir / "pairwise_nei_distance.csv",
                index=True,
                header=True,
                float_format="%.8f",
            )

            self.logger.info(f"Nei distance files saved to {self.outdir}")

            return df_obs, df_lower, df_upper, df_pval

        # ------------------------------------------
        # If none of the above matched, raise error
        # ------------------------------------------
        msg = "Unrecognized structure in result_dict. Expected either a DataFrame or a dictionary with specific keys and structures."
        self.logger.error(msg)
        raise ValueError(msg)

    def _combine_upper_lower_ci(
        self, df_upper: pd.DataFrame, df_lower: pd.DataFrame, diagonal="zero"
    ) -> pd.DataFrame:
        """Combines two square DataFrames into one, using the upper triangle from one and the lower from the other.

        Args:
            df_upper (pd.DataFrame): DataFrame to provide the upper triangle (including diagonal if diagonal='upper').
            df_lower (pd.DataFrame): DataFrame to provide the lower triangle (including diagonal if diagonal='lower').
            diagonal (str): Which DataFrame should provide the diagonal values. Options are 'upper', 'lower', "zero", or 'nan'.

        Returns:
            pd.DataFrame: Combined DataFrame with upper/lower triangle values from respective inputs.

        Raises:
            ValueError: If input DataFrames are not square or do not match in shape.
        """
        if df_upper.shape != df_lower.shape:
            msg = "Both DataFrames must have the same shape."
            self.logger.error(msg)
            raise AssertionError(msg)

        if df_upper.shape[0] != df_upper.shape[1]:
            msg = "Input DataFrames must be square."
            self.logger.error(msg)
            raise AssertionError(msg)

        n = df_upper.shape[0]

        upper_mask = np.triu(np.ones((n, n)), k=1)
        lower_mask = np.tril(np.ones((n, n)), k=-1)
        diag_mask = np.eye(n)

        combined = np.full_like(df_upper, np.nan, dtype="float64")

        combined[upper_mask.astype(bool)] = df_upper.values[upper_mask.astype(bool)]
        combined[lower_mask.astype(bool)] = df_lower.values[lower_mask.astype(bool)]

        if diagonal == "upper":
            combined[diag_mask.astype(bool)] = df_upper.values[diag_mask.astype(bool)]
        elif diagonal == "lower":
            combined[diag_mask.astype(bool)] = df_lower.values[diag_mask.astype(bool)]
        elif diagonal == "nan":
            pass  # leave diagonal as NaN
        elif diagonal == "zero":
            np.fill_diagonal(combined, 0.0)
        else:
            msg = (
                "Invalid option for 'diagonal'. Choose from 'upper', 'lower', or 'nan'."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return pd.DataFrame(combined, index=df_upper.index, columns=df_upper.columns)

    def _precompute_allele_counts(self, matrix, indices):
        """Precompute allele counts per locus for a given population."""
        n_loci = matrix.shape[1]
        allele_counts = {}

        for locus in range(n_loci):
            genos = [matrix[i, locus] for i in indices]
            clean = self._clean_inds(genos)
            if not clean:
                allele_counts[locus] = {}
                continue

            alleles = self._get_alleles(clean)
            counts = pd.Series(alleles).value_counts().to_dict()
            allele_counts[locus] = counts

        return allele_counts

    def _neis_distance_from_counts(self, counts1, counts2):
        """Calculate Nei's genetic distance from allele count dictionaries."""
        total_num = 0.0
        total_denom1 = 0.0
        total_denom2 = 0.0

        for locus in counts1:
            if locus not in counts2:
                continue

            p1_counts = counts1[locus]
            p2_counts = counts2[locus]

            total1 = sum(p1_counts.values())
            total2 = sum(p2_counts.values())

            if total1 == 0 or total2 == 0:
                continue

            p1_freqs = {a: c / total1 for a, c in p1_counts.items()}
            p2_freqs = {a: c / total2 for a, c in p2_counts.items()}

            shared = set(p1_freqs) & set(p2_freqs)
            if not shared:
                continue

            total_num += sum(p1_freqs[a] * p2_freqs[a] for a in shared)
            total_denom1 += sum(f**2 for f in p1_freqs.values())
            total_denom2 += sum(f**2 for f in p2_freqs.values())

        if total_num == 0 or total_denom1 == 0 or total_denom2 == 0:
            return np.nan

        I = total_num / np.sqrt(total_denom1 * total_denom2)
        D = -np.log(I) if I > 0 else np.inf
        return D
