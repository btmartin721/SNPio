import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from snpio.utils.logging import LoggerManager
from snpio.utils.multiqc_reporter import SNPioMultiQC

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData
    from snpio.plotting.plotting import Plotting


class GeneticDistance:
    """Class for computing pairwise Nei's genetic distance between populations with optional permutation and bootstrap inference."""

    def __init__(
        self,
        genotype_data: "GenotypeData",
        plotter: "Plotting",
        verbose: bool = False,
        debug: bool = False,
        seed: int | None = None,
    ) -> None:
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.seed = seed

        if self.genotype_data.was_filtered:
            self.outdir = Path(
                f"{self.genotype_data.prefix}_output", "nremover", "analysis"
            )
        else:
            self.outdir = Path(f"{self.genotype_data.prefix}_output", "analysis")

        self.outdir.mkdir(parents=True, exist_ok=True)

        logman = LoggerManager(
            __name__, self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()

        self.iupac_decoder = self.genotype_data.reverse_iupac_mapping
        self.snpio_mqc = SNPioMultiQC

    @staticmethod
    def _clean_inds(inds: list[str]) -> list[str]:
        """Filters out IUPAC genotypes that are missing data codes."""
        missing_codes = {"N", "-", "?", "."}
        return [ind for ind in inds if ind not in missing_codes]

    def _get_alleles(self, iupac_list: list[str]) -> list[str]:
        """Decodes a list of IUPAC codes into a flat list of alleles."""
        alleles = []
        for iupac_code in iupac_list:
            # The decoder returns a tuple like ('A', 'G') for 'R'
            # It defaults to an empty tuple for unknown/missing codes
            decoded_pair = self.iupac_decoder.get(iupac_code, ())
            alleles.extend(decoded_pair)
        return alleles

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
        """Calculate Nei's genetic distance (1972) across all loci using global sums."""
        total_num, total_denom1, total_denom2 = 0.0, 0.0, 0.0
        for locus in f1:
            if locus not in f2:
                continue
            p1, p2 = f1[locus], f2[locus]
            shared = set(p1) & set(p2)
            if not shared:
                continue
            total_num += sum(p1[a] * p2[a] for a in shared)
            total_denom1 += sum(freq**2 for freq in p1.values())
            total_denom2 += sum(freq**2 for freq in p2.values())

        if total_num == 0 or total_denom1 == 0 or total_denom2 == 0:
            return np.nan

        I = total_num / np.sqrt(total_denom1 * total_denom2)
        return -np.log(I) if I > 0 else np.inf

    def _nei_permutation_pvalue(
        self, pop1_inds, pop2_inds, full_matrix, n_permutations=1000, seed=42
    ):
        """Calculates a p-value for Nei distance using a permutation test."""
        f1_obs = self._calculate_allele_freqs(full_matrix, list(pop1_inds))
        f2_obs = self._calculate_allele_freqs(full_matrix, list(pop2_inds))
        obs_nei = self._neis_distance(f1_obs, f2_obs)

        pooled_indices = np.concatenate((pop1_inds, pop2_inds))
        n1_size = len(pop1_inds)
        perm_distances = np.full(n_permutations, np.nan)
        rng = np.random.default_rng(seed)

        for i in range(n_permutations):
            rng.shuffle(pooled_indices)
            perm_inds1 = pooled_indices[:n1_size]
            perm_inds2 = pooled_indices[n1_size:]
            f1_perm = self._calculate_allele_freqs(full_matrix, list(perm_inds1))
            f2_perm = self._calculate_allele_freqs(full_matrix, list(perm_inds2))
            perm_distances[i] = self._neis_distance(f1_perm, f2_perm)

        perm_distances = perm_distances[~np.isnan(perm_distances)]
        p_value = (np.sum(perm_distances >= obs_nei) + 1) / (len(perm_distances) + 1)
        return obs_nei, p_value, perm_distances

    def _permutation_worker(
        self,
        pop_pair_keys: tuple[str, str],
        pop_indices: dict[str, np.ndarray],
        full_matrix: np.ndarray,
        n_reps: int,
        seed: int,
    ):
        """Worker function for a single permutation test between two populations.

        Args:
            pop_pair_keys (tuple[str, str]): A tuple containing the two population names.
            pop_indices (dict[str, np.ndarray]): A mapping of population names to their sample indices
            full_matrix (np.ndarray): The full genotype matrix.
            n_reps (int): Number of permutations to perform.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple: A tuple containing the population pair keys and a dictionary with observed Nei distance, p-value, and permutation distribution.
        """
        p1_key, p2_key = pop_pair_keys
        i1 = pop_indices[p1_key]
        i2 = pop_indices[p2_key]

        obs, pval, dist = self._nei_permutation_pvalue(
            i1, i2, full_matrix, n_permutations=n_reps, seed=seed
        )

        # Return the key with the result to reassemble the final dictionary
        return (p1_key, p2_key), {"nei": obs, "pvalue": pval, "perm_dist": dist}

    def _bootstrap_replicate(
        self,
        seed: int,
        n_loci: int,
        full_matrix: np.ndarray,
        pop_pairs: list[tuple[str, str]],
        pop_indices: dict[str, np.ndarray],
    ) -> dict[tuple[str, str], float]:
        """Worker function for a single bootstrap replicate for Nei's distance.

        Args:
            seed (int): Random seed for reproducibility.
            n_loci (int): Total number of loci in the dataset.
            full_matrix (np.ndarray): The full genotype matrix.
            pop_pairs (list[tuple[str, str]]): List of population pairs to compute distances for.
            pop_indices (dict[str, np.ndarray]): Mapping of population names to their sample indices.

        Returns:
            dict[tuple[str, str], float]: A dictionary mapping population pairs to their Nei distance for this replicate.
        """
        rng = np.random.default_rng(seed)
        resampled_loci = rng.choice(n_loci, size=n_loci, replace=True)
        replicate = {}
        resampled_matrix = full_matrix[:, resampled_loci]

        for p1_key, p2_key in pop_pairs:
            i1 = pop_indices[p1_key]
            i2 = pop_indices[p2_key]
            f1 = self._calculate_allele_freqs(resampled_matrix, list(i1))
            f2 = self._calculate_allele_freqs(resampled_matrix, list(i2))
            replicate[(p1_key, p2_key)] = self._neis_distance(f1, f2)
        return replicate

    def nei_distance(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
    ):
        """Calculate pairwise Nei's genetic distance with optional statistical tests.

        Args:
            method (str): The calculation method to use.'observed': (Default) Computes the observed Nei's distance matrix. 'permutation': Performs a permutation test to calculate p-values for the observed distances. 'bootstrap': Performs a bootstrap by resampling loci to generate confidence intervals.
            n_reps (int): The number of replicates for 'permutation' or 'bootstrap' methods. Default is 1000.
            n_jobs (int): Number of parallel jobs to run. Default is 1. Use -1 for all available CPUs.

        Returns:
            Tuple[pd.DataFrame | dict, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]: Depending on the method, returns either pd.DataFrame for 'observed' method, a matrix of pairwise distances. dict for 'permutation' or 'bootstrap' methods, a dictionary containing results. Use `parse_nei_result` to convert this dict to DataFrames.
        """
        self.logger.info(
            f"Calculating Nei's distance using method='{method}' with {n_reps} replicates and {n_jobs} jobs."
        )

        if n_jobs == 0 or n_jobs < -1:
            msg = f"n_jobs must be -1 or a positive integer, but got: {n_jobs}"
            self.logger.error(msg)
            raise ValueError(msg)

        rng = np.random.default_rng(self.seed)

        popmap = self.genotype_data.popmap_inverse
        sample_to_idx = {s: i for i, s in enumerate(self.genotype_data.samples)}
        pop_indices = {
            pop: np.array([sample_to_idx[s] for s in popmap[pop]]) for pop in popmap
        }
        pop_keys = sorted(popmap.keys())
        full_matrix = self.genotype_data.snp_data.astype(str)
        n_loci = full_matrix.shape[1]
        num_pops = len(pop_keys)

        # BRANCH 1: Calculate observed distance only
        if method == "observed":
            nei_mat = np.full((num_pops, num_pops), np.nan)
            for ia, ib in itertools.combinations(range(num_pops), 2):
                p1_key, p2_key = pop_keys[ia], pop_keys[ib]
                i1, i2 = pop_indices[p1_key], pop_indices[p2_key]
                f1 = self._calculate_allele_freqs(full_matrix, list(i1))
                f2 = self._calculate_allele_freqs(full_matrix, list(i2))
                dist = self._neis_distance(f1, f2)
                nei_mat[ia, ib] = nei_mat[ib, ia] = dist

            np.fill_diagonal(nei_mat, 0.0)
            df = pd.DataFrame(nei_mat, index=pop_keys, columns=pop_keys)
            self.logger.info("Nei distance calculation complete!")
            return df

        # BRANCH 2: Perform permutation test for p-values
        elif method == "permutation":
            result = {}
            pop_pairs = list(itertools.combinations(pop_keys, 2))

            # Create a partial function with the arguments that are the same for all jobs
            worker_func = partial(
                self._permutation_worker,
                pop_indices=pop_indices,
                full_matrix=full_matrix,
                n_reps=n_reps,
                seed=self.seed,
            )

            # Run the jobs in parallel
            with ProcessPoolExecutor(
                max_workers=(mp.cpu_count() if n_jobs == -1 else n_jobs)
            ) as pool:
                # The result is an iterator of ((pop1, pop2), {results_dict})
                res_iterator = pool.map(worker_func, pop_pairs)
                for pop_pair, res_dict in tqdm(
                    res_iterator, desc="Nei permutations", total=len(pop_pairs)
                ):
                    result[pop_pair] = res_dict

            # Plotting should be done after all parallel results are collected
            for (p1_key, p2_key), res_dict in result.items():
                self.plotter.plot_permutation_dist(
                    res_dict["nei"],
                    res_dict["perm_dist"],
                    p1_key,
                    p2_key,
                    dist_type="nei",
                )

            self.logger.info("Nei distance permutation test complete!")
            return result

        # BRANCH 3: Perform bootstrap for confidence intervals
        elif method == "bootstrap":
            pop_pairs = list(itertools.combinations(pop_keys, 2))
            result = {pair: np.zeros(n_reps, dtype=float) for pair in pop_pairs}

            # Use functools.partial to create a pickleable worker function
            worker_func = partial(
                self._bootstrap_replicate,
                n_loci=n_loci,
                full_matrix=full_matrix,
                pop_pairs=pop_pairs,
                pop_indices=pop_indices,
            )

            # Use the master rng to generate a deterministic list of seeds
            seeds = rng.integers(0, 1e9, size=n_reps)

            with ProcessPoolExecutor(
                max_workers=(mp.cpu_count() if n_jobs == -1 else n_jobs)
            ) as pool:
                # Map the new worker function over the seeds
                reps = pool.map(worker_func, seeds)
                for i, rep_data in tqdm(
                    enumerate(reps), desc="Nei bootstrapping", total=n_reps
                ):
                    for pair, dist_val in rep_data.items():
                        result[pair][i] = dist_val

            self.logger.info("Nei distance bootstrap complete!")
            return result

        else:
            msg = f"Unknown method '{method}'. Choose from 'observed', 'permutation', or 'bootstrap'."
            self.logger.error(msg)
            raise ValueError(msg)

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

    def parse_nei_result(self, result_dict: dict, alpha: float = 0.05) -> tuple[
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
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
            tuple: (df_mean, df_lower, df_upper, df_pval), where each is a pandas DataFrame (or None if not applicable).
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

        # Create empty DataFrames for storing results.
        # Fill them if the dictionary structure allows it.
        df_mean = df_lower = df_upper = df_pval = None

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
                # Extract the distribution of permuted Nei distances if exists
                # If not, we can still compute mean & CIs
                dist = subdict.get("perm_dist", None)

                # Fill in the observed Nei distance
                df_obs.loc[p1, p2] = obs_val
                df_obs.loc[p2, p1] = obs_val

                # Extract the p-value (one-tailed)
                p_value = subdict["pvalue"]
                df_pval.loc[p1, p2] = p_value
                df_pval.loc[p2, p1] = p_value

                # If have a distribution of permuted Nei, compute its mean & CIs
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

            self.logger.info(f"Nei distance files saved to {self.outdir}")

            return df_obs, df_lower, df_upper, df_pval

        # If none of the above matched, raise error
        msg = "Unrecognized structure in result_dict. Expected either a DataFrame or a dictionary with specific keys and structures."
        self.logger.error(msg)
        raise ValueError(msg)
