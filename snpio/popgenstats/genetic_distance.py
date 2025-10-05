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

    @staticmethod
    def _get_alleles(
        iupac_decoder: dict[str, tuple[str, str]], iupac_list: list[str]
    ) -> list[str]:
        """Decodes a list of IUPAC codes into a flat list of alleles.

        Args:
            iupac_decoder: Mapping like {'R': ('A','G'), ...}.
            iupac_list: List of IUPAC genotype codes (string) per individual.

        Returns:
            Flat list of single-letter alleles for the input list.
        """
        alleles: list[str] = []
        for iupac_code in iupac_list:
            decoded_pair = iupac_decoder.get(iupac_code, ())
            alleles.extend(decoded_pair)
        return alleles

    @staticmethod
    def _calculate_allele_freqs(
        matrix: np.ndarray,
        indices: list[int],
        iupac_decoder: dict[str, tuple[str, str]],
    ) -> dict[int, dict[str, float]]:
        """Compute per-locus allele frequencies for a subset of individuals.

        Args:
            matrix: SNP matrix (individuals x loci) as string IUPAC codes.
            indices: Selected individual indices.
            iupac_decoder: Mapping like {'R': ('A','G'), ...}.

        Returns:
            Dict: locus_index -> {allele: frequency}.
        """
        n_loci = matrix.shape[1]
        allele_freqs: dict[int, dict[str, float]] = {}

        for locus in range(n_loci):
            genos = [matrix[i, locus] for i in indices]
            # Reuse class-level missing filter (doesn't capture self)
            clean = GeneticDistance._clean_inds(genos)
            if not clean:
                allele_freqs[locus] = {}
                continue

            alleles = GeneticDistance._get_alleles(iupac_decoder, clean)
            total = len(alleles)
            if total == 0:
                allele_freqs[locus] = {}
                continue
            # Use numpy for speed; small dict at the end is fine
            vals, counts = np.unique(np.asarray(alleles), return_counts=True)
            freqs = {a: c / total for a, c in zip(vals.tolist(), counts.tolist())}
            allele_freqs[locus] = freqs

        return allele_freqs

    @staticmethod
    def _neis_distance(
        f1: dict[int, dict[str, float]], f2: dict[int, dict[str, float]]
    ) -> float:
        """Calculate Nei's genetic distance (Nei 1972) across all loci using global sums."""
        total_num = 0.0
        total_denom1 = 0.0
        total_denom2 = 0.0

        # Iterate only over loci present in both dicts
        for locus, p1 in f1.items():
            p2 = f2.get(locus)
            if not p2:
                continue
            shared = set(p1) & set(p2)
            if not shared:
                continue
            total_num += sum(p1[a] * p2[a] for a in shared)
            total_denom1 += sum(freq * freq for freq in p1.values())
            total_denom2 += sum(freq * freq for freq in p2.values())

        if total_num == 0.0 or total_denom1 == 0.0 or total_denom2 == 0.0:
            return np.nan

        I = total_num / np.sqrt(total_denom1 * total_denom2)
        return -np.log(I) if I > 0.0 else np.inf

    @staticmethod
    def _nei_permutation_pvalue(
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
        iupac_decoder: dict[str, tuple[str, str]],
        n_permutations: int,
        seed: int,
    ) -> tuple[float, float, np.ndarray]:
        """Calculates a p-value for Nei distance using a permutation test (pure/static)."""
        f1_obs = GeneticDistance._calculate_allele_freqs(
            full_matrix, list(pop1_inds), iupac_decoder
        )
        f2_obs = GeneticDistance._calculate_allele_freqs(
            full_matrix, list(pop2_inds), iupac_decoder
        )
        obs_nei = GeneticDistance._neis_distance(f1_obs, f2_obs)

        pooled_indices = np.concatenate((pop1_inds, pop2_inds))
        n1_size = len(pop1_inds)
        perm_distances = np.full(n_permutations, np.nan)
        rng = np.random.default_rng(seed)

        for i in range(n_permutations):
            rng.shuffle(pooled_indices)
            perm_inds1 = pooled_indices[:n1_size]
            perm_inds2 = pooled_indices[n1_size:]
            f1_perm = GeneticDistance._calculate_allele_freqs(
                full_matrix, list(perm_inds1), iupac_decoder
            )
            f2_perm = GeneticDistance._calculate_allele_freqs(
                full_matrix, list(perm_inds2), iupac_decoder
            )
            perm_distances[i] = GeneticDistance._neis_distance(f1_perm, f2_perm)

        perm_distances = perm_distances[~np.isnan(perm_distances)]
        p_value = (
            (np.sum(perm_distances >= obs_nei) + 1) / (len(perm_distances) + 1)
            if len(perm_distances) > 0
            else np.nan
        )
        return obs_nei, p_value, perm_distances

    @staticmethod
    def _permutation_worker(
        pop_pair_keys: tuple[str, str],
        pop_indices: dict[str, np.ndarray],
        full_matrix: np.ndarray,
        iupac_decoder: dict[str, tuple[str, str]],
        n_reps: int,
        seed: int,
    ):
        """Top-level/staticmethod worker: runs Nei permutation for one pair with its own seed."""
        p1_key, p2_key = pop_pair_keys
        i1 = pop_indices[p1_key]
        i2 = pop_indices[p2_key]

        obs, pval, dist = GeneticDistance._nei_permutation_pvalue(
            i1, i2, full_matrix, iupac_decoder, n_permutations=n_reps, seed=seed
        )
        return (p1_key, p2_key), {"nei": obs, "pvalue": pval, "perm_dist": dist}

    @staticmethod
    def _bootstrap_replicate(
        seed: int,
        n_loci: int,
        full_matrix: np.ndarray,
        iupac_decoder: dict[str, tuple[str, str]],
        pop_pairs: list[tuple[str, str]],
        pop_indices: dict[str, np.ndarray],
    ) -> dict[tuple[str, str], float]:
        """Top-level/staticmethod worker: one bootstrap replicate over loci (pure/static)."""
        rng = np.random.default_rng(seed)
        resampled_loci = rng.choice(n_loci, size=n_loci, replace=True)
        resampled_matrix = full_matrix[:, resampled_loci]

        replicate: dict[tuple[str, str], float] = {}
        for p1_key, p2_key in pop_pairs:
            i1 = pop_indices[p1_key]
            i2 = pop_indices[p2_key]
            f1 = GeneticDistance._calculate_allele_freqs(
                resampled_matrix, list(i1), iupac_decoder
            )
            f2 = GeneticDistance._calculate_allele_freqs(
                resampled_matrix, list(i2), iupac_decoder
            )
            replicate[(p1_key, p2_key)] = GeneticDistance._neis_distance(f1, f2)

        return replicate

    def nei_distance(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
    ):
        """Calculate pairwise Nei's genetic distance with optional statistical tests.

        Args:
            method (Literal["observed", "permutation", "bootstrap"]): The method to use for calculating Nei's distance.
            n_reps (int): The number of replicates to use for permutation and bootstrap methods.
            n_jobs (int): The number of parallel jobs to use. -1 uses all available cores.

        Returns:
            dict: A dictionary containing the results of the Nei distance calculation.
            If method is "observed":
                pd.DataFrame: A symmetric matrix of Nei distances between populations.

            If method is "permutation":
                dict: A dictionary where keys are (pop1, pop2) tuples and values are dicts with:
                    - "nei": observed Nei distance (float)
                    - "pvalue": p-value from permutation test (float)
                    - "perm_dist": array of Nei distances from permutations (np.ndarray)

            If method is "bootstrap":
                dict: A dictionary where keys are (pop1, pop2) tuples and values are np.ndarrays of Nei distances from bootstrap replicates.
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
        iupac_decoder = self.genotype_data.reverse_iupac_mapping

        if method == "observed":
            nei_mat = np.full((num_pops, num_pops), np.nan)
            for ia, ib in itertools.combinations(range(num_pops), 2):
                p1_key, p2_key = pop_keys[ia], pop_keys[ib]
                i1, i2 = pop_indices[p1_key], pop_indices[p2_key]
                f1 = GeneticDistance._calculate_allele_freqs(
                    full_matrix, list(i1), iupac_decoder
                )
                f2 = GeneticDistance._calculate_allele_freqs(
                    full_matrix, list(i2), iupac_decoder
                )
                dist = GeneticDistance._neis_distance(f1, f2)
                nei_mat[ia, ib] = nei_mat[ib, ia] = dist

            np.fill_diagonal(nei_mat, 0.0)
            df = pd.DataFrame(nei_mat, index=pop_keys, columns=pop_keys)
            self.logger.info("Nei distance calculation complete!")
            return df

        elif method == "permutation":
            result: dict[tuple[str, str], dict[str, object]] = {}
            pop_pairs = list(itertools.combinations(pop_keys, 2))

            # --- per-pair independent seeds ---
            base_seed = self.seed if self.seed is not None else 0
            ss = np.random.SeedSequence(base_seed)
            children = ss.spawn(len(pop_pairs))
            pair_seeds = {
                pair: int(child.entropy) & 0x7FFFFFFF  # stable 31-bit int
                for pair, child in zip(pop_pairs, children)
            }

            from concurrent.futures import as_completed

            max_workers = mp.cpu_count() if n_jobs == -1 else n_jobs
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        GeneticDistance._permutation_worker,
                        pair,
                        pop_indices,
                        full_matrix,
                        iupac_decoder,
                        n_reps,
                        pair_seeds[pair],
                    )
                    for pair in pop_pairs
                ]
                for fut in tqdm(
                    as_completed(futures), desc="Nei permutations", total=len(futures)
                ):
                    pop_pair, res_dict = fut.result()
                    result[pop_pair] = res_dict

            # Plot after collecting results
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

        elif method == "bootstrap":
            pop_pairs = list(itertools.combinations(pop_keys, 2))
            result = {pair: np.zeros(n_reps, dtype=float) for pair in pop_pairs}

            worker_func = partial(
                GeneticDistance._bootstrap_replicate,
                n_loci=n_loci,
                full_matrix=full_matrix,
                iupac_decoder=iupac_decoder,
                pop_pairs=pop_pairs,
                pop_indices=pop_indices,
            )

            seeds = rng.integers(0, 1_000_000_000, size=n_reps)

            with ProcessPoolExecutor(
                max_workers=(mp.cpu_count() if n_jobs == -1 else n_jobs)
            ) as pool:
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
