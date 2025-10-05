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
from snpio.utils.misc import IUPAC
from snpio.utils.multiqc_reporter import SNPioMultiQC

if TYPE_CHECKING:
    from snpio.plotting.plotting import Plotting
    from snpio.read_input.genotype_data import GenotypeData

phased_encoding = IUPAC().get_phased_encoding()


class FstDistance:
    """Class for calculating Fst distance between populations using Weir and Cockerham's method.

    This class implements Weir and Cockerham's (1984) Fst calculation for two populations, s1 and s2. It supports:

    - Observed Fst calculation
    - Permutation tests for statistical significance
    - Bootstrap resampling for confidence intervals

    References:
        ..[1] Weir, B. S., & Cockerham, C. C. (1984). Estimating F-statistics for the analysis of population structure. Evolution, 38(6), 1358-1370.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        plotter: "Plotting",
        seed: int | None = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.verbose = verbose
        self.seed = seed

        if self.genotype_data.was_filtered:
            self.outdir = Path(
                f"{genotype_data.prefix}_output", "nremover", "analysis", "fst"
            )
        else:
            self.outdir = Path(f"{genotype_data.prefix}_output", "analysis", "fst")
        self.outdir.mkdir(parents=True, exist_ok=True)

        logman = LoggerManager(
            __name__, self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()
        self.iupac = IUPAC(logger=self.logger)
        self.snpio_mqc = SNPioMultiQC

    @staticmethod
    def _get_alleles(phased_list: list[str]) -> list[str]:
        """Flattens a list of 'A/T' strings into a list of single alleles."""
        return [allele for genotype in phased_list for allele in genotype.split("/")]

    @staticmethod
    def _get_het_from_phased(allele: str, phased_list: list[str], count: bool = False):
        """Calculates observed heterozygosity for an allele from phased genotypes."""
        hets = 0.0
        for genotype in phased_list:
            a1, a2 = genotype.split("/")
            if (a1 == allele and a2 != allele) or (a2 == allele and a1 != allele):
                hets += 1.0

        if count:
            return hets
        else:
            # Correctly divide by number of individuals (N), not 2N
            return hets / len(phased_list) if phased_list else 0.0

    @staticmethod
    def _clean_inds(inds: list[str]) -> list[str]:
        """Filters out IUPAC genotypes that are missing data codes."""
        missing_codes = {"N", "-", "?", "."}
        return [ind for ind in inds if ind not in missing_codes]

    @staticmethod
    def _permutation_worker(
        pop_pair_keys: tuple[str, str],
        pop_indices: dict[str, np.ndarray],
        full_matrix: np.ndarray,
        n_reps: int,
        seed: int,
    ):
        """Worker function for a single Fst permutation test between two populations.

        Args:
            pop_pair_keys (tuple[str, str]): Tuple of population keys (pop1, pop2) to compare.
            pop_indices (dict[str, np.ndarray]): Dictionary mapping population keys to individual indices in the genotype matrix.
            full_matrix (np.ndarray): Full genotype matrix with shape (num_individuals, num_loci).
            n_reps (int): Number of permutations to perform for p-value estimation.
            seed (int): Random seed for reproducibility in permutations.

        Returns:
            tuple: ((pop1, pop2), {"fst": float, "pvalue": float, "perm_dist": np.ndarray})
        """
        p1_key, p2_key = pop_pair_keys
        i1 = pop_indices[p1_key]
        i2 = pop_indices[p2_key]

        obs, pval, dist = FstDistance._fst_permutation_pvalue(
            i1, i2, full_matrix, n_permutations=n_reps, seed=seed
        )

        # Return the key with the result to reassemble the final dictionary
        return (p1_key, p2_key), {"fst": obs, "pvalue": pval, "perm_dist": dist}

    @staticmethod
    def _two_pop_weir_cockerham_fst_locus(s1: list[str], s2: list[str]):
        """Computes Weir & Cockerham's Fst components for a SINGLE locus.

        Edge-case guard:
            - If either pop has <1 individual => (0,0)
            - If n_bar <= 1 or nC == 0 (e.g., n1==n2==1) => (0,0)

        Args:
            s1 (list[str]): Phased genotypes for population 1.
            s2 (list[str]): Phased genotypes for population 2.

        Returns:
            tuple: A tuple containing the Fst numerator and denominator.
        """
        if not s1 or not s2:
            return 0.0, 0.0

        alleles1 = FstDistance._get_alleles(s1)
        alleles2 = FstDistance._get_alleles(s2)
        unique_alleles = set(alleles1) | set(alleles2)

        r = 2.0
        n1, n2 = float(len(s1)), float(len(s2))

        if n1 < 1.0 or n2 < 1.0:
            return 0.0, 0.0

        n_bar = (n1 + n2) / r

        if n_bar <= 1.0:
            return 0.0, 0.0

        denom_nc = n1 + n2
        if denom_nc == 0.0:
            return 0.0, 0.0

        nC = (1.0 / (r - 1.0)) * (n1 + n2 - (n1**2 + n2**2) / denom_nc)

        if nC == 0.0:
            return 0.0, 0.0

        num, denom = 0.0, 0.0
        for allele in unique_alleles:
            p1 = alleles1.count(allele) / (2.0 * n1)
            p2 = alleles2.count(allele) / (2.0 * n2)
            p_bar = (n1 * p1 + n2 * p2) / (n1 + n2)

            h1 = FstDistance._get_het_from_phased(allele, s1, count=True) / n1
            h2 = FstDistance._get_het_from_phased(allele, s2, count=True) / n2
            h_bar = (n1 * h1 + n2 * h2) / (n1 + n2)

            s_squared = ((n1 * (p1 - p_bar) ** 2) + (n2 * (p2 - p_bar) ** 2)) / (
                (r - 1.0) * n_bar
            )

            a = (n_bar / nC) * (
                s_squared
                - (1.0 / (n_bar - 1.0))
                * (p_bar * (1.0 - p_bar) - ((r - 1.0) / r) * s_squared - h_bar / 4.0)
            )
            b = (n_bar / (n_bar - 1.0)) * (
                p_bar * (1.0 - p_bar)
                - ((r - 1.0) / r) * s_squared
                - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar
            )
            c = h_bar / 2.0

            num += a
            denom += a + b + c

        return num, denom

    @staticmethod
    def _compute_multilocus_fst(pop1_inds, pop2_inds, full_matrix):
        """Computes the final Fst by summing single-locus components."""
        total_num, total_denom = 0.0, 0.0
        for loc in range(full_matrix.shape[1]):
            # 1. Extract IUPAC codes for the locus
            s1_iupac = full_matrix[pop1_inds, loc]
            s2_iupac = full_matrix[pop2_inds, loc]

            # 2. Decode IUPAC to phased strings (e.g., 'R' -> 'A/G')
            s1_phased = [phased_encoding.get(gt, "N/N") for gt in s1_iupac]
            s2_phased = [phased_encoding.get(gt, "N/N") for gt in s2_iupac]

            # 3. Clean missing data
            s1_clean = [gt for gt in s1_phased if "N" not in gt]
            s2_clean = [gt for gt in s2_phased if "N" not in gt]

            # 4. Calculate components for this locus
            num, denom = FstDistance._two_pop_weir_cockerham_fst_locus(
                s1_clean, s2_clean
            )

            if not np.isnan(num) and not np.isnan(denom):
                total_num += num
                total_denom += denom

        return total_num / total_denom if total_denom != 0 else np.nan

    @staticmethod
    def _fst_permutation_pvalue(
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
        n_permutations: int,
        seed: int,
    ):
        """Calculates a p-value for Fst using a permutation test.

        Args:
            pop1_inds (np.ndarray): Indices of individuals in population
            pop2_inds (np.ndarray): Indices of individuals in population 2
            full_matrix (np.ndarray): Full genotype matrix
            n_permutations (int): Number of permutations to perform
            seed (int): Random seed for reproducibility

        Returns:
            Tuple[float, float, np.ndarray]: (observed Fst, p-value, permutation distribution array)
        """
        obs_fst = FstDistance._compute_multilocus_fst(pop1_inds, pop2_inds, full_matrix)

        pooled_indices = np.concatenate((pop1_inds, pop2_inds))
        n1_size = len(pop1_inds)
        perm_dist = np.full(n_permutations, np.nan)
        rng = np.random.default_rng(seed)

        for i in range(n_permutations):
            rng.shuffle(pooled_indices)
            perm_inds1 = pooled_indices[:n1_size]
            perm_inds2 = pooled_indices[n1_size:]
            perm_dist[i] = FstDistance._compute_multilocus_fst(
                perm_inds1, perm_inds2, full_matrix
            )

        perm_dist = perm_dist[~np.isnan(perm_dist)]
        p_value = (
            (np.sum(perm_dist >= obs_fst) + 1) / (len(perm_dist) + 1)
            if len(perm_dist) > 0
            else np.nan
        )
        return obs_fst, p_value, perm_dist

    @staticmethod
    def _bootstrap_replicate(
        seed: int,
        pop_indices: dict,
        pop_pairs: list,
        full_matrix: np.ndarray,
        n_loci: int,
    ):
        """Worker function for a single bootstrap replicate."""
        rng = np.random.default_rng(seed)
        resampled_loci = rng.choice(n_loci, size=n_loci, replace=True)
        replicate = {}
        resampled_matrix = full_matrix[:, resampled_loci]

        for p1_key, p2_key in pop_pairs:
            replicate[(p1_key, p2_key)] = FstDistance._compute_multilocus_fst(
                pop_indices[p1_key], pop_indices[p2_key], resampled_matrix
            )
        return replicate

    def weir_cockerham_fst(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
    ):
        """Calculates pairwise Fst with optional statistical tests.

        Args:
            method (Literal["observed", "permutation", "bootstrap"]): The method to use for Fst calculation.
            n_reps (int): The number of replicates to use for permutation/bootstrap methods.
            n_jobs (int): The number of parallel jobs to run.

        Returns:
            pd.DataFrame: A DataFrame containing pairwise Fst values.
        """
        self.logger.info(
            f"Calculating Fst using method='{method}' with {n_reps} replicates."
        )

        rng = np.random.default_rng(self.seed)

        if n_jobs == 0 or n_jobs < -1:
            msg = f"n_jobs must be -1 or a positive integer, but got: {n_jobs}"
            self.logger.error(msg)
            raise ValueError(msg)

        popmap = self.genotype_data.popmap_inverse
        sample_to_idx = {s: i for i, s in enumerate(self.genotype_data.samples)}
        pop_indices = {
            pop: np.array([sample_to_idx[s] for s in samples if s in sample_to_idx])
            for pop, samples in popmap.items()
        }

        pop_keys = sorted(pop_indices.keys())
        full_matrix = self.genotype_data.snp_data
        n_loci = full_matrix.shape[1]
        num_pops = len(pop_keys)

        if method == "observed":
            fst_mat = np.full((num_pops, num_pops), np.nan)
            for ia, ib in itertools.combinations(range(num_pops), 2):
                p1_key, p2_key = pop_keys[ia], pop_keys[ib]
                fst_val = self._compute_multilocus_fst(
                    pop_indices[p1_key], pop_indices[p2_key], full_matrix
                )
                fst_mat[ia, ib] = fst_mat[ib, ia] = fst_val

            np.fill_diagonal(fst_mat, 0.0)
            df = pd.DataFrame(fst_mat, index=pop_keys, columns=pop_keys)
            self.logger.info("Fst calculation complete!")
            return df

        elif method == "permutation":
            result = {}
            pop_pairs = list(itertools.combinations(pop_keys, 2))

            # --- per-pair independent seeds ---
            seed_base = self.seed if self.seed is not None else 0
            ss = np.random.SeedSequence(seed_base)
            children = ss.spawn(len(pop_pairs))
            pair_seeds = {
                pair: int(child.entropy) & 0x7FFFFFFF  # stable 31-bit int
                for pair, child in zip(pop_pairs, children)
            }

            from concurrent.futures import (
                as_completed,
            )  # local import to keep top clean

            max_workers = mp.cpu_count() if n_jobs == -1 else n_jobs
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        FstDistance._permutation_worker,
                        pair,
                        pop_indices,
                        full_matrix,
                        n_reps,
                        pair_seeds[pair],
                    )
                    for pair in pop_pairs
                ]
                for fut in tqdm(
                    as_completed(futures), desc="Fst permutations", total=len(futures)
                ):
                    pop_pair, res_dict = fut.result()
                    result[pop_pair] = res_dict

            for (p1_key, p2_key), res_dict in result.items():
                self.plotter.plot_permutation_dist(
                    res_dict["fst"],
                    res_dict["perm_dist"],
                    p1_key,
                    p2_key,
                    dist_type="Fst",
                )

            self.logger.info("Fst permutation testing complete!")
            return result

        elif method == "bootstrap":
            pop_pairs = list(itertools.combinations(pop_keys, 2))
            result = {pair: np.zeros(n_reps, dtype=float) for pair in pop_pairs}

            worker_func = partial(
                FstDistance._bootstrap_replicate,
                pop_indices=pop_indices,
                pop_pairs=pop_pairs,
                full_matrix=full_matrix,
                n_loci=n_loci,
            )

            seeds = rng.integers(0, 1e9, size=n_reps)

            with ProcessPoolExecutor(
                max_workers=(mp.cpu_count() if n_jobs == -1 else n_jobs)
            ) as pool:
                reps = pool.map(worker_func, seeds)
                for i, rep_data in tqdm(
                    enumerate(reps), desc="Fst bootstraps", total=n_reps
                ):
                    for pair, fst_val in rep_data.items():
                        result[pair][i] = fst_val

            self.logger.info("Fst bootstrapping complete!")
            return result

        else:
            msg = f"Unknown method '{method}'. Choose from 'observed', 'permutation', or 'bootstrap'."
            self.logger.error(msg)
            raise ValueError(msg)

    def parse_wc_fst(self, result_dict, alpha: float = 0.05):
        """Convert the output of `weir_cockerham_fst()` into summary DataFrames.

        Args:
            result_dict (dict or pd.DataFrame): The output from `weir_cockerham_fst()`.
            alpha (float): Significance level for confidence intervals (default: 0.05 for 95% CIs).

        Returns:
            tuple: (df_mean, df_lower, df_upper, df_pval).
        """
        if isinstance(result_dict, pd.DataFrame):
            self.snpio_mqc.queue_heatmap(
                df=result_dict,
                panel_id="wc_fst_observed",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst (Observed)",
                description=(
                    "Observed pairwise Weir & Cockerham (1984) Fst for all population pairs. No resampling (bootstrap/permutation) was performed."
                ),
                index_label="Population",
                pconfig={
                    "title": "SNPio: Pairwise Weir & Cockerham Fst (Observed)",
                    "id": "wc_fst_observed",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Observed Fst",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                },
            )
            return result_dict, None, None, None

        first_val = next(iter(result_dict.values()))

        def all_populations_from_keys(dict_keys):
            pop_set = set()
            for k in dict_keys:
                pop_set.update(k)
            return sorted(pop_set)

        df_mean, df_lower, df_upper, df_pval = None, None, None, None

        if isinstance(first_val, np.ndarray):
            pop_pairs = list(result_dict.keys())
            pops = all_populations_from_keys(pop_pairs)

            df_mean = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_lower = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_upper = pd.DataFrame(np.nan, index=pops, columns=pops)

            for (p1, p2), arr in result_dict.items():
                arr_nonan = arr[~np.isnan(arr)]
                if len(arr_nonan) == 0:
                    mean_val = np.nan
                    lower_val = np.nan
                    upper_val = np.nan
                else:
                    mean_val = np.mean(arr_nonan)
                    lower_val = np.percentile(arr_nonan, 100 * alpha / 2)
                    upper_val = np.percentile(arr_nonan, 100 * (1 - alpha / 2))

                df_mean.loc[p1, p2] = df_mean.loc[p2, p1] = mean_val
                df_lower.loc[p1, p2] = df_lower.loc[p2, p1] = lower_val
                df_upper.loc[p1, p2] = df_upper.loc[p2, p1] = upper_val

            np.fill_diagonal(df_mean.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            df_ul_combined = self._combine_upper_lower_ci(
                df_upper, df_lower, diagonal="zero"
            )

            self.snpio_mqc.queue_heatmap(
                df=df_mean,
                panel_id="wc_fst_bootstrap_mean",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst (Bootstrap Mean)",
                description=(
                    "Mean pairwise Weir & Cockerham (1984) Fst across bootstrap replicates. Bootstrap resampling was performed by resampling loci with replacement."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst (Bootstrap Mean)",
                    "id": "wc_fst_bootstrap_mean",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Mean Fst (Bootstrap)",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                    "reverse_colors": False,
                },
            )

            self.snpio_mqc.queue_heatmap(
                df=df_ul_combined,
                panel_id="wc_fst_bootstrap_ci95",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst - 95% CIs (Bootstrap)",
                description=(
                    "95 percent confidence intervals from bootstrap replicates. Upper triangle shows upper CI; lower triangle shows lower CI."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst 95% CIs (Bootstrap)",
                    "id": "wc_fst_bootstrap_ci95",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Fst with 95% CIs (Bootstrap)",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                    "reverse_colors": False,
                },
            )

            return df_mean, df_lower, df_upper, df_pval

        if isinstance(first_val, dict) and "fst" in first_val and "pvalue" in first_val:
            pop_pairs = list(result_dict.keys())
            pops = all_populations_from_keys(pop_pairs)

            df_obs = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_pval = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_mean = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_lower = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_upper = pd.DataFrame(np.nan, index=pops, columns=pops)

            for (p1, p2), subdict in result_dict.items():
                obs_val = subdict["fst"]
                # --- changed: treat pvalue as float directly ---
                pv_raw = subdict["pvalue"]
                p_value = (
                    float(pv_raw)
                    if pv_raw is not None and not np.isnan(pv_raw)
                    else np.nan
                )

                dist = subdict.get("perm_dist", None)

                df_obs.loc[p1, p2] = df_obs.loc[p2, p1] = obs_val
                df_pval.loc[p1, p2] = df_pval.loc[p2, p1] = p_value

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

                    df_mean.loc[p1, p2] = df_mean.loc[p2, p1] = mean_val
                    df_lower.loc[p1, p2] = df_lower.loc[p2, p1] = lower_val
                    df_upper.loc[p1, p2] = df_upper.loc[p2, p1] = upper_val

            np.fill_diagonal(df_obs.values, 0.0)
            np.fill_diagonal(df_pval.values, 1.0)
            np.fill_diagonal(df_mean.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            self.snpio_mqc.queue_heatmap(
                df=df_obs,
                panel_id="wc_fst_permutation_observed",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst (Observed; Permutation Test)",
                description=(
                    "Observed pairwise Weir & Cockerham (1984) Fst. Significance is assessed via a permutation test: p = Pr(Fst_perm ≥ Fst_obs)."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst (Observed; Permutation Test)",
                    "id": "wc_fst_permutation_observed",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Observed Fst",
                    "reverse_colors": False,
                    "min": 0.0,
                    "max": 1.0,
                    "tt_decimals": 3,
                },
            )

            self.snpio_mqc.queue_heatmap(
                df=df_pval,
                panel_id="wc_fst_permutation_pvalues",
                section="genetic_differentiation",
                title="SNPio: P-values for Pairwise Weir & Cockerham Fst (Permutation Test)",
                description=(
                    "One-tailed permutation p-values: probability that a permuted Fst is greater than or equal to the observed Fst."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst Permutation p-values",
                    "id": "wc_fst_permutation_pvalues",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Permutation p-value",
                    "reverse_colors": True,
                    "min": 0.0,
                    "max": 1.0,
                    "tt_decimals": 3,
                },
            )

            return df_obs, df_lower, df_upper, df_pval

        msg = "Unrecognized structure in result_dict when estimating Fst. Expected either a DataFrame or a dictionary with specific keys and structures."
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

    def fst_variance_components_per_locus(self, pop1_inds, pop2_inds, full_matrix):
        """Compute per-locus numerator and denominator (Weir & Cockerham) for Fst.

        The numerator is the variance among populations, and the denominator is the total variance. This method processes the SNP matrix for two populations and computes the variance components for each locus. This is useful for calculating Fst values on a per-locus basis, which can help identify loci with significant differentiation between populations.

        Args:
            pop1_inds (np.ndarray): Population 1 individual indices.
            pop2_inds (np.ndarray): Population 2 individual indices.
            full_matrix (np.ndarray): SNP matrix (individuals x loci).

        Notes:
            - a_vals is the numerator (variance among populations),
            - d_vals is the denominator (total variance).
            - This method assumes that the input matrix is a 2D numpy array with individuals as rows and loci as columns.
            - a_vals and d_vals should be of shape (n_loci,).

        Returns:
            Tuple: (a_vals, d_vals) where:
                - a_vals (np.ndarray): Array of numerator values for each locus.
                - d_vals (np.ndarray): Array of denominator values for each locus.
        """
        n_loci = full_matrix.shape[1]
        a_vals = np.zeros(n_loci)
        d_vals = np.zeros(n_loci)

        for loc in range(n_loci):
            s1 = [full_matrix[i, loc] for i in pop1_inds]
            s2 = [full_matrix[j, loc] for j in pop2_inds]

            s1 = FstDistance._clean_inds(s1)
            s2 = FstDistance._clean_inds(s2)

            if not s1 or not s2:
                a_vals[loc] = np.nan
                d_vals[loc] = np.nan
                continue

            s1 = [phased_encoding.get(x, x) for x in s1]
            s2 = [phased_encoding.get(x, x) for x in s2]

            try:
                a, d = FstDistance._two_pop_weir_cockerham_fst_locus(s1, s2)
                a_vals[loc] = a
                d_vals[loc] = d
            except ValueError:
                a_vals[loc] = np.nan
                d_vals[loc] = np.nan

        return a_vals, d_vals
