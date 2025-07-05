from itertools import combinations
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from snpio.plotting.plotting import Plotting
from snpio.popgenstats.fst_distance import FstDistance
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


class PermutationOutlierDetector:
    """Detects Fst outliers using a univariate, per-locus permutation test.

    This class performs a per-locus permutation test for Fst outliers between all pairs of populations. It computes the Weir & Cockerham Fst for each locus, permutes the sample labels, and calculates p-values based on the distribution of permuted Fst values. The results are adjusted for multiple testing using methods like FDR or Bonferroni correction.
    """

    def __init__(self, genotype_data, verbose: bool = False, debug: bool = False):
        """Initialize the permutation outlier detector.

        Initializes the permutation outlier detector with genotype data and sets up necessary components for Fst distance calculation and plotting.

        Args:
            genotype_data (GenotypeData): Genotype data object containing the genotype matrix.
            verbose (bool): Verbose logging.
            debug (bool): Debug logging.
        """
        self.genotype_data = genotype_data
        self.prefix = genotype_data.prefix
        self.full_matrix = genotype_data.snp_data
        self.popmap_inverse = genotype_data.popmap_inverse
        self.pop_names = list(self.popmap_inverse.keys())

        self.plotter = Plotting(self.genotype_data, genotype_data.plot_kwargs)

        self.fst_dist = FstDistance(
            self.genotype_data, self.plotter, verbose=verbose, debug=debug
        )

        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=verbose, debug=debug
        )
        self.logger = logman.get_logger()

        self.iupac = IUPAC(logger=self.logger)
        self.phased_encoding = self.iupac.get_phased_encoding()

    def run(
        self,
        n_perm: int = 1000,
        correction_method: (
            Literal["fdr_bh", "bonferroni", "holm", "hommel", "holm-sidak"] | None
        ) = "fdr_bh",
        alpha: float = 0.05,
        seed: int | None = None,
        alternative: Literal["upper", "lower", "both"] = "upper",
    ) -> pd.DataFrame:
        """Run the per-locus permutation test for all population pairs.

        This method performs a per-locus permutation test for Fst outliers between all pairs of populations in the genotype data. It computes the Weir & Cockerham Fst for each locus, permutes the sample labels, and calculates p-values based on the distribution of permuted Fst values. The results are adjusted for multiple testing using the specified correction method.

        Args:
            n_perm (int): Number of permutations per locus.
            correction_method (Literal["fdr_bh", "bonferroni", "holm", "hommel", "holm-sidak"] | None): 'fdr_bh', 'bonferroni', or None.
            alpha (float): Significance threshold for identifying outliers.
            seed (int | None): RNG seed for reproducibility.
            alternative (Literal["upper", "lower", "both"]): 'upper', 'lower', or 'both'.

        Returns:
            pd.DataFrame: A long-form DataFrame of significant outlier results.
        """
        self.logger.info(
            f"Running per-locus permutation test for Fst outliers with {n_perm} permutations, alpha={alpha}, seed={seed}, alternative={alternative}."
        )

        all_results = []
        self.logger.info("Starting per-locus permutation tests for all pairs...")

        for pop1, pop2 in combinations(self.pop_names, 2):
            # List of sample IDs for given populations
            pop1_inds: List[str] = self.popmap_inverse[pop1]
            pop2_inds: List[str] = self.popmap_inverse[pop2]

            pop1_indices: List[int] = [
                self.genotype_data.samples.index(i) for i in pop1_inds
            ]
            pop2_indices: List[int] = [
                self.genotype_data.samples.index(i) for i in pop2_inds
            ]

            observed_fst, raw_pvals = self._per_locus_permutation_test(
                pop1_indices, pop2_indices, n_perm, seed, alternative
            )

            if correction_method:
                adj_pvals = self._correct_pvals(raw_pvals, method=correction_method)
            else:
                adj_pvals = raw_pvals

            # Store results for this pair
            for loc_idx, (fst, q_val) in enumerate(zip(observed_fst, adj_pvals)):
                if not np.isnan(q_val) and q_val <= alpha:
                    all_results.append(
                        {
                            "Locus": self.genotype_data.marker_names[loc_idx]
                            or f"locus_{loc_idx}",
                            "Population_Pair": f"{pop1}_{pop2}",
                            "Fst": fst,
                            "q_value": q_val,
                        }
                    )

        self.logger.info(
            f"Permutation method found {len(all_results)} significant outlier entries. Analysis complete."
        )
        return pd.DataFrame(all_results)

    def _compute_per_locus_fst(
        self, pop1_inds: np.ndarray, pop2_inds: np.ndarray
    ) -> np.ndarray:
        """Compute Weir & Cockerham Fst for each locus between two populations.

        This method computes the Weir & Cockerham (1984) Fst for each locus between two populations. It handles missing data and monomorphic loci, returning NaN for invalid cases.

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
            g1 = [self.phased_encoding.get(x, x) for x in g1]
            g2 = [self.phased_encoding.get(x, x) for x in g2]

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

    def _per_locus_permutation_test(
        self,
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        n_perm: int = 1000,
        seed: int | None = None,
        alternative: Literal["upper", "lower", "both"] = "both",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Permute sample labels for each locus and compute p-values.

        This method performs a per-locus permutation test for Fst outliers between two populations. It computes the Weir & Cockerham Fst for each locus, permutes the sample labels, and calculates p-values based on the distribution of permuted Fst values.

        Args:
            pop1_inds (np.ndarray): Indices of population 1 individuals.
            pop2_inds (np.ndarray): Indices of population 2 individuals.
            n_perm (int): Number of permutation replicates.
            seed (int | None): RNG seed.
            bandwidth (str | float): KDE bandwidth.
            alternative (str): 'upper', 'lower', or 'both'.
            mode (str): 'auto', 'kde', or 'empirical'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Observed Fst and p-values.
        """
        if seed is None:
            rng = np.random.default_rng()
        else:
            if not isinstance(seed, int):
                msg = f"Seed must be an integer, but got: {type(seed)}."
                self.logger.error(msg)
                raise ValueError(msg)
            rng = np.random.default_rng(seed)

        full_mat = self.full_matrix.copy()
        n_loci = full_mat.shape[1]
        observed_fst = self._compute_per_locus_fst(pop1_inds, pop2_inds)

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

                g1 = [self.phased_encoding.get(x, x) for x in g1]
                g2 = [self.phased_encoding.get(x, x) for x in g2]

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
                f"Skipped {len(skipped_loci)} loci due to invalid values. Setting P-values at the corresponding loci to 1.0."
            )

        if flat_loci:
            for loc in flat_loci:
                p_values[loc] = 1.0
            self.logger.warning(
                f"{len(flat_loci)} loci have flat distributions: {list(set(flat_loci))}. Setting P-values at the corresponding loci to 1.0."
            )

        if few_valids:
            for loc in few_valids:
                p_values[loc] = 1.0
            self.logger.warning(
                f"{len(few_valids)} loci have too few valid permutations: {list(set(few_valids))}. Setting P-values at the corresponding loci to 1.0."
            )

        return observed_fst, p_values

    @staticmethod
    def _correct_pvals(pvals: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
        """Apply multiple-testing correction to an array of p-values.

        This method uses the statsmodels library to apply the specified correction method.

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
