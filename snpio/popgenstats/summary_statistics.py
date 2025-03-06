import itertools
import multiprocessing as mp
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


class SummaryStatistics:
    """Class for calculating summary statistics from genotype data."""

    def __init__(self, genotype_data, alignment_012, logger, plotter):
        """Initialize the SummaryStatistics object.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing genotype data.
            alignment_012 (np.ndarray): Genotype data in 012-encoded format.
            logger (logging.Logger): Logger object.
            plotter (snpio.plotting.Plotting): Plotting object.
        """
        self.genotype_data = genotype_data
        self.alignment_012 = alignment_012
        self.logger = logger
        self.plotter = plotter

    def observed_heterozygosity(self) -> np.ndarray:
        """Calculate observed heterozygosity (Ho) for each locus.

        Observed heterozygosity (Ho) is defined as the proportion of heterozygous individuals at a given locus.

        Returns:
            np.ndarray: An array containing observed heterozygosity values for each locus.
        """
        alignment, n_individuals = self._prepare_alignment_and_individuals()
        ho = self._calculate_heterozygosity(alignment, n_individuals, observed=True)
        return ho

    def expected_heterozygosity(self) -> np.ndarray:
        """Calculate expected heterozygosity (He) for each locus.

        Expected heterozygosity (He) is the expected proportion of heterozygous individuals under Hardy-Weinberg equilibrium.

        Returns:
            np.ndarray: An array containing expected heterozygosity values for each locus.
        """
        alignment, n_individuals = self._prepare_alignment_and_individuals()
        he = self._calculate_heterozygosity(alignment, n_individuals, observed=False)
        return he

    def _calculate_heterozygosity(
        self, alignment: np.ndarray, n_individuals: np.ndarray, observed: bool
    ) -> np.ndarray:
        """Calculate heterozygosity (Ho or He) for each locus.

        Args:
            alignment (np.ndarray): The alignment array.
            n_individuals (np.ndarray): Number of non-missing individuals per locus.
            observed (bool): If True, calculate observed heterozygosity (Ho); otherwise, expected heterozygosity (He).

        Returns:
            np.ndarray: Heterozygosity values for each locus.
        """
        if observed:
            # Calculate observed heterozygosity
            heterozygous_counts = np.sum(alignment == 1, axis=0)
            ho = np.divide(heterozygous_counts, n_individuals, where=n_individuals > 0)
            ho[n_individuals == 0] = np.nan  # Handle loci with no valid data
            return ho
        else:
            # Calculate expected heterozygosity
            alt_allele_counts = np.nansum(alignment, axis=0, dtype=np.float64)
            total_alleles = 2 * n_individuals  # Assuming diploid organisms
            with np.errstate(divide="ignore", invalid="ignore"):
                # Frequency of alternate alleles
                p = np.divide(alt_allele_counts, total_alleles, where=total_alleles > 0)
                q = 1 - p
                he = 2 * p * q  # He = 2pq

                # Handle loci with no valid data
                he[total_alleles == 0] = np.nan
            return he

    def nucleotide_diversity(self) -> np.ndarray:
        """Calculate nucleotide diversity (Pi) for each locus.

        Nucleotide diversity (Pi) is the average number of nucleotide differences per site between two sequences.

        Notes:
            A bias correction is applied in the calculation.

        Returns:
            np.ndarray: An array containing nucleotide diversity values for each locus.
        """
        _, n_individuals = self._prepare_alignment_and_individuals()
        he = self.expected_heterozygosity()

        # Calculate nucleotide diversity
        pi = np.full_like(he, np.nan, dtype=float)
        valid = n_individuals > 1  # Need at least 2 individuals for diversity
        pi[valid] = he[valid] * n_individuals[valid] / (n_individuals[valid] - 1)
        return pi

    def calculate_summary_statistics(
        self,
        n_bootstraps=0,
        n_jobs=1,
        save_plots: bool = True,
        use_pvalues: bool = False,
    ):
        """Calculate a suite of summary statistics for SNP data.

        Computes overall and per-population observed heterozygosity (Ho), expected heterozygosity (He), nucleotide diversity (Pi),
        and pairwise Fst between populations. When bootstrapping is used for Fst, the bootstrap replicates (and optionally p-values)
        are included in the summary.

        Args:
            n_bootstraps (int): Number of bootstrap replicates for estimating variance of Fst per SNP. Defaults to 0.
            n_jobs (int): Number of parallel jobs. If -1, all available cores are used. Defaults to 1.
            save_plots (bool): Whether to save plots of the summary statistics. Defaults to True.
            use_pvalues (bool): If True, compute p-values for pairwise Fst comparisons. Defaults to False.

        Returns:
            dict: A dictionary containing summary statistics per population and overall.
        """
        self.logger.info("Calculating summary statistics...")

        # Overall statistics.
        ho_overall = pd.Series(self.observed_heterozygosity())
        he_overall = pd.Series(self.expected_heterozygosity())
        pi_overall = pd.Series(self.nucleotide_diversity())

        # Per-population statistics.
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

        # Fst between populations.
        fst_between_pops = self.weir_cockerham_fst(
            n_bootstraps=n_bootstraps, n_jobs=n_jobs, return_pvalues=use_pvalues
        )
        summary_stats["Fst_between_populations"] = fst_between_pops

        if save_plots:
            self.plotter.plot_summary_statistics(summary_stats, use_pvalues=use_pvalues)

        self.logger.info("Summary statistics calculation complete!")
        return summary_stats

    def observed_heterozygosity_per_population(self):
        """Calculate observed heterozygosity (Ho) for each locus per population.

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
            num_heterozygotes = np.nansum(pop_alignment == 1, axis=0)

            # Calculate Ho
            ho = np.full(pop_alignment.shape[1], np.nan, dtype=np.float64)
            valid = n_individuals > 0
            ho[valid] = num_heterozygotes[valid] / n_individuals[valid]

            # Store results as a pandas Series with locus indices
            ho_per_population[pop_id] = pd.Series(
                ho, index=np.arange(pop_alignment.shape[1]), name="Ho"
            )

        return ho_per_population

    def expected_heterozygosity_per_population(self, return_n: bool = False):
        """Calculate expected heterozygosity (He) for each locus per population.

        Args:
            return_n (bool): If True, also return the number of non-missing individuals per locus.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the expected heterozygosity values per locus for that population. If return_n is True, returns a tuple (he, n_individuals) per population.
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
            total_alleles = 2 * n_individuals

            # Frequency of alternate allele (p)
            alt_allele_counts = np.nansum(pop_alignment, axis=0, dtype=np.float64)
            p = np.zeros_like(alt_allele_counts, dtype=float)
            valid = total_alleles > 0
            p[valid] = alt_allele_counts[valid] / total_alleles[valid]
            q = 1 - p

            # Expected heterozygosity
            he = np.zeros_like(p, dtype=float)
            he[valid] = 2 * p[valid] * q[valid]

            if return_n:
                he_per_population[pop_id] = (
                    pd.Series(he, index=np.arange(pop_alignment.shape[1]), name="He"),
                    n_individuals,
                )
            else:
                he_per_population[pop_id] = pd.Series(
                    he, index=np.arange(pop_alignment.shape[1]), name="He"
                )

        return he_per_population

    def nucleotide_diversity_per_population(self):
        """Calculate nucleotide diversity (Pi) for each locus per population.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the nucleotide diversity values per locus for that population.
        """
        he_and_n_per_population = self.expected_heterozygosity_per_population(
            return_n=True
        )
        pi_per_population = {}

        for pop_id, (he, n_individuals) in he_and_n_per_population.items():
            n = n_individuals.astype(float)

            # Calculate Pi with bias correction
            pi = np.zeros_like(he, dtype=float)
            valid = n > 1
            pi[valid] = (n[valid] / (n[valid] - 1)) * he[valid]

            # Store results as a pandas Series
            pi_per_population[pop_id] = pd.Series(
                pi, index=np.arange(len(pi)), name="Pi"
            )

        return pi_per_population

    def weir_cockerham_fst(
        self, n_bootstraps: int = 0, n_jobs: int = 1, return_pvalues: bool = False
    ):
        """Calculate Weir and Cockerham's Fst between populations.

        When bootstrapping (n_bootstraps > 0) and return_pvalues is False, the function computes an overall Fst
        (by averaging per-locus Fst values) for each bootstrap replicate using resampling of loci.

        When return_pvalues is True, a permutation approach is used instead. Individuals are randomly reassigned
        to populations (preserving sample sizes) to generate a null distribution of overall Fst values. The observed
        overall Fst (computed from the original assignments) is then compared to this null distribution to obtain
        a one-tailed p-value for each population pair.

        Args:
            n_bootstraps (int): Number of replicates. If 0, no resampling is done and per-locus Fst is returned.
            n_jobs (int): Number of parallel jobs. Use -1 to use all available cores.
            return_pvalues (bool): If True, use the permutation approach to compute p-values.
                If False, return the raw bootstrap replicate overall Fst values (using loci resampling).

        Returns:
            dict:
            - If n_bootstraps == 0, returns a dictionary where keys are population pairs and values are pandas Series
                of per-locus Fst values.
            - If n_bootstraps > 0 and return_pvalues is False, returns a dictionary where keys are population pairs and
                values are numpy arrays of overall Fst values (length n_bootstraps) obtained via bootstrap.
            - If n_bootstraps > 0 and return_pvalues is True, returns a dictionary where keys are population pairs and
                values are dicts with keys "fst" (the observed overall Fst) and "pvalue" (a pandas Series of the computed
                p-value repeated for each replicate).
        """
        # Get population indices and number of loci.
        pop_indices = self.genotype_data.get_population_indices()
        populations = list(pop_indices.keys())
        n_loci = self.alignment_012.shape[1]

        # Precompute alignments for each population; convert missing data (< 0) to np.nan.
        pop_alignments = {
            pop: self.alignment_012[indices, :].astype(float)
            for pop, indices in pop_indices.items()
        }
        for alignment in pop_alignments.values():
            alignment[alignment < 0] = np.nan

        def compute_fst_pair(alignment1, alignment2):
            """Compute Fst per SNP between two populations.

            Args:
                alignment1 (np.ndarray): Genotype matrix for population 1 (individuals x loci).
                alignment2 (np.ndarray): Genotype matrix for population 2 (individuals x loci).

            Returns:
                np.ndarray: Fst values for each locus.
            """
            # Count non-missing calls per locus.
            n1 = np.count_nonzero(~np.isnan(alignment1), axis=0)
            n2 = np.count_nonzero(~np.isnan(alignment2), axis=0)
            n_total = n1 + n2

            # Count alternate alleles per locus.
            alt1 = np.nansum(alignment1, axis=0, dtype=np.float64)
            alt2 = np.nansum(alignment2, axis=0, dtype=np.float64)
            total_alt = alt1 + alt2

            # Compute allele frequencies.
            p1 = np.divide(alt1, 2 * n1, where=(n1 > 0))
            p2 = np.divide(alt2, 2 * n2, where=(n2 > 0))
            p_total = np.divide(total_alt, 2 * n_total, where=(n_total > 0))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Calculate variance in allele frequencies (unbiased estimator).
                s2 = (n1 * (p1 - p_total) ** 2 + n2 * (p2 - p_total) ** 2) / np.where(
                    n_total > 1, (n_total - 1), np.nan
                )

            # Expected heterozygosity.
            he_total = 2 * p_total * (1 - p_total)

            # Compute Fst as ratio of variance to total heterozygosity.
            fst = np.full_like(he_total, np.nan, dtype=np.float64)
            valid = (he_total > 0) & (n_total > 1)
            fst[valid] = s2[valid] / he_total[valid]
            return fst

        # Case 1: No resampling requested.
        if n_bootstraps == 0:
            fst_per_population_pair = {}
            for pop1, pop2 in itertools.combinations(populations, 2):
                alignment1 = pop_alignments[pop1]
                alignment2 = pop_alignments[pop2]
                fst_values = compute_fst_pair(alignment1, alignment2)
                fst_per_population_pair[(pop1, pop2)] = pd.Series(
                    fst_values, index=np.arange(n_loci), name=f"Fst {pop1}-{pop2}"
                )
            return fst_per_population_pair

        # Case 2: Resampling is requested.
        if return_pvalues:
            # Use permutation approach.
            # First, compute the observed overall Fst for each population pair.
            observed_overall = {}
            for pop1, pop2 in itertools.combinations(populations, 2):
                alignment1 = pop_alignments[pop1]
                alignment2 = pop_alignments[pop2]
                observed_overall[(pop1, pop2)] = np.nanmean(
                    compute_fst_pair(alignment1, alignment2)
                )

            # Prepare dictionary to store permutation replicates overall Fst.
            fst_permutation_overall = {
                (pop1, pop2): np.zeros(n_bootstraps, dtype=np.float64)
                for pop1, pop2 in itertools.combinations(populations, 2)
            }

            def permutation_replicate(seed):
                """Perform one permutation replicate by randomly reassigning individuals to populations.

                Args:
                    seed (int): Random seed for reproducibility.

                Returns:
                    dict: Keys are population pairs and values are overall Fst values (averaged over loci)
                        computed from the permuted assignments.
                """
                rng = np.random.default_rng(seed)
                # Combine all individual indices.
                all_inds = np.concatenate([pop_indices[pop] for pop in populations])
                # Permute all indices.
                permuted = rng.permutation(all_inds)
                new_assignments = {}
                start = 0
                # Assign individuals to populations preserving original sample sizes.
                for pop in populations:
                    n_pop = len(pop_indices[pop])
                    new_assignments[pop] = permuted[start : start + n_pop]
                    start += n_pop

                replicate_results = {}
                for pop1, pop2 in itertools.combinations(populations, 2):
                    # Extract new alignments from the overall genotype matrix.
                    alignment1_perm = self.alignment_012[
                        new_assignments[pop1], :
                    ].astype(float)
                    alignment2_perm = self.alignment_012[
                        new_assignments[pop2], :
                    ].astype(float)
                    alignment1_perm[alignment1_perm < 0] = np.nan
                    alignment2_perm[alignment2_perm < 0] = np.nan
                    fst_values_perm = compute_fst_pair(alignment1_perm, alignment2_perm)
                    overall_fst_perm = np.nanmean(fst_values_perm)
                    replicate_results[(pop1, pop2)] = overall_fst_perm
                return replicate_results

            seeds = np.random.default_rng().integers(0, int(1e9), size=n_bootstraps)
            n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                permutation_results = list(executor.map(permutation_replicate, seeds))

            # Collate permutation replicates.
            for b, replicate in enumerate(permutation_results):
                for pop_pair, overall_fst in replicate.items():
                    fst_permutation_overall[pop_pair][b] = overall_fst

            # Compute one-tailed p-values for each population pair.
            results = {}
            for pop_pair, perm_array in fst_permutation_overall.items():
                obs = observed_overall[pop_pair]
                # p-value: proportion of permutation replicates with overall Fst >= observed.
                p_val = (np.sum(perm_array >= obs) + 1) / (n_bootstraps + 1)
                # Return a dict containing the observed Fst and a p-value repeated (for compatibility with plotting).
                results[pop_pair] = {
                    "fst": np.array([obs]),  # Observed overall Fst.
                    "pvalue": pd.Series(
                        [p_val] * n_bootstraps,
                        name=f"P-value {pop_pair[0]}-{pop_pair[1]}",
                    ),
                }
            return results

        else:
            # Use bootstrap (loci resampling) approach.
            fst_bootstrap_overall = {
                (pop1, pop2): np.zeros(n_bootstraps, dtype=np.float64)
                for pop1, pop2 in itertools.combinations(populations, 2)
            }

            def bootstrap_replicate(seed):
                """Compute overall Fst between two populations for one bootstrap replicate by resampling loci.

                Args:
                    seed (int): Random seed for reproducibility.

                Returns:
                    dict: Keys are population pairs and values are overall Fst values.
                """
                rng = np.random.default_rng(seed)
                resample_indices = rng.choice(n_loci, size=n_loci, replace=True)
                replicate_overall = {}
                for pop1, pop2 in itertools.combinations(populations, 2):
                    alignment1 = pop_alignments[pop1][:, resample_indices]
                    alignment2 = pop_alignments[pop2][:, resample_indices]
                    fst_values = compute_fst_pair(alignment1, alignment2)
                    overall_fst = np.nanmean(fst_values)
                    replicate_overall[(pop1, pop2)] = overall_fst
                return replicate_overall

            seeds = np.random.default_rng().integers(0, int(1e9), size=n_bootstraps)
            n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                bootstrap_results = list(executor.map(bootstrap_replicate, seeds))

            for b, replicate in enumerate(bootstrap_results):
                for pop_pair, overall_fst in replicate.items():
                    fst_bootstrap_overall[pop_pair][b] = overall_fst

            return fst_bootstrap_overall

    def _prepare_alignment_and_individuals(self):
        """Prepare alignment and count non-missing individuals per locus.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Alignment array and counts of non-missing individuals per locus.
        """
        alignment = self.alignment_012.astype(float).copy()

        # Replace missing data (-9) with NaN
        alignment[alignment == -9] = np.nan

        # Count valid individuals per locus
        n_individuals = np.sum(~np.isnan(alignment), axis=0)
        return alignment, n_individuals
