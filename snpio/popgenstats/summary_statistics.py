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
        self, n_bootstraps=0, n_jobs=1, save_plots: bool = True
    ):
        """Calculate a suite of summary statistics for SNP data.

        This method calculates a suite of summary statistics for SNP data, including observed heterozygosity (Ho), expected heterozygosity (He), nucleotide diversity (Pi), and Fst between populations. Summary statistics are calculated both overall and per population.

        Args:
            n_bootstraps (int): Number of bootstrap replicates to use for estimating variance of Fst per SNP. If 0, then bootstrapping is not used and confidence intervals are estimated from the data. Defaults to 0.
            n_jobs (int): Number of parallel jobs. If set to -1, all available CPU threads are used. Defaults to 1.
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
        fst_between_pops = self.weir_cockerham_fst(
            n_bootstraps=n_bootstraps, n_jobs=n_jobs
        )
        summary_stats["Fst_between_populations"] = fst_between_pops

        if save_plots:
            self.plotter.plot_summary_statistics(summary_stats)

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

    def weir_cockerham_fst(self, n_bootstraps: int = 0, n_jobs: int = -1):
        """Calculate pairwise per-population Weir and Cockerham's Fst.

        This method calculates pairwise Weir and Cockerham's Fst between populations. Fst is a measure of population differentiation due to genetic structure. Fst values range from 0 to 1, where 0 indicates no genetic differentiation and 1 indicates complete differentiation. Bootstrapping can be used to estimate the variance of Fst per SNP.

        Args:
            n_bootstraps (int): Number of bootstrap replicates. Default is 0 (no bootstrapping).
            n_jobs (int): Number of parallel jobs for bootstrapping. Default is -1 (use all available cores).

        Returns:
            dict: If n_bootstraps is 0, returns a dictionary where keys are population pairs and values are pandas Series of Fst values per locus. If n_bootstraps > 0, returns a dictionary where keys are population pairs and values are numpy arrays with shape (n_loci, n_bootstraps).
        """
        # Prepare population indices and get number of loci
        pop_indices = self.genotype_data.get_population_indices()
        populations = list(pop_indices.keys())
        n_loci = self.alignment_012.shape[1]

        # Precompute alignments for each population; convert missing data (< 0) to np.nan
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
            # Count non-missing calls per locus for each population.
            n1 = np.count_nonzero(~np.isnan(alignment1), axis=0)
            n2 = np.count_nonzero(~np.isnan(alignment2), axis=0)
            n_total = n1 + n2

            # Count alternate alleles per locus.
            alt1 = np.nansum(alignment1, axis=0, dtype=np.float64)
            alt2 = np.nansum(alignment2, axis=0, dtype=np.float64)
            total_alt = alt1 + alt2

            # Compute allele frequencies where valid.
            p1 = np.divide(alt1, 2 * n1, where=(n1 > 0))
            p2 = np.divide(alt2, 2 * n2, where=(n2 > 0))
            p_total = np.divide(total_alt, 2 * n_total, where=(n_total > 0))

            with warnings.catch_warnings():
                # Suppress RuntimeWarning for NaN values.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Calculate variance in allele frequencies (unbiased estimator).
                s2 = (n1 * (p1 - p_total) ** 2 + n2 * (p2 - p_total) ** 2) / np.where(
                    n_total > 1, (n_total - 1), np.nan
                )

            # Expected heterozygosity in the total population.
            he_total = 2 * p_total * (1 - p_total)

            # Compute Fst as the ratio of variance to total heterozygosity.
            fst = np.full_like(he_total, np.nan, dtype=np.float64)
            valid = (he_total > 0) & (n_total > 1)
            fst[valid] = s2[valid] / he_total[valid]
            return fst

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
        else:
            # Prepare dictionary for bootstrap results.
            fst_bootstrap_per_population_pair = {
                (pop1, pop2): np.zeros((n_loci, n_bootstraps), dtype=np.float64)
                for pop1, pop2 in itertools.combinations(populations, 2)
            }

            def bootstrap_replicate(seed):
                """Compute Fst per SNP between two populations for one bootstrap replicate.

                Args:
                    seed (int): Random seed for reproducibility.

                Returns:
                    dict: Dictionary with keys as population pairs and values as Fst arrays per locus.
                """
                rng = np.random.default_rng(seed)
                resample_indices = rng.choice(n_loci, size=n_loci, replace=True)
                replicate_results = {}
                for pop1, pop2 in itertools.combinations(populations, 2):
                    alignment1 = pop_alignments[pop1][:, resample_indices]
                    alignment2 = pop_alignments[pop2][:, resample_indices]
                    fst_values = compute_fst_pair(alignment1, alignment2)
                    replicate_results[(pop1, pop2)] = fst_values
                return replicate_results

            seeds = np.random.default_rng().integers(0, 1e9, size=n_bootstraps)
            n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                bootstrap_results = list(executor.map(bootstrap_replicate, seeds))

            # Collate bootstrap replicates.
            for b, replicate in enumerate(bootstrap_results):
                for pop_pair, fst_values in replicate.items():
                    fst_bootstrap_per_population_pair[pop_pair][:, b] = fst_values

            return fst_bootstrap_per_population_pair

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
