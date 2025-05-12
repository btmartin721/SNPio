import numpy as np
import pandas as pd

from snpio.popgenstats.fst_distance import FstDistance
from snpio.utils.logging import LoggerManager


class SummaryStatistics:
    """Class for calculating summary statistics from genotype data."""

    def __init__(
        self, genotype_data, alignment_012, plotter, verbose=False, debug=False
    ):
        """Initialize the SummaryStatistics object.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing genotype data.
            alignment_012 (np.ndarray): Genotype data in 012-encoded format.
            plotter (snpio.plotting.Plotting): Plotting object.
            verbose (bool): If True, enable verbose logging.
            debug (bool): If True, enable debug logging.
        """
        self.genotype_data = genotype_data
        self.alignment_012 = alignment_012
        self.verbose = verbose
        self.debug = debug

        logman = LoggerManager(
            __name__, prefix=self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()
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
        n_permutations=0,
        n_jobs=1,
        save_plots: bool = True,
        use_pvalues: bool = False,
    ):
        """Calculate a suite of summary statistics for SNP data.

        Computes overall and per-population observed heterozygosity (Ho), expected heterozygosity (He), nucleotide diversity (Pi),
        and pairwise Fst between populations. When bootstrapping is used for Fst, the bootstrap replicates (and optionally p-values)
        are included in the summary.

        Args:
            n_permutations (int): Number of permutation replicates for estimating variance of Fst per SNP. Defaults to 0.
            n_jobs (int): Number of parallel jobs. If -1, all available cores are used. Defaults to 1.
            save_plots (bool): Whether to save plots of the summary statistics. Defaults to True.
            use_pvalues (bool): If True, compute p-values for pairwise Fst comparisons. Defaults to False.

        Returns:
            dict: A dictionary containing summary statistics per population and overall.
        """
        self.logger.info("Calculating summary statistics...")

        self.logger.info("Calculating heterozygosity and nucleotide diversity...")

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

        self.logger.info("Calculating pairwise Weir & Cockerham Fst...")
        self.logger.info(f"Number of bootstraps: {n_permutations}")
        self.logger.info(f"Number of CPU threads: {n_jobs}")
        self.logger.info(f"Use p-values: {use_pvalues}")
        self.logger.info(f"Save plots: {save_plots}")

        # Pairwise Fst between populations.
        fst = FstDistance(
            self.genotype_data, self.plotter, verbose=self.verbose, debug=self.debug
        )

        fst_pw = fst.weir_cockerham_fst(
            n_bootstraps=n_permutations, n_jobs=n_jobs, return_pvalues=use_pvalues
        )

        df_observed, df_lower, df_upper, df_pvals = fst._parse_wc_fst(
            fst_pw, alpha=0.05
        )

        summary_stats["Fst_between_populations_obs"] = df_observed
        summary_stats["Fst_between_populations_lower"] = df_lower
        summary_stats["Fst_between_populations_upper"] = df_upper
        summary_stats["Fst_between_populations_pvalues"] = df_pvals

        if save_plots:
            self.plotter.plot_summary_statistics(summary_stats, use_pvalues=use_pvalues)

        self.logger.info("Fst calculation complete!")
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

    def ind_count(self, geno):
        """Count the number of non-missing individuals in a genotype array.

        Args:
            geno (np.ndarray): 1D array of genotype values (with np.nan for missing).

        Returns:
            int: Count of non-missing individuals.
        """
        return int(np.sum(~np.isnan(geno)))

    def pop_het(self, geno):
        """Compute the observed heterozygosity (proportion of heterozygotes) from a 1D genotype array (assumes heterozygote is coded as 1).

        Args:
            geno (np.ndarray): 1D array of genotype values.

        Returns:
            float: Proportion of heterozygotes, or np.nan if no individuals are typed.
        """
        n = self.ind_count(geno)
        if n == 0:
            return np.nan
        return np.sum(geno[~np.isnan(geno)] == 1) / n

    def pop_freq(self, geno):
        """Compute the allele frequency from a 1D genotype array.

        Assumes genotypes are coded as 0, 1, or 2 (number of copies of the alternate allele).
        Only non-missing individuals are used.

        Args:
            geno (np.ndarray): 1D array of genotype values.

        Returns:
            float: Allele frequency, or np.nan if no individuals are typed.
        """
        n = self.ind_count(geno)
        if n == 0:
            return np.nan
        # Total alternate allele count divided by total allele copies.
        return np.nansum(geno) / (2 * n)

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
