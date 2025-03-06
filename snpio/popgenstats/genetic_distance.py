import itertools
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


class GeneticDistance:
    """Class for calculating Nei's genetic distance between populations.

    This class calculates genetic distance between populations using allele frequencies. The genetic distance is calculated using Nei's method, which is based on the proportion of shared alleles between populations. The genetic distance is calculated as ``-log(I)``, where I is the proportion of shared alleles between populations.
    """

    def __init__(self, genotype_data, alignment_012, logger, plotter):
        """Initialize the GeneticDistance object.

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

    def calculate_neis_distance(
        self,
        n_bootstraps: int = 0,
        palette: str = "coolwarm",
        supress_plot: bool = False,
    ):
        """Calculate Nei's genetic distance between all pairs of populations.

        When n_bootstraps is greater than 0, a permutation approach is used to compute p-values for each population pair. Individuals are randomly reassigned to populations (preserving original sample sizes) to generate a null distribution of Nei's distances. The p-value for a given pair is defined as the proportion of permutation replicates with a distance greater than or equal to the observed distance.

        Args:
            n_bootstraps (int): Number of permutation replicates to compute p-values. Defaults to 0 (only distances are returned).
            palette (str): Color palette for the distance matrix plot. Defaults to 'coolwarm'.
            supress_plot (bool): If True, suppresses plotting of the distance matrix. Defaults to False.

        Returns:
            pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
                If n_bootstraps == 0, returns a DataFrame of Nei's distances.
                If n_bootstraps > 0, returns a tuple of (distance matrix, p-value matrix).
        """
        self.logger.info("Calculating Nei's genetic distance...")

        # Step 1: Get allele frequencies per population.
        allele_freqs_per_pop = self._calculate_allele_frequencies()
        populations = list(allele_freqs_per_pop.keys())

        # Initialize the observed distance matrix.
        dist_matrix = pd.DataFrame(
            np.nan, index=populations, columns=populations, dtype=float
        )
        # Also initialize the p-value matrix if replicates are requested.
        pval_matrix = (
            pd.DataFrame(np.nan, index=populations, columns=populations, dtype=float)
            if n_bootstraps > 0
            else None
        )

        # Step 2: Loop over all population pairs to compute observed Nei's distances.
        for i, pop1 in enumerate(populations):
            freqs_pop1 = allele_freqs_per_pop[pop1]
            for j, pop2 in enumerate(populations):
                if i == j:
                    dist_matrix.loc[pop1, pop2] = 0.0
                    if pval_matrix is not None:
                        pval_matrix.loc[pop1, pop2] = np.nan
                elif j < i:
                    # Use symmetry: copy the value from the other half.
                    dist_matrix.loc[pop1, pop2] = dist_matrix.loc[pop2, pop1]
                    if pval_matrix is not None:
                        pval_matrix.loc[pop1, pop2] = pval_matrix.loc[pop2, pop1]
                else:
                    freqs_pop2 = allele_freqs_per_pop[pop2]
                    D_obs = self._calculate_neis_distance_between_pops(
                        freqs_pop1, freqs_pop2
                    )
                    dist_matrix.loc[pop1, pop2] = D_obs
                    dist_matrix.loc[pop2, pop1] = D_obs

        # Step 3: If permutation replicates are requested, compute p-values.
        if n_bootstraps > 0:
            # Dictionary to store permutation replicate distances.
            permuted_distances = {
                (pop1, pop2): np.zeros(n_bootstraps, dtype=np.float64)
                for pop1, pop2 in itertools.combinations(populations, 2)
            }
            # Get original population indices.
            pop_indices = self.genotype_data.get_population_indices()
            all_inds = np.concatenate([pop_indices[pop] for pop in populations])
            seeds = np.random.default_rng().integers(0, int(1e9), size=n_bootstraps)
            n_jobs = mp.cpu_count()

            def permutation_replicate(seed):
                """Perform one permutation replicate by reassigning individuals to populations.

                Args:
                    seed (int): Random seed.
                Returns:
                    dict: Keys are population pairs; values are Nei's distance computed from the permuted assignments.
                """
                rng = np.random.default_rng(seed)
                permuted = rng.permutation(all_inds)
                new_assignments = {}
                start = 0
                for pop in populations:
                    n_pop = len(pop_indices[pop])
                    new_assignments[pop] = permuted[start : start + n_pop]
                    start += n_pop
                # Compute allele frequencies for each permuted population.
                perm_allele_freqs = {}
                for pop in populations:
                    pop_alignment = self.alignment_012[new_assignments[pop], :].astype(
                        float
                    )
                    # Replace missing data (assuming missing is indicated by -9).
                    pop_alignment[pop_alignment == -9] = np.nan
                    total_alleles = 2 * np.sum(~np.isnan(pop_alignment), axis=0)
                    alt_allele_counts = np.nansum(pop_alignment, axis=0)
                    freqs = np.divide(
                        alt_allele_counts, total_alleles, where=total_alleles > 0
                    )
                    perm_allele_freqs[pop] = freqs
                replicate_results = {}
                for pop1, pop2 in itertools.combinations(populations, 2):
                    D_perm = self._calculate_neis_distance_between_pops(
                        perm_allele_freqs[pop1], perm_allele_freqs[pop2]
                    )
                    replicate_results[(pop1, pop2)] = D_perm
                return replicate_results

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                permutation_results = list(executor.map(permutation_replicate, seeds))
            for b, replicate in enumerate(permutation_results):
                for pop_pair, D_perm in replicate.items():
                    permuted_distances[pop_pair][b] = D_perm

            # Compute one-tailed p-values: p-value is the proportion of permutation replicates with
            # Nei's distance greater than or equal to the observed value.
            for pop_pair, perm_array in permuted_distances.items():
                D_obs = dist_matrix.loc[pop_pair[0], pop_pair[1]]
                p_val = np.mean(perm_array >= D_obs)
                pval_matrix.loc[pop_pair[0], pop_pair[1]] = p_val
                pval_matrix.loc[pop_pair[1], pop_pair[0]] = p_val

            args = (dist_matrix, pval_matrix)
        else:
            args = dist_matrix

        # Step 4: Plot the distance matrix (and p-value matrix if available).
        if not supress_plot:
            self.plotter.plot_dist_matrix(
                *args, palette=palette, title="Nei's Genetic Distance"
            )

        self.logger.info("Nei's genetic distance calculation complete!")
        return args

    def _calculate_neis_distance_between_pops(
        self, freqs_pop1: np.ndarray, freqs_pop2: np.ndarray
    ) -> float:
        """Calculate Nei's genetic distance between two populations using averaged per-locus values.

        The function computes the per-locus genetic identity (``I = p*q + (1-p)*(1-q)``) and then averages over loci.

        Nei's genetic distance is defined as ``D = -ln(Ī)``, where Ī is the ratio of the average identity to the geometric  mean of the average homozygosities.

        Args:
            freqs_pop1 (np.ndarray): Allele frequencies for population 1.
            freqs_pop2 (np.ndarray): Allele frequencies for population 2.

        Returns:
            float: Nei's genetic distance. Returns np.nan if no valid loci exist, or np.inf if the identity ratio is non-positive.
        """
        valid = (
            ~np.isnan(freqs_pop1)
            & ~np.isnan(freqs_pop2)
            & ~np.isinf(freqs_pop1)
            & ~np.isinf(freqs_pop2)
        )
        if np.sum(valid) == 0:
            return np.nan

        p = np.clip(freqs_pop1[valid], 0, 1)
        q = np.clip(freqs_pop2[valid], 0, 1)

        I_locus = p * q + (1 - p) * (1 - q)
        I_bar = np.mean(I_locus)

        homo1 = p**2 + (1 - p) ** 2
        homo2 = q**2 + (1 - q) ** 2
        avg_homo1 = np.mean(homo1)
        avg_homo2 = np.mean(homo2)

        denom = np.sqrt(avg_homo1 * avg_homo2)
        if denom <= 0:
            return np.nan

        I_ratio = I_bar / denom
        if I_ratio <= 0:
            return np.inf

        nei_distance = -np.log(I_ratio)
        return nei_distance

    def _calculate_allele_frequencies(self) -> dict:
        """Calculate allele frequencies for each population.

        Returns:
            dict: A dictionary where keys are population IDs and values are arrays of allele frequencies per locus.
        """
        pop_indices = self.genotype_data.get_population_indices()
        allele_freqs_per_pop = {}
        for pop_id, indices in pop_indices.items():
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan  # Replace missing data
            total_alleles = 2 * np.sum(~np.isnan(pop_alignment), axis=0)
            alt_allele_counts = np.nansum(pop_alignment, axis=0)
            freqs = np.divide(alt_allele_counts, total_alleles, where=total_alleles > 0)
            allele_freqs_per_pop[pop_id] = freqs
        return allele_freqs_per_pop
