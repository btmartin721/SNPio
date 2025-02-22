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

        Optionally computes bootstrap-based p-values for each population pair if n_bootstraps > 0.

        Nei's genetic distance is defined as D = -ln( Ī ), where Ī is the ratio of the average genetic identity to the geometric mean of the average homozygosities.

        Args:
            n_bootstraps (int): Number of bootstrap replicates to compute p-values. Defaults to 0 (only distances are returned).
            palette (str): Color palette for the distance matrix plot. Can use any matplotlib gradient-based palette. Some frequently used options include: "coolwarm", "viridis", "magma", and "inferno". Defaults to 'coolwarm'.
            supress_plot (bool): If True, suppresses the plotting of the distance matrix. Defaults to False.

        Returns:
            pd.DataFrame: If n_bootstraps == 0, returns a DataFrame of Nei's distances.
            Tuple[pd.DataFrame, pd.DataFrame]: If n_bootstraps > 0, returns a tuple of (distance matrix, p-value matrix).
        """
        self.logger.info("Calculating Nei's genetic distance...")

        # Step 1: Get allele frequencies per population.
        allele_freqs_per_pop = self._calculate_allele_frequencies()
        populations = list(allele_freqs_per_pop.keys())

        # Initialize the distance matrix and, if needed, the p-value matrix.
        dist_matrix = pd.DataFrame(
            np.nan, index=populations, columns=populations, dtype=float
        )

        pval_matrix = (
            pd.DataFrame(np.nan, index=populations, columns=populations, dtype=float)
            if n_bootstraps > 0
            else None
        )

        # Step 2: Loop over all pairs.
        for i, pop1 in enumerate(populations):
            freqs_pop1 = allele_freqs_per_pop[pop1]
            for j, pop2 in enumerate(populations):
                if i == j:
                    dist_matrix.loc[pop1, pop2] = 0.0
                    if pval_matrix is not None:
                        pval_matrix.loc[pop1, pop2] = np.nan
                elif j < i:
                    # Use symmetry: copy from the other half.
                    dist_matrix.loc[pop1, pop2] = dist_matrix.loc[pop2, pop1]
                    if pval_matrix is not None:
                        pval_matrix.loc[pop1, pop2] = pval_matrix.loc[pop2, pop1]
                else:
                    freqs_pop2 = allele_freqs_per_pop[pop2]
                    # Calculate observed Nei's distance.
                    D_obs = self._calculate_neis_distance_between_pops(
                        freqs_pop1, freqs_pop2
                    )
                    dist_matrix.loc[pop1, pop2] = D_obs
                    dist_matrix.loc[pop2, pop1] = D_obs

                    # If bootstrapping is requested, compute a p-value.
                    if n_bootstraps > 0:
                        # Identify loci with valid data.
                        valid = (
                            ~np.isnan(freqs_pop1)
                            & ~np.isnan(freqs_pop2)
                            & ~np.isinf(freqs_pop1)
                            & ~np.isinf(freqs_pop2)
                        )
                        if np.sum(valid) == 0:
                            pval = np.nan
                        else:
                            p1_valid = freqs_pop1[valid]
                            q_valid = freqs_pop2[valid]
                            # Clip allele frequencies to [0, 1] to avoid numerical issues.
                            p1_valid = np.clip(p1_valid, 0, 1)
                            q_valid = np.clip(q_valid, 0, 1)
                            L = len(p1_valid)
                            boot_dists = []
                            for b in range(n_bootstraps):
                                sample_idx = np.random.choice(L, size=L, replace=True)
                                p_boot = p1_valid[sample_idx]
                                q_boot = q_valid[sample_idx]

                                # Compute bootstrap replicate of Nei's distance.
                                I_locus = p_boot * q_boot + (1 - p_boot) * (1 - q_boot)

                                I_bar = np.mean(I_locus)

                                homo1 = p_boot**2 + (1 - p_boot) ** 2
                                homo2 = q_boot**2 + (1 - q_boot) ** 2

                                avg_homo1 = np.mean(homo1)
                                avg_homo2 = np.mean(homo2)

                                denom = np.sqrt(avg_homo1 * avg_homo2)

                                if denom <= 0:
                                    boot_d = np.nan
                                else:
                                    I_ratio = I_bar / denom
                                    if I_ratio <= 0:
                                        boot_d = np.inf
                                    else:
                                        boot_d = -np.log(I_ratio)
                                boot_dists.append(boot_d)

                            boot_dists = np.array(boot_dists)
                            valid_boot = ~np.isnan(boot_dists)

                            if np.sum(valid_boot) == 0:
                                pval = np.nan
                            else:
                                # One-tailed p-value: proportion of bootstrap replicates with distance >= observed.
                                pval = np.mean(boot_dists[valid_boot] >= D_obs)

                        pval_matrix.loc[pop1, pop2] = pval
                        pval_matrix.loc[pop2, pop1] = pval

        # Replace any infinite values with NaN.
        dist_matrix.replace(np.inf, np.nan, inplace=True)
        if pval_matrix is not None:
            pval_matrix.replace(np.inf, np.nan, inplace=True)
            args = (dist_matrix, pval_matrix)
        else:
            args = dist_matrix

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

        The function computes the per-locus genetic identity `(I = p*q + (1-p)*(1-q)) and then averages over loci. Nei's genetic distance is then defined as D = -ln( Ī ), where Ī is the ratio of the average identity to the geometric mean of the average homozygosities.

        Args:
            freqs_pop1 (np.ndarray): Allele frequencies for population 1.
            freqs_pop2 (np.ndarray): Allele frequencies for population 2.

        Returns:
            float: Nei's genetic distance. Returns np.nan if no valid loci exist, or np.inf if the
                identity ratio is non-positive.
        """
        # Identify loci with valid data.
        valid = (
            ~np.isnan(freqs_pop1)
            & ~np.isnan(freqs_pop2)
            & ~np.isinf(freqs_pop1)
            & ~np.isinf(freqs_pop2)
        )
        if np.sum(valid) == 0:
            return np.nan

        p = freqs_pop1[valid]
        q = freqs_pop2[valid]
        # Clip values to ensure they lie within [0, 1].
        p = np.clip(p, 0, 1)
        q = np.clip(q, 0, 1)

        # Compute per-locus genetic identity.
        I_locus = p * q + (1 - p) * (1 - q)
        I_bar = np.mean(I_locus)

        # Compute homozygosities.
        homo1 = p**2 + (1 - p) ** 2
        homo2 = q**2 + (1 - q) ** 2
        avg_homo1 = np.mean(homo1)
        avg_homo2 = np.mean(homo2)

        # Compute the geometric mean of average homozygosities.
        denom = np.sqrt(avg_homo1 * avg_homo2)
        if denom <= 0:
            return np.nan

        I_ratio = I_bar / denom
        if I_ratio <= 0:
            return np.inf

        nei_distance = -np.log(I_ratio)
        return nei_distance

    def _calculate_allele_frequencies(self) -> dict:
        """
        Helper method to calculate allele frequencies for each population at each locus.

        Returns:
            dict: A dictionary where keys are population IDs and values are arrays of allele
            frequencies per locus.
        """
        pop_indices = self.genotype_data.get_population_indices()
        allele_freqs_per_pop = {}

        for pop_id, indices in pop_indices.items():
            # Subset alignment for population
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan  # Replace missing data

            # Calculate allele frequencies
            # Since diploid
            total_alleles = 2 * np.sum(~np.isnan(pop_alignment), axis=0)
            alt_allele_counts = np.nansum(pop_alignment, axis=0)

            # Avoid division by zero
            freqs = np.divide(alt_allele_counts, total_alleles, where=total_alleles > 0)

            allele_freqs_per_pop[pop_id] = freqs

        return allele_freqs_per_pop

    def _calculate_allele_frequencies(self):
        """Calculate allele frequencies for each population.

        This method calculates allele frequencies for each population in the genotype data. The allele frequencies are calculated as the proportion of alternate alleles at each locus.

        Returns:
            dict: Dictionary of allele frequencies for each population.
        """
        pop_indices = self.genotype_data.get_population_indices()
        allele_freqs_per_pop = {}

        for pop_id, indices in pop_indices.items():
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan
            total_alleles = 2 * np.sum(~np.isnan(pop_alignment), axis=0)
            alt_allele_counts = np.nansum(pop_alignment, axis=0)
            freqs = np.divide(alt_allele_counts, total_alleles, where=total_alleles > 0)
            allele_freqs_per_pop[pop_id] = freqs

        return allele_freqs_per_pop
