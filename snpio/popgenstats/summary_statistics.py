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

    def filter_loci_by_call_rate(self, alignment1, alignment2, min_call_rate=0.8):
        """Filter loci based on a minimum call rate in both populations.
            Args:
            alignment1 (np.ndarray): Genotype matrix for population 1 (individuals x loci),
                with missing values as np.nan.
            alignment2 (np.ndarray): Genotype matrix for population 2 (individuals x loci),
                with missing values as np.nan.
            min_call_rate (float): Minimum proportion of individuals with non-missing data required.

        Returns:
            tuple: Filtered (alignment1, alignment2) containing only loci with call rate >= min_call_rate
                in both populations.
        """
        call_rate1 = (
            np.count_nonzero(~np.isnan(alignment1), axis=0) / alignment1.shape[0]
        )
        call_rate2 = (
            np.count_nonzero(~np.isnan(alignment2), axis=0) / alignment2.shape[0]
        )
        valid = (call_rate1 >= min_call_rate) & (call_rate2 >= min_call_rate)
        return alignment1[:, valid], alignment2[:, valid]

    def weir_cockerham_fst(
        self, n_bootstraps: int = 0, n_jobs: int = 1, return_pvalues: bool = False
    ):
        """
        Calculate Weir and Cockerham's multi-locus Fst between populations,
        matching HierFstat's 'wc' approach for pairs of populations. We do so by
        summation of variance components (a, b, c) at each locus, using 0/1/2 biallelic encoding.

        When bootstrapping (n_bootstraps > 0) and return_pvalues=False,
        the function computes one overall Fst for each bootstrap replicate
        by resampling loci.

        When return_pvalues=True, a permutation (of individuals) approach
        is used to generate a null distribution of Fst values for each population
        pair, from which a one-tailed p-value is computed.

        Args:
            n_bootstraps (int): Number of replicates. If 0, no resampling is done
                                and a single multi-locus Fst is returned for each pair
                                (but stored as "abc_values" arrays for ratio-of-sums).
            n_jobs (int): Number of parallel jobs. Use -1 to use all available cores.
            return_pvalues (bool): If True, use the permutation approach to compute p-values.
                                   If False, return raw bootstrap replicate Fst values (via resampling loci).

        Returns:
            dict: The structure depends on the mode:

                1) If n_bootstraps == 0 and not return_pvalues:
                   returns { (pop1, pop2): {"abc_values": (a_arr, b_arr, c_arr)} }
                   i.e. the per-locus variance components for each pair.
                   You can convert them to a single multi-locus Fst by summation:
                       A = sum(a_arr); B = sum(b_arr); C = sum(c_arr)
                       Fst = A / (A + B + C)

                2) If n_bootstraps > 0 and return_pvalues == False:
                   returns { (pop1, pop2): np.array([...]) }
                   An array of length = n_bootstraps, each entry = multi-locus Fst
                   for that bootstrap replicate.

                3) If n_bootstraps > 0 and return_pvalues == True:
                   returns { (pop1, pop2):
                             { "fst": floatObservedFst,
                               "pvalue": pd.Series([...]) } }
                   Where "fst" is the observed multi-locus Fst, and "pvalue" is
                   the one-tailed permutation p-value (stored in a pd.Series).
        """

        pop_indices = self.genotype_data.get_population_indices()
        populations = list(pop_indices.keys())
        n_loci = self.alignment_012.shape[1]

        # Convert negative genotype calls to np.nan for missing data
        full_matrix = self.alignment_012.astype(float)
        full_matrix[full_matrix < 0] = np.nan

        # Extract sub-matrices for each population
        pop_alignments = {
            pop: full_matrix[inds, :] for pop, inds in pop_indices.items()
        }

        # ---------------------------
        # 1) Helper: Return per-locus a_i, b_i, c_i arrays (no ratio-of-sums yet)
        # ---------------------------
        def compute_fst_pair_return_abc(geno_pop1, geno_pop2):
            """
            Return arrays (a_vals, b_vals, c_vals) for each locus, so that
            multi-locus Fst = sum(a_vals)/sum(a_vals + b_vals + c_vals).

            This is the same formula used for 2-pop Weir & Cockerham (1984).
            """
            n_loci_local = geno_pop1.shape[1]
            a_vals, b_vals, c_vals = [], [], []

            for locus_i in range(n_loci_local):
                g1 = geno_pop1[:, locus_i]
                g2 = geno_pop2[:, locus_i]

                valid1 = ~np.isnan(g1)
                valid2 = ~np.isnan(g2)
                n1 = np.sum(valid1)
                n2 = np.sum(valid2)
                if n1 < 2 or n2 < 2:
                    # Skip locus if not enough typed individuals in either pop
                    continue

                # Allele frequencies
                p1 = np.nansum(g1[valid1]) / (2.0 * n1)
                p2 = np.nansum(g2[valid2]) / (2.0 * n2)

                # Observed heterozygosities
                H1 = np.mean(g1[valid1] == 1)
                H2 = np.mean(g2[valid2] == 1)

                N = n1 + n2
                if (N - 1) <= 0:
                    continue

                p_bar = (n1 * p1 + n2 * p2) / N
                H_bar = (n1 * H1 + n2 * H2) / N

                # Among-pops sum of squares
                D = n1 * (p1 - p_bar) ** 2 + n2 * (p2 - p_bar) ** 2

                # Effective sample size factor
                n_c = N - ((n1**2 + n2**2) / float(N))

                # Weir & Cockerham formula
                a_i = (n_c / (2.0 * (N - 1))) * (
                    D - (p_bar * (1 - p_bar) - (H_bar / 4.0)) / (N - 1)
                )
                b_i = p_bar * (1.0 - p_bar) - ((N - 1) / N) * D - (H_bar / 4.0)
                c_i = H_bar / 2.0

                denom = a_i + b_i + c_i
                if (not np.isnan(denom)) and (denom > 0):
                    a_vals.append(a_i)
                    b_vals.append(b_i)
                    c_vals.append(c_i)

            return np.array(a_vals), np.array(b_vals), np.array(c_vals)

        # ---------------------------
        # 2) Helper: Return single multi-locus Fst (ratio-of-sums)
        # ---------------------------
        def compute_fst_pair(geno_pop1, geno_pop2):
            """
            Directly compute sum(a_i) etc. to yield a single multi-locus Fst.
            """
            a_arr, b_arr, c_arr = compute_fst_pair_return_abc(geno_pop1, geno_pop2)
            if len(a_arr) == 0:
                return np.nan
            A = np.sum(a_arr)
            B = np.sum(b_arr)
            C = np.sum(c_arr)
            denom = A + B + C
            if denom <= 0:
                return np.nan
            return A / denom

        # ---------------------------------------------------------------------
        # CASE 1) No resampling: store "abc_values" for each pair
        # ---------------------------------------------------------------------
        if n_bootstraps == 0 and not return_pvalues:
            fst_per_population_pair = {}
            for pop1, pop2 in itertools.combinations(populations, 2):
                a_arr, b_arr, c_arr = compute_fst_pair_return_abc(
                    pop_alignments[pop1], pop_alignments[pop2]
                )
                # Instead of returning a series of Fst, store the raw components
                fst_per_population_pair[(pop1, pop2)] = {
                    "abc_values": (a_arr, b_arr, c_arr)
                }
            return fst_per_population_pair

        # ---------------------------------------------------------------------
        # CASE 2) If n_bootstraps > 0 AND we want p-values (permutation approach)
        # ---------------------------------------------------------------------
        if return_pvalues:
            # 2a) Observed overall Fst for each pair
            observed_overall = {}
            for pop1, pop2 in itertools.combinations(populations, 2):
                observed_overall[(pop1, pop2)] = compute_fst_pair(
                    pop_alignments[pop1], pop_alignments[pop2]
                )

            # 2b) Prepare placeholders for permutation replicates
            fst_permutation_overall = {
                (pop1, pop2): np.zeros(n_bootstraps, dtype=np.float64)
                for pop1, pop2 in itertools.combinations(populations, 2)
            }

            # 2c) Permutation replicate worker
            def permutation_replicate(seed):
                rng = np.random.default_rng(seed)
                # Combine all individuals
                all_inds = np.concatenate([pop_indices[pop] for pop in populations])
                permuted = rng.permutation(all_inds)

                # Reassign to each pop
                new_assignments = {}
                start = 0
                for pop in populations:
                    n_pop = len(pop_indices[pop])
                    new_assignments[pop] = permuted[start : start + n_pop]
                    start += n_pop

                # Build permuted genotype submatrices, compute Fst
                replicate_dict = {}
                for pop1, pop2 in itertools.combinations(populations, 2):
                    g1 = full_matrix[new_assignments[pop1], :]
                    g2 = full_matrix[new_assignments[pop2], :]
                    replicate_dict[(pop1, pop2)] = compute_fst_pair(g1, g2)
                return replicate_dict

            # 2d) Parallel permutations
            seeds = np.random.default_rng().integers(0, int(1e9), size=n_bootstraps)
            max_workers = mp.cpu_count() if n_jobs == -1 else n_jobs

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                permutation_results = list(executor.map(permutation_replicate, seeds))

            # 2e) Collate results
            for b_idx, replicate_dict in enumerate(permutation_results):
                for pop_pair, fst_val in replicate_dict.items():
                    fst_permutation_overall[pop_pair][b_idx] = fst_val

            # 2f) Compute one-tailed p-value
            results = {}
            for pop_pair, perm_array in fst_permutation_overall.items():
                obs = observed_overall[pop_pair]
                if np.isnan(obs):
                    p_val = 1.0
                else:
                    p_val = (np.sum(perm_array >= obs) + 1) / (n_bootstraps + 1)
                results[pop_pair] = {
                    "fst": obs,
                    "pvalue": pd.Series(
                        [p_val] * n_bootstraps,
                        name=f"P-value {pop_pair[0]}-{pop_pair[1]}",
                    ),
                }
            return results

        # ---------------------------------------------------------------------
        # CASE 3) If n_bootstraps > 0 and we do NOT want p-values (locus-resampling)
        # ---------------------------------------------------------------------
        else:
            fst_bootstrap_overall = {
                (pop1, pop2): np.zeros(n_bootstraps, dtype=np.float64)
                for pop1, pop2 in itertools.combinations(populations, 2)
            }

            def bootstrap_replicate(seed):
                """
                Resample loci with replacement, then compute one multi-locus Fst for each pair.
                """
                rng = np.random.default_rng(seed)
                resample_indices = rng.choice(n_loci, size=n_loci, replace=True)

                replicate_vals = {}
                for pop1, pop2 in itertools.combinations(populations, 2):
                    alignment1 = pop_alignments[pop1][:, resample_indices]
                    alignment2 = pop_alignments[pop2][:, resample_indices]
                    replicate_vals[(pop1, pop2)] = compute_fst_pair(
                        alignment1, alignment2
                    )
                return replicate_vals

            seeds = np.random.default_rng().integers(0, int(1e9), size=n_bootstraps)
            max_workers = mp.cpu_count() if n_jobs == -1 else n_jobs

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                boot_results = list(executor.map(bootstrap_replicate, seeds))

            # Collate
            for b_idx, replicate_dict in enumerate(boot_results):
                for pop_pair, fst_val in replicate_dict.items():
                    fst_bootstrap_overall[pop_pair][b_idx] = fst_val

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
