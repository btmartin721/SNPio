import itertools
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from pysam import index

import snpio.utils.custom_exceptions as exceptions
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC
from snpio.utils.multiqc_reporter import SNPioMultiQC


class FstDistance:
    """Class for calculating Fst distance between populations using Weir and Cockerham's method.

    This class provides methods to compute pairwise Fst values between populations based on genotype data. It supports both direct computation and permutation-based p-value calculations. The class also allows for locus-resampling bootstrap replicates to estimate confidence intervals. The results are saved to a CSV file for further analysis and plotted as a heatmap.
    """

    def __init__(self, genotype_data, plotter, verbose=False, debug=False):
        """Initialize the FstDistance class.

        This method sets up the logger and stores the genotype data. The genotype data should contain the necessary information for computing Fst values.

        Args:
            genotype_data: An object containing genotype data and population mapping.
            plotter: An object for plotting results.
            verbose (bool): If True, enables verbose logging.
            debug (bool): If True, enables debug logging.
        """
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.verbose = verbose

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
    def _uniq_alleles(str_list):
        """Return a set of unique alleles found in a list of phased genotype strings.

        Args:
            str_list (list): A list of phased genotype strings, where each element is a string of the form 'allele1/allele2'.

        Returns:
            set: A set of unique alleles found in the input list.

        Notes:
            - Assumes the input strings are in the format 'allele1/allele2'.
            - The function flattens the list of strings into a single list of alleles before extracting unique values.
            - For example, if the input is ['A/T', 'C/G', 'A/A'], the output will be {'A', 'T', 'C', 'G'}.
            - This function is useful for identifying all unique alleles present in a set of phased genotypes.

        Example:
            >>> str_list = ['A/T', 'C/G', 'A/A']
            >>> FstDistance._uniq_alleles(str_list)
            {'A', 'T', 'C', 'G'}
        """
        return set(sum([x.split("/") for x in str_list], []))

    @staticmethod
    def _get_alleles(str_list):
        """Flatten a list of phased genotype strings into a single list of alleles.

        Args:
            str_list (list): A list of phased genotype strings, where each element is a string of the form 'allele1/allele2'.

        Returns:
            list: A flattened list of alleles extracted from the input list.

        Notes:
            - This function splits each string in the input list by the '/' character and concatenates the resulting lists.
            - For example, if the input is ['A/T', 'C/G', 'A/A'], the output will be ['A', 'T', 'C', 'G', 'A', 'A'].
            - This function is useful for preparing allele data for further analysis.

        Example:
            >>> str_list = ['A/T', 'C/G', 'A/A']
            >>> FstDistance._get_alleles(str_list)
            ['A', 'T', 'C', 'G', 'A', 'A']
        """
        return sum([x.split("/") for x in str_list], [])

    def _get_het_from_phased(self, allele, phasedList, count=False):
        """Returns observed heterozygosity of an allele given a list of phased genotypes (e.g. allele1/allele2 for each individual). Assumes diploid.

        This function counts the number of heterozygotes for a given allele in a list of phased genotypes. It can return either the count or the proportion of heterozygotes. The function checks if the input genotypes are phased (i.e., in the format 'allele1/allele2') and raises an error if not. It also handles cases where the allele is not present in the genotypes.

        Args:
            allele (str): The allele for which to compute heterozygosity.
            phasedList (list): A list of phased genotypes, where each element is a string of the form 'allele1/allele2'.
            count (bool): Whether to return the number of heterozygotes instead of the proportion.

        Returns:
            float: The proportion of heterozygotes for the given allele, unless `count` is True, in which case the count of heterozygotes is returned.

        Raises:
            ValueError: If the input genotypes are not phased (i.e., do not contain '/').
            ValueError: If the input list is empty or contains invalid genotypes.
            ValueError: If the allele is not found in the genotypes.

        Notes:
            - This function assumes that the input genotypes are diploid and in the format 'allele1/allele2'.
            - The function counts heterozygotes as individuals with one allele different from the specified allele.
            - If the allele is not found in any of the genotypes, the function will return 0.0 for the proportion or count.
            - The function can be used to assess genetic diversity in populations by calculating heterozygosity.

        Example:
            >>> phased_list = ['A/A', 'A/T', 'C/G', 'T/T']
            >>> allele = 'A'
            >>> self._get_het_from_phased(allele, phased_list)
            0.5
        """
        hets = 0.0
        twoN = (len(phasedList)) * 2.0

        if not all("/" in x for x in phasedList):
            msg = "All inputs must be phased genotypes, e.g. 'A/T'."
            self.logger.error(msg)
            raise exceptions.InvalidGenotypeError(msg)

        for genotype in phasedList:
            gens = genotype.split("/")
            if gens[0] == allele and gens[1] != allele:
                hets += 1.0
                continue
            elif gens[1] == allele and gens[0] != allele:
                hets += 1.0
                continue
            else:
                continue
        if count:
            return hets
        else:
            return hets / twoN

    def _two_pop_weir_cockerham_fst(self, s1, s2):
        """Computes Weir and Cockerham's THETAst Fst approximation for a SINGLE locus in two populations.

        This function calculates the Fst value for a single locus between two populations using Weir and Cockerham's method. It takes two lists of phased genotypes (e.g., 'A/A', 'A/T') as input and returns the numerator and denominator for the Fst calculation. The function also performs input validation to ensure that the inputs are in the correct format.

        Args:
            s1 (array-like): A 1D array (or list) of phased genotypes for population 1 at a single locus. Each element is a string like 'A/A' or 'A/T'.
            s2 (array-like): A 1D array (or list) of phased genotypes for population 2 at the same single locus.

        Returns:
            tuple: (num, denom), i.e. the single-locus numerator and denominator. Fst at this locus would be num / denom (if denom > 0).

        Raises:
            ValueError: If the input lists are not valid phased genotypes.
            ValueError: If the input lists are empty or contain invalid genotypes.
            ValueError: If the input lists contain unknown or gap alleles.
            ValueError: If the input lists contain non-string elements.
            ValueError: If the input lists contain invalid phased genotypes (not in the form 'allele1/allele2').
        Notes:
            - This function assumes that the input genotypes are diploid and in the format 'allele1/allele2'.
            - The function counts heterozygotes as individuals with one allele different from the specified allele.
            - The function can be used to assess genetic diversity in populations by calculating heterozygosity.
            - The function is designed to be used as part of a larger analysis of genetic diversity and population structure.
            - The function is not intended for use with unphased genotypes or non-genetic data.

        Example:
            >>> s1 = ['A/A', 'A/T', 'C/G', 'T/T']
            >>> s2 = ['A/A', 'T/T', 'C/G', 'G/G']
            >>> num, denom = self._two_pop_weir_cockerham_fst(s1, s2)
            >>> print(num, denom)
            (0.5, 1.0)
        """
        # Quick validation
        if (not isinstance(s1, list)) or (not isinstance(s2, list)):
            msg = "Inputs must be lists."
            self.logger.error(msg)
            raise exceptions.InvalidGenotypeError(msg)

        if (not s1) or (not s2):
            msg = "Inputs must not be empty."
            self.logger.error(msg)
            raise exceptions.EmptyIterableError(msg)

        if (not all(isinstance(x, str) for x in s1)) or (
            not all(isinstance(x, str) for x in s2)
        ):
            msg = "All inputs must be strings (phased genotypes)."
            self.logger.error(msg)
            raise exceptions.InvalidGenotypeError(msg)

        if (not all("/" in x for x in s1)) or (not all("/" in x for x in s2)):
            msg = "All inputs must be phased genotypes, e.g. 'A/T'."
            self.logger.error(msg)
            raise exceptions.InvalidGenotypeError(msg)

        # Initialize variables
        num = 0.0
        denom = 0.0

        # mean sample size
        alleles1 = FstDistance._get_alleles(s1)  # split alleles s1
        alleles2 = FstDistance._get_alleles(s2)  # split alleles s2
        uniques = FstDistance._uniq_alleles(s1 + s2)  # list of unique alleles only

        r = 2.0  # number of pops
        n1 = float(len(s1))  # pop size of pop 1
        n2 = float(len(s2))  # pop size of pop 2
        csd = np.std([n1, n2])
        cm = np.mean([n1, n2])
        nbar = cm
        csquare = (csd * csd) / (cm * cm)
        nC = nbar * (1.0 - (csquare / r))  # coeff of pop size variance
        for allele in uniques:
            ac1 = float(alleles1.count(allele))
            ac2 = float(alleles2.count(allele))

            # p1 is the frequency of allele in pop1.
            # p2 is the frequency of allele in pop2.
            p1 = ac1 / float(len(alleles1))
            p2 = ac2 / float(len(alleles2))

            # pbar is the mean frequency of the allele across both pops.
            h1 = self._get_het_from_phased(allele, s1, count=True)
            h2 = self._get_het_from_phased(allele, s2, count=True)
            pbar = (ac1 + ac2) / (float(len(alleles1)) + float(len(alleles2)))

            # Compute the numerator and denominator for Fst
            # using Weir and Cockerham's method
            # ssquare is the variance of the allele frequency
            # across the two populations.
            # hbar is the mean heterozygosity across the two populations.
            # nbar is the mean sample size across the two populations.
            # nC is the coefficient of variation of the sample size.
            # a is the numerator for Fst.
            # b is the denominator for Fst.
            # c is a correction term for the denominator.
            # d is the final denominator for Fst.
            # num is the sum of the numerators for all alleles.
            # denom is the sum of the denominators for all alleles.
            ssquare = (
                np.sum([(n1 * (np.square(p1 - pbar))), (n2 * (np.square(p2 - pbar)))])
            ) / ((r - 1.0) * nbar)
            hbar = (h1 + h2) / (r * nbar)

            if nbar != 1.0:
                a = (nbar / nC) * (
                    ssquare
                    - (
                        (1.0 / (nbar - 1.0))
                        * (
                            (pbar * (1.0 - pbar))
                            - ((r - 1.0) * ssquare / r)
                            - (hbar / 4.0)
                        )
                    )
                )

                b = (nbar / (nbar - 1.0)) * (
                    (pbar * (1.0 - pbar))
                    - ((r - 1.0) * ssquare / r)
                    - (((2.0 * nbar) - 1.0) * hbar / (4.0 * nbar))
                )

                c = hbar / 2.0
                d = a + b + c
                num += a
                denom += d

        return num, denom

    @staticmethod
    def _clean_inds(inds):
        """Removes individuals with unknown or gap alleles.

        This function filters out individuals from a list of phased genotypes that contain unknown or gap alleles. It checks for the presence of '-', '?', 'n', or 'N' in the genotype strings and excludes those individuals from the output list.

        Args:
            inds (list): A list of individuals.

        Returns:
            list: A filtered list of individuals without unknown or gap alleles.

        Notes:
            - The function assumes that the input list contains strings representing phased genotypes.
            - The function is useful for cleaning genotype data before further analysis.

        Example:
            >>> inds = ['A/A', 'A/T', 'C/G', 'N/N', 'T/T']
            >>> FstDistance._clean_inds(inds)
            ['A/A', 'A/T', 'C/G', 'T/T']
        """
        ret = []
        for ind in inds:
            if "-" not in ind and "?" not in ind and "n" not in ind and "N" not in ind:
                ret.append(ind)
        return ret

    def _compute_multilocus_fst(
        self, full_matrix, pop_indices, n_loci, pop1_name, pop2_name, locus_subset=None
    ):
        """Compute multilocus Fst (sum(a_i)/sum(a_i+b_i+c_i)) using vectorized operations."""
        if locus_subset is None:
            locus_subset = range(n_loci)

        p1_inds = pop_indices[pop1_name]
        p2_inds = pop_indices[pop2_name]

        pop1_matrix = full_matrix[p1_inds, :]
        pop2_matrix = full_matrix[p2_inds, :]

        phased_encoding = self.iupac.get_phased_encoding()
        encode = np.vectorize(lambda x: phased_encoding.get(x, x))

        num_list = []
        denom_list = []

        for loc in locus_subset:
            seqs1 = pop1_matrix[:, loc].astype("U")
            seqs2 = pop2_matrix[:, loc].astype("U")

            # Vectorized masking of ambiguous genotypes
            def is_clean(arr):
                return (
                    (np.char.find(arr, "-") == -1)
                    & (np.char.find(arr, "?") == -1)
                    & (np.char.find(arr, "n") == -1)
                    & (np.char.find(arr, "N") == -1)
                )

            clean1 = seqs1[is_clean(seqs1)]
            clean2 = seqs2[is_clean(seqs2)]

            if clean1.size == 0 or clean2.size == 0:
                continue

            encoded1 = encode(clean1)
            encoded2 = encode(clean2)

            try:
                a_val, d_val = self._two_pop_weir_cockerham_fst(
                    encoded1.tolist(), encoded2.tolist()
                )
                num_list.append(a_val)
                denom_list.append(d_val)
            except ValueError:
                continue

        numerator = np.nansum(num_list)
        denominator = np.nansum(denom_list)
        return np.nan if denominator == 0 else numerator / denominator

    def _pairwise_permutation_test(
        self,
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
        n_permutations: int = 1000,
        seed: int | None = None,
    ):
        """Permutation test for multilocus Fst using locus resampling (hierfstat-style).

        This method performs a permutation test to compute the observed Fst value and its empirical p-value by resampling loci. It uses the Weir and Cockerham method for Fst calculation and returns the observed Fst, empirical p-value, and bootstrap distribution of Fst values.

        Args:
            pop1_inds (np.ndarray): Indices of individuals in population 1.
            pop2_inds (np.ndarray): Indices of individuals in population 2.
            full_matrix (np.ndarray): Genotype matrix (individuals x loci).
            n_permutations (int): Number of locus permutation resampling replicates.
            seed (int | None): Random number generator seed.

        Returns:
            tuple: (observed Fst, empirical P-value, bootstrap Fst distribution)
        """
        return self.fst_permutation_pval_locus_resampling(
            pop1_inds, pop2_inds, full_matrix, n_permutations=n_permutations, seed=seed
        )

    def weir_cockerham_fst(
        self, n_permutations: int = 0, n_jobs: int = 1, return_pvalues: bool = False
    ):
        """Calculate Weir and Cockerham's multi-locus Fst between populations.

            This method can perform three different analyses:

            1. Calculate pairwise Fst between all populations (no bootstraps, no p-values).
            2. Calculate pairwise Fst between all populations and compute permutation-based p-values.
            3. Calculate pairwise Fst between all populations using locus-resampling bootstrap replicates (no p-values).

        Notes:
            The method returns a DataFrame of pairwise Fst values, or a dictionary of bootstrap or permutation results.
            The method also saves the results to a CSV file.
            The output structure depends on the case:
            - Case 1: DataFrame of pairwise Fst values.
            - Case 2: Dictionary with observed Fst and p-values.
            - Case 3: Dictionary with bootstrap replicate Fst values.
            The method also handles parallel processing for speedup.
            The method saves the results to a CSV file in the specified output directory.
            The output file is named "{prefix}_output/analysis/pairwise_WC_fst.json" or, if filtering was done, then "{prefix}_output/nremover/analysis/pairwise_WC_fst.json".

        Args:
            n_permutations (int): Number of permutation replicates. Defaults to 0 (no bootstraps, no p-values).
            n_jobs (int): Number of parallel jobs. If -1, all available cores are used. Defaults to 1.
            return_pvalues (bool): If True, compute a permutation-based p-value (Case 2). Otherwise, if n_permutations>0, do locus-resampling bootstrap (Case 3).

        Returns:
            Depending on the case:
            - Case 1 (n_permutations=0, return_pvalues=False): pd.DataFrame of pairwise Fst.
            - Case 2 (n_permutations>0, return_pvalues=True): Dict of form
                {
                    (pop1, pop2): {
                        "fst": observedValue,
                        "pvalue": pd.Series([...])  # same p-value repeated or entire distribution
                    }
                }
            - Case 3 (n_bootstraps>0, return_pvalues=False): Dict of form
                {
                    (pop1, pop2): np.array([...])  # one Fst per bootstrap replicate
                }

        Raises:
            ValueError: If n_permutations < 0 or n_jobs < -1.
            ValueError: If return_pvalues is True and n_permutations <= 0.
            ValueError: If n_jobs is not an integer or is less than -1.

        Examples:
            >>> fst_distance = FstDistance(genotype_data, verbose=True, debug=False)
            >>> result = fst_distance.weir_cockerham_fst(n_permutations=100, n_jobs=4, return_pvalues=True)
            >>> print(result)
            {
                ('pop1', 'pop2'): {
                    "fst": observedValue,
                    "pvalue": pd.Series([...])
                }
            }

            >>> result = fst_distance.weir_cockerham_fst(n_bootstraps=0, n_jobs=1, return_pvalues=False)
            >>> print(result)
            >>> # pd.DataFrame of pairwise Fst values
        """
        self.logger.info(
            "Calculating pairwise Weir and Cockerham's Fst between populations."
        )
        # Validate inputs
        if n_permutations < 0:
            msg = "n_permutations must be >= 0."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_jobs < -1:
            msg = "n_jobs must be an integer >= -1."
            self.logger.error(msg)
            raise ValueError(msg)

        if return_pvalues and n_permutations <= 0:
            msg = "If return_pvalues is True, n_permutations must be > 0."
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(n_jobs, int):
            msg = "n_jobs must be an integer."
            self.logger.error(msg)
            raise ValueError(msg)

        # If n_jobs is -1, use all available cores
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs == 0 or n_jobs < -1:
            msg = "n_jobs must be a positive integer or -1 for all cores."
            self.logger.error(msg)
            raise ValueError(msg)

        # A) Build sample -> index map
        # e.g. {'EA': [...], 'GU': [...], ...}
        popmap = self.genotype_data.popmap_inverse

        # e.g. list of all sample IDs in the correct row order
        sample_names = self.genotype_data.samples

        # e.g. {'sample1': 0, 'sample2': 1, ...}
        sample_to_idx = {name: i for i, name in enumerate(sample_names)}

        # B) Convert popmap into integer row indices for each
        # population
        pop_indices = {}
        for pop_name, sample_list in popmap.items():
            # turn each sample ID into an integer row index
            pop_indices[pop_name] = np.array(
                [sample_to_idx[s] for s in sample_list], dtype=int
            )

        # C) Prelim variables
        full_matrix = self.genotype_data.snp_data.astype(str)
        n_loci = full_matrix.shape[1]
        pop_keys = list(popmap.keys())
        num_populations = len(pop_keys)

        # ----------------------------------------------------
        # CASE 1: No bootstraps, no p-values
        # ----------------------------------------------------
        if n_permutations == 0 and not return_pvalues:
            return self._compute_pw_fst_no_bootstrap(
                full_matrix.copy(), num_populations, pop_keys, pop_indices, n_loci
            )

        # --------------------------------------------------
        # CASE 2: Permutation test for p-values
        # (return_pvalues=True)
        # --------------------------------------------------
        elif return_pvalues and n_permutations > 0:
            result_dict = {}
            for ia, ib in itertools.combinations(range(num_populations), 2):
                pop1_name = pop_keys[ia]
                pop2_name = pop_keys[ib]
                pop1_inds = pop_indices[pop1_name]
                pop2_inds = pop_indices[pop2_name]

                # Check and log Fst stability of the denominator
                df_fst_stability = self._check_fst_denominator_stability(
                    pop1_inds, pop2_inds, full_matrix.copy()
                )

                df_fst_stability.to_csv(
                    self.outdir / f"{pop1_name}_{pop2_name}_fst_variance_stability.csv",
                    index=True,
                )

                obs_fst, p_val, perm_dist = self._pairwise_permutation_test(
                    pop1_inds,
                    pop2_inds,
                    full_matrix.copy(),
                    n_permutations=n_permutations,
                )

                result_dict[(pop1_name, pop2_name)] = {
                    "fst": obs_fst,
                    "pvalue": p_val,
                    "perm_dist": perm_dist,
                }

                self.plotter.plot_permutation_dist(
                    obs_fst, perm_dist, pop1_name, pop2_name
                )

            return result_dict

        # ---------------------------------------------------
        # CASE 3: Locus-resampling bootstrap (no p-values)
        # ---------------------------------------------------
        else:  # n_bootstraps > 0 and not return_pvalues
            # Does confidence interval calculation.
            return self._compute_pw_fst_with_bootstrap(
                full_matrix.copy(),
                pop_indices,
                n_loci,
                num_populations,
                pop_keys,
                n_permutations,
                n_jobs,
            )

    def _compute_pw_fst_with_bootstrap(
        self,
        full_matrix,
        pop_indices,
        n_loci,
        num_populations,
        pop_keys,
        n_bootstraps,
        n_jobs,
    ):
        """Compute pairwise Fst for all populations using bootstrap replicates.

        This function computes pairwise Fst values for all pairs of populations in the genotype data using bootstrap replicates. It handles parallel processing for speedup.

        Args:
            full_matrix (np.ndarray): The full genotype matrix.
            pop_indices (dict): A dictionary mapping population names to indices.
            n_loci (int): The number of loci in the genotype matrix.
            num_populations (int): The number of populations.
            pop_keys (list): A list of population names.
            n_bootstraps (int): Number of bootstrap replicates.
            n_jobs (int): Number of parallel jobs.

        Returns:
            dict: A dictionary containing bootstrap replicate Fst values for each pair of populations.

        Notes:
            - This function assumes that the genotype data is stored in a specific format and that the necessary libraries are available.
            - The function handles missing values and computes Fst values for all pairs of populations.
            - The results are saved to a CSV file in the specified output directory.
        """
        # Store an array of length n_bootstraps for each pop pair
        result_dict = {
            (pop_keys[ia], pop_keys[ib]): np.zeros(n_bootstraps, dtype=float)
            for ia, ib in itertools.combinations(range(num_populations), 2)
        }

        # Worker function for a single bootstrap replicate
        def bootstrap_replicate(seed):
            rng = np.random.default_rng(seed)
            resample_idx = rng.choice(n_loci, size=n_loci, replace=True)

            replicate_vals = {}
            for ia, ib in itertools.combinations(range(num_populations), 2):
                pop1_name = pop_keys[ia]
                pop2_name = pop_keys[ib]
                fst_val = self._compute_multilocus_fst(
                    full_matrix.copy(),
                    pop_indices,
                    n_loci,
                    pop1_name,
                    pop2_name,
                    locus_subset=resample_idx,
                )
                replicate_vals[(pop1_name, pop2_name)] = fst_val
            return replicate_vals

        seeds = np.random.default_rng().integers(0, 1e9, size=n_bootstraps)
        max_workers = mp.cpu_count() if n_jobs == -1 else n_jobs

        # Run in parallel
        boot_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for replicate_dict in executor.map(bootstrap_replicate, seeds):
                boot_results.append(replicate_dict)

        # Collate bootstrap results
        for b_idx, replicate_dict in enumerate(boot_results):
            for pop_pair, fst_val in replicate_dict.items():
                result_dict[pop_pair][b_idx] = fst_val

        return result_dict

    def _compute_pw_fst_no_bootstrap(
        self, full_matrix, num_populations, pop_keys, pop_indices, n_loci
    ):
        """Compute pairwise Fst for all populations without bootstrapping or permutations.

        This function computes pairwise Fst values for all pairs of populations in the genotype data. It uses the Weir and Cockerham method for Fst calculation and saves the results to a CSV file.

        Args:
            full_matrix (np.ndarray): The full genotype matrix.
            num_populations (int): The number of populations.
            pop_keys (list): A list of population names.
            pop_indices (dict): A dictionary mapping population names to indices.
            n_loci (int): The number of loci in the genotype matrix.

        Returns:
            pd.DataFrame: A DataFrame containing pairwise Fst values for all populations.
        Notes:
            - The function assumes that the genotype data is stored in a specific format and that the necessary libraries are available.
            - The function handles missing values and computes Fst values for all pairs of populations.
            - The results are saved to a CSV file in the specified output directory.
        """
        genmat = np.full((num_populations, num_populations), np.nan, dtype=float)

        for ia, ib in itertools.combinations(range(num_populations), 2):
            pop1_name = pop_keys[ia]
            pop2_name = pop_keys[ib]

            fst_val = self._compute_multilocus_fst(
                full_matrix.copy(), pop_indices, n_loci, pop1_name, pop2_name
            )
            genmat[ia, ib] = fst_val
            genmat[ib, ia] = fst_val

        # Diagonal = 0.0
        np.fill_diagonal(genmat, 0.0)

        df_fst = pd.DataFrame(genmat, index=pop_keys, columns=pop_keys, dtype=float)

        self.logger.info(
            f"Pairwise Fsts saved to {str(self.outdir / 'pairwise_WC_fst_mqc.json')}"
        )

        return df_fst

    def _parse_wc_fst(self, result_dict, alpha: float = 0.05):
        """Convert the output of `weir_cockerham_fst()` into DataFrames for:

        - The mean Fst among permutations or bootstraps,
        - The lower and upper confidence intervals,
        - And the p-values (if return_pvalues=True).

        This method auto-detects which case of result_dict it has:
        1) A direct DataFrame (case: no bootstrap, no p-values).
        2) A dict {(pop1, pop2): np.array([...])} for bootstrap replicates.
        3) A dict {(pop1, pop2): {"fst": float, "pvalue": pd.Series, "perm_dist": np.array([...])} for permutation results with an optional distribution array.

        Args:
            result_dict: The structure returned by `weir_cockerham_fst()`.
            alpha (float): Significance level for CIs. Default 0.05 => 95% CIs.

        Returns:
            tuple: (df_mean, df_lower, df_upper, df_pval), where each is a
            pandas DataFrame (or None if not applicable).
            - df_mean: matrix of average Fst across replicates (or observed if no replicates).
            - df_lower: matrix of lower CI bounds.
            - df_upper: matrix of upper CI bounds.
            - df_pval: matrix of p-values if p-values exist; otherwise None.

        Notes:
            - For bootstrap results, df_mean, df_lower, and df_upper are computed from the replicate arrays in result_dict.
            - For permutation results, the method looks for "perm_dist" to compute a distribution-based mean and CIs. If "perm_dist" is missing, df_lower and df_upper will remain NaN.
            - If result_dict is just a DataFrame, it returns that as df_mean  and None for the others, since no replicates/p-values exist.
        """
        # ------------------------------------------------------------
        # CASE 1: No permutations, no p-values
        # Expected data structure (DataFrame): pairwise Fst matrix
        # ------------------------------------------------------------
        if isinstance(result_dict, pd.DataFrame):
            # No distribution or p-values to parse, just plot and return matrix
            self.snpio_mqc.queue_heatmap(
                df=result_dict,
                panel_id="pairwise_WC_fst",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham (1984) Fst",
                description="Pairwise Weir & Cockerham (1984) Fst values computed for all population pairs. No permutation replicates were performed.",
                index_label="Population",
                pconfig={
                    "title": "SNPio: Pairwise Weir & Cockerham Fst",
                    "id": "pairwise_WC_fst",
                    "xlab": "Population ID",
                    "ylab": "Population ID",
                    "zlab": "Observed Fst",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                },
            )
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
        df_mean, df_lower, df_upper, df_pval = None, None, None, None

        # ---------------------------------------------------------------------
        # CASE 2: With permutation replicates, but no P-value estimation
        # Expected data structure (Dictionary): (pop1, pop2) -> np.array([...])
        # ---------------------------------------------------------------------
        if isinstance(first_val, np.ndarray):
            # This indicates we likely have arrays of replicate
            # Fst values => permutation replicates.
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

            # Diagonal is typically 0 for self-Fst
            np.fill_diagonal(df_mean.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            # P-values don't exist for a standard bootstrap approach
            df_pval = None

            # Write df_lower and df_upper to CSV
            df_ul_combined = self._combine_upper_lower_ci(
                df_upper, df_lower, diagonal="zero"
            )

            self.snpio_mqc.queue_heatmap(
                df=df_mean,
                panel_id="pairwise_WC_fst",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst (Permutation Mean)",
                description="Mean Pairwise Weir & Cockerham (1984) Fst estimates computed for all population pairs. Mean values are based on permutation replicates.",
                index_label="Population",
                pconfig={
                    "title": "SNPio: Mean Pairwise Weir & Cockerham Fst",
                    "id": "pairwise_WC_fst",
                    "xlab": "Population ID",
                    "ylab": "Population ID",
                    "zlab": "Mean Observed Fst",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                },
            )

            self.snpio_mqc.queue_heatmap(
                df=df_ul_combined,
                panel_id="pairwise_WC_fst_ci95",
                section="genetic_differentiation",
                title="SNPio: Pairwise 95 Percent CIs (upper and lower) for Weir & Cockerham Fst Estimates",
                description="Upper and lower 95 percent confidence intervals for Pairwise Weir & Cockerham (1984) Fst values computed for all population pairs. The upper and lower confidence intervals are computed from the permutation replicates and are represented in the upper and lower triangles of the heatmap, respectively.",
                index_label="Population",
                pconfig={
                    "title": "SNPio: Pairwise Weir & Cockerham Fst with 95 Percent CIs",
                    "id": "pairwise_WC_fst_ci95",
                    "xlab": "Population ID",
                    "ylab": "Population ID",
                    "zlab": "Fst with 95% CIs",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                },
            )

            return df_mean, df_lower, df_upper, df_pval

        # -----------------------------------------------------------------
        # CASE 3: Fst estimation with permutations and P-values
        # Expected data structure:
        # (Permutation): (pop1, pop2) ->
        # {"fst": float, "pvalue": pd.Series, "perm_dist": optional ...}
        # -----------------------------------------------------------------
        if isinstance(first_val, dict) and "fst" in first_val and "pvalue" in first_val:
            pop_pairs = list(result_dict.keys())
            pops = all_populations_from_keys(pop_pairs)

            # Observed Fst
            df_obs = pd.DataFrame(np.nan, index=pops, columns=pops)

            # We store p-values in df_pval
            df_pval = pd.DataFrame(np.nan, index=pops, columns=pops)

            # If there's a distribution, we can compute mean & CI
            df_mean = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_lower = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_upper = pd.DataFrame(np.nan, index=pops, columns=pops)

            for (p1, p2), subdict in result_dict.items():
                obs_val = subdict["fst"]
                pval_series = pd.Series(subdict["pvalue"])

                # Check if we have a distribution of permuted Fst
                # If not, we can still compute mean & CIs
                dist = subdict.get("perm_dist", None)

                # Fill in the observed Fst
                df_obs.loc[p1, p2] = obs_val
                df_obs.loc[p2, p1] = obs_val

                # Extract the p-value (one-tailed)
                if not pval_series.empty:
                    p_value = pval_series.mean()
                    df_pval.loc[p1, p2] = p_value
                    df_pval.loc[p2, p1] = p_value

                # If we have a distribution of permuted Fst, compute its mean & CIs
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

            self.snpio_mqc.queue_heatmap(
                df=df_obs,
                panel_id="pairwise_WC_fst",
                section="genetic_differentiation",
                title="SNPio: Mean Pairwise Weir & Cockerham Fst",
                description="Mean Pairwise Weir & Cockerham (1984) Fst estimates computed for all population pairs, calculated from permutation replicates.",
                index_label="Population",
                pconfig={
                    "title": "SNPio: Mean Pairwise Weir & Cockerham Fst",
                    "id": "pairwise_WC_fst",
                    "xlab": "Population ID 1",
                    "ylab": "Population ID 2",
                    "zlab": "Mean Observed Fst",
                    "reverse_colors": True,
                    "min": 0.0,
                    "max": 1.0,
                    "tt_decimals": 3,
                },
            )

            self.snpio_mqc.queue_heatmap(
                df=df_pval,
                panel_id="pairwise_WC_fst_pvalues",
                section="genetic_differentiation",
                title="SNPio: P-values from Pairwise Weir & Cockerham Fst Estimates",
                description="Pairwise Weir & Cockerham (1984) Fst P-values computed for all population pairs. The p-values are based on the distribution estimated from permutation replicates and represent the statistical significance of the observed Fst values.",
                index_label="Population",
                pconfig={
                    "title": "SNPio: P-values from Pairwise Weir & Cockerham Fst Estimates",
                    "id": "pairwise_WC_fst_pvalues",
                    "xlab": "Population ID 1",
                    "ylab": "Population ID 2",
                    "zlab": "P-value",
                    "reverse_colors": False,
                    "min": 0.0,
                    "max": 1.0,
                    "tt_decimals": 3,
                },
            )

            return df_obs, df_lower, df_upper, df_pval

        # --------------------------------------------------------
        # Else: If none of the above conditions match, raise error
        # --------------------------------------------------------
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

            phased_encoding = self.iupac.get_phased_encoding()
            s1 = [phased_encoding.get(x, x) for x in s1]
            s2 = [phased_encoding.get(x, x) for x in s2]

            try:
                a, d = self._two_pop_weir_cockerham_fst(s1, s2)
                a_vals[loc] = a
                d_vals[loc] = d
            except ValueError:
                a_vals[loc] = np.nan
                d_vals[loc] = np.nan

        return a_vals, d_vals

    def fst_permutation_pval_locus_resampling(
        self,
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
        n_permutations: int = 1000,
        seed: int | None = None,
    ):
        """Compute permutation p-value by shuffling individual labels between populations.

        This method calculates the observed multi-locus Fst between two populations and then generates a null distribution under the hypothesis of no population structure by randomly permuting individuals between the two groups.

        Args:
            pop1_inds (np.ndarray): Indices of individuals in population 1.
            pop2_inds (np.ndarray): Indices of individuals in population 2.
            full_matrix (np.ndarray): Genotype matrix (individuals x loci).
            n_permutations (int): Number of permutation replicates to perform. Must be a positive integer.
            seed (int | None): Random seed for reproducibility. If None, uses a random seed.

        Returns:
            tuple:
                float: Observed Fst between the two populations.
                float: Empirical p-value as (count(null >= observed) + 1) / (n + 1).
                np.ndarray: Array of permuted Fst values.
        """
        if seed is None:
            rng = np.random.default_rng()
        else:
            if not isinstance(seed, int):
                msg = f"Seed must be an integer or None, but got: {type(seed)}"
                self.logger.error(msg)
                raise TypeError(msg)
            if seed < 0:
                msg = f"Seed must be a non-negative integer, but got a negative value: {seed}"
                self.logger.error(msg)
                raise ValueError(msg)

            # Use the provided seed for reproducibility
            rng = np.random.default_rng(seed)

        # Combine and count sizes
        combined = np.concatenate([pop1_inds, pop2_inds])
        n1 = len(pop1_inds)

        # 1. Observed Fst
        pop_indices = {"A": pop1_inds, "B": pop2_inds}
        obs_fst = self._compute_multilocus_fst(
            full_matrix, pop_indices, full_matrix.shape[1], "A", "B"
        )

        # 2. Build null distribution by permuting labels
        dist = np.empty(n_permutations, dtype=float)
        for i in tqdm(range(n_permutations), desc="Fst Permutations", unit="perms"):
            permuted = rng.permutation(combined)
            perm_A = permuted[:n1]
            perm_B = permuted[n1:]
            perm_indices = {"A": perm_A, "B": perm_B}
            dist[i] = self._compute_multilocus_fst(
                full_matrix, perm_indices, full_matrix.shape[1], "A", "B"
            )

        # 3. Filter out any NaNs
        valid = ~np.isnan(dist)
        if not np.any(valid):
            # No valid permutations => return p-value 1.0
            return obs_fst, 1.0, dist
        dist = dist[valid]

        # 4. One-tailed p-value: fraction of permuted >= observed
        p_val = (np.sum(dist >= obs_fst) + 1) / (len(dist) + 1)
        return obs_fst, p_val, dist

    def _check_fst_denominator_stability(
        self,
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
        threshold: float = 1e-6,
        report: bool = True,
    ) -> pd.DataFrame:
        """Check denominator stability (d_i) per-locus to assess Fst stability.

        This method evaluates the stability of the Fst denominator (d_i) for each locus by checking for zero or near-zero values. It computes the numerator (a_i) and denominator (d_i) for each locus and categorizes them into 'OK', 'Zero', or 'Unstable' based on the value of d_i. It also logs a summary of loci that may yield unstable Fst values.

        Args:
            pop1_inds (np.ndarray): Indices of individuals in population 1.
            pop2_inds (np.ndarray): Indices of individuals in population 2.
            full_matrix (np.ndarray): Genotype matrix (individuals x loci).
            threshold (float): Minimum acceptable value for the denominator (d_i). Default is 1e-6.
            report (bool): If True, logs a summary of unstable loci.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - 'Locus': Locus index
                - 'a': Numerator
                - 'd': Denominator
                - 'Status': 'OK', 'Zero', or 'Unstable'
        """
        a_vals, d_vals = self.fst_variance_components_per_locus(
            pop1_inds, pop2_inds, full_matrix
        )

        results = []
        for i, (a, d) in enumerate(zip(a_vals, d_vals)):
            if np.isnan(d):
                status = "Failed"
            elif d == 0.0:
                status = "Zero"
            elif abs(d) < threshold:
                status = "Unstable"
            else:
                status = "OK"
            results.append((i, a, d, status))

        df = pd.DataFrame(results, columns=["Locus", "a", "d", "Status"])

        if report:
            num_zero = (df["Status"] == "Zero").sum()
            num_unstable = (df["Status"] == "Unstable").sum()
            num_nan = (df["Status"] == "NaN").sum()
            total = len(df)

            self.logger.info(f"Fst denominator stability check:")
            self.logger.info(f"  Total loci evaluated: {total}")
            self.logger.info(f"  Zero denominator loci: {num_zero}")
            self.logger.info(f"  Near-zero (< {threshold}) loci: {num_unstable}")
            self.logger.info(f"  NaN loci: {num_nan}")

            if num_zero + num_unstable + num_nan > 0:
                if self.verbose:
                    self.logger.warning(
                        "  Some loci have zero or near-zero denominators, which may lead to unstable Fst values."
                    )
        return df
