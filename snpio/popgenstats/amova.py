import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from snpio.utils.misc import IUPAC


class AMOVA:
    """Analysis of Molecular Variance (AMOVA) class.

    This class provides methods for conducting AMOVA with 1, 2, or 3 hierarchical levels. The AMOVA method partitions genetic variation into components due to differences among populations, among individuals within populations, and within individuals. This method calculates variance components and Phi statistics for a given number of hierarchical levels (1, 2, or 3). If bootstrapping is enabled, it also estimates p-values for the variance components. The number of hierarchical levels determines the structure of the AMOVA model: 1 => populations only, 2 => region -> populations, 3 => region -> population -> individuals. If regionmap is provided, it is used to map populations to regions.

    Notes:
        - Algorithm adapted from the R package 'poppr' (Kamvar et al., 2014) and Excoffier et al. (1992).
        - The Phi statistic is a measure of genetic differentiation.
        - Bootstrapping is implemented by resampling SNP loci with replacement.
    """

    def __init__(
        self,
        genotype_data: Any,
        alignment: np.ndarray,
        logger: logging.Logger | None = None,
    ):
        """Initialize the AMOVA object.

        Args:
            genotype_data (GenotypeData): Genotype data object (must have .popmap_inverse, .samples).
            alignment (np.ndarray): Genotype data in 012 format (shape: [n_samples, n_loci]).
            logger (Logger | None): Logger object for debug/info output.
        """
        self._popmap_inverse = genotype_data.popmap_inverse
        self._samples = genotype_data.samples

        # self._alignment = IUPAC-coded 2D array.
        self._alignment = alignment.copy()

        # Set up logger
        self.logger = logger or logging.getLogger(__name__)

        # Cache structures used by single-level or two-level AMOVA
        self._cached_regions_three_level = None

        iupac_data = IUPAC(logger=self.logger)

        # Get dictionary mapping IUPAC codes to integers.
        # This is used for computing pairwise distances.
        self._iupac_map = iupac_data.get_iupac_int_map()

        # Precompute IUPAC distance matrix
        self._iupac_distance_matrix = np.array(
            [
                [
                    self._iupac_nucleotide_distance(c1, c2)
                    for c2 in self._iupac_map.keys()
                ]
                for c1 in self._iupac_map.keys()
            ]
        )

    def run(
        self,
        regionmap: Dict[str, str] | None = None,
        n_bootstraps: int = 0,
        n_jobs: int = 1,
        random_seed: int | None = None,
    ) -> Dict[str, float]:
        """Conduct AMOVA with 1, 2, or 3 hierarchical levels.

        Args:
            regionmap (dict, optional): Mapping from population_id -> region_id. Must be provided for hierarchical_levels > 1. Defaults to None.
            n_bootstraps (int): Number of bootstrap replicates (across loci). Default: 0 (no bootstrapping).
            n_jobs (int): Number of parallel jobs. -1 uses all cores. Defaults to 1
            random_seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            dict: AMOVA results (variance components, Phi statistics, p-values if bootstrapping).
        """
        # Input validation
        if not isinstance(n_bootstraps, int) or n_bootstraps < 0:
            msg = (
                f"n_bootstraps must be a non-negative integer, but got: {n_bootstraps}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info(f"Starting AMOVA...")
        results = self._amova_three_level(regionmap, self._alignment)

        # Handle bootstrapping
        if n_bootstraps > 0:
            self.logger.info(
                f"Performing {n_bootstraps} bootstrap replicates with {n_jobs} parallel jobs..."
            )
            p_values = self._amova_bootstrap(
                results,
                n_bootstraps,
                n_jobs,
                random_seed,
                regionmap,
            )
            results.update({f"{k}_p_value": v for k, v in p_values.items()})

        self.logger.info("AMOVA completed!")

        return results

    def _amova_three_level(
        self, regionmap: Dict[str, str], alignment: np.ndarray
    ) -> Dict[str, float]:
        """Perform a three-level hierarchical AMOVA.

        This method calculates variance components and Phi statistics for a three-level AMOVA model: region -> population -> individuals. It computes among-region, among-population-within-region, and within-population variance components. Returns Phi_RT, Phi_PR, Phi_PT, and variance components.

        In layman's terms, this method helps in understanding genetic differentiation at three hierarchical levels: regions, populations within regions, and individuals within populations. It calculates variance components that quantify the genetic diversity at each level. The Phi statistics are measures of genetic differentiation between regions, populations, and individuals.

        Args:
            regionmap (dict): pop_id -> region_id. Required for this method. If not provided, raises ValueError. Format: {pop_id: region_id}.
            alignment (np.ndarray): Genotype data in IUPAC format (shape: [n_samples, n_loci]).

        Returns:
            Dict[str, float]: Dictionary of variance components and Phi statistics in float format. Keys: "Among_region_variance", "Among_population_within_region_variance", "Within_population_variance", "Phi_RT", "Phi_PR", "Phi_PT". Values: float. If any variance component is NaN, the corresponding Phi statistic is also NaN.

        Example:
            {
                "Among_region_variance": 0.123,
                "Among_population_within_region_variance": 0.456,
                "Within_population_variance": 0.789,
                "Phi_RT": 0.123,
                "Phi_PR": 0.456,
                "Phi_PT": 0.789
            }

        Notes:
            - This method uses a pairwise distance-based approach to calculate variance components.
            - The Phi statistics are measures of genetic differentiation.
            - Sample-size weighting is applied to variance components.
            - The method caches the region -> pop -> sample indices structure for efficiency.
            - If any variance component is NaN, the corresponding Phi statistic is also NaN.
            - If any variance component is <= 0, the corresponding Phi statistic is NaN.
            - If the total variance is <= 0, all Phi statistics are NaN.
            - If the total variance is NaN, all Phi statistics are NaN.
            - The algorithm is inherently O(N^2) due to pairwise distance calculations.
        """
        # NOTE: Skip major vectorization here because pairwise distance-based
        # AMOVA is inherently O(N^2).
        # However, we can still cache the structure that maps region -> pop ->
        # sample indices.
        if self._cached_regions_three_level is None:
            popmap_inverse = self._popmap_inverse
            sample_ids = self._samples
            sample_id_to_index = {sid: i for i, sid in enumerate(sample_ids)}

            regions: Dict[str, Dict[str, List[int]]] = {}
            for pop_id, samples in popmap_inverse.items():
                reg_id = regionmap[pop_id]
                idxs = [sample_id_to_index[s] for s in samples]
                if reg_id not in regions:
                    regions[reg_id] = {}
                regions[reg_id][pop_id] = idxs

            self._cached_regions_three_level = regions

        regions = self._cached_regions_three_level
        region_ids = list(regions.keys())
        R = len(region_ids)

        # total pops
        P = sum(len(pop_dict) for pop_dict in regions.values())

        # total individuals
        sample_ids = self._samples
        N = len(sample_ids)

        SSR_total = 0.0  # among regions
        SSp_r_total = 0.0  # among populations within regions
        SSW_total = 0.0  # within populations
        df_reg_total = 0
        df_pop_reg_total = 0
        df_within_total = 0

        n_loci = alignment.shape[1]
        for locus_idx in range(n_loci):
            dist_matrix = self._compute_locus_pairwise_distances(locus_idx, alignment)

            if dist_matrix is None or np.all(np.isnan(dist_matrix)):
                continue  # Skip entirely missing loci

            valid_inds = [i for i in range(N) if not np.isnan(dist_matrix[i, i])]
            N_l = len(valid_inds)
            if N_l < 2:
                continue

            # degrees of freedom
            df_reg = R - 1
            df_pop_reg = max(P - R, 0)
            df_within = max(N_l - P, 0)
            if df_reg <= 0 or df_pop_reg <= 0 or df_within <= 0:
                continue

            # global mean distance
            dist_sum_all = 0.0
            valid_pairs = 0
            for i in range(N_l):
                for j in range(i + 1, N_l):
                    d_ij = dist_matrix[valid_inds[i], valid_inds[j]]
                    if not np.isnan(d_ij):
                        dist_sum_all += d_ij
                        valid_pairs += 1
            if valid_pairs < 1:
                continue
            global_mean_dist = dist_sum_all / valid_pairs

            SSR_locus = 0.0
            SSp_r_locus = 0.0
            SSW_locus = 0.0

            # among regions
            region_indices_map = {}
            for reg_id in region_ids:
                region_indices_map[reg_id] = []
                for pop_id, idx_list in regions[reg_id].items():
                    region_indices_map[reg_id].extend(idx_list)

            region_sizes = {}
            region_avg_dist = {}
            for reg_id in region_ids:
                members = [x for x in region_indices_map[reg_id] if x in valid_inds]
                region_sizes[reg_id] = len(members)
                if len(members) < 2:
                    region_avg_dist[reg_id] = np.nan
                    continue
                sum_reg = 0.0
                valid_reg_pairs = 0
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        d_ij = dist_matrix[members[i], members[j]]
                        if not np.isnan(d_ij):
                            sum_reg += d_ij
                            valid_reg_pairs += 1
                region_avg_dist[reg_id] = (
                    sum_reg / valid_reg_pairs if valid_reg_pairs > 0 else np.nan
                )

            for reg_id in region_ids:
                n_r = region_sizes[reg_id]
                if n_r < 2 or np.isnan(region_avg_dist[reg_id]):
                    continue
                SSR_locus += n_r * (region_avg_dist[reg_id] - global_mean_dist) ** 2

            # among populations within regions
            for reg_id in region_ids:
                reg_mean = region_avg_dist[reg_id]
                if np.isnan(reg_mean):
                    continue
                for pop_id, idx_list in regions[reg_id].items():
                    pop_members = [x for x in idx_list if x in valid_inds]
                    if len(pop_members) < 2:
                        continue
                    sum_pop = 0.0
                    valid_pop_pairs = 0
                    for i in range(len(pop_members)):
                        for j in range(i + 1, len(pop_members)):
                            d_ij = dist_matrix[pop_members[i], pop_members[j]]
                            if not np.isnan(d_ij):
                                sum_pop += d_ij
                                valid_pop_pairs += 1
                    if valid_pop_pairs > 0:
                        pop_mean = sum_pop / valid_pop_pairs
                        SSp_r_locus += len(pop_members) * (pop_mean - reg_mean) ** 2

            # within populations
            for reg_id in region_ids:
                for pop_id, idx_list in regions[reg_id].items():
                    pop_members = [x for x in idx_list if x in valid_inds]
                    if len(pop_members) < 2:
                        continue
                    sum_pop = 0.0
                    valid_pop_pairs = 0
                    for i in range(len(pop_members)):
                        for j in range(i + 1, len(pop_members)):
                            d_ij = dist_matrix[pop_members[i], pop_members[j]]
                            if not np.isnan(d_ij):
                                sum_pop += d_ij
                                valid_pop_pairs += 1
                    if valid_pop_pairs == 0:
                        continue

                    pop_mean_dist = sum_pop / valid_pop_pairs
                    d_ij_values = dist_matrix[np.ix_(pop_members, pop_members)]
                    valid_values = d_ij_values[~np.isnan(d_ij_values)]
                    pop_mean_dist = np.mean(valid_values)
                    sum_sq_dev = np.sum((valid_values - pop_mean_dist) ** 2)
                    SSW_locus += sum_sq_dev

            SSR_total += SSR_locus
            SSp_r_total += SSp_r_locus
            SSW_total += SSW_locus
            df_reg_total += df_reg
            df_pop_reg_total += df_pop_reg
            df_within_total += df_within

        # finalize
        if df_reg_total <= 0 or df_pop_reg_total <= 0 or df_within_total <= 0:
            return {
                "Among_region_variance": np.nan,
                "Among_population_within_region_variance": np.nan,
                "Within_population_variance": np.nan,
                "Phi_RT": np.nan,
                "Phi_PR": np.nan,
                "Phi_PT": np.nan,
            }

        MSR = SSR_total / df_reg_total
        MSP_r = SSp_r_total / df_pop_reg_total
        MSW = SSW_total / df_within_total

        # Sample-size weighting (Excoffier approach)
        n_c_reg, n_c_pop, n_c_within = self._excoffier_sample_size_weighting(regions)
        if n_c_reg <= 0 or n_c_pop <= 0 or n_c_within <= 0:
            return {
                "Among_region_variance": np.nan,
                "Among_population_within_region_variance": np.nan,
                "Within_population_variance": np.nan,
                "Phi_RT": np.nan,
                "Phi_PR": np.nan,
                "Phi_PT": np.nan,
            }

        sigma_region = (MSR - MSP_r) / n_c_reg
        sigma_region = max(sigma_region, 0)

        sigma_pop_region = (MSP_r - MSW) / n_c_pop
        sigma_pop_region = max(sigma_pop_region, 0)

        sigma_within = MSW / n_c_within
        sigma_within = max(sigma_within, 0)

        if any(np.isnan([sigma_region, sigma_pop_region, sigma_within])):
            return {
                "Among_region_variance": np.nan,
                "Among_population_within_region_variance": np.nan,
                "Within_population_variance": np.nan,
                "Phi_RT": np.nan,
                "Phi_PR": np.nan,
                "Phi_PT": np.nan,
            }

        sigma_total = sigma_region + sigma_pop_region + sigma_within
        if sigma_total <= 0:
            return {
                "Among_region_variance": float(sigma_region),
                "Among_population_within_region_variance": float(sigma_pop_region),
                "Within_population_variance": float(sigma_within),
                "Phi_RT": np.nan,
                "Phi_PR": np.nan,
                "Phi_PT": np.nan,
            }

        phi_rt = sigma_region / sigma_total
        phi_pr = (
            sigma_pop_region / (sigma_pop_region + sigma_within)
            if (sigma_pop_region + sigma_within) > 0
            else np.nan
        )
        phi_pt = (sigma_region + sigma_pop_region) / sigma_total

        return {
            "Among_region_variance": float(sigma_region),
            "Among_population_within_region_variance": float(sigma_pop_region),
            "Within_population_variance": float(sigma_within),
            "Phi_RT": float(phi_rt),
            "Phi_PR": float(phi_pr),
            "Phi_PT": float(phi_pt),
        }

    def _compute_locus_pairwise_distances(self, locus_idx, alignment):
        """Compute pairwise distances for one locus using IUPAC-coded alignment.

        Args:
            locus_idx (int): Column index in self._alignment (locus).
            alignment (np.ndarray): Genotype data in IUPAC format (shape: [n_samples, n_loci]).

        Returns:
            np.ndarray or None: A square pairwise distance matrix or None if all missing.

        Notes:
            - The IUPAC code distance matrix is computed using the IUPAC nucleotide distance metric.
            - The distance metric is based on the number of nucleotide differences between two IUPAC codes.
            - The distance matrix is a squareform matrix.
        """
        locus_bases = alignment[:, locus_idx]
        if np.all(np.isin(locus_bases, {"N", "-", "?", "."})):
            return None

        processed_bases = np.array(
            [self._iupac_map.get(base, 16) for base in locus_bases]
        )

        def fast_iupac_distance(x, y):
            return self._iupac_distance_matrix[int(x), int(y)]

        pairwise_distances = pdist(processed_bases[:, None], metric=fast_iupac_distance)

        return squareform(pairwise_distances)

    def _iupac_nucleotide_distance(self, code1: str, code2: str) -> float:
        """Compute distance between two IUPAC codes.

        Args:
            code1 (str): First IUPAC code.
            code2 (str): Second IUPAC code.

        Returns:
            float: Distance between the two IUPAC codes.

        Notes:
            - The IUPAC nucleotide distance metric is based on the number of nucleotide differences between two IUPAC codes.
            - The distance is 0 if the codes overlap, else 1.
        """
        iupac_dict = {
            "A": {"A"},
            "C": {"C"},
            "G": {"G"},
            "T": {"T"},
            "R": {"A", "G"},
            "Y": {"C", "T"},
            "S": {"G", "C"},
            "W": {"A", "T"},
            "K": {"G", "T"},
            "M": {"A", "C"},
            "B": {"C", "G", "T"},
            "D": {"A", "G", "T"},
            "H": {"A", "C", "T"},
            "V": {"A", "C", "G"},
            "N": {"A", "C", "G", "T"},
            "-": set(),
            "?": set(),
            ".": set(),
        }
        s1 = iupac_dict.get(code1.upper(), set())
        s2 = iupac_dict.get(code2.upper(), set())
        if not s1 or not s2:
            # Gap or unknown => treat as mismatch
            return 1.0
        # Overlap => distance = 0, else 1
        return 0.0 if (s1 & s2) else 1.0

    def _excoffier_sample_size_weighting(
        self, regions: Dict[str, Dict[str, List[int]]]
    ) -> Tuple[float, float, float]:
        """Compute multi-level sample-size weighting factors for 3-level AMOVA.

        This method calculates the weighting factors needed for Analysis of Molecular Variance (AMOVA) at three hierarchical levels: regions, populations, and within populations. These factors adjust the variance components and are essential for computing Phi statistics and p-values in AMOVA.

        In layman's terms, this method helps in adjusting the calculations for genetic diversity analysis by considering the sizes of different groups (regions and populations). It ensures that the calculations are fair and accurate, even if some groups have more samples than others.

        Args:
            regions (dict): region_id -> {pop_id -> list_of_sample_indices}.

        Returns:
            Tuple[float, float, float]: Weighting factors for regions, populations, and within populations.

        Notes:
            - The sample-size weighting factors are calculated according to Excoffier et al. (1992).
            - The factors are typically set to 1.0 for distance-based calculations.
            - The factors are capped at 1e-9 to avoid division by zero.
            - The factors are used to adjust variance components in AMOVA calculations.
            - The factors are used to compute Phi statistics in AMOVA.
            - The factors are used to compute p-values in AMOVA bootstrapping.

        """
        region_sizes = {}
        pop_sizes = []
        N_total = 0
        for reg_id, pop_dict in regions.items():
            region_sum = 0
            for _, idx_list in pop_dict.items():
                n_pop = len(idx_list)
                pop_sizes.append(n_pop)
                region_sum += n_pop
            region_sizes[reg_id] = region_sum
            N_total += region_sum

        R = len(region_sizes)
        P = len(pop_sizes)

        if R > 1:
            sum_sq_r = sum(sz * sz for sz in region_sizes.values())
            n_c_reg = (N_total - sum_sq_r / N_total) / (R - 1) if N_total > 0 else 1.0
        else:
            n_c_reg = 1.0

        if P > R:
            sum_sq_p = sum(sz * sz for sz in pop_sizes)
            n_c_pop = (N_total - sum_sq_p / N_total) / (P - R) if N_total > 0 else 1.0
        else:
            n_c_pop = 1.0

        # Typically set to 1.0 for distance-based calculations
        n_c_within = 1.0
        return max(n_c_reg, 1e-9), max(n_c_pop, 1e-9), max(n_c_within, 1e-9)

    @staticmethod
    def _single_bootstrap_run(
        n_loci: int,
        alignment: np.ndarray,
        regionmap: Dict[str, str],
        phi_keys: List[str],
        func: Callable[[Dict[str, str], np.ndarray], Dict[str, float]],
        boot_seed: int,
    ) -> Dict[str, float]:
        """Run a single bootstrap replicate.

        Args:
            n_loci (int): Number of loci in the alignment.
            alignment (np.ndarray): Genotype data in IUPAC format (shape: [n_samples, n_loci]).
            regionmap (dict): pop_id -> region_id mapping.
            phi_keys (list): List of Phi statistic keys.
            func (callable): AMOVA function to run.
            boot_seed (int): Random seed for bootstrapping.

        Returns:
            Dict[str, float]: Dictionary of Phi statistics for the bootstrap replicate.
        """
        # Resample loci (columns) with replacement
        local_rng = np.random.default_rng(boot_seed)
        idx = local_rng.choice(np.arange(n_loci), size=n_loci, replace=True)
        aln = alignment[:, idx]

        # Run AMOVA
        res = func(regionmap, aln)
        return {k: res.get(k, np.nan) for k in phi_keys}

    def _amova_bootstrap(
        self,
        observed_results: Dict[str, float],
        n_bootstraps: int,
        n_jobs: int,
        random_seed: int | None = None,
        regionmap: Dict[str, str] | None = None,
    ) -> Dict[str, float]:
        """Bootstrap across loci to get p-values for Phi statistics.

        This method performs bootstrapping across SNP loci to estimate p-values for Phi statistics obtained from AMOVA. It resamples loci with replacement and re-calculates AMOVA for each replicate. The p-value is calculated as the fraction of replicates where the Phi statistic is greater than or equal to the observed value.

        Args:
            observed_results (dict): Observed results from AMOVA.
            n_bootstraps (int): Number of bootstrap replicates.
            n_jobs (int): Number of parallel jobs.
            random_seed (int | None): Random seed for reproducibility.
            regionmap (Dict[str, str] | None): population_id -> region_id mapping.

        Returns:
            Dict[str, float]: Dictionary of p-values for Phi statistics. Keys: "Phi_XXX_p_value". Values: float.

        Notes:
            - The p-value is calculated as the fraction of replicates where the Phi statistic is greater than or equal to the observed value.
            - The method uses a frequency-based approach for single-level AMOVA.
            - The method uses a pairwise distance-based approach for two-level and three-level AMOVA.
            - The method caches the region -> pop -> sample indices structure for efficiency.
            - If the observed value is NaN, the p-value is NaN.
            - If the observed value is not available in the replicate, the p-value is NaN.
        """
        phi_keys = [k for k in observed_results if k.startswith("Phi_")]
        n_loci = self._alignment.shape[1]

        rng = np.random.default_rng(random_seed)
        seeds = rng.choice(range(int(1e9)), size=n_bootstraps, replace=False)

        n_jobs_resolved = n_jobs if n_jobs != -1 else mp.cpu_count()
        if n_jobs_resolved <= 0:
            msg = f"n_jobs must be a positive integer, but got: {n_jobs_resolved}."
            self.logger.error(msg)
            raise ValueError(msg)

        original_alignment = self._alignment.copy()
        partial_single_boot = partial(
            AMOVA._single_bootstrap_run,
            n_loci,
            original_alignment,
            regionmap,
            phi_keys,
            self._amova_three_level,
        )

        with ProcessPoolExecutor(max_workers=n_jobs_resolved) as executor:
            replicate_results = list(executor.map(partial_single_boot, seeds))

        aggregated = {k: [] for k in phi_keys}
        for rep_dict in replicate_results:
            for k in phi_keys:
                val = rep_dict.get(k, np.nan)
                if not np.isnan(val):
                    aggregated[k].append(val)

        # Compute p-values
        p_values = {}
        for k in phi_keys:
            obs_val = observed_results.get(k, np.nan)
            if np.isnan(obs_val):
                p_values[k] = 1.0
                continue

            arr = np.array(aggregated[k], dtype=float)
            arr = arr[~np.isnan(arr)]  # Remove NaNs

            # If no valid bootstrap replicates, assume p-value is high
            if len(arr) == 0:
                p_values[k] = 1.0
                continue

            more_extreme = np.sum(arr >= obs_val)
            p_values[k] = (more_extreme + 1) / (len(arr) + 1)

            # Ensure p-values are between 0 and 1
            p_values[k] = min(max(p_values[k], 0.0001), 0.9999)

        return {k: float(v) for k, v in p_values.items()}
