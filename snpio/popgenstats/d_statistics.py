import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

# Mapping of IUPAC codes to sets of alleles
iupac_to_set = {
    "A": {"A"},
    "T": {"T"},
    "G": {"G"},
    "C": {"C"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "N": set(),  # N for missing data
}


def bootstrap_iteration(
    alignment,
    d1_inds,
    d2_inds,
    d3_inds,
    d4_inds,
    outgroup_inds,
    sample_ids,
    d_statistic_func,
    include_heterozygous,
) -> List[Tuple[Tuple[str, str, str, str | None, str], float | None]]:
    """Standalone function for a single bootstrap iteration to enable tracking sample IDs with D-statistics."""

    snp_indices = np.random.choice(
        alignment.shape[1], size=alignment.shape[1], replace=True
    )

    if d4_inds:
        individual_results = d_statistic_func(
            alignment,
            d1_inds,
            d2_inds,
            d3_inds,
            d4_inds,
            outgroup_inds,
            sample_ids,
            include_heterozygous,
            snp_indices,
        )
    else:
        individual_results = d_statistic_func(
            alignment,
            d1_inds,
            d2_inds,
            d3_inds,
            outgroup_inds,
            sample_ids,
            include_heterozygous,
            snp_indices,
        )

    return individual_results


class DStatistics:
    """Class to calculate D-statistics (Patterson, Partitioned, and DFOIL) from SNP data.

    This class provides methods to calculate D-statistics using the ABBA/BABA method from SNP data. The D-statistic is a measure of the degree of allele sharing between populations and is used to infer the presence of gene flow between populations.

    Attributes:
        alignment (np.ndarray): Array of SNP data with rows representing individuals and columns representing
            SNP sites.

    Example:
        >>> import numpy as np
        >>> alignment = np.array(
        >>> [["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
        >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
        >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
        >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
        >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"]]
        >>> d1_inds = [0, 1, 2]
        >>> d2_inds = [3, 4, 5]
        >>> d3_inds = [6, 7, 8]
        >>> outgroup_inds = [9, 10, 11]
        >>> d_statistic = DStatistic(alignment)
        >>> observed_d_statistic, z_score, p_value = d_statistic.calculate_z_and_p_values(
        >>>     d1_inds, d2_inds, d3_inds, outgroup_inds, num_bootstraps=1000
        >>> )
        ...
        >>> # Expected output: (1.0, 0.0, 1.0)
    """

    def __init__(
        self, alignment: np.ndarray, sample_ids: List[int], logger: logging.Logger
    ) -> None:
        """Initialize DStatisticCalculator with SNP alignment data.

        This class provides methods to calculate D-statistics using the ABBA/BABA method from SNP data. The D-statistic is a measure of the degree of allele sharing between populations and is used to infer the presence of gene flow between populations.

        Args:
            alignment (np.ndarray): Array of SNP data with rows representing individuals and columns representing SNP sites.
            logger (logging.Logger): Logger object for logging messages.
        """
        self.alignment = alignment
        self.sample_ids = sample_ids
        self.logger = logger

    @staticmethod
    def _consensus_genotype(bases: List[str], include_heterozygous: bool) -> set[str]:
        """Get consensus genotype with optional heterozygote inclusion.

        Args:
            bases (list[str]): List of bases to consider.
            include_heterozygous (bool): Whether to include heterozygotes.

        Returns:
            set[str]: Set of consensus genotypes.
        """
        allele_counts = {}
        for base in bases:
            alleles = iupac_to_set.get(base, set())
            for allele in alleles:
                allele_counts[allele] = allele_counts.get(allele, 0) + 1

        # If including heterozygotes, treat both alleles equally
        if include_heterozygous:
            sorted_alleles = sorted(allele_counts.items(), key=lambda x: -x[1])
            if len(sorted_alleles) > 1 and sorted_alleles[0][1] == sorted_alleles[1][1]:
                return set(
                    [sorted_alleles[0][0], sorted_alleles[1][0]]
                )  # Return both as heterozygous
            return {sorted_alleles[0][0]}
        else:
            # Only consider the most frequent single allele
            if allele_counts:
                return {max(allele_counts, key=allele_counts.get)}
            return set()

    @staticmethod
    def bootstrap_d_statistic(
        alignment,
        d1_inds,
        d2_inds,
        d3_inds,
        d4_inds,
        outgroup_inds,
        sample_ids,
        include_heterozygous=False,
        num_bootstraps=1000,
        method="patterson",
        n_jobs=-1,
    ) -> List[Tuple[Tuple[str, str, str, str | None, str], List[float | None]]]:
        """Bootstrap specified D-statistic method over SNP sites, tracking sample combinations."""

        method_map = {
            "patterson": DStatistics.patterson_d_statistic,
            "partitioned": DStatistics.partitioned_d_statistic,
            "dfoil": DStatistics.dfoil_statistic,
        }

        if method not in method_map:
            raise KeyError(f"Invalid method provided: {method}")

        d_statistic_func = method_map[method]

        # Configure parallel processing
        parallel = n_jobs != 1
        if parallel and n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        if parallel and n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()

        # Perform bootstrapping with progress tracking
        if parallel:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Use tqdm to wrap the executor map for progress tracking
                bootstrapped_results = list(
                    tqdm(
                        executor.map(
                            lambda _: bootstrap_iteration(
                                alignment,
                                d1_inds,
                                d2_inds,
                                d3_inds,
                                d4_inds,
                                outgroup_inds,
                                sample_ids,
                                d_statistic_func,
                                include_heterozygous,
                            ),
                            range(num_bootstraps),
                        ),
                        total=num_bootstraps,
                        desc="Bootstrapping (parallel)",
                        unit="replicate",
                    )
                )
        else:
            bootstrapped_results = [
                bootstrap_iteration(
                    alignment,
                    d1_inds,
                    d2_inds,
                    d3_inds,
                    d4_inds,
                    outgroup_inds,
                    sample_ids,
                    d_statistic_func,
                    include_heterozygous,
                )
                for _ in tqdm(
                    range(num_bootstraps), desc="Bootstrapping", unit="replicate"
                )
            ]

        # Flatten results from all replicates and organize by sample ID
        # combinations
        combo_results = {}
        for replicate_results in bootstrapped_results:
            for sample_combo, d_stat in replicate_results:
                if sample_combo not in combo_results:
                    combo_results[sample_combo] = []
                combo_results[sample_combo].append(d_stat)

        # Return individual results with mean D-statistics for each sample combo
        return combo_results

    def calculate_z_and_p_values(
        self,
        d1_inds,
        d2_inds,
        d3_inds,
        outgroup_inds,
        d4_inds=None,
        include_heterozygous=False,
        num_bootstraps=1000,
        method="patterson",
        n_jobs=-1,
    ) -> Tuple[
        List[
            Tuple[
                Tuple[str, str, str, str | None, str],
                float | None,
                float | None,
                float | None,
            ]
        ],
        float | None,
        float | None,
    ]:
        """Calculate Z-scores and P-values from D-statistic using bootstrapping with sample tracking

        Args:
            d1_inds (List[int]): Indices of individuals in population 1.
            d2_inds (List[int]): Indices of individuals in population 2.
            d3_inds (List[int]): Indices of individuals in population 3.
            outgroup_inds (List[int]): Indices of individuals in the outgroup.
            d4_inds (List[int], optional): Indices of individuals in population 4. Defaults to None.
            include_heterozygous (bool): Whether to include heterozygous individuals.
            num_bootstraps (int): Number of bootstrap iterations.
            method (str): Method to use for D-statistic calculation. Options are "patterson", "partitioned", or "dfoil".
            n_jobs (int): Number of parallel jobs to run. Defaults to -1 (use all available cores).
            debug (bool): Whether to enable debug mode. Defaults to False.
        Returns:
            Tuple[
                List[
                    Tuple[
                        Tuple[str, str, str, str | None, str],
                        float | None,
                        float | None,
                        float | None,
                    ]
                ],
                float | None,
                float | None,
            ]: Tuple containing:
                - List of tuples with sample combinations, mean D-statistics, Z-scores, and P-values.
                - Overall Z-score.
                - Overall P-value.
        """
        method = method.lower()
        if method not in {"patterson", "partitioned", "dfoil"}:
            self.logger.error(f"Invalid method specified: {method}")
            raise ValueError(f"Invalid method specified: {method}")

        args = (
            self.alignment,
            d1_inds,
            d2_inds,
            d3_inds,
            d4_inds,
            outgroup_inds,
            self.sample_ids,
        )
        kwargs = {
            "include_heterozygous": include_heterozygous,
            "num_bootstraps": num_bootstraps,
            "method": method,
            "n_jobs": n_jobs,
        }

        # Get the bootstrapped D-statistics per sample combination
        combo_results = DStatistics.bootstrap_d_statistic(*args, **kwargs)

        combo_z_p_values = []
        overall_bootstrapped_values = []

        for sample_combo, bootstrapped_d_stats in combo_results.items():
            # Remove None values (cases where D-statistic couldn't be
            # calculated)
            bootstrapped_d_stats = [d for d in bootstrapped_d_stats if d is not None]

            if not bootstrapped_d_stats:
                self.logger.warning(
                    f"No valid bootstrapped D-statistics for sample combination: {sample_combo}"
                )
                continue

            # Calculate mean D-statistic
            mean_d_stat = np.mean(bootstrapped_d_stats)

            with np.errstate(divide="ignore", invalid="ignore"):
                # Calculate standard deviation of bootstrapped D-statistics
                std_d_stat = np.std(bootstrapped_d_stats)

            # Calculate Z-score and P-value
            if std_d_stat == 0:
                z_score = 0.0
                p_value = 1.0
                self.logger.warning(
                    f"Zero variance in bootstrapped D-statistics for {sample_combo}, skipping Z/P calculation."
                )
            else:
                z_score = mean_d_stat / std_d_stat
                p_value = 2 * (1 - norm.cdf(abs(z_score)))

            combo_z_p_values.append((sample_combo, mean_d_stat, z_score, p_value))
            overall_bootstrapped_values.extend(bootstrapped_d_stats)

        # Calculate overall Z-score and P-value
        overall_bootstrapped_values = [
            d for d in overall_bootstrapped_values if d is not None
        ]
        mean_overall = np.mean(overall_bootstrapped_values)
        std_overall = np.std(overall_bootstrapped_values)

        if std_overall == 0:
            overall_z_score = 0.0
            overall_p_value = 1.0
            self.logger.warning(
                "Zero variance in overall bootstrapped D-statistics, skipping overall Z/P calculation."
            )
        else:
            overall_z_score = mean_overall / std_overall
            overall_p_value = 2 * (1 - norm.cdf(abs(overall_z_score)))

        return combo_z_p_values, overall_z_score, overall_p_value

    @staticmethod
    def patterson_d_statistic(
        alignment: np.ndarray,
        d1_inds: List[int],
        d2_inds: List[int],
        d3_inds: List[int],
        outgroup_inds: List[int],
        sample_ids: List[str],
        include_heterozygous: bool = False,
        snp_indices: np.ndarray | List[int] | None = None,
    ) -> float | None:
        """Calculate Patterson's D-statistic for all individual combinations across populations.

        Args:
            alignment (np.ndarray): SNP data with individuals as rows and SNP sites as columns.
            d1_inds (List[int]): Indices of individuals in population 1.
            d2_inds (List[int]): Indices of individuals in population 2.
            d3_inds (List[int]): Indices of individuals in population 3.
            outgroup_inds (List[int]): Indices of individuals in the outgroup.
            sample_ids (List[str]): List of sample IDs.
            include_heterozygous (bool): Whether to include heterozygous individuals.
            snp_indices (np.ndarray | List[int] | None): Indices of SNP sites to consider.

        Returns:
            float | None: List of tuples with sample combinations and D-statistics.
                If no valid patterns are found, returns None.
        """

        if snp_indices is None:
            snp_indices = np.arange(alignment.shape[1])

        individual_results = []

        # Pre-compute consensus for each individual and SNP site
        consensus_matrix = np.empty(
            (alignment.shape[0], alignment.shape[1]), dtype=object
        )
        for ind in range(alignment.shape[0]):
            consensus_matrix[ind] = [
                DStatistics._consensus_genotype([base], include_heterozygous)
                for base in alignment[ind, snp_indices]
            ]

        # Iterate over all individual combinations using product
        for d1_ind, d2_ind, d3_ind, outgroup_ind in product(
            d1_inds, d2_inds, d3_inds, outgroup_inds
        ):

            sample_id_combo = (
                sample_ids[d1_ind],
                sample_ids[d2_ind],
                sample_ids[d3_ind],
                sample_ids[outgroup_ind],
            )

            # Access precomputed consensus data for each SNP
            d1_consensus = consensus_matrix[d1_ind]
            d2_consensus = consensus_matrix[d2_ind]
            d3_consensus = consensus_matrix[d3_ind]
            outgroup_consensus = consensus_matrix[outgroup_ind]

            # Define valid data mask and patterns
            valid_mask = (
                (d1_consensus != set())
                & (d2_consensus != set())
                & (d3_consensus != set())
                & (outgroup_consensus != set())
            )
            abba_mask = (
                (d1_consensus == d2_consensus)
                & (d3_consensus != outgroup_consensus)
                & valid_mask
            )
            baba_mask = (
                (d1_consensus == d3_consensus)
                & (d2_consensus != outgroup_consensus)
                & valid_mask
            )

            abba_count = np.count_nonzero(abba_mask)
            baba_count = np.count_nonzero(baba_mask)

            total_count = abba_count + baba_count

            # Handle edge case with no valid patterns
            if total_count == 0:
                # No valid patterns found
                individual_results.append((sample_id_combo, None))
            else:
                # Calculate final D-statistic
                d_statistic = (abba_count - baba_count) / total_count
                individual_results.append((sample_id_combo, d_statistic))

        return individual_results

    @staticmethod
    def partitioned_d_statistic(
        alignment: np.ndarray,
        d1_inds: List[int],
        d2_inds: List[int],
        d3_inds: List[int],
        d4_inds: List[int],
        outgroup_inds: List[int],
        sample_ids: List[str],
        include_heterozygous: bool = False,
        snp_indices: List[int] = None,
    ) -> float | None:
        """Calculate the partitioned D-statistic for all individual combinations across populations.

        Args:
            alignment (np.ndarray): SNP data with individuals as rows and SNP sites as columns.
            d1_inds (List[int]): Indices of individuals in population 1.
            d2_inds (List[int]): Indices of individuals in population
                2.
            d3_inds (List[int]): Indices of individuals in population 3.
            d4_inds (List[int]): Indices of individuals in population 4.
            outgroup_inds (List[int]): Indices of individuals in the outgroup.
            sample_ids (List[str]): List of sample IDs.
            include_heterozygous (bool): Whether to include heterozygous individuals.
            snp_indices (List[int] | None): Indices of SNP sites to consider. If None, all SNPs are used.

        Returns:
            float | None: List of tuples with sample combinations and partitioned D-statistics.
                If no valid patterns are found, returns None.
        Note:
            The partitioned D-statistic is calculated using the formula:
                .. math:: D = (ABCD - DCBA) / (ABCD + DCBA)
            where ABCD and DCBA are the counts of the respective patterns.
            This method is used to detect gene flow between populations.
            The function returns a list of tuples, each containing the sample combination and the corresponding partitioned D-statistic.
            If no valid patterns are found for a sample combination, the D-statistic is set to None.
        Example:
            >>> import numpy as np
            >>> alignment = np.array(
            >>> [["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
            >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
            >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
            >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"],
            >>> ["A", "A", "A", "T", "T", "T", "C", "C", "C", "G", "G", "G"]]
            >>> d1_inds = [0, 1, 2]
            >>> d2_inds = [3, 4, 5]
            >>> d3_inds = [6, 7, 8]
            >>> d4_inds = [9, 10, 11]
            >>> outgroup_inds = [12, 13, 14]
            >>> sample_ids = ["pop1", "pop2", "pop3", "pop4", "outgroup"]
            >>> d_statistic = DStatistics(alignment, sample_ids)
            >>> observed_partitioned_d_statistic = d_statistic.partitioned_d_statistic(
            >>>     d1_inds, d2_inds, d3_inds, d4_inds, outgroup_inds
            >>> )
            >>> print(observed_partitioned_d_statistic)
            >>> # Expected output: [(("pop1", "pop2", "pop3", "pop4", "outgroup"), 0.0)]
        """

        if snp_indices is None:
            snp_indices = np.arange(alignment.shape[1])

        individual_results = []

        # Pre-compute consensus for each individual and SNP site
        consensus_matrix = np.empty(
            (alignment.shape[0], alignment.shape[1]), dtype=object
        )
        for ind in range(alignment.shape[0]):
            consensus_matrix[ind] = [
                DStatistics._consensus_genotype([base], include_heterozygous)
                for base in alignment[ind, snp_indices]
            ]

        for d1_ind, d2_ind, d3_ind, d4_ind, outgroup_ind in product(
            d1_inds, d2_inds, d3_inds, d4_inds, outgroup_inds
        ):

            sample_id_combo = (
                sample_ids[d1_ind],
                sample_ids[d2_ind],
                sample_ids[d3_ind],
                sample_ids[d4_ind],
                sample_ids[outgroup_ind],
            )

            # Access precomputed consensus data for each SNP
            d1_consensus = consensus_matrix[d1_ind]
            d2_consensus = consensus_matrix[d2_ind]
            d3_consensus = consensus_matrix[d3_ind]
            d4_consensus = consensus_matrix[d4_ind]
            outgroup_consensus = consensus_matrix[outgroup_ind]

            valid_mask = (
                (d1_consensus != set())
                & (d2_consensus != set())
                & (d3_consensus != set())
                & (d4_consensus != set())
                & (outgroup_consensus != set())
            )
            abcd_mask = (
                (d1_consensus == d2_consensus)
                & (d3_consensus == d4_consensus)
                & (d3_consensus != outgroup_consensus)
                & valid_mask
            )
            dcba_mask = (
                (d1_consensus == d4_consensus)
                & (d2_consensus == d3_consensus)
                & (d2_consensus != outgroup_consensus)
                & valid_mask
            )

            abcd_count = np.count_nonzero(abcd_mask)
            dcba_count = np.count_nonzero(dcba_mask)

            total_count = abcd_count + dcba_count

            if total_count == 0:
                # No valid patterns found
                individual_results.append((sample_id_combo, None))
            else:
                partitioned_d_statistic = (abcd_count - dcba_count) / total_count
                individual_results.append((sample_id_combo, partitioned_d_statistic))

            return individual_results

    @staticmethod
    def dfoil_statistic(
        alignment: np.ndarray,
        d1_inds: List[int],
        d2_inds: List[int],
        d3_inds: List[int],
        d4_inds: List[int],
        outgroup_inds: List[int],
        sample_ids: List[str],
        include_heterozygous: bool = False,
        snp_indices: np.ndarray | List[int] | None = None,
    ) -> List[float | None]:
        """Calculate the DFOIL statistic for all individual combinations across populations.

        This method calculates the DFOIL statistic for all individual combinations across populations. The DFOIL statistic is a measure of the degree of allele sharing between populations and is used to infer the presence of gene flow between populations.

        Args:
            alignment (np.ndarray): SNP data with individuals as rows and SNP sites as columns.
            d1_inds (List[int]): Indices of individuals in population 1.
            d2_inds (List[int]): Indices of individuals in population 2.
            d3_inds (List[int]): Indices of individuals in population 3.
            d4_inds (List[int]): Indices of individuals in population 4.
            outgroup_inds (List[int]): Indices of individuals in the outgroup.
            sample_ids (List[str]): List of sample IDs.
            include_heterozygous (bool): Whether to include heterozygous individuals.
            snp_indices (np.ndarray | List[int] | None): Indices of SNP sites to consider. If None, all SNPs are used.

        Returns:
            List[float | None]: List of tuples with sample combinations and DFOIL statistics.
                If no valid patterns are found, returns None.

        Note:
            This function calculates the DFOIL statistic for all individual combinations across populations.
            The DFOIL statistic is a measure of the degree of allele sharing between populations and is used to infer the presence of gene flow between populations.
            The function returns a list of tuples, each containing the sample combination and the corresponding DFOIL statistic.
            If no valid patterns are found for a sample combination, the DFOIL statistic is set to None.
            The DFOIL statistic is calculated using the formula:

                .. math:: DFOIL = (ABBA - BABA + FABA - FOAB) / (ABBA + BABA + FABA + FOAB)
            where ABBA, BABA, FABA, and FOAB are the counts of the respective patterns.
            This method is used to detect gene flow between populations.
        Example:
            >>> DStatistics.dfoil_statistic(
            >>>     alignment,
            >>>     d1_inds,
            >>>     d2_inds,
            >>>     d3_inds,
            >>>     d4_inds,
            >>>     outgroup_inds,
            >>>     sample_ids,
            >>>     include_heterozygous=True,
            >>>     snp_indices=None,
            >>> )
            >>> # Expected output: [(("pop1", "pop2", "pop3", "pop4", "outgroup"), 0.0)]
            >>> )
        """

        if snp_indices is None:
            snp_indices = np.arange(alignment.shape[1])

        individual_results = []

        # Pre-compute consensus for each individual and SNP site
        consensus_matrix = np.empty(
            (alignment.shape[0], alignment.shape[1]), dtype=object
        )
        for ind in range(alignment.shape[0]):
            consensus_matrix[ind] = [
                DStatistics._consensus_genotype([base], include_heterozygous)
                for base in alignment[ind, snp_indices]
            ]

        # Iterate over all combinations of individuals across populations
        for d1_ind, d2_ind, d3_ind, d4_ind, outgroup_ind in product(
            d1_inds, d2_inds, d3_inds, d4_inds, outgroup_inds
        ):
            sample_id_combo = (
                sample_ids[d1_ind],
                sample_ids[d2_ind],
                sample_ids[d3_ind],
                sample_ids[d4_ind],
                sample_ids[outgroup_ind],
            )

            # Access precomputed consensus data for each SNP
            d1_consensus = consensus_matrix[d1_ind]
            d2_consensus = consensus_matrix[d2_ind]
            d3_consensus = consensus_matrix[d3_ind]
            d4_consensus = consensus_matrix[d4_ind]
            d5_consensus = consensus_matrix[outgroup_ind]

            valid_mask = (
                (d1_consensus != set())
                & (d2_consensus != set())
                & (d3_consensus != set())
                & (d4_consensus != set())
                & (d5_consensus != set())
            )

            abba_mask = (
                (d1_consensus == d2_consensus)
                & (d3_consensus != d4_consensus)
                & (d4_consensus == d5_consensus)
                & valid_mask
            )
            baba_mask = (
                (d1_consensus != d2_consensus)
                & (d3_consensus == d4_consensus)
                & (d2_consensus == d5_consensus)
                & valid_mask
            )
            faba_mask = (
                (d1_consensus != d2_consensus)
                & (d2_consensus == d3_consensus)
                & (d3_consensus != d4_consensus)
                & valid_mask
            )
            foab_mask = (
                (d1_consensus == d3_consensus)
                & (d2_consensus != d4_consensus)
                & (d3_consensus == d5_consensus)
                & valid_mask
            )

            abba_count = np.count_nonzero(abba_mask)
            baba_count = np.count_nonzero(baba_mask)
            faba_count = np.count_nonzero(faba_mask)
            foab_count = np.count_nonzero(foab_mask)

            total_count = abba_count + baba_count + faba_count + foab_count

            if total_count == 0:
                # No valid patterns found
                individual_results.append((sample_id_combo, None))
            else:
                # Calculate individual DFOIL statistic for this combination
                dfoil_statistic = (
                    abba_count - baba_count + faba_count - foab_count
                ) / total_count
                individual_results.append((sample_id_combo, dfoil_statistic))

        return individual_results
