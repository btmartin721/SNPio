import random
from typing import List, Optional, Tuple

import numpy as np

from snpio.utils.custom_exceptions import NoValidAllelesError
from snpio.utils.logging import LoggerManager


class BaseGenotypeData:
    def __init__(
        self, filename: Optional[str] = None, filetype: Optional[str] = "auto"
    ):
        self.filename = filename
        self.filetype = filetype.lower()
        self._snp_data = None  # Initialize as None, load on demand
        self._samples = []
        self._populations = []
        self._ref = []
        self._alt = []

        logman = LoggerManager(__name__, verbose=False, debug=False)
        self.logger = logman.get_logger()

    def _load_data(self) -> None:
        """Method to load data from file based on filetype"""
        raise NotImplementedError("Subclasses should implement this method")

    def get_ref_alt_alleles(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Determine the most common, second most common, and less common alleles in each column of a 2D numpy array, excluding 'N', '.', '?', and '-' alleles. The reference allele is determined by frequency and by the fewest number of heterozygous genotypes. If tied, a random allele is selected.

        Args:
            data (np.ndarray): A 2D numpy array where each column represents different SNP data.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
                - Most common alleles (likely ref).
                - Second most common alleles (likely alt).
                - Less common alleles (for potential multi-allelic sites).
        """
        # Initialize arrays to hold results
        most_common_alleles = np.full(data.shape[1], None, dtype=object)
        second_most_common_alleles = np.full(data.shape[1], None, dtype=object)
        less_common_alleles_list = []

        for i in range(data.shape[1]):
            column = data[:, i]

            # Flatten alleles and remove missing and gap values.
            valid_alleles = []
            heterozygous_counts = {}
            for genotype in column:
                if genotype not in ["N", "?", ".", "-"]:
                    # Split heterozygous genotypes (e.g., 'A/G') and add each
                    # allele separately
                    alleles = genotype.split("/")
                    valid_alleles.extend(alleles)
                    # Count heterozygous genotypes for each allele
                    if len(alleles) == 2:
                        for allele in alleles:
                            heterozygous_counts[allele] = (
                                heterozygous_counts.get(allele, 0) + 1
                            )

            # Convert valid_alleles to a numpy array
            valid_alleles = np.array(valid_alleles)

            if valid_alleles.size == 0:
                # If no valid alleles, log an error and raise an exception
                self.logger.error(f"No valid alleles found in column {i}")
                raise NoValidAllelesError(i)

            # Use numpy's unique function with return_counts for counting
            alleles, counts = np.unique(valid_alleles, return_counts=True)

            # Sort by counts (descending order)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_alleles = alleles[sorted_indices]
            sorted_counts = counts[sorted_indices]

            # Warning for low allele counts or borderline cases
            if sorted_counts[0] <= 2:
                self.logger.warning(
                    f"Low allele count in column {i}: {sorted_alleles[0]} occurs only {sorted_counts[0]} times."
                )

            # If the top two alleles have the same count, choose by fewest
            # heterozygous occurrences or randomly
            if len(sorted_alleles) > 1 and sorted_counts[0] == sorted_counts[1]:
                top_alleles = [sorted_alleles[0], sorted_alleles[1]]
                heterozygous_top_counts = [
                    heterozygous_counts.get(allele, 0) for allele in top_alleles
                ]
                if heterozygous_top_counts[0] == heterozygous_top_counts[1]:
                    # Randomly choose the reference allele if heterozygous
                    # counts are tied
                    chosen_ref_index = random.choice([0, 1])
                    most_common_alleles[i] = top_alleles[chosen_ref_index]
                    second_most_common_alleles[i] = top_alleles[1 - chosen_ref_index]
                else:
                    # Choose by fewest heterozygous occurrences
                    chosen_ref_index = np.argmin(heterozygous_top_counts)
                    most_common_alleles[i] = top_alleles[chosen_ref_index]
                    second_most_common_alleles[i] = top_alleles[1 - chosen_ref_index]
            else:
                # Assign most common and second most common alleles
                if len(sorted_alleles) > 0:
                    most_common_alleles[i] = sorted_alleles[0]
                if len(sorted_alleles) > 1:
                    second_most_common_alleles[i] = sorted_alleles[1]

            # Less common alleles are those beyond the second most common
            less_common_alleles = (
                sorted_alleles[2:] if len(sorted_alleles) > 2 else np.array([])
            )
            less_common_alleles_list.append(less_common_alleles)

        return most_common_alleles, second_most_common_alleles, less_common_alleles_list
