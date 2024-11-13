import logging
import random
from typing import List, Optional, Tuple

import numpy as np

from snpio.utils.custom_exceptions import NoValidAllelesError
from snpio.utils.logging import LoggerManager


class BaseGenotypeData:
    def __init__(
        self, filename: Optional[str] = None, filetype: Optional[str] = "auto"
    ):
        self.filename: str | None = filename

        self.filetype: str | None = filetype if filetype is not None else None
        if filetype is not None:
            self.filetype = filetype.lower()

        # Initialize as None, load on demand
        self._snp_data: Optional[List[List[str]]] = None
        self._samples: List[str] = []
        self._populations: List[str | int] = []
        self._ref: List[str] = []
        self._alt: List[List[str] | str] = []

        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=self.debug
        )
        self.logger: Optional[logging.Logger] = logman.get_logger()

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
                - Most common alleles (ref).
                - Second most common alleles (alt).
                - Less common alleles (for potential multi-allelic sites).
        """
        num_cols = data.shape[1]
        most_common_alleles = np.full(num_cols, None, dtype=object)
        second_most_common_alleles = np.full(num_cols, None, dtype=object)
        less_common_alleles_list = []

        for i in range(num_cols):
            column_data = data[:, i]
            valid_mask = ~np.isin(column_data, {"N", ".", "?", "-"})
            valid_genotypes = column_data[valid_mask]

            if valid_genotypes.size == 0:
                # If no valid alleles, log an error and raise an exception
                self.logger.error(f"No valid alleles found in column {i}")
                raise NoValidAllelesError(i)

            # Split genotypes into alleles
            alleles_list = np.char.split(valid_genotypes, sep="/")
            alleles_flat = np.concatenate(alleles_list)

            # Use numpy's unique function with return_counts for counting
            alleles, counts = np.unique(alleles_flat, return_counts=True)

            # Get heterozygous genotypes
            heterozygous_mask = np.char.count(valid_genotypes, "/") == 1
            heterozygous_genotypes = valid_genotypes[heterozygous_mask]

            heterozygous_counts_dict = {}
            if heterozygous_genotypes.size > 0:
                heterozygous_alleles_list = np.char.split(
                    heterozygous_genotypes, sep="/"
                )
                heterozygous_alleles_flat = np.concatenate(heterozygous_alleles_list)
                heterozygous_alleles, heterozygous_counts = np.unique(
                    heterozygous_alleles_flat, return_counts=True
                )
                heterozygous_counts_dict = dict(
                    zip(heterozygous_alleles, heterozygous_counts)
                )

            # Sort by counts (descending order)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_alleles = alleles[sorted_indices]
            sorted_counts = counts[sorted_indices]

            # Warning for low allele counts or borderline cases
            if sorted_counts[0] <= 2:
                self.logger.warning(
                    f"Low allele count in column {i}: {sorted_alleles[0]} occurs only {sorted_counts[0]} times."
                )

            # If the top two alleles have the same count, choose by fewest heterozygous occurrences or randomly
            if len(sorted_alleles) > 1 and sorted_counts[0] == sorted_counts[1]:
                top_alleles = [sorted_alleles[0], sorted_alleles[1]]
                heterozygous_top_counts = [
                    heterozygous_counts_dict.get(allele, 0) for allele in top_alleles
                ]
                if heterozygous_top_counts[0] == heterozygous_top_counts[1]:
                    # Randomly choose the reference allele if heterozygous counts are tied
                    chosen_ref_index = random.choice([0, 1])
                else:
                    # Choose by fewest heterozygous occurrences
                    chosen_ref_index = int(np.argmin(heterozygous_top_counts))
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
