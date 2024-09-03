import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from snpio.utils.custom_exceptions import (
    NoValidAllelesError,
    SequenceLengthError,
    AlignmentFormatError,
)
from snpio.utils.logging import setup_logger

logger = setup_logger(__name__)


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

    def _load_data(self):
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
                - Most common alleles (likely ref)
                - Second most common alleles (likely alt)
                - Less common alleles (for potential multi-allelic sites)
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
                logger.error(f"No valid alleles found in column {i}")
                raise NoValidAllelesError(i)

            # Use numpy's unique function with return_counts for counting
            alleles, counts = np.unique(valid_alleles, return_counts=True)

            # Sort by counts (descending order)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_alleles = alleles[sorted_indices]
            sorted_counts = counts[sorted_indices]

            # Warning for low allele counts or borderline cases
            if sorted_counts[0] <= 2:
                logger.warning(
                    f"Low allele count in column {i}: {sorted_alleles[0]} occurs only {sorted_counts[0]} times."
                )

            # If the top two alleles have the same count, choose by fewest heterozygous occurrences or randomly
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

    @property
    def snp_data(self) -> np.ndarray:
        """Get the genotypes as a 2D list of shape (n_samples, n_loci)."""
        is_array = True if isinstance(self._snp_data, np.ndarray) else False

        load = True if is_array and not self._snp_data.size > 0 else False
        load = (
            True
            if not is_array and isinstance(self._snp_data, list) and not self._snp_data
            else False
        )
        load = True if self._snp_data is None else False

        if load:
            self.load_aln()

        if isinstance(self._snp_data, (np.ndarray, pd.DataFrame, list)):
            if isinstance(self._snp_data, list):
                return np.array(self._snp_data)
            elif isinstance(self._snp_data, pd.DataFrame):
                return self._snp_data.to_numpy()
            return self._snp_data  # is numpy.ndarray
        else:
            msg = f"Invalid 'snp_data' type. Expected numpy.ndarray, pandas.DataFrame, or list, but got: {type(self._snp_data)}"
            logger.error(msg)
            raise TypeError(msg)

    @snp_data.setter
    def snp_data(self, value) -> None:
        """Set snp_data. Input can be a 2D list, numpy array, or pandas DataFrame object."""
        if isinstance(value, (np.ndarray, pd.DataFrame, list)):
            if isinstance(value, list):
                value = np.array(value)
            elif isinstance(value, pd.DataFrame):
                value = value.to_numpy()
            self._snp_data = value
            self._validate_seq_lengths()
        else:
            msg = f"Attempt to set 'snp_data' to invalid type. Must be a list, numpy.ndarray, pandas.DataFrame, but got: {type(value)}"
            logger.error(msg)
            raise TypeError(msg)

    def _validate_seq_lengths(self):
        """Ensure that all SNP data rows have the same length."""
        lengths = {len(row) for row in self.snp_data}
        if len(lengths) > 1:
            n_snps = len(self.snp_data[0])
            for i, row in enumerate(self.snp_data):
                if len(row) != n_snps:
                    msg = f"Invalid sequence length for Sample {self.samples[i]}. Expected {len(self.snp_data[0])}, but got: {len(row)}"
                    logger.error(msg)
                    raise SequenceLengthError(self.samples[i])
