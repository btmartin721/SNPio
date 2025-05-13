import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

# Custom imports
import snpio.utils.custom_exceptions as exceptions
from snpio.utils.custom_exceptions import AlignmentFormatError
from snpio.utils.misc import IUPAC

if TYPE_CHECKING:
    from snpio.filtering.nremover2 import NRemover2


class FilteringMethods:
    def __init__(self, nremover_instance: "NRemover2") -> None:
        """Initialize FilteringMethods class with an instance of NRemover2.

        This class contains methods for filtering loci (columns) and samples (rows) based on various criteria.

        Args:
            nremover_instance (NRemover2): An instance of NRemover2 to access its attributes. The instance should be initialized with the alignment data and other attributes.

        Note:
            - This class uses the NRemover2 instance to access alignment data, genotype data, and other attributes.
            - The logger attribute is used for logging messages.
            - Missing data values are defined using the `missing_vals` attribute.
            - Filtering results are appended to the global list using `_append_global_list`.
            - Minor allele counts are computed using `_calculate_minor_allele_counts`.
            - Minor allele frequencies are computed using `_compute_maf_proportions`.
            - Loci and sample indices are updated using `_update_loci_indices` and `_update_sample_indices`, respectively.

        Attributes:
            nremover (NRemover2): An instance of NRemover2 to access its attributes.
            logger (Logger): A logger object to log messages.
            resource_data (dict): A dictionary containing resource data.
            missing_vals (list): A list of missing data values.
        """
        self.nremover: "NRemover2" = nremover_instance
        self.logger: logging.Logger = self.nremover.logger
        self.resource_data: Dict[str, Any] = self.nremover.genotype_data.resource_data
        self.missing_vals: List[str] = ["N", "-", ".", "?"]
        self.ambiguous_bases: List[str] = ["R", "Y", "S", "W", "K", "M"]
        self.exclude_hets: List[str] = self.missing_vals + self.ambiguous_bases

        if not self.resource_data:
            self.resource_data: Dict[str, Any] = {}

        self.iupac: IUPAC = IUPAC(logger=self.logger)

    def filter_missing(self, threshold: float) -> "NRemover2":
        """Filters out columns (loci) with missing data proportion greater than the given threshold.

        The method calculates the proportion of missing data for each locus and retains only those loci with a proportion of missing data less than or equal to the threshold.

        Args:
            threshold (float): The maximum proportion of missing data allowed for a locus to be retained.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean loci_indices array set.

        Raises:
            TypeError: If the threshold is not a float.
            ValueError: If the threshold is not between 0.0 and 1.0 inclusive.

        Notes:
            - The method uses the `self.nremover` instance to access the alignment data and sample indices.
            - The method also uses the `self.missing_vals` attribute to identify missing data values.
            - The method logs the filtering process and results.
        """

        self.logger.info(
            f"Filtering loci with missing data proportion > {threshold:.2f}"
        )

        self.nremover.propagate_chain()

        if not isinstance(threshold, float):
            msg: str = f"Threshold must be a float value, but got: {type(threshold)}"
            self.logger.error(msg)
            raise TypeError(msg)
        if threshold < 0.0 or threshold > 1.0:
            msg = f"Threshold must be between [0.0, 1.0], but got: {threshold:.2f}"
            self.logger.error(msg)
            raise exceptions.InvalidThresholdError(threshold, msg)

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Try adjusting filtering parameters."
            )
            self.nremover.resolve()

        alignment_array: np.ndarray = (
            self.nremover.alignment[self.nremover.sample_indices, :][
                :, self.nremover.loci_indices
            ]
            .copy()
            .astype(str)
        )

        missing_counts: int = np.count_nonzero(
            np.isin(alignment_array, self.missing_vals), axis=0
        )

        num_samples: int = np.count_nonzero(self.nremover.sample_indices)

        self.logger.debug(f"Total Loci Before {inspect.stack()[0][3]}: {num_samples}")

        missing_props: float = missing_counts / num_samples

        self.logger.debug(f"Total Loci Before {inspect.stack()[0][3]}: {num_samples}")
        self.logger.debug(f"missing_props: {missing_props}")
        self.logger.debug(f"missing_props shape: {missing_props.shape}")

        mask: np.ndarray = missing_props <= threshold

        self.logger.debug(f"mask {inspect.stack()[0][3]}: {mask}")
        self.logger.debug(f"mask shape {inspect.stack()[0][3]}: {mask.shape}")

        full_mask: np.ndarray = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        full_mask[self.nremover.loci_indices] = mask

        # Update loci indices
        n_to_keep: int = np.count_nonzero(full_mask)

        if n_to_keep == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            n_removed: int = np.count_nonzero(~full_mask)
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        t: None | float = None if self.nremover.search_mode else threshold

        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)

        return self.nremover

    def filter_missing_pop(self, threshold: float) -> "NRemover2":
        """Filters loci based on missing data proportion per population.

        This method calculates the proportion of missing data for each locus in each population and retains only those loci that meet the specified threshold across all populations.

        Args:
            threshold (float): The maximum proportion of missing data allowed for a locus to be retained.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean loci_indices array set.
        """
        self.logger.info(
            f"Filtering loci with missing data proportion > {threshold:.2f} in any population."
        )

        if self.nremover.popmap_inverse is None:
            msg = "No population map data found. Cannot filter by population."
            self.logger.error(msg)
            raise exceptions.MissingPopulationMapError(msg)

        self.nremover.propagate_chain()

        # Check if any loci are still active
        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Try adjusting the filtering parameters."
            )
            return self.nremover

        # Extract the active samples and loci from the alignment
        alignment_array: np.ndarray = (
            self.nremover.alignment[self.nremover.sample_indices, :][
                :, self.nremover.loci_indices
            ]
            .copy()
            .astype(str)
        )

        if alignment_array.size == 0:
            self.logger.warning(
                "No samples remain in the alignment. Try adjusting the filtering parameters."
            )
            return self.nremover

        populations: Dict[str | int, List[str]] = self.nremover.popmap_inverse
        samples: np.ndarray = np.array(self.nremover.samples)[
            self.nremover.sample_indices
        ]

        # Initialize a list to collect population-specific masks for loci
        pop_masks: list = []

        for sample_ids in populations.values():
            # Filter only the sample IDs in the current sample set
            sample_ids_filt: List[str] = [sid for sid in sample_ids if sid in samples]
            indices: List[int] = [
                i for i, sid in enumerate(samples) if sid in sample_ids_filt
            ]

            if not indices:
                continue

            # Calculate the missing data proportion for the current population
            missing_props: npt.NDArray[np.float64] = np.array(
                [
                    np.count_nonzero(np.isin(col[indices], self.missing_vals))
                    / len(indices)
                    for col in alignment_array.T
                ]
            )

            # Mask loci where the missing proportion exceeds the threshold
            pop_mask: npt.NDArray[npt.Bool] = missing_props <= threshold

            self.logger.debug(f"pop_mask: {pop_mask}")
            self.logger.debug(f"pop_mask shape: {pop_mask.shape}")
            self.logger.debug(f"pop_mask sum: {np.count_nonzero(pop_mask)}")

            pop_masks.append(pop_mask)

        # Combine all population masks: Loci must be retained if they meet the
        # threshold in all populations
        if pop_masks:
            cumulative_mask: npt.NDArray[npt.Bool] = np.all(
                np.vstack(pop_masks), axis=0
            )
        else:
            cumulative_mask: npt.NDArray[npt.Bool] = np.zeros(
                alignment_array.shape[1], dtype=bool
            )

        self.logger.debug(f"cumulative_mask: {cumulative_mask}")
        self.logger.debug(f"cumulative_mask shape: {cumulative_mask.shape}")

        # Initialize full_mask with the same length as the alignment's loci
        full_mask: npt.NDArray[npt.Bool] = np.zeros(
            self.nremover.alignment.shape[1], dtype=bool
        )

        # Update loci_indices using the cumulative mask, applied only to active loci
        full_mask[self.nremover.loci_indices] = cumulative_mask

        # Check if any loci remain after filtering
        if np.count_nonzero(full_mask) == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            n_removed: int = np.count_nonzero(~full_mask)
            # Append this information to the filtering results dataframe
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        t: None | float = None if self.nremover.search_mode else threshold

        # Update the loci indices with the final mask
        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)

        return self.nremover

    def _compute_mac_counts(
        self, min_count: int, exclude_heterozygous: bool = False
    ) -> np.ndarray:
        """
        Computes a mask based on minor allele counts (MAC) for each locus.

        Args:
            min_count (int): The minimum minor allele count required to keep a locus.

            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the MAC calculation. Defaults to False.

        Returns:
            np.ndarray: A boolean mask where True means the locus passes the filter.
        """
        mac_counts: npt.NDArray[np.float64 | np.int64] = (
            self._calculate_minor_allele_counts(
                exclude_heterozygous=exclude_heterozygous
            )
        )
        return mac_counts >= min_count

    def filter_mac(
        self, min_count: int, exclude_heterozygous: bool = False
    ) -> "NRemover2":
        """Filters loci where the minor allele count is below the given minimum count.

        The minor allele count is calculated for each locus, and loci with a count below the threshold are removed.

        Args:
            min_count (int): The minimum minor allele count to retain a locus.

            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the MAC calculation. Defaults to False.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean loci_indices array set.
        """
        self.logger.info(f"Filtering loci with minor allele count < {min_count}")

        # Ensure propagation within the chain if needed
        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        # Compute the minor allele counts for the loci
        mac_mask = self.nremover._compute_mac_counts(
            min_count, exclude_heterozygous=exclude_heterozygous
        )

        self.logger.debug(f"mask {inspect.stack()[0][3]}: {mac_mask}")
        self.logger.debug(f"mask shape {inspect.stack()[0][3]}: {mac_mask.shape}")

        if not np.any(mac_mask):
            return self.nremover

        # Create a full mask to track loci to keep
        full_mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        full_mask[self.nremover.loci_indices] = mac_mask

        # Determine how many loci will be kept after applying the mask
        n_to_keep = np.count_nonzero(full_mask)

        if n_to_keep == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            n_removed = np.count_nonzero(~full_mask)
            # Append filtering information to the global list for tracking
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        # If not in search mode, use the count threshold for tracking
        t = None if self.nremover.search_mode else min_count

        # Update the existing loci indices based on the mask
        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)

        return self.nremover

    def filter_maf(
        self, threshold: float, exclude_heterozygous: bool = False
    ) -> "NRemover2":
        """Filters loci where the minor allele frequency is below the threshold.

        The minor allele frequency is calculated as the proportion of the minor allele at each locus.

        Args:
            threshold (float): The minimum minor allele frequency required to keep a locus.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean loci_indices array set.
        """
        self.logger.info(
            f"Filtering loci with minor allele frequency < {threshold:.2f}"
        )

        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        maf_vals = self.nremover._compute_maf_proportions(
            exclude_heterozygous=exclude_heterozygous
        )
        maf_mask = maf_vals >= threshold

        self.logger.debug(f"mask {inspect.stack()[0][3]}: {maf_mask}")
        self.logger.debug(f"mask shape {inspect.stack()[0][3]}: {maf_mask.shape}")

        if not np.any(maf_mask):
            return self.nremover

        full_mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        full_mask[self.nremover.loci_indices] = maf_mask

        # Update loci indices
        n_to_keep = np.count_nonzero(full_mask)

        if n_to_keep == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            n_removed = np.count_nonzero(~full_mask)
            # Append this information to the filtering results dataframe
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        t = None if self.nremover.search_mode else threshold

        # Update existing loci indices
        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)

        return self.nremover

    def filter_linked(self) -> "NRemover2":
        """Filters out linked loci based on VCF file CHROM information.

        Randomly selects one locus from each unique chromosome, ensuring that the selected
        loci are not already filtered out by `self.nremover.loci_indices`.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
        Raises:
            AlignmentFormatError: If the file type is not 'vcf'.
            FileNotFoundError: If the HDF5 file does not exist.
            KeyError: If the key 'chrom' is not present in the HDF5 file.
        """
        self.logger.info("Filtering linked loci based on VCF file CHROM data.")

        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        if self.genotype_data.filetype != "vcf":
            msg = f"Only 'vcf' file type is supported for filtering linked loci, but got {self.genotype_data.filetype}"
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

        # Construct the path to the HDF5 file
        hdf5_path = self.genotype_data.vcf_attributes_fn

        # Check if the HDF5 file exists
        if not Path(hdf5_path).is_file():
            msg = f"The vcf attributes HDF5 file {hdf5_path} does not exist."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Read the chromosome information from the HDF5 file
        with h5py.File(hdf5_path, "r") as f:
            if "chrom" not in f.keys():
                msg = f"'chrom' key absent from the HDF5 file: {hdf5_path}"
                self.logger.error(msg)
                raise KeyError(msg)
            chrom_data = f["chrom"][:]

        # Ensure we're only considering loci that are currently set to True in
        # self.nremover.loci_indices
        valid_loci_indices = np.where(self.nremover.loci_indices)[0]
        valid_chrom_data = chrom_data[self.nremover.loci_indices]

        # Find unique chromosomes from the valid loci
        unique_chroms = np.unique(valid_chrom_data)

        # Create a boolean mask for each unique chromosome
        chrom_masks = {chrom: valid_chrom_data == chrom for chrom in unique_chroms}

        # Initialize an empty mask
        mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)

        # Iterate over the unique chromosomes
        for chrom_mask in chrom_masks.values():
            # Get the indices of loci belonging to the current chromosome
            indices_of_chrom = valid_loci_indices[chrom_mask]

            # Randomly select one locus from the available loci for the current
            # chromosome
            random_index = np.random.choice(indices_of_chrom)

            # Set the selected locus to True in the mask
            mask[random_index] = True

        # Update loci indices
        n_to_keep = np.count_nonzero(mask)

        if n_to_keep == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            n_removed = np.count_nonzero(~mask)
            # Append this information to the filtering results dataframe
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        # Update the loci indices with the mask
        self.nremover._update_loci_indices(mask, inspect.stack()[0][3])

        return self.nremover

    def filter_biallelic(self, exclude_heterozygous: bool = False) -> "NRemover2":
        """Filters alignment to retain ONLY biallelic loci.

        Biallelic loci are those that contain exactly two unique valid alleles.

        Args:
            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the biallelic filtering. This means that when set to True, only homozygous genotypes will be considered, and heterozygous genotypes will be ignored. Defaults to False.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
        """
        self.logger.info(
            f"Filtering loci to retain only biallelic loci (exclude_heterozygous={exclude_heterozygous})."
        )

        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        # Ensure alignment_array is not empty before continuing
        if not np.any(self.nremover.loci_indices) or not np.any(
            self.nremover.sample_indices
        ):
            msg = "No data remain in the alignment before filtering. Try adjusting the filtering parameters."
            self.logger.warning(msg)
            return self.nremover

        # Define valid and invalid bases
        heterozygous_bases = {
            "R": ("A", "G"),
            "Y": ("C", "T"),
            "S": ("G", "C"),
            "W": ("A", "T"),
            "K": ("G", "T"),
            "M": ("A", "C"),
        }
        invalid_bases = np.array(self.missing_vals)

        # Function to count unique valid bases
        def count_valid_bases(col, exclude_heterozygous):
            col = col[~np.isin(col, invalid_bases)]  # Remove invalid bases
            if not exclude_heterozygous:
                expanded_alleles = []
                for base in col:
                    if base in heterozygous_bases:
                        expanded_alleles.extend(heterozygous_bases[base])
                    else:
                        expanded_alleles.append(base)
                col = np.array(expanded_alleles)
            unique_bases = np.unique(col)
            return len(unique_bases)

        # Apply the count_valid_bases function
        unique_base_counts = np.squeeze(
            np.apply_along_axis(
                count_valid_bases,
                0,
                self.nremover.alignment[self.nremover.sample_indices, :],
                exclude_heterozygous,
            )
        )

        # Create a mask for biallelic loci (those with exactly 2 unique bases)
        mask = np.where(unique_base_counts == 2, True, False)

        # Map the final mask back to the original loci_indices
        final_mask = np.logical_and(mask, self.nremover.loci_indices)

        # Update loci indices
        n_to_keep = np.count_nonzero(final_mask)

        if n_to_keep == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            return self.nremover

        t = None if self.nremover.search_mode else int(exclude_heterozygous)

        # Update the loci indices with the final mask
        self.nremover._update_loci_indices(final_mask, inspect.stack()[0][3], t)

        return self.nremover

    def filter_monomorphic(self, exclude_heterozygous: bool = False) -> "NRemover2":
        """Filters out monomorphic sites from an alignment. Monomorphic sites are those that contain only one unique valid allele.

        This method checks each column of the alignment and retains only those columns that are polymorphic.

        Args:
            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the monomorphic filtering.
                                                Defaults to False.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
        """
        self.logger.info(
            f"Filtering out monomorphic loci (exclude_heterozygous={exclude_heterozygous})."
        )

        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        # Define invalid and ambiguous bases based on whether to exclude
        # heterozygous sites
        if exclude_heterozygous:
            invalid_bases = [self.exclude_hets]
        else:
            invalid_bases = [self.missing_vals]

        # Filter out invalid bases and count the unique alleles in each column
        polymorphic_mask = []

        for i in range(self.nremover.alignment.shape[1]):
            col = self.nremover.alignment[:, i]
            # Filter out invalid bases
            valid_alleles = col[~np.isin(col, invalid_bases)]
            unique_alleles = np.unique(valid_alleles)

            # Check if the locus is polymorphic (more than one unique valid allele)
            if len(unique_alleles) > 1:
                polymorphic_mask.append(True)
            else:
                polymorphic_mask.append(False)

        polymorphic_mask = np.array(polymorphic_mask, dtype=bool)

        # Map the polymorphic_mask back to the full loci_indices
        final_mask = np.logical_and(self.nremover.loci_indices, polymorphic_mask)

        if np.count_nonzero(final_mask) == 0:
            self.logger.warning(
                f"No data remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            return self.nremover

        t = None if self.nremover.search_mode else int(exclude_heterozygous)

        self.nremover._update_loci_indices(final_mask, inspect.stack()[0][3], t)

        return self.nremover

    def filter_singletons(self, exclude_heterozygous: bool = False) -> "NRemover2":
        """Filters out singletons from an alignment.

        This method checks each column of the alignment and retains only those columns that are not singletons. A singleton is defined as a locus where a variant appears only once.

        Args:
            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the singleton filtering. This means that when set to True, only homozygous genotypes will be considered, and heterozygous genotypes will be ignored. Defaults to False.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
        """
        self.logger.info("Filtering out singleton loci.")

        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices) or not np.any(
            self.nremover.sample_indices
        ):
            self.logger.warning(
                "No data remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        # Define invalid bases based on whether we are excluding heterozygous positions
        if exclude_heterozygous:
            invalid_bases = np.array(self.exclude_hets)
        else:
            invalid_bases = np.array(self.missing_vals)

        # Function to count valid alleles in a column
        def count_valid_alleles(col):
            valid_col = col[~np.isin(col, invalid_bases)]
            return {
                allele: np.count_nonzero(valid_col == allele)
                for allele in set(valid_col)
            }

        # Determine if a locus is a singleton
        def is_singleton_column(allele_count):
            if len(allele_count) >= 2:
                min_allele_count = min(allele_count.values())
                return min_allele_count == 1
            return False

        # Apply the is_singleton_column function to each column and create the
        # mask
        mask = np.squeeze(
            np.apply_along_axis(
                lambda col: is_singleton_column(count_valid_alleles(col)),
                0,
                self.nremover.alignment[self.nremover.sample_indices, :],
            )
        )

        # Map the mask back to the original self.nremover.loci_indices
        final_mask = np.logical_and(~mask, self.nremover.loci_indices)

        # Update loci indices
        n_to_keep = np.count_nonzero(final_mask)

        if n_to_keep == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            # Append this information to the filtering results dataframe
            return self.nremover

        t = None if self.nremover.search_mode else int(exclude_heterozygous)

        self.nremover._update_loci_indices(final_mask, inspect.stack()[0][3], t)

        return self.nremover

    def random_subset_loci(
        self, size: int | float, seed: int | None = None
    ) -> "NRemover2":
        """Randomly subsets loci based on the `size` parameter.

        The `size` can be an integer (number of loci) or a float (proportion of loci).

        Args:
            size (int or float): Number or proportion of loci to keep.
                                If int, must be >0 and ≤ total loci.
                                If float, must be in (0, 1].
            seed (int | None): Optional random seed for reproducibility.

        Returns:
            NRemover2: NRemover2 object with updated loci_indices.

        Raises:
            ValueError: If size is invalid (not in range or type).
            TypeError: If size is not an int or float.
        """
        self.logger.info("Randomly subsetting loci.")
        self.nremover.propagate_chain()

        # Current mask for loci that are still active
        current_mask = self.nremover.loci_indices

        if not np.any(current_mask):
            self.logger.warning("No loci remain in alignment. Aborting subset.")
            return self.nremover

        total_loci = np.count_nonzero(current_mask)

        # Determine number to keep
        if isinstance(size, int):
            if size <= 0 or size > total_loci:
                msg = f"Invalid size={size}. Must be between 1 and {total_loci}."
                self.logger.error(msg)
                raise exceptions.InvalidThresholdError(size, msg)
            n_to_keep = size
        elif isinstance(size, float):
            if size <= 0.0 or size > 1.0:
                msg = "If float, `size` must be in (0, 1]."
                self.logger.error(msg)
                raise exceptions.InvalidThresholdError(size, msg)
            n_to_keep = int(np.round(total_loci * size))
            if n_to_keep == 0:
                msg = (
                    "Resulting size of 'random_subset_loci' is 0. "
                    "Increase `size` parameter."
                )
                self.logger.error(msg)
                raise exceptions.EmptyLocusSetError(msg)
        else:
            msg = "Size must be an int or float."
            self.logger.error(msg)
            raise TypeError(msg)

        # RNG for reproducibility
        rng = np.random.default_rng(seed)

        # Get absolute indices of loci currently active
        active_loci = np.flatnonzero(current_mask)
        selected_global_indices = rng.choice(active_loci, size=n_to_keep, replace=False)

        # Create new full-length boolean mask
        subset_mask = np.zeros_like(current_mask, dtype=bool)
        subset_mask[selected_global_indices] = True

        self.logger.debug(
            f"Subset selected {n_to_keep}/{total_loci} loci. Global mask shape: {subset_mask.shape}"
        )
        self.logger.debug(
            f"Subset indices: {selected_global_indices[:10]}{'...' if len(selected_global_indices) > 10 else ''}"
        )

        t = None if self.nremover.search_mode else size

        self.nremover._update_loci_indices(subset_mask, inspect.stack()[0][3], t)

        return self.nremover

    def thin_loci(self, size: int | float) -> "NRemover2":
        """Thins loci that are within `size` bases of another SNP.

        This method removes all but one locus within the specified size of another SNP. It is particularly useful for reducing linkage disequilibrium in genomic data. The method uses the VCF file format to identify loci and their positions. It requires the VCF file to be in a specific format with 'chrom' and 'pos' attributes.

        Args:
            size (int): The thinning size. Removes all but one locus within `size` bases of another SNP.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean loci_indices array set.

        Raises:
            AlignmentFormatError: If the file type is not 'vcf'.
            TypeError: If the alignment is NoneType.
            ValueError: If no loci remain in the alignment after filtering.
        """
        self.logger.info(f"Thinning loci within {size} bases of another SNP.")

        self.nremover.propagate_chain()

        if self.nremover.genotype_data.filetype != "vcf":
            msg = f"Only 'vcf' file type is supported for thinning loci, but got {self.nremover.genotype_data.filetype}"
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

        if self.nremover.alignment is None:
            msg = "Alignment must be provided, but got NoneType."
            self.logger.error(msg)
            raise exceptions.AlignmentError(msg)

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        # Get all chrom and pos VCF attributes.
        with h5py.File(self.nremover.genotype_data.vcf_attributes_fn, "r") as f:
            chrom_field = f["chrom"][:]
            pos = f["pos"][:]

        decoder = np.vectorize(lambda x: x.decode("UTF-8"))
        chrom_field = decoder(chrom_field)
        pos = pos.astype(str)

        # Create an array to store which loci to keep
        to_keep = np.ones_like(pos, dtype=bool)
        to_keep[~self.nremover.loci_indices] = False

        unique_chroms = np.unique(chrom_field)

        for chrom in unique_chroms:
            chrom_mask = chrom_field == chrom
            chrom_positions = pos[chrom_mask].astype(int)
            chrom_global_indices = np.flatnonzero(chrom_mask)

            current_keep_mask = to_keep[chrom_global_indices]
            current_chrom_positions = chrom_positions[current_keep_mask]
            current_global_indices = chrom_global_indices[current_keep_mask]

            if len(current_chrom_positions) < 2:
                continue  # Nothing to thin if only one SNP left

            sorted_order = np.argsort(current_chrom_positions)
            sorted_positions = current_chrom_positions[sorted_order]
            sorted_indices = current_global_indices[sorted_order]

            diff = np.diff(sorted_positions)

            bad_indices = sorted_indices[:-1][diff <= size]

            to_keep[bad_indices] = False

        if np.count_nonzero(to_keep) == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Try adjusting the filtering parameters."
            )
            n_removed = np.count_nonzero(~to_keep)
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        t = None if self.nremover.search_mode else size

        self.nremover._update_loci_indices(to_keep, inspect.stack()[0][3], t)

        return self.nremover

    def _append_global_list(
        self, method_name: str, loci_removed: int, loci_removed_prop: float
    ) -> None:
        """Appends filtering results to the global list.

        This method is used to track the filtering process and the number of loci removed at each step.

        Args:
            method_name (str): The name of the filtering method.
            loci_removed (int): The number of loci removed.
            loci_removed_prop (float): The proportion of loci removed.
        """
        self.nremover.df_global_list.append(
            pd.DataFrame(
                {
                    "Step": self.nremover.step_index,
                    "Filter_Method": method_name,
                    "Threshold": self.nremover.current_threshold,
                    "Removed_Count": loci_removed,
                    "Removed_Prop": loci_removed_prop,
                },
                index=[0],
            )
        )

    def _append_sample_list(
        self, method_name: str, removed: int, removed_prop: float
    ) -> None:
        """Appends filtering results to the sample list.

        This method is used to track the filtering process and the number of samples removed at each step.

        Args:
            method_name (str): The name of the filtering method.
            removed (int): The number of samples removed.
            removed_prop (float): The proportion of samples removed.
        """
        self.nremover.df_sample_list.append(
            pd.DataFrame(
                {
                    "Step": self.nremover.step_index,
                    "Filter_Method": method_name,
                    "Threshold": self.nremover.current_threshold,
                    "Removed_Count": removed,
                    "Removed_Prop": removed_prop,
                },
                index=[0],
            )
        )

    def filter_missing_sample(self, threshold: float) -> "NRemover2":
        """Filters samples with a proportion of missing data greater than the specified threshold.

        This method calculates the proportion of missing data for each sample and removes those that exceed the threshold.

        Args:
            threshold (float): The maximum proportion of missing data allowed for a sample.

        Returns:
            NRemover2: The NRemover2 object with the filtered alignment's boolean sample_indices array set.

        Raises:
            ValueError: If the threshold is not in the range [0, 1].
        """
        if not 0 <= threshold <= 1:
            msg = f"Threshold must be in the range [0, 1], but got: {threshold}."
            self.logger.error(msg)
            raise exceptions.InvalidThresholdError(threshold, msg)

        self.logger.info(
            f"Filtering sequences (samples) with missing data proportion > {threshold:.2f}"
        )

        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        alignment_array = (
            self.nremover.alignment[self.nremover.sample_indices, :][
                :, self.nremover.loci_indices
            ]
            .copy()
            .astype(str)
        )

        if alignment_array.size == 0:
            self.logger.warning(
                "No samples remain in the alignment before filtering. Try adjusting the filtering parameters."
            )
            return self.nremover

        missing_counts = np.count_nonzero(
            np.isin(alignment_array, self.missing_vals), axis=1
        )

        # Calculate the proportion of missing data per sample based on the number of loci
        num_loci = alignment_array.shape[1]
        missing_props = missing_counts / num_loci

        mask = missing_props <= threshold

        self.logger.debug(f"mask {inspect.stack()[0][3]}: {mask}")
        self.logger.debug(f"mask shape {inspect.stack()[0][3]}: {mask.shape}")

        # Map the mask back to the full sample_indices
        final_mask = np.zeros_like(self.nremover.sample_indices, dtype=bool)
        final_mask[self.nremover.sample_indices] = mask

        # Update loci indices
        n_to_keep = np.count_nonzero(final_mask)

        if n_to_keep == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Adjust filtering parameters."
            )
            # Append this information to the filtering results dataframe
            self._append_sample_list(
                inspect.stack()[0][3],
                np.count_nonzero(self.nremover.sample_indices) - n_to_keep,
                1.0,
            )
            return self.nremover

        t = None if self.nremover.search_mode else threshold

        # Update the sample indices with the final mask
        self.nremover._update_sample_indices(final_mask, inspect.stack()[0][3], t)

        return self.nremover

    def _compute_maf_proportions(
        self, exclude_heterozygous: bool = False
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Compute MAF values and a boolean mask for loci.

        This method calculates the minor allele frequency (MAF) for each locus in the alignment and returns a mask indicating which loci are active. The MAF is calculated as the proportion of the minor allele at each locus.

        Args:
            exclude_heterozygous (bool): If True, ignore heterozygotes in allele counts.

        Returns:
            maf_vals: np.ndarray of shape (n_active_loci,) with minor‐allele frequencies.
        """
        # select only active samples & loci
        aln = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)
        if aln.size == 0:
            n = self.nremover.loci_indices.sum()
            return np.zeros(n, float), np.zeros(n, bool)

        # prepare IUPAC mapping
        # Filter ambiguity codes to only 2-base ones
        ambig = {
            code: (value[0], value[1])
            for code, value in self.iupac.ambiguous_dna_values.items()
            if len(value) == 2
        }
        bases = ("A", "C", "G", "T")

        def locus_maf(col: np.ndarray) -> float:
            # count alleles diploid
            counts: Dict[str, int] = {b: 0 for b in bases}
            for b in bases:
                counts[b] += 2 * np.sum(col == b)
            if not exclude_heterozygous:
                for code, (b1, b2) in ambig.items():
                    n = np.sum(col == code)
                    counts[b1] += n
                    counts[b2] += n
            vals = np.array([counts[b] for b in bases], float)
            tot = vals.sum()
            if tot == 0 or np.count_nonzero(vals) < 2:
                return 0.0
            sorted_vals = np.sort(vals)[::-1]
            return sorted_vals[1] / tot

        maf_vals = np.apply_along_axis(locus_maf, 0, aln)
        return maf_vals

    def _calculate_minor_allele_counts(
        self, exclude_heterozygous: bool = False
    ) -> np.ndarray:
        """Calculate minor allele counts for each locus in the alignment.

        This method computes the minor allele count (MAC) for each locus in the alignment, which is the count of the second most common allele.

        Args:
            exclude_heterozygous (bool): Whether to exclude heterozygous sites from the MAC calculation.

        Returns:
            np.ndarray: An array of minor allele counts for each locus.
        """
        ACGT = ("A", "C", "G", "T")
        arr = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)

        if arr.size == 0:
            return np.zeros(arr.shape[1], dtype=int)

        # Filter ambiguity codes to only 2-base ones
        ambig = {
            code: (value[0], value[1])
            for code, value in self.iupac.ambiguous_dna_values.items()
            if len(value) == 2
        }

        def count_alleles(col):
            # Start counts at zero
            counts = dict.fromkeys(ACGT, 0)
            # Homozygotes contribute two
            for base in ACGT:
                counts[base] += 2 * np.sum(col == base)
            # Heterozygotes contribute one of each
            if not exclude_heterozygous:
                for code, (b1, b2) in ambig.items():
                    mask = col == code
                    n = np.sum(mask)
                    counts[b1] += n
                    counts[b2] += n
            # now pull out counts and compute minor allele count
            vals = np.array([counts[b] for b in ACGT])
            # if fewer than two alleles observed, minor = 0
            if np.count_nonzero(vals) < 2:
                return 0
            # sort descending; minor = second largest
            return np.sort(vals)[-2]

        mac = np.apply_along_axis(count_alleles, 0, arr)
        return mac
