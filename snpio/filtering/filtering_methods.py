import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy._typing._array_like import NDArray

# Custom imports
from snpio.utils.benchmarking import Benchmark
from snpio.utils.custom_exceptions import AlignmentFormatError
from snpio.utils.misc import IUPAC

measure_execution_time: Callable = Benchmark.measure_execution_time


class FilteringMethods:
    def __init__(self, nremover_instance: Any) -> None:
        """Initialize FilteringMethods class with an instance of NRemover2.

        This class contains methods for filtering loci (columns) and samples (rows) based on various criteria.

        Args:
            nremover_instance (NRemover2): An instance of NRemover2 to access its attributes. The instance should be initialized with the alignment data and other attributes.

        Returns:
            None

        Note:
            The class uses the NRemover2 instance to access the alignment data and other attributes.

            The class also uses the genotype_data attribute to access the resource data and other attributes.

            The class uses the logger attribute to log messages.

            The class uses the missing_vals attribute to define missing data values.

            The class uses the _append_global_list method to append filtering results to the global list.

            The class uses the _calculate_minor_allele_counts method to compute minor allele counts.

            The class uses the _compute_maf_proportions method to compute minor allele frequencies.

            The class uses the _update_loci_indices method to update loci indices based on a mask.

            The class uses the _update_sample_indices method to update sample indices based on a mask.

        Attributes:
            nremover (NRemover2): An instance of NRemover2 to access its attributes.
            logger (Logger): A logger object to log messages.
            resource_data (dict): A dictionary containing resource data.
            missing_vals (list): A list of missing data values.
        """
        self.nremover: Any = nremover_instance
        self.logger: logging.Logger = self.nremover.logger
        self.resource_data: dict = self.nremover.genotype_data.resource_data
        self.missing_vals: List[str] = ["N", "-", ".", "?"]
        self.ambiguous_bases: List[str] = ["R", "Y", "S", "W", "K", "M"]
        self.exclude_hets: List[str] = self.missing_vals + self.ambiguous_bases

        if not self.resource_data:
            self.resource_data = {}

        self.iupac = IUPAC()

    @measure_execution_time
    def filter_missing(self, threshold: float) -> Any:
        """Filters out columns (loci) with missing data proportion greater than the given threshold.

        Args:
            threshold (float): The maximum proportion of missing data allowed for a locus to be retained.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean loci_indices array set.

        Raises:
            TypeError: If the threshold is not a float.
            ValueError: If the threshold is not between 0.0 and 1.0 inclusive.
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
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0 inclusive, but got: {threshold:.2f}"
            )

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

    @measure_execution_time
    def filter_missing_pop(self, threshold: float) -> Any:
        """Filters loci based on missing data proportion per population.

        The method filters loci where the missing data proportion exceeds the given threshold in any population.

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
            raise ValueError(msg)

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

    @measure_execution_time
    def filter_mac(self, min_count: int, exclude_heterozygous: bool = False) -> Any:
        """Filters loci where the minor allele count is below the given minimum count.

        Args:
            min_count (int): The minimum minor allele count to retain a locus.

            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the MAC calculation. Defaults to False.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean loci_indices array set.
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

    @measure_execution_time
    def filter_maf(self, threshold: float) -> Any:
        """Filters loci where the minor allele frequency is below the threshold.

        Args:
            threshold (float): The minimum minor allele frequency required to keep a locus.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean loci_indices array set.
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

        maf_mask = self.nremover._compute_maf_proportions(threshold)

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

    @measure_execution_time
    def filter_linked(self) -> Any:
        """
        Filters out linked loci based on VCF file CHROM information.

        Randomly selects one locus from each unique chromosome, ensuring that the selected
        loci are not already filtered out by `self.nremover.loci_indices`.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
        Raises:
            OSError: Unsupported file type provided.
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
                raise KeyError()
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

    @measure_execution_time
    def filter_biallelic(self, exclude_heterozygous: bool = False) -> Any:
        """Filters alignment to retain ONLY biallelic loci.

        Args:
            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the biallelic filtering. This means that when set to True, only homozygous genotypes will be considered, and heterozygous genotypes will be ignored. Defaults to False.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
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
            "R": ("A", "G"),  # A or G
            "Y": ("C", "T"),  # C or T
            "S": ("G", "C"),  # G or C
            "W": ("A", "T"),  # A or T
            "K": ("G", "T"),  # G or T
            "M": ("A", "C"),  # A or C
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

    @measure_execution_time
    def filter_monomorphic(self, exclude_heterozygous: bool = False) -> Any:
        """Filters out monomorphic sites from an alignment. Monomorphic sites are those that contain only one unique valid allele.

        Args:
            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the monomorphic filtering.
                                                Defaults to False.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
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

    @measure_execution_time
    def filter_singletons(self, exclude_heterozygous: bool = False) -> Any:
        """Filters out singletons from an alignment. A singleton is defined as a locus where one variant appears only once.

        Args:
            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the singleton filtering. This means that when set to True, only homozygous genotypes will be considered, and heterozygous genotypes will be ignored. Defaults to False.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean sample_indices array set.
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

    @measure_execution_time
    def random_subset_loci(self, size: Union[int, float]) -> Any:
        """Randomly subsets loci based on the `size` parameter.

        Args:
            size (int or float): The number or proportion of loci to subset. If int, the exact number of loci to keep. Must be less than the total number of loci. If float, the proportion of loci to keep (must be in (0, 1]). If the number of loci to keep is greater than the total number of loci, the loci will be randomly sampled with replacement.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean loci_indices array set.
        """
        self.logger.info("Randomly subsetting loci.")

        self.nremover.propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        total_loci = np.count_nonzero(self.nremover.loci_indices)

        # Validate size and calculate the number of loci to keep
        if isinstance(size, int):
            if size <= 0 or size > total_loci:
                msg = f"If size is an integer, it must be between 0 and the total number of remaining loci: Total loci={total_loci}, size={size}."
                self.logger.error(msg)
                raise RuntimeError(msg)
            n_to_keep = size
        else:
            if size <= 0.0 or size > 1.0:
                raise ValueError(
                    "If size is a float, it must be in the interval (0, 1]."
                )
            n_to_keep = int(np.round(total_loci * size))

        replace = n_to_keep > total_loci

        # Randomly select loci to keep
        subset_indices = np.random.choice(total_loci, size=n_to_keep, replace=replace)

        # Create the subset mask, ensuring it's aligned with loci_indices
        subset_mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        active_loci = np.where(self.nremover.loci_indices)[0]  # Get active loci
        subset_mask[active_loci[subset_indices]] = True

        # Update loci indices
        n_to_keep = np.count_nonzero(subset_mask)

        if not n_to_keep:
            self.logger.warning(
                "No loci remain after randomly subsetting loci. Adjust filtering parameters."
            )
            n_removed = np.count_nonzero(~subset_mask)
            # Append this information to the filtering results dataframe
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        t = None if self.nremover.search_mode else size

        self.nremover._update_loci_indices(subset_mask, inspect.stack()[0][3], t)

        return self.nremover

    @measure_execution_time
    def thin_loci(self, size: Union[int, float]) -> Any:
        """Thins loci that are within `size` bases of another SNP.

        Uses the CHROM and POS fields of a VCF file to determine the locations of the loci.

        Args:
            size (int): The thinning size. Removes all but one locus within `size` bases of another SNP.

        Returns:
            self: The NRemover2 object with the filtered alignment's boolean loci_indices array set.
        """
        self.logger.infO(f"Thinning loci within {size} bases of another SNP.")

        self.nremover.propagate_chain()

        if self.genotype_data.filetype != "vcf":
            msg = f"Only 'vcf' file type is supported for thinning loci, but got {self.genotype_data.filetype}"
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

        if self.nremover.alignment is None:
            msg = "Alignment must be provided, but got NoneType."
            self.logger.error(msg)
            raise TypeError(msg)

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        # Get all chrom and pos VCF attributes.
        with h5py.File(self.genotype_data.vcf_attributes_fn, "r") as f:
            chrom_field = f["chrom"][:]
            pos = f["pos"][:]

        decoder = np.vectorize(lambda x: x.decode("UTF-8"))
        chrom_field = decoder(chrom_field)
        pos = pos.astype(str)

        # Create an array to store which loci to keep
        to_keep = np.ones_like(pos, dtype=bool)
        to_keep[~self.nremover.loci_indices] = False

        # Loop through each chromosome
        unique_chroms = np.unique(chrom_field)

        for chrom in unique_chroms:
            chrom_mask = chrom_field == chrom  # Only current chromosome
            chrom_positions = pos[chrom_mask]  # Subset to current CHROM value.
            chrom_indices = np.logical_and(chrom_mask, to_keep)  # Get mask

            # Sort positions and corresponding indices
            sorted_order = np.argsort(chrom_positions.astype(int))
            sorted_positions = chrom_positions[sorted_order]
            sorted_indices = chrom_indices[sorted_order]

            diff = np.diff(sorted_positions.astype(int))
            to_keep[sorted_indices[:-1][diff <= size]] = False

        if np.count_nonzero(to_keep) == 0:
            self.logger.warning(
                f"No loci remain in the alignment after {inspect.stack()[0][3]}. Try adjusting the filtering parameters."
            )
            n_removed = np.count_nonzero(~to_keep)

            # Append this information to the filtering results dataframe
            self._append_global_list(inspect.stack()[0][3], n_removed, 1.0)
            return self.nremover

        t = None if self.nremover.search_mode else size

        self.nremover._update_loci_indices(to_keep, inspect.stack()[0][3], t)

        return self.nremover

    def _append_global_list(
        self, method_name: str, loci_removed: int, loci_removed_prop: float
    ) -> None:
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

    @measure_execution_time
    def filter_missing_sample(self, threshold: float) -> Any:
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
        self, maf_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MAF proportions and create a mask for loci based on the MAF threshold.

        Args:
            maf_threshold (float): The minimum minor allele frequency required to keep a locus.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the MAF proportions and the mask for loci that pass the MAF threshold.
        """
        alignment_array: npt.NDArray[Any] = (
            self.nremover.alignment[self.nremover.sample_indices, :][
                :, self.nremover.loci_indices
            ]
            .copy()
            .astype(str)
        )

        # Ensure alignment_array is not empty before continuing
        if alignment_array.size == 0:
            msg = "No samples remain in the alignment before filtering MAF. Try adjusting the filtering parameters."
            self.logger.warning(msg)
            return np.zeros_like(self.nremover.loci_indices, dtype=bool)

        def count_bases(column):
            """
            Count occurrences of A, C, G, and T in a SNP data column, distributing heterozygous counts across alleles.
            """
            base_count: Dict[str, int] = {"A": 0, "C": 0, "G": 0, "T": 0}
            valid_bases: NDArray[Any] = np.array(["A", "C", "G", "T"])

            for base in valid_bases:
                base_count[base] = np.sum(column == base)

            ambiguous_bases: Dict[str, str] = {
                base: self.iupac.ambiguous_dna_values[base]
                for base in self.iupac.ambiguous_dna_values
            }

            for ambig_base, mapping in ambiguous_bases.items():
                mask = column == ambig_base
                split_count = np.sum(mask) / len(mapping)
                for mapped_base in mapping:
                    base_count[mapped_base] += split_count

            return base_count

        def minor_allele_frequency(column: np.ndarray) -> float:
            """Calculates the minor allele frequency for a given column of SNP data.

            Args:
                column (numpy.ndarray): A numpy array of bases.

            Returns:
                float: The minor allele frequency for the given column.
            """
            counts: Dict[str, int] = count_bases(column)
            valid_bases: set[str] = {"A", "C", "G", "T"}
            counts = {
                base: count for base, count in counts.items() if base in valid_bases
            }

            if not counts or all(count == 0 for count in counts.values()):
                return 0

            sorted_counts: NDArray[Any] = np.array(
                sorted(counts.values(), reverse=True)
            )
            total: np.int64 = np.sum(sorted_counts)
            if total == 0:
                return 0
            freqs = sorted_counts / total
            return freqs[1] if len(freqs) > 1 else 0

        # Calculate MAF for each column in alignment_array
        maf: NDArray[Any] = np.apply_along_axis(
            minor_allele_frequency, 0, alignment_array
        )
        return maf >= maf_threshold

    def _calculate_minor_allele_counts(
        self, exclude_heterozygous: bool = False
    ) -> np.ndarray:
        """Calculate the minor allele counts (MAC) for each locus in the alignment.

        Args:
            exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the MAC calculation. Defaults to False.

        Returns:
            numpy.ndarray: An array containing the minor allele count for each locus.
        """
        alignment_array: npt.NDArray[Any] = (
            self.nremover.alignment[self.nremover.sample_indices, :][
                :, self.nremover.loci_indices
            ]
            .copy()
            .astype(str)
        )

        # Ensure alignment_array is not empty before continuing
        if alignment_array.size == 0:
            msg = "No samples remain in the alignment. Try adjusting the filtering parameters."
            self.logger.warning(msg)
            return np.zeros(alignment_array.shape[1], dtype=int)

        def count_bases(column: np.ndarray) -> Dict[str, int]:
            """Count occurrences of A, C, G, and T in a SNP data column, distributing heterozygous counts across alleles.

            Args:
                column (numpy.ndarray): A numpy array of bases.

            Returns:
                Dict[str, int]: A dictionary with counts of 'A', 'C', 'G', 'T'.
            """
            base_count: Dict[str, int] = {"A": 0, "C": 0, "G": 0, "T": 0}

            if exclude_heterozygous:
                valid_bases: npt.NDArray[Any] = np.array(["A", "C", "G", "T"])
            else:
                valid_bases: npt.NDArray[Any] = np.array(
                    ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M"]
                )

            for base in valid_bases:
                base_count[base] = np.sum(column == base)

            ambiguous_bases: Dict[str, str] = {
                base: self.iupac.ambiguous_dna_values[base]
                for base in self.iupac.ambiguous_dna_values
            }

            for ambig_base, mapping in ambiguous_bases.items():
                mask = column == ambig_base
                split_count = np.sum(mask) / len(mapping)
                for mapped_base in mapping:
                    base_count[mapped_base] += split_count

            return base_count

        def minor_allele_count(column: np.ndarray) -> int:
            """Calculates the minor allele count for a given column of SNP data.

            Args:
                column (np.ndarray): A numpy array of bases.

            Returns:
                int: The minor allele count for the given column.
            """
            counts: Dict[str, int] = count_bases(column)
            valid_bases: set[str] = {"A", "C", "G", "T"}
            counts = {
                base: count for base, count in counts.items() if base in valid_bases
            }

            if not counts or all(count == 0 for count in counts.values()):
                return 0

            # Sort the allele counts and return the second highest (minor allele count)
            sorted_counts: NDArray[Any] = np.array(
                sorted(counts.values(), reverse=True)
            )
            return sorted_counts[1] if len(sorted_counts) > 1 else 0

        # Calculate MAC for each column (locus) in the alignment_array
        mac = np.apply_along_axis(minor_allele_count, 0, alignment_array)
        return mac
