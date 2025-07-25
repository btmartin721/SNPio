import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

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

        This class contains methods for filtering loci (columns) and samples (rows) based on various criteria. The methods are designed to work with the NRemover2 instance, which contains the alignment data and other attributes necessary for filtering.

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
        self.missing_vals: List[str] = ["N", "-", ".", "?"]
        self.ambiguous_bases: List[str] = ["R", "Y", "S", "W", "K", "M"]
        self.exclude_hets: List[str] = self.missing_vals + self.ambiguous_bases

        self.iupac: IUPAC = IUPAC(logger=self.logger)

    def filter_missing(self, threshold: float) -> "NRemover2":
        """Filters out loci (columns) with missing data proportion greater than the specified threshold.

        A genotype is considered missing for if it contains any of the following values: 'N', '-', '.', or '?'.

        Args:
            threshold (float): Maximum allowable proportion of missing data per locus (0.0 to 1.0 inclusive).

        Returns:
            NRemover2: The NRemover2 object with updated loci_indices.

        Raises:
            TypeError: If threshold is not a float.
            InvalidThresholdError: If threshold is outside the range [0.0, 1.0].

        Notes:
            - Uses vectorized NumPy operations to calculate missingness.
            - The mask is applied only to currently active loci and samples.
        """

        self.logger.info(
            f"Filtering loci with missing data proportion > {threshold:.3f}"
        )

        self.nremover._propagate_chain()

        if not isinstance(threshold, float):
            msg: str = f"Threshold must be a float value, but got: {type(threshold)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if threshold < 0.0 or threshold > 1.0:
            msg = f"Threshold must be between [0.0, 1.0], but got: {threshold:.3f}"
            self.logger.error(msg)
            raise exceptions.InvalidThresholdError(threshold, msg)

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Try adjusting filtering parameters."
            )
            return self.nremover

        alignment_array: np.ndarray = self.nremover.alignment[
            self.nremover.sample_indices, :
        ][:, self.nremover.loci_indices].astype(str, copy=False)

        missing_counts: int = np.count_nonzero(
            np.isin(alignment_array, self.missing_vals), axis=0
        )

        num_samples: int = np.count_nonzero(self.nremover.sample_indices)

        self.logger.debug(f"Total Loci Before {inspect.stack()[0][3]}: {num_samples}")

        missing_props: float = missing_counts / num_samples

        self.logger.debug(f"Total Loci Before {inspect.stack()[0][3]}: {num_samples}")
        self.logger.debug(f"missing_props: {missing_props}")
        self.logger.debug(f"missing_props shape: {missing_props.shape}")

        retained_loci_mask = missing_props <= threshold

        self.logger.debug(f"mask {inspect.stack()[0][3]}: {retained_loci_mask}")
        self.logger.debug(
            f"mask shape {inspect.stack()[0][3]}: {retained_loci_mask.shape}"
        )

        full_mask: np.ndarray = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        full_mask[self.nremover.loci_indices] = retained_loci_mask

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
        """Filters loci (columns) based on missing data per population.

        This method calculates the proportion of missing genotype values per locus for each user-defined population. A locus is retained only if the missing proportion is ≤ threshold in **every** population. Missing genotypes are defined as: "N", "-", ".", or "?". Sample-to-population assignments are taken from `self.nremover.popmap_inverse`.

        Args:
            threshold (float): Maximum allowable proportion of missing data per population.

        Returns:
            NRemover2: NRemover2 object with updated `loci_indices`.

        Raises:
            MissingPopulationMapError: If `popmap_inverse` is not defined.
        """

        self.logger.info(
            f"Filtering loci with missing data proportion > {threshold:.3f} in any population."
        )

        if self.nremover.popmap_inverse is None:
            msg = "No population map data found. Cannot filter by population."
            self.logger.error(msg)
            raise exceptions.MissingPopulationMapError(msg)

        self.nremover._propagate_chain()

        # Check if any loci are still active
        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Try adjusting the filtering parameters."
            )
            return self.nremover

        # Extract the active samples and loci from the alignment
        alignment_array: np.ndarray = self.nremover.alignment[
            self.nremover.sample_indices, :
        ][:, self.nremover.loci_indices].astype(str, copy=False)

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
            sample_ids_filt_set = set(sample_ids)
            indices = np.where(np.isin(samples, list(sample_ids_filt_set)))[0]

            if indices.size == 0:
                continue

            # shape: (n_sub_samples, n_loci)
            submatrix = alignment_array[indices, :]
            missing_matrix = np.isin(submatrix, self.missing_vals)
            missing_props = np.mean(missing_matrix, axis=0)

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
            self.logger.warning(
                "No populations had retained samples; all loci will be filtered."
            )
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

        # Update the loci indices with the final mask
        t: None | float = None if self.nremover.search_mode else threshold
        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)
        return self.nremover

    def filter_missing_sample(self, threshold: float) -> "NRemover2":
        """Remove samples with missing data proportion > threshold.

        This method evaluates the proportion of missing genotypes for each sample and removes those whose missing rate exceeds the user-defined ``threshold`` argument.

        Args:
            threshold (float): Proportion cutoff (0 ≤ threshold ≤ 1). Samples with more missing data than this threshold will be removed.

        Returns:
            NRemover2: The NRemover2 object with updated `sample_indices`.

        Raises:
            InvalidThresholdError: If `threshold` is not in the range [0, 1].
        """
        if not 0.0 <= threshold <= 1.0:
            msg = f"Threshold must be in the range [0, 1], but got: {threshold}."
            self.logger.error(msg)
            raise exceptions.InvalidThresholdError(threshold, msg)

        self.logger.info(
            f"Filtering samples with missing data proportion > {threshold:.3f}"
        )

        self.nremover._propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning("No loci remain before filtering samples.")
            return self.nremover

        # Subset alignment to retained samples and loci
        alignment = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)

        if alignment.size == 0:
            self.logger.warning("No data remain before filtering samples.")
            return self.nremover

        # Compute proportion of missing values per sample
        is_missing = np.isin(alignment, self.missing_vals)
        missing_props = is_missing.sum(axis=1) / alignment.shape[1]

        # Samples to keep: those with missing proportion <= threshold
        sample_mask = missing_props <= threshold

        # Map to full-length sample_indices
        final_mask = np.zeros_like(self.nremover.sample_indices, dtype=bool)
        final_mask[self.nremover.sample_indices] = sample_mask

        n_to_keep = final_mask.sum()

        if n_to_keep == 0:
            self.logger.warning(
                f"No samples remain after {inspect.stack()[0][3]}. Try adjusting the threshold."
            )
            self._append_sample_list(
                inspect.stack()[0][3],
                np.count_nonzero(self.nremover.sample_indices),
                1.0,
            )
            return self.nremover

        t = None if self.nremover.search_mode else threshold
        self.nremover._update_sample_indices(final_mask, inspect.stack()[0][3], t)

        return self.nremover

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
        self.nremover._propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        # Compute the minor allele counts for the loci
        mac_mask = self._compute_mac_counts(
            min_count, exclude_heterozygous=exclude_heterozygous
        )

        self.logger.debug(f"mask {inspect.stack()[0][3]}: {mac_mask}")
        self.logger.debug(f"mask shape {inspect.stack()[0][3]}: {mac_mask.shape}")

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

        self.nremover._propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        maf_vals = self._compute_maf_proportions(
            exclude_heterozygous=exclude_heterozygous
        )
        maf_mask = maf_vals >= threshold

        self.logger.debug(f"mask {inspect.stack()[0][3]}: {maf_mask}")
        self.logger.debug(f"mask shape {inspect.stack()[0][3]}: {maf_mask.shape}")

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

    def filter_linked(self, seed: int | None = None) -> "NRemover2":
        """Filter to retain only one locus per chromosome/scaffold from a VCF file.

        For each unique value in the VCF 'CHROM' field, this method randomly selects a single locus among currently active loci (`loci_indices`) and removes all others. This is intended to reduce linkage by ensuring no more than one SNP per chromosome or scaffold.

        Args:
            seed (int | None): Optional seed for reproducibility. If None, uses a random number generator without a fixed seed.

        Returns:
            NRemover2: The updated NRemover2 object with filtered `loci_indices`.

        Raises:
            AlignmentFormatError: If filetype is not VCF.
            FileNotFoundError: If the HDF5 file containing CHROM data is missing.
            KeyError: If the HDF5 file lacks a 'chrom' field.
        """
        self.logger.info("Filtering linked loci based on VCF file CHROM data.")

        # Establish a random number generator for reproducibility
        # (if seed is provided)
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

        self.nremover._propagate_chain()

        if not np.any(self.nremover.loci_indices):
            self.logger.warning(
                "No loci remain in the alignment. Adjust filtering parameters."
            )
            return self.nremover

        if self.nremover.genotype_data.filetype != "vcf":
            msg = (
                "Only 'vcf' file type is supported for filtering linked loci, "
                f"but got {self.nremover.genotype_data.filetype}"
            )
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

        # Construct the path to the HDF5 file
        hdf5_path = self.nremover.genotype_data.vcf_attributes_fn

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

        # Initialize an empty mask
        mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)

        for chrom in np.unique(valid_chrom_data):
            chrom_mask = valid_chrom_data == chrom
            indices_of_chrom = valid_loci_indices[chrom_mask]

            # Get the indices of loci belonging to the current chromosome
            indices_of_chrom = valid_loci_indices[chrom_mask]

            # Randomly select one locus from the available loci for the current
            # chromosome
            random_index = rng.choice(indices_of_chrom)

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
        if not hasattr(self.nremover, "current_thresholds"):
            self.nremover.current_thresholds = (None, None, None, None)
        self.nremover._update_loci_indices(mask, inspect.stack()[0][3])

        return self.nremover

    def thin_loci(
        self, size: int, remove_all: bool = False, seed: int | None = None
    ) -> "NRemover2":
        """Thin loci that are within `size` base pairs of each other.

        Operates per chromosome/scaffold and either:

        - Retains one SNP per cluster of nearby SNPs (`remove_all=False`, default).
        - Removes all loci that have another SNP within `size` bp (`remove_all=True`).

        Args:
            size (int): Distance in base pairs for defining SNP proximity.
            remove_all (bool): If True, removes *all* SNPs within `size` of another. If False, retains one SNP per proximity cluster.
            seed (int | None): Optional seed for reproducibility.

        Returns:
            NRemover2: The updated NRemover2 object with thinned loci.

        Raises:
            AlignmentFormatError: If filetype is not VCF.
            AlignmentError: If alignment is missing.
        """
        self.logger.info(f"Thinning loci within {size} bp.")
        self.logger.info(
            f"Mode: {'remove all' if remove_all else 'retain one randomly'}."
        )

        self.nremover._propagate_chain()

        if self.nremover.genotype_data.filetype != "vcf":
            msg = f"Only 'vcf' file type is supported for thinning, but got {self.nremover.genotype_data.filetype}"
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

        if self.nremover.alignment is None:
            msg = "Alignment must be provided, but got None."
            self.logger.error(msg)
            raise exceptions.AlignmentError(msg)

        if not np.any(self.nremover.loci_indices):
            self.logger.warning("No loci remain in the alignment. Skipping thinning.")
            return self.nremover

        # Load CHROM and POS attributes
        with h5py.File(self.nremover.genotype_data.vcf_attributes_fn, "r") as f:
            chroms = np.array([x.decode("utf-8") for x in f["chrom"][:]])
            positions = f["pos"][:].astype(int)

        to_keep = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

        # Only consider loci that are currently retained
        active_mask = self.nremover.loci_indices
        active_chroms = chroms[active_mask]
        active_positions = positions[active_mask]
        active_indices = np.flatnonzero(active_mask)

        for chrom in np.unique(active_chroms):
            # Get loci on this chromosome that are still active
            chrom_mask = active_chroms == chrom
            chrom_pos = active_positions[chrom_mask]
            chrom_inds = active_indices[chrom_mask]

            if len(chrom_pos) < 2:
                to_keep[chrom_inds] = True
                continue

            # Sort by position
            order = np.argsort(chrom_pos)
            sorted_pos = chrom_pos[order]
            sorted_inds = chrom_inds[order]

            if remove_all:
                # Find close pairs and mark both for removal
                diff = np.diff(sorted_pos)
                close = np.where(diff <= size)[0]
                bad_inds = np.unique(np.concatenate([close, close + 1]))
                retain_inds = np.setdiff1d(np.arange(len(sorted_inds)), bad_inds)
                to_keep[sorted_inds[retain_inds]] = True
            else:
                # Build proximity clusters and retain one SNP randomly per
                # cluster
                clusters = []
                cluster = [sorted_inds[0]]

                for i in range(1, len(sorted_pos)):
                    if sorted_pos[i] - sorted_pos[i - 1] <= size:
                        cluster.append(sorted_inds[i])
                    else:
                        if len(cluster) > 1:
                            clusters.append(cluster)
                        else:
                            to_keep[cluster[0]] = True
                        cluster = [sorted_inds[i]]

                # Handle final cluster
                if len(cluster) > 1:
                    clusters.append(cluster)
                else:
                    to_keep[cluster[0]] = True

                # Randomly keep one per cluster
                for cluster in clusters:
                    keep_idx = rng.choice(cluster)
                    to_keep[keep_idx] = True

        if np.count_nonzero(to_keep) == 0:
            self.logger.warning(
                f"No loci remain after thinning with size={size}. Adjust filtering parameters."
            )
            self._append_global_list(
                inspect.stack()[0][3], np.count_nonzero(~to_keep), 1.0
            )
            return self.nremover

        t = None if self.nremover.search_mode else size
        self.nremover._update_loci_indices(to_keep, inspect.stack()[0][3], t)
        return self.nremover

    def filter_biallelic(self, exclude_heterozygous: bool = False) -> "NRemover2":
        """Retain only biallelic loci and remove those with more than two alleles.

        This method keeps loci with exactly 2 alleles from A, C, G, T. IUPAC ambiguity codes contribute both alleles if `exclude_heterozygous=False`.

        Args:
            exclude_heterozygous (bool): If True, ignores heterozygous genotypes in allele counts.

        Returns:
            NRemover2: The updated NRemover2 object.
        """
        self.logger.info(
            f"Filtering to biallelic loci (exclude_heterozygous={exclude_heterozygous})"
        )
        self.nremover._propagate_chain()

        if not np.any(self.nremover.loci_indices) or not np.any(
            self.nremover.sample_indices
        ):
            self.logger.warning("No data remain before filtering.")
            return self.nremover

        ACGT = ("A", "C", "G", "T")
        base_to_idx = {b: i for i, b in enumerate(ACGT)}
        aln = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)
        n_bases, n_loci = len(ACGT), aln.shape[1]
        allele_presence = np.zeros((n_bases, n_loci), dtype=bool)

        for code, alleles in self.iupac.ambiguous_dna_values.items():
            # Skip invalid/missing codes
            if code in ("N", "X", "-", "?", "."):
                continue

            if exclude_heterozygous and len(set(alleles)) > 1:
                continue

            mask = aln == code
            col_mask = mask.any(axis=0)
            for base in alleles:
                if base in base_to_idx:
                    allele_presence[base_to_idx[base]] |= col_mask

        allele_counts = allele_presence.sum(axis=0)
        biallelic_mask = allele_counts == 2

        full_mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        full_mask[self.nremover.loci_indices] = biallelic_mask

        n_retained = full_mask.sum()
        if n_retained == 0:
            self.logger.warning("No loci remain after filtering to biallelic.")
            return self.nremover

        t = None if self.nremover.search_mode else int(exclude_heterozygous)
        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)

        return self.nremover

    def filter_monomorphic(self, exclude_heterozygous: bool = False) -> "NRemover2":
        """Filter out monomorphic loci with only one valid allele.

        If `exclude_heterozygous` is True, ambiguity codes are ignored. Otherwise, they contribute each allele once to allele presence.

        Args:
            exclude_heterozygous (bool): If True, heterozygotes are excluded from allele presence.

        Returns:
            NRemover2: The updated NRemover2 object with filtered loci.
        """
        self.logger.info(
            f"Filtering out monomorphic loci (exclude_heterozygous={exclude_heterozygous})"
        )
        self.nremover._propagate_chain()

        if not np.any(self.nremover.loci_indices) or not np.any(
            self.nremover.sample_indices
        ):
            self.logger.warning("No data remain before filtering.")
            return self.nremover

        ACGT = ("A", "C", "G", "T")
        base_to_idx = {b: i for i, b in enumerate(ACGT)}
        aln = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)
        n_loci = aln.shape[1]
        allele_presence = np.zeros((4, n_loci), dtype=bool)

        for code, alleles in self.iupac.ambiguous_dna_values.items():
            if code in ("N", "X", "-", ".", "?"):
                continue
            if exclude_heterozygous and len(set(alleles)) > 1:
                continue
            mask = aln == code
            col_mask = mask.any(axis=0)
            for base in alleles:
                if base in base_to_idx:
                    allele_presence[base_to_idx[base]] |= col_mask

        n_alleles = allele_presence.sum(axis=0)
        polymorphic_mask = n_alleles > 1

        full_mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        full_mask[self.nremover.loci_indices] = polymorphic_mask

        n_retained = full_mask.sum()
        if n_retained == 0:
            self.logger.warning("No loci remain after filtering out monomorphic sites.")
            return self.nremover

        t = None if self.nremover.search_mode else int(exclude_heterozygous)
        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)

        return self.nremover

    def filter_singletons(self, exclude_heterozygous: bool = False) -> "NRemover2":
        """Filter out singleton loci (minor allele count == 1, with ≥2 alleles present).

        Homozygous genotypes count as 2. Heterozygotes count as 1 per allele if `exclude_heterozygous=False`.

        Args:
            exclude_heterozygous (bool): If True, heterozygotes are ignored entirely.

        Returns:
            NRemover2: The updated NRemover2 object with filtered loci.
        """
        self.logger.info(
            f"Filtering out singleton loci (exclude_heterozygous={exclude_heterozygous})"
        )
        self.nremover._propagate_chain()

        if not np.any(self.nremover.loci_indices) or not np.any(
            self.nremover.sample_indices
        ):
            self.logger.warning("No data remain in the alignment.")
            return self.nremover

        ACGT = ("A", "C", "G", "T")
        base_to_idx = {b: i for i, b in enumerate(ACGT)}
        aln = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)
        n_loci = aln.shape[1]
        counts = np.zeros((4, n_loci), dtype=int)

        for code, alleles in self.iupac.ambiguous_dna_values.items():
            if code in ("N", "X", "-", ".", "?"):
                continue
            if exclude_heterozygous and len(set(alleles)) > 1:
                continue
            mask = aln == code
            col_sum = mask.sum(axis=0)
            for base in alleles:
                if base in base_to_idx:
                    add = col_sum if len(set(alleles)) > 1 else 2 * col_sum
                    counts[base_to_idx[base]] += add

        nonzero_alleles = (counts > 0).sum(axis=0)
        counts_sorted = np.sort(counts, axis=0)[::-1]
        nonzero = (counts > 0).sum(axis=0)
        singleton_mask = (nonzero == 2) & (counts_sorted[1] == 1)
        retained_mask = ~singleton_mask

        full_mask = np.zeros_like(self.nremover.loci_indices, dtype=bool)
        full_mask[self.nremover.loci_indices] = retained_mask

        n_retained = full_mask.sum()
        if n_retained == 0:
            self.logger.warning("No loci remain after filtering out singletons.")
            return self.nremover

        t = None if self.nremover.search_mode else int(exclude_heterozygous)
        self.nremover._update_loci_indices(full_mask, inspect.stack()[0][3], t)

        return self.nremover

    def random_subset_loci(
        self, size: int | float, seed: int | None = None
    ) -> "NRemover2":
        """Randomly subset loci from the current alignment.

        This method randomly selects a subset of loci (columns) to retain.

        The size can be provided as:
            - an integer (absolute number of loci to keep), or
            - a float (proportion of remaining loci to keep).

        Args:
            size (int | float): If int, must be >0 and ≤ total number of retained loci.
                                If float, must be in the interval (0, 1].
            seed (int | None): Optional seed for reproducibility.

        Returns:
            NRemover2: The updated NRemover2 object with a subset of loci retained.

        Raises:
            ValueError: If size is invalid or results in no loci.
            TypeError: If size is not int or float.
        """

        self.logger.info(f"Randomly subsetting {size} loci.")

        self.nremover._propagate_chain()

        # Current mask for loci that are still active
        current_mask = self.nremover.loci_indices

        if not np.any(current_mask):
            self.logger.warning(
                "No loci remain in alignment. Aborting random subsetting."
            )
            return self.nremover

        total_loci = np.count_nonzero(current_mask)

        # Determine number to keep
        if isinstance(size, int):
            if size <= 0 or size > total_loci:
                if size > total_loci:
                    msg = f"Requested size={size} exceeds total loci={total_loci}."
                    self.logger.warning(msg)
                    return self.nremover
                else:
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

            self.logger.debug(f"Rounded float size={size:.3f} to n_to_keep={n_to_keep}")

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

    def filter_allele_depth(self, min_total_depth: int = 10) -> "NRemover2":
        """Filter loci where total allele depth (AD) across all retained samples is below threshold.

        This method reads the 'AD' FORMAT field from the HDF5-backed VCF attributes and removes loci where the sum of all allele depths across all retained samples is less than `min_total_depth`.

        Args:
            min_total_depth (int): Minimum summed allele depth required to retain a locus.

        Returns:
            NRemover2: The updated NRemover2 object with filtered loci.

        Raises:
            AlignmentFormatError: If the filetype is not VCF.
            KeyError: If the AD field is missing from VCF metadata.
        """
        self.logger.info(f"Filtering loci with summed AD < {min_total_depth}")
        self.nremover._propagate_chain()

        if self.nremover.genotype_data.filetype != "vcf":
            msg = "Allele depth filtering is only available for VCF-formatted input."
            self.logger.error(msg)
            raise exceptions.AlignmentFormatError(msg)

        try:
            with h5py.File(self.nremover.genotype_data.vcf_attributes_fn, "r") as h5:
                # shape: (n_loci, n_samples), dtype: str
                ad_raw = h5["fmt_metadata"]["AD"][:]
        except KeyError:
            msg = "VCF FORMAT field 'AD' is not stored. Enable `store_format_fields=True` when initializing VCFReader."
            self.logger.error(msg)
            raise KeyError("AD")

        # Subset AD to retained loci and samples
        ad_subset = ad_raw[self.nremover.loci_indices][:, self.nremover.sample_indices]

        # Vectorized AD parsing: replace missing with "0,0", split, and sum
        flat_ad = ad_subset.flatten()

        flat_ad_cleaned = []
        for val in flat_ad:
            if isinstance(val, bytes):
                val = val.decode()

            if val in (".", "", "None", None) or val is np.nan:
                flat_ad_cleaned.append("0,0")
            else:
                flat_ad_cleaned.append(val)

        def parse_ad(val):
            if isinstance(val, bytes):
                val = val.decode()
            if val in (".", "", "None", None):
                return 0
            return sum(map(int, val.split(","))) if "," in val else int(val)

        ref_alt_sums = np.array([parse_ad(val) for val in flat_ad_cleaned]).reshape(
            ad_subset.shape
        )

        # Total allele depth across all retained samples per locus
        total_depths = np.sum(ref_alt_sums, axis=1)

        # Mask: loci with total AD ≥ threshold
        mask = total_depths >= min_total_depth

        if not np.any(mask):
            msg = f"No loci remain after allele depth filtering with min_total_depth={min_total_depth}."
            self.logger.warning(msg)
            self._append_global_list(inspect.stack()[0][3], 0, 0.0)
            return self.nremover

        # Map to full-length loci mask
        final_mask = np.copy(self.nremover.loci_indices)
        final_mask[self.nremover.loci_indices] = mask

        t = None if self.nremover.search_mode else min_total_depth
        self.nremover._update_loci_indices(final_mask, inspect.stack()[0][3], t)

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

    def _compute_maf_proportions(
        self, exclude_heterozygous: bool = False
    ) -> npt.NDArray[np.float64]:
        """Compute minor allele frequencies (MAF) for each locus.

        MAF is defined as the frequency of the second most common base among A/C/G/T. Heterozygotes contribute one count to each allele unless excluded.

        Args:
            exclude_heterozygous (bool): If True, ignore heterozygotes entirely.

        Returns:
            np.ndarray: MAF values per locus (shape: [n_active_loci]).
        """
        ACGT = ("A", "C", "G", "T")
        base_to_idx = {b: i for i, b in enumerate(ACGT)}

        aln = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)
        if aln.size == 0:
            return np.zeros(self.nremover.loci_indices.sum(), dtype=float)

        counts = np.zeros((4, aln.shape[1]), dtype=int)

        for code, alleles in self.iupac.ambiguous_dna_values.items():
            if code in ("N", "X", "-", "?", "."):
                continue
            if exclude_heterozygous and len(set(alleles)) > 1:
                continue

            mask = aln == code
            col_sum = mask.sum(axis=0)

            for base in alleles:
                if base in base_to_idx:
                    add = col_sum if len(set(alleles)) > 1 else 2 * col_sum
                    counts[base_to_idx[base]] += add

        totals = counts.sum(axis=0).astype(float)
        sorted_counts = np.sort(counts, axis=0)[::-1]
        maf = np.where(totals == 0, 0.0, sorted_counts[1] / totals)
        maf[(totals == 0) | ((counts > 0).sum(axis=0) < 2)] = 0.0

        return maf

    def _calculate_minor_allele_counts(
        self, exclude_heterozygous: bool = False
    ) -> np.ndarray:
        """Calculate minor allele counts (MAC) per locus.

        A MAC is the count of the second most frequent valid allele (A/C/G/T). Ambiguity codes contribute one count per allele unless excluded.

        Args:
            exclude_heterozygous (bool): Whether to exclude heterozygotes from MAC counting.

        Returns:
            np.ndarray: Minor allele counts per locus.
        """
        ACGT = ("A", "C", "G", "T")
        base_to_idx = {base: i for i, base in enumerate(ACGT)}

        arr = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].astype(str)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return np.zeros(0, dtype=int)

        counts = np.zeros((4, arr.shape[1]), dtype=int)

        for code, alleles in self.iupac.ambiguous_dna_values.items():
            if code in ("N", "X", "-", "?", "."):
                continue
            if exclude_heterozygous and len(set(alleles)) > 1:
                continue

            mask = arr == code
            col_sum = mask.sum(axis=0)

            for base in alleles:
                if base in base_to_idx:
                    add = col_sum if len(set(alleles)) > 1 else 2 * col_sum
                    counts[base_to_idx[base]] += add

        counts_sorted = np.sort(counts, axis=0)[::-1]
        n_nonzero = (counts > 0).sum(axis=0)
        mac = np.where(n_nonzero < 2, 0, counts_sorted[1])

        return mac

    def _compute_mac_counts(
        self, min_count: int, exclude_heterozygous: bool = False
    ) -> np.ndarray:
        """Computes a mask based on minor allele counts (MAC) for each locus.

                Args:
                    min_count (int): The minimum minor allele count required to keep a locus.

                    exclude_heterozygous (bool, optional): Whether to exclude heterozygous sites from the MAC calculation. Defaults to False.

        Returns:
            np.ndarray: A boolean array indicating whether each locus passes the MAC threshold.

        """
        mac_counts: npt.NDArray[np.float64 | np.int64] = (
            self._calculate_minor_allele_counts(
                exclude_heterozygous=exclude_heterozygous
            )
        )
        return mac_counts >= min_count
