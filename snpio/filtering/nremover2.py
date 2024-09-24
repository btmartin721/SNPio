# Standard library imports
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

from snpio.filtering.filtering_helper import FilteringHelper
from snpio.filtering.filtering_methods import FilteringMethods
from snpio.plotting.plotting import Plotting

# Custom imports
from snpio.utils.logging import setup_logger


class NRemover2:
    """
    A class for filtering alignments based on the proportion of missing data in a genetic alignment, by minor allele frequency, and by linked loci.

    The class can filter out sequences (samples) and loci (columns) that exceed a missing data threshold.

    The loci can be filtered by global missing data proportions or if any given population exceeds the missing data threshold.

    A number of informative plots are also generated.

    Note:
        NRemover2 handles the following characters as missing data:
            - 'N'
            - '-'
            - '?'
            - '.'

        Thus, it treats gaps as missing data. Please keep this in mind when using NRemover2.

    Args:
        genotype_data (GenotypeData): An instance of the GenotypeData class containing the genetic data alignment, population map, and populations.

    Attributes:
        alignment (list of Bio.SeqRecord.SeqRecord): The input alignment to filter.

        populations (list of str): The population for each sequence in the alignment.

        samples (list): The sample IDs for each sequence in the alignment.

        prefix (str): The prefix for the output files.

        popmap (dict): A dictionary mapping sample IDs to population names.

        popmap_inverse (dict): A dictionary mapping population names to lists of sample IDs.

        sample_indices (np.ndarray): A boolean array indicating which samples to keep.

        loci_indices (np.ndarray): A boolean array indicating which loci to keep.

        loci_removed_per_step (dict): A dictionary tracking the number of loci removed at each filtering step.

    Methods:
        filter_missing: Filters out sequences from the alignment that have more than a given proportion of missing data.

        filter_missing_pop: Filters out sequences from the alignment that have more than a given proportion of missing data in a specific population.

        filter_missing_sample: Filters out samples from the alignment that have more than a given proportion of missing data.

        filter_maf: Filters out loci (columns) where the minor allele frequency is below the threshold.

        filter_monomorphic: Filters out monomorphic sites.

        filter_singletons: Filters out loci (columns) where the only variant is a singleton.

        filter_biallelic: Filter out loci (columns) that have more than 2 alleles.

        filter_linked: Filter out linked loci using VCF file CHROM field.

        thin_loci: Thin out loci within ``thin`` bases of each other.

        random_subset_loci: Randomly subset the loci (columns) in the SNP dataset.

        search_thresholds: Plots the proportion of missing data against the filtering thresholds.

        plot_sankey_filtering_report: Makes a Sankey plot showing the number of loci removed at each filtering step.

        print_filtering_report: Prints a summary of the filtering results.
    """

    def __init__(self, genotype_data) -> None:
        """A class to filter genetic data based on missing data, minor allele frequency, linked loci, and other criteria.

        Args:
            genotype_data (GenotypeData): An instance of the GenotypeData class containing the genetic data alignment, population map, and populations.
        """
        self.genotype_data = genotype_data
        self.popmap: Dict[str, Union[str, int]] = genotype_data.popmap
        self.popmap_inverse: Dict[Union[str, int], List[str]] = (
            genotype_data.popmap_inverse
        )
        self.populations: List[Union[str, int]] = list(self.popmap_inverse.keys())
        self.samples: List[str] = genotype_data.samples
        self.prefix: str = genotype_data.prefix

        # Store the original alignment and indices as backups
        self.original_alignment: np.ndarray = np.copy(self.genotype_data.snp_data)

        self.current_threshold: float = 0.0

        self.original_loci_count = genotype_data.snp_data.shape[1]
        self.original_sample_count = genotype_data.snp_data.shape[0]
        self.original_loci_indices: np.ndarray = np.ones(self.original_loci_count, bool)
        self.original_sample_indices: np.ndarray = np.ones(
            self.original_sample_count, bool
        )

        self.current_loci_count = self.original_loci_count
        self.current_sample_count = self.original_sample_count

        # Use the originals for the active filtering
        self.alignment: np.ndarray = np.copy(self.original_alignment)
        self.original_alignment_shape: Tuple[int, int] = self.alignment.shape
        self._loci_indices: np.ndarray = np.copy(self.original_loci_indices)
        self._sample_indices: np.ndarray = np.copy(self.original_sample_indices)

        self.step_index: int = 0
        self.loci_removed_per_step: Dict[str, Tuple[int, int]] = {}
        self.samples_removed_per_step: Dict[str, Tuple[int, int]] = {}
        self.kept_per_step: Dict[str, Tuple[int, float]] = {}

        # Initializing other properties
        self.df_sample_list: List[pd.DataFrame] = []
        self.df_global_list: List[pd.DataFrame] = []

        self._chain_active = False
        self._chain_resolved = True  # Initially, no chain has been started

        self._search_mode = False

        self.debug = genotype_data.debug
        self.verbose = genotype_data.verbose

        if self.genotype_data.logger is None:
            self.log_file = Path(f"{self.prefix}_output", "logs", "nremover2.log")
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.logger = setup_logger(
                __name__,
                log_file=self.log_file,
                level="DEBUG" if self.debug else "INFO",
            )
        else:
            self.logger = self.genotype_data.logger

        # Initialize the filtering helper and methods
        self.filtering_helper = FilteringHelper(self)
        self.filtering_methods = FilteringMethods(self)

    def _reset_filtering_state(self):
        """Resets the filtering state of the alignment and indices."""
        self.logger.debug("Resetting alignment and indices to original state.")
        self.alignment = self.original_alignment.copy()
        self.loci_indices = self.original_loci_indices.copy()
        self.sample_indices = self.original_sample_indices.copy()
        self.step_index = 0
        self.loci_removed_per_step = {}
        self.samples_removed_per_step = {}
        self.kept_per_step = {}

    def _reset_filtering_results(self):
        """Resets the filtering results (i.e., DataFrame lists)."""
        self.logger.debug("Resetting filtering results.")
        self.df_global_list = self.df_global_list or []
        self.df_sample_list = self.df_sample_list or []
        self.df_sample_list.clear()
        self.df_global_list.clear()

    def _get_filter_methods(
        self, threshold, maf_threshold, bool_threshold, mac_threshold
    ):
        thresholds = (threshold, maf_threshold, bool_threshold, mac_threshold)

        return {
            "filter_missing_sample": (self.filter_missing_sample, thresholds),
            "filter_missing": (self.filter_missing, thresholds),
            "filter_missing_pop": (self.filter_missing_pop, thresholds),
            "filter_maf": (self.filter_maf, thresholds),
            "filter_mac": (self.filter_mac, thresholds),
            "filter_monomorphic": (self.filter_monomorphic, thresholds),
            "filter_biallelic": (self.filter_biallelic, thresholds),
            "filter_singletons": (self.filter_singletons, thresholds),
        }

    def _validate_filter_order(
        self, filter_order: Optional[List[str]], filter_methods: Dict[str, Callable]
    ) -> List[str]:
        """Validate the filter order or set the default order."""
        if filter_order is None:
            filter_order = [
                "filter_missing_sample",
                "filter_missing",
                "filter_missing_pop",
                "filter_maf",
                "filter_mac",
                "filter_monomorphic",
                "filter_biallelic",
                "filter_singletons",
            ]

        # Validate the filter order to ensure that only valid methods are
        # included
        for step in filter_order:
            if step not in filter_methods:
                msg = f"Unknown step in filter_order: {step}"
                self.logger.error(msg)
                raise ValueError(msg)

        return {k: filter_methods[k] for k in filter_order}

    def _set_threshold_ranges(
        self,
        thresholds: Optional[Union[List[float], np.ndarray]],
        maf_thresholds: Optional[Union[List[float], np.ndarray]],
        mac_thresholds: Optional[Union[List[int], np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """Set the ranges of thresholds for filtering."""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, num=9, endpoint=True, dtype=float)
        else:
            if not all([0.0 < t <= 1.0 for t in thresholds]):
                msg = f"Invalid missing data sthreshold provided. Thresholds must be between 0.0 and 1.0, but got: {thresholds}"
                self.logger.error(msg)
                raise ValueError(msg)

        if maf_thresholds is None:
            maf_thresholds = np.array([0.01, 0.05, 0.075, 0.1, 0.15, 0.2], dtype=float)
        else:
            if not all([0.0 <= t < 1.0 for t in maf_thresholds]):
                msg = "Invalid MAF threshold provided. MAF thresholds must be between 0.0 and 1.0."
                self.logger.error(msg)
                raise ValueError(msg)

        if mac_thresholds is None:
            mac_thresholds = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
        else:
            if not all([t > 1 for t in mac_thresholds]):
                msg = "Invalid MAC threshold provided. MAC thresholds must be greater than 1."
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.debug(f"Max Thresholds: {thresholds}")
        self.logger.debug(f"MAF Thresholds: {maf_thresholds}")
        self.logger.debug(f"MAC Thresholds: {mac_thresholds}")

        return (
            thresholds,
            maf_thresholds,
            np.array([True, False], dtype=bool),
            mac_thresholds,
        )

    def _apply_filtering_methods(
        self,
        filter_methods: Dict[str, Tuple[Callable, Optional[Union[np.ndarray, bool]]]],
    ) -> None:
        self.propagate_chain()
        for method_name, (method, thresholds) in filter_methods.items():
            self.current_thresholds = thresholds

            if method_name in {
                "filter_missing_sample",
                "filter_missing_pop",
                "filter_missing",
            }:
                threshold = thresholds[0]
            elif method_name == "filter_maf":
                threshold = thresholds[1]
            elif method_name in {
                "filter_singletons",
                "filter_biallelic",
                "filter_monomorphic",
            }:
                threshold = thresholds[2]
            elif method_name == "filter_mac":
                threshold = thresholds[3]
            else:
                msg = f"Unknown method: {method_name}"
                self.logger.error(msg)
                raise ValueError(msg)

            try:
                method(threshold)
            except Exception as e:
                msg = f"Error applying filter {method_name}: {e}"
                self.logger.error(msg)
                raise e

            self.step_index += 1
        self.resolve()

    def _plot_results(self, df_combined):
        """Plot and save the filtering results."""

        if self.verbose:
            self.logger.info("Plotting filtering results...")

        kwargs = self.genotype_data.plot_kwargs
        plotting = Plotting(self.genotype_data, **kwargs)

        # Call the plot_search_results method from the Plotting class
        plotting.plot_search_results(df_combined=df_combined)

    def plot_sankey_filtering_report(self):
        """Plot a Sankey diagram showing the number of loci removed at each filtering step."""

        if self.verbose:
            self.logger.info("Plotting Sankey filtering report...")

        kwargs = self.genotype_data.plot_kwargs
        plotting = Plotting(self.genotype_data, **kwargs)

        df = pd.concat(self.df_global_list)
        plotting.plot_sankey_filtering_report(df, search_mode=self.search_mode)

    def propagate_chain(self):
        """Propagate the filtering chain to the next step."""
        self._chain_active = True

    def _start_new_filter_chain(self):
        if not self._chain_resolved:
            msg = "The previous filtering chain was not resolved. Please call resolve() before starting a new chain."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Proceed with filtering
        self._reset_filtering_state()
        self._chain_resolved = False

    def _update_loci_indices(self, included_indices, method_name, threshold=None):
        self.loci_indices, self.loci_removed_per_step, self.df_global_list = (
            self._record_filtering_step(
                self.loci_removed_per_step,
                included_indices,
                self.df_global_list,
                method_name,
                threshold,
            )
        )

    def _update_sample_indices(self, included_indices, method_name, threshold=None):
        self.sample_indices, self.samples_removed_per_step, self.df_sample_list = (
            self._record_filtering_step(
                self.samples_removed_per_step,
                included_indices,
                self.df_sample_list,
                method_name,
                threshold,
            )
        )

    def _record_filtering_step(
        self,
        removed_per_step,
        included_indices,
        current_list,
        method_name,
        threshold=None,
    ):

        all_indices = (
            self.loci_indices.copy()
            if method_name != "filter_missing_sample"
            else self.sample_indices.copy()
        )
        total_input = np.count_nonzero(all_indices)
        n_to_keep = np.count_nonzero(included_indices & all_indices)
        n_removed = total_input - n_to_keep  # number removed at this step

        # Append this information to the filtering results dataframe
        current_list.append(
            pd.DataFrame(
                {
                    "Step": self.step_index,
                    "Filter_Method": method_name,
                    "Missing_Threshold": (
                        self.current_thresholds[0] if threshold is None else threshold
                    ),
                    "MAF_Threshold": (
                        self.current_thresholds[1] if threshold is None else threshold
                    ),
                    "Bool_Threshold": (
                        self.current_thresholds[2] if threshold is None else threshold
                    ),
                    "MAC_Threshold": (
                        self.current_thresholds[3] if threshold is None else threshold
                    ),
                    "Removed_Count": n_removed,
                    "Removed_Prop": (
                        np.round(n_removed / total_input, 2) if total_input > 0 else 0
                    ),
                    "Kept_Count": n_to_keep,
                    "Kept_Prop": (
                        np.round(n_to_keep / total_input, 2) if total_input > 0 else 0
                    ),
                    "Total_Loci": total_input,
                },
                index=[0],
            )
        )

        all_indices[~included_indices] = False

        return all_indices.copy(), removed_per_step, current_list

    def __getattr__(self, name):
        """
        Custom attribute access method that handles delegating calls to FilteringMethods or FilteringHelper.
        """
        # Ensure filtering_methods is initialized before accessing it
        if name in ["filtering_methods", "filtering_helper"]:
            if name == "filtering_methods" and "filtering_methods" not in self.__dict__:
                self.filtering_methods = FilteringMethods(
                    self
                )  # Initialize if not present
            if name == "filtering_helper" and "filtering_helper" not in self.__dict__:
                self.filtering_helper = FilteringHelper(self)
            return self.__dict__[name]

        # Try accessing the attribute directly from NRemover2
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        # Safely access filtering_methods and filtering_helper attributes
        if "filtering_methods" in self.__dict__ and hasattr(
            self.filtering_methods, name
        ):
            return getattr(self.filtering_methods, name)

        if "filtering_helper" in self.__dict__ and hasattr(self.filtering_helper, name):
            return getattr(self.filtering_helper, name)

        # Raise AttributeError if the attribute is not found
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def resolve(self, threshold=None):
        """
        Resolve the method chain and finalize the filtering process.
        """
        gd = self._finalize_chain()  # Finalizes the alignment
        self._chain_resolved = True  # Mark the chain as resolved
        return gd

    def _finalize_chain(self):
        """
        This method is called when the method chain is completed.
        It sets the alignment attribute automatically.
        """
        self.filtering_helper.print_filtering_report()

        if not self._chain_active:
            msg = "No active chain to finalize."
            self.logger.warning(msg)
            raise RuntimeError(msg)

        si = self.sample_indices.astype(bool)
        li = self.loci_indices.astype(bool)

        self.logger.debug(f"Search Mode: {self.search_mode}")
        self.logger.debug(f"Sample Indices: {si}")
        self.logger.debug(f"Loci Indices: {li}")

        if not np.any(si) or si.size == 0:
            if self.search_mode:
                if self.verbose:
                    msg = f"No samples left after filtering at threshold: {self.current_threshold}."
                    self.logger.info(msg)
                self._chain_active = False
                return
            else:
                self.logger.error(msg)
                raise ValueError(msg)

        if not np.any(li) or li.size == 0:
            if self.search_mode:
                if self.verbose:
                    msg = f"No loci left after filtering at {self.current_threshold}."
                    self.logger.info(msg)
                self._chain_active = False
                return
            else:
                msg = "No loci left after filtering."
                self.logger.error(msg)
                raise ValueError(msg)

        if self.search_mode:
            self._chain_active = False
            return

        self.alignment = self.alignment[si, :][:, li].copy()

        self.samples = (
            np.array(self.samples)[si].tolist()
            if isinstance(self.samples, list)
            else self.samples[si].copy()
        )

        # Update the genotype data
        self.genotype_data.set_alignment(self.alignment, self.samples, si, li)

        # Reset the chain active flag
        self._chain_active = False

        return self.genotype_data

    @property
    def search_mode(self):
        return self._search_mode

    @search_mode.setter
    def search_mode(self, value):
        self._search_mode = value

    @property
    def loci_indices(self):
        return self._loci_indices

    @loci_indices.setter
    def loci_indices(self, value):
        self._loci_indices = value

    @property
    def sample_indices(self):
        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, value):
        self._sample_indices = value
