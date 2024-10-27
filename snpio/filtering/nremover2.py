# Standard library imports
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

# Custom imports
from snpio.filtering.filtering_helper import FilteringHelper
from snpio.filtering.filtering_methods import FilteringMethods
from snpio.plotting.plotting import Plotting
from snpio.utils.logging import LoggerManager


class NRemover2:
    """A class for filtering alignments based on various criteria.

    The class can filter out sequences (samples) and loci (columns) that exceed a missing data threshold (per-column and per-population), minor allele frequency, minor allele count, and other criteria. It can also filter out monomorphic sites, singletons, and loci with more than two alleles. Finally, it can removed all but one linked locus, thin out loci within a specified distance of each other, and randomly subset loci in the SNP dataset. The class provides a flexible and extensible framework for filtering genetic data alignments based on user-defined criteria. It can be used to clean up SNP datasets, remove low-quality loci, and prepare data for downstream analyses.

    Note:
        NRemover2 handles the following characters as missing data:
            - 'N'
            - '-'
            - '?'
            - '.'

        Thus, it treats gaps as missing data. Please keep this in mind when using NRemover2.

        The class is designed to be used with the GenotypeData class, which contains the genetic data alignment, population map, and populations (if a popmap is provided).

        The filtering classes use either a threshold or a boolean value to determine whether to consider heterozygous gentoypes in the filtering logic. The following methods use the ``exclude_heterozygous`` parameter:

            - filter_monomorphic
            - filter_singletons
            - filter_biallelic

        If ``exclude_heterozygous`` is set to ``True``, the filtering methods will exclude heterozygous genotypes from the filtering logic. If set to ``False`` (default), heterozygous genotypes will be included in the filtering logic.

        The class can be used to search for optimal filtering thresholds by plotting the proportion of missing data against the filtering thresholds. The ``search_thresholds()`` method can be used to search across various combinations of filtering thresholds and plot the results.

        The class can also be used to thin out loci within a specified distance of each other using the ``thin_loci`` method.

        The class can be used to randomly subset the loci (columns) in the SNP dataset using the ``random_subset_loci`` method.

        The class can be used to filter out linked loci using the VCF file CHROM field using the ``filter_linked`` method.

        The class can be used to plot a Sankey diagram showing the number of loci removed at each filtering step using the ``plot_sankey_filtering_report`` method.

        The class can be used to print a summary of the filtering results using the ``print_filtering_report`` method.

        The class can be used to filter out monomorphic sites using the ``filter_monomorphic`` method.

        The class can be used to filter out loci (columns) where the only variant is a singleton using the ``filter_singletons`` method.

        The class can be used to filter out loci (columns) that have more than 2 alleles using the ``filter_biallelic`` method.

        The class can be used to filter out loci (columns) where the minor allele frequency is below the threshold using the ``filter_maf`` method.

        The class can be used to filter out loci (columns) where the minor allele count is below the threshold using the ``filter_mac`` method.

        The class can be used to filter out loci (columns) from the alignment that have more than a given proportion of missing data using the ``filter_missing`` method.

        The class can be used to filter out sequences from the alignment that have more than a given proportion of missing data using the ``filter_missing_sample`` method.

        The class can be used to filter out loci (columns) from the alignment that have more than a given proportion of missing data in a specific population using the ``filter_missing_pop`` method.

    Example:
        >>> from snpio import VCFReader
        >>>
        >>> # Specify the genetic data from a VCF file
        >>> vcf_file = "snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz"
        >>>
        >>> # Specify the population map file
        >>> popmap_file = "snpio/example_data/popmaps/phylogen_nomx.popmap"
        >>>
        >>> # Read the genetic data from the VCF file
        >>>
        >>> gd = VCFReader(filename=vcf_file, popmapfile=popmap_file)
        >>>
        >>> # Initialize the NRemover2 class with the GenotypeData instance
        >>> nrm = NRemover2(gd)
        >>>
        >>> # Filter samples and loci.
        >>> nrm.filter_missing_sample(0.75).filter_.filter_missing(0.75).filter_missing_pop(0.75).filter_mac(2).filter_monomorphic(exclude_heterozygous=False).filter_singletons(exclude_heterozygous=False).filter_biallelic(exclude_heterozygous=False).resolve()
        >>>
        >>> # Plot the Sankey diagram showing the number of loci removed at each filtering step.
        >>> nrm.plot_sankey_filtering_report()
        >>>
        >>> # Run a threshold search and plot the results.
        >>> nrm.search_thresholds(thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], maf_thresholds=[0.01, 0.05, 0.1], mac_thresholds=[2, 3, 4, 5], filter_order=["filter_missing_sample", "filter_missing", "filter_missing_pop", "filter_maf", "filter_mac", "filter_monomorphic", "filter_singletons", "filter_biallelic"])

    Attributes:
        genotype_data (GenotypeData): An instance of the GenotypeData class.

        filtering_helper (FilteringHelper): An instance of the FilteringHelper class.

        filtering_methods (FilteringMethods): An instance of the FilteringMethods class.

        df_sample_list (List[pd.DataFrame]): A list of DataFrames containing filtering results for samples.

        df_global_list (List[pd.DataFrame]): A list of DataFrames containing global filtering results.

        _chain_active (bool): A boolean flag indicating whether a filtering chain is active.

        _chain_resolved (bool): A boolean flag indicating whether a filtering chain has been resolved.

        _search_mode (bool): A boolean flag indicating whether the search mode is active.

        debug (bool): A boolean flag indicating whether to enable debug mode.

        verbose (bool): A boolean flag indicating whether to enable verbose mode.

        logger (Logger): An instance of the Logger class for logging messages.

        alignment (np.ndarray): The input alignment to filter.

        populations (List[str]): The population for each sequence in the alignment.

        samples (List[str]): The sample IDs for each sequence in the alignment.

        prefix (str): The prefix for the output files.

        popmap (Dict[str, Union[str, int]]): A dictionary mapping sample IDs to population names.

        popmap_inverse (Dict[Union[str, int], List[str]]): A dictionary mapping population names to lists of sample IDs.

        sample_indices (np.ndarray): A boolean array indicating which samples to keep.

        loci_indices (np.ndarray): A boolean array indicating which loci to keep.

        loci_removed_per_step (Dict[str, Tuple[int, int]]): A dictionary tracking the number of loci removed at each filtering step.

        samples_removed_per_step (Dict[str, Tuple[int, int]]): A dictionary tracking the number of samples removed at each filtering step.

        kept_per_step (Dict[str, Tuple[int, float]]): A dictionary tracking the number of loci or samples kept at each filtering step.

        step_index (int): The current step index in the filtering process.

        current_threshold (float): The current threshold for missing data.

        original_loci_count (int): The original number of loci in the alignment.

        original_sample_count (int): The original number of samples in the alignment.

        original_loci_indices (np.ndarray): A boolean array indicating the original loci indices.

        original_sample_indices (np.ndarray): A boolean array indicating the original sample indices.

    Methods:
        filter_missing: Filters out sequences from the alignment that have more than a given proportion of missing data.

        filter_missing_pop: Filters out sequences from the alignment that have more than a given proportion of missing data in a specific population.

        filter_missing_sample: Filters out samples from the alignment that have more than a given proportion of missing data.

        filter_maf: Filters out loci (columns) where the minor allele frequency is below the threshold.

        filter_monomorphic: Filters out monomorphic sites.

        filter_singletons: Filters out loci (columns) where the only variant is a singleton.

        filter_biallelic: Filter out loci (columns) that have more than 2 alleles.

        filter_linked: Filter out linked loci using VCF file CHROM field.

        thin_loci: Thin out loci within a specified distance of each other.

        random_subset_loci: Randomly subset the loci (columns) in the SNP dataset.

        search_thresholds: Plots the proportion of missing data against the filtering thresholds.

        plot_sankey_filtering_report: Makes a Sankey plot showing the number of loci removed at each filtering step.

        print_filtering_report: Prints a summary of the filtering results.

        resolve: Finalizes the method chain and returns the updated GenotypeData instance.

        __repr__: Returns a string representation of the NRemover2 instance.

        __str__: Returns a string representation of the NRemover2 instance.

        __getattr__: Custom attribute access method that handles delegating calls to FilteringMethods or FilteringHelper.
    """

    def __init__(self, genotype_data: Any) -> None:
        """Initializes the NRemover2 class.

        Args:
            genotype_data (GenotypeData): An instance of the GenotypeData class containing the genetic data alignment, population map,  populations, and other relevant data.
        """
        self.genotype_data = genotype_data
        self.popmap: Dict[str, Union[str, int]] = genotype_data.popmap
        self.popmap_inverse: Dict[Union[str, int], List[str]] = (
            genotype_data.popmap_inverse
        )

        if self.popmap_inverse is not None:
            self.populations: List[Union[str, int]] = list(self.popmap_inverse.keys())
            if not all(isinstance(pop, str) for pop in self.populations):
                self.populations = [str(pop) for pop in self.populations]

        self.samples: List[str] = genotype_data.samples
        self.prefix: str = genotype_data.prefix

        # Store the original alignment and indices as backups
        self.original_alignment: np.ndarray = np.copy(self.genotype_data.snp_data)

        self.current_threshold: float = 0.0

        self.original_loci_count = genotype_data.snp_data.shape[1]
        self.original_sample_count = genotype_data.snp_data.shape[0]
        self.original_loci_indices: np.ndarray = np.ones(
            self.original_loci_count, dtype=bool
        )
        self.original_sample_indices: np.ndarray = np.ones(
            self.original_sample_count, dtype=bool
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
            kwargs = {"prefix": self.prefix, "debug": self.debug}
            kwargs["verbose"] = self.verbose
            logman = LoggerManager(__name__, **kwargs)
            self.logger = logman.get_logger()
        else:
            self.logger = self.genotype_data.logger

        # Initialize the filtering helper and methods
        self.filtering_helper = FilteringHelper(self)
        self.filtering_methods = FilteringMethods(self)

    def _reset_filtering_state(self) -> None:
        """Resets the filtering state of the alignment and indices to their original state."""
        self.logger.debug("Resetting alignment and indices to original state.")
        self.alignment = self.original_alignment.copy()
        self.loci_indices = self.original_loci_indices.copy()
        self.sample_indices = self.original_sample_indices.copy()
        self.step_index = 0
        self.loci_removed_per_step = {}
        self.samples_removed_per_step = {}
        self.kept_per_step = {}

    def _reset_filtering_results(self) -> None:
        """Resets the filtering results (i.e., DataFrame lists) for future filtering steps."""
        self.logger.debug("Resetting filtering results.")
        self.df_global_list.clear()
        self.df_sample_list.clear()

    def _get_filter_methods(
        self,
        threshold: float,
        maf_threshold: float,
        bool_threshold: bool,
        mac_threshold: int,
    ) -> Dict[str, Tuple[Callable, Tuple[float, float, bool, int]]]:
        """Retrieves the filter methods based on provided thresholds.

        Args:
            threshold (float): The threshold for missing data.
            maf_threshold (float): The threshold for minor allele frequency.
            bool_threshold (bool): A boolean threshold for filtering methods.
            mac_threshold (int): The minimum allele count threshold.

        Returns:
            Dict[str, Tuple[Callable, Tuple[float, float, bool, int]]]: A dictionary mapping method names to their corresponding callable and thresholds.
        """
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
        """Validates the filter order or sets the default order if none is provided.

        Args:
            filter_order (Optional[List[str]]): A list of filter methods in order of application.
            filter_methods (Dict[str, Callable]): A dictionary of available filtering methods.

        Returns:
            List[str]: The validated or default filter order.

        Raises:
            ValueError: If an unknown method is found in the filter order.
        """
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sets the ranges of thresholds for filtering.

        Args:
            thresholds (Optional[Union[List[float], np.ndarray]]): Thresholds for missing data (0.0 < threshold ≤ 1.0).
            maf_thresholds (Optional[Union[List[float], np.ndarray]]): Thresholds for minor allele frequency (0.0 ≤ maf < 1.0).
            mac_thresholds (Optional[Union[List[int], np.ndarray]]): Thresholds for minimum allele count (MAC > 1).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The validated thresholds, maf thresholds, boolean array, and mac thresholds.

        Raises:
            ValueError: If any of the provided thresholds are invalid.
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, num=9, endpoint=True, dtype=float)
        else:
            if not all([0.0 < t <= 1.0 for t in thresholds]):
                msg = f"Invalid missing data threshold provided. Thresholds must be between 0.0 and 1.0, but got: {thresholds}"
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
        """Applies the specified filtering methods in sequence.

        Args:
            filter_methods (Dict[str, Tuple[Callable, Optional[Union[np.ndarray, bool]]]]): A dictionary of filter methods and their corresponding thresholds.
        """
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

    def _plot_results(self, df_combined: pd.DataFrame) -> None:
        """Plot and save the filtering results.

        Args:
            df_combined (pd.DataFrame): Combined DataFrame containing filtering results to plot.
        """
        self.logger.info("Plotting filtering results...")

        kwargs = self.genotype_data.plot_kwargs
        plotting = Plotting(self.genotype_data, **kwargs)

        # Call the plot_search_results method from the Plotting class
        plotting.plot_search_results(df_combined=df_combined)

    def plot_sankey_filtering_report(self) -> None:
        """Plots a Sankey diagram showing the number of loci removed at each filtering step.

        This method generates a Sankey diagram showing the number of loci removed at each filtering step. The diagram is saved as a PNG file in the output directory. The Sankey diagram provides a visual representation of the filtering process, showing the number of loci removed at each step and the proportion of loci removed relative to the total number of loci in the alignment. It also shows the number of loci that were retained at each step.

        Returns:
            None

        Raises:
            RuntimeError: If the filtering chain has not been resolved.

        Note:
            The Sankey diagram is generated using the Plotting class.

            The Sankey diagram is saved as PNG and HTML files in the output directory.

            The Sankey diagram shows the number of loci removed at each filtering step and the proportion of loci removed relative to the total number of loci in the alignment.

            The Sankey diagram also shows the number of loci that were retained at each step.

            The Sankey diagram provides a visual representation of the filtering process, making it easier to understand the impact of each filtering step on the alignment.

            The Sankey diagram is useful for visualizing the filtering process and identifying the most effective filtering steps.

        Example:
            To plot the Sankey diagram showing the number of loci removed at each filtering step, use the following code:

            >>> gd = GenotypeData("snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz", "snpio/example_data/popmaps/phylogen_nomx.popmap")
            >>> nrm = NRemover2(gd)
            >>> nrm.filter_missing_sample(0.75).filter_missing(0.75).filter_missing_pop(0.75).filter_mac(2).filter_monomorphic(exclude_heterozygous=False).filter_singletons(exclude_heterozygous=False).filter_biallelic(exclude_heterozygous=False).resolve()
            >>> nrm.plot_sankey_filtering_report()
            >>> # The Sankey diagram will be saved as a PNG file in the output directory.
        """
        self.logger.info("Plotting Sankey filtering report...")

        kwargs = self.genotype_data.plot_kwargs
        plotting = Plotting(self.genotype_data, **kwargs)

        df = pd.concat(self.df_global_list)
        plotting.plot_sankey_filtering_report(df, search_mode=self.search_mode)

    def propagate_chain(self) -> None:
        """Propagates the filtering chain to the next step, marking the chain as active.

        Raises:
            RuntimeError: If the filtering chain has not been resolved.

        Note:
            - This method is used to propagate the filtering chain to the next step in the filtering process.
            - It marks the chain as active, allowing further filtering steps to be applied.
            - The chain must be resolved before starting a new chain.
            - The chain is resolved by calling the resolve() method.
        """
        self._chain_active = True

    def _start_new_filter_chain(self) -> None:
        """Initializes a new filtering chain by resetting the filtering state.

        Raises:
            RuntimeError: If a previous filtering chain has not been resolved.
        """
        if not self._chain_resolved:
            msg = "The previous filtering chain was not resolved. Please call resolve() before starting a new chain."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Proceed with filtering
        self._reset_filtering_state()
        self._chain_resolved = False

    def _update_loci_indices(
        self,
        included_indices: np.ndarray,
        method_name: str,
        threshold: Optional[float] = None,
    ) -> None:
        """Updates the loci indices based on the filtering results and records the step.

        This method updates the loci indices based on the filtering results and records the step in the filtering results DataFrame. It also updates the loci_removed_per_step dictionary to track the number of loci removed at each filtering step.

        Args:
            included_indices (np.ndarray): Boolean array indicating which loci are included.
            method_name (str): The name of the filtering method used.
            threshold (Optional[float]): The threshold value used for filtering.

        Returns:
            None

        Raises:
            ValueError: If the method_name is not recognized

        Note:
            - This method updates the loci indices based on the filtering results and records the step in the filtering results DataFrame.
            - It also updates the loci_removed_per_step dictionary to track the number of loci removed at each filtering step.
            - The method_name parameter specifies the filtering method used, such as "filter_missing", "filter_maf", "filter_mac", "filter_monomorphic", "filter_singletons", or "filter_biallelic".
            - The threshold parameter specifies the threshold value used for filtering.
            - The included_indices parameter is a boolean array indicating which loci are included after filtering.
        """
        self.loci_indices, self.loci_removed_per_step, self.df_global_list = (
            self._record_filtering_step(
                self.loci_removed_per_step,
                included_indices,
                self.df_global_list,
                method_name,
                threshold,
            )
        )

    def _update_sample_indices(
        self,
        included_indices: np.ndarray,
        method_name: str,
        threshold: Optional[float] = None,
    ) -> None:
        """Updates the sample indices based on the filtering results and records the step.

        This method updates the sample indices based on the filtering results and records the step in the filtering results DataFrame. It also updates the samples_removed_per_step dictionary to track the number of samples removed at each filtering step.

        Args:
            included_indices (np.ndarray): Boolean array indicating which samples are included.
            method_name (str): The name of the filtering method used.
            threshold (Optional[float]): The threshold value used for filtering.

        Raises:
            ValueError: If the method_name is not recognized

        Note:
            - This method updates the sample indices based on the filtering results and records the step in the filtering results DataFrame.
            - It also updates the samples_removed_per_step dictionary to track the number of samples removed at each filtering step.
            - The method_name parameter specifies the filtering method used, such as "filter_missing_sample", "filter_missing", "filter_missing_pop", "filter_maf", "filter_mac", "filter_monomorphic", "filter_singletons", or "filter_biallelic".
            - The threshold parameter specifies the threshold value used for filtering.
            - The included_indices parameter is a boolean array indicating which samples are included after filtering.
        """
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
        removed_per_step: Dict[str, Tuple[int, int]],
        included_indices: np.ndarray,
        current_list: List[pd.DataFrame],
        method_name: str,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[pd.DataFrame]]:
        """Records the results of a filtering step and updates the indices.

        This method records the results of a filtering step, updates the indices, and appends the results to the filtering results DataFrame. It is used to track the number of loci or samples removed at each filtering step.

        Args:
            removed_per_step (Dict[str, Tuple[int, int]]): Dictionary tracking the number of removed loci or samples per step.
            included_indices (np.ndarray): Boolean array indicating which loci or samples are included after filtering.
            current_list (List[pd.DataFrame]): List to append filtering results DataFrames.
            method_name (str): The name of the filtering method used.
            threshold (Optional[float]): The threshold value used for filtering.

        Returns:
            Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[pd.DataFrame]]: Updated indices, removed per step, and the current list of DataFrames.

        Raises:
            ValueError: If the method_name is not recognized

        Note:
            - This method records the results of a filtering step, updates the indices, and appends the results to the filtering results DataFrame.
            - It is used to track the number of loci or samples removed at each filtering step.
            - The removed_per_step dictionary tracks the number of loci or samples removed at each filtering step.
            - The included_indices parameter is a boolean array indicating which loci or samples are included after filtering.
            - The method_name parameter specifies the filtering method used, such as "filter_missing", "filter_maf", "filter_mac", "filter_monomorphic", "filter_singletons", or "filter_biallelic".
            - The threshold parameter specifies the threshold value used for filtering.
        """
        all_indices = (
            self.loci_indices.copy()
            if method_name != "filter_missing_sample"
            else self.sample_indices.copy()
        )
        total_input = np.count_nonzero(all_indices)
        n_to_keep = np.count_nonzero(included_indices & all_indices)
        n_removed = total_input - n_to_keep  # number removed at this step

        # Append this information to the filtering results DataFrame
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

    def resolve(self) -> Any:
        """Resolve the method chain and finalize the filtering process.

        This method resolves the method chain and finalizes the filtering process. It applies the selected filters to the alignment, updates the alignment, sample indices, and loci indices based on the filtering results, and resets the chain active flag. It returns the updated GenotypeData instance after filtering has been applied.

        Returns:
            GenotypeData: The updated GenotypeData instance after filtering has been applied.

        Note:
            - This method is used to finalize the method chain by applying the selected filters to the alignment.
            - It updates the alignment, sample indices, and loci indices based on the filtering results.
            - It also resets the chain active flag and returns the updated GenotypeData instance after filtering has been applied.
        """
        gd = self._finalize_chain()  # Finalizes the alignment
        self._chain_resolved = True  # Mark the chain as resolved
        return gd

    def _finalize_chain(self) -> None:
        """Finalizes the method chain by applying the selected filters to the alignment.

        This method finalizes the method chain by applying the selected filters to the alignment. It updates the alignment, sample indices, and loci indices based on the filtering results. It also resets the chain active flag and returns the updated GenotypeData instance after filtering has been applied.

        Raises:
            ValueError: If no samples or loci remain after filtering.
        """
        if not self.search_mode and self.verbose:
            self.filtering_helper.print_filtering_report()

        if not self._chain_active:
            msg = "No active chain to finalize."
            self.logger.error(msg)
            raise RuntimeError(msg)

        si = self.sample_indices.astype(bool)
        li = self.loci_indices.astype(bool)

        self.logger.debug(f"Search Mode: {self.search_mode}")
        self.logger.debug(f"Sample Indices: {si}")
        self.logger.debug(f"Loci Indices: {li}")

        if not np.any(si) or si.size == 0:
            if self.search_mode:
                msg = f"No samples left after filtering at threshold: {self.current_threshold}."
                self.logger.info(msg)
                self._chain_active = False
                return
            else:
                self.logger.error(msg)
                raise ValueError(msg)

        if not np.any(li) or li.size == 0:
            if self.search_mode:
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
    def search_mode(self) -> bool:
        """Gets the current search mode status.

        Returns:
            bool: The status of search mode.
        """
        return self._search_mode

    @search_mode.setter
    def search_mode(self, value: bool) -> None:
        """Sets the search mode status.

        Args:
            value (bool): The new status of search mode.
        """
        self._search_mode = value

    @property
    def loci_indices(self) -> np.ndarray:
        """Gets the current loci_indices.

        Returns:
            np.ndarray: The boolean array indicating which loci to keep.
        """
        return self._loci_indices

    @loci_indices.setter
    def loci_indices(self, value: np.ndarray) -> None:
        """Sets loci_indices.

        Args:
            value (np.ndarray): The new boolean array indicating which loci to keep.
        """
        self._loci_indices = value

    @property
    def sample_indices(self) -> np.ndarray:
        """Gets the current sample_indices.

        Returns:
            np.ndarray: The boolean array indicating which samples to keep.
        """
        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, value: np.ndarray) -> None:
        """Sets sample_indices.

        Args:
            value (np.ndarray): The new boolean array indicating which samples to keep.
        """
        self._sample_indices = value

    def __getattr__(self, name: str) -> Union[Callable, None]:
        """
        Custom attribute access method that handles delegating calls to FilteringMethods or FilteringHelper.

        This method allows for transparent access to the filtering methods and helper classes from NRemover2.

        Args:
            name (str): The name of the attribute to access.

        Returns:
            Union[Callable, None]: The requested attribute, or None if not found.

        Raises:
            AttributeError: If the attribute is not found in either NRemover2 or the filtering methods or helper.

        Note:
            - This method allows for transparent access to the filtering methods and helper classes from NRemover2.
            - It delegates attribute access to the FilteringMethods and FilteringHelper classes if the attribute is not found in NRemover2.
            - This method is used to provide a more user-friendly interface for accessing the filtering methods and helper classes.
            - It allows users to access the filtering methods and helper classes directly from the NRemover2 instance without having to access them through the filtering_methods or filtering_helper attributes.
            - If the attribute is not found in NRemover2, FilteringMethods, or FilteringHelper, an AttributeError is raised.
        """
        # Ensure filtering_methods is initialized before accessing it
        if name in ["filtering_methods", "filtering_helper"]:
            if name == "filtering_methods" and "filtering_methods" not in self.__dict__:
                self.filtering_methods = FilteringMethods(self)
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

    def __repr__(self) -> str:
        """Returns a string representation of the NRemover2 instance.

        Returns:
            str: A string representation of the NRemover2 instance.
        """
        return f"NRemover2(genotype_data={self.genotype_data})"

    def __str__(self) -> str:
        """Returns a string representation of the NRemover2 instance.

        Returns:
            str: A string representation of the NRemover2 instance.
        """
        return f"NRemover2(genotype_data={self.genotype_data})"
