# Standard library imports
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Custom imports
import snpio.utils.custom_exceptions as exceptions
from snpio.filtering.filtering_helper import FilteringHelper
from snpio.filtering.filtering_methods import FilteringMethods
from snpio.plotting.plotting import Plotting
from snpio.utils.logging import LoggerManager

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class NRemover2:
    """A class for filtering alignments based on various criteria.

    The class provides various methods for filtering genetic data alignments based on user-defined criteria. These include filtering sequences (samples) and loci (columns) by missing data thresholds, minor allele frequency, minor allele count, and other criteria. It can also remove monomorphic sites, singletons, loci with more than two alleles, and linked loci. Additional functionality includes thinning loci within a specified distance, randomly subsetting loci, and plotting filtering results.

        Key features:
            - Search for optimal filtering thresholds and plot results using `search_thresholds`.
            - Thin loci within a specified distance using `thin_loci`.
            - Randomly subset loci using `random_subset_loci`.
            - Filter linked loci using the VCF CHROM field with `filter_linked`.
            - Plot a Sankey diagram of loci removed at each step using `plot_sankey_filtering_report`.
            - Print a summary of filtering results with `print_filtering_report`.
            - Remove monomorphic sites using `filter_monomorphic`.
            - Remove loci where the only variant is a singleton using `filter_singletons`.
            - Remove loci with more than two alleles using `filter_biallelic`.
            - Filter loci by minor allele frequency (`filter_maf`) or count (`filter_mac`).
            - Filter loci with excessive missing data using `filter_missing`.
            - Filter samples with excessive missing data using `filter_missing_sample`.
            - Filter loci with excessive missing data in specific populations using `filter_missing_pop`.

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
        >>> nrm.filter_missing_sample(0.75)
                .filter_missing(0.75)
                .filter_missing_pop(0.75)
                .filter_mac(2)
                .filter_monomorphic(exclude_heterozygous=False)
                .filter_singletons(exclude_heterozygous=False)
                .filter_biallelic(exclude_heterozygous=False)
                .resolve()
        >>> # Plot the Sankey diagram showing the number of loci removed at each filtering step.
        >>> nrm.plot_sankey_filtering_report()
        >>>
        >>> # Run a threshold search and plot the results.
        >>> nrm.search_thresholds(
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
            maf_thresholds=[0.01, 0.05, 0.1],
            mac_thresholds=[2, 3, 4, 5],
            filter_order=[
                "filter_missing_sample",
                "filter_missing",
                "filter_missing_pop",
                "filter_maf",
                "filter_mac",
                "filter_monomorphic",
                "filter_singletons",
                "filter_biallelic",
            ])

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

        popmap (Dict[str, str | int): A dictionary mapping sample IDs to population names.

        popmap_inverse (Dict[str | int], List[str]]): A dictionary mapping population names to lists of sample IDs.

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
    """

    def __init__(self, genotype_data: "GenotypeData") -> None:
        """Initializes the NRemover2 class.

        This method initializes the NRemover2 class with the provided GenotypeData instance. It sets up the filtering state, including the alignment, sample indices, loci indices, and other relevant attributes. It also initializes the filtering helper and methods.

        Args:
            genotype_data (GenotypeData): An instance of the GenotypeData class containing the genetic data alignment, population map,  populations, and other relevant data.
        """
        self.genotype_data = genotype_data
        self.popmap: Dict[str, str | int] = genotype_data.popmap
        self.popmap_inverse: Dict[str | int, List[str]] = genotype_data.popmap_inverse

        if self.popmap_inverse is not None:
            self.populations: List[str | int] = list(self.popmap_inverse.keys())
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
        self, filter_order: List[str] | None, filter_methods: Dict[str, Callable]
    ) -> List[str]:
        """Validates the filter order or sets the default order if none is provided.

        Args:
            filter_order (List[str] | None): A list of filter methods in order of application.
            filter_methods (Dict[str, Callable]): A dictionary of available filtering methods.

        Returns:
            Dict[str, Callable]: A dictionary of filter methods in the specified order.

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
        thresholds: List[float] | np.ndarray | None,
        maf_thresholds: List[float] | np.ndarray | None,
        mac_thresholds: List[int] | np.ndarray | None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sets the ranges of thresholds for filtering.

        Args:
            thresholds (List[float] | np.ndarray | None): Thresholds for missing data (0.0 < threshold ≤ 1.0).
            maf_thresholds (List[float] | np.ndarray | None): Thresholds for minor allele frequency (0.0 ≤ maf < 1.0).
            mac_thresholds (List[int] | np.ndarray | None): Thresholds for minimum allele count (MAC > 1).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The validated thresholds, maf thresholds, boolean array, and mac thresholds.

        Raises:
            ValueError: If any of the thresholds are invalid.

        Notes:
            - The thresholds for missing data must be between 0.0 and 1.0 (exclusive).
            - The maf thresholds must be between 0.0 and 1.0 (inclusive).
            - The mac thresholds must be greater than 1.
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, num=9, endpoint=True, dtype=float)
        else:
            if not all([0.0 < t <= 1.0 for t in thresholds]):
                msg = f"Invalid missing data threshold provided. Thresholds must be in the range [0.0, 1.0], but got: {','.join(map(str, thresholds))}"
                self.logger.error(msg)
                raise exceptions.InvalidThresholdError(
                    ",".join(map(str, thresholds)), msg
                )

        if maf_thresholds is None:
            maf_thresholds = np.array([0.01, 0.05, 0.075, 0.1, 0.15, 0.2], dtype=float)
        else:
            if not all([0.0 <= t < 1.0 for t in maf_thresholds]):
                msg = f"Invalid MAF threshold provided. MAF thresholds must be in the range [0.0, 1.0], but got: {','.join(map(str, maf_thresholds))}"
                self.logger.error(msg)
                raise exceptions.InvalidThresholdError(
                    ",".join(map(str, maf_thresholds)), msg
                )

        if mac_thresholds is None:
            mac_thresholds = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
        else:
            if not all([t > 1 for t in mac_thresholds]):
                msg = "Invalid MAC threshold provided. MAC thresholds must be greater than 1."
                self.logger.error(msg)
                raise exceptions.InvalidThresholdError(
                    ",".join(map(str, mac_thresholds)), msg
                )

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
        filter_methods: Dict[str, Tuple[Callable, np.ndarray | bool | None]],
    ) -> None:
        """Applies the specified filtering methods in sequence.

        This method applies the specified filtering methods in sequence, updating the alignment and indices based on the filtering results. It also records the filtering results in the filtering results DataFrame.

        Args:
            filter_methods (Dict[str, Tuple[Callable, np.ndarray | bool | None]]): A dictionary of filter methods and their corresponding thresholds.
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

        This method generates a plot of the filtering results and saves it to the output directory. The plot shows the number of loci removed at each filtering step and the proportion of loci removed relative to the total number of loci in the alignment.

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

        Raises:
            RuntimeError: If the filtering chain has not been resolved.

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

        This method marks the filtering chain as active, allowing further filtering steps to be applied. It should be called after resolving the previous chain and before starting a new chain.

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
        threshold: float | None = None,
    ) -> None:
        """Updates the loci indices based on the filtering results and records the step.

        This method updates the loci indices based on the filtering results and records the step in the filtering results DataFrame. It also updates the loci_removed_per_step dictionary to track the number of loci removed at each filtering step.

        Args:
            included_indices (np.ndarray): Boolean array indicating which loci are included.
            method_name (str): The name of the filtering method used.
            threshold (float | None): The threshold value used for filtering.

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
        threshold: float | None = None,
    ) -> None:
        """Updates the sample indices based on the filtering results and records the step.

        This method updates the sample indices based on the filtering results and records the step in the filtering results DataFrame. It also updates the samples_removed_per_step dictionary to track the number of samples removed at each filtering step.

        Args:
            included_indices (np.ndarray): Boolean array indicating which samples are included.
            method_name (str): The name of the filtering method used.
            threshold (float | None): The threshold value used for filtering.

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
        threshold: float | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[pd.DataFrame]]:
        """Records the results of a filtering step and updates the indices.

        This method records the results of a filtering step and updates the indices based on the filtering results.

        Args:
            removed_per_step (Dict[str, Tuple[int, int]]): A dictionary tracking the number of loci or samples removed at each filtering step.
            included_indices (np.ndarray): Boolean array indicating which loci or samples are included.
            current_list (List[pd.DataFrame]): A list of DataFrames containing filtering results for the current step.
            method_name (str): The name of the filtering method used.
            threshold (float | None): The threshold value used for filtering.

        Returns:
            Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[pd.DataFrame]]: The updated indices, removed loci or samples per step, and the current list of filtering results.
        """

        # Assert shape match
        reference_indices = (
            self.loci_indices
            if method_name != "filter_missing_sample"
            else self.sample_indices
        )
        assert included_indices.shape == reference_indices.shape, (
            f"Shape mismatch for {method_name}: "
            f"expected {reference_indices.shape}, got {included_indices.shape}"
        )

        total_input = np.count_nonzero(reference_indices)
        n_to_keep = np.count_nonzero(included_indices)
        n_removed = total_input - n_to_keep

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

        return included_indices.copy(), removed_per_step, current_list

    def resolve(self, benchmark_mode: bool = False) -> "GenotypeData":
        """Resolve the method chain and finalize the filtering process.

        This method resolves the method chain and finalizes the filtering process. It applies the selected filters to the alignment, updates the alignment, sample indices, and loci indices based on the filtering results, and resets the chain active flag. It returns the updated GenotypeData instance after filtering has been applied.

        Args:
            benchmark_mode (bool): If True, enables benchmark mode for performance measurement. Default is False.

        Returns:
            GenotypeData: The updated GenotypeData instance after filtering has been applied.

        Note:
            - This method is used to finalize the method chain by applying the selected filters to the alignment.
            - It updates the alignment, sample indices, and loci indices based on the filtering results.
            - It also resets the chain active flag and returns the updated GenotypeData instance after filtering has been applied.
        """
        gd = self._finalize_chain(benchmark_mode=benchmark_mode)
        self._chain_resolved = True  # Mark the chain as resolved
        return gd

    def _finalize_chain(self, benchmark_mode: bool = False) -> "GenotypeData":
        """Finalizes the method chain by applying the selected filters to the alignment.

        This method finalizes the method chain by applying the selected filters to the alignment. It updates the alignment, sample indices, and loci indices based on the filtering results. It also resets the chain active flag and returns the updated GenotypeData instance after filtering has been applied.

        Args:
            benchmark_mode (bool): If True, enables benchmark mode for performance measurement. Default is False.

        Raises:
            ValueError: If no samples or loci remain after filtering.
            RuntimeError: If no active filter chain is found.
            RuntimeError: If the filtering chain has not been resolved.

        Returns:
            GenotypeData: The updated GenotypeData instance after filtering has been applied, or None if no filtering was performed.
        """

        if not self._chain_active:
            msg = "No active filter chain to resolve."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if not self.search_mode and self.verbose:
            self.filtering_helper.print_filtering_report()

        si = self.sample_indices.astype(bool)
        li = self.loci_indices.astype(bool)

        if not np.any(si):
            msg = f"No samples remain after filtering at threshold: {self.current_threshold}."
            if not self.search_mode:
                self.logger.error(msg)
                raise exceptions.EmptyLocusSetError(msg)
            else:
                self.logger.info(msg)
                self._chain_active = False
                return

        if not np.any(li):
            msg = f"No loci remain after filtering at threshold: {self.current_threshold}."
            if not self.search_mode:
                self.logger.error(msg)
                raise exceptions.EmptyLocusSetError(msg)
            else:
                self.logger.info(msg)
                self._chain_active = False
                return

        if not np.any(si) or si.size == 0:
            if self.search_mode:
                msg = f"No samples left after filtering at threshold: {self.current_threshold}."
                self.logger.info(msg)
                self._chain_active = False
                return
            else:
                self.logger.error(msg)
                raise exceptions.EmptyLocusSetError(msg)

        self._chain_active = False

        if benchmark_mode:
            return self.genotype_data  # Do not update during benchmark

        filtered_alignment = self.alignment[np.ix_(si, li)].copy()
        filtered_samples = np.array(self.samples)[si].tolist()

        gd_filtered = self.genotype_data.copy()

        # Set updated alignment and force recomputation of attributes
        gd_filtered.set_alignment(
            snp_data=filtered_alignment,
            samples=filtered_samples,
            sample_indices=si,
            loci_indices=li,
            reset_attributes=True,  # Ensure metadata files are rewritten
        )

        return gd_filtered

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

    def __getattr__(self, name: str) -> Callable | None:
        """Custom attribute access method that handles delegating calls to `FilteringMethods` or `FilteringHelper`.

        This method allows for transparent access to the filtering methods and helper classes from NRemover2.

        Args:
            name (str): The name of the attribute to access.

        Returns:
            Callable | None: The requested attribute, or None if not found.

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
