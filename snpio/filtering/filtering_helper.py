import itertools
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


class FilteringHelper:
    def __init__(self, nremover_instance: Any) -> None:
        """
        Initialize the FilteringHelper class.

        Args:
            nremover_instance (NRemover2): An instance of the NRemover2 class to access its attributes.
        """
        self.nremover = nremover_instance
        self.logger = self.nremover.logger
        self.missing_vals = ["N", "-", ".", "?"]

    def print_filtering_report(self) -> None:
        """
        Print a filtering report to the terminal.

        This method generates and prints a report detailing the filtering process, including the number of loci and samples before and after filtering, and the percentage of missing data.

        Returns:
            None. The report is printed to the terminal.

        Raises:
            RuntimeError: If there is no data left after filtering.

        Note:
            This method should be called after filtering is complete.

            If no data is removed during filtering, a warning is logged.
        """
        self.logger.info("Printing filtering report.")

        before_alignment = self.nremover.alignment.copy()
        after_alignment = self.nremover.alignment[self.nremover.sample_indices, :][
            :, self.nremover.loci_indices
        ].copy()

        num_loci_before = len(before_alignment[0])
        num_samples_before = len(before_alignment)
        num_loci_after = len(after_alignment[0])
        num_samples_after = len(after_alignment)
        samples_removed = num_samples_before - num_samples_after

        def missing_data_percent(aln):
            """Calculates the percentage of missing data in an alignment.

            Args:
                aln (np.ndarray): A 2D numpy array representing an alignment.

            Returns:
                float: The percentage of missing data in the alignment.

            Raises:
                RuntimeError: If there is no data left after filtering.
            """
            total = len(aln) * len(aln[0])
            if total == 0:
                msg = "There is no data left after filtering."
                self.logger.error(msg)
                raise RuntimeError(msg)
            missing = np.count_nonzero(np.isin(aln, self.missing_vals))
            return (missing / total) * 100 if total else 0.0

        before = missing_data_percent(before_alignment)
        after = missing_data_percent(after_alignment)

        loci_unchanged = [
            self.nremover.loci_removed_per_step[k][1] == 0
            for k in self.nremover.loci_removed_per_step.keys()
        ]

        samp_unchanged = [
            self.nremover.samples_removed_per_step[k][1] == 0
            for k in self.nremover.samples_removed_per_step.keys()
        ]

        if all(loci_unchanged) and all(samp_unchanged) and before == after:
            msg = "The alignment was unchanged, nothing was filtered."
            self.logger.warning(msg)

        rem_per_step = [
            f"{key} (Step {value[0]}): {value[1]}"
            for key, value in self.nremover.loci_removed_per_step.items()
        ]

        msg = f"\nFiltering Report:"
        msg += f"\nLoci before filtering: {num_loci_before}"
        msg += f"\nSamples before filtering: {num_samples_before}"
        msg += "\n".join(rem_per_step)
        msg += f"\nSamples removed: {samples_removed}"
        msg += f"\nLoci remaining: {num_loci_after}"
        msg += f"\nSamples remaining: {num_samples_after}"
        msg += f"\nMissing data before filtering: {before:.2f}%"
        msg += f"\nMissing data after filtering: {after:.2f}%\n\n"
        self.logger.info(msg)

    def search_thresholds(
        self,
        thresholds: Optional[List[float]] = None,
        maf_thresholds: Optional[List[float]] = None,
        mac_thresholds: Optional[List[int]] = None,
        filter_order: Optional[List[str]] = None,
    ) -> None:
        """
        Search across filtering thresholds and plot the filtering proportions.

        This method searches through various combinations of filtering thresholds and plots the results. The filtering logic is handled by the NRemover2 class, while this method manages the plotting and threshold search.

        Args:
            thresholds (List[float], optional): A list of missing data thresholds to search. Defaults to None.
            maf_thresholds (List[float], optional): A list of minor allele frequency thresholds to search. Defaults to None.
            mac_thresholds (List[int], optional): A list of minor allele count thresholds to search. Defaults to None.
            filter_order (List[str], optional): A list of filtering methods to apply in order. If None, the default order is: "filter_missing_sample", "filter_missing_pop", "filter_maf", "filter_mac", "filter_monomorphic", "filter_biallelic", "filter_singletons". Defaults to None.

        Returns:
            None
        """
        self.logger.info("Searching and plotting filtering thresholds.")

        self.nremover.search_mode = True

        # Set up threshold ranges
        thresholds, maf_thresholds, bool_thresholds, mac_thresholds = (
            self.nremover._set_threshold_ranges(
                thresholds, maf_thresholds, mac_thresholds
            )
        )

        # Create all combinations of thresholds
        threshold_combinations = list(
            itertools.product(
                thresholds, maf_thresholds, bool_thresholds, mac_thresholds
            )
        )

        self.logger.info(
            f"Searching {len(threshold_combinations)} threshold combinations..."
        )

        for (
            threshold,
            maf_threshold,
            bool_threshold,
            mac_threshold,
        ) in tqdm(threshold_combinations, desc="Threshold search: ", unit="combo"):
            self.nremover._reset_filtering_state()

            # Validate and set filter order
            filter_methods = self.nremover._get_filter_methods(
                threshold, maf_threshold, bool_threshold, mac_threshold
            )

            filter_methods = self.nremover._validate_filter_order(
                filter_order, filter_methods
            )

            # Apply filtering methods
            self.nremover._apply_filtering_methods(filter_methods)

        # Prepare DataFrames
        df_combined = self._prepare_dataframes()

        # Plot results
        self.nremover._plot_results(df_combined=df_combined)
        self.nremover.search_mode = False
        self.nremover._reset_filtering_state()
        self.nremover._reset_filtering_results()
        self.nremover.genotype_data.resource_data.update(
            self.nremover.filtering_methods.resource_data
        )

    def _prepare_dataframes(self) -> pd.DataFrame:
        """Prepare DataFrames for plotting.

        This method concatenates the sample and global DataFrames from the NRemover2 instance for plotting purposes.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing sample and global data.

        Raises:
            ValueError: If there is no data to plot.

        Note:
            - This method should be called after filtering is complete.
            - If no data is available for plotting, a ValueError is raised.
            - The concatenated DataFrame is used for plotting the filtering results.
            - The concatenated DataFrame contains the sample and global data.
            - The sample data is stored in the 'df_sample_list' attribute of the NRemover2 instance.
            - The global data is stored in the 'df_global_list' attribute of the NRemover2 instance.
            - If no data is available for plotting, a ValueError is raised.
        """
        df_sample = (
            pd.concat(self.nremover.df_sample_list)
            if self.nremover.df_sample_list
            else pd.DataFrame()
        )
        df_global = (
            pd.concat(self.nremover.df_global_list)
            if self.nremover.df_global_list
            else pd.DataFrame()
        )

        if df_sample.empty and df_global.empty:
            msg = "No data to plot. Please check the filtering thresholds."
            self.logger.error(msg)
            raise ValueError(msg)

        return pd.concat([df_sample, df_global])
