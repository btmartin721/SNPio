import itertools
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from snpio.utils.benchmarking import Benchmark
from snpio.utils.custom_exceptions import SequenceLengthError


class FilteringHelper:
    def __init__(self, nremover_instance):
        """
        Initialize the helper class for NRemover2.

        Args:
            nremover_instance (NRemover2): An instance of the NRemover2 class to access its attributes.
        """
        self.nremover = nremover_instance
        self.logger = self.nremover.logger

    def print_filtering_report(self):
        """
        Print a filtering report to the terminal.

        Returns:
            None.

        Raises:
            SequenceLengthError: If there is no data left after filtering.
        """
        if self.nremover.verbose:
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
            total = len(aln) * len(aln[0])
            if total == 0:
                msg = "There is no data left after filtering."
                self.logger.error(msg)
                raise SequenceLengthError(msg)
            missing = np.count_nonzero(np.isin(aln, ["N", "-", ".", "?"]))
            return (missing / total) * 100 if total else 0.0

        before = missing_data_percent(before_alignment)
        after = missing_data_percent(after_alignment)

        msg = "Filtering Report: "
        msg += f"\n\tLoci before filtering: {num_loci_before}"
        msg += f"\n\tSamples before filtering: {num_samples_before}"
        self.logger.info(msg)

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

        msg = "\n".join(rem_per_step)
        msg += f"\n\nSamples removed: {samples_removed}"
        msg += f"\nLoci remaining: {num_loci_after}"
        msg += f"\nSamples remaining: {num_samples_after}"
        msg += f"\nMissing data before filtering: {before:.2f}%"
        msg += f"\nMissing data after filtering: {after:.2f}%\n"
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

        The filtering logic is kept in the NRemover2 class while the plotting and threshold search logic is in this method.

        Args:
            thresholds (List[float], optional): A list of missing data thresholds to search. Defaults to None.
            maf_thresholds (List[float], optional): A list of minor allele frequency thresholds to search. Defaults to None.
            mac_thresholds (List[int], optional): A list of minor allele count thresholds to search. Defaults to None.
            filter_order (List[str], optional): A list of filtering methods to apply in order. If None, then filtering methods will be provided in the following order: "filter_missing_sample", "filter_missing_pop", "filter_maf", "filter_mac", "filter_monomorphic", "filter_biallelic", "filter_singletons". Defaults to None.
        """
        if self.nremover.verbose:
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

        if self.nremover.verbose:
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

    def _prepare_dataframes(self):
        """Prepare DataFrames for plotting."""
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
