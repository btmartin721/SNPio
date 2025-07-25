from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from snpio.plotting.plotting import Plotting
from snpio.utils.logging import LoggerManager
from snpio import GenotypeEncoder

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class AlleleSummaryStats:
    """Class for summarizing allele data from SNP genotypes.

    This class provides methods to summarize allele data, including missingness, heterozygosity, minor allele frequency (MAF), and other statistics.
    """

    def __init__(
        self, genotype_data: "GenotypeData", verbose: bool = False, debug: bool = False
    ) -> None:
        """Initialize the AlleleSummary with genotype data.

        This method sets up the allele mapping and prepares the genotype data for analysis.

        Args:
            genotype_data (GenotypeData): An object containing SNP data in a format compatible with this class.
        """
        self.genotype_data = genotype_data

        self.plotter = Plotting(self.genotype_data, **self.genotype_data.plot_kwargs)

        logman = LoggerManager(
            __name__, prefix=self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()

        ge = GenotypeEncoder(self.genotype_data)
        self.alleles = ge.two_channel_alleles

    def summarise(self, sample_indices: np.ndarray | None = None) -> pd.Series:
        """Summarize allele data from the genotype data.

        Computes missingness, heterozygosity, MAF, and other statistics for the SNP data. This method is a wrapper around ``summarize()`` and is intended as an alias.

        Args:
            sample_indices (np.ndarray | None): Optional subset of individuals. If None, uses all samples.

        Returns:
            pd.Series: summary statistics including missingness, heterozygosity, MAF.
        """
        return self.summarize(sample_indices)

    def summarize(self, sample_indices: np.ndarray | None = None) -> pd.Series:
        """Summarize allele data from the genotypes as proportions.

        Computes a suite of proportional statistics for the SNP dataset:
        - Missingness (overall, per-sample/locus, median, % with any, quartiles)
        - Heterozygosity (overall, per-sample/locus, quartiles)
        - Allelic spectrum (mono- through quad-allelic; mean # alleles; effective # alleles)
        - Expected heterozygosity and F_IS (per-locus, averaged)
        - Singleton loci and MAF (mean, median, rare-variant proportion, spectrum bins)

        Args:
            sample_indices (np.ndarray | None): Optional subset of individuals.
                If None, uses all samples.

        Returns:
            pd.Series: Each entry is a proportion (0-1).
        """
        alleles1, alleles2 = self.alleles
        if sample_indices is not None:
            alleles1 = alleles1[sample_indices]
            alleles2 = alleles2[sample_indices]

        # 1. Masks & basic counts
        missing_mask = (alleles1 == -1) | (alleles2 == -1)
        non_missing = ~missing_mask
        n_non_missing = non_missing.sum()
        het_mask = (alleles1 != alleles2) & non_missing

        # 2. Sample- / locus-level missingness & het
        sample_miss = missing_mask.mean(axis=1)  # (n_samples,)
        locus_miss = missing_mask.mean(axis=0)  # (n_loci,)
        sample_het = np.divide(
            het_mask.sum(axis=1),
            non_missing.sum(axis=1),
            where=non_missing.sum(axis=1) > 0 & np.isfinite(non_missing.sum(axis=1)),
        )
        locus_het = np.divide(
            het_mask.sum(axis=0),
            non_missing.sum(axis=0),
            where=non_missing.sum(axis=0) > 0 & np.isfinite(non_missing.sum(axis=0)),
        )

        # 3. Allele counts per locus
        # (2*n_samples, n_loci)
        flat = np.concatenate([alleles1, alleles2], axis=0)

        # (n_loci,4)
        counts = np.vstack([np.sum(flat == i, axis=0) for i in range(4)]).T
        n_alleles = np.sum(counts > 0, axis=1)  # distinct alleles per locus
        tot_alleles = counts.sum(axis=1)
        # allele_freqs per locus
        freqs = np.divide(
            counts,
            tot_alleles[:, None],
            where=tot_alleles[:, None] > 0 & np.isfinite(tot_alleles[:, None]),
        )
        minor_counts = tot_alleles - counts.max(axis=1)

        # minor allele frequency per locus
        # (n_loci,)
        maf = np.divide(
            minor_counts, tot_alleles, where=tot_alleles > 0 & np.isfinite(tot_alleles)
        )

        # 4. Derived metrics
        # per locus
        biallelic = np.sum(freqs**2, axis=1)  # check for biallelic loci
        effective_alleles = np.divide(
            1.0, biallelic, where=biallelic > 0 & np.isfinite(biallelic)
        )
        exp_het = 1.0 - np.sum(freqs**2, axis=1)  # expected heterozygosity

        # 1) Build an explicit boolean mask
        mask = (exp_het > 0) & np.isfinite(exp_het) & np.isfinite(locus_het)

        # 2) Allocate F_IS = 0 everywhere
        F_IS = np.zeros_like(exp_het)

        # 3) Only do the division on the good entries
        F_IS[mask] = 1.0 - (locus_het[mask] / exp_het[mask])

        # 5. MAF bins
        maf_bins = {
            "MAF < 0.01": np.mean(maf < 0.01),
            "0.01 ≤ MAF < 0.05": np.mean((maf >= 0.01) & (maf < 0.05)),
            "0.05 ≤ MAF < 0.10": np.mean((maf >= 0.05) & (maf < 0.10)),
            "0.10 ≤ MAF < 0.20": np.mean((maf >= 0.10) & (maf < 0.20)),
            "MAF ≥ 0.20": np.mean(maf >= 0.20),
        }

        summary = {
            # Missingness
            "Overall Missing Prop.": missing_mask.mean(),
            "Median Sample Missing": np.median(sample_miss),
            "Median Locus Missing": np.median(locus_miss),
            "Pct Samples with Missing": np.mean(sample_miss > 0),
            "Pct Loci with Missing": np.mean(locus_miss > 0),
            "Sample Miss Q1": np.percentile(sample_miss, 25),
            "Sample Miss Q3": np.percentile(sample_miss, 75),
            "Locus Miss Q1": np.percentile(locus_miss, 25),
            "Locus Miss Q3": np.percentile(locus_miss, 75),
            # Heterozygosity
            "Overall Heterozygosity Prop.": het_mask.sum() / n_non_missing,
            "Mean Sample Heterozygosity Prop.": np.nanmean(sample_het),
            "Mean Locus Heterozygosity Prop.": np.nanmean(locus_het),
            "Sample Heterozygosity Q1": np.nanpercentile(sample_het, 25),
            "Sample Heterozygosity Q3": np.nanpercentile(sample_het, 75),
            "Locus Heterozygosity Q1": np.nanpercentile(locus_het, 25),
            "Locus Heterozygosity Q3": np.nanpercentile(locus_het, 75),
            # Allelic spectrum
            "Prop. Monomorphic": np.mean(n_alleles == 1),
            "Prop. Biallelic": np.mean(n_alleles == 2),
            "Prop. Triallelic": np.mean(n_alleles == 3),
            "Prop. Quadallelic": np.mean(n_alleles == 4),
            "Mean Alleles per Locus": np.mean(n_alleles),
            "Mean Effective Alleles": np.mean(effective_alleles),
            # Expected het & F_IS
            "Mean Expected Heterozygosity": np.nanmean(exp_het),
            "Mean F_IS": np.nanmean(F_IS),
            # Singleton & MAF summary
            "Prop. Singleton Loci": np.mean(minor_counts == 1),
            "MAF Mean": np.nanmean(maf),
            "MAF Median": np.nanmedian(maf),
            "Prop. Rare Variants": np.mean(maf < 0.05),
            **maf_bins,
        }

        self.logger.info("Allele-based summary statistics complete!")
        self.logger.debug(f"Summary statistics: {summary}")

        series = pd.Series(summary)
        self._plot(series)
        return series

    def _plot(self, summary: pd.Series) -> None:
        """Plot allele summary statistics.

        This method is intended to visualize the allele summary statistics using the Plotting class.

        Args:
            summary (pd.Series): Summary statistics to be plotted.
        """
        self.logger.info("Plotting allele summary statistics...")
        self.plotter.plot_allele_summary(summary, figsize=(12, 6))
        self.logger.info("Allele summary statistics plotting complete!")

    def __repr__(self) -> str:
        """String representation of the AlleleSummaryStats object."""
        return (
            f"AlleleSummaryStats(genotype_data={self.genotype_data}, "
            f"alleles_shape={self.alleles[0].shape})"
        )

    def __str__(self) -> str:
        """String representation of the AlleleSummaryStats object."""
        return (
            f"AlleleSummaryStats with {self.genotype_data.n_samples} samples and "
            f"{self.genotype_data.n_loci} loci."
        )
