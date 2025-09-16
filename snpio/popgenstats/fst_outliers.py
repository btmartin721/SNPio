import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from snpio.popgenstats.fst_distance import FstDistance
from snpio.popgenstats.fst_outliers_dbscan import DBSCANOutlierDetector
from snpio.popgenstats.fst_outliers_perm import PermutationOutlierDetector
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


class FstOutliers:
    """Class for detecting Fst outliers between populations."""

    def __init__(
        self,
        genotype_data: Any,
        plotter: Any,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the FstOutliers object.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing genotype data.
            logger (logging.Logger): Logger object.
        """
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.verbose = verbose
        self.debug = debug

        self.fst_dist = FstDistance(
            genotype_data, plotter, verbose=verbose, debug=debug
        )

        if genotype_data.was_filtered:
            outdir = Path(f"{self.genotype_data.prefix}_output", "nremover", "analysis")
        else:
            outdir = Path(f"{self.genotype_data.prefix}_output", "analysis")
        outdir.mkdir(parents=True, exist_ok=True)

        logman = LoggerManager(
            __name__, prefix=self.genotype_data.prefix, verbose=verbose, debug=debug
        )
        self.logger = logman.get_logger()

        self.full_matrix = self.genotype_data.snp_data.astype(str)

        self.logger.debug(f"{self.full_matrix.shape=}")

        # Build an index for sample -> row
        self.sample_to_idx = {s: i for i, s in enumerate(self.genotype_data.samples)}

        # Convert population -> integer indices
        self.pop_indices = {
            pop: np.array([self.sample_to_idx[s] for s in sample_list], dtype=int)
            for pop, sample_list in self.genotype_data.popmap_inverse.items()
        }

        self.pop_names = sorted(self.pop_indices.keys())

        self.iupac = IUPAC(logger=self.logger)
        self.phased_encoding = self.iupac.get_phased_encoding()

    def detect_fst_outliers_dbscan(
        self,
        correction_method: Literal["bonferroni", "fdr_bh"] | None = None,
        alpha: float = 0.05,
        n_simulations: int = 1000,
        n_jobs: int = 1,
        seed: int | None = None,
        min_samples: int = 5,
    ) -> pd.DataFrame:
        """Detect Fst outliers from SNP data using a multivariate DBSCAN method.

        Args:
            correction_method (Literal["bonferroni", "fdr_bh"] | None): Multiple testing correction method, e.g., 'bonferroni', 'fdr_bh', or None for no correction.
            alpha (float): Significance level for multiple testing correction.
            n_simulations (int): Number of permutations for significance testing.
            n_jobs (int): Number of CPU threads to use for parallelization.
            seed (int | None): Random seed for reproducibility. Defaults to None.
            min_samples (int): Minimum number of samples for DBSCAN clustering. Defaults to 5.
        """
        outliers_df = self._dbscan_fst(
            correction_method=correction_method,
            alpha=alpha,
            n_simulations=n_simulations,
            n_jobs=n_jobs,
            seed=seed,
            min_samples=min_samples,
        )

        if outliers_df.empty:
            self.logger.warning("No Fst outliers detected. Skipping Fst outlier plot.")
        else:
            self.logger.info(
                f"{len(outliers_df['Locus'].unique())} Fst outlier loci detected using DBSCAN method."
            )

        self.logger.info("DBSCAN Fst outlier detection complete!")
        return outliers_df

    def _dbscan_fst(
        self,
        correction_method: str | None,
        alpha: float,
        n_simulations: int,
        n_jobs: int,
        seed: int | None = None,
        min_samples: int = 5,
        z_score_threshold: float = 2.5,
    ) -> pd.DataFrame:
        """Core logic for detecting Fst outliers using DBSCAN.

        This method applies the DBSCAN clustering algorithm to identify outlier loci based on their Fst values. It returns a DataFrame containing the detected outliers and their associated metadata.

        Args:
            correction_method (str | None): Multiple testing correction method, e.g., 'bonferroni', 'fdr_bh', or None for no correction.
            alpha (float): Significance level for multiple testing correction.
            n_simulations (int): Number of permutations for significance testing.
            n_jobs (int): Number of CPU threads to use for parallelization.
            seed (int | None): Random seed for reproducibility. Defaults to None.
            min_samples (int): Minimum number of samples for DBSCAN clustering. Defaults to 5.
            z_score_threshold (float): Threshold for z-scores to determine contributing pairs. Defaults to 2.5.

        Returns:
            pd.DataFrame: A DataFrame containing the Fst outliers and contributing population pairs.
        """
        self.logger.info("Detecting Fst outliers with DBSCAN method…")

        # 1. Prepare the (loci x pairs) Fst matrix
        fst_dict = self._calculate_observed_per_locus_fst_all_pairs()

        fst_long = pd.concat(
            [
                df.assign(Population_Pair=f"{p1}_{p2}")
                for (p1, p2), df in fst_dict.items()
            ]
        )
        fst_wide = fst_long.pivot(
            index="Locus", columns="Population_Pair", values="Fst"
        ).sort_index()

        if fst_wide.shape[1] < 2:
            msg = f"DBSCAN requires at least 2 population pairs, but got: {fst_wide.shape[1]}"
            self.logger.error(msg)
            raise ValueError(msg)

        # 2. Impute and prepare data
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(fst_wide)
        df_imputed = pd.DataFrame(X, index=fst_wide.index, columns=fst_wide.columns)
        df_imputed = df_imputed.clip(lower=0.0)

        # 3. Identify Global Outliers using the multivariate method
        est = DBSCANOutlierDetector(
            eps="auto",
            min_samples=min_samples,
            n_jobs=n_jobs,
            standardize="standardize",
            null_model="shuffle",
            n_simulations=n_simulations,
            random_state=seed,
            verbose=self.verbose,
            debug=self.debug,
        )
        est.fit(df_imputed)
        p_values_1d = est.estimate_pvalues()
        is_outlier_series, q_values_series = est.identify_outliers(
            p_values_1d, alpha=alpha, correction_method=correction_method
        )
        is_outlier_series.index = df_imputed.index
        q_values_series.index = df_imputed.index
        significant_loci = is_outlier_series[is_outlier_series].index

        if significant_loci.empty:
            self.logger.warning(
                "No significant outliers were detected by DBSCAN method."
            )
            return pd.DataFrame()

        # 4. Post-Hoc: Analyze which pairs contributed
        z_scores_df = pd.DataFrame(
            est._scaler.transform(df_imputed),
            index=df_imputed.index,
            columns=df_imputed.columns,
        )
        significant_z_scores = z_scores_df.loc[significant_loci]
        contrib_mask = significant_z_scores.abs() >= z_score_threshold

        final_results = []
        for locus in significant_loci:
            locus_data = fst_wide.loc[locus]
            locus_q_value = q_values_series.loc[locus]
            contributing_pairs = locus_data[contrib_mask.loc[locus]].index.tolist()
            for pair in contributing_pairs:
                final_results.append(
                    {
                        "Locus": locus,
                        "Population_Pair": pair,
                        "Fst": locus_data[pair],
                        "q_value": locus_q_value,
                    }
                )

        return pd.DataFrame(final_results)

    def detect_outliers_permutation(
        self,
        n_perm: int = 1000,
        correction_method: Literal["fdr_bh", "bonferroni"] | None = "fdr_bh",
        alpha: float = 0.05,
        seed: int = 42,
        alternative: Literal["upper", "lower", "both"] = "both",
    ) -> pd.DataFrame:
        """Detects all per-locus outliers in all pairs using permutation tests."""
        perm_detector = PermutationOutlierDetector(
            genotype_data=self.genotype_data, verbose=self.verbose, debug=self.debug
        )
        return perm_detector.run(
            n_perm=n_perm,
            correction_method=correction_method,
            alpha=alpha,
            seed=seed,
            alternative=alternative,
        )

    def _calculate_observed_per_locus_fst_all_pairs(
        self,
    ) -> dict[Tuple[str, str], pd.DataFrame]:
        """Compute observed per-locus Weir & Cockerham's Fst for each population pair."""
        full_matrix = self.genotype_data.snp_data
        popmap = self.genotype_data.popmap_inverse
        sample_to_idx = {name: i for i, name in enumerate(self.genotype_data.samples)}
        pop_indices = {
            pop_name: np.array([sample_to_idx[s] for s in samples], dtype=int)
            for pop_name, samples in popmap.items()
        }
        n_loci = full_matrix.shape[1]

        result = {}
        for pop1, pop2 in combinations(popmap.keys(), 2):
            inds1, inds2 = pop_indices[pop1], pop_indices[pop2]

            # This helper from FstDistance computes
            # the (a) and (d) components per locus
            a_vals, d_vals = self.fst_dist.fst_variance_components_per_locus(
                inds1, inds2, full_matrix
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                observed_fst = np.where(d_vals != 0, a_vals / d_vals, np.nan)

            if (
                hasattr(self.genotype_data, "marker_names")
                and self.genotype_data.marker_names
            ):
                locus_names = self.genotype_data.marker_names
            else:
                locus_names = [f"locus_{i}" for i in range(n_loci)]

            df = pd.DataFrame({"Locus": locus_names, "Fst": observed_fst})
            result[(pop1, pop2)] = df

        return result
