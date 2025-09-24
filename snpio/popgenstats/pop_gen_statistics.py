import warnings
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import snpio.utils.custom_exceptions as exceptions
from snpio import GenotypeEncoder, Plotting
from snpio.analysis.allele_summary_stats import AlleleSummaryStats
from snpio.popgenstats.amova import AMOVA
from snpio.popgenstats.d_statistics import DStatistics
from snpio.popgenstats.fst_distance import FstDistance
from snpio.popgenstats.fst_outliers import FstOutliers
from snpio.popgenstats.genetic_distance import GeneticDistance
from snpio.popgenstats.summary_statistics import SummaryStatistics
from snpio.utils.logging import LoggerManager
from snpio.utils.multiqc_reporter import SNPioMultiQC

if TYPE_CHECKING:
    from snpio.plotting.plotting import Plotting
    from snpio.read_input.genotype_data import GenotypeData


class PopGenStatistics:
    """Class for calculating population genetics statistics from SNP data.

    This class provides methods for calculating population genetics statistics from SNP data. It is designed to work with GenotypeData objects. The PopGenStatistics class can calculate Patterson's D-statistic, partitioned D-statistic, D-foil statistic, summary statistics, and perform PCA dimensionality reduction analysis. It also includes methods for calculating Fst, Nei's genetic distance, and detecting Fst outliers using permutation or DBSCAN clustering methods. Finally, it can conduct AMOVA (Analysis of Molecular Variance) to partition genetic variation among and within populations.
    """

    def __init__(
        self, genotype_data: "GenotypeData", verbose: bool = False, debug: bool = False
    ) -> None:
        """Initialize the PopGenStatistics object.

        This class provides methods for calculating population genetics statistics from SNP data. It is designed to work with GenotypeData objects. The PopGenStatistics class can calculate Patterson's D-statistic, partitioned D-statistic, D-foil statistic, summary statistics, and perform PCA dimensionality reduction analysis.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing SNP data and metadata.
            verbose (bool): Whether to display verbose output. Defaults to False.
            debug (bool): Whether to display debug output. Defaults to False.
        """
        self.genotype_data: "GenotypeData" = genotype_data
        self.verbose: bool = verbose
        self.debug: bool = debug
        self.alignment: np.ndarray = genotype_data.snp_data
        self.popmap: Dict[str, str | int] = genotype_data.popmap
        self.populations: List[str | int] = genotype_data.populations

        plot_kwargs: dict = genotype_data.plot_kwargs
        plot_kwargs["debug"] = debug
        plot_kwargs["verbose"] = verbose

        # Initialize plotting and dstats objects
        self.plotter: "Plotting" = Plotting(genotype_data, **plot_kwargs)

        # Initialize logger
        logman = LoggerManager(
            __name__, prefix=genotype_data.prefix, debug=debug, verbose=verbose
        )

        # Get logger object and set logging level
        level: str = "DEBUG" if debug else "INFO"
        logman.set_level(level)
        self.logger: Logger = logman.get_logger()
        self.logger.verbose = verbose

        self.d_stats = DStatistics(
            self.genotype_data,
            self.alignment,
            self.genotype_data.samples,
            self.logger,
            self.verbose,
            self.debug,
        )

        self.encoder = GenotypeEncoder(self.genotype_data)
        self.alignment_012: np.ndarray = self.encoder.genotypes_012.astype(np.float64)

        if self.genotype_data.was_filtered:
            self.outdir = Path(
                f"{self.genotype_data.prefix}_output", "nremover", "reports", "analysis"
            )
        else:
            self.outdir = Path(
                f"{self.genotype_data.prefix}_output", "reports", "analysis"
            )
        self.outdir.mkdir(exist_ok=True, parents=True)

        self.snpio_mqc = SNPioMultiQC

    def calculate_d_statistics(
        self,
        method: Literal["patterson", "partitioned", "dfoil"],
        population1: str | List[str],
        population2: str | List[str],
        population3: str | List[str],
        *,
        population4: str | List[str] | None = None,
        outgroup: str | List[str] | None = None,
        num_bootstraps: int = 1000,
        max_individuals_per_pop: int | None = None,
        individual_selection: str | Dict[str, List[str]] = "random",
        save_plot: bool = True,
        seed: int | None = None,
        per_combination: bool = True,
        calc_overall: bool = True,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Tuple[pd.DataFrame, dict]:
        """Calculate D-statistics with bootstrap support and return a summary DataFrame and overall stats.

        This method calculates D-statistics using the specified method (patterson, partitioned, or dfoil) for the given populations and outgroup. It supports bootstrapping for statistical significance and can save results to a file.

        Args:
            method (Literal["patterson", "partitioned", "dfoil"]): The method to use for D-statistics calculation.
            population1 (str | List[str]): The first population to compare.
            population2 (str | List[str]): The second population to compare.
            population3 (str | List[str]): The third population to compare.
            population4 (str | List[str] | None): The fourth population to compare (if applicable).
            outgroup (str | List[str]): The outgroup population.
            snp_indices (np.ndarray | List[int] | None): Specific SNP indices to include in the analysis.
            num_bootstraps (int): Number of bootstrap replicates to perform.
            max_individuals_per_pop (int | None): Maximum individuals to sample per population.
            individual_selection (str | Dict[str, List[str]]): Method for individual selection.
            save_plot (bool): Whether to save the plot.
            seed (int | None): Random seed for reproducibility.
            per_combination (bool): Whether to calculate D-statistics for each combination of populations.
            calc_overall (bool): Whether to calculate overall D-statistics.
            use_jackknife (bool): Whether to use jackknife resampling instead of bootstrap. Use this if you want to estimate the variance of the D-statistics and there is the possibility of linkage disequilibrium in the data. Defaults to False, which uses bootstrap resampling instead.
            block_size (int): Block size for jackknife resampling. Defaults to 500.


        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing a DataFrame with D-statistics results and a dictionary with overall statistics.
        """
        self.logger.info("Calculating D-statistics...")
        self.logger.info(f"Using D-statistics method: {method}")
        self.logger.info(f"Number of bootstraps: {num_bootstraps}")
        self.logger.info(f"Max individuals per population: {max_individuals_per_pop}")
        self.logger.info(f"Seed: {seed if seed is not None else 'None'}")
        self.logger.info(f"Individual selection: {individual_selection}")
        self.logger.info(f"Save plot: {save_plot}")

        # Validate method
        if method not in {"patterson", "partitioned", "dfoil"}:
            msg = f"Invalid method: {method}. Supported methods: 'patterson', 'partitioned', 'dfoil'."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        df, overall = self.d_stats.run(
            method=method,
            population1=population1,
            population2=population2,
            population3=population3,
            population4=population4,
            outgroup=outgroup,
            num_bootstraps=num_bootstraps,
            max_individuals_per_pop=max_individuals_per_pop,
            individual_selection=individual_selection,
            seed=seed,  # For reproducibility
            per_combination=per_combination,
            calc_overall=calc_overall,
            use_jackknife=use_jackknife,
            block_size=block_size,
        )

        if save_plot:
            self.plotter.plot_d_statistics(df, method=method)

        return df, overall

    def detect_fst_outliers(
        self,
        correction_method: (
            Literal[
                "bonferroni",
                "fdr_bh",
                "fdr_by",
                "holm",
                "sidak",
                "holm-sidak",
                "simes-hochberg",
                "hommel",
                "fdr_tsbh",
                "fdr_tsbky",
            ]
            | None
        ) = None,
        alpha: float = 0.05,
        use_dbscan: bool = False,
        n_permutations: int = 1000,
        n_jobs: int = 1,
        seed: int | None = None,
        min_samples: int = 5,
        max_outliers_to_plot: int | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect Fst outliers from SNP data using permutation or DBSCAN.

        This method detects Fst outliers from SNP data using permutation or DBSCAN clustering. Outliers are identified based on the distribution of Fst values between population pairs. The method returns a DataFrame containing the Fst outliers and contributing population pairs, as well as a DataFrame containing the adjusted or unadjusted P-values, depending on whether a multiple testing correction method was specified.

        Args:
            correction_method (Literal["bonferroni", "fdr_bh", "fdr_by", "holm", "sidak", "holm-sidak", "simes-hochberg", "hommel", "fdr_tsbh", "fdr_tsbky"]): Multiple testing correction method that performs P-value adjustment. Shoould be either 'bonf' (Bonferroni) or 'fdr' (FDR B-H), or None. If not specified, no correction or P-value adjustment is applied. Defaults to None (no correction).
            alpha (float): Significance level for multiple test correction with adjusted P-values. Defaults to 0.05.
            use_dbscan (bool): Whether to use DBSCAN clustering to estimate Fst outliers per SNP. If False, permutation variance method is used instead. Defaults to False.
            n_permutations (int): Number of permutation replicates to use for estimating variance of Fst per SNP. Defaults to 1000.
            n_jobs (int): Number of parallel jobs. Only applies to DBSCAN method. -1 uses all available CPU threads. Defaults to 1.
            seed (int | None): Random seed for reproducibility. Defaults to None, which means a random seed will be used.
            min_samples (int): Minimum number of samples for DBSCAN clustering. Defaults to 5.
            max_outliers_to_plot (int | None): Maximum number of outliers to plot. If None, all outliers are plotted. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the locus indices for Fst outliers, the Fst values, contributing population pairs to each outlier, and the unadjusted and adjusted P-values (if applicable).
        """
        self.logger.info("Detecting Fst outliers...")
        self.logger.info(
            f"Using {'DBSCAN' if use_dbscan else 'permutation'} method for outlier detection."
        )

        self.logger.info(
            f"Correction method: {correction_method if correction_method else 'None'}"
        )

        self.logger.info(f"Alpha level: {alpha}")

        self.logger.info(f"Number of permutations: {n_permutations}")
        self.logger.info(f"Number of parallel jobs: {n_jobs}")
        self.logger.info(f"Random seed: {seed if seed is not None else 'None'}")
        self.logger.info(f"Minimum samples for DBSCAN: {min_samples}")

        fo = FstOutliers(
            self.genotype_data, self.plotter, verbose=self.verbose, debug=self.debug
        )

        if correction_method is not None:
            if isinstance(correction_method, str):
                # Convert to lowercase for consistency
                correction_method = correction_method.lower()
            else:
                msg = f"Invalid type for correction_method: {type(correction_method)}. Expected str or None."
                self.logger.error(msg)
                raise TypeError(msg)

            # Validate correction_method
            corr_method_set = {
                "bonferroni",
                "fdr_bh",
                "fdr_by",
                "fdr",
                "bonf",
                "sidak",
                "holm",
                "holm-sidak",
                "simes-hochberg",
                "hommel",
                "fdr_tsbh",
                "fdr_tsbky",
            }

            if correction_method not in corr_method_set:
                msg = f"Invalid correction_method. Supported options: {','.join(list(corr_method_set))}; but got: {correction_method}"
                self.logger.error(msg)
                raise ValueError(msg)

        if not 0 < alpha <= 1:
            msg = f"Invalid alpha value. Must be in the range (0, 1], but got: {alpha}"
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(n_permutations, int) or n_permutations < 1:
            msg = f"Invalid 'n_permutations' value. Must be an integer greater than 0, but got: {n_permutations}"
            self.logger.error(msg)
            raise exceptions.PermutationInferenceError(msg)

        method = "dbscan" if use_dbscan else "permutation"

        self.logger.info(
            f"Starting Fst outlier detection ({method.capitalize()} method)..."
        )

        # Returns outlier SNPs, Fst-values, and adjusted
        # p-values per population pair in a long-form DataFrame.
        if use_dbscan:
            # Use DBSCAN method for outlier detection
            df_fst_outliers = fo.detect_fst_outliers_dbscan(
                correction_method,
                alpha,
                n_permutations,
                n_jobs,
                seed=seed,
                min_samples=min_samples,
            )
        else:
            df_fst_outliers = fo.detect_outliers_permutation(
                n_perm=n_permutations,
                correction_method=correction_method,
                alpha=alpha,
                seed=seed,
                alternative="upper",
            )

        try:
            self.plotter.plot_fst_outliers(
                df_fst_outliers,
                method=method,
                max_outliers_to_plot=max_outliers_to_plot,
            )
        except Exception as e:
            # Continue without plotting if an error occurs
            self.logger.warning(
                f"Error plotting Fst outliers: {e}. Plotting skipped for {method} method."
            )

        self.logger.info(f"{method.capitalize()} Fst outlier detection complete!")

        return df_fst_outliers

    def summary_statistics(
        self,
        fst_method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
        save_plots: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """Calculate a suite of summary statistics for SNP data.

        This method calculates heterozygosity, nucleotide diversity, and Fst. It can also perform permutation tests or bootstrapping for Fst. The method returns a DataFrame with allele-based summary statistics and a dictionary with other summary statistics.

        Args:
            fst_method (str): The Fst calculation method to use.
                - 'observed': (Default) Computes the observed Fst matrix.
                - 'permutation': Performs a permutation test to get p-values.
                - 'bootstrap': Performs a bootstrap to get confidence intervals.
            n_reps (int): Number of replicates for permutation or bootstrap.
            n_jobs (int): Number of parallel jobs (-1 for all cores).
            save_plots (bool): Whether to save plots of the summary statistics.

        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing a DataFrame with allele-based summary statistics and a dictionary with all other summary statistics.
        """
        allele_sumstats = AlleleSummaryStats(
            self.genotype_data, verbose=self.verbose, debug=self.debug
        )
        allele_sumstats_df = allele_sumstats.summarize()

        summary_stats = SummaryStatistics(
            self.genotype_data,
            self.alignment_012,
            self.plotter,
            verbose=self.verbose,
            debug=self.debug,
        )

        # Call the helper method with the new, clearer API
        sumstats = summary_stats.calculate_summary_statistics(
            fst_method=fst_method, n_reps=n_reps, n_jobs=n_jobs, save_plots=save_plots
        )

        return sumstats, allele_sumstats_df

    def amova(
        self,
        regionmap: Dict[str, str] | None = None,
        n_permutations: int = 0,
        n_jobs: int = 1,
        random_seed: int | None = None,
    ) -> Dict[str, float]:
        """Conduct AMOVA (Analysis Of Molecular Variance).

        AMOVA (Analysis of Molecular Variance) is a method for partitioning genetic variation into components due to differences among populations, among individuals within populations, and within individuals. This method calculates variance components and Phi statistics for a given number of hierarchical levels (1, 2, or 3). If bootstrapping is enabled, it also estimates p-values for the variance components. The number of hierarchical levels determines the structure of the AMOVA model: 1 => populations only, 2 => region -> populations, 3 => region -> population -> individuals. If regionmap is provided, it is used to map populations to regions.

        Notes:
            - Algorithm adapted from the R package 'poppr' (Kamvar et al., 2014).
            - The Phi statistic is a measure of genetic differentiation between populations.
            - Algorithm description: First, the total variance is calculated. Then, the variance components are calculated by summing the squared differences between the mean of each group and the global mean. The Phi statistic is calculated as the ratio of the among-group variance to the total variance. P-values are estimated by bootstrapping.

        Args:
            regionmap (dict, optional): Mapping from population_id -> region_id.
                If None but hierarchical_levels>1, raises ValueError.
            n_permutations (int): Number of bootstrap replicates across SNP loci.
            n_jobs (int): Number of parallel jobs. -1 uses all cores.
            random_seed (int | None): Random seed for reproducibility.

        Returns:
            dict: AMOVA results (variance components, Phi statistics, and possibly p-values).

        Raises:
            ValueError: If regionmap is required but not provided.
        """
        self.logger.info("Running AMOVA...")
        amova_instance = AMOVA(self.genotype_data, self.alignment, self.logger)

        amv = amova_instance.run(
            regionmap=regionmap,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )
        self.logger.info("AMOVA complete!")
        self.logger.debug(f"AMOVA results: {amv}")

        return amv

    def fst_distance(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
        palette: str = "viridis",
        suppress_plot: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Estimate Weir & Cockerham's Fst, optionally run statistical tests.

        This method calculates Weir & Cockerham's Fst between populations and can perform permutation tests for p-values or bootstrapping for 95 percent confidence intervals (CIs). The resulting Fst matrix can be visualized as a heatmap. The method returns a dictionary containing the observed Fst matrix, and optionally lower/upper confidence intervals or p-values depending on the chosen method.

        Args:
            method (str): The calculation method to use. 'observed': (Default) Computes the observed Fst matrix. 'permutation': Performs a permutation test to get p-values. 'bootstrap': Performs a bootstrap to get confidence intervals.
            n_reps (int): Number of replicates for 'permutation' or 'bootstrap'.
            n_jobs (int): Number of parallel jobs (-1 for all cores).
            palette (str): Color palette for the distance matrix heatmap.
            suppress_plot (bool): If True, suppresses plotting the heatmap.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the resulting DataFrames. Keys may include 'observed', 'pvalues', 'lower_ci', 'upper_ci'.
        """
        fst = FstDistance(
            self.genotype_data, self.plotter, verbose=self.verbose, debug=self.debug
        )
        self.logger.info(f"Calculating Weir & Cockerham Fst (method: '{method}')...")

        if len(list(set(self.populations))) < 2:
            msg = "At least two populations are required to calculate Fst."
            self.logger.error(msg)
            raise exceptions.MissingPopulationMapError(msg)

        # Call the backend with the new, explicit API
        fst_results = fst.weir_cockerham_fst(
            method=method, n_reps=n_reps, n_jobs=n_jobs
        )

        # The parser handles the output from any method
        df_obs, df_lower, df_upper, df_pval = fst.parse_wc_fst(fst_results)

        if not suppress_plot:
            # Plotting logic now depends on the method chosen
            self.plotter.plot_dist_matrix(
                df_obs,
                pvals=df_pval if method == "permutation" else None,
                palette=palette,
                title="SNPio: Weir & Cockerham $F_{ST}$",
                dist_type="fst",
            )

        self.logger.info("Fst calculation complete!")

        # Return a structured dictionary
        final_results = {"observed": df_obs}
        if df_lower is not None:
            final_results["lower_ci"] = df_lower
        if df_upper is not None:
            final_results["upper_ci"] = df_upper
        if df_pval is not None:
            final_results["pvalues"] = df_pval

        return final_results

    def _combine_upper_lower_ci(
        self, df_upper: pd.DataFrame, df_lower: pd.DataFrame, diagonal="zero"
    ) -> pd.DataFrame:
        """Combines two square DataFrames into one, using the upper triangle from one and the lower from the other.

        Args:
            df_upper (pd.DataFrame): DataFrame to provide the upper triangle (including diagonal if diagonal='upper').
            df_lower (pd.DataFrame): DataFrame to provide the lower triangle (including diagonal if diagonal='lower').
            diagonal (str): Which DataFrame should provide the diagonal values. Options are 'upper', 'lower', "zero", or 'nan'.

        Returns:
            pd.DataFrame: Combined DataFrame with upper/lower triangle values from respective inputs.

        Raises:
            ValueError: If input DataFrames are not square or do not match in shape.
        """
        if df_upper.shape != df_lower.shape:
            msg = "Both DataFrames must have the same shape."
            self.logger.error(msg)
            raise AssertionError(msg)

        if df_upper.shape[0] != df_upper.shape[1]:
            msg = "Input DataFrames must be square."
            self.logger.error(msg)
            raise AssertionError(msg)

        n = df_upper.shape[0]

        upper_mask = np.triu(np.ones((n, n)), k=1)
        lower_mask = np.tril(np.ones((n, n)), k=-1)
        diag_mask = np.eye(n)

        combined = np.full_like(df_upper, np.nan, dtype="float64")

        combined[upper_mask.astype(bool)] = df_upper.values[upper_mask.astype(bool)]
        combined[lower_mask.astype(bool)] = df_lower.values[lower_mask.astype(bool)]

        if diagonal == "upper":
            combined[diag_mask.astype(bool)] = df_upper.values[diag_mask.astype(bool)]
        elif diagonal == "lower":
            combined[diag_mask.astype(bool)] = df_lower.values[diag_mask.astype(bool)]
        elif diagonal == "nan":
            pass  # leave diagonal as NaN
        elif diagonal == "zero":
            np.fill_diagonal(combined, 0.0)
        else:
            msg = (
                "Invalid option for 'diagonal'. Choose from 'upper', 'lower', or 'nan'."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return pd.DataFrame(combined, index=df_upper.index, columns=df_upper.columns)

    def neis_genetic_distance(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
        palette: str = "magma",
        suppress_plot: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Calculate Nei's genetic distance and optionally run statistical tests.

        This method calculates Nei's genetic distance between populations and can perform permutation tests for p-values or bootstrapping for 95 percent confidence intervals (CIs). The resulting distance matrix can be visualized as a heatmap. The method returns a dictionary containing the observed distance matrix, and optionally lower/upper confidence intervals or p-values depending on the chosen method.

        Args:
            method (str): The calculation method to use. 'observed': (Default) Computes the observed Nei's distance matrix. 'permutation': Performs a permutation test to get p-values. 'bootstrap': Performs a bootstrap to get confidence intervals.
            n_reps (int): Number of replicates for 'permutation' or 'bootstrap'.
            n_jobs (int): Number of parallel jobs (-1 for all cores).
            palette (str): Color palette for the distance matrix heatmap.
            suppress_plot (bool): If True, suppresses plotting the heatmap.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the resulting DataFrames. Keys may include 'observed', 'pvalues', 'lower_ci', 'upper_ci'.
        """
        gd = GeneticDistance(
            self.genotype_data, self.plotter, verbose=self.verbose, debug=self.debug
        )
        self.logger.info(f"Calculating Nei's genetic distance (method: '{method}')...")

        if len(list(set(self.populations))) < 2:
            msg = "At least two populations are required to calculate Nei's genetic distance."
            self.logger.error(msg)
            raise exceptions.MissingPopulationMapError(msg)

        # Call the backend with the new, explicit API
        nei_results = gd.nei_distance(method=method, n_reps=n_reps, n_jobs=n_jobs)

        # The parser handles the output from any method
        df_obs, df_lower, df_upper, df_pval = gd.parse_nei_result(nei_results)

        if not suppress_plot:
            # ---- Force diagonals + clamp for MultiQC display copies ----
            if df_obs is not None:
                # Coerce all values to numeric.
                # Invalid strings (like "NaN") become np.nan.
                df_obs_plot = df_obs.apply(pd.to_numeric, errors="coerce")
                df_obs_plot = df_obs_plot.fillna(0.0)

            if df_pval is not None:
                # Coerce all values to numeric.
                # Invalid strings (like "NaN") become np.nan.
                df_pval_plot = df_pval.apply(pd.to_numeric, errors="coerce")
                df_pval_plot = df_pval_plot.fillna(1.0)

            # ----- Plotting & MultiQC queuing -----
            self.plotter.plot_dist_matrix(
                df_obs_plot,
                pvals=(df_pval_plot if method == "permutation" else None),
                palette=palette,
                title="SNPio: Nei's Genetic Distance",
                dist_type="nei",
            )

            # Unique panel IDs per method
            obs_id = {
                "observed": "nei_observed",
                "permutation": "nei_permutation_observed",
                "bootstrap": "nei_bootstrap_mean",
            }[method]

            title = (
                "SNPio: Nei's Genetic Distance — Observed"
                if method == "observed"
                else (
                    "SNPio: Nei's Genetic Distance — Observed (Permutation)"
                    if method == "permutation"
                    else "SNPio: Nei's Genetic Distance — Mean (Bootstrap)"
                )
            )

            # Observed / mean heatmap
            self.snpio_mqc.queue_heatmap(
                df=df_obs_plot,
                panel_id=obs_id,
                section="genetic_differentiation",
                title=title,
                description=(
                    "Observed Nei's (1972) genetic distance between pairwise populations."
                    if method == "observed"
                    else (
                        "Observed Nei's distance; significance assessed via permutation (p = Pr(D_perm ≥ D_obs)). This tests the null hypothesis of no genetic differentiation (D = 0). D_perm is the distance from permuted data; D_obs is the observed distance."
                        if method == "permutation"
                        else "Mean Nei's distance across bootstrap replicates (loci resampled with replacement). This provides a measure of uncertainty around the observed distance."
                    )
                ),
                index_label="Population",
                pconfig={
                    "title": f"Nei's Distance ({method.capitalize()})",
                    "id": obs_id,
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Nei's Genetic Distance",
                    "tt_decimals": 3,
                    "reverse_colors": False,
                    "min": 0.0,
                    "max": 1.0,
                },
            )

            # Permutation p-values heatmap (diagonal already = 1.0)
            if method == "permutation" and df_pval_plot is not None:
                pval_id = "nei_permutation_pvalues"
                self.snpio_mqc.queue_heatmap(
                    df=df_pval_plot,
                    panel_id=pval_id,
                    section="genetic_differentiation",
                    title="SNPio: Nei's Genetic Distance — P-values (Permutation)",
                    description="One-tailed permutation p-values: Pr(D_perm ≥ D_obs). This tests the null hypothesis of no genetic differentiation (D = 0). D_perm is the distance from permuted data; D_obs is the observed distance.",
                    index_label="Population",
                    pconfig={
                        "title": "Nei's Distance P-values (Permutation)",
                        "id": pval_id,
                        "xlab": "Population",
                        "ylab": "Population",
                        "zlab": "Permutation p-value",
                        "reverse_colors": True,
                        "tt_decimals": 3,
                        "min": 0.0,
                        "max": 1.0,
                    },
                )

            # 95% CI heatmap for bootstrap
            if method == "bootstrap" and df_lower is not None and df_upper is not None:
                df_combined = self._combine_upper_lower_ci(
                    df_upper.clip(0.0, 1.0),
                    df_lower.clip(0.0, 1.0),
                    diagonal="zero",
                )
                ci95_id = "nei_bootstrap_ci95"
                self.snpio_mqc.queue_heatmap(
                    df=df_combined,
                    panel_id=ci95_id,
                    section="genetic_differentiation",
                    title="SNPio: Nei's Genetic Distance — 95% CIs (Bootstrap)",
                    description=(
                        "95 percent confidence intervals from bootstrap replicates. Upper triangle: upper CI; lower triangle: lower CI. The lower and upper CIs are calculated as the 2.5th and 97.5th percentiles of the bootstrap distribution, respectively. Bootstrapping resamples loci with replacement to provide a measure of uncertainty around the observed distance."
                    ),
                    index_label="Population",
                    pconfig={
                        "title": "Nei's Distance 95% CI (Bootstrap)",
                        "id": ci95_id,
                        "xlab": "Population",
                        "ylab": "Population",
                        "zlab": "95% CI (Bootstrap)",
                        "tt_decimals": 3,
                        "min": 0.0,
                        "max": 1.0,
                    },
                )

            self.logger.info("Nei's genetic distance calculation complete!")

            # Return a structured dictionary, which is more robust than a tuple
            final_results = {"observed": df_obs}
            if df_lower is not None:
                final_results["lower_ci"] = df_lower
            if df_upper is not None:
                final_results["upper_ci"] = df_upper
            if df_pval is not None:
                final_results["pvalues"] = df_pval

            return final_results

    def pca(
        self,
        n_components: int | None = None,
        center: bool = True,
        scale: bool = False,
        n_axes: Literal[2, 3] = 2,
        seed: int | None = None,
        point_size: int = 15,
        bottom_margin: float = 0,
        top_margin: float = 0,
        left_margin: float = 0,
        right_margin: float = 0,
        width: int = 1088,
        height: int = 700,
        plot_format: Literal["png", "jpeg", "pdf", "svg"] | None = None,
    ) -> Tuple[np.ndarray, PCA]:
        """Run PCA on genotype data and generate a scatterplot colored by missing data proportions.

        This method performs Principal Component Analysis (PCA) on the genotype data, handling missing values and scaling as specified. It generates a scatterplot of the first two or three principal components, colored by the proportion of missing data per sample. The plot is saved as an interactive HTML file and as a static image format supplied as `plot_format` to ``<prefix>_output/analysis`` (if NRemover2 has not been run) or ``<prefix>_output/nremover/analysis`` (after filtering with NRemover2).

        The workflow is:

        1. **Encode genotypes** to 0/1/2 integers (-9 → missing, NaN → missing).
        2. **Impute missing values** with K-nearest neighbours (per sample).
        3. **Remove or neutral-ise constant loci** (zero variance after imputation).
        4. Optional **centering / scaling** (``StandardScaler``).
        5. **Sanity-check** the matrix to ensure it is finite (`np.nan_to_num`).
        6. **Fit PCA**, return components, and plot (2-D or 3-D).

        The plot is saved as an interactive HTML file and as a static image format supplied as `plot_format` to ``<prefix>_output/analysis`` (if NRemover2 has not been run) or ``<prefix>_output/nremover/analysis`` (after filtering with NRemover2) with the base name ``pca_missingness_mqc``. If `plot_format` is not specified, it defaults to the format supplied to the GenotypeData object (e.g., VCFReader, StructureReader, GenePopReader, PhylipReader).


        Args:
            n_components (int | None): Number of PCs to compute. ``None`` ⇒ all.
            center (bool): Subtract mean per locus before PCA.
            scale (bool): Divide by standard deviation per locus before PCA.
            n_axes (Literal[2, 3]): Number of axes to plot (must be integer of 2 or 3).
            seed (int | None): Random seed for reproducibility. Defaults to None.
            point_size (int): Size of points in the scatter.
            bottom_margin (float): Plotly margin. Defaults to 0.
            top_margin (float): Plotly margin. Defaults to 0.
            left_margin (float): Plotly margin. Defaults to 0.
            right_margin (float): Plotly margin. Defaults to 0.
            width (int): Output figure width (pixels). Defaults to 1088.
            height (int): Output figure height (pixels). Defaults to 700.
            plot_format (Literal["png", "jpeg", "pdf", "svg"] | None): Format to save the plot. If None, saves as the format supplied to the GenotypeData object (e.g., VCFReader, StructureReader, GenePopReader, PhylipReader). Defaults to None.

        Returns:
            Tuple[np.ndarray, PCA]: ``components``; shape ``(n_samples, n_components)`` and the fitted ``sklearn.decomposition.PCA`` object.

        Raises:
            ValueError: If ``n_axes`` is not an integer of 2 or 3.
            TypeError: If ``n_components`` is not an integer or None.
            RuntimeError: If PCA fails to converge.
            ValueError: If the input data is not suitable for PCA.
        """
        self.logger.info("Running Principal Component Analysis (PCA)...")

        if plot_format is not None:
            if plot_format not in {"png", "jpeg", "pdf", "svg"}:
                msg = f"Invalid plot format: {plot_format}. Supported formats: png, jpeg, pdf, svg."
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            plot_format = self.genotype_data.plot_kwargs.get("plot_format", "png")

        output_dir = Path(f"{self.genotype_data.prefix}_output")
        if self.genotype_data.was_filtered:
            output_dir = output_dir / "nremover" / "analysis"
        else:
            output_dir = output_dir / "analysis"

        output_dir.mkdir(exist_ok=True, parents=True)

        if self.genotype_data.was_filtered:
            description_prefix = "PCA (Post-NRemover2 Filtering): "
        else:
            description_prefix = "PCA (Pre-filtering): "

        with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
            if n_axes not in {2, 3}:
                msg = f"{n_axes} axes are not supported; choose 2 or 3 with the `n_axes` argument."
                self.logger.error(msg)
                raise ValueError(msg)

            # 1. Integer-encode genotypes; -9/“-9” → NaN
            # Use the pre-computed 012 matrix from __init__ for efficiency.
            df_raw = pd.DataFrame(self.alignment_012, dtype=float)
            df_raw = df_raw.replace([-9, "-9"], np.nan)

            # 2. Impute missing values
            imputer = KNNImputer(
                weights="uniform", n_neighbors=5, metric="nan_euclidean"
            )

            X_imp = imputer.fit_transform(df_raw)

            # 3. Drop / neutral-ise constant loci
            var = X_imp.var(axis=0)
            non_constant_mask = var > 0

            if (~non_constant_mask).any():
                dropped = (~non_constant_mask).sum()
                self.logger.warning(f"Dropping {dropped} loci with zero variance.")
                X_imp = X_imp[:, non_constant_mask]

            # 4. Center / scale (optional)
            if center or scale:
                scaler = StandardScaler(with_mean=center, with_std=scale)
                X_proc = scaler.fit_transform(X_imp)

                # Guard against divide-by-zero in scaling (rare but possible)
                if scale and hasattr(scaler, "scale_"):
                    zero_scale = scaler.scale_ == 0
                    if zero_scale.any():
                        X_proc[:, zero_scale] = 0.0
            else:
                X_proc = X_imp

            # 5. Final finite-value guard
            X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)

            if seed is not None and not isinstance(seed, int) and not seed > 0:
                msg = f"Invalid seed type ({type(seed)}) or value ({seed}). Expected positive int or None."
                self.logger.error(msg)
                raise TypeError(msg)

            # 6. PCA
            pca = PCA(n_components=n_components, random_state=seed)
            components = pca.fit_transform(X_proc)

            # 7. Assemble plotting DataFrame
            axis_labels = ["Axis1", "Axis2", "Axis3"][: max(3, n_axes)]
            df_pca = pd.DataFrame(
                components[:, : len(axis_labels)], columns=axis_labels
            )
            df_pca["SampleID"] = self.genotype_data.samples
            df_pca["Population"] = self.genotype_data.populations
            df_pca["Size"] = point_size

            # Missingness colouring
            miss_stats = self.genotype_data.calc_missing(df_raw, use_pops=False)

            df_pca["Missing Prop."] = miss_stats.per_individual.to_numpy()

            # Labels with % variance explained
            pc_var = pca.explained_variance_ratio_ * 100
            labels = {
                "Axis1": f"PC1 ({pc_var[0]:.2f}% Explained Variance)",
                "Axis2": f"PC2 ({pc_var[1]:.2f}% Explained Variance)",
                "Missing Prop.": "Missing Prop.",
                "Population": "Population",
            }
            if n_axes == 3:
                labels["Axis3"] = f"PC3 ({pc_var[2]:.2f}% Explained Variance)"

            # Plotly colour scale (ggplot-ish)
            my_scale = ["rgb(19, 43, 67)", "rgb(86, 177, 247)"]

            scatter_fn = px.scatter_3d if n_axes == 3 else px.scatter
            scatter_kwargs = dict(x="Axis1", y="Axis2")
            if n_axes == 3:
                scatter_kwargs["z"] = "Axis3"

            fig = scatter_fn(
                df_pca,
                **scatter_kwargs,
                color="Missing Prop.",
                symbol="Population",
                color_continuous_scale=my_scale,
                custom_data=["SampleID", "Population", "Missing Prop."],
                size="Size",
                size_max=point_size,
                labels=labels,
                range_color=[0.0, 1.0],
                title="SNPio: PCA with Points Colored by Per-population Missing Proportions",
            )

            # Nice hover + layout
            hover_lines = [
                "Axis 1: %{x}",
                "Axis 2: %{y}",
                "Sample ID: %{customdata[0]}",
                "Population: %{customdata[1]}",
                "Missing Prop.: %{customdata[2]}",
            ]
            if n_axes == 3:
                hover_lines.insert(2, "Axis 3: %{z}")

            fig.update_traces(hovertemplate="<br>".join(hover_lines))
            fig.update_layout(
                showlegend=True,
                margin=dict(
                    b=bottom_margin,
                    t=top_margin + 100,
                    l=left_margin,
                    r=right_margin,
                ),
                width=width,
                height=height,
                legend_orientation="h",
                legend_title="Population",
                legend_title_side="top",
                font=dict(size=24),
            )

            # 8. Save
            out_base = output_dir / "pca_missingness"
            fig.write_html(out_base.with_suffix(".html"))
            fig.write_image(out_base.with_suffix(f".{plot_format}"), format=plot_format)

        self.snpio_mqc.queue_table(
            df=df_pca,
            panel_id="pca_missingness",
            section="population_structure",
            title=f"SNPio: {description_prefix}PCA with Points Colored by Per-population Missing Proportions",
            description=(
                f"{description_prefix.capitalize()}PCA (Principal Component Analysis) scatterplot with points (samples) colored by per-population missing proportions and each population represented by distinct shapes."
            ),
            index_label="SampleID",
        )

        self.snpio_mqc.queue_html(
            out_base.with_suffix(".html"),
            panel_id="pca_missingness_scatter",
            section="population_structure",
            title=f"SNPio: {description_prefix}PCA Scatterplot with Points Colored by Per-population Missing Proportions",
            description=(
                f"{description_prefix.capitalize()}PCA (Principal Component Analysis) scatterplot with points (samples) colored by per-population missing proportions and each population represented by distinct shapes."
            ),
            index_label="SampleID",
        )

        self.logger.info("PCA completed successfully.")
        return components, pca
