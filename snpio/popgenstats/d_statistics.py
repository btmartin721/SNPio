import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm.contrib.itertools import product

from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.plotting.plotting import Plotting
from snpio.popgenstats.dfoil import DfoilStats
from snpio.popgenstats.dstat import PattersonDStats
from snpio.popgenstats.partd import PartitionedDStats

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class DStatistics:
    """Class to calculate D-statistics (Patterson's D, Partitioned D, and D-FOIL) with bootstrap support.

    This class provides methods to compute various D-statistics using genotype data, including Patterson's D, Partitioned D, and D-FOIL. It supports bootstrapping for statistical inference and can handle missing data represented as 'N' or '.' in genotype strings. The class uses Numba for efficient computation, especially for large datasets.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        alignment: np.ndarray,
        sample_ids: List[str],
        logger: logging.Logger,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize DStatistics object.

        This class is designed to compute D-statistics from genotype data. The initialization process involves encoding the genotype data and setting up the necessary data structures for subsequent analysis.

        Args:
            genotype_data (GenotypeData): Genotype data object containing genotype information.
            alignment (np.ndarray): 2D array of genotype strings (shape: n_samples x n_snps).
            sample_ids (List[str]): List of sample identifiers corresponding to alignment rows.
            logger (logging.Logger): Logger for diagnostic messages.
            verbose (bool): If True, enables verbose logging.
            debug (bool): If True, enables debug logging for detailed output.
        """
        self.genotype_data = genotype_data
        self.alignment = alignment
        self.sample_ids = sample_ids
        self.logger = logger
        self.verbose = verbose
        self.debug = debug

        # ACGT → 0, 1, 2, 3
        self._IUPAC_MAP = {
            "A": (0, 0),
            "C": (1, 1),
            "G": (2, 2),
            "T": (3, 3),
            "R": (0, 2),
            "Y": (1, 3),
            "S": (1, 2),
            "W": (0, 3),
            "K": (2, 3),
            "M": (0, 1),
            "N": (-1, -1),
            "-": (-1, -1),
            ".": (-1, -1),
            "?": (-1, -1),
        }

        self._allele_code = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "R": -2,
            "Y": -2,
            "S": -2,
            "W": -2,
            "K": -2,
            "M": -2,
            "N": -1,
            ".": -1,
        }

        self.valid_methods = {"patterson", "partitioned", "dfoil"}

        encoder = GenotypeEncoder(self.genotype_data)
        self.geno012 = encoder.genotypes_012.astype(int, copy=False)

    def run(
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
        seed: int | None = None,
        per_combination: bool = True,
        calc_overall: bool = True,
        use_jackknife=False,
        block_size: int = 500,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Wrapper that returns both per-quartet and overall D-stats.

        This method calculates D-statistics for the specified populations and returns both per-quartet and overall statistics. It supports various methods (Patterson's D, Partitioned D, D-FOIL) and can handle bootstrapping for statistical inference.

        Args:
            method (Literal["patterson", "partitioned", "dfoil"]): The D-statistics method to use.
            population1 (str | List[str]): The first population to compare.
            population2 (str | List[str]): The second population to compare.
            population3 (str | List[str]): The third population to compare.
            population4 (str | List[str] | None): The fourth population to compare (optional).
            outgroup (str | List[str]): The outgroup population.
            num_bootstraps (int): Number of bootstrap replicates. Ignored if `use_jackknife` is True.
            max_individuals_per_pop (int | None): Maximum individuals to sample per population (optional).
            individual_selection (str | Dict[str, List[str]]): Method for individual selection (random or specific).
            seed (int | None): Random seed for reproducibility (optional).
            per_combination (bool): Whether to calculate D-statistics for each combination of populations.
            calc_overall (bool): Whether to calculate overall D-statistics.
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrapping. Use this if you have possible linkage disequilibrium (LD) issues.
            block_size (int): Block size for jackknife resampling. Defaults to 500.

        Returns:
            Tuple[pd.DataFrame, Dict[str, float | int | str]]: DataFrame of per-quartet D-statistics and a dictionary of overall statistics.

        Raises:
            ValueError: If `method` is not one of the valid methods or if neither `calc_overall` nor `per_combination` is True.
            TypeError: If any of the population arguments are not strings or lists of strings.
        """
        method = self._validate_d_statistics_method(method)

        if not calc_overall and not per_combination:
            msg = "At least one of 'calc_overall' or 'per_combination' must be True."
            self.logger.error(msg)
            raise ValueError(msg)

        # 1) --- Get population indices -----------------------------
        pops = (population1, population2, population3, population4, outgroup)

        ds = self._get_all_pop_indices(
            *pops,
            max_individuals_per_pop=max_individuals_per_pop,
            individual_selection=individual_selection,
            seed=seed,
        )

        args = (method, self.geno012, *ds, num_bootstraps, seed)
        kwargs = {"use_jackknife": use_jackknife, "block_size": block_size}

        if calc_overall:
            # 5) --- Overall mode -----------------------------------
            self.logger.info("Calculating overall D-statistics...")

            overall_stats, bootstats = self._d_stats_overall(*args, **kwargs)

            self.logger.info("Overall D-statistics calculation completed.")

        if per_combination:
            # 2a) --- Per-combination mode -----------------------------
            self.logger.info("Calculating per-combination D-statistics...")

            df = self._d_stats_per_combination(*args, **kwargs)

            self.logger.info("Per-combination D-statistics calculation completed.")
        else:
            df = pd.DataFrame()

        df, overall_stats = self._proc_all_dstat_results(
            method, num_bootstraps, per_combination, calc_overall, df, overall_stats
        )

        return df, overall_stats

    def _d_stats_per_combination(
        self,
        method: Literal["patterson", "partitioned", "dfoil"],
        geno012: np.ndarray,
        population1: List[int],
        population2: List[int],
        population3: List[int],
        population4: List[int] | None = None,
        outgroup: List[int] | None = None,
        num_bootstraps: int = 1000,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> pd.DataFrame:
        """Calculate D-statistics for each combination of individuals.

        Args:
            method (Literal["patterson", "partitioned", "dfoil"]): The D-statistics method to use.
            geno012 (np.ndarray): Genotype matrix in 0/1/2 format.
            population1 (List[int]): List of indices for the first population.
            population2 (List[int]): List of indices for the second population.
            population3 (List[int]): List of indices for the third population.
            population4 (List[int] | None): List of indices for the fourth population (optional).
            outgroup (List[int] | None): List of indices for the outgroup population (optional).
            num_bootstraps (int): Number of bootstrap replicates. Ignored if `use_jackknife` is True.
            seed (int | None): Random seed for reproducibility (optional).
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrapping. Use this if you have possible linkage disequilibrium (LD) issues.
            block_size (int): Block size for jackknife resampling. Defaults to 500.

        Returns:
            pd.DataFrame: DataFrame containing D-statistics for each combination of individuals.
        """
        class_args = (self.genotype_data, geno012)
        class_kwargs = {"verbose": self.verbose, "debug": self.debug}
        if method == "patterson":
            dstat = PattersonDStats(*class_args, **class_kwargs)
        elif method == "partitioned":
            dstat = PartitionedDStats(*class_args, **class_kwargs)
        else:  # method == "dfoil"
            dstat = DfoilStats(*class_args, **class_kwargs)

        results_list = []
        sample_ids = self.genotype_data.samples

        # Prepare the population lists for itertools.product
        # For patterson's D, pop4 is not used, so we use a placeholder.
        pop4_iterable = population4 if method != "patterson" else [None]

        all_pop_combinations = product(
            population1,
            population2,
            population3,
            pop4_iterable,
            outgroup,
            desc=f"{method.capitalize()} D-stats",
            unit=" quartets",
        )

        for i1, i2, i3, i4, out in all_pop_combinations:
            if method == "patterson":
                popargs = (
                    np.array([i1]),
                    np.array([i2]),
                    np.array([i3]),
                    np.array([out]),
                )
            else:  # partitioned or dfoil
                popargs = (
                    np.array([i1]),
                    np.array([i2]),
                    np.array([i3]),
                    np.array([i4]),
                    np.array([out]),
                )

            combo, bootstats = dstat.calculate(
                *popargs,
                n_boot=num_bootstraps,
                seed=seed,
                use_jackknife=use_jackknife,
                block_size=block_size,
            )

            # Create the quartet/quintet name string.
            # `i1`, `i2`, etc., are single integer indices here.
            if method == "patterson":
                combo_str = f"{sample_ids[i1]}-{sample_ids[i2]}-{sample_ids[i3]}-{sample_ids[out]}"
            else:
                combo_str = f"{sample_ids[i1]}-{sample_ids[i2]}-{sample_ids[i3]}-{sample_ids[i4]}-{sample_ids[out]}"

            # Append results without incorrect indexing.
            # `combo['D']`, `combo['Z']`, etc., are single float values.
            if method == "patterson":
                results_list.append(
                    (
                        combo_str,
                        combo["D"],
                        combo["Z"],
                        combo["P"],
                        combo["X2"],
                        combo["P_X2"],
                        combo["Method"],
                        combo["Bootstraps"],
                        combo["Seed"],
                    )
                )
            elif method == "partitioned":
                results_list.append(
                    (
                        combo_str,
                        combo["D1"],
                        combo["D2"],
                        combo["D12"],
                        combo["Z_D1"],
                        combo["Z_D2"],
                        combo["Z_D12"],
                        combo["P_D1"],
                        combo["P_D2"],
                        combo["P_D12"],
                        combo["X2_D1"],
                        combo["X2_D2"],
                        combo["X2_D12"],
                        combo["P_X2_D1"],
                        combo["P_X2_D2"],
                        combo["P_X2_D12"],
                        combo["Method"],
                        combo["Bootstraps"],
                        combo["Seed"],
                    )
                )
            else:  # dfoil
                results_list.append(
                    (
                        combo_str,
                        combo["DFO"],
                        combo["DFI"],
                        combo["DOL"],
                        combo["DIL"],
                        combo["Z_DFO"],
                        combo["Z_DFI"],
                        combo["Z_DOL"],
                        combo["Z_DIL"],
                        combo["P_DFO"],
                        combo["P_DFI"],
                        combo["P_DOL"],
                        combo["P_DIL"],
                        combo["X2_DFO"],
                        combo["X2_DFI"],
                        combo["X2_DOL"],
                        combo["X2_DIL"],
                        combo["P_X2_DFO"],
                        combo["P_X2_DFI"],
                        combo["P_X2_DOL"],
                        combo["P_X2_DIL"],
                        combo["Method"],
                        combo["Bootstraps"],
                        combo["Seed"],
                    )
                )

        # 3. Create DataFrame from results
        if method == "patterson":
            columns = [
                "Quartet",
                "D-statistic",
                "Z",
                "P",
                "X2",
                "P_X2",
                "Method",
                "Bootstraps",
                "Seed",
            ]

        elif method == "partitioned":
            columns = [
                "Quartet",
                "D1",
                "D2",
                "D12",
                "Z_D1",
                "Z_D2",
                "Z_D12",
                "P_D1",
                "P_D2",
                "P_D12",
                "X2_D1",
                "X2_D2",
                "X2_D12",
                "P_X2_D1",
                "P_X2_D2",
                "P_X2_D12",
                "Method",
                "Bootstraps",
                "Seed",
            ]

        else:
            columns = [
                "Quartet",
                "DFO",
                "DFI",
                "DOL",
                "DIL",
                "Z_DFO",
                "Z_DFI",
                "Z_DOL",
                "Z_DIL",
                "P_DFO",
                "P_DFI",
                "P_DOL",
                "P_DIL",
                "X2_DFO",
                "X2_DFI",
                "X2_DOL",
                "X2_DIL",
                "P_X2_DFO",
                "P_X2_DFI",
                "P_X2_DOL",
                "P_X2_DIL",
                "Method",
                "Bootstraps",
                "Seed",
            ]

        results_list = [
            x[0] if isinstance(x, (tuple, list, np.ndarray)) and len(x) == 1 else x
            for x in results_list
        ]

        # 5. Return DataFrame with combination results
        # Create DataFrame with Quartet as index
        df = pd.DataFrame(results_list, columns=columns).set_index("Quartet")
        self.logger.info(f"Calculated {len(df)} per-combination D-statistics.")
        return df

    def _proc_all_dstat_results(
        self,
        method: Literal["patterson", "partitioned", "dfoil"],
        num_bootstraps: int,
        per_combination: bool,
        overall: bool,
        df: pd.DataFrame,
        overall_stats: dict[str, float] | None = None,
    ):
        """Process and save D-statistics results.

        This method processes the calculated D-statistics, saves them to a JSON file, and returns the DataFrame of per-combination D-statistics and overall statistics.

        Args:
            method (Literal["patterson", "partitioned", "dfoil"]): The D-statistics method used.
            num_bootstraps (int): The number of bootstrap replicates.
            per_combination (bool): Whether to compute per-combination D-statistics.
            overall (bool): Whether to compute overall D-statistics.
            df (pd.DataFrame): The DataFrame of per-combination D-statistics to process.
            overall_stats (dict[str, float]): The overall statistics to process.

        Returns:
            Tuple[pd.DataFrame, dict[str, float | tuple | list]]: DataFrame of per-combination D-statistics and a dictionary of overall statistics.
        """

        base = self._make_results_dir()

        if per_combination:
            # Index should be "Quartet" for per-combination results
            df = df.sort_index()

            if df.empty:
                msg = "No valid D-statistics were calculated. Returning empty outputs."
                self.logger.warning(msg)
            else:
                self.logger.info(f"Calculated {len(df)} per-combination D-statistics.")

                df = self._proc_dstats_per_combo(df, method=method)

                # Save per-combination results to JSON and CSV
                output_file = base / f"dstats_{method}_combinations.json"
                df.to_json(output_file, orient="records", indent=4)
                df.to_csv(output_file.with_suffix(".csv"), index=True)
        else:
            df = pd.DataFrame()

        if overall:
            overall_stats.update({"Method": method, "Bootstraps": num_bootstraps})
            overall_stats = {
                k: (
                    v[0]
                    if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 1
                    else v
                )
                for k, v in overall_stats.items()
            }

            # Save overall results to JSON
            output_file = base / f"dstats_{method}_overall.json"

            with open(output_file, "w") as f:
                json.dump(overall_stats, f, indent=4)

        return df, overall_stats

    def _make_results_dir(self) -> Path:
        """Create results directory based on output file or default path.

        This method creates a directory for saving results based on the provided output file path. If no output file is specified, it uses a default path based on the genotype data prefix.

        Returns:
            Path: The base directory for saving results.
        """
        base = Path(self.genotype_data.prefix + "_output")
        if self.genotype_data.was_filtered:
            base = base / "nremover"
        base = base / "analysis" / "d_stats"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _proc_dstats_per_combo(
        self,
        df: pd.DataFrame,
        method: Literal["patterson", "partitioned", "dfoil"] = "patterson",
    ):
        """Process per-combination D-statistics DataFrame.

        This method adds multiple testing corrections and significance flags to the DataFrame of per-combination D-statistics.

        Args:
            df (pd.DataFrame): DataFrame containing per-combination D-statistics.
            method (Literal["patterson", "partitioned", "dfoil"]): The D-statistics method used.

        Returns:
            pd.DataFrame: Processed DataFrame with significance flags and corrections.
        """
        # Ensure DataFrame is a copy to avoid SettingWithCopyWarning
        df = df.copy()

        if method == "patterson":
            # 5. Apply multiple testing corrections
            # Bonferroni and FDR-BH corrections for P-values
            bonf_key = "Bonferroni"
            fdr_key = "FDR-BH"

            # Create a mask for non-null p-values
            mask = df["P"].notna()

            # Initialize columns for significance flags
            df[bonf_key] = 1.0
            df[fdr_key] = 1.0
            df["Significant (Raw)"] = False
            df["Significant (Bonferroni)"] = False
            df["Significant (FDR-BH)"] = False

            # Bonferroni correction
            df.loc[mask, bonf_key] = multipletests(
                df.loc[mask, "P"].to_numpy(), method="bonferroni"
            )[1]

            # FDR-BH correction
            df.loc[mask, fdr_key] = multipletests(
                df.loc[mask, "P"].to_numpy(), method="fdr_bh"
            )[1]

            # Significance flags
            df.loc[mask, "Significant (Raw)"] = df.loc[mask, "P"] < 0.05
            df.loc[mask, "Significant (Bonferroni)"] = df.loc[mask, bonf_key] < 0.05
            df.loc[mask, "Significant (FDR-BH)"] = df.loc[mask, fdr_key] < 0.05

        elif method == "partitioned":
            # 5a. correction for partitioned D (D1, D2, D12)
            for stat in ["D1", "D2", "D12"]:
                key_p = f"P_{stat}"
                bonf_key = f"{key_p}_bonf"
                fdr_key = f"{key_p}_fdr"

                # Create a mask for non-null p-values
                mask = df[key_p].notna()

                # Initialize columns for significance flags
                df[bonf_key] = 1.0
                df[fdr_key] = 1.0
                df[f"Significant (Raw) {stat}"] = False
                df[f"Significant (Bonferroni) {stat}"] = False
                df[f"Significant (FDR-BH) {stat}"] = False

                # Bonferroni
                df.loc[mask, bonf_key] = multipletests(
                    df.loc[mask, key_p].to_numpy(), method="bonferroni"
                )[1]

                # FDR-BH
                df.loc[mask, fdr_key] = multipletests(
                    df.loc[mask, key_p].to_numpy(), method="fdr_bh"
                )[1]

                # significance flags
                df.loc[mask, f"Significant (Raw) {stat}"] = df.loc[mask, key_p] < 0.05
                df.loc[mask, f"Significant (Bonferroni) {stat}"] = (
                    df.loc[mask, bonf_key] < 0.05
                )
                df.loc[mask, f"Significant (FDR-BH) {stat}"] = (
                    df.loc[mask, fdr_key] < 0.05
                )

        elif method == "dfoil":
            # 5b. correction for D-FOIL (DFO, DFI, DOL, DIL)
            for stat in ["DFO", "DFI", "DOL", "DIL"]:
                key_p = f"P_{stat}"
                bonf_key = f"{key_p}_bonf"
                fdr_key = f"{key_p}_fdr"

                # Create a mask for non-null p-values
                mask = df[key_p].notna()

                # Initialize columns for significance flags
                df[bonf_key] = 1.0
                df[fdr_key] = 1.0
                df[f"Significant (Raw) {stat}"] = False
                df[f"Significant (Bonferroni) {stat}"] = False
                df[f"Significant (FDR-BH) {stat}"] = False

                # Bonferroni
                df.loc[mask, bonf_key] = multipletests(
                    df.loc[mask, key_p].to_numpy(), method="bonferroni"
                )[1]

                # FDR-BH
                df.loc[mask, fdr_key] = multipletests(
                    df.loc[mask, key_p].to_numpy(), method="fdr_bh"
                )[1]

                # significance flags
                df.loc[mask, f"Significant (Raw) {stat}"] = df.loc[mask, key_p] < 0.05
                df.loc[mask, f"Significant (Bonferroni) {stat}"] = (
                    df.loc[mask, bonf_key] < 0.05
                )
                df.loc[mask, f"Significant (FDR-BH) {stat}"] = (
                    df.loc[mask, fdr_key] < 0.05
                )
        else:
            msg = f"Unknown method: {method}. Supported: 'patterson', 'partitioned'."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        return df

    def _d_stats_overall(
        self,
        method: Literal["patterson", "partitioned", "dfoil"],
        geno012: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
        d3: np.ndarray,
        d4: np.ndarray | None,
        out: np.ndarray,
        n_boot: int,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Dict[str, Any]:
        """Calculate overall Patterson's (4-taxon) D-statistics.

        This method computes overall D-statistics based on the specified method (Patterson's D or Partitioned D). It uses bootstrapping to estimate the distribution of the D-statistic and calculates z-scores and p-values.

        Args:
            method (str): The D-statistics method to use. Valid options are "patterson", "partitioned", and "dfoil".
            geno012 (np.ndarray): Genotype data in 0/1/2 format.
            use_jackknife (bool): Whether to use jackknife resampling.
            d1 (np.ndarray): Population 1 indices to process.
            d2 (np.ndarray): Population 2 indices to process.
            d3 (np.ndarray): Population 3 indices to process.
            d4 (np.ndarray | None): Population 4 indices to process.
            out (np.ndarray): Outgroup indices to process.
            n_boot (int): Number of bootstrap replicates.
            seed (int | None): Random seed for reproducibility (optional). If None, a random seed will be used.
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrapping. Use this if you have possible linkage disequilibrium (LD) issues.
            block_size (int): Block size for jackknife resampling. Defaults to 500.

        Returns:
            Dict[str, Any]: A dictionary containing the overall D-statistics, z-scores, and p-values.
        """
        if method == "patterson":
            return self._patterson_d_overall(
                d1,
                d2,
                d3,
                out,
                n_boot,
                seed=seed,
                use_jackknife=use_jackknife,
                block_size=block_size,
            )

        elif method == "partitioned":
            return self._part_d_overall(
                d1,
                d2,
                d3,
                d4,
                out,
                n_boot,
                seed=seed,
                use_jackknife=use_jackknife,
                block_size=block_size,
            )

        else:  # D-FOIL
            return self._dfoil_overall(
                d1,
                d2,
                d3,
                d4,
                out,
                n_boot,
                seed=seed,
                use_jackknife=use_jackknife,
                block_size=block_size,
            )

    def _dfoil_overall(
        self,
        d1: np.ndarray,
        d2: np.ndarray,
        d3: np.ndarray,
        d4: np.ndarray,
        out: np.ndarray,
        n_boot: int,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Tuple[dict, np.ndarray]:
        """Calculate overall DFOIL D-statistics.

        This method computes overall DFOIL D-statistics using bootstrapping. It calculates the observed D-statistic and its z-score and p-value.

        Args:
            d1 (np.ndarray): Indices for population 1.
            d2 (np.ndarray): Indices for population 2.
            d3 (np.ndarray): Indices for population 3.
            d4 (np.ndarray): Indices for population 4.
            out (np.ndarray): Indices for outgroup.
            n_boot (int): Number of bootstrap replicates.
            seed (int | None): Random seed for reproducibility (optional).
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrapping. Use this if you have possible linkage disequilibrium (LD) issues.
            block_size (int): Block size for jackknife resampling. Defaults to 500.

        Returns:
            Dict[str, float | int | str]: A dictionary containing the overall DFOIL D-statistic, z-score, and p-value.
        """
        self.logger.info("Calculating overall DFOIL D-statistics...")

        dfs = DfoilStats(
            self.genotype_data, self.geno012, verbose=self.verbose, debug=self.debug
        )
        return dfs.calculate(
            population1=d1,
            population2=d2,
            population3=d3,
            population4=d4,
            outgroup=out,
            n_boot=n_boot,
            seed=seed,
            use_jackknife=use_jackknife,
            block_size=block_size,
        )

    def _part_d_overall(
        self,
        d1: np.ndarray,
        d2: np.ndarray,
        d3: np.ndarray,
        d4: np.ndarray,
        out: np.ndarray,
        n_boot: int,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Dict[str, float | int | str]:
        """Calculate overall Partitioned D-statistics.

        This method computes overall Partitioned D-statistics (D1, D2, D12) using bootstrapping. It calculates the observed D-statistics and their z-scores and p-values.

        Args:
            d1 (np.ndarray): Indices for population 1.
            d2 (np.ndarray): Indices for population 2.
            d3 (np.ndarray): Indices for population 3.
            d4 (np.ndarray): Indices for population 4.
            out (np.ndarray): Indices for outgroup.
            n_boot (int): Number of bootstrap replicates.
            seed (int | None): Random seed for reproducibility (optional).
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrapping. Use this if you have possible linkage disequilibrium (LD) issues.
            block_size (int): Block size for jackknife resampling. Defaults to 500.

        Returns:
            Dict[str, float | int | str]: A dictionary containing the overall Partitioned D-statistics, z-scores, and p-values.
        """
        self.logger.info("Calculating overall Partitioned D-statistics...")

        pds = PartitionedDStats(
            self.genotype_data, self.geno012, verbose=self.verbose, debug=self.debug
        )
        return pds.calculate(
            d1,
            d2,
            d3,
            d4,
            out,
            n_boot,
            seed=seed,
            use_jackknife=use_jackknife,
            block_size=block_size,
        )

    def _patterson_d_overall(
        self,
        d1: np.ndarray,
        d2: np.ndarray,
        d3: np.ndarray,
        out: np.ndarray,
        n_boot: int,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Dict[str, float | int | str]:
        """Calculate overall Patterson's D-statistics.

        This method computes overall Patterson's D-statistics using bootstrapping. It calculates the observed D-statistic and its z-score and p-value.

        Args:
            d1 (np.ndarray): Indices for population 1.
            d2 (np.ndarray): Indices for population 2.
            d3 (np.ndarray): Indices for population 3.
            out (np.ndarray): Indices for outgroup.
            n_boot (int): Number of bootstrap replicates.
            seed (int | None): Random seed for reproducibility (optional).
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrapping. Use this if you have possible linkage disequilibrium (LD) issues.
            block_size (int): Block size for jackknife resampling. Defaults to 500.

        Returns:
            Dict[str, float | int | str]: A dictionary containing the overall Patterson's D-statistic, z-score, and p-value.
        """
        self.logger.info("Calculating overall Patterson's D-statistics...")

        pds = PattersonDStats(
            self.genotype_data, self.geno012, verbose=self.verbose, debug=self.debug
        )
        return pds.calculate(
            population1=d1,
            population2=d2,
            population3=d3,
            outgroup=out,
            n_boot=n_boot,
            seed=seed,
            use_jackknife=use_jackknife,
            block_size=block_size,
        )

    def _validate_d_statistics_method(
        self, method: Literal["patterson", "partitioned", "dfoil"]
    ):
        """Validate the D-statistics method.

        This method checks if the provided method is a valid D-statistics method. It raises an error if the method is not recognized.

        Args:
            method (str): The D-statistics method to validate.

        Returns:
            str: The validated method in lowercase.

        Raises:
            TypeError: If the method is not a string.
            ValueError: If the method is not one of the valid methods.
        """
        if not isinstance(method, str):
            msg = f"'method' argument must be a string, but got: {type(method)}"
            self.logger.error(msg)
            raise TypeError(msg)

        method = method.lower()

        if method not in self.valid_methods:
            msg = f"Unsupported D-statistics method '{method}'. Valid methods are: {self.valid_methods}"
            self.logger.error(msg)
            raise ValueError(msg)

        return method

    def _get_single_pop_indices(
        self,
        pop: str | List[str],
        max_individuals_per_pop: int | None,
        individual_selection: Literal["random"] | Dict[str, List[str]],
        seed: int | None,
    ) -> List[int]:
        """Get indices of individuals from specified populations.

        This method selects individuals from specified populations based on the provided selection strategy. It supports both random selection and predefined lists of individual IDs.

        Args:
            pop (str | List[str]): Population ID or list of IDs to select individuals from.
            max_individuals_per_pop (int | None): Maximum number of individuals to select from each population.
            individual_selection (Literal["random"] | Dict[str, List[str]]): Selection strategy for individuals, either 'random' or a dictionary mapping population IDs to lists of individual IDs.
            seed (int | None): Random seed for reproducibility.

        Returns:
            List[int]: Indices of selected individuals in the genotype data.

        Raises:
            KeyError: If the specified population is not found in the population map.
            ValueError: If the individual selection strategy is invalid or if no individuals are selected.
            ValueError: If no individuals are selected after applying the selection criteria.
        """
        # Establish a random seed for reproducibility
        rng = np.random.default_rng(seed)

        pops = [pop] if isinstance(pop, str) else pop
        selected = []
        for p in pops:
            if p not in self.genotype_data.popmap_inverse:
                msg = f"Population '{p}' not found in popmap"
                self.logger.error(msg)
                raise KeyError(msg)
            samples = self.genotype_data.popmap_inverse[p]
            if (
                max_individuals_per_pop is not None
                and len(samples) > max_individuals_per_pop
            ):
                if individual_selection == "random":
                    chosen = rng.choice(samples, max_individuals_per_pop, replace=False)
                elif isinstance(individual_selection, dict):
                    if max_individuals_per_pop is not None:
                        chosen = individual_selection[p][:max_individuals_per_pop]
                    else:
                        chosen = individual_selection[p]
                else:
                    msg = f"Invalid individual_selection argument supplied: {individual_selection}. Must be 'random' or a dictionary mapping population IDs to lists of individual IDs. But got: {individual_selection} of type {type(individual_selection)}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                selected.extend(chosen)
            else:
                selected.extend(samples)

        if not selected:
            msg = f"No individuals selected for population(s): {pops}. Check your population definitions and individual selection criteria."
            self.logger.error(msg)
            raise ValueError(msg)

        # Map selected sample IDs to their indices in the genotype data
        return [self.genotype_data.samples.index(s) for s in selected]

    def _get_all_pop_indices(
        self,
        population1: str | List[str],
        population2: str | List[str],
        population3: str | List[str],
        population4: str | List[str],
        outgroup: str | List[str],
        max_individuals_per_pop: int | None = None,
        individual_selection: Literal["random"] | Dict[str, List[str]] = "random",
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get population index arrays in Comp-D partitioned D-statistics order using user-defined labels.

        Args:
            population1 (str | List[str]): Label of P1 population or list of individuals to use.
            population2 (str | List[str]): Label of P2 population or list of individuals to use.
            population3 (str | List[str]): Label of P3 population or list of individuals to use.
            population4 (str | List[str]): Label of P4 population or list of individuals to use.
            outgroup (str | List[str]): Label of outgroup or list of individuals to use.
            max_individuals_per_pop (int | None): Max number of individuals per population. Defaults to None (use all individuals).
            individual_selection (Literal["random"] | Dict[str, List[str]]): 'random' or dict[str, list[str]] for custom individual choices. Defaults to 'random'.
            seed (int | None): Optional seed for reproducibility.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (p1_inds, p2_inds, p3a_inds, p3b_inds, out_inds)
        """
        out_inds = np.array(
            self._get_single_pop_indices(
                outgroup, max_individuals_per_pop, individual_selection, seed
            )
        )

        p4_inds = None
        if population4 is not None:
            p4_inds = np.array(
                self._get_single_pop_indices(
                    population4, max_individuals_per_pop, individual_selection, seed
                )
            )
        p3_inds = np.array(
            self._get_single_pop_indices(
                population3, max_individuals_per_pop, individual_selection, seed
            )
        )
        p2_inds = np.array(
            self._get_single_pop_indices(
                population2, max_individuals_per_pop, individual_selection, seed
            )
        )
        p1_inds = np.array(
            self._get_single_pop_indices(
                population1, max_individuals_per_pop, individual_selection, seed
            )
        )

        return p1_inds, p2_inds, p3_inds, p4_inds, out_inds
