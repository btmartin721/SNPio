import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

from snpio.utils.logging import LoggerManager

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


@dataclass
class DStatsConfig:
    """Configuration for D-statistics calculations.

    This class holds parameters for calculating D-statistics, including the populations involved, number of bootstraps, and method used.

    Attributes:
        geno012 (np.ndarray): Genotype data in 012 format (-9 for missing). 0 = homozygous reference, 1 = heterozygous, 2 = homozygous alternate.
        pop1 (np.ndarray): Indices of individuals in population 1.
        pop2 (np.ndarray): Indices of individuals in population 2.
        pop3 (np.ndarray): Indices of individuals in population 3.
        pop4 (np.ndarray): Indices of individuals in population 4.
        outgroup (np.ndarray): Indices of individuals in the outgroup population.
        n_boot (int): Number of bootstrap replicates to compute.
        seed (int | None): Random seed for reproducibility. If None, uses a random seed.
        MISSING (int): Value representing missing data in the genotype matrix.
        EPS (float): Small value to avoid division by zero in calculations.
        method (str): Method for calculating D-statistics. Currently supports "patterson" for 4-taxon D-statistics.
    """

    geno012: np.ndarray
    pop1: np.ndarray
    pop2: np.ndarray
    pop3: np.ndarray
    pop4: np.ndarray | None
    outgroup: np.ndarray
    n_boot: int
    seed: int | None
    method: Literal["patterson", "partitioned", "dfoil"] = "patterson"
    MISSING: int = -9
    EPS: float = 1e-10

    def __post_init__(self):
        """Validate the configuration after initialization."""
        if not isinstance(self.geno012, np.ndarray):
            raise TypeError("geno012 must be a numpy.ndarray.")
        if not isinstance(self.pop1, np.ndarray):
            raise TypeError("pop1 must be a numpy.ndarray.")
        if not isinstance(self.pop2, np.ndarray):
            raise TypeError("pop2 must be a numpy.ndarray.")
        if not isinstance(self.pop3, np.ndarray):
            raise TypeError("pop3 must be a numpy.ndarray.")
        if not isinstance(self.pop4, (np.ndarray, type(None))):
            raise TypeError("pop4 must be a numpy.ndarray or None.")
        if not isinstance(self.outgroup, np.ndarray):
            raise TypeError("outgroup must be a numpy.ndarray.")
        if self.n_boot <= 1:
            raise ValueError("n_boot must be a positive integer greater than 1.")
        if self.method not in {"patterson", "partitioned", "dfoil"}:
            raise NotImplementedError(
                f"Method '{self.method}' is not implemented. Supported methods: 'patterson', 'partitioned', 'dfoil'."
            )
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("seed must be an integer or None.")

        self.rng = (
            np.random.default_rng(self.seed)
            if self.seed is not None
            else np.random.default_rng()
        )

        if self.method != "patterson" and self.pop4 is None:
            raise ValueError(
                "'pop4' must be specified for partitioned and D-FOIL methods."
            )

        self.pop3a = self.pop3
        self.pop3b = self.pop4 if self.method != "patterson" else None

        if self.method == "patterson":
            self.pops = [self.pop1, self.pop2, self.pop3, self.outgroup]
        else:
            # For partitioned D-statistics, we have two populations for the
            # third taxon
            self.pops = [self.pop1, self.pop2, self.pop3a, self.pop3b, self.outgroup]

        if self.method == "patterson":
            self.n_dstats = 1
        elif self.method == "partitioned":
            self.n_dstats = 3
        else:  # D-FOIL
            # D-FOIL has 4 D-statistics
            self.n_dstats = 4

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        This method returns a dictionary representation of the ``DStatsConfig`` instance, which can be useful for logging or serialization.

        Returns:
            dict: A dictionary containing the configuration parameters.
        """
        if self.method == "patterson":
            return {
                "geno012": self.geno012.shape,
                "pop1": self.pop1.shape,
                "pop2": self.pop2.shape,
                "pop3": self.pop3a.shape,
                "outgroup": self.outgroup.shape,
                "n_boot": self.n_boot,
                "seed": self.seed,
                "MISSING": self.MISSING,
                "method": self.method,
            }

        else:
            return {
                "geno012": self.geno012.shape,
                "pop1": self.pop1.shape,
                "pop2": self.pop2.shape,
                "pop3": self.pop3a.shape,
                "pop4": self.pop3b.shape if self.pop3b is not None else None,
                "outgroup": self.outgroup.shape,
                "n_boot": self.n_boot,
                "seed": self.seed,
                "MISSING": self.MISSING,
                "method": self.method,
            }


@dataclass
class DStatsResults:
    """Results of D-statistics calculations.

    This class holds the results of D-statistics calculations, including the observed D-statistic, Z-score, p-value, chi-squared statistic, and method used.

    Attributes:
        D (float | None): Observed D-statistic.
        Z (float | None): Z-score of the D-statistic.
        P (float | None): p-value associated with the Z-score.
        X2 (float | None): Chi-squared statistic.
        P_X2 (float | None): p-value associated with the chi-squared statistic.
        D1 (float | None): D-statistic for population 1.
        D2 (float | None): D-statistic for population 2.
        D12 (float | None): D-statistic for populations 1 and 2.
        Z_D1 (float | None): Z-score for D1.
        Z_D2 (float | None): Z-score for D2.
        Z_D12 (float | None): Z-score for D12.
        P_D1 (float | None): p-value for D1.
        P_D2 (float | None): p-value for D2.
        P_D12 (float | None): p-value for D12.
        X2_D1 (float | None): Chi-squared statistic for D1.
        X2_D2 (float | None): Chi-squared statistic for D2.
        X2_D12 (float | None): Chi-squared statistic for D12.
        P_X2_D1 (float | None): p-value for chi-squared statistic of D1.
        P_X2_D2 (float | None): p-value for chi-squared statistic of D2.
        P_X2_D12 (float | None): p-value for chi-squared statistic of D12.
        n_boot (int): Number of bootstrap replicates used.
        seed (int | None): Random seed used for reproducibility, or None if random.
        method (Literal["patterson", "partitioned", "dfoil"]): Method used for calculating the D-statistic.
    """

    n_boot: int
    seed: int | None

    # Patterson D-statistics
    D: float | None = None
    Z: float | None = None
    P: float | None = None
    X2: float | None = None
    P_X2: float | None = None

    # Partitioned D-statistics
    D1: float | None = None
    D2: float | None = None
    D12: float | None = None
    Z_D1: float | None = None
    Z_D2: float | None = None
    Z_D12: float | None = None
    P_D1: float | None = None
    P_D2: float | None = None
    P_D12: float | None = None
    X2_D1: float | None = None
    X2_D2: float | None = None
    X2_D12: float | None = None
    P_X2_D1: float | None = None
    P_X2_D2: float | None = None
    P_X2_D12: float | None = None

    # DFOIL statistics
    DFO: float | None = None
    DFI: float | None = None
    DOL: float | None = None
    DIL: float | None = None
    Z_DFO: float | None = None
    Z_DFI: float | None = None
    Z_DOL: float | None = None
    Z_DIL: float | None = None
    P_DFO: float | None = None
    P_DFI: float | None = None
    P_DOL: float | None = None
    P_DIL: float | None = None
    X2_DFO: float | None = None
    X2_DFI: float | None = None
    X2_DOL: float | None = None
    X2_DIL: float | None = None
    P_X2_DFO: float | None = None
    P_X2_DFI: float | None = None
    P_X2_DOL: float | None = None
    P_X2_DIL: float | None = None

    method: Literal["patterson", "partitioned", "dfoil"] = "patterson"

    def __post_init__(self):
        """Validate the results after initialization."""
        if not isinstance(self.D, (float, tuple, list, np.ndarray, type(None))):
            raise TypeError("D must be a float, tuple, list, or np.ndarray.")
        if not isinstance(self.X2, (float, tuple, list, np.ndarray, type(None))):
            raise TypeError("X2 must be a float, tuple, list, or np.ndarray.")
        if not isinstance(self.P_X2, (float, tuple, list, np.ndarray, type(None))):
            raise TypeError("P_X2 must be a float, tuple, list, or np.ndarray.")
        if not isinstance(self.n_boot, int) or self.n_boot <= 1:
            raise ValueError("n_boot must be a positive integer greater than 1.")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("seed must be an integer or None.")

        if self.method not in {"patterson", "partitioned", "dfoil"}:
            raise NotImplementedError(
                f"Method '{self.method}' is not implemented. Supported methods: 'patterson', 'partitioned', 'dfoil'."
            )

        self.rng = (
            np.random.default_rng(self.seed)
            if self.seed is not None
            else np.random.default_rng()
        )

        if self.method == "patterson":
            assert self.D is not None, "D must be provided."
            assert self.Z is not None, "Z must be provided."
            assert self.P is not None, "P must be provided."
            assert self.X2 is not None, "X2 must be provided."
            assert self.P_X2 is not None, "P_X2 must be provided."

        elif self.method == "partitioned":
            assert self.D1 is not None, "D1 must be provided."
            assert self.D2 is not None, "D2 must be provided."
            assert self.D12 is not None, "D12 must be provided."
            assert self.Z_D1 is not None, "Z_D1 must be provided."
            assert self.Z_D2 is not None, "Z_D2 must be provided."
            assert self.Z_D12 is not None, "Z_D12 must be provided."
            assert self.P_D1 is not None, "P_D1 must be provided."
            assert self.P_D2 is not None, "P_D2 must be provided."
            assert self.P_D12 is not None, "P_D12 must be provided."
            assert self.X2_D1 is not None, "X2_D1 must be provided."
            assert self.X2_D2 is not None, "X2_D2 must be provided."
            assert self.X2_D12 is not None, "X2_D12 must be provided."
            assert self.P_X2_D1 is not None, "P_X2_D1 must be provided."
            assert self.P_X2_D2 is not None, "P_X2_D2 must be provided."
            assert self.P_X2_D12 is not None, "P_X2_D12 must be provided."

        else:  # D-FOIL
            assert self.DFO is not None, "DFO must be provided."
            assert self.DFI is not None, "DFI must be provided."
            assert self.DOL is not None, "DOL must be provided."
            assert self.DIL is not None, "DIL must be provided."
            assert self.Z_DFO is not None, "Z_DFO must be provided."
            assert self.Z_DFI is not None, "Z_DFI must be provided."
            assert self.Z_DOL is not None, "Z_DOL must be provided."
            assert self.Z_DIL is not None, "Z_DIL must be provided."
            assert self.P_DFO is not None, "P_DFO must be provided."
            assert self.P_DFI is not None, "P_DFI must be provided."
            assert self.P_DOL is not None, "P_DOL must be provided."
            assert self.P_DIL is not None, "P_DIL must be provided."
            assert self.X2_DFO is not None, "X2_DFO must be provided."
            assert self.X2_DFI is not None, "X2_DFI must be provided."
            assert self.X2_DOL is not None, "X2_DOL must be provided."
            assert self.X2_DIL is not None, "X2_DIL must be provided."
            assert self.P_X2_DFO is not None, "P_X2_DFO must be provided."
            assert self.P_X2_DFI is not None, "P_X2_DFI must be provided."
            assert self.P_X2_DOL is not None, "P_X2_DOL must be provided."
            assert self.P_X2_DIL is not None, "P_X2_DIL must be provided."

    def to_dict(self) -> dict:
        """Convert the results to a dictionary.

        This method returns a dictionary representation of the DStatsResults instance, which can be useful for logging or serialization.

        Returns:
            dict: A dictionary containing the configuration parameters.
        """
        if self.method == "patterson":
            return {
                "D": self.D,
                "Z": self.Z,
                "P": self.P,
                "X2": self.X2,
                "P_X2": self.P_X2,
                "Method": self.method,
                "Bootstraps": self.n_boot,
                "Seed": self.seed,
            }
        elif self.method == "partitioned":
            return {
                "D1": self.D1,
                "D2": self.D2,
                "D12": self.D12,
                "Z_D1": self.Z_D1,
                "Z_D2": self.Z_D2,
                "Z_D12": self.Z_D12,
                "P_D1": self.P_D1,
                "P_D2": self.P_D2,
                "P_D12": self.P_D12,
                "X2_D1": self.X2_D1,
                "X2_D2": self.X2_D2,
                "X2_D12": self.X2_D12,
                "P_X2_D1": self.P_X2_D1,
                "P_X2_D2": self.P_X2_D2,
                "P_X2_D12": self.P_X2_D12,
                "Method": self.method,
                "Bootstraps": self.n_boot,
                "Seed": self.seed,
            }

        elif self.method == "dfoil":
            return {
                "DFO": self.DFO,
                "DFI": self.DFI,
                "DOL": self.DOL,
                "DIL": self.DIL,
                "Z_DFO": self.Z_DFO,
                "Z_DFI": self.Z_DFI,
                "Z_DOL": self.Z_DOL,
                "Z_DIL": self.Z_DIL,
                "P_DFO": self.P_DFO,
                "P_DFI": self.P_DFI,
                "P_DOL": self.P_DOL,
                "P_DIL": self.P_DIL,
                "X2_DFO": self.X2_DFO,
                "X2_DFI": self.X2_DFI,
                "X2_DOL": self.X2_DOL,
                "X2_DIL": self.X2_DIL,
                "P_X2_DFO": self.P_X2_DFO,
                "P_X2_DFI": self.P_X2_DFI,
                "P_X2_DOL": self.P_X2_DOL,
                "P_X2_DIL": self.P_X2_DIL,
                "Method": self.method,
                "Bootstraps": self.n_boot,
                "Seed": self.seed,
            }

        else:
            raise NotImplementedError(
                f"Method '{self.method}' is not implemented. Supported methods: 'patterson', 'partitioned', 'dfoil'."
            )

    def log_results(self, logger):
        """Log the results using the provided logger.

        This method logs the D-statistic results in a human-readable format.

        Args:
            logging.Logger: Logger instance to log the results.
        """
        logger = LoggerManager(__name__, to_file=False).get_logger()
        for key, value in self.to_dict().items():
            logger.info(
                f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            )


class DStatsBase:
    """Base class for D-statistics calculations.

    Provides methods for computing discrete-count and frequency-based D-statistics, bootstrap/jackknife resampling, summary statistics, and plotting/logging.

    Attributes:
        genotype_data (GenotypeData): SNP data container.
        verbose (bool): If True, enables verbose logging.
        debug (bool): If True, enables debug logging.
        logger: configured logger instance.
    """

    def __init__(
        self, genotype_data: "GenotypeData", verbose: bool = False, debug: bool = False
    ) -> None:
        self.genotype_data = genotype_data
        self.verbose = verbose
        self.debug = debug

        logman = LoggerManager(
            __name__, prefix=self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()

    def summary_statistics(self, boots: np.ndarray) -> dict:
        """Summarize bootstrap replicates: mean, std (ddof=1), and 95% CI.

        This method computes the mean, standard deviation, and 95% confidence intervals for each statistic across bootstrap replicates.

        Args:
            boots (np.ndarray): Bootstrap replicates of D-statistics.

        Returns:
            dict: Dictionary containing:
                - "mean": Mean of bootstrap replicates.
                - "std": Standard deviation of bootstrap replicates (ddof=1).
                - "95% CI lower": Lower bound of the 95% confidence interval.
                - "95% CI upper": Upper bound of the 95% confidence interval.
        """
        if np.any(np.all(np.isnan(boots), axis=0)):
            bad = np.where(np.all(np.isnan(boots), axis=0))[0]
            self.logger.warning(f"Bootstrap contains all-NaN columns: {bad.tolist()}")
        means = np.nanmean(boots, axis=0)
        stds = np.nanstd(boots, axis=0, ddof=1)
        ci_lo = np.nanpercentile(boots, 2.5, axis=0)
        ci_hi = np.nanpercentile(boots, 97.5, axis=0)
        return {
            "mean": means,
            "std": stds,
            "95% CI lower": ci_lo,
            "95% CI upper": ci_hi,
        }

    def multiple_test_correction(
        self,
        pvals: np.ndarray,
        method: Literal["bonferroni", "fdr_bh"] = "bonferroni",
    ) -> np.ndarray:
        """Apply multiple test correction to P-values.

        This method applies either Bonferroni or FDR correction to the provided P-values.

        Args:
            pvals (np.ndarray): Array of P-values to correct.
            method (Literal): Correction method, either "bonferroni" or "fdr_bh".

        Raises:
            ValueError: If an invalid method is specified.

        Returns:
            np.ndarray: Corrected P-values.
        """
        if method not in {"bonferroni", "fdr_bh"}:
            msg = (
                f"Invalid method: {method}. Supported methods: 'bonferroni', 'fdr_bh'."
            )
            self.logger.error(msg)
            raise ValueError(f"Invalid method: {method}")

        _, corrected_pvals, _, _ = multipletests(pvals, method=method)
        return corrected_pvals

    def zscore(
        self,
        observed: np.ndarray | list[float] | tuple[float, ...],
        boots: np.ndarray | list[float] | tuple[float, ...],
        *,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Z-scores and two-sided P-values from bootstrap/jackknife replicates.

        Guards against zero/near-zero SD so divisions do not explode to inf.

        Args:
            observed (np.ndarray | list[float] | tuple[float, ...]): Observed statistics, length = n_stats.
            boots (np.ndarray): Replicate statistics, shape (n_reps, n_stats).
            eps (float): Minimum SD treated as nonzero.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (z_scores, p_values), each shape (n_stats,).
        """
        observed = np.array(observed, dtype=float)

        if not isinstance(boots, np.ndarray):
            boots = np.array(boots, dtype=float)

        if boots.ndim != 2 or boots.shape[1] != observed.size:
            raise ValueError(
                f"`boots` must be (n_reps, {observed.size}); got {boots.shape}."
            )

        stds = np.nanstd(boots, axis=0, ddof=1)

        # keep only finite, > eps
        good = (~np.isnan(stds)) & (stds > eps)

        if np.any(~good):
            bad_ix = np.where(~good)[0].tolist()
            self.logger.warning(
                f"Unreliable variance for stats at indices {bad_ix}; Z and P set to NaN."
            )

        z_scores = np.full_like(observed, np.nan, dtype=float)
        p_vals = np.full_like(observed, np.nan, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores[good] = observed[good] / stds[good]
            p_vals[good] = 2.0 * (1.0 - norm.cdf(np.abs(z_scores[good])))

        return z_scores, p_vals

    def pvalue(
        self,
        z_scores: np.ndarray,
        alternative: Literal["two-sided", "upper", "lower"] = "two-sided",
    ) -> np.ndarray:
        """Compute P-values for Z-scores under the normal distribution.

        Args:
            z_scores (np.ndarray): Array of Z-scores.
            alternative (Literal["two-sided", "upper", "lower"]): Alternative hypothesis: ``"two-sided"``, ``"upper"``,
                or ``"lower"``.

        Returns:
            np.ndarray: Array of P-values.

        Raises:
            ValueError: If ``alternative`` is invalid.
        """
        if alternative not in {"two-sided", "upper", "lower"}:
            msg = f"Invalid alternative: {alternative}"
            self.logger.error(msg)
            raise ValueError(msg)

        z_scores = np.asarray(z_scores, dtype=np.float64)
        pvals = np.full(z_scores.shape, np.nan, dtype=np.float64)

        finite = np.isfinite(z_scores)

        if alternative == "two-sided":
            pvals[finite] = 2.0 * (1.0 - norm.cdf(np.abs(z_scores[finite])))
        elif alternative == "upper":
            pvals[finite] = 1.0 - norm.cdf(z_scores[finite])
        else:
            pvals[finite] = norm.cdf(z_scores[finite])

        return pvals

    def chisq(
        self,
        observed: Sequence[float],
        boots: np.ndarray | list[float] | tuple[float, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute chi-squared stats and P-values for each observed D vs bootstrap variance.

        This method computes the chi-squared statistic for each observed value against the variance of the bootstrap replicates. It returns both the chi-squared statistics and their corresponding P-values.

        Args:
            observed (Sequence[float]): Observed D-statistics.
            boots (np.ndarray | list[float] | tuple[float, ...]): Bootstrap replicates of D-statistics.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing chisqs (np.ndarray): Chi-squared statistics for each observed value. pvals (np.ndarray): P-values corresponding to the chi-squared statistics.
        """
        chisqs = np.zeros(len(observed), dtype=float)
        pvals = np.zeros(len(observed), dtype=float)

        if not isinstance(boots, np.ndarray):
            boots = np.array(boots, dtype=float)

        for i, obs in enumerate(observed):
            var = np.nanvar(boots[:, i], ddof=1)
            chisqs[i] = obs**2 / var if var > 1e-10 else np.nan
            pvals[i] = chi2.sf(chisqs[i], df=1) if var > 1e-10 else np.nan
        return chisqs, pvals

    def to_dataframe(
        self,
        observed: np.ndarray,
        boots: np.ndarray,
        labels: list[str] | None = None,
        method_name: str = "D",
    ) -> pd.DataFrame:
        """Convert observed and bootstrap summaries into a pandas DataFrame.

        This method creates a DataFrame containing the observed statistics, mean, standard deviation, and 95% confidence intervals from the bootstrap replicates.

        Args:
            observed (np.ndarray): Observed statistics.
            boots (np.ndarray): Bootstrap replicates.
            labels (list[str] | None): Optional labels for the rows.
            method_name (str): Name of the method for the statistics, used in column names.

        Returns:
            pd.DataFrame: DataFrame containing observed, mean, std, and 95% CI values.
        """
        stats = self.summary_statistics(boots)
        df = pd.DataFrame(
            {
                f"{method_name}_obs": observed,
                f"{method_name}_mean": stats["mean"],
                f"{method_name}_std": stats["std"],
                f"{method_name}_CI_low": stats["95% CI lower"],
                f"{method_name}_CI_high": stats["95% CI upper"],
            }
        )
        if labels is not None:
            df.insert(0, "Label", labels)
        return df

    def jackknife_indices(self, n_snps: int, block_size: int = 500) -> np.ndarray:
        """Generate leave-one-block-out jackknife index sets.

        Truncates to full blocks. If n_snps < block_size, reduces block_size to
        the largest valid value >= 1 that yields at least one block. Raises if
        impossible.

        Args:
            n_snps (int): Total SNP count.
            block_size (int): Block length to omit per replicate.

        Returns:
            np.ndarray: Shape (n_blocks, usable_snps - block_size) of kept indices.

        Raises:
            ValueError: If no valid jackknife blocks can be formed.
        """
        if not isinstance(n_snps, int) or n_snps <= 1:
            raise ValueError(
                "Jackknife requires n_snps to be an integer greater than 1."
            )

        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer.")

        if n_snps <= 1:
            raise ValueError("Jackknife requires at least 2 SNPs.")

        # Downshift block_size if needed
        if n_snps < block_size:
            new_bs = max(1, n_snps // 2)
            self.logger.warning(
                f"jackknife block_size ({block_size}) > n_snps ({n_snps}); "
                f"reducing block_size to {new_bs}."
            )
            block_size = new_bs

        n_blocks = n_snps // block_size
        if n_blocks < 1:
            # as a last resort, use leave-one-out by SNP
            if n_snps >= 2:
                self.logger.warning(
                    "Could not form any full blocks; using leave-one-out jackknife."
                )
                block_size = 1
                n_blocks = n_snps
            else:
                raise ValueError("Not enough SNPs for jackknife.")

        usable_snps = n_blocks * block_size
        if usable_snps - block_size < 1:
            raise ValueError(
                f"Jackknife would leave zero SNPs per replicate: "
                f"n_snps={n_snps}, block_size={block_size}."
            )

        all_snps = np.arange(usable_snps, dtype=np.int32)
        out = np.empty((n_blocks, usable_snps - block_size), dtype=np.int32)
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            keep = np.concatenate([all_snps[:start], all_snps[end:]])
            out[i] = keep
        return out

    def bootstrap_indices(
        self, n_snps: int, n_bootstraps: int, seed: int | None = None
    ) -> np.ndarray:
        """Return bootstrap-resampled SNP indices.

        Args:
            n_snps: Number of SNPs.
            n_bootstraps: Number of bootstrap replicates.
            seed: Optional random seed.

        Returns:
            np.ndarray: Integer array with shape ``(n_bootstraps, n_snps)``.

        Raises:
            ValueError: If ``n_snps`` or ``n_bootstraps`` is invalid.
        """
        if not isinstance(n_snps, int) or n_snps <= 0:
            raise ValueError("n_snps must be a positive integer.")

        if not isinstance(n_bootstraps, int) or n_bootstraps <= 0:
            raise ValueError("n_bootstraps must be a positive integer.")

        rng = np.random.default_rng(seed)
        return rng.choice(n_snps, size=(n_bootstraps, n_snps), replace=True)

    def _map_geno_to_pops(
        self, geno012: np.ndarray, config: DStatsConfig
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Map the genotype matrix to population-specific row indices.

        Args:
            geno012: Genotype matrix with shape ``(n_samples, n_snps)``.
            config: D-statistics configuration containing population sample indices.

        Returns:
            tuple[np.ndarray, tuple[np.ndarray, ...]]: A tuple containing the subsetted genotype matrix containing all unique population samples and a tuple of mapped population indices relative to the subsetted matrix.

        Raises:
            ValueError: If input dimensions, population indices, or overlaps are invalid.
            IndexError: If population indices are out of bounds.
            TypeError: If population indices are not integers.
        """
        if not isinstance(geno012, np.ndarray) or geno012.ndim != 2:
            msg = "`geno012` must be a 2D numpy array."
            self.logger.error(msg)
            raise ValueError(msg)

        n_samples, n_snps = geno012.shape
        if n_samples == 0 or n_snps == 0:
            msg = "`geno012` must have at least one sample and one SNP."
            self.logger.error(msg)
            raise ValueError(msg)

        pops = tuple(config.pops)
        if len(pops) == 0:
            msg = "At least one population must be provided."
            self.logger.error(msg)
            raise ValueError(msg)

        cleaned_pops: list[np.ndarray] = []
        seen_global: set[int] = set()

        for pop_idx, inds in enumerate(pops):
            inds = np.asarray(inds)

            if inds.ndim != 1:
                msg = f"Population {pop_idx} indices must be one-dimensional."
                self.logger.error(msg)
                raise ValueError(msg)

            if inds.size == 0:
                msg = f"Population {pop_idx} is empty."
                self.logger.error(msg)
                raise ValueError(msg)

            if not np.issubdtype(inds.dtype, np.integer):
                msg = f"Population {pop_idx} indices must be integers."
                self.logger.error(msg)
                raise TypeError(msg)

            if np.any(inds < 0) or np.any(inds >= n_samples):
                bad = inds[(inds < 0) | (inds >= n_samples)]
                msg = f"Population {pop_idx} contains out-of-bounds sample indices: {bad.tolist()} for n_samples={n_samples}."
                self.logger.error(msg)
                raise IndexError(msg)

            uniq_local = np.unique(inds)
            if uniq_local.size != inds.size:
                msg = f"Population {pop_idx} contains duplicate sample indices. Duplicates would overweight samples in population-frequency estimates."
                self.logger.error(msg)
                raise ValueError(msg)

            overlap = seen_global.intersection(inds.tolist())
            if overlap:
                msg = f"Population {pop_idx} shares sample indices with another population: {sorted(overlap)}."
                self.logger.error(msg)
                raise ValueError(msg)

            seen_global.update(inds.tolist())
            cleaned_pops.append(inds.astype(int, copy=False))

        all_inds = np.concatenate(cleaned_pops)
        uniq, first_pos = np.unique(all_inds, return_index=True)
        uniq = uniq[np.argsort(first_pos)]

        geno_sub = geno012[uniq, :]

        inv = {int(orig): i for i, orig in enumerate(uniq)}
        pops_mapped = tuple(
            np.array([inv[int(i)] for i in grp], dtype=np.int64) for grp in cleaned_pops
        )

        return geno_sub, pops_mapped

    def _extract_pop_freqs(
        self,
        config: DStatsConfig,
        geno_sub: np.ndarray,
        pops_mapped: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """Extract polarized population allele frequencies from 012 genotypes.

        Genotypes are interpreted as alternate-allele dosage:
        ``0 = homozygous reference``, ``1 = heterozygous``, ``2 = homozygous alternate``, and ``config.MISSING`` is treated as missing. Sites are retained only if every population has at least one non-missing genotype and the outgroup is not tied at frequency 0.5. Sites where the outgroup alternate-allele frequency is greater than 0.5 are flipped so that the returned frequency corresponds to the allele opposite the outgroup-major state.

        Args:
            config: D-statistics configuration.
            geno_sub: Genotype matrix subsetted to the samples used by the test.
            pops_mapped: Population indices relative to ``geno_sub``. The final
                population is assumed to be the outgroup.

        Returns:
            np.ndarray: Array with shape ``(n_pops, n_retained_sites)`` containing population allele frequencies.

        Raises:
            ValueError: If input dimensions, population mappings, or genotype values are invalid.
            RuntimeError: If no SNPs survive coverage and outgroup-tie filtering.
        """
        if not isinstance(geno_sub, np.ndarray) or geno_sub.ndim != 2:
            msg = "`geno_sub` must be a 2D numpy array."
            self.logger.error(msg)
            raise ValueError(msg)

        if geno_sub.size == 0 or geno_sub.shape[0] == 0 or geno_sub.shape[1] == 0:
            msg = "`geno_sub` must be a non-empty 2D genotype matrix."
            self.logger.error(msg)
            raise ValueError(msg)

        if not pops_mapped:
            msg = "`pops_mapped` must contain at least one population."
            self.logger.error(msg)
            raise ValueError(msg)

        n_samples, n_sites = geno_sub.shape

        for pop_idx, inds in enumerate(pops_mapped):
            inds = np.asarray(inds)

            if inds.ndim != 1:
                msg = f"Population {pop_idx} mapped indices must be one-dimensional."
                self.logger.error(msg)
                raise ValueError(msg)

            if inds.size == 0:
                msg = f"Population {pop_idx} has no mapped samples."
                self.logger.error(msg)
                raise ValueError(msg)

            if not np.issubdtype(inds.dtype, np.integer):
                msg = f"Population {pop_idx} mapped indices must be integers."
                self.logger.error(msg)
                raise TypeError(msg)

            if np.any(inds < 0) or np.any(inds >= n_samples):
                bad = inds[(inds < 0) | (inds >= n_samples)]
                msg = (
                    f"Population {pop_idx} contains mapped indices outside geno_sub: "
                    f"{bad.tolist()} for n_samples={n_samples}."
                )
                self.logger.error(msg)
                raise IndexError(msg)

        valid = (
            (geno_sub == 0)
            | (geno_sub == 1)
            | (geno_sub == 2)
            | (geno_sub == config.MISSING)
        )
        if not np.all(valid):
            bad_vals = np.unique(geno_sub[~valid])
            msg = (
                "Invalid 012 genotype values detected. Expected only "
                f"{{0, 1, 2, {config.MISSING}}}; found {bad_vals.tolist()}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        mask_cov = np.ones(n_sites, dtype=bool)
        for inds in pops_mapped:
            mask_cov &= np.any(geno_sub[inds, :] != config.MISSING, axis=0)

        freq = geno_sub.astype(np.float64, copy=True)
        freq[freq == config.MISSING] = np.nan
        freq *= 0.5

        out_idx = pops_mapped[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out_mean = np.nanmean(freq[out_idx, :], axis=0)

        tie_mask = np.isclose(out_mean, 0.5, atol=1e-12, rtol=0.0)
        keep_mask = mask_cov & np.isfinite(out_mean) & ~tie_mask

        if not np.any(keep_mask):
            msg = "No sites survived the coverage/outgroup-tie filter. Consider stricter upstream SNP filtering, removing highly missing sites, or checking that the outgroup has usable genotype calls."
            self.logger.error(msg)
            raise RuntimeError(msg)

        flip_mask = keep_mask & (out_mean > 0.5)
        freq[:, flip_mask] = 1.0 - freq[:, flip_mask]

        freq_kept = freq[:, keep_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pop_freqs = np.vstack(
                [np.nanmean(freq_kept[inds, :], axis=0) for inds in pops_mapped]
            )

        if not np.all(np.isfinite(pop_freqs)):
            msg = "Non-finite population frequencies detected after filtering. This indicates an unexpected coverage-filtering failure."
            self.logger.error(msg)
            raise RuntimeError(msg)

        return pop_freqs
