from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence
import numpy as np
import pandas as pd

from snpio.popgenstats.fst_distance import FstDistance
from snpio.popgenstats.genetic_distance import GeneticDistance
from snpio.utils.logging import LoggerManager
from snpio.utils.numeric import safe_divide

if TYPE_CHECKING:
    from snpio.plotting.plotting import Plotting
    from snpio.read_input.genotype_data import GenotypeData


@dataclass(frozen=True)
class PairwiseTableInputs:
    """Normalized pairwise table inputs for manuscript-compatible outputs.

    This class standardizes pairwise population-statistic results across Fst and
    Nei's genetic distance. It keeps backward-compatible keys while also exposing
    explicit bootstrap and permutation fields.

    Attributes:
        metric: Metric prefix, e.g. "Fst" or "Nei".
        method: Method used to generate the result.
        observed: Observed pairwise statistic table.
        boot_lower: Bootstrap lower confidence limit table.
        boot_upper: Bootstrap upper confidence limit table.
        pvalues: Permutation p-value table.
        perm_lower: Lower bound of the permutation/null distribution.
        perm_upper: Upper bound of the permutation/null distribution.
    """

    metric: str
    method: Literal["observed", "permutation", "bootstrap"]
    observed: pd.DataFrame
    boot_lower: pd.DataFrame | None = None
    boot_upper: pd.DataFrame | None = None
    pvalues: pd.DataFrame | None = None
    perm_lower: pd.DataFrame | None = None
    perm_upper: pd.DataFrame | None = None

    @classmethod
    def from_method_result(
        cls,
        metric: str,
        method: Literal["observed", "permutation", "bootstrap"],
        observed: pd.DataFrame,
        lower: pd.DataFrame | None = None,
        upper: pd.DataFrame | None = None,
        pvalues: pd.DataFrame | None = None,
    ) -> "PairwiseTableInputs":
        """Construct normalized pairwise table inputs from one method result.

        Args:
            metric: Metric prefix, e.g. "Fst" or "Nei".
            method: Method used to generate the result.
            observed: Observed pairwise statistic table.
            lower: Lower bound table.
            upper: Upper bound table.
            pvalues: P-value table.

        Returns:
            PairwiseTableInputs object.
        """
        if method == "bootstrap":
            return cls(
                metric=metric,
                method=method,
                observed=observed,
                boot_lower=lower,
                boot_upper=upper,
                pvalues=pvalues,
            )

        if method == "permutation":
            return cls(
                metric=metric,
                method=method,
                observed=observed,
                pvalues=pvalues,
                perm_lower=lower,
                perm_upper=upper,
            )

        return cls(
            metric=metric,
            method=method,
            observed=observed,
            pvalues=pvalues,
        )

    def to_summary_stats_dict(self, include_legacy_keys: bool = True) -> dict[str, Any]:
        """Convert normalized pairwise inputs to summary-statistics dictionary keys.

        Args:
            include_legacy_keys: If True, also include the historical
                ``*_lower`` and ``*_upper`` keys used elsewhere in SNPio.

        Returns:
            Dictionary of summary-statistics keys and values.
        """
        base = f"{self.metric}_between_populations"

        out: dict[str, Any] = {
            f"{base}_obs": self.observed,
            f"{base}_boot_lower": self.boot_lower,
            f"{base}_boot_upper": self.boot_upper,
            f"{base}_pvalues": self.pvalues,
            f"{base}_perm_lower": self.perm_lower,
            f"{base}_perm_upper": self.perm_upper,
        }

        if include_legacy_keys:
            if self.method == "bootstrap":
                active_lower = self.boot_lower
                active_upper = self.boot_upper
            elif self.method == "permutation":
                active_lower = self.perm_lower
                active_upper = self.perm_upper
            else:
                active_lower = None
                active_upper = None

            out.update(
                {
                    f"{base}_lower": active_lower,
                    f"{base}_upper": active_upper,
                }
            )

        return out


class SummaryStatistics:
    """Class for calculating summary statistics from genotype data.

    This class contains methods for estimating population genetic summary statistics.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        alignment_012: np.ndarray,
        plotter: "Plotting",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the SummaryStatistics object.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing genotype data.
            alignment_012 (np.ndarray): Genotype data in 012-encoded format.
            plotter (Plotting): Plotting object to use.
            verbose (bool): If True, enable verbose logging.
            debug (bool): If True, enable debug logging.
        """
        self.genotype_data = genotype_data
        self.alignment_012 = alignment_012
        self.verbose = verbose
        self.debug = debug

        logman = LoggerManager(
            __name__, prefix=self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()
        self.plotter = plotter

    def observed_heterozygosity(self) -> np.ndarray:
        """Calculate observed heterozygosity (Ho) for each locus.

        Observed heterozygosity (Ho) is defined as the proportion of heterozygous individuals at a given locus.

        Returns:
            np.ndarray: An array containing observed heterozygosity values for each locus.
        """
        alignment, n_individuals = self._prepare_alignment_and_individuals()
        ho = self._calculate_heterozygosity(alignment, n_individuals, observed=True)
        return ho

    def expected_heterozygosity(self) -> np.ndarray:
        """Calculate expected heterozygosity (He) for each locus.

        Expected heterozygosity (He) is the expected proportion of heterozygous individuals under Hardy-Weinberg equilibrium.

        Returns:
            np.ndarray: An array containing expected heterozygosity values for each locus.
        """
        alignment, n_individuals = self._prepare_alignment_and_individuals()
        he = self._calculate_heterozygosity(alignment, n_individuals, observed=False)
        return he

    def _calculate_heterozygosity(
        self, alignment: np.ndarray, n_individuals: np.ndarray, observed: bool
    ) -> np.ndarray:
        """Calculate heterozygosity (Ho or He) for each locus.

        Args:
            alignment (np.ndarray): The alignment array.
            n_individuals (np.ndarray): Number of non-missing individuals per locus.
            observed (bool): If True, calculate observed heterozygosity (Ho); otherwise, expected heterozygosity (He).

        Returns:
            np.ndarray: Heterozygosity values for each locus.
        """
        if observed:
            # Calculate observed heterozygosity
            heterozygous_counts = np.sum(alignment == 1, axis=0)
            return safe_divide(heterozygous_counts, n_individuals)

        # Calculate expected heterozygosity
        alt_allele_counts = np.nansum(alignment, axis=0, dtype=np.float64)
        total_alleles = 2 * n_individuals  # Assuming diploid organisms
        p = safe_divide(alt_allele_counts, total_alleles)
        q = 1 - p
        return 2 * p * q  # He = 2pq

    def nucleotide_diversity(self) -> np.ndarray:
        """Calculate nucleotide diversity (Pi) for each locus.

        Nucleotide diversity (Pi) is the average number of nucleotide differences per site between two sequences.

        Notes:
            A bias correction is applied in the calculation.

        Returns:
            np.ndarray: An array containing nucleotide diversity values for each locus.
        """
        _, n_individuals = self._prepare_alignment_and_individuals()
        he = self.expected_heterozygosity()

        # Calculate nucleotide diversity
        pi = np.full_like(he, np.nan, dtype=float)
        valid = n_individuals > 1  # Need at least 2 individuals for diversity
        pi[valid] = he[valid] * n_individuals[valid] / (n_individuals[valid] - 1)
        return pi

    def calculate_summary_statistics(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
        save_plots: bool = True,
        include_nei: bool = True,
    ) -> dict:
        """Calculate a suite of summary statistics for SNP data.

        Computes overall and per-population heterozygosity, nucleotide diversity, pairwise Weir and Cockerham's Fst, and optionally pairwise Nei's genetic distance.

        Args:
            method: Method for pairwise inferential statistics. "observed": Computes observed pairwise matrices only. "permutation": Performs permutation tests to estimate p-values. "bootstrap": Performs bootstrap resampling to estimate confidence intervals.
            n_reps: Number of permutation or bootstrap replicates.
            n_jobs: Number of parallel jobs. Use -1 for all available cores.
            save_plots: Whether to save summary-statistic plots.
            include_nei: Whether to include Nei's genetic distance in the returned
                summary-statistics dictionary.

        Returns:
            Dictionary containing calculated summary statistics.
        """
        self.logger.info("Calculating summary statistics...")

        self.logger.info("Calculating heterozygosity and nucleotide diversity...")
        ho_overall = pd.Series(self.observed_heterozygosity())
        he_overall = pd.Series(self.expected_heterozygosity())
        pi_overall = pd.Series(self.nucleotide_diversity())

        summary_stats = {
            "overall": pd.DataFrame(
                {
                    "Ho": ho_overall,
                    "He": he_overall,
                    "Pi": pi_overall,
                }
            ),
            "per_population": {},
        }

        neisdist = None

        if self.genotype_data.has_popmap:
            ho_per_population = self.observed_heterozygosity_per_population()
            he_per_population = self.expected_heterozygosity_per_population()
            pi_per_population = self.nucleotide_diversity_per_population()

            for pop_id in ho_per_population.keys():
                summary_stats["per_population"][pop_id] = pd.DataFrame(
                    {
                        "Ho": ho_per_population[pop_id],
                        "He": he_per_population[pop_id],
                        "Pi": pi_per_population[pop_id],
                    }
                )

        if self.genotype_data.has_popmap:
            self.logger.info(
                f"Calculating pairwise Weir & Cockerham Fst using method: "
                f"'{method}'..."
            )

            fst = FstDistance(
                self.genotype_data,
                self.plotter,
                verbose=self.verbose,
                debug=self.debug,
            )

            if include_nei:
                neisdist = GeneticDistance(
                    self.genotype_data,
                    self.plotter,
                    verbose=self.verbose,
                    debug=self.debug,
                )

            fst_results = fst.weir_cockerham_fst(
                method=method,
                n_reps=n_reps,
                n_jobs=n_jobs,
            )

            df_observed, df_lower, df_upper, df_pvals = fst.parse_wc_fst(
                fst_results,
                alpha=0.05,
            )

            fst_table_inputs = PairwiseTableInputs.from_method_result(
                metric="Fst",
                method=method,
                observed=df_observed,
                lower=df_lower,
                upper=df_upper,
                pvalues=df_pvals,
            )

            summary_stats.update(fst_table_inputs.to_summary_stats_dict())

        else:
            self.logger.info(
                "No population map provided; skipping Fst calculation between populations."
            )

        if save_plots:
            use_pvalues = method == "permutation"
            self.plotter.plot_summary_statistics(summary_stats, use_pvalues=use_pvalues)

        if self.genotype_data.has_popmap and include_nei:
            self.logger.info(
                f"Calculating pairwise Nei's genetic distance using method: '{method}'..."
            )

            if neisdist is None:
                msg = "GeneticDistance object was not initialized. Cannot calculate Nei's distance."
                self.logger.error(msg)
                raise RuntimeError(msg)

            nei_results = neisdist.nei_distance(
                method=method,
                n_reps=n_reps,
                n_jobs=n_jobs,
            )

            df_nei_observed, df_nei_lower, df_nei_upper, df_nei_pvals = (
                neisdist.parse_nei_result(nei_results)
            )

            summary_stats["Nei_between_populations_obs"] = df_nei_observed
            summary_stats["Nei_between_populations_lower"] = df_nei_lower
            summary_stats["Nei_between_populations_upper"] = df_nei_upper
            summary_stats["Nei_between_populations_pvalues"] = df_nei_pvals

            # Backward-compatible explicit aliases for manuscript-table construction. These do not change existing outputs; they only clarify whether lower/upper came from bootstrap CIs or permutation/null intervals.
            if method == "bootstrap":
                summary_stats["Nei_between_populations_boot_lower"] = df_nei_lower
                summary_stats["Nei_between_populations_boot_upper"] = df_nei_upper
                summary_stats["Nei_between_populations_perm_lower"] = None
                summary_stats["Nei_between_populations_perm_upper"] = None

            elif method == "permutation":
                summary_stats["Nei_between_populations_boot_lower"] = None
                summary_stats["Nei_between_populations_boot_upper"] = None
                summary_stats["Nei_between_populations_perm_lower"] = df_nei_lower
                summary_stats["Nei_between_populations_perm_upper"] = df_nei_upper

            else:
                summary_stats["Nei_between_populations_boot_lower"] = None
                summary_stats["Nei_between_populations_boot_upper"] = None
                summary_stats["Nei_between_populations_perm_lower"] = None
                summary_stats["Nei_between_populations_perm_upper"] = None

        elif include_nei:
            self.logger.info(
                "No population map provided; skipping Nei's genetic distance calculation."
            )

        self.logger.info("Pairwise population-statistic calculations complete!")
        self.logger.info("Summary statistics calculation complete!")

        return summary_stats

    @staticmethod
    def _is_pairwise_population_key(key: object) -> bool:
        """Check whether a dictionary key represents a pairwise population comparison.

        Args:
            key: Dictionary key.

        Returns:
            True if key is a two-element tuple.
        """
        return isinstance(key, tuple) and len(key) == 2

    @classmethod
    def _is_flat_pairwise_results_dict(cls, results: Mapping[Any, Any]) -> bool:
        """Check whether a dictionary is keyed by pairwise population tuples.

        Args:
            results: Result dictionary.

        Returns:
            True if all keys are two-element population-pair tuples.
        """
        if not results:
            return False

        return all(cls._is_pairwise_population_key(key) for key in results.keys())

    @classmethod
    def _infer_population_order_from_pairwise_keys(
        cls,
        pairwise_results: Mapping[Any, Any],
        population_order: Sequence[str] | None = None,
    ) -> list[str]:
        """Infer population order from pairwise tuple keys.

        Args:
            pairwise_results: Pairwise dictionary keyed by two-element tuples.
            population_order: Optional explicit population order.

        Returns:
            Population labels.

        Raises:
            ValueError: If no population labels can be inferred.
        """
        observed_pops: set[str] = set()

        for key in pairwise_results:
            if cls._is_pairwise_population_key(key):
                pop1, pop2 = key
                observed_pops.add(str(pop1))
                observed_pops.add(str(pop2))

        if not observed_pops:
            raise ValueError("Could not infer population labels from pairwise keys.")

        if population_order is None:
            return sorted(observed_pops)

        ordered = [str(pop) for pop in population_order if str(pop) in observed_pops]
        missing = sorted(observed_pops - set(ordered))

        return ordered + missing

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        """Convert scalar-like value to float.

        Args:
            value: Value to convert.

        Returns:
            Float value, or None if conversion is not possible.
        """
        if value is None:
            return None

        try:
            if isinstance(value, pd.Series):
                if len(value) != 1:
                    return None
                value = value.iloc[0]

            elif isinstance(value, np.ndarray):
                if value.size != 1:
                    return None
                value = value.item()

            return float(value)

        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_pairwise_record_value(
        cls,
        record: Any,
        aliases: tuple[str, ...],
        scalar_fallback: bool = False,
    ) -> float | None:
        """Extract a numeric value from a pairwise result record.

        Args:
            record: Pairwise result record.
            aliases: Candidate aliases if record is dictionary-like.
            scalar_fallback: Whether scalar records should be treated as the target value.

        Returns:
            Extracted float value, or None.
        """
        if record is None:
            return None

        if isinstance(record, Mapping):
            value = cls._get_result_by_alias(record, aliases)
            return cls._to_float_or_none(value)

        if isinstance(record, pd.Series):
            value = cls._get_result_by_alias(record.to_dict(), aliases)

            if value is not None:
                return cls._to_float_or_none(value)

            if scalar_fallback and len(record) == 1:
                return cls._to_float_or_none(record.iloc[0])

            return None

        if hasattr(record, "_asdict"):
            value = cls._get_result_by_alias(record._asdict(), aliases)

            if value is not None:
                return cls._to_float_or_none(value)

        if hasattr(record, "__dict__"):
            value = cls._get_result_by_alias(vars(record), aliases)

            if value is not None:
                return cls._to_float_or_none(value)

        if isinstance(record, (tuple, list, np.ndarray)):
            vector = cls._as_flat_numeric_vector(record)

            if scalar_fallback and vector is not None and vector.size == 1:
                return float(vector[0])

            return None

        if scalar_fallback:
            return cls._to_float_or_none(record)

        return None

    @classmethod
    def _extract_positional_pairwise_values(
        cls,
        record: Any,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Extract observed, lower, upper, and p-value values positionally.

        This is a fallback for pairwise records returned as tuples, lists, arrays,
        or unlabeled Series.

        Supported positional assumptions:
            - length 1: observed
            - length 2: observed, p-value
            - length 3: observed, lower, upper
            - length >= 4: observed, lower, upper, p-value

        Args:
            record: Pairwise result record.

        Returns:
            Tuple of observed, lower, upper, and p-value values.
        """
        vector = cls._as_flat_numeric_vector(record)

        if vector is None or vector.size == 0:
            return None, None, None, None

        if vector.size == 1:
            return float(vector[0]), None, None, None

        if vector.size == 2:
            return float(vector[0]), None, None, float(vector[1])

        if vector.size == 3:
            return float(vector[0]), float(vector[1]), float(vector[2]), None

        # For very long vectors, assume this is a replicate distribution rather
        # than an observed/lower/upper/p-value record.
        if vector.size > 10:
            return (
                float(np.nanmean(vector)),
                float(np.nanquantile(vector, 0.025)),
                float(np.nanquantile(vector, 0.975)),
                None,
            )

        return float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3])

    @classmethod
    def _extract_numeric_vector_from_record(
        cls,
        record: Any,
        aliases: tuple[str, ...],
    ) -> np.ndarray | None:
        """Extract a numeric vector from a pairwise record.

        Args:
            record: Pairwise result record.
            aliases: Candidate aliases for vector-valued entries.

        Returns:
            One-dimensional numeric array, or None.
        """
        if record is None:
            return None

        if isinstance(record, Mapping):
            value = cls._get_result_by_alias(record, aliases)
            return cls._as_flat_numeric_vector(value)

        if isinstance(record, pd.Series):
            value = cls._get_result_by_alias(record.to_dict(), aliases)
            return cls._as_flat_numeric_vector(value)

        return cls._as_flat_numeric_vector(record)

    @staticmethod
    def _as_flat_numeric_vector(value: Any) -> np.ndarray | None:
        """Convert a value to a one-dimensional numeric vector when possible.

        Args:
            value: Value to convert.

        Returns:
            Numeric vector, or None if conversion is not possible.
        """
        if value is None:
            return None

        if isinstance(value, Mapping):
            return None

        if isinstance(value, pd.Series):
            values = value.to_numpy()

        elif isinstance(value, pd.DataFrame):
            return None

        elif isinstance(value, np.ndarray):
            values = value

        elif isinstance(value, (list, tuple)):
            values = value

        else:
            scalar = SummaryStatistics._to_float_or_none(value)
            if scalar is None:
                return None
            return np.array([scalar], dtype=float)

        try:
            arr = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            return None

        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return None

        return arr

    @classmethod
    def _flat_pairwise_dict_to_symmetric_frames(
        cls,
        pairwise_results: Mapping[Any, Any],
        population_order: Sequence[str] | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
        """Convert flat pairwise results into symmetric square DataFrames.

        This method supports flat pairwise dictionaries keyed by population tuples, e.g. ``{("EA", "GU"): value}``, where each value may be a scalar, mapping, Series, tuple/list, NumPy array, or bootstrap/permutation replicate vector.

        Args:
            pairwise_results: Dictionary keyed by two-element population tuples.
            population_order: Optional explicit population order.

        Returns:
            Tuple containing observed, lower, upper, and p-value DataFrames.

        Raises:
            ValueError: If observed pairwise values cannot be extracted.
        """
        pops = cls._infer_population_order_from_pairwise_keys(
            pairwise_results,
            population_order=population_order,
        )

        observed = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)
        lower = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)
        upper = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)
        pvalues = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)

        np.fill_diagonal(observed.values, 0.0)
        np.fill_diagonal(lower.values, 0.0)
        np.fill_diagonal(upper.values, 0.0)
        np.fill_diagonal(pvalues.values, 1.0)

        observed_found = False
        lower_found = False
        upper_found = False
        pvalues_found = False

        observed_aliases = (
            "Nei_between_populations_obs",
            "Nei_between_populations_observed",
            "Nei_observed",
            "Nei_obs",
            "observed",
            "obs",
            "estimate",
            "est",
            "statistic",
            "stat",
            "distance",
            "dist",
            "d",
            "nei",
            "nei_distance",
            "neis_distance",
            "nei_genetic_distance",
            "neis_genetic_distance",
            "Nei's genetic distance",
            "mean",
            "avg",
            "average",
            "value",
        )
        lower_aliases = (
            "Nei_between_populations_lower",
            "Nei_between_populations_boot_lower",
            "Nei_between_populations_perm_lower",
            "Nei_lower",
            "lower",
            "lower_ci",
            "ci_lower",
            "ci_low",
            "lcl",
            "lower_bound",
            "bootstrap_lower",
            "boot_lower",
            "permutation_lower",
            "perm_lower",
            "q025",
            "q_025",
            "q2.5",
            "2.5%",
            "2.5",
        )
        upper_aliases = (
            "Nei_between_populations_upper",
            "Nei_between_populations_boot_upper",
            "Nei_between_populations_perm_upper",
            "Nei_upper",
            "upper",
            "upper_ci",
            "ci_upper",
            "ci_high",
            "ucl",
            "upper_bound",
            "bootstrap_upper",
            "boot_upper",
            "permutation_upper",
            "perm_upper",
            "q975",
            "q_975",
            "q97.5",
            "97.5%",
            "97.5",
        )
        pvalue_aliases = (
            "Nei_between_populations_pvalues",
            "Nei_pvalues",
            "pvalues",
            "p_values",
            "pvalue",
            "p_value",
            "p",
            "pval",
            "p_val",
            "permutation_pvalues",
            "perm_pvalues",
            "empirical_pvalue",
            "empirical_p",
        )
        replicate_aliases = (
            "replicates",
            "reps",
            "bootstrap",
            "bootstraps",
            "bootstrap_replicates",
            "bootstrap_values",
            "bootstrap_distribution",
            "permutations",
            "permutation",
            "permutation_replicates",
            "permutation_values",
            "null_distribution",
        )

        for key, record in pairwise_results.items():
            if not cls._is_pairwise_population_key(key):
                continue

            pop1, pop2 = str(key[0]), str(key[1])

            obs_value = cls._extract_pairwise_record_value(
                record,
                aliases=observed_aliases,
                scalar_fallback=True,
            )
            lower_value = cls._extract_pairwise_record_value(
                record,
                aliases=lower_aliases,
                scalar_fallback=False,
            )
            upper_value = cls._extract_pairwise_record_value(
                record,
                aliases=upper_aliases,
                scalar_fallback=False,
            )
            pvalue = cls._extract_pairwise_record_value(
                record,
                aliases=pvalue_aliases,
                scalar_fallback=False,
            )

            replicate_values = cls._extract_pairwise_record_value(
                record,
                aliases=replicate_aliases,
                scalar_fallback=False,
            )

            # If direct alias extraction failed, try positional extraction.
            if (
                obs_value is None
                and lower_value is None
                and upper_value is None
                and pvalue is None
            ):
                (
                    obs_value,
                    lower_value,
                    upper_value,
                    pvalue,
                ) = cls._extract_positional_pairwise_values(record)

            # If record contains a replicate vector, use it for bounds if needed.
            replicate_array = cls._extract_numeric_vector_from_record(
                record,
                aliases=replicate_aliases,
            )

            if replicate_array is not None and replicate_array.size > 1:
                if lower_value is None:
                    lower_value = float(np.nanquantile(replicate_array, 0.025))
                if upper_value is None:
                    upper_value = float(np.nanquantile(replicate_array, 0.975))

                # Last-resort fallback. Ideally GeneticDistance.nei_distance()
                # should return the observed distance explicitly. If it does not,
                # use the mean of the replicate distribution so table construction
                # does not fail.
                if obs_value is None:
                    obs_value = float(np.nanmean(replicate_array))

            if obs_value is not None:
                observed.loc[pop1, pop2] = obs_value
                observed.loc[pop2, pop1] = obs_value
                observed_found = True

            if lower_value is not None:
                lower.loc[pop1, pop2] = lower_value
                lower.loc[pop2, pop1] = lower_value
                lower_found = True

            if upper_value is not None:
                upper.loc[pop1, pop2] = upper_value
                upper.loc[pop2, pop1] = upper_value
                upper_found = True

            if pvalue is not None:
                pvalues.loc[pop1, pop2] = pvalue
                pvalues.loc[pop2, pop1] = pvalue
                pvalues_found = True

        if not observed_found:
            first_key = next(iter(pairwise_results))
            first_record = pairwise_results[first_key]
            raise ValueError(
                "Could not extract observed Nei's distance values from the flat "
                "pairwise dictionary. Example record: "
                f"key={first_key!r}, type={type(first_record)}, value={first_record!r}"
            )

        return (
            observed,
            lower if lower_found else None,
            upper if upper_found else None,
            pvalues if pvalues_found else None,
        )

    @staticmethod
    def _normalize_result_key(key: object) -> str:
        """Normalize a result key for flexible dictionary lookup.

        Args:
            key: Dictionary key.

        Returns:
            Normalized key string.
        """
        return (
            str(key)
            .lower()
            .replace("'", "")
            .replace('"', "")
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )

    @classmethod
    def _get_result_by_alias(
        cls,
        results: Mapping[str, Any],
        aliases: tuple[str, ...],
    ) -> Any | None:
        """Get a result object from a dictionary using flexible key aliases.

        Args:
            results: Result dictionary.
            aliases: Candidate key aliases.

        Returns:
            Matching result object, or None if no match is found.
        """
        normalized_aliases = {cls._normalize_result_key(alias) for alias in aliases}

        for key, value in results.items():
            normalized_key = cls._normalize_result_key(key)
            if normalized_key in normalized_aliases:
                return value

        for key, value in results.items():
            normalized_key = cls._normalize_result_key(key)
            if any(alias in normalized_key for alias in normalized_aliases):
                return value

        return None

    @staticmethod
    def _coerce_pairwise_table(value: Any, table_name: str) -> pd.DataFrame | None:
        """Convert a pairwise result object to a square DataFrame.

        Args:
            value: Pairwise result object.
            table_name: Name used in error messages.

        Returns:
            Pairwise DataFrame, or None if value is None.

        Raises:
            TypeError: If the value cannot be converted to a DataFrame.
            ValueError: If the DataFrame is not square.
        """
        if value is None:
            return None

        if isinstance(value, pd.DataFrame):
            df = value.copy()

        elif isinstance(value, pd.Series):
            raise TypeError(
                f"{table_name} was returned as a Series, but a square pairwise "
                "DataFrame is required."
            )

        elif isinstance(value, dict):
            df = pd.DataFrame.from_dict(value, orient="index")

        else:
            raise TypeError(
                f"Could not convert {table_name} to a DataFrame. "
                f"Got object of type: {type(value)}"
            )

        df = df.apply(pd.to_numeric, errors="coerce")

        if df.shape[0] != df.shape[1]:
            raise ValueError(f"{table_name} must be square, but got shape {df.shape}.")

        if list(df.index) != list(df.columns):
            raise ValueError(
                f"{table_name} index and columns must contain the same population labels."
            )

        return df

    @classmethod
    def parse_nei_distance_results(
        cls,
        nei_results: Any,
        population_order: Sequence[str] | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
        """Parse Nei's genetic distance results into observed, bounds, and p-values.

        Supports square DataFrames, dictionary-wrapped square DataFrames, and flat
        dictionaries keyed by population-pair tuples such as ``("EA", "GU")``.

        Args:
            nei_results: Output from GeneticDistance.nei_distance().
            population_order: Optional explicit population order.

        Returns:
            Tuple containing observed distances, lower bounds, upper bounds, and p-values.

        Raises:
            TypeError: If the result object has an unsupported type.
            ValueError: If no observed Nei's distance table can be identified.
        """
        if isinstance(nei_results, pd.DataFrame):
            observed = cls._coerce_pairwise_table(nei_results, "Nei observed")
            return observed, None, None, None

        if not isinstance(nei_results, Mapping):
            raise TypeError(
                "Expected Nei's distance results to be a DataFrame or dictionary, "
                f"but got: {type(nei_results)}"
            )

        if cls._is_flat_pairwise_results_dict(nei_results):
            return cls._flat_pairwise_dict_to_symmetric_frames(
                nei_results,
                population_order=population_order,
            )

        observed_raw = cls._get_result_by_alias(
            nei_results,
            aliases=(
                "Nei_between_populations_obs",
                "Nei_between_populations_observed",
                "Nei_observed",
                "Nei_obs",
                "observed",
                "obs",
                "distance",
                "nei_distance",
                "neis_distance",
                "neis_genetic_distance",
                "Nei's genetic distance",
            ),
        )

        lower_raw = cls._get_result_by_alias(
            nei_results,
            aliases=(
                "Nei_between_populations_lower",
                "Nei_between_populations_boot_lower",
                "Nei_between_populations_perm_lower",
                "Nei_lower",
                "lower",
                "lower_ci",
                "ci_lower",
                "bootstrap_lower",
                "boot_lower",
                "permutation_lower",
                "perm_lower",
            ),
        )

        upper_raw = cls._get_result_by_alias(
            nei_results,
            aliases=(
                "Nei_between_populations_upper",
                "Nei_between_populations_boot_upper",
                "Nei_between_populations_perm_upper",
                "Nei_upper",
                "upper",
                "upper_ci",
                "ci_upper",
                "bootstrap_upper",
                "boot_upper",
                "permutation_upper",
                "perm_upper",
            ),
        )

        pvalues_raw = cls._get_result_by_alias(
            nei_results,
            aliases=(
                "Nei_between_populations_pvalues",
                "Nei_pvalues",
                "pvalues",
                "p_values",
                "pvalue",
                "p_value",
                "p",
                "permutation_pvalues",
                "perm_pvalues",
            ),
        )

        if observed_raw is None:
            candidate_tables: list[pd.DataFrame] = []

            for value in nei_results.values():
                try:
                    candidate = cls._coerce_pairwise_table(value, "candidate Nei table")
                except (TypeError, ValueError):
                    continue

                if candidate is not None:
                    candidate_tables.append(candidate)

            if len(candidate_tables) == 1:
                observed = candidate_tables[0]
            else:
                raise ValueError(
                    "Could not identify the observed Nei's distance table from "
                    f"GeneticDistance.nei_distance() output. Available keys: "
                    f"{list(nei_results.keys())}"
                )
        else:
            observed = cls._coerce_pairwise_table(observed_raw, "Nei observed")

        lower = cls._coerce_pairwise_table(lower_raw, "Nei lower")
        upper = cls._coerce_pairwise_table(upper_raw, "Nei upper")
        pvalues = cls._coerce_pairwise_table(pvalues_raw, "Nei p-values")

        return observed, lower, upper, pvalues

    def observed_heterozygosity_per_population(self):
        """Calculate observed heterozygosity (Ho) for each locus per population.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the observed heterozygosity values per locus for that population.
        """
        pop_indices = self.genotype_data.get_population_indices()
        ho_per_population = {}

        for pop_id, indices in pop_indices.items():
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan  # Replace missing data

            if pop_alignment.shape[0] == 0 or np.all(np.isnan(pop_alignment)):
                continue  # Skip populations with no data

            # Number of non-missing individuals per locus
            n_individuals = np.sum(~np.isnan(pop_alignment), axis=0)
            num_heterozygotes = np.nansum(pop_alignment == 1, axis=0)

            # Calculate Ho
            ho = np.full(pop_alignment.shape[1], np.nan, dtype=np.float64)
            valid = n_individuals > 0
            ho[valid] = num_heterozygotes[valid] / n_individuals[valid]

            # Store results as a pandas Series with locus indices
            ho_per_population[pop_id] = pd.Series(
                ho, index=np.arange(pop_alignment.shape[1]), name="Ho"
            )

        return ho_per_population

    def expected_heterozygosity_per_population(self, return_n: bool = False):
        """Calculate expected heterozygosity (He) for each locus per population.

        Args:
            return_n (bool): If True, also return the number of non-missing individuals per locus.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the expected heterozygosity values per locus for that population. If return_n is True, returns a tuple (he, n_individuals) per population.
        """
        pop_indices = self.genotype_data.get_population_indices()
        he_per_population = {}

        for pop_id, indices in pop_indices.items():
            pop_alignment = self.alignment_012[indices, :].astype(float).copy()
            pop_alignment[pop_alignment == -9] = np.nan  # Replace missing data

            if pop_alignment.shape[0] == 0 or np.all(np.isnan(pop_alignment)):
                continue  # Skip populations with no data

            # Number of non-missing individuals per locus
            n_individuals = np.sum(~np.isnan(pop_alignment), axis=0)
            total_alleles = 2 * n_individuals

            # Frequency of alternate allele (p)
            alt_allele_counts = np.nansum(pop_alignment, axis=0, dtype=float)

            p = np.zeros_like(alt_allele_counts, dtype=float)
            valid = total_alleles > 0
            p[valid] = alt_allele_counts[valid] / total_alleles[valid]
            q = 1 - p

            # Expected heterozygosity
            he = np.zeros_like(p, dtype=float)
            he[valid] = 2 * p[valid] * q[valid]

            if return_n:
                he_per_population[pop_id] = (
                    pd.Series(he, index=np.arange(pop_alignment.shape[1]), name="He"),
                    n_individuals,
                )
            else:
                he_per_population[pop_id] = pd.Series(
                    he, index=np.arange(pop_alignment.shape[1]), name="He"
                )

        return he_per_population

    def nucleotide_diversity_per_population(self):
        """Calculate nucleotide diversity (Pi) for each locus per population.

        Returns:
            dict: A dictionary where keys are population IDs and values are pandas Series containing the nucleotide diversity values per locus for that population.
        """
        he_and_n_per_population = self.expected_heterozygosity_per_population(
            return_n=True
        )
        pi_per_population = {}

        for pop_id, (he, n_individuals) in he_and_n_per_population.items():
            n = n_individuals.astype(float)

            # Calculate Pi with bias correction
            pi = np.zeros_like(he, dtype=float)
            valid = n > 1
            pi[valid] = (n[valid] / (n[valid] - 1)) * he[valid]

            # Store results as a pandas Series
            pi_per_population[pop_id] = pd.Series(
                pi, index=np.arange(len(pi)), name="Pi"
            )

        return pi_per_population

    def ind_count(self, geno):
        """Count the number of non-missing individuals in a genotype array.

        Args:
            geno (np.ndarray): 1D array of genotype values (with np.nan for missing).

        Returns:
            int: Count of non-missing individuals.
        """
        return int(np.sum(~np.isnan(geno)))

    def pop_het(self, geno):
        """Compute the observed heterozygosity (proportion of heterozygotes) from a 1D genotype array (assumes heterozygote is coded as 1).

        Args:
            geno (np.ndarray): 1D array of genotype values.

        Returns:
            float: Proportion of heterozygotes, or np.nan if no individuals are typed.
        """
        n = self.ind_count(geno)
        if n == 0:
            return np.nan
        return np.sum(geno[~np.isnan(geno)] == 1) / n

    def pop_freq(self, geno):
        """Compute the allele frequency from a 1D genotype array.

        Assumes genotypes are coded as 0, 1, or 2 (number of copies of the alternate allele).
        Only non-missing individuals are used.

        Args:
            geno (np.ndarray): 1D array of genotype values.

        Returns:
            float: Allele frequency, or np.nan if no individuals are typed.
        """
        n = self.ind_count(geno)
        if n == 0:
            return np.nan
        # Total alternate allele count divided by total allele copies.
        return np.nansum(geno) / (2 * n)

    def _prepare_alignment_and_individuals(self):
        """Prepare alignment and count non-missing individuals per locus.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Alignment array and counts of non-missing individuals per locus.
        """
        alignment = self.alignment_012.astype(float).copy()

        # Replace missing data (-9) with NaN
        alignment[alignment == -9] = np.nan

        # Count valid individuals per locus
        n_individuals = np.sum(~np.isnan(alignment), axis=0)
        return alignment, n_individuals
