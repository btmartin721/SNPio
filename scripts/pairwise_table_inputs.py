from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairwiseParts:
    """Parsed pairwise result components.

    Attributes:
        observed: Observed pairwise statistic table.
        lower: Lower bound table.
        upper: Upper bound table.
        pvalues: P-value table.
    """

    observed: pd.DataFrame | None = None
    lower: pd.DataFrame | None = None
    upper: pd.DataFrame | None = None
    pvalues: pd.DataFrame | None = None


@dataclass(frozen=True)
class PairwiseTableInputs:
    """Normalized pairwise table inputs for manuscript table generation.

    Attributes:
        metric: Metric prefix, e.g. "Fst" or "Nei".
        observed: Observed pairwise statistic table.
        boot_lower: Bootstrap lower confidence limit table.
        boot_upper: Bootstrap upper confidence limit table.
        pvalues: Permutation p-value table.
        perm_lower: Lower bound of the permutation/null distribution.
        perm_upper: Upper bound of the permutation/null distribution.
    """

    metric: str
    observed: pd.DataFrame
    boot_lower: pd.DataFrame | None = None
    boot_upper: pd.DataFrame | None = None
    pvalues: pd.DataFrame | None = None
    perm_lower: pd.DataFrame | None = None
    perm_upper: pd.DataFrame | None = None

    @classmethod
    def from_results(
        cls,
        metric: str,
        perm_results: Any,
        boot_results: Any,
        population_order: Sequence[str] | None = None,
    ) -> PairwiseTableInputs:
        """Create normalized manuscript-table inputs from permutation and bootstrap outputs.

        Args:
            metric: Metric prefix, e.g. "Fst" or "Nei".
            perm_results: Permutation result object.
            boot_results: Bootstrap result object.
            population_order: Optional explicit population order.

        Returns:
            Normalized pairwise table inputs.

        Raises:
            ValueError: If an observed pairwise table cannot be found.
        """
        perm_parts = _parse_pairwise_results(
            results=perm_results,
            metric=metric,
            population_order=population_order,
        )
        boot_parts = _parse_pairwise_results(
            results=boot_results,
            metric=metric,
            population_order=population_order,
        )

        observed = boot_parts.observed
        if observed is None:
            observed = perm_parts.observed

        if observed is None:
            raise ValueError(
                f"Could not identify observed {metric} pairwise table from "
                "permutation or bootstrap results."
            )

        return cls(
            metric=metric,
            observed=observed,
            boot_lower=boot_parts.lower,
            boot_upper=boot_parts.upper,
            pvalues=perm_parts.pvalues,
            perm_lower=perm_parts.lower,
            perm_upper=perm_parts.upper,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PairwiseTableInputs:
        """Load normalized pairwise table inputs from JSON.

        Args:
            path: Path to normalized pairwise table-input JSON.

        Returns:
            Normalized pairwise table inputs.
        """
        path = Path(path)

        with path.open("r") as f:
            data = json.load(f)

        metric = data["metric"]

        return cls(
            metric=metric,
            observed=_nested_dict_to_frame(data[f"{metric}_between_populations_obs"]),
            boot_lower=_nested_dict_to_frame(
                data.get(f"{metric}_between_populations_boot_lower")
            ),
            boot_upper=_nested_dict_to_frame(
                data.get(f"{metric}_between_populations_boot_upper")
            ),
            pvalues=_nested_dict_to_frame(
                data.get(f"{metric}_between_populations_pvalues")
            ),
            perm_lower=_nested_dict_to_frame(
                data.get(f"{metric}_between_populations_perm_lower")
            ),
            perm_upper=_nested_dict_to_frame(
                data.get(f"{metric}_between_populations_perm_upper")
            ),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Convert table inputs to a JSON-safe dictionary.

        Returns:
            JSON-safe dictionary.
        """
        base = f"{self.metric}_between_populations"

        return {
            "metric": self.metric,
            f"{base}_obs": _frame_to_nested_dict(self.observed),
            f"{base}_boot_lower": _frame_to_nested_dict(self.boot_lower),
            f"{base}_boot_upper": _frame_to_nested_dict(self.boot_upper),
            f"{base}_pvalues": _frame_to_nested_dict(self.pvalues),
            f"{base}_perm_lower": _frame_to_nested_dict(self.perm_lower),
            f"{base}_perm_upper": _frame_to_nested_dict(self.perm_upper),
        }

    def write_json(self, path: str | Path) -> None:
        """Write normalized table inputs to JSON.

        Args:
            path: Output JSON path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.to_json_dict(), f, indent=4)


def _parse_pairwise_results(
    results: Any,
    metric: str,
    population_order: Sequence[str] | None = None,
) -> PairwiseParts:
    """Parse Fst or Nei results into observed/lower/upper/p-value tables.

    Args:
        results: Result object to parse.
        metric: Metric prefix, e.g. "Fst" or "Nei".
        population_order: Optional population order.

    Returns:
        Parsed pairwise result parts.
    """
    if results is None:
        return PairwiseParts()

    if isinstance(results, pd.DataFrame):
        return PairwiseParts(observed=_coerce_square_frame(results, "observed"))

    if not isinstance(results, Mapping):
        raise TypeError(
            f"Expected {metric} results to be a DataFrame or mapping, "
            f"but got: {type(results)}"
        )

    if _is_flat_pairwise_dict(results):
        return _parse_flat_pairwise_dict(
            results=results,
            population_order=population_order,
        )

    base = f"{metric}_between_populations"

    observed_raw = _get_by_alias(
        results,
        aliases=(
            f"{base}_obs",
            f"{base}_observed",
            f"{metric}_observed",
            f"{metric}_obs",
            "observed",
            "obs",
            "distance",
            "value",
        ),
    )
    lower_raw = _get_by_alias(
        results,
        aliases=(
            f"{base}_lower",
            f"{base}_boot_lower",
            f"{base}_perm_lower",
            f"{metric}_lower",
            "lower",
            "lower_ci",
            "ci_lower",
            "boot_lower",
            "bootstrap_lower",
            "perm_lower",
            "permutation_lower",
        ),
    )
    upper_raw = _get_by_alias(
        results,
        aliases=(
            f"{base}_upper",
            f"{base}_boot_upper",
            f"{base}_perm_upper",
            f"{metric}_upper",
            "upper",
            "upper_ci",
            "ci_upper",
            "boot_upper",
            "bootstrap_upper",
            "perm_upper",
            "permutation_upper",
        ),
    )
    pvalues_raw = _get_by_alias(
        results,
        aliases=(
            f"{base}_pvalues",
            f"{metric}_pvalues",
            "pvalues",
            "p_values",
            "pvalue",
            "p_value",
            "p",
            "perm_pvalues",
            "permutation_pvalues",
        ),
    )

    return PairwiseParts(
        observed=_coerce_square_frame(observed_raw, f"{metric} observed"),
        lower=_coerce_square_frame(lower_raw, f"{metric} lower"),
        upper=_coerce_square_frame(upper_raw, f"{metric} upper"),
        pvalues=_coerce_square_frame(pvalues_raw, f"{metric} p-values"),
    )


def _parse_flat_pairwise_dict(
    results: Mapping[Any, Any],
    population_order: Sequence[str] | None = None,
) -> PairwiseParts:
    """Parse a flat pairwise dictionary keyed by population-pair tuples.

    Args:
        results: Flat pairwise result dictionary.
        population_order: Optional population order.

    Returns:
        Parsed pairwise result parts.
    """
    pops = _infer_population_order(results, population_order=population_order)

    observed = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)
    lower = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)
    upper = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)
    pvalues = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)

    np.fill_diagonal(observed.values, 0.0)
    np.fill_diagonal(lower.values, 0.0)
    np.fill_diagonal(upper.values, 0.0)
    np.fill_diagonal(pvalues.values, 1.0)

    found_observed = False
    found_lower = False
    found_upper = False
    found_pvalues = False

    for key, record in results.items():
        pop1, pop2 = str(key[0]), str(key[1])

        obs = _extract_record_value(
            record,
            aliases=(
                "observed",
                "obs",
                "distance",
                "dist",
                "value",
                "nei",
                "nei_distance",
                "neis_distance",
                "neis_genetic_distance",
            ),
            scalar_fallback=True,
        )
        lo = _extract_record_value(
            record,
            aliases=("lower", "lower_ci", "ci_lower", "boot_lower", "bootstrap_lower"),
            scalar_fallback=False,
        )
        hi = _extract_record_value(
            record,
            aliases=("upper", "upper_ci", "ci_upper", "boot_upper", "bootstrap_upper"),
            scalar_fallback=False,
        )
        pval = _extract_record_value(
            record,
            aliases=("p", "pvalue", "p_value", "pvalues", "p_values"),
            scalar_fallback=False,
        )

        if obs is not None:
            observed.loc[pop1, pop2] = obs
            observed.loc[pop2, pop1] = obs
            found_observed = True

        if lo is not None:
            lower.loc[pop1, pop2] = lo
            lower.loc[pop2, pop1] = lo
            found_lower = True

        if hi is not None:
            upper.loc[pop1, pop2] = hi
            upper.loc[pop2, pop1] = hi
            found_upper = True

        if pval is not None:
            pvalues.loc[pop1, pop2] = pval
            pvalues.loc[pop2, pop1] = pval
            found_pvalues = True

    return PairwiseParts(
        observed=observed if found_observed else None,
        lower=lower if found_lower else None,
        upper=upper if found_upper else None,
        pvalues=pvalues if found_pvalues else None,
    )


def _coerce_square_frame(value: Any, name: str) -> pd.DataFrame | None:
    """Convert a value into a square pairwise DataFrame.

    Args:
        value: Object to convert.
        name: Name used in error messages.

    Returns:
        Square DataFrame or None.
    """
    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        df = value.copy()

    elif isinstance(value, Mapping):
        df = pd.DataFrame.from_dict(value, orient="index")

    else:
        return None

    df = df.apply(pd.to_numeric, errors="coerce")

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"{name} must be square, but got shape {df.shape}.")

    if list(df.index) != list(df.columns):
        raise ValueError(f"{name} index and columns must match.")

    return df


def _nested_dict_to_frame(value: Any) -> pd.DataFrame | None:
    """Convert nested dictionary to DataFrame.

    Args:
        value: Nested dictionary or None.

    Returns:
        DataFrame or None.
    """
    if value is None:
        return None

    df = pd.DataFrame.from_dict(value, orient="index")
    return df.apply(pd.to_numeric, errors="coerce")


def _frame_to_nested_dict(
    df: pd.DataFrame | None,
) -> dict[str, dict[str, float]] | None:
    """Convert a DataFrame to nested dictionary.

    Args:
        df: DataFrame or None.

    Returns:
        Nested dictionary or None.
    """
    if df is None:
        return None

    return df.to_dict(orient="index")


def _is_flat_pairwise_dict(results: Mapping[Any, Any]) -> bool:
    """Check whether a dictionary uses tuple pairwise population keys.

    Args:
        results: Dictionary to check.

    Returns:
        True if all keys are two-element tuple keys.
    """
    if not results:
        return False

    return all(isinstance(key, tuple) and len(key) == 2 for key in results)


def _infer_population_order(
    results: Mapping[Any, Any],
    population_order: Sequence[str] | None = None,
) -> list[str]:
    """Infer population order from pairwise tuple keys.

    Args:
        results: Flat pairwise result dictionary.
        population_order: Optional explicit population order.

    Returns:
        Population order.
    """
    observed_pops: set[str] = set()

    for key in results:
        observed_pops.add(str(key[0]))
        observed_pops.add(str(key[1]))

    if population_order is None:
        return sorted(observed_pops)

    ordered = [str(pop) for pop in population_order]
    missing = sorted(observed_pops - set(ordered))
    return ordered + missing


def _extract_record_value(
    record: Any,
    aliases: tuple[str, ...],
    scalar_fallback: bool,
) -> float | None:
    """Extract a numeric value from a pairwise record.

    Args:
        record: Pairwise result record.
        aliases: Candidate aliases if the record is dictionary-like.
        scalar_fallback: Whether to interpret scalar records as the target value.

    Returns:
        Extracted float or None.
    """
    if record is None:
        return None

    if isinstance(record, Mapping):
        value = _get_by_alias(record, aliases)
        return _to_float_or_none(value)

    if isinstance(record, pd.Series):
        value = _get_by_alias(record.to_dict(), aliases)
        if value is not None:
            return _to_float_or_none(value)

        if scalar_fallback and len(record) == 1:
            return _to_float_or_none(record.iloc[0])

        return None

    if scalar_fallback:
        return _to_float_or_none(record)

    return None


def _get_by_alias(data: Mapping[Any, Any], aliases: tuple[str, ...]) -> Any | None:
    """Get dictionary value using flexible aliases.

    Args:
        data: Dictionary-like object.
        aliases: Candidate aliases.

    Returns:
        Matching value or None.
    """
    normalized_aliases = {_normalize_key(alias) for alias in aliases}

    for key, value in data.items():
        if _normalize_key(key) in normalized_aliases:
            return value

    return None


def _normalize_key(key: Any) -> str:
    """Normalize dictionary key for flexible matching.

    Args:
        key: Dictionary key.

    Returns:
        Normalized key.
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


def _to_float_or_none(value: Any) -> float | None:
    """Convert scalar-like value to float.

    Args:
        value: Value to convert.

    Returns:
        Float or None.
    """
    if value is None:
        return None

    try:
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.item()

        if isinstance(value, pd.Series):
            if len(value) != 1:
                return None
            value = value.iloc[0]

        return float(value)

    except (TypeError, ValueError):
        return None
