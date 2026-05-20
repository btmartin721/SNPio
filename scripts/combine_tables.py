#!/usr/bin/env python3
"""Create manuscript-ready pairwise SNPio Fst or Nei tables from JSON outputs.

Recommended manuscript table:
    Upper triangle: observed statistic with bootstrap 95% CI
    Lower triangle: permutation p-value

Examples:
    Fst from combined JSON:

        python combine_fst_tables.py \
            --fst-json snpio_fst_table_inputs.json \
            --output wc_fst_manuscript_table.xlsx \
            --table-type obs-ci-pvalues \
            --population-order EA,GU,TT,ON,OG \
            --decimals 3 \
            --pvalue-decimals 3 \
            --diagonal emdash \
            --stars

    Nei from combined JSON:

        python combine_fst_tables.py \
            --nei-json snpio_nei_table_inputs.json \
            --output nei_distance_manuscript_table.xlsx \
            --table-type obs-ci-pvalues \
            --population-order EA,GU,TT,ON,OG \
            --decimals 3 \
            --pvalue-decimals 3 \
            --diagonal emdash \
            --stars

    Fst from separate permutation/bootstrap summary JSON files:

        python combine_fst_tables.py \
            --metric Fst \
            --perm-json snpio_allele_summary_stats_perm.json \
            --boot-json snpio_allele_summary_stats_boot.json \
            --output wc_fst_manuscript_table.xlsx \
            --table-type obs-ci-pvalues \
            --stars

    Nei from separate permutation/bootstrap summary JSON files:

        python combine_fst_tables.py \
            --metric Nei \
            --perm-json snpio_allele_summary_stats_perm.json \
            --boot-json snpio_allele_summary_stats_boot.json \
            --output nei_distance_manuscript_table.xlsx \
            --table-type obs-ci-pvalues \
            --stars
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

Metric = Literal["Fst", "Nei"]
TableType = Literal[
    "obs-ci-pvalues",
    "obs-pvalues",
    "boot-ci-bounds",
    "perm-null-bounds",
]
DiagonalMode = Literal["emdash", "blank", "zero", "nan"]


@dataclass(frozen=True)
class PairwiseTableInputs:
    """Container for pairwise population-statistic table components.

    Attributes:
        metric: Metric prefix, either "Fst" or "Nei".
        observed: Observed pairwise statistic table.
        boot_lower: Bootstrap lower confidence limit table.
        boot_upper: Bootstrap upper confidence limit table.
        pvalues: Permutation p-value table.
        perm_lower: Optional lower bound of the permutation/null distribution.
        perm_upper: Optional upper bound of the permutation/null distribution.
    """

    metric: Metric
    observed: pd.DataFrame
    boot_lower: pd.DataFrame | None = None
    boot_upper: pd.DataFrame | None = None
    pvalues: pd.DataFrame | None = None
    perm_lower: pd.DataFrame | None = None
    perm_upper: pd.DataFrame | None = None


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        TypeError: If the top-level JSON object is not a dictionary.
    """
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object to be a dict: {path}")

    return data


def normalize_metric(metric: str) -> Metric:
    """Normalize a metric name.

    Args:
        metric: Metric name.

    Returns:
        Normalized metric name.

    Raises:
        ValueError: If the metric is unsupported.
    """
    metric_norm = metric.strip().lower().replace("_", "").replace("-", "")

    if metric_norm in {"fst", "fstatistic"}:
        return "Fst"

    if metric_norm in {
        "nei",
        "neis",
        "neidistance",
        "neisdistance",
        "neigeneticdistance",
        "neisgeneticdistance",
    }:
        return "Nei"

    raise ValueError("Unsupported metric. Choose 'Fst' or 'Nei'.")


def metric_display_name(metric: Metric) -> str:
    """Return display name for a metric.

    Args:
        metric: Metric prefix.

    Returns:
        Human-readable metric name.
    """
    if metric == "Fst":
        return "Fst"

    if metric == "Nei":
        return "Nei's genetic distance"

    raise ValueError(f"Unsupported metric: {metric}")


def metric_label(metric: Metric) -> str:
    """Return LaTeX-safe label fragment for a metric.

    Args:
        metric: Metric prefix.

    Returns:
        Label fragment.
    """
    if metric == "Fst":
        return "fst"

    if metric == "Nei":
        return "nei_distance"

    raise ValueError(f"Unsupported metric: {metric}")


def infer_metric_from_json(
    data: Mapping[str, Any],
    metric: str | None = None,
) -> Metric:
    """Infer metric prefix from a JSON object.

    Args:
        data: Parsed JSON object.
        metric: Optional explicitly provided metric.

    Returns:
        Metric prefix.

    Raises:
        ValueError: If the metric cannot be inferred unambiguously.
    """
    if metric is not None:
        return normalize_metric(metric)

    metric_from_file = data.get("metric")
    if isinstance(metric_from_file, str):
        return normalize_metric(metric_from_file)

    detected = []
    for candidate in ("Fst", "Nei"):
        prefix = f"{candidate}_between_populations"
        if any(str(key).startswith(prefix) for key in data):
            detected.append(candidate)

    if len(detected) == 1:
        return detected[0]  # type: ignore[return-value]

    if len(detected) > 1:
        raise ValueError(
            "JSON contains both Fst and Nei keys. Provide --metric Fst or --metric Nei."
        )

    raise ValueError(
        "Could not infer metric from JSON. Provide --metric Fst or --metric Nei."
    )


def get_first_present(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    """Return the first present, non-None value from a dictionary.

    Args:
        data: Dictionary to query.
        keys: Candidate keys.

    Returns:
        First matching value, or None.
    """
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return None


def nested_dict_to_frame(
    data: Mapping[str, Mapping[str, Any]] | None,
    name: str,
    population_order: list[str] | None = None,
) -> pd.DataFrame | None:
    """Convert a nested pairwise dictionary into a square DataFrame.

    Args:
        data: Nested dictionary of pairwise values.
        name: Name of the table for error messages.
        population_order: Optional population ordering.

    Returns:
        Square pairwise DataFrame, or None if data is None.

    Raises:
        TypeError: If data is not a nested dictionary.
        ValueError: If the resulting table is not square or labels are inconsistent.
    """
    if data is None:
        return None

    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a nested dictionary, got {type(data)}.")

    if not data:
        raise ValueError(f"{name} is empty.")

    if not all(isinstance(value, Mapping) for value in data.values()):
        raise TypeError(f"{name} must be a nested dictionary of dictionaries.")

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.apply(pd.to_numeric, errors="coerce")

    if population_order is None:
        population_order = list(data.keys())

    missing_rows = sorted(set(population_order) - set(df.index))
    missing_cols = sorted(set(population_order) - set(df.columns))

    if missing_rows:
        raise ValueError(f"{name} is missing row labels: {missing_rows}")

    if missing_cols:
        raise ValueError(f"{name} is missing column labels: {missing_cols}")

    df = df.loc[population_order, population_order]

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"{name} must be square, but got shape {df.shape}.")

    if list(df.index) != list(df.columns):
        raise ValueError(f"{name} index and columns must contain the same labels.")

    return df


def load_pairwise_inputs_from_combined_json(
    table_json: str | Path,
    metric: str | None = None,
    population_order: list[str] | None = None,
) -> PairwiseTableInputs:
    """Load pairwise table inputs from a combined metric table-input JSON file.

    Args:
        table_json: Path to combined table-input JSON.
        metric: Metric prefix, either "Fst" or "Nei". If omitted, inferred from JSON.
        population_order: Optional population ordering.

    Returns:
        PairwiseTableInputs object.

    Raises:
        KeyError: If the observed table is missing.
    """
    data = read_json(table_json)
    metric_name = infer_metric_from_json(data, metric)
    base = f"{metric_name}_between_populations"

    observed_raw = get_first_present(
        data,
        (f"{base}_obs", f"{base}_observed"),
    )

    if observed_raw is None:
        raise KeyError(f"{table_json} is missing required key: {base}_obs")

    observed = nested_dict_to_frame(
        observed_raw,
        f"{base}_obs",
        population_order=population_order,
    )

    if observed is None:
        raise ValueError(f"{base}_obs could not be converted to a table.")

    if population_order is None:
        population_order = list(observed.index)

    boot_lower = nested_dict_to_frame(
        get_first_present(
            data,
            (
                f"{base}_boot_lower",
                f"{base}_bootstrap_lower",
            ),
        ),
        f"{base}_boot_lower",
        population_order=population_order,
    )
    boot_upper = nested_dict_to_frame(
        get_first_present(
            data,
            (
                f"{base}_boot_upper",
                f"{base}_bootstrap_upper",
            ),
        ),
        f"{base}_boot_upper",
        population_order=population_order,
    )
    pvalues = nested_dict_to_frame(
        get_first_present(
            data,
            (
                f"{base}_pvalues",
                f"{base}_p_values",
                f"{base}_pvalue",
                f"{base}_p_value",
            ),
        ),
        f"{base}_pvalues",
        population_order=population_order,
    )
    perm_lower = nested_dict_to_frame(
        get_first_present(
            data,
            (
                f"{base}_perm_lower",
                f"{base}_permutation_lower",
            ),
        ),
        f"{base}_perm_lower",
        population_order=population_order,
    )
    perm_upper = nested_dict_to_frame(
        get_first_present(
            data,
            (
                f"{base}_perm_upper",
                f"{base}_permutation_upper",
            ),
        ),
        f"{base}_perm_upper",
        population_order=population_order,
    )

    return PairwiseTableInputs(
        metric=metric_name,
        observed=observed,
        boot_lower=boot_lower,
        boot_upper=boot_upper,
        pvalues=pvalues,
        perm_lower=perm_lower,
        perm_upper=perm_upper,
    )


def load_pairwise_inputs_from_separate_jsons(
    perm_json: str | Path,
    boot_json: str | Path,
    metric: str,
    population_order: list[str] | None = None,
) -> PairwiseTableInputs:
    """Load pairwise table inputs from separate permutation and bootstrap JSON files.

    Args:
        perm_json: Path to permutation summary JSON.
        boot_json: Path to bootstrap summary JSON.
        metric: Metric prefix, either "Fst" or "Nei".
        population_order: Optional population ordering.

    Returns:
        PairwiseTableInputs object.

    Raises:
        KeyError: If the observed table is missing from both JSON files.
    """
    metric_name = normalize_metric(metric)
    base = f"{metric_name}_between_populations"

    perm = read_json(perm_json)
    boot = read_json(boot_json)

    observed_raw = get_first_present(
        boot,
        (
            f"{base}_obs",
            f"{base}_observed",
        ),
    )

    if observed_raw is None:
        observed_raw = get_first_present(
            perm,
            (
                f"{base}_obs",
                f"{base}_observed",
            ),
        )

    if observed_raw is None:
        raise KeyError(f"Could not find {base}_obs in bootstrap or permutation JSON.")

    observed = nested_dict_to_frame(
        observed_raw,
        f"{base}_obs",
        population_order=population_order,
    )

    if observed is None:
        raise ValueError(f"{base}_obs could not be converted to a table.")

    if population_order is None:
        population_order = list(observed.index)

    boot_lower = nested_dict_to_frame(
        get_first_present(
            boot,
            (
                f"{base}_boot_lower",
                f"{base}_bootstrap_lower",
                f"{base}_lower",
            ),
        ),
        f"{base}_boot_lower",
        population_order=population_order,
    )
    boot_upper = nested_dict_to_frame(
        get_first_present(
            boot,
            (
                f"{base}_boot_upper",
                f"{base}_bootstrap_upper",
                f"{base}_upper",
            ),
        ),
        f"{base}_boot_upper",
        population_order=population_order,
    )
    pvalues = nested_dict_to_frame(
        get_first_present(
            perm,
            (
                f"{base}_pvalues",
                f"{base}_p_values",
                f"{base}_pvalue",
                f"{base}_p_value",
            ),
        ),
        f"{base}_pvalues",
        population_order=population_order,
    )
    perm_lower = nested_dict_to_frame(
        get_first_present(
            perm,
            (
                f"{base}_perm_lower",
                f"{base}_permutation_lower",
                f"{base}_lower",
            ),
        ),
        f"{base}_perm_lower",
        population_order=population_order,
    )
    perm_upper = nested_dict_to_frame(
        get_first_present(
            perm,
            (
                f"{base}_perm_upper",
                f"{base}_permutation_upper",
                f"{base}_upper",
            ),
        ),
        f"{base}_perm_upper",
        population_order=population_order,
    )

    return PairwiseTableInputs(
        metric=metric_name,
        observed=observed,
        boot_lower=boot_lower,
        boot_upper=boot_upper,
        pvalues=pvalues,
        perm_lower=perm_lower,
        perm_upper=perm_upper,
    )


def validate_matching_tables(
    reference: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame | None],
) -> None:
    """Validate that all available tables match the reference table.

    Args:
        reference: Reference square DataFrame.
        tables: Mapping of table names to DataFrames.

    Raises:
        ValueError: If any table has mismatched shape, index, or columns.
    """
    for name, table in tables.items():
        if table is None:
            continue

        if table.shape != reference.shape:
            raise ValueError(
                f"{name} shape does not match observed pairwise table. "
                f"Expected {reference.shape}, got {table.shape}."
            )

        if list(table.index) != list(reference.index):
            raise ValueError(f"{name} row labels do not match observed table.")

        if list(table.columns) != list(reference.columns):
            raise ValueError(f"{name} column labels do not match observed table.")


def significance_stars(p_value: float) -> str:
    """Return conventional significance stars for a p-value.

    Args:
        p_value: P-value.

    Returns:
        Significance star string.
    """
    if np.isnan(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return ""


def format_numeric(value: float, decimals: int) -> str:
    """Format a numeric value.

    Args:
        value: Numeric value.
        decimals: Number of decimal places.

    Returns:
        Formatted numeric string.
    """
    if pd.isna(value):
        return ""

    return f"{float(value):.{decimals}f}"


def format_pvalue(
    value: float,
    decimals: int,
    stars: bool = False,
) -> str:
    """Format a p-value for manuscript display.

    Args:
        value: P-value.
        decimals: Number of decimal places.
        stars: Whether to append significance stars.

    Returns:
        Formatted p-value string.
    """
    if pd.isna(value):
        return ""

    value = float(value)
    threshold = 10 ** (-decimals)

    if value < threshold:
        formatted = f"<{threshold:.{decimals}f}"
    else:
        formatted = f"{value:.{decimals}f}"

    if stars:
        formatted = f"{formatted}{significance_stars(value)}"

    return formatted


def format_stat_ci(
    observed: float,
    lower: float,
    upper: float,
    decimals: int,
    ci_separator: str = ", ",
) -> str:
    """Format observed statistic with bootstrap confidence interval.

    Args:
        observed: Observed statistic value.
        lower: Bootstrap lower confidence limit.
        upper: Bootstrap upper confidence limit.
        decimals: Number of decimal places.
        ci_separator: Separator between lower and upper CI values.

    Returns:
        Formatted statistic confidence interval string.
    """
    if pd.isna(observed):
        return ""

    observed_fmt = format_numeric(observed, decimals)

    if pd.isna(lower) or pd.isna(upper):
        return observed_fmt

    lower_fmt = format_numeric(lower, decimals)
    upper_fmt = format_numeric(upper, decimals)

    return f"{observed_fmt} [{lower_fmt}{ci_separator}{upper_fmt}]"


def diagonal_value(mode: DiagonalMode, decimals: int) -> str:
    """Generate the requested diagonal display value.

    Args:
        mode: Diagonal display mode.
        decimals: Number of decimal places for zero diagonal.

    Returns:
        Diagonal display value.
    """
    if mode == "emdash":
        return "—"

    if mode == "blank":
        return ""

    if mode == "zero":
        return f"{0.0:.{decimals}f}"

    if mode == "nan":
        return "NaN"

    raise ValueError(
        "Invalid diagonal mode. Choose from: 'emdash', 'blank', 'zero', or 'nan'."
    )


def make_manuscript_table(
    pairwise_inputs: PairwiseTableInputs,
    table_type: TableType = "obs-ci-pvalues",
    decimals: int = 3,
    pvalue_decimals: int = 3,
    diagonal: DiagonalMode = "emdash",
    stars: bool = False,
) -> pd.DataFrame:
    """Create a manuscript-ready pairwise population-statistic table.

    Args:
        pairwise_inputs: Pairwise table input object.
        table_type: Type of manuscript table to create.
        decimals: Number of decimal places for statistic and CI values.
        pvalue_decimals: Number of decimal places for p-values.
        diagonal: Diagonal display mode.
        stars: Whether to append significance stars to p-values.

    Returns:
        Manuscript-ready pairwise table as strings.

    Raises:
        ValueError: If requested table type requires unavailable fields.
    """
    observed = pairwise_inputs.observed

    validate_matching_tables(
        reference=observed,
        tables={
            "boot_lower": pairwise_inputs.boot_lower,
            "boot_upper": pairwise_inputs.boot_upper,
            "pvalues": pairwise_inputs.pvalues,
            "perm_lower": pairwise_inputs.perm_lower,
            "perm_upper": pairwise_inputs.perm_upper,
        },
    )

    if table_type == "obs-ci-pvalues":
        if pairwise_inputs.boot_lower is None or pairwise_inputs.boot_upper is None:
            raise ValueError(
                "table_type='obs-ci-pvalues' requires bootstrap lower and upper "
                "confidence interval tables."
            )
        if pairwise_inputs.pvalues is None:
            raise ValueError(
                "table_type='obs-ci-pvalues' requires permutation p-values."
            )

    if table_type == "obs-pvalues" and pairwise_inputs.pvalues is None:
        raise ValueError("table_type='obs-pvalues' requires permutation p-values.")

    if table_type == "boot-ci-bounds":
        if pairwise_inputs.boot_lower is None or pairwise_inputs.boot_upper is None:
            raise ValueError(
                "table_type='boot-ci-bounds' requires bootstrap lower and upper "
                "confidence interval tables."
            )

    if table_type == "perm-null-bounds":
        if pairwise_inputs.perm_lower is None or pairwise_inputs.perm_upper is None:
            raise ValueError(
                "table_type='perm-null-bounds' requires permutation lower and upper "
                "null-distribution interval tables."
            )

    pops = list(observed.index)
    n_pops = len(pops)
    combined = pd.DataFrame("", index=pops, columns=pops, dtype=object)
    diag = diagonal_value(diagonal, decimals)

    for i in range(n_pops):
        for j in range(n_pops):
            if i == j:
                combined.iat[i, j] = diag
                continue

            if table_type == "obs-ci-pvalues":
                if i < j:
                    combined.iat[i, j] = format_stat_ci(
                        observed=observed.iat[i, j],
                        lower=pairwise_inputs.boot_lower.iat[i, j],
                        upper=pairwise_inputs.boot_upper.iat[i, j],
                        decimals=decimals,
                    )
                else:
                    combined.iat[i, j] = format_pvalue(
                        pairwise_inputs.pvalues.iat[i, j],
                        decimals=pvalue_decimals,
                        stars=stars,
                    )

            elif table_type == "obs-pvalues":
                if i < j:
                    combined.iat[i, j] = format_numeric(observed.iat[i, j], decimals)
                else:
                    combined.iat[i, j] = format_pvalue(
                        pairwise_inputs.pvalues.iat[i, j],
                        decimals=pvalue_decimals,
                        stars=stars,
                    )

            elif table_type == "boot-ci-bounds":
                if i < j:
                    combined.iat[i, j] = format_numeric(
                        pairwise_inputs.boot_upper.iat[i, j],
                        decimals,
                    )
                else:
                    combined.iat[i, j] = format_numeric(
                        pairwise_inputs.boot_lower.iat[i, j],
                        decimals,
                    )

            elif table_type == "perm-null-bounds":
                if i < j:
                    combined.iat[i, j] = format_numeric(
                        pairwise_inputs.perm_upper.iat[i, j],
                        decimals,
                    )
                else:
                    combined.iat[i, j] = format_numeric(
                        pairwise_inputs.perm_lower.iat[i, j],
                        decimals,
                    )

            else:
                raise ValueError(f"Unsupported table type: {table_type}")

    return combined


def write_table(
    df: pd.DataFrame,
    output: str | Path,
    table_type: TableType,
    metric: Metric,
) -> None:
    """Write a manuscript table to CSV, TSV, Excel, or LaTeX.

    Args:
        df: Output table.
        output: Output file path.
        table_type: Table type, used for LaTeX captions.
        metric: Metric prefix, either "Fst" or "Nei".

    Raises:
        ValueError: If the file extension is unsupported.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output)
        return

    if suffix in {".tsv", ".txt"}:
        df.to_csv(output, sep="\t")
        return

    if suffix == ".xlsx":
        df.to_excel(output)
        return

    if suffix == ".tex":
        display = metric_display_name(metric)
        label_metric = metric_label(metric)

        captions = {
            "obs-ci-pvalues": (
                f"Pairwise {display} estimates with bootstrap confidence intervals "
                "and permutation p-values."
            ),
            "obs-pvalues": f"Pairwise {display} estimates and permutation p-values.",
            "boot-ci-bounds": (
                f"Bootstrap confidence interval bounds for pairwise {display}."
            ),
            "perm-null-bounds": (
                f"Permutation null-distribution interval bounds for pairwise {display}."
            ),
        }

        labels = {
            "obs-ci-pvalues": f"tab:pairwise_{label_metric}_ci_pvalues",
            "obs-pvalues": f"tab:pairwise_{label_metric}_pvalues",
            "boot-ci-bounds": f"tab:pairwise_{label_metric}_bootstrap_ci",
            "perm-null-bounds": f"tab:pairwise_{label_metric}_permutation_null",
        }

        latex = df.to_latex(
            escape=False,
            index=True,
            bold_rows=False,
            caption=captions[table_type],
            label=labels[table_type],
        )
        output.write_text(latex)
        return

    raise ValueError(f"Unsupported output file extension: {suffix}")


def parse_population_order(population_order: str | None) -> list[str] | None:
    """Parse a comma-separated population order string.

    Args:
        population_order: Comma-separated population labels.

    Returns:
        List of population labels, or None.
    """
    if population_order is None:
        return None

    pops = [pop.strip() for pop in population_order.split(",") if pop.strip()]

    if not pops:
        raise ValueError("--population-order was provided but no labels were parsed.")

    return pops


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript-ready pairwise SNPio Fst or Nei tables from "
            "permutation and bootstrap JSON outputs."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--json",
        help=(
            "Generic combined table-input JSON file. Use with --metric if the "
            "metric cannot be inferred."
        ),
    )
    input_group.add_argument(
        "--fst-json",
        help=(
            "Combined Fst table-input JSON file, e.g. "
            "<prefix>_fst_table_inputs.json."
        ),
    )
    input_group.add_argument(
        "--nei-json",
        help=(
            "Combined Nei table-input JSON file, e.g. "
            "<prefix>_nei_table_inputs.json."
        ),
    )
    input_group.add_argument(
        "--perm-json",
        help=(
            "Permutation summary JSON file, e.g. "
            "<prefix>_allele_summary_stats_perm.json. Requires --boot-json."
        ),
    )

    parser.add_argument(
        "--boot-json",
        help=(
            "Bootstrap summary JSON file, e.g. "
            "<prefix>_allele_summary_stats_boot.json. Required with --perm-json."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["Fst", "Nei"],
        default=None,
        help=(
            "Metric to extract from JSON files. Required when using --perm-json "
            "with summary JSONs containing both Fst and Nei. Defaults to Fst "
            "for separate JSON mode if omitted."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path: .csv, .tsv, .txt, .xlsx, or .tex.",
    )
    parser.add_argument(
        "--table-type",
        choices=[
            "obs-ci-pvalues",
            "obs-pvalues",
            "boot-ci-bounds",
            "perm-null-bounds",
        ],
        default="obs-ci-pvalues",
        help=(
            "Type of table to create. Default: obs-ci-pvalues, where the upper "
            "triangle is observed statistic with bootstrap CI and the lower "
            "triangle is the permutation p-value."
        ),
    )
    parser.add_argument(
        "--population-order",
        default=None,
        help=(
            "Optional comma-separated population order, e.g. 'EA,GU,TT,ON,OG'. "
            "If omitted, the order from the JSON file is used."
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimal places for statistic and CI values.",
    )
    parser.add_argument(
        "--pvalue-decimals",
        type=int,
        default=3,
        help="Number of decimal places for p-values.",
    )
    parser.add_argument(
        "--diagonal",
        choices=["emdash", "blank", "zero", "nan"],
        default="emdash",
        help="How to display diagonal self-comparisons.",
    )
    parser.add_argument(
        "--stars",
        action="store_true",
        help="Append significance stars to p-values.",
    )

    args = parser.parse_args()

    if args.perm_json and not args.boot_json:
        parser.error("--boot-json is required when using --perm-json.")

    if args.boot_json and not args.perm_json:
        parser.error("--boot-json can only be used together with --perm-json.")

    return args


def main() -> None:
    """Run the manuscript-table workflow."""
    args = parse_args()
    population_order = parse_population_order(args.population_order)

    if args.fst_json:
        pairwise_inputs = load_pairwise_inputs_from_combined_json(
            table_json=args.fst_json,
            metric="Fst",
            population_order=population_order,
        )

    elif args.nei_json:
        pairwise_inputs = load_pairwise_inputs_from_combined_json(
            table_json=args.nei_json,
            metric="Nei",
            population_order=population_order,
        )

    elif args.json:
        pairwise_inputs = load_pairwise_inputs_from_combined_json(
            table_json=args.json,
            metric=args.metric,
            population_order=population_order,
        )

    else:
        metric = args.metric or "Fst"
        pairwise_inputs = load_pairwise_inputs_from_separate_jsons(
            perm_json=args.perm_json,
            boot_json=args.boot_json,
            metric=metric,
            population_order=population_order,
        )

    manuscript_table = make_manuscript_table(
        pairwise_inputs=pairwise_inputs,
        table_type=args.table_type,
        decimals=args.decimals,
        pvalue_decimals=args.pvalue_decimals,
        diagonal=args.diagonal,
        stars=args.stars,
    )

    write_table(
        df=manuscript_table,
        output=args.output,
        table_type=args.table_type,
        metric=pairwise_inputs.metric,
    )


if __name__ == "__main__":
    main()
