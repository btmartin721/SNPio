#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pysam import include

from snpio import PopGenStatistics, VCFReader

"""
run_snpio.py

A helper script to run SNPio programmatically from within Docker or CLI.

Usage:
    python run_snpio.py \
        --input /app/data/0_original_alignments/example.vcf \
        --popmap /app/data/1_popmaps/example_popmap.txt \
        --prefix /app/results/snpio \
        --verbose \
        --debug \
        --version \
        --plot-format <png|pdf|svg> \
        --n-jobs <int> \
        --n-reps <int>
"""


@dataclass(frozen=True)
class PairwiseTableInputs:
    """Container for pairwise population-statistic table components.

    Attributes:
        observed: Observed pairwise statistic table.
        boot_lower: Bootstrap lower confidence limit table.
        boot_upper: Bootstrap upper confidence limit table.
        pvalues: Permutation p-value table.
        perm_lower: Optional lower bound of the permutation/null distribution.
        perm_upper: Optional upper bound of the permutation/null distribution.
        metric_prefix: Metric label used for captions and JSON keys, e.g. "Fst" or "Nei".
    """

    observed: pd.DataFrame
    boot_lower: pd.DataFrame | None = None
    boot_upper: pd.DataFrame | None = None
    pvalues: pd.DataFrame | None = None
    perm_lower: pd.DataFrame | None = None
    perm_upper: pd.DataFrame | None = None
    metric_prefix: str = "Fst"


def version():
    from snpio import __version__

    return str(__version__)


def to_jsonable(obj):
    """Recursively convert common scientific Python objects to JSON-safe objects.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable representation of the object.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(value) for value in obj]

    return obj


def load_pairwise_inputs_from_combined_json(
    table_json: str | Path,
    metric_prefix: str,
    population_order: list[str] | None = None,
) -> PairwiseTableInputs:
    """Load pairwise table inputs from a combined metric table-input JSON file.

    Args:
        table_json: Path to the combined table-input JSON file.
        metric_prefix: Metric prefix, e.g. "Fst" or "Nei".
        population_order: Optional population ordering.

    Returns:
        FstTableInputs-compatible object containing observed values, confidence
        intervals, p-values, and optional null bounds.
    """
    data = read_json(table_json)
    base = f"{metric_prefix}_between_populations"

    observed = nested_dict_to_frame(
        get_required_key(data, f"{base}_obs", str(table_json)),
        f"{base}_obs",
        population_order=population_order,
    )

    if population_order is None:
        population_order = list(observed.index)

    boot_lower = nested_dict_to_frame(
        get_required_key(data, f"{base}_boot_lower", str(table_json)),
        f"{base}_boot_lower",
        population_order=population_order,
    )

    boot_upper = nested_dict_to_frame(
        get_required_key(data, f"{base}_boot_upper", str(table_json)),
        f"{base}_boot_upper",
        population_order=population_order,
    )

    pvalues = nested_dict_to_frame(
        get_required_key(data, f"{base}_pvalues", str(table_json)),
        f"{base}_pvalues",
        population_order=population_order,
    )

    perm_lower = nested_dict_to_frame(
        data.get(f"{base}_perm_lower"),
        f"{base}_perm_lower",
        population_order=population_order,
    )

    perm_upper = nested_dict_to_frame(
        data.get(f"{base}_perm_upper"),
        f"{base}_perm_upper",
        population_order=population_order,
    )

    return FstTableInputs(
        observed=observed,
        boot_lower=boot_lower,
        boot_upper=boot_upper,
        pvalues=pvalues,
        perm_lower=perm_lower,
        perm_upper=perm_upper,
    )


def write_json(data, outfile: str | Path) -> None:
    """Write data to JSON after recursively converting unsupported objects.

    Args:
        data: Data object to serialize.
        outfile (str | Path): Output JSON path.

    Raises:
        ValueError: If the serialized object is an empty dictionary.
    """
    jsonable = to_jsonable(data)

    if isinstance(jsonable, dict) and not jsonable:
        raise ValueError(f"Refusing to write empty JSON object: {outfile}")

    with open(outfile, "w") as f:
        json.dump(jsonable, f, indent=4)


def validate_file(path: str, name: str) -> None:
    pth = Path(path)
    if not pth.exists() or not pth.is_file():
        print(f"ERROR: {name} file not found at: {path}")
        raise FileNotFoundError(f"{name} file not found: {path}")


def extract_pairwise_table_inputs(
    perm_stats: dict,
    boot_stats: dict,
    metric_prefix: str,
    include_nei: bool = True,
) -> dict:
    """Extract pairwise table inputs from permutation and bootstrap summaries.

    Args:
        perm_stats: Summary statistics generated with method="permutation".
        boot_stats: Summary statistics generated with method="bootstrap".
        metric_prefix: Metric prefix, e.g. "Fst" or "Nei".
        include_nei: Whether to include Nei's distance results. If False, only Fst results will be included.

    Returns:
        Dictionary containing observed values, bootstrap confidence intervals,
        permutation p-values, and optional permutation null intervals.

    Raises:
        KeyError: If required fields are missing.
    """
    base = f"{metric_prefix}_between_populations"

    obs_key = f"{base}_obs"
    lower_key = f"{base}_lower"
    upper_key = f"{base}_upper"
    pvalues_key = f"{base}_pvalues"

    required_perm = {obs_key, pvalues_key}
    required_boot = {obs_key, lower_key, upper_key}

    missing_perm = required_perm - set(perm_stats)
    missing_boot = required_boot - set(boot_stats)

    is_nei = metric_prefix.lower() == "nei"

    if missing_perm:
        if not include_nei and is_nei:
            print(
                f"Warning: Missing permutation results for {metric_prefix}, but skipping error since include_nei=False."
            )
        else:
            raise KeyError(
                f"Permutation {metric_prefix} results missing keys: {sorted(missing_perm)}"
            )

    if missing_boot:
        if not include_nei and is_nei:
            print(
                f"Warning: Missing bootstrap results for {metric_prefix}, but skipping error since include_nei=False."
            )
        else:
            raise KeyError(
                f"Bootstrap {metric_prefix} results missing keys: {sorted(missing_boot)}"
            )

    if is_nei and not include_nei:
        return {}

    return {
        f"{base}_obs": boot_stats[obs_key],
        f"{base}_boot_lower": boot_stats[lower_key],
        f"{base}_boot_upper": boot_stats[upper_key],
        f"{base}_pvalues": perm_stats[pvalues_key],
        f"{base}_perm_lower": perm_stats.get(lower_key),
        f"{base}_perm_upper": perm_stats.get(upper_key),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SNPio",
        description="Run SNPio with specified input, popmap, and output prefix.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (VCF, PHYLIP, or STRUCTURE format).",
    )
    parser.add_argument(
        "--popmap",
        type=str,
        required=True,
        help="Path to popmap file mapping samples to populations. Format: <sample>\t<population>",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Output prefix for results (output files will be saved as <prefix>_output/*)",
    )
    parser.add_argument(
        "--include-nei",
        action="store_true",
        default=False,
        help="Whether to include Nei's distance results. If not set, only Fst results will be included in the outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging. Includes additional logging information during processing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Includes additional logging and checks. This may slow down processing.",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for output plots. Options: png, pdf, svg (default: png)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to use for computations. Default is -1 (use all available cores).",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=1000,
        help="Number of replicates for permutation tests or bootstrapping. Default is 1000.",
    )

    parser.add_argument(
        "--version",
        default=False,
        required=False,
        action="store_true",
        help="Show the version of SNPio and exit.",
    )

    args = parser.parse_args()

    if args.version:
        print(f"SNPio version {version()}")
        exit(0)

    return args


def main():
    args = parse_args()

    # Validate paths
    validate_file(args.input, "Input")
    validate_file(args.popmap, "Popmap")

    print(f"🧬 Running SNPio version {version()} with the following arguments:")
    print(f"  📥 Input file:     {args.input}")
    print(f"  🧾 Popmap file:    {args.popmap}")
    print(f"  ✅ Include Nei's distance: {args.include_nei}")
    print(f"  🔁 Replicates:      {args.n_reps}")
    print(f"  ⚙️ Jobs:            {args.n_jobs}")
    print(f"  📁 Output prefix:  {args.prefix}")
    print(f"  🖼️ Plot format:     {args.plot_format}")
    print(f"  🔍 Verbose:         {args.verbose}")
    print(f"  🐛 Debug:           {args.debug}")
    print()

    gd = VCFReader(
        filename=args.input,
        popmapfile=args.popmap,
        force_popmap=True,
        chunk_size=5000,
        include_pops=["EA", "GU", "TT", "ON", "OG"],
        prefix=args.prefix,
        plot_format=args.plot_format,
        verbose=args.verbose,
        debug=args.debug,
        # allele_encoding={"0": "A", "1": "C", "2": "G", "3": "T", "-9": "N"},
    )

    gd.write_popmap(filename=f"{args.prefix}_popmap.txt")
    gd.write_vcf(f"{args.prefix}_popgen.vcf.gz")

    pgs = PopGenStatistics(gd, verbose=args.verbose, debug=args.debug)

    allele_summary_stats_perm, summary_stats_perm = pgs.summary_statistics(
        method="permutation",
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        save_plots=True,
        include_nei=args.include_nei,
    )

    allele_summary_stats_boot, summary_stats_boot = pgs.summary_statistics(
        method="bootstrap",
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        save_plots=True,
        include_nei=args.include_nei,
    )

    results_dir = Path(f"{args.prefix}")
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    write_json(
        allele_summary_stats_perm,
        f"{args.prefix}_allele_summary_stats_perm.json",
    )

    write_json(
        allele_summary_stats_boot,
        f"{args.prefix}_allele_summary_stats_boot.json",
    )

    fst_table_inputs = extract_pairwise_table_inputs(
        perm_stats=allele_summary_stats_perm,
        boot_stats=allele_summary_stats_boot,
        metric_prefix="Fst",
        include_nei=args.include_nei,
    )

    write_json(
        fst_table_inputs,
        f"{args.prefix}_fst_table_inputs.json",
    )

    nei_table_inputs = extract_pairwise_table_inputs(
        perm_stats=allele_summary_stats_perm,
        boot_stats=allele_summary_stats_boot,
        metric_prefix="Nei",
        include_nei=args.include_nei,
    )

    if args.include_nei:
        write_json(
            nei_table_inputs,
            f"{args.prefix}_nei_table_inputs.json",
        )

    summary_stats_perm.to_csv(f"{args.prefix}_summary_stats_perm.csv", index=False)
    summary_stats_boot.to_csv(f"{args.prefix}_summary_stats_boot.csv", index=False)


if __name__ == "__main__":
    main()
