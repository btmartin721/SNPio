#!/usr/bin/env python3
"""Run independent validation workflows for SNPio linkage disequilibrium."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Iterator, Sequence

import pandas as pd
from tqdm.auto import tqdm

from snpio import GenePopReader, PopGenStatistics, VCFReader
from snpio.validation.ld_simulation import (
    Fwdpy11CalibrationConfig,
    run_fwdpy11_calibration,
)
from snpio.validation.ld_plots import (
    plot_exact_expectation_errors,
    plot_golden_reference_errors,
    plot_pair_convergence,
    plot_published_island_fox_comparison,
    plot_simulation_calibration,
)
from snpio.validation.linkage_disequilibrium import (
    PUBLISHED_ISLAND_FOX_ESTIMATES,
    compare_published_estimates,
    prepare_island_fox_genepop,
    summarize_convergence,
    validate_exact_expectations,
    validate_golden_reference,
)

DEFAULT_OUTPUT = Path("validation_results") / "linkage_disequilibrium"
CONVERGENCE_HEARTBEAT_SECONDS = 15.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _open_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must lie between zero and one")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Validation output root (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png", "pdf"],
        help="Static validation plot formats (default: png pdf).",
    )
    parser.add_argument(
        "--plot-dpi",
        type=_positive_int,
        default=300,
        help="Resolution for PNG validation plots (default: 300).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    exact = subparsers.add_parser(
        "exact", help="Enumerate exact small-sample multinomial expectations."
    )
    exact.add_argument("--sample-sizes", type=_positive_int, nargs="+", default=[4, 6])
    exact.add_argument("--atol", type=float, default=1e-12)

    golden = subparsers.add_parser(
        "golden", help="Compare against the frozen moments-popgen 1.6.0 corpus."
    )
    golden.add_argument("--fixture", type=Path, default=None)
    golden.add_argument("--rtol", type=float, default=1e-12)
    golden.add_argument("--atol", type=float, default=1e-14)

    simulation = subparsers.add_parser(
        "simulate", help="Run neutral forward-time fwdpy11/tskit calibration."
    )
    simulation.add_argument(
        "--population-sizes",
        type=_positive_int,
        nargs="+",
        default=[10, 25, 50, 100, 400],
    )
    simulation.add_argument(
        "--sample-sizes", type=_positive_int, nargs="+", default=[6, 8, 20, 50]
    )
    simulation.add_argument("--replicates", type=_positive_int, default=100)
    simulation.add_argument("--chromosomes", type=_positive_int, default=8)
    simulation.add_argument("--loci-per-chromosome", type=_positive_int, default=100)
    simulation.add_argument(
        "--burnin-multiplier",
        type=_positive_int,
        default=10,
        help="Forward burn-in in units of census N (default: 10).",
    )
    selfing = simulation.add_mutually_exclusive_group()
    selfing.add_argument(
        "--allow-residual-selfing",
        dest="allow_residual_selfing",
        action="store_true",
        help=(
            "Use standard Wright-Fisher parent sampling with replacement "
            "(default; retained as an explicit compatibility flag)."
        ),
    )
    selfing.add_argument(
        "--prohibit-residual-selfing",
        dest="allow_residual_selfing",
        action="store_false",
        help=(
            "Require two distinct parents as a model-sensitivity analysis; "
            "this does not target the standard 1/(3N) Wright-Fisher result."
        ),
    )
    simulation.set_defaults(allow_residual_selfing=True)
    simulation.add_argument("--n-bootstraps", type=_nonnegative_int, default=200)
    simulation.add_argument(
        "--minimum-model-population-size",
        type=_positive_int,
        default=100,
        help=(
            "Smallest N used for formal asymptotic 1/(3N) model checks "
            "(default: 100)."
        ),
    )
    simulation.add_argument(
        "--minimum-coverage-sample-size",
        type=_positive_int,
        default=8,
        help=(
            "Smallest diploid sample eligible for the optional strict "
            "population-coverage check (default: 8)."
        ),
    )
    simulation.add_argument(
        "--model-relative-bias-tolerance",
        type=_open_unit_interval,
        default=0.05,
        help=(
            "Practical relative-bias margin for formal r2D model checks "
            "(default: 0.05)."
        ),
    )
    simulation.add_argument(
        "--require-population-coverage",
        action="store_true",
        help=(
            "Make matched-census population-target coverage a formal gate. "
            "By default it is diagnostic because the interval resamples "
            "chromosomes, not sampled individuals."
        ),
    )
    simulation.add_argument("--n-jobs", type=int, default=1)
    simulation.add_argument("--seed", type=_nonnegative_int, default=20260715)
    simulation.add_argument(
        "--quick",
        action="store_true",
        help="Use a small smoke grid (N=10,25; n=6,8; 3 replicates).",
    )

    published = subparsers.add_parser(
        "published", help="Reproduce the published island-fox table 1 benchmark."
    )
    published.add_argument("--genepop", type=Path, required=True)
    published.add_argument("--n-jobs", type=int, default=1)
    published.add_argument("--seed", type=_nonnegative_int, default=20260715)
    published.add_argument("--relative-tolerance", type=float, default=0.05)

    convergence = subparsers.add_parser(
        "convergence", help="Measure LD stability across locus-pair budgets and seeds."
    )
    convergence.add_argument("--vcf", type=Path, required=True)
    convergence.add_argument("--popmap", type=Path, required=True)
    convergence.add_argument(
        "--pair-budgets",
        type=_positive_int,
        nargs="+",
        default=[250_000, 1_000_000, 4_000_000],
    )
    convergence.add_argument(
        "--seeds", type=_nonnegative_int, nargs="+", default=[101, 203, 307, 409, 503]
    )
    convergence.add_argument("--n-jobs", type=int, default=1)
    convergence.add_argument("--n-bootstraps", type=_nonnegative_int, default=0)
    convergence.add_argument("--assume-unlinked", action="store_true")
    convergence.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the weighted run-level progress bar.",
    )
    return parser


def _write_status(directory: Path, frame: pd.DataFrame, *, name: str) -> bool:
    """Write one result table and compact machine-readable status."""

    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{name}.csv", index=False)
    passed = bool(frame["Passed"].all()) if "Passed" in frame else True
    payload = {
        "rows": int(len(frame)),
        "passed": passed,
        "failed_rows": int((~frame["Passed"]).sum()) if "Passed" in frame else 0,
    }
    (directory / f"{name}_status.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return passed


def _report_plot_files(files: dict[str, Path]) -> None:
    """Print generated plots so command-line users can find them immediately."""

    for path in files.values():
        print(f"Plot: {path}")


@contextmanager
def _progress_heartbeat(
    progress: Any,
    *,
    postfix: dict[str, object],
    enabled: bool,
) -> Iterator[None]:
    """Refresh one long-running progress item without inventing partial work."""

    if not enabled:
        yield
        return

    stopped = Event()
    started = perf_counter()

    def refresh() -> None:
        while not stopped.wait(CONVERGENCE_HEARTBEAT_SECONDS):
            progress.set_postfix(
                **postfix,
                run_elapsed=f"{perf_counter() - started:.0f}s",
                refresh=True,
            )

    thread = Thread(target=refresh, name="ld-convergence-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _exact(args: argparse.Namespace) -> bool:
    output = args.output / "exact"
    results = validate_exact_expectations(
        sample_sizes=args.sample_sizes, atol=args.atol
    )
    passed = _write_status(output, results, name="exact_expectations")
    plot_files = plot_exact_expectation_errors(
        results,
        output / "plots",
        formats=args.plot_formats,
        dpi=args.plot_dpi,
    )
    print(results.groupby("Statistic")["Absolute_Error"].max().to_string())
    _report_plot_files(plot_files)
    return passed


def _golden(args: argparse.Namespace) -> bool:
    output = args.output / "golden_reference"
    results = validate_golden_reference(args.fixture, rtol=args.rtol, atol=args.atol)
    passed = _write_status(output, results, name="golden_reference_comparison")
    plot_files = plot_golden_reference_errors(
        results,
        output / "plots",
        formats=args.plot_formats,
        dpi=args.plot_dpi,
    )
    print(results.groupby("Statistic")["Absolute_Error"].max().to_string())
    _report_plot_files(plot_files)
    return passed


def _simulate(args: argparse.Namespace) -> bool:
    if args.quick:
        population_sizes = (10, 25)
        sample_sizes = (6, 8)
        replicates = 3
        chromosomes = 4
        loci_per_chromosome = 25
        n_bootstraps = min(args.n_bootstraps, 20)
    else:
        population_sizes = tuple(args.population_sizes)
        sample_sizes = tuple(args.sample_sizes)
        replicates = args.replicates
        chromosomes = args.chromosomes
        loci_per_chromosome = args.loci_per_chromosome
        n_bootstraps = args.n_bootstraps
    config = Fwdpy11CalibrationConfig(
        population_sizes=population_sizes,
        sample_sizes=sample_sizes,
        replicates=replicates,
        chromosomes=chromosomes,
        loci_per_chromosome=loci_per_chromosome,
        burnin_multiplier=args.burnin_multiplier,
        allow_residual_selfing=args.allow_residual_selfing,
        n_bootstraps=n_bootstraps,
        minimum_model_population_size=args.minimum_model_population_size,
        minimum_coverage_sample_size=args.minimum_coverage_sample_size,
        model_relative_bias_tolerance=args.model_relative_bias_tolerance,
        require_population_coverage=args.require_population_coverage,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    _, summary = run_fwdpy11_calibration(
        config, output_directory=args.output / "simulation"
    )
    plot_files = plot_simulation_calibration(
        summary,
        args.output / "simulation" / "plots",
        formats=args.plot_formats,
        dpi=args.plot_dpi,
    )
    print(summary.to_string(index=False))
    _report_plot_files(plot_files)
    return bool(summary["Passed_Calibration"].all()) if replicates >= 10 else True


def _published(args: argparse.Namespace) -> bool:
    if not args.genepop.is_file():
        raise FileNotFoundError(args.genepop)
    output = args.output / "published_island_fox"
    output.mkdir(parents=True, exist_ok=True)
    normalized_genepop, popmap = prepare_island_fox_genepop(args.genepop, output)
    reader = GenePopReader(
        filename=str(normalized_genepop),
        popmapfile=str(popmap),
        force_popmap=True,
        prefix=str(output / "island_fox"),
        verbose=False,
    )
    result = PopGenStatistics(reader).calculate_linkage_disequilibrium(
        assume_unlinked=True,
        n_bootstraps=200,
        n_bootstrap_blocks=20,
        n_jobs=args.n_jobs,
        max_pairs=None,
        pairwise_sample_size=0,
        alpha=0.10,
        seed=args.seed,
        save_pairwise=False,
        save_plots=False,
    )
    population_names = dict(
        zip(
            ("SMI", "SRI", "SCI", "SCA", "SCL", "SNI"),
            PUBLISHED_ISLAND_FOX_ESTIMATES["Population"],
        )
    )
    result.summary["Population"] = result.summary["Population"].replace(
        population_names
    )
    comparison = compare_published_estimates(
        result.summary, relative_tolerance=args.relative_tolerance
    )
    result.summary.to_csv(output / "snpio_island_fox_summary.csv", index=False)
    passed = _write_status(output, comparison, name="published_comparison")
    plot_files = plot_published_island_fox_comparison(
        comparison,
        output / "plots",
        formats=args.plot_formats,
        dpi=args.plot_dpi,
    )
    print(comparison.to_string(index=False))
    _report_plot_files(plot_files)
    return passed


def _convergence(args: argparse.Namespace) -> bool:
    if not args.vcf.is_file():
        raise FileNotFoundError(args.vcf)
    if not args.popmap.is_file():
        raise FileNotFoundError(args.popmap)
    output = args.output / "convergence"
    output.mkdir(parents=True, exist_ok=True)
    reader = VCFReader(
        filename=str(args.vcf),
        popmapfile=str(args.popmap),
        force_popmap=True,
        disable_progress_bar=True,
        prefix=str(output / "input"),
        verbose=False,
    )
    statistics = PopGenStatistics(reader)
    rows: list[dict[str, Any]] = []
    tasks = [
        (pair_budget, seed) for pair_budget in args.pair_budgets for seed in args.seeds
    ]
    total_budget = sum(pair_budget for pair_budget, _ in tasks)
    with tqdm(
        total=total_budget,
        desc="Pair convergence",
        unit="budgeted pair",
        unit_scale=True,
        dynamic_ncols=True,
        disable=args.no_progress,
    ) as progress:
        for run_number, (pair_budget, seed) in enumerate(tasks, start=1):
            postfix: dict[str, object] = {
                "run": f"{run_number}/{len(tasks)}",
                "budget": f"{pair_budget:,}",
                "seed": seed,
            }
            progress.set_postfix(**postfix, run_elapsed="0s", refresh=True)
            reader.prefix = str(
                output / "runs" / f"pairs_{pair_budget}" / f"seed_{seed}"
            )
            started = perf_counter()
            with _progress_heartbeat(
                progress,
                postfix=postfix,
                enabled=not args.no_progress,
            ):
                result = statistics.calculate_linkage_disequilibrium(
                    assume_unlinked=args.assume_unlinked,
                    n_bootstraps=args.n_bootstraps,
                    n_bootstrap_blocks=20,
                    n_jobs=args.n_jobs,
                    max_pairs=pair_budget,
                    pairwise_sample_size=0,
                    seed=seed,
                    save_pairwise=False,
                    save_plots=False,
                )
            for summary_row in result.summary.to_dict(orient="records"):
                rows.append(
                    {
                        **summary_row,
                        "max_pairs": pair_budget,
                        "seed": seed,
                        "group_source": result.metadata["group_source"],
                    }
                )
            progress.update(pair_budget)
            if not args.no_progress:
                progress.write(
                    "Completed convergence run "
                    f"{run_number}/{len(tasks)}: budget={pair_budget:,}, "
                    f"seed={seed}, populations={len(result.summary)}, "
                    f"elapsed={perf_counter() - started:.1f}s"
                )
    replicates = pd.DataFrame(rows)
    summary = summarize_convergence(replicates)
    replicates.to_csv(output / "convergence_runs.csv", index=False)
    summary.to_csv(output / "convergence_summary.csv", index=False)
    plot_files = plot_pair_convergence(
        summary,
        output / "plots",
        formats=args.plot_formats,
        dpi=args.plot_dpi,
    )
    print(summary.to_string(index=False))
    _report_plot_files(plot_files)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output = args.output.resolve()
    commands = {
        "exact": _exact,
        "golden": _golden,
        "simulate": _simulate,
        "published": _published,
        "convergence": _convergence,
    }
    passed = commands[args.command](args)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
