#!/usr/bin/env python3

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from pprint import pprint

from snpio import NRemover2, PopGenStatistics, SNPioMultiQC, VCFReader
from snpio.utils.output_paths import OutputPaths


@dataclass
class Arguments:
    input_files: dict[str, dict] = field(
        default_factory=lambda: {
            "input": {
                "type": str,
                "help": "Path to the VCF input used by the bundled full workflow.",
            },
            "popmap": {
                "type": str,
                "help": "Path to the sample-to-population map required by the bundled full workflow. Format: <sample>\\t<population>",
            },
            "prefix": {
                "type": str,
                "help": "Output prefix for results (output files will be saved as <prefix>_output/*)",
            },
            "include_pops": {
                "type": list[str],
                "help": "Optional list of populations to include in the analysis. If not specified, all populations in the popmap will be included.",
            },
        }
    )
    filtering: dict[str, dict] = field(
        default_factory=lambda: {
            "sample_missing_threshold": {
                "type": float,
                "help": "Threshold for filtering samples based on missing data (default: 0.8). Samples with missing data above this threshold will be removed.",
            },
            "locus_missing_threshold": {
                "type": float,
                "help": "Threshold for filtering loci based on missing data (default: 0.75). Loci with missing data above this threshold will be removed.",
            },
            "locus_missing_pop_threshold": {
                "type": float,
                "help": "Threshold for filtering loci based on missing data within populations (default: 0.75). Loci with missing data above this threshold in any population will be removed.",
            },
            "maf_threshold": {
                "type": float,
                "help": "Threshold for filtering loci based on minor allele frequency (MAF) (default: 0.01). Loci with MAF below this threshold will be removed.",
            },
            "mac_threshold": {
                "type": int,
                "help": "Threshold for filtering loci based on minor allele count (MAC) (default: 2). Loci with MAC below this threshold will be removed.",
            },
            "exclude_heterozygous": {
                "type": bool,
                "help": "Exclude heterozygous loci from filtering and analysis (default: False).",
            },
        }
    )
    threading: dict[str, dict] = field(
        default_factory=lambda: {
            "n_jobs": {
                "type": int,
                "help": "Number of parallel jobs to run for computations that support multi-threading. Use -1 for all available cores (default: 1).",
            },
        }
    )
    plotting: dict[str, dict] = field(
        default_factory=lambda: {
            "plot_format": {
                "type": str,
                "help": "Format for output plots. Options: png, pdf, svg (default: png)",
            },
        }
    )
    bootstrap: dict[str, dict] = field(
        default_factory=lambda: {
            "n_boot_fst": {
                "type": int,
                "help": "Number of bootstrap replicates for FST and D-statistics calculations (default: 1000).",
            },
            "n_perm_fst": {
                "type": int,
                "help": "Number of permutation replicates for FST outlier detection (default: 1000).",
            },
            "n_boot_neis": {
                "type": int,
                "help": "Number of bootstrap replicates for Nei's genetic distance calculations (default: 1000).",
            },
            "n_perm_neis": {
                "type": int,
                "help": "Number of permutation replicates for Nei's genetic distance calculations (default: 1000).",
            },
            "n_boot_dstats": {
                "type": int,
                "help": "Number of bootstrap replicates for D-statistics calculations (default: 1000).",
            },
            "n_boot_ld": {
                "type": int,
                "help": "Number of bootstrap replicates for LD estimates (default: 1000).",
            },
        }
    )
    ld: dict[str, dict] = field(
        default_factory=lambda: {
            "include_overall": {
                "type": bool,
                "help": "Include overall LD estimates across all populations in addition to population-specific estimates. If not toggled, only population-specific LD estimates are calculated (default: False).",
            },
            "assume_unlinked": {
                "type": bool,
                "help": "Explicitly assert that every supplied locus is unlinked. Use only for independently pruned data without usable linkage-group labels (default: False).",
            },
        }
    )
    dtest: dict[str, dict] = field(
        default_factory=lambda: {
            "population1": {
                "type": str,
                "help": "Population 1 for D-statistics calculations (default: EA).",
            },
            "population2": {
                "type": str,
                "help": "Population 2 for D-statistics calculations (default: GU).",
            },
            "population3": {
                "type": str,
                "help": "Population 3 for D-statistics calculations (default: TT).",
            },
            "population4": {
                "type": str,
                "help": "Population 4 for partitioned D-statistics calculations (default: ON).",
            },
            "outgroup": {
                "type": str,
                "help": "Outgroup population for D-statistics calculations (default: OG).",
            },
            "individual_selection": {
                "type": str,
                "help": "Method for selecting individuals for D-statistics calculations. Options: random (random subset), least_missing (samples with the fewest unusable genotypes), all (ignores the per-population cap) (default: random).",
            },
        }
    )
    advanced: dict[str, dict] = field(
        default_factory=lambda: {
            "random_seed": {
                "type": int,
                "help": "Random seed for reproducibility of results. If not set, results will be non-deterministic (default: None).",
            },
            "overwrite_multiqc": {
                "type": bool,
                "help": "Overwrite existing MultiQC report if it exists. Use with caution as this will replace any existing report in the output directory.",
            },
            "pvalue_correction_method": {
                "type": str,
                "help": "Method for p-value correction in multiple testing scenarios. Options: bonferroni, fdr_bh, holm, hochberg, hommel, fdr_tsbh (default: fdr_bh).",
            },
            "force_popmap": {
                "type": bool,
                "help": "Force the use of the provided popmap file even if it does not match the samples in the input file. Use with caution.",
            },
            "chunk_size": {
                "type": int,
                "help": "Chunk size for processing large VCF files. Adjust based on available memory (default: 5000).",
            },
        }
    )
    verbosity: dict[str, dict] = field(
        default_factory=lambda: {
            "version": {
                "type": bool,
                "help": "Show the version of SNPio and exit.",
            },
            "verbose": {
                "type": bool,
                "help": "Enable verbose logging. Includes additional logging information during processing.",
            },
            "debug": {
                "type": bool,
                "help": "Enable debug mode. Includes additional logging and checks. This may slow down processing.",
            },
        }
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent=1) -> str:
        return json.dumps(asdict(self), indent=indent)


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
        --plot-format <png|pdf|svg>
"""


def seconds_to_hms(seconds):
    """Convert seconds to hours, minutes, and seconds.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted string in "HH:MM:SS" format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def timer(func):
    """Decorator to measure the execution time of a function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function with execution time measurement.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("")
        print(f"Executed SNPio workflow in: {seconds_to_hms(elapsed_time)}")
        print("")
        return result

    return wrapper


def version():
    from snpio import __version__

    return str(__version__)


def validate_file(path: str, name: str) -> None:
    pth = Path(path)
    if not pth.exists() or not pth.is_file():
        print(f"ERROR: {name} file not found at: {path}")
        raise FileNotFoundError(f"{name} file not found: {path}")


def _run_fst_outlier_detection(
    pgs: PopGenStatistics,
    args: argparse.Namespace,
):
    """Run the single Fst outlier method selected by the CLI."""

    return pgs.detect_fst_outliers(
        n_permutations=args.n_perm_fst,
        correction_method=args.pvalue_correction_method,
        use_dbscan=args.use_dbscan,
        n_jobs=args.n_jobs,
        min_samples=args.min_samples_dbscan,
        seed=args.random_seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SNPio",
        description="Run SNPio's bundled full VCF workflow with a population map and output prefix. The workflow filters the data, runs summary statistics, FST and Nei distances, D-statistics, PCA, unbiased LD and LD-based Ne, and then builds a MultiQC report.",
    )
    input_group = parser.add_argument_group(
        title="Input Options",
        description="Specify the VCF input and population map for the bundled full workflow. Use the Python API for PHYLIP, STRUCTURE, or GENEPOP inputs. The output prefix determines the artifact root.",
    )
    input_group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the VCF input used by the bundled full workflow.",
    )
    input_group.add_argument(
        "--popmap",
        type=str,
        required=False,
        default=None,
        help="Path to the sample-to-population map required by the bundled full workflow. Format: <sample>\t<population>.",
    )
    input_group.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Output prefix for results (output files will be saved as <prefix>_output/*)",
    )
    input_group.add_argument(
        "--include-pops",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of populations to include in the analysis. If not specified, all populations in the popmap will be included. Specify populations as space-separated values, e.g., --include-pops pop1 pop2 pop3",
    )
    filter_group = parser.add_argument_group(
        title="Filtering Options",
        description="Options for filtering SNP data based on missingness, MAF, MAC, and other criteria.",
    )
    filter_group.add_argument(
        "--sample-missing-threshold",
        type=float,
        default=0.8,
        help="Threshold for filtering samples based on missing data (default: 0.8). Samples with missing data above this threshold will be removed.",
    )
    filter_group.add_argument(
        "--locus-missing-threshold",
        type=float,
        default=0.75,
        help="Threshold for filtering loci based on missing data (default: 0.75). Loci with missing data above this threshold will be removed.",
    )
    filter_group.add_argument(
        "--locus-missing-pop-threshold",
        type=float,
        default=0.75,
        help="Threshold for filtering loci based on missing data within populations (default: 0.75). Loci with missing data above this threshold in any population will be removed.",
    )
    filter_group.add_argument(
        "--maf-threshold",
        type=float,
        default=0.01,
        help="Threshold for filtering loci based on minor allele frequency (MAF) (default: 0.01). Loci with MAF below this threshold will be removed.",
    )
    filter_group.add_argument(
        "--mac-threshold",
        type=int,
        default=2,
        help="Threshold for filtering loci based on minor allele count (MAC) (default: 2). Loci with MAC below this threshold will be removed.",
    )
    filter_group.add_argument(
        "--exclude-heterozygous",
        action="store_true",
        default=False,
        help="Exclude heterozygous loci from filtering and analysis (default: False).",
    )
    thread_group = parser.add_argument_group(
        title="Threading Options",
        description="Control the number of parallel jobs for computations that support multi-threading.",
    )
    thread_group.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run for computations that support multi-threading. Use -1 for all available cores (default: 1).",
    )
    plot_group = parser.add_argument_group(
        title="Plotting Options",
        description="Options for controlling the format of output plots generated by SNPio.",
    )
    plot_group.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for output plots. Options: png, pdf, svg (default: png)",
    )
    fst_group = parser.add_argument_group(
        title="FST and Nei's Options",
        description="Options for calculating FST distances and detecting outliers. Also includes options for Nei's genetic distance calculations.",
    )
    fst_group.add_argument(
        "--use-dbscan",
        action="store_true",
        default=False,
        help="Use DBSCAN clustering for FST outlier detection (default: False). If not toggled, standard permutation-based outlier detection is used.",
    )
    fst_group.add_argument(
        "--min-samples-dbscan",
        type=int,
        default=5,
        help="Minimum number of samples for DBSCAN clustering in FST outlier detection (default: 5).",
    )
    boot_group = parser.add_argument_group(
        title="Bootstrap Options",
        description="Options for controlling the number of bootstrap replicates for various analyses.",
    )
    boot_group.add_argument(
        "--n-boot-fst",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for FST and D-statistics calculations (default: 1000).",
    )
    boot_group.add_argument(
        "--n-perm-fst",
        type=int,
        default=1000,
        help="Number of permutation replicates for FST outlier detection (default: 1000).",
    )
    boot_group.add_argument(
        "--n-boot-neis",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for Nei's genetic distance calculations (default: 1000).",
    )
    boot_group.add_argument(
        "--n-perm-neis",
        type=int,
        default=1000,
        help="Number of permutation replicates for Nei's genetic distance calculations (default: 1000).",
    )
    boot_group.add_argument(
        "--n-boot-dstats",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for D-statistics calculations (default: 1000).",
    )
    boot_group.add_argument(
        "--n-boot-ld",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for LD estimates (default: 1000).",
    )
    ld_group = parser.add_argument_group(
        title="Linkage Disequilibrium Options",
        description="Options for finite-sample-unbiased linkage disequilibrium (LD) and LD-based recent effective population size (Ne) following Ragsdale and Gravel (2020).",
    )
    ld_group.add_argument(
        "--include-overall",
        action="store_true",
        help="Include overall LD estimates across all populations in addition to population-specific estimates. If not toggled, only population-specific LD estimates are calculated (default: False).",
    )
    ld_group.add_argument(
        "--assume-unlinked",
        action="store_true",
        help="Explicitly assert that every supplied locus is unlinked. Use only for independently pruned data without usable linkage-group labels (default: False).",
    )
    dtest_group = parser.add_argument_group(
        title="D-statistics Options",
        description="Options for calculating D-statistics (ABBA-BABA tests) for detecting introgression and gene flow.",
    )
    dtest_group.add_argument(
        "--population1",
        type=str,
        default="EA",
        help="Population 1 for D-statistics calculations (default: EA).",
    )
    dtest_group.add_argument(
        "--population2",
        type=str,
        default="GU",
        help="Population 2 for D-statistics calculations (default: GU).",
    )
    dtest_group.add_argument(
        "--population3",
        type=str,
        default="TT",
        help="Population 3 for D-statistics calculations (default: TT).",
    )
    dtest_group.add_argument(
        "--population4",
        type=str,
        default="ON",
        help="Population 4 for partitioned D-statistics calculations (default: ON).",
    )
    dtest_group.add_argument(
        "--outgroup",
        type=str,
        default="OG",
        help="Outgroup population for D-statistics calculations (default: OG).",
    )
    dtest_group.add_argument(
        "--individual-selection",
        type=str,
        default="random",
        choices=["random", "least_missing", "all"],
        help="Method for selecting individuals for D-statistics calculations. Options: random (random subset), least_missing (samples with the fewest unusable genotypes), all (ignores --max-individuals-per-pop) (default: random).",
    )
    dtest_group.add_argument(
        "--max-individuals-per-pop",
        type=int,
        default=5,
        help="Maximum number of individuals to include per population for D-statistics calculations (default: 5).",
    )
    advanced_group = parser.add_argument_group(
        title="Advanced Options",
        description="Advanced options for controlling SNPio behavior, including random seed for reproducibility.",
    )
    advanced_group.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility of results. If not set, results will be non-deterministic (default: None).",
    )
    advanced_group.add_argument(
        "--overwrite-multiqc",
        action="store_true",
        default=False,
        help="Overwrite existing MultiQC report if it exists. Use with caution as this will replace any existing report in the output directory.",
    )
    advanced_group.add_argument(
        "--pvalue-correction-method",
        type=str,
        default="fdr_bh",
        choices=["bonferroni", "fdr_bh", "holm", "hochberg", "hommel", "fdr_tsbh"],
        help="Method for p-value correction in multiple testing scenarios. Options: bonferroni, fdr_bh, holm, hochberg, hommel, fdr_tsbh (default: fdr_bh).",
    )
    advanced_group.add_argument(
        "--force-popmap",
        action="store_true",
        default=False,
        help="Force the use of the provided popmap file even if it does not match the samples in the input file. Use with caution.",
    )
    advanced_group.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Chunk size for processing large VCF files. Adjust based on available memory (default: 5000).",
    )
    verbosity_group = parser.add_argument_group(
        title="Verbosity Options",
        description="Control the verbosity of logging and output during SNPio processing.",
    )
    verbosity_group.add_argument(
        "--version",
        default=False,
        action="store_true",
        help="Show the version of SNPio and exit.",
    )
    verbosity_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging. Includes additional logging information during processing.",
    )
    verbosity_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Includes additional logging and checks. This may slow down processing.",
    )

    args = parser.parse_args()

    if args.version:
        print(f"SNPio version {version()}")
        sys.exit(0)

    return args


@timer
def main():
    """Main function to run SNPio with specified arguments."""

    args = parse_args()

    # Initialize and validate arguments
    args_dc = initialize_args(args)
    export_args_to_json(args, args_dc)

    # Validate paths
    validate_file(args.input, "Input")
    validate_file(args.popmap, "Popmap")

    args_dict = args_dc.to_dict()
    pprint(args_dict, indent=1)

    genotype_data = VCFReader(
        filename=args.input,
        popmapfile=args.popmap,
        force_popmap=args.force_popmap,
        chunk_size=args.chunk_size,
        include_pops=args.include_pops,
        prefix=args.prefix,
        plot_format=args.plot_format,
        verbose=args.verbose,
        debug=args.debug,
    )

    # Generate missingness reports before filtering
    genotype_data.missingness_reports(prefix=args.prefix)

    nrm = NRemover2(genotype_data)

    nrm.search_thresholds(
        thresholds=[0.25, 0.5, 0.75],
        maf_thresholds=[0.01, 0.05],
        mac_thresholds=[2, 3],
        filter_order=[
            "filter_missing_sample",
            "filter_missing",
            "filter_missing_pop",
            "filter_monomorphic",
            "filter_singletons",
            "filter_biallelic",
            "filter_mac",
            "filter_maf",
        ],
    )

    gd_filt = (
        nrm.filter_biallelic(exclude_heterozygous=args.exclude_heterozygous)
        .filter_missing(args.locus_missing_threshold)
        .filter_missing_pop(args.locus_missing_threshold)
        .filter_singletons(exclude_heterozygous=args.exclude_heterozygous)
        .filter_missing_sample(args.sample_missing_threshold)
        .resolve()
    )

    nrm.plot_sankey_filtering_report()
    gd_filt.missingness_reports(gd_filt.prefix)

    pgs = PopGenStatistics(gd_filt, verbose=args.verbose, debug=args.debug)

    allele_summary_stats, summary_stats = pgs.summary_statistics(
        method="observed", n_reps=args.n_boot_fst, n_jobs=args.n_jobs
    )

    fst_dist = pgs.fst_distance(
        method="permutation",
        n_reps=args.n_perm_fst,
        n_jobs=args.n_jobs,
        palette="magma",
    )

    fst_dist = pgs.fst_distance(
        method="bootstrap", n_reps=args.n_boot_fst, n_jobs=args.n_jobs, palette="magma"
    )

    neis_dist_boot = pgs.neis_genetic_distance(
        method="bootstrap", n_reps=args.n_boot_neis, n_jobs=args.n_jobs
    )

    neis_dist_perm = pgs.neis_genetic_distance(
        method="permutation", n_reps=args.n_perm_neis, n_jobs=args.n_jobs
    )

    _run_fst_outlier_detection(pgs, args)

    dstats = pgs.calculate_d_statistics(
        method="patterson",
        population1=args.population1,
        population2=args.population2,
        population3=args.population3,
        outgroup=args.outgroup,
        num_bootstraps=args.n_boot_dstats,
        individual_selection=args.individual_selection,
        max_individuals_per_pop=args.max_individuals_per_pop,
        seed=args.random_seed,
    )

    dstats_partitioned = pgs.calculate_d_statistics(
        method="partitioned",
        population1=args.population1,
        population2=args.population2,
        population3=args.population3,
        population4=args.population4,
        outgroup=args.outgroup,
        num_bootstraps=args.n_boot_dstats,
        individual_selection=args.individual_selection,
        max_individuals_per_pop=args.max_individuals_per_pop,
        seed=args.random_seed,
    )

    dstats_dfoil = pgs.calculate_d_statistics(
        method="dfoil",
        population1=args.population1,
        population2=args.population2,
        population3=args.population3,
        population4=args.population4,
        outgroup=args.outgroup,
        num_bootstraps=args.n_boot_dstats,
        individual_selection=args.individual_selection,
        max_individuals_per_pop=args.max_individuals_per_pop,
        seed=args.random_seed,
    )

    # Run PCA
    pgs.pca()

    ldr = pgs.calculate_linkage_disequilibrium(
        include_overall=args.include_overall,
        assume_unlinked=args.assume_unlinked,
        n_jobs=args.n_jobs,
        n_bootstraps=args.n_boot_ld,
        seed=args.random_seed,
    )

    # Build MultiQC report
    print("📊 Building MultiQC report...")
    html_obj = SNPioMultiQC.build(prefix=args.prefix, overwrite=args.overwrite_multiqc)
    print(f"✅ MultiQC report generated at: {str(html_obj.resolve())}")


def export_args_to_json(args, args_dc):
    """Export the parsed arguments to a JSON file for record-keeping."""
    args_json = args_dc.to_json(indent=4)

    json_dir = OutputPaths(args.prefix).logs
    json_dir.mkdir(parents=True, exist_ok=True)
    json_out = json_dir / "arguments.json"

    if json_out.exists():
        print(
            f"WARNING: arguments.json already exists at {json_out}. It will be overwritten."
        )

    with open(json_out, "w") as f:
        f.write(args_json)


def initialize_args(args):
    """Initialize and validate arguments for SNPio."""
    return Arguments(
        input_files={
            "input": args.input,
            "popmap": args.popmap,
            "prefix": args.prefix,
            "include_pops": args.include_pops,
        },
        filtering={
            "sample_missing_threshold": args.sample_missing_threshold,
            "locus_missing_threshold": args.locus_missing_threshold,
            "locus_missing_pop_threshold": args.locus_missing_pop_threshold,
            "maf_threshold": args.maf_threshold,
            "mac_threshold": args.mac_threshold,
            "exclude_heterozygous": args.exclude_heterozygous,
        },
        threading={"n_jobs": args.n_jobs},
        plotting={"plot_format": args.plot_format},
        bootstrap={
            "n_boot_fst": args.n_boot_fst,
            "n_perm_fst": args.n_perm_fst,
            "n_boot_neis": args.n_boot_neis,
            "n_perm_neis": args.n_perm_neis,
            "n_boot_dstats": args.n_boot_dstats,
            "n_boot_ld": args.n_boot_ld,
        },
        ld={
            "include_overall": args.include_overall,
            "assume_unlinked": args.assume_unlinked,
        },
        dtest={
            "population1": args.population1,
            "population2": args.population2,
            "population3": args.population3,
            "population4": args.population4,
            "outgroup": args.outgroup,
            "individual_selection": args.individual_selection,
            "max_individuals_per_pop": args.max_individuals_per_pop,
        },
        advanced={
            "random_seed": args.random_seed,
            "overwrite_multiqc": args.overwrite_multiqc,
            "pvalue_correction_method": args.pvalue_correction_method,
            "force_popmap": args.force_popmap,
            "chunk_size": args.chunk_size,
        },
        verbosity={
            "version": args.version,
            "verbose": args.verbose,
            "debug": args.debug,
        },
    )


if __name__ == "__main__":
    main()
