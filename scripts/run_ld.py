#!/usr/bin/env python3

import argparse
from pathlib import Path

from snpio import PopGenStatistics, SNPioMultiQC, VCFReader


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SNPio",
        description="Run SNPio LD analysis with specified input, popmap, and output prefix.",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input VCF file."
    )
    parser.add_argument(
        "--popmap",
        type=str,
        required=False,
        default=None,
        help="Path to population map file mapping samples to populations. Format: <sample>\t<population>",
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Prefix for output files."
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=100,
        help="Number of bootstrap replicates for LD calculations (default: 100).",
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        default=False,
        help="Include overall LD calculations.",
    )
    parser.add_argument(
        "--assume-unlinked",
        action="store_true",
        default=False,
        help="Assume loci are unlinked.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for LD calculations (default: 1). Use -1 for all available cores.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Format for plots (default: png).",
    )

    return parser.parse_args()


def main(args):
    # Validate input files
    input_path = Path(args.input)

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    popmap_path = Path(args.popmap) if args.popmap else None

    if popmap_path and (not popmap_path.exists() or not popmap_path.is_file()):
        raise FileNotFoundError(f"Popmap file not found: {args.popmap}")

    # Read VCF file
    vcf_reader = VCFReader(
        str(input_path),
        popmapfile=str(popmap_path) if popmap_path else None,
        chunk_size=5000,
        force_popmap=True,
        verbose=args.verbose,
        debug=args.debug,
        plot_format=args.plot_format,
        prefix=args.prefix,
    )

    pgs = PopGenStatistics(vcf_reader, verbose=args.verbose, debug=args.debug)

    ldr = pgs.calculate_linkage_disequilibrium(
        include_overall=args.include_overall,
        assume_unlinked=args.assume_unlinked,
        n_jobs=args.n_jobs,
        n_bootstraps=args.n_boot,
        seed=args.random_seed,
    )

    # Generate multiQC report
    SNPioMultiQC.build(
        prefix=args.prefix, title="SNPio LD Analysis Report", overwrite=True
    )


if __name__ == "__main__":
    main(parse_args())
