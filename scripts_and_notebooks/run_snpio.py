#!/usr/bin/env python3

import argparse
from pathlib import Path

from snpio import NRemover2, PopGenStatistics, SNPioMultiQC, VCFReader

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
        --plot-format <png|pdf|svg>
"""


def version():
    from snpio import __version__

    return str(__version__)


def validate_file(path: str, name: str) -> None:
    pth = Path(path)
    if not pth.exists() or not pth.is_file():
        print(f"ERROR: {name} file not found at: {path}")
        raise FileNotFoundError(f"{name} file not found: {path}")


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
    print(f"  📁 Output prefix:  {args.prefix}")
    print(f"  🖼️ Plot format:     {args.plot_format}")
    print(f"  🔍 Verbose:         {args.verbose}")
    print(f"  🐛 Debug:           {args.debug}")
    print()

    genotype_data = VCFReader(
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
        nrm.filter_biallelic(exclude_heterozygous=True)
        .filter_missing(0.75)
        .filter_missing_pop(0.75)
        .filter_singletons(exclude_heterozygous=True)
        .filter_missing_sample(0.8)
        .resolve()
    )

    nrm.plot_sankey_filtering_report()
    gd_filt.missingness_reports(gd_filt.prefix)

    pgs = PopGenStatistics(gd_filt, verbose=args.verbose, debug=args.debug)

    allele_summary_stats, summary_stats = pgs.summary_statistics(
        n_permutations=100, n_jobs=8, use_pvalues=True
    )

    neis_dist = pgs.neis_genetic_distance(
        n_permutations=1000, n_jobs=8, use_pvalues=True
    )

    fst_permutation = pgs.detect_fst_outliers(
        n_permutations=100,
        correction_method="fdr_bh",
        use_dbscan=False,
        n_jobs=8,
        min_samples=5,
    )

    fst_dbscan = pgs.detect_fst_outliers(
        n_permutations=1000,
        correction_method="fdr_bh",
        use_dbscan=True,
        n_jobs=8,
        min_samples=5,
    )

    dstats = pgs.calculate_d_statistics(
        method="patterson",
        population1="EA",
        population2="GU",
        population3="TT",
        outgroup="ON",
        num_bootstraps=1000,
        individual_selection="random",
        max_individuals_per_pop=3,
        seed=42,
    )

    dstats_partitioned = pgs.calculate_d_statistics(
        method="partitioned",
        population1="EA",
        population2="GU",
        population3="TT",
        population4="ON",
        outgroup="OG",
        num_bootstraps=1000,
        individual_selection="random",
        max_individuals_per_pop=3,
        seed=42,
    )

    dstats_dfoil = pgs.calculate_d_statistics(
        method="dfoil",
        population1="EA",
        population2="GU",
        population3="TT",
        population4="ON",
        outgroup="OG",
        num_bootstraps=1000,
        individual_selection="random",
        max_individuals_per_pop=3,
        seed=42,
    )

    # Run PCA
    pgs.pca()

    # Build MultiQC report
    print("📊 Building MultiQC report...")
    SNPioMultiQC.build(
        prefix="Example Report",
        output_dir=f"{args.prefix}_output/multiqc",
        overwrite=True,
    )


if __name__ == "__main__":
    main()
