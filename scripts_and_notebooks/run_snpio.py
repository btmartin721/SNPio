#!/usr/bin/env python

"""
run_snpio.py

A helper script to run SNPio programmatically from within Docker or CLI.

Usage:
    python run_snpio.py \
        --input /app/data/0_original_alignments/example.vcf \
        --popmap /app/data/1_popmaps/example_popmap.txt \
        --prefix /app/results/snpio
"""

import argparse
import os
import sys

from snpio import NRemover2, Plotting, PopGenStatistics, VCFReader


def validate_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        print(f"ERROR: {name} file not found at: {path}", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SNPio with specified input, popmap, and output prefix."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (VCF, PHYLIP, or STRUCTURE format)",
    )
    parser.add_argument(
        "--popmap",
        type=str,
        required=True,
        help="Path to popmap file mapping samples to populations",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Output prefix for results (output files will be saved as <prefix>_output/*)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for output plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate paths
    validate_file(args.input, "Input")
    validate_file(args.popmap, "Popmap")

    print(f"🧬 Running SNPio with:")
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
        exclude_pops=["OG", "DS"],
        prefix=args.prefix,
        plot_format=args.plot_format,
        verbose=args.verbose,
        debug=args.debug,
    )

    nrm = NRemover2(genotype_data)

    gd_filt = (
        nrm.filter_missing_sample(0.8)
        .filter_missing(0.5)
        .filter_missing_pop(0.5)
        .filter_maf(0.01)
        .filter_biallelic(exclude_heterozygous=True)
        .filter_monomorphic(exclude_heterozygous=True)
        .filter_singletons(exclude_heterozygous=True)
        .thin_loci(100)
        .resolve()
    )

    nrm.plot_sankey_filtering_report()
    genotype_data.missingness_reports(prefix=args.prefix)

    pgs = PopGenStatistics(gd_filt, verbose=args.verbose, debug=args.debug)

    summary_stats = pgs.summary_statistics(
        n_permutations=10, n_jobs=1, use_pvalues=True
    )

    neis_dist = pgs.neis_genetic_distance(n_permutations=10, n_jobs=1, use_pvalues=True)

    gd_filt.write_vcf(f"results/{args.prefix}_output/filtered.vcf")

    plotter = Plotting(
        genotype_data,
        plot_format=args.plot_format,
        verbose=args.verbose,
        debug=args.debug,
    )

    plotter.run_pca()


if __name__ == "__main__":
    main()
