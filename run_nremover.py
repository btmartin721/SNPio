# nremover.py
import argparse
from popgenstats.read_files import popgenio as pgio
from popgenstats.filtering import nremover
from popgenstats.pop_gen_statistics import PopGenStatistics
from popgenstats.plotting import Plotting


def main(args):
    # Read the alignment and popmap files
    pg = pgio.PopGenIO(args.alignment, args.popmap)
    pg.read_files()

    # Run the NRemover class
    nrm = nremover.NRemover(pg)
    pg = nrm.nremover(
        max_missing_global=args.max_missing_global,
        max_missing_pop=args.max_missing_pop,
        max_missing_sample=args.max_missing_sample,
        singletons=args.singletons,
        biallelic=args.biallelic,
        monomorphic=args.monomorphic,
        min_maf=args.min_maf,
        plot_missingness_report=args.plot_missingness_report,
        plot_dir=args.plot_dir,
    )

    pgs = PopGenStatistics(pg)
    Plotting.plot_sfs(pgs, "ON", "EA")


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the NRemover class on the given input files."
    )

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "--alignment", required=True, help="Path to the alignment file."
    )
    required_args.add_argument(
        "--popmap", required=True, help="Path to the popmap file."
    )

    parser.add_argument(
        "--max_missing_global",
        type=float,
        default=1.0,
        help="Maximum global missing rate.",
    )
    parser.add_argument(
        "--max_missing_pop",
        type=float,
        default=1.0,
        help="Maximum population missing rate.",
    )
    parser.add_argument(
        "--max_missing_sample",
        type=float,
        default=1.0,
        help="Maximum sample missing rate.",
    )
    parser.add_argument(
        "--singletons", action="store_true", help="Filter singletons."
    )
    parser.add_argument(
        "--biallelic", action="store_true", help="Filter biallelic sites."
    )
    parser.add_argument(
        "--monomorphic", action="store_true", help="Filter monomorphic sites."
    )
    parser.add_argument(
        "--min_maf",
        type=float,
        default=0.0,
        help="Minimum minor allele frequency.",
    )
    parser.add_argument(
        "--plot_missingness_report",
        action="store_true",
        help="Plot a missingness report.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots",
        help="Directory to save plots to.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
