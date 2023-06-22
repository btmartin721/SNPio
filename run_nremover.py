import argparse
from snpio.read_input.genotype_data import GenotypeData
from snpio.filtering import nremover2


def main(args):
    # Read the alignment and popmap files
    gd = GenotypeData(
        filename=args.alignment,
        popmapfile=args.popmap,
        force_popmap=True,
        filetype=args.filetype,
        qmatrix_iqtree=args.qmat_iqtree,
        siterates_iqtree=args.siterates_iqtree,
        guidetree=args.tree,
    )

    # Run the NRemover class
    nrm = nremover2.NRemover2(gd)
    gd_filtered = nrm.nremover(
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

    gd_filtered.write_vcf("example_data/vcf_files/nremover_test.vcf")
    gd_filtered.write_phylip("example_data/phylip_files/nremover_test.phy")
    gd_filtered.write_structure(
        "example_data/structure_files/nremover_test.str"
    )

    print(gd_filtered.tree)


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

    required_args.add_argument(
        "--filetype",
        required=True,
        help="Supported file types: 'phylip', 'structure1row', 'structure1rowPopID', 'structure2row', 'structure2rowPopID'",
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
    parser.add_argument(
        "--qmat", type=str, default=None, help="Path to Q-matrix file."
    )
    parser.add_argument(
        "--qmat_iqtree",
        type=str,
        default=None,
        help="Path to Q-matrix IQTREE file.",
    )
    parser.add_argument(
        "--siterates", type=str, default=None, help="Path to Site Rates file."
    )
    parser.add_argument(
        "--siterates_iqtree",
        type=str,
        default=None,
        help="Path to Site Rates IQTREE file.",
    )
    parser.add_argument(
        "--tree", type=str, default=None, help="Path to newick treefile."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
