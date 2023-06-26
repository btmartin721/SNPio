import argparse
from snpio.read_input.genotype_data import GenotypeData
from snpio.filtering import nremover2
from snpio.plotting.plotting import Plotting


def main():
    # Read the alignment, popmap, and tree files
    gd = GenotypeData(
        filename="example_data/phylip_files/phylogen_nomx.u.snps.phy",
        popmapfile="example_data/popmaps/phylogen_nomx.popmap",
        force_popmap=True,
        filetype="auto",
        qmatrix_iqtree="example_data/trees/test.qmat",
        siterates_iqtree="example_data/trees/test.rate",
        guidetree="example_data/trees/test.tre",
        exclude_pops=["DS", "ON"],
    )

    # Make missingness report plots.
    gd.missingness_reports(prefix="unfiltered")

    # Run a PCA and make a scatterplot on the unfiltered data.
    Plotting.run_pca(gd, prefix="unfiltered", plot_format="png")

    # Run the NRemover class to filter out missing data.
    nrm = nremover2.NRemover2(gd)
    gd_filtered = nrm.nremover(
        max_missing_global=0.5,
        max_missing_pop=0.5,
        max_missing_sample=0.8,
        singletons=True,
        biallelic=True,
        monomorphic=True,
        min_maf=0.01,
        search_thresholds=False,
        plot_dir="plots",
    )

    # Make another missingness report plot for filtered data.
    gd_filtered.missingness_reports(prefix="filtered")

    # Run a PCA on the filtered data and make a scatterplot.
    Plotting.run_pca(gd_filtered, prefix="filtered", plot_format="png")

    gd_filtered.write_vcf("example_data/vcf_files/nremover_test.vcf")
    gd_filtered.write_phylip("example_data/phylip_files/nremover_test.phy")
    gd_filtered.write_structure(
        "example_data/structure_files/nremover_test.str"
    )

    print(gd_filtered.alignment)
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
    main()
