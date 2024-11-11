import pprint

import pandas as pd

from snpio import (
    GenotypeEncoder,
    NRemover2,
    Plotting,
    PopGenStatistics,
    TreeParser,
    VCFReader,
)

# Uncomment the following line to import the Benchmark class.
# NOTE: For development purposes. Comment out for normal use.
# from snpio.utils.benchmarking import Benchmark


def main():
    # Read the alignment, popmap, and tree files.
    gd = VCFReader(
        filename="snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz",
        popmapfile="snpio/example_data/popmaps/phylogen_nomx.popmap",
        force_popmap=True,  # Remove samples not in the popmap, or vice versa.
        chunk_size=5000,
        exclude_pops=["OG"],
        plot_format="pdf",
    )

    pgs = PopGenStatistics(gd, verbose=True)

    summary_stats = pgs.summary_statistics(save_plots=True)

    df_fst_outliers, df_fst_outlier_pvalues = pgs.detect_fst_outliers(
        correction_method="bonf", use_bootstrap=False
    )

    amova_results = pgs.amova()

    print(summary_stats)
    print(amova_results)
    print(df_fst_outliers.head())

    dstats_df, overall_results = pgs.calculate_d_statistics(
        method="patterson",
        population1="EA",
        population2="GU",
        population3="TT",
        outgroup="ON",
        num_bootstraps=10,
        n_jobs=6,
        max_individuals_per_pop=3,
    )

    print(dstats_df.head())
    pprint.pprint(overall_results, indent=4)

    # Run PCA and make missingness report plots.
    plotting = Plotting(genotype_data=gd)
    gd_components, gd_pca = plotting.run_pca()
    gd.missingness_reports()

    nrm = NRemover2(gd)

    nrm.search_thresholds(
        thresholds=[0.25, 0.5, 0.75, 1.0],
        maf_thresholds=[0.0, 0.01, 0.025, 0.05],
        mac_thresholds=[2, 5],
    )

    # # Plot benchmarking results.
    # NOTE: For development purposes. Comment out for normal use.
    # Benchmark.plot_performance(nrm.genotype_data, nrm.genotype_data.resource_data)

    gd_filt = (
        nrm.filter_missing_sample(0.75)
        .filter_missing(0.75)
        .filter_missing_pop(0.75)
        .filter_mac(2)
        .filter_monomorphic(exclude_heterozygous=False)
        .filter_singletons(exclude_heterozygous=False)
        .filter_biallelic(exclude_heterozygous=False)
        .resolve()
    )

    nrm.plot_sankey_filtering_report()

    # Make missingness report plots.
    plotting2 = Plotting(genotype_data=gd_filt)
    filt_components, filt_pca = plotting2.run_pca()
    gd_filt.missingness_reports(prefix="filtered")

    # Write the filtered VCF file.
    gd_filt.write_vcf("snpio/example_data/vcf_files/nremover_test.vcf")

    # Encode the genotypes into 012, one-hot, and integer formats.
    ge = GenotypeEncoder(gd_filt)
    gt_012 = ge.genotypes_012
    gt_onehot = ge.genotypes_onehot
    gt_int = ge.genotypes_int

    df012 = pd.DataFrame(gt_012)
    dfint = pd.DataFrame(gt_int)

    print(df012.head())
    print(gt_onehot[:5])
    print(dfint.head())

    tp = TreeParser(
        genotype_data=gd_filt,
        treefile="snpio/example_data/trees/test.tre",
        qmatrix="snpio/example_data/trees/test.iqtree",
        siterates="snpio/example_data/trees/test14K.rate",
        verbose=True,
        debug=False,
    )

    # Get a toytree object by reading the tree file.
    tree = tp.read_tree()

    # Get the tree stats. Returns a dictionary of tree stats.
    print(tp.tree_stats())

    # Reroot the tree at any nodes containing the string 'EA' in the sampleID.
    tp.reroot_tree("~EA")

    # Get a distance matrix between all nodes in the tree.
    print(tp.get_distance_matrix())

    # Get the Rate Matrix Q from the Qmatrix file.
    print(tp.qmat)

    # Get the Site Rates from the Site Rates file.
    print(tp.site_rates)

    # Get a subtree with only the samples containing 'EA' in the sampleID.
    subtree = tp.get_subtree("~EA")

    # Prune the tree to remove samples containing 'ON' in the sampleID.
    pruned_tree = tp.prune_tree("~ON")

    # Write the subtree and pruned tree. Returns a Newick string if 'save_path'
    # is None.
    print(tp.write_tree(subtree, save_path=None))
    print(tp.write_tree(pruned_tree, save_path=None))


if __name__ == "__main__":
    main()
