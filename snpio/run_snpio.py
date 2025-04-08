import pandas as pd

import snpio
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

    print("Using snpio version:", snpio.__version__)

    # Read the alignment, popmap, and tree files.
    gd = VCFReader(
        filename="snpio/example_data/vcf_files/nremover_test.vcf.gz",
        popmapfile="snpio/example_data/popmaps/nremover_test.popmap",
        force_popmap=True,  # Remove samples not in the popmap, or vice versa.
        chunk_size=5000,
        exclude_pops=["DS", "OG"],
        plot_format="png",
        prefix="BOX_analysis",
    )

    pgs = PopGenStatistics(gd, verbose=True)

    summary_stats = pgs.summary_statistics(
        save_plots=True, n_bootstraps=1000, n_jobs=8, use_pvalues=True
    )

    nei = pgs.neis_genetic_distance(n_bootstraps=100, n_jobs=8, return_pvalues=True)

    # df_fst_outliers_boot, df_fst_outlier_pvalues_boot = pgs.detect_fst_outliers(
    #     correction_method="bonf",
    #     use_bootstrap=True,
    #     n_bootstraps=1000,
    #     n_jobs=-1,
    #     tail_direction="upper",
    # )

    # df_fst_outliers_dbscan, df_fst_outlier_pvalues_dbscan = pgs.detect_fst_outliers(
    #     correction_method="bonf", use_bootstrap=False, n_jobs=1
    # )

    # NOTE: Takes a while to run.
    # amova_results = pgs.amova(
    #     regionmap={
    #         "EA": "Eastern",
    #         "GU": "Eastern",_only
    #         "TT": "Eastern",
    #         "TC": "Eastern",
    #         "ON": "Ornate",
    #         "DS": "Ornate",
    #     },
    #     n_bootstraps=10,
    #     n_jobs=1,
    #     random_seed=42,
    # )

    # dstats_df, overall_results = pgs.calculate_d_statistics(
    #     method="patterson",
    #     population1="EA",
    #     population2="GU",
    #     population3="TT",
    #     outgroup="ON",
    #     num_bootstraps=10,
    #     n_jobs=1,
    #     max_individuals_per_pop=6,
    # )

    # # Run PCA and make missingness report plots.
    # plotting = Plotting(genotype_data=gd)
    # gd_components, gd_pca = plotting.run_pca()
    # gd.missingness_reports()

    # nrm.search_thresholds(
    #     thresholds=[0.25, 0.5, 0.75, 1.0],
    #     maf_thresholds=[0.0],
    #     mac_thresholds=[2, 5],
    # )

    # Plot benchmarking results.
    # NOTE: For development purposes. Comment out for normal use.
    # Benchmark.plot_performance(nrm.genotype_data, nrm.genotype_data.resource_data)

    # nrm.plot_sankey_filtering_report()

    # # Make missingness report plots.
    # plotting2 = Plotting(genotype_data=gd_filt)
    # filt_components, filt_pca = plotting2.run_pca()
    # gd_filt.missingness_reports(prefix="filtered")

    # # Write the filtered VCF file.
    # gd_filt.write_vcf("snpio/example_data/vcf_files/nremover_test.vcf")
    # gd_filt.write_popmap("snpio/example_data/popmaps/nremover_test.popmap")

    # # Encode the genotypes into 012, one-hot, and integer formats.
    # ge = GenotypeEncoder(gd_filt)
    # gt_012 = ge.genotypes_012
    # gt_onehot = ge.genotypes_onehot
    # gt_int = ge.genotypes_int

    # df012 = pd.DataFrame(gt_012)
    # dfint = pd.DataFrame(gt_int)

    # tp = TreeParser(
    #     genotype_data=gd_filt,
    #     treefile="snpio/example_data/trees/test.tre",
    #     qmatrix="snpio/example_data/trees/test.iqtree",
    #     siterates="snpio/example_data/trees/test14K.rate",
    #     verbose=True,
    #     debug=False,
    # )

    # # # Get a toytree object by reading the tree file.
    # tree = tp.read_tree()

    # # # Reroot the tree at any nodes containing the string 'EA' in the sampleID.
    # tp.reroot_tree("~EA")

    # # # Get a subtree with only the samples containing 'EA' in the sampleID.
    # subtree = tp.get_subtree("~EA")

    # # # Prune the tree to remove samples containing 'ON' in the sampleID.
    # pruned_tree = tp.prune_tree("~ON")


if __name__ == "__main__":
    main()
