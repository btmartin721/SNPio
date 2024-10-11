import pandas as pd

from snpio import GenotypeEncoder, NRemover2, Plotting, VCFReader

# from snpio.utils.benchmarking import Benchmark


def main():
    # Read the alignment, popmap, and tree files.
    vcf = "example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz"
    pm = "example_data/popmaps/phylogen_nomx.popmap"
    cs = 5000
    fp = True
    gd = VCFReader(filename=vcf, popmapfile=pm, force_popmap=fp, chunk_size=cs)

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
    gd_filt.write_vcf("example_data/vcf_files/nremover_test.vcf")

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


if __name__ == "__main__":
    main()
