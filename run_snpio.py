from snpio import NRemover2, Plotting, VCFReader
from snpio.utils.benchmarking import Benchmark


def main():
    # Read the alignment, popmap, and tree files
    gd = VCFReader(
        filename="example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz",
        popmapfile="example_data/popmaps/phylogen_nomx.popmap",
        force_popmap=True,
        chunk_size=5000,
        verbose=False,
        benchmark=True,
    )

    nrm = NRemover2(gd)

    nrm.search_thresholds(
        thresholds=[0.5, 0.75, 1.0],
        maf_thresholds=[0.0, 0.01, 0.05],
        mac_thresholds=[2],
    )

    Benchmark.plot_performance(nrm.genotype_data, nrm.genotype_data.resource_data)

    # gd.missingness_reports()

    gd_filt = (
        nrm.filter_missing_pop(0.8)
        .filter_singletons(exclude_heterozygous=False)
        .filter_biallelic(exclude_heterozygous=False)
        .filter_monomorphic(exclude_heterozygous=False)
        .filter_mac(2)
        .filter_missing_sample(0.8)
        .filter_missing(0.8)
        .resolve()
    )

    nrm.plot_sankey_filtering_report()

    # Make missingness report plots.
    gd_filt.missingness_reports(prefix="filtered")
    gd_filt.write_vcf("example_data/vcf_files/nremover_test.vcf")


if __name__ == "__main__":
    main()
