from snpio import GenotypeData, NRemover2, Plotting


def main():
    # Read the alignment, popmap, and tree files
    gd = GenotypeData(
        filename="example_data/phylip_files/phylogen_nomx.u.snps.phy",
        popmapfile="example_data/popmaps/phylogen_nomx.popmap",
        force_popmap=True,
        filetype="auto",
        iqtree_filename="example_data/trees/test.iqtree",
        siterates_iqtree="example_data/trees/test_n500.rate",
        guidetree="example_data/trees/test.tre",
        chunk_size=5000,
    )

    print(gd.iqtree)

    # # Make missingness report plots.
    # gd.missingness_reports(file_prefix="unfiltered")

    # # Run a PCA and make a scatterplot on the unfiltered data.
    # Plotting.run_pca(gd, file_prefix="unfiltered")

    # Run the NRemover class to filter out missing data.
    # nrm = NRemover2(gd)
    # gd_filtered = nrm.nremover(
    #     max_missing_global=0.5,
    #     max_missing_pop=0.5,
    #     max_missing_sample=0.8,
    #     singletons=True,
    #     biallelic=True,
    #     unlinked_only=True,
    #     monomorphic=True,
    #     min_maf=0.01,
    #     search_thresholds=True,
    # )

    # # Make another missingness report plot for filtered data.
    # gd_filtered.missingness_reports(file_prefix="filtered")

    # # Run a PCA on the filtered data and make a scatterplot.
    # Plotting.run_pca(gd_filtered, file_prefix="filtered")

    # gd_filtered.write_vcf("example_data/vcf_files/nremover_test.vcf")
    # gd_filtered.write_phylip("example_data/phylip_files/nremover_test.phy")
    # gd_filtered.write_structure(
    #     "example_data/structure_files/nremover_test.str"
    # )

    # print(gd_filtered.alignment)
    # print(gd_filtered.tree)
    # print(gd_filtered.iqtree)


if __name__ == "__main__":
    main()
