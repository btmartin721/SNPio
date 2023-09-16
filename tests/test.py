import unittest
import os
from snpio import GenotypeData, NRemover2, Plotting


class TestGenotypeData(unittest.TestCase):
    def setUp(self):
        self.file_data = {
            "example_data/structure_files/phylogen_nomx.ustr": 14760,
            "example_data/phylip_files/phylogen_nomx.u.snps.phy": 14760,
            "example_data/vcf_files/phylogen_subset14K.vcf.gz": 14000
        }

    def test_file_loading(self):
        for filename, expected_loci in self.file_data.items():
            gd = GenotypeData(
                filename=filename,
                popmapfile="example_data/popmaps/phylogen_nomx.popmap",
                force_popmap=True,
                filetype="auto",
                chunk_size=1000,
            )
            # Test if the number of loci is correct
            self.assertEqual(len(gd.snp_data[0]), expected_loci)

            # Test if the number of samples is correct.
            self.assertEqual(len(gd.snp_data), 203)

    def test_write_and_reload_vcf(self):
        for filename in self.file_data.keys():
            gd = GenotypeData(
                filename=filename,
                popmapfile="example_data/popmaps/phylogen_nomx.popmap",
                force_popmap=True,
                filetype="auto",
                chunk_size=1000,
            )
            gd.write_vcf("example_data/vcf_files/gtdata_test.vcf")
            gd_reloaded = GenotypeData(
                "example_data/vcf_files/gtdata_test.vcf",
                popmapfile="example_data/popmaps/phylogen_nomx.popmap",
                force_popmap=True,
                filetype="auto",
                chunk_size=1000,
            )
            self.assertEqual(len(gd_reloaded.snp_data), len(gd.snp_data))
            self.assertEqual(len(gd_reloaded.snp_data[0]), len(gd.snp_data[0]))

    def test_plot_functions(self):
        gd = GenotypeData(
            filename="example_data/vcf_files/phylogen_subset14K.vcf.gz",
            popmapfile="example_data/popmaps/phylogen_nomx.popmap",
            force_popmap=True,
            filetype="auto",
            chunk_size=1000,
        )
        gd.missingness_reports(file_prefix="unfiltered")
        Plotting.run_pca(gd, file_prefix="unfiltered")
        # Check if the plot files exist and are not empty
        for file_prefix in [
            "unfiltered_missingness.png",
            "unfiltered_pca.png",
            "unfiltered_pca.html",
        ]:
            filepath = f"snpio_output/gtdata/plots/{file_prefix}"
            self.assertTrue(os.path.exists(filepath))
            self.assertGreater(os.path.getsize(filepath), 0)

    def test_nremover_function(self):
        filter_params = {
            "max_missing_global": 0.5,
            "max_missing_pop": 0.5,
            "max_missing_sample": 0.8,
            "singletons": True,
            "biallelic": True,
            "monomorphic": True,
            "min_maf": 0.01,
            "search_thresholds": False,
        }
        expected_dimensions = {
            "example_data/structure_files/phylogen_nomx.ustr": (5093, 193),
            "example_data/phylip_files/phylogen_nomx.u.snps.phy": (8154, 194),
            "example_data/vcf_files/phylogen_subset14K.vcf.gz": (7870, 193),
        }
        for filename, (
            expected_loci,
            expected_samples,
        ) in expected_dimensions.items():
            gd = GenotypeData(
                filename=filename,
                popmapfile="example_data/popmaps/phylogen_nomx.popmap",
                force_popmap=True,
                filetype="auto",
                chunk_size=1000,
            )
            nrm = NRemover2(gd)
            gd_filtered = nrm.nremover(**filter_params)

            gd_filtered.write_vcf("test.vcf")
            self.assertEqual(len(gd_filtered.snp_data[0]), expected_loci)
            self.assertEqual(len(gd_filtered.snp_data), expected_samples)


if __name__ == "__main__":
    unittest.main()
