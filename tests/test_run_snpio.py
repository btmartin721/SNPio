import unittest
from unittest.mock import MagicMock, patch, call

from snpio.run_snpio import main
from snpio import NRemover2, Plotting, VCFReader


class TestRunSnpio(unittest.TestCase):

    @patch("run_snpio.VCFReader")
    @patch("run_snpio.Plotting")
    @patch("run_snpio.NRemover2")
    def test_main(self, MockNRemover2, MockPlotting, MockVCFReader):
        # Mock the VCFReader instance
        mock_vcf_reader = MockVCFReader.return_value
        mock_vcf_reader.missingness_reports = MagicMock()

        # Mock the Plotting instance
        mock_plotting = MockPlotting.return_value
        mock_plotting.run_pca = MagicMock(return_value=(MagicMock(), MagicMock()))

        # Mock the NRemover2 instance
        mock_nremover2 = MockNRemover2.return_value
        mock_nremover2.filter_missing_sample.return_value = mock_nremover2
        mock_nremover2.filter_missing.return_value = mock_nremover2
        mock_nremover2.filter_missing_pop.return_value = mock_nremover2
        mock_nremover2.filter_mac.return_value = mock_nremover2
        mock_nremover2.filter_monomorphic.return_value = mock_nremover2
        mock_nremover2.filter_singletons.return_value = mock_nremover2
        mock_nremover2.filter_biallelic.return_value = mock_nremover2
        mock_nremover2.resolve.return_value = mock_nremover2
        mock_nremover2.plot_sankey_filtering_report = MagicMock()

        # Mock the filtered genotype data
        mock_gd_filt = mock_nremover2
        mock_gd_filt.missingness_reports = MagicMock()
        mock_gd_filt.write_vcf = MagicMock()

        # Run the main function
        main()

        # Assertions to ensure the methods were called
        MockVCFReader.assert_called_once_with(
            filename="snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz",
            popmapfile="snpio/example_data/popmaps/phylogen_nomx.popmap",
            force_popmap=True,
            chunk_size=5000,
        )
        mock_vcf_reader.missingness_reports.assert_called_once()

        # Instead of assert_has_calls, use assert_any_call for individual calls
        MockPlotting.assert_any_call(genotype_data=mock_vcf_reader)
        MockPlotting.assert_any_call(genotype_data=mock_gd_filt)

        mock_plotting.run_pca.assert_called()  # Check if run_pca was called at least once
        MockNRemover2.assert_called_with(mock_vcf_reader)
        mock_nremover2.filter_missing_sample.assert_called_once_with(0.75)
        mock_nremover2.filter_missing.assert_called_once_with(0.75)
        mock_nremover2.filter_missing_pop.assert_called_once_with(0.75)
        mock_nremover2.filter_mac.assert_called_once_with(2)
        mock_nremover2.filter_monomorphic.assert_called_once_with(
            exclude_heterozygous=False
        )
        mock_nremover2.filter_singletons.assert_called_once_with(
            exclude_heterozygous=False
        )
        mock_nremover2.filter_biallelic.assert_called_once_with(
            exclude_heterozygous=False
        )
        mock_nremover2.resolve.assert_called_once()
        mock_nremover2.plot_sankey_filtering_report.assert_called_once()

        mock_gd_filt.missingness_reports.assert_called_once_with(prefix="filtered")
        mock_gd_filt.write_vcf.assert_called_once_with(
            "snpio/example_data/vcf_files/nremover_test.vcf"
        )


if __name__ == "__main__":
    unittest.main()
