import csv
import unittest
from pathlib import Path

from snpio import NRemover2, VCFReader


class TestNRemover2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a small test VCF file with controlled data for all filter tests
        cls.test_vcf_file = "test_data.vcf"
        if not Path(cls.test_vcf_file).exists():
            with open(cls.test_vcf_file, "w", newline="") as vcf:
                writer = csv.writer(vcf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)

                # Write the VCF header
                writer.writerow(["##fileformat=VCFv4.2"])
                writer.writerow(
                    ["##FORMAT=<ID=GT,Number=1,Type=String,Description='Genotype'>"]
                )
                writer.writerow(
                    [
                        "#CHROM",
                        "POS",
                        "ID",
                        "REF",
                        "ALT",
                        "QUAL",
                        "FILTER",
                        "INFO",
                        "FORMAT",
                        "Sample1",
                        "Sample2",
                        "Sample3",
                    ]
                )

                # Write the VCF data rows (each field strictly tab-delimited)
                writer.writerow(
                    [
                        "NW123.1",
                        100,
                        ".",
                        "A",
                        "T",
                        ".",
                        "PASS",
                        ".",
                        "GT",
                        "0/0",
                        "0/1",
                        "1/1",
                    ]
                )
                writer.writerow(
                    [
                        "NW123.1",
                        200,
                        ".",
                        "G",
                        "A",
                        ".",
                        "PASS",
                        ".",
                        "GT",
                        "0/1",
                        "./.",
                        "0/1",
                    ]
                )
                writer.writerow(
                    [
                        "XM123.1",
                        100,
                        ".",
                        "C",
                        ".",
                        ".",
                        "PASS",
                        ".",
                        "GT",
                        "0/0",
                        "0/0",
                        "0/0",
                    ]
                )
                writer.writerow(
                    [
                        "XM123.1",
                        201,
                        ".",
                        "T",
                        "G",
                        ".",
                        "PASS",
                        ".",
                        "GT",
                        "1/1",
                        "1/1",
                        "0/1",
                    ]
                )
                writer.writerow(
                    [
                        "XM123.1",
                        305,
                        ".",
                        "C",
                        "G,A",
                        ".",
                        "PASS",
                        ".",
                        "GT",
                        "0/2",
                        "0/1",
                        "./.",
                    ]
                )

    def setUp(self):
        # Initialize the VCFReader with the test VCF file
        self.vcf_reader = VCFReader(
            filename=self.test_vcf_file,
            popmapfile=None,
            force_popmap=False,
            chunk_size=100,
        )

        # Initialize NRemover2 with the VCFReader instance
        self.n_remover = NRemover2(self.vcf_reader)

    def test_filter_missing_sample(self):
        # Test filter_missing_sample with a threshold of 0.19 (19% missing allowed)
        filtered_data = self.n_remover.filter_missing_sample(0.19).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.sample_indices) if keep
        ]
        expected_indices = [0]  # Indices of retained samples
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_missing(self):
        # Test filter_missing loci with a threshold of 0.3 (30% missing allowed)
        filtered_data = self.n_remover.filter_missing(0.3).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        expected_indices = [0, 2, 3]  # Indices of retained loci
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_mac(self):
        # Test filter_mac to keep loci with MAC >= 2
        filtered_data = self.n_remover.filter_mac(2).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        expected_indices = [0]  # Indices of retained loci
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_monomorphic(self):
        # Test filter_monomorphic to remove monomorphic loci
        filtered_data = self.n_remover.filter_monomorphic().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        expected_indices = [0, 3, 4]  # Locus 2 filtered out as monomorphic
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_singletons(self):
        # Test filter_singletons to remove loci that only have one
        # non-reference allele
        filtered_data = self.n_remover.filter_singletons().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        expected_indices = [1, 2]  # Indices of retained loci
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_biallelic(self):
        # Test filter_biallelic to keep only loci with two alleles
        filtered_data = self.n_remover.filter_biallelic().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        expected_indices = [0, 1, 3]  # Locus 2 monomorphic; locus 4 multiallelic
        self.assertEqual(retained_indices, expected_indices)

    @classmethod
    def tearDownClass(cls):
        # Clean up by deleting the test VCF file
        if Path(cls.test_vcf_file).exists():
            Path(cls.test_vcf_file).unlink()

        if Path(cls.test_vcf_file + ".tbi").exists():
            Path(cls.test_vcf_file + ".tbi").unlink()


if __name__ == "__main__":
    unittest.main()
