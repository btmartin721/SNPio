import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from snpio import NRemover2, VCFReader


class TestNRemover2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_vcf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf")
        cls.tmp_popmap_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".popmap"
        )
        cls.tmp_output_vcf_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".vcf"
        )
        cls.tmp_output_popmap_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".popmap"
        )
        cls.test_popmap_content = ["Sample1\tpop1", "Sample2\tpop1", "Sample3\tpop2"]

        with open(cls.tmp_vcf_file.name, "w", newline="") as vcf:
            writer = csv.writer(vcf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["##fileformat=VCFv4.2"])
            writer.writerow(
                ["##FORMAT=<ID=GT,Number=1,Type=String,Description='Genotype'>"]
            )
            writer.writerow(
                ["##FORMAT=<ID=AD,Number=R,Type=Integer,Description='Allele depths'>"]
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
                    "GT:AD",
                    "0/0:10,0",
                    "0/1:5,5",
                    "1/1:0,10",
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
                    "GT:AD",
                    "0/1:3,2",
                    "./.:.",
                    "0/1:4,4",
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
                    "GT:AD",
                    "0/0:8",
                    "0/0:9",
                    "0/0:10",
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
                    "GT:AD",
                    "1/1:0,9",
                    "1/1:0,8",
                    "0/1:4,4",
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
                    "GT:AD",
                    "0/2:5,0,5",
                    "0/1:3,3,0",
                    "./.:.",
                ]
            )
            writer.writerow(
                [
                    "XM123.1",
                    400,
                    ".",
                    "A",
                    "C",
                    ".",
                    "PASS",
                    ".",
                    "GT:AD",
                    "0/0:10,0",
                    "0/0:5,5",
                    "0/1:0,10",
                ]
            )

        with open(cls.tmp_popmap_file.name, "w") as popmap:
            for line in cls.test_popmap_content:
                popmap.write(line + "\n")

    def setUp(self):
        self.vcf_reader = VCFReader(
            filename=self.tmp_vcf_file.name,
            popmapfile=self.tmp_popmap_file.name,
            chunk_size=100,
            prefix="test_read_vcf",
            verbose=False,
            store_format_fields=True,
        )
        self.nrm = NRemover2(self.vcf_reader)

    def test_filter_missing_sample(self):
        filtered_data = self.nrm.filter_missing_sample(0.15).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.sample_indices) if keep
        ]
        expected_indices = [0]  # Only Sample1 has enough data
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_missing(self):
        filtered_data = self.nrm.filter_missing(0.3).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        expected_indices = [0, 2, 3, 5]  # Loci with less than 30% missing data
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_mac(self):
        filtered_data = self.nrm.filter_mac(2, exclude_heterozygous=True).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        self.assertEqual(retained_indices, [0])

    def test_filter_monomorphic(self):
        filtered_data = self.nrm.filter_monomorphic().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]

        expected_indices = [0, 1, 3, 4, 5]
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_singletons(self):
        filtered_data = self.nrm.filter_singletons().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]

        # The third and fifth loci are filtered out (singletons)
        # The first, second, and fourth loci are retained
        expected_indices = [0, 1, 2, 4]
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_biallelic(self):
        filtered_data = self.nrm.filter_biallelic().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        self.assertEqual(retained_indices, [0, 1, 3, 5])

    def test_filter_linked(self):
        np.random.seed(0)
        filtered_data = self.nrm.filter_linked().resolve()
        self.assertEqual(filtered_data.num_snps, 2)
        self.assertEqual(np.count_nonzero(filtered_data.loci_indices), 2)

    def test_filter_allele_depth(self):
        # Test filter_allele_depth with minimum total AD = 26
        filtered_data = self.nrm.filter_allele_depth(min_total_depth=26).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]

        # Based on AD values in test VCF:
        # all loci have at least one sample with AD sum ≥ 26
        expected_indices = [0, 2, 5]
        self.assertEqual(retained_indices, expected_indices)

    @classmethod
    def tearDownClass(cls):
        for path in [
            cls.tmp_vcf_file.name,
            cls.tmp_popmap_file.name,
            cls.tmp_output_vcf_file.name,
            cls.tmp_output_popmap_file.name,
            cls.tmp_vcf_file.name + ".tbi",
        ]:
            Path(path).unlink(missing_ok=True)

        shutil.rmtree(Path("test_read_vcf_output"), ignore_errors=True)

        for pattern in ["tmp*.vcf.gz", "tmp*.vcf.gz.tbi", "tmp*.popmap"]:
            for f in Path(".").glob(pattern):
                f.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
