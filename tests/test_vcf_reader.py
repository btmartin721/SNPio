import os
import textwrap
import unittest

import numpy as np

from snpio.io.vcf_reader import VCFReader


class TestVCFReader(unittest.TestCase):

    def setUp(self):
        self.test_vcf = "test.vcf"
        self.test_hdf5 = "test_vcf_attributes.h5"
        self.test_output_vcf = "output_test.vcf"

        vcf_content = textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##source=test
            ##contig=<ID=1,length=249250621>
            ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2
            1\t100\t.\tG\tA\t.\tPASS\t.\tGT\t0/1\t0/0
            1\t200\t.\tT\tC\t.\tPASS\t.\tGT\t0/0\t0/1
            """
        )

        with open(self.test_vcf, "w") as f:
            f.write(vcf_content.strip())

        self.reader = VCFReader(
            filename=self.test_vcf, filetype="vcf", chunk_size=2, verbose=False
        )

    def tearDown(self):
        if os.path.exists(self.test_vcf):
            os.remove(self.test_vcf)
        if os.path.exists(self.test_hdf5):
            os.remove(self.test_hdf5)
        if os.path.exists(self.test_output_vcf):
            os.remove(self.test_output_vcf)

    def test_read_vcf(self):
        self.reader.load_aln()
        self.assertEqual(len(self.reader.samples), 2)
        expected_snp_data = [["R", "G"], ["T", "Y"]]
        np.testing.assert_array_equal(self.reader.snp_data, expected_snp_data)

    def test_write_vcf(self):
        self.reader.load_aln()
        self.reader.write_vcf(output_filename=self.test_output_vcf, chunk_size=2)

        with open(self.test_output_vcf, "r") as f:
            output_vcf_content = f.readlines()

        expected_output_vcf_content = [
            "##fileformat=VCFv4.2\n",
            "##source=SNPio\n",
            '##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">\n',
            '##INFO=<ID=VAF,Number=A,Type=Float,Description="Variant Allele Frequency">\n',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2\n",
            "1\t100\t.\tG\tA\t.\t.\t.\tGT\t0/1\t0/0\n",
            "1\t200\t.\tT\tC\t.\t.\t.\tGT\t0/0\t0/1\n",
        ]

        self.assertEqual(output_vcf_content, expected_output_vcf_content)


if __name__ == "__main__":
    unittest.main()
