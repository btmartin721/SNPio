import os
import textwrap
import unittest

import h5py
import numpy as np
from pysam import VariantFile

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

        self.reader = VCFReader(filename=self.test_vcf, verbose=False)

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

    def test_get_vcf_attributes(self):
        hdf5_file_path, snp_data, _ = self.reader.get_vcf_attributes(
            VariantFile(self.test_vcf, mode="r")
        )
        self.assertTrue(os.path.exists(hdf5_file_path))

        with h5py.File(hdf5_file_path, "r") as f:
            chrom_data = f["chrom"][:]
            pos_data = f["pos"][:]
            ref_data = f["ref"][:]
            alt_data = f["alt"][:]
            snp_data = self.reader.snp_data
            np.testing.assert_array_equal(chrom_data, np.array([b"1", b"1"]))
            np.testing.assert_array_equal(pos_data, np.array([100, 200]))
            np.testing.assert_array_equal(ref_data, np.array([b"G", b"T"]))
            np.testing.assert_array_equal(alt_data, np.array([b"A", b"C"]))
            self.assertEqual(snp_data.shape, (2, 2))

    def test_write_vcf(self):
        self.reader.load_aln()
        self.reader.write_vcf(
            output_filename=self.test_output_vcf,
            hdf5_file_path=self.reader._vcf_attributes,
        )

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
