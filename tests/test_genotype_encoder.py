import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from snpio import GenotypeEncoder, PhylipReader, VCFReader


class TestGenotypeEncoder(unittest.TestCase):
    def setUp(self):
        self.tmp_phy_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".phy", mode="w"
        )
        self.tmp_popmap_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".popmap", mode="w"
        )
        self.tmp_vcf_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".vcf", mode="w"
        )

        self.phylip_data_biallelic = textwrap.dedent("""\
            5 10
            Sample1\tACACRTAGTG
            Sample2\tACACGTRGTG
            Sample3\tTGGTRCGTAC
            Sample4\tTGGTACGTAC
            Sample5\tTGGTACGTAC
            """)

        self.vcf_data = textwrap.dedent("""\
            ##fileformat=VCFv4.2
            ##contig=<ID=chr1>
            ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2\tSample3\tSample4\tSample5
            chr1\t1\tref_a_alt_g\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t./.\t0|1
            chr1\t2\tref_g_alt_a\tG\tA\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0\t./.\t1|0
            chr1\t3\tmulti_alt\tA\tC,G\t.\tPASS\t.\tGT\t0/0\t0/1\t0/2\t1/2\t2/2
            chr1\t4\tvcf_ref_is_minor\tA\tG\t.\tPASS\t.\tGT\t0/0\t1/1\t1/1\t1/1\t0/1
            """)

        self.test_popmap_content = [
            "Sample1\tpop1",
            "Sample2\tpop1",
            "Sample3\tpop2",
            "Sample4\tpop2",
            "Sample5\tpop2",
        ]

        with open(self.tmp_phy_file.name, "w") as f:
            f.write(self.phylip_data_biallelic)

        with open(self.tmp_popmap_file.name, "w") as f:
            f.write("\n".join(self.test_popmap_content))

        with open(self.tmp_vcf_file.name, "w") as f:
            f.write(self.vcf_data)

        self.genotype_data = PhylipReader(
            filename=self.tmp_phy_file.name,
            popmapfile=self.tmp_popmap_file.name,
            prefix="test_read_phylip",
            verbose=False,
        )

        self.vcf_genotype_data = VCFReader(
            filename=self.tmp_vcf_file.name,
            popmapfile=self.tmp_popmap_file.name,
            prefix="test_read_vcf",
            verbose=False,
        )

        self.encoder = GenotypeEncoder(self.genotype_data)
        self.vcf_encoder = GenotypeEncoder(self.vcf_genotype_data)

    def test_convert_012(self):
        expected = np.array(
            [
                [2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        )

        result = self.encoder.genotypes_012
        np.testing.assert_array_equal(result, expected)

    def test_convert_012_from_vcf(self):
        """Test VCF-backed 012 encoding for REF/ALT, phased, missing, and multi-ALT calls.

        Expected VCF 012 semantics:
            0 = homozygous REF
            1 = heterozygous REF/ALT
            2 = homozygous ALT or two non-REF alleles
            -9 = missing

        The final locus intentionally makes the VCF REF allele the minor allele. This verifies that VCF REF/ALT metadata, not inferred major/minor allele logic, controls the 012 orientation.
        """
        expected = np.array(
            [
                [0, 2, 0, 0],  # Sample1: 0/0, 1/1, 0/0, 0/0
                [1, 1, 1, 2],  # Sample2: 0/1, 0/1, 0/1, 1/1
                [2, 0, 1, 2],  # Sample3: 1/1, 0/0, 0/2, 1/1
                [-9, -9, 2, 2],  # Sample4: ./., ./., 1/2, 1/1
                [1, 1, 2, 1],  # Sample5: 0|1, 1|0, 2/2, 0/1
            ],
            dtype=np.int8,
        )

        result = self.vcf_encoder.genotypes_012
        np.testing.assert_array_equal(result, expected)

    def test_convert_onehot(self):
        # One-hot columns are [A, C, G, T]
        expected = np.array(
            [
                [  # Sample1: ACACRTAGTG
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [1.0, 0.0, 1.0, 0.0],  # R (A/G)
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [0.0, 0.0, 1.0, 0.0],  # G
                ],
                [  # Sample2: ACACGTRGTG
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 1.0, 0.0],  # R (A/G)
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [0.0, 0.0, 1.0, 0.0],  # G
                ],
                [  # Sample3: TGGTRCGTAC
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 1.0, 0.0],  # R (A/G)
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                ],
                [  # Sample4: TGGTACGTAC
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                ],
                [  # Sample5: TGGTACGTAC
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                ],
            ],
            dtype=np.float32,
        )

        result = self.encoder.genotypes_onehot
        np.testing.assert_array_equal(result, expected)

    def test_convert_integer(self):
        # Integer codes: A=0, C=1, G=2, T=3, R=5
        expected = np.array(
            [
                [0, 1, 0, 1, 5, 3, 0, 2, 3, 2],  # Sample1 ACACRTAGTG
                [0, 1, 0, 1, 2, 3, 5, 2, 3, 2],  # Sample2 ACACGTRGTG
                [3, 2, 2, 3, 5, 1, 2, 3, 0, 1],  # Sample3 TGGTRCGTAC
                [3, 2, 2, 3, 0, 1, 2, 3, 0, 1],  # Sample4 TGGTACGTAC
                [3, 2, 2, 3, 0, 1, 2, 3, 0, 1],  # Sample5 TGGTACGTAC
            ],
            dtype=np.int8,
        )

        result = self.encoder.genotypes_int
        np.testing.assert_array_equal(result, expected)

    def tearDown(self):
        # Clean up temporary files.
        for f in [self.tmp_phy_file, self.tmp_popmap_file, self.tmp_vcf_file]:
            Path(f.name).unlink(missing_ok=True)

        # Clean up output directories if they exist.
        for output_dir in [
            Path("test_read_phylip_output"),
            Path("test_read_vcf_output"),
        ]:
            if output_dir.is_dir():
                shutil.rmtree(output_dir)

        # Clean up any temporary files left in the current working directory.
        for pattern in ["tmp*.phy", "tmp*.popmap", "tmp*.vcf"]:
            for f in Path(".").glob(pattern):
                if f.is_file():
                    f.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
