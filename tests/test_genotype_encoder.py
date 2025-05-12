import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from snpio import GenotypeEncoder, PhylipReader


class TestGenotypeEncoder(unittest.TestCase):
    def setUp(self):

        self.tmp_phy_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".phy", mode="w"
        )
        self.tmp_popmap_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".popmap", mode="w"
        )

        self.phylip_data_biallelic = textwrap.dedent(
            """\
            5 10
            Sample1\tACACRTAGTG
            Sample2\tACACGTRGTG
            Sample3\tTGGTRCGTAC
            Sample4\tTGGTACGTAC
            Sample5\tTGGTACGTAC
            """
        )

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

        self.genotype_data = PhylipReader(
            filename=self.tmp_phy_file.name,
            popmapfile=self.tmp_popmap_file.name,
            prefix="test_read_phylip",
            verbose=False,
        )
        with open(self.tmp_phy_file.name, "w") as f:
            f.write(self.phylip_data_biallelic)

        with open(self.tmp_popmap_file.name, "w") as f:
            f.write("\n".join(self.test_popmap_content))

        self.genotype_data = PhylipReader(
            filename=self.tmp_phy_file.name,
            popmapfile=self.tmp_popmap_file.name,
            prefix="test_read_phylip",
            verbose=False,
        )

        self.encoder = GenotypeEncoder(self.genotype_data)

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

    def test_convert_onehot(self):
        expected = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=np.float32,
        )

        result = self.encoder.genotypes_onehot
        np.testing.assert_array_equal(result, expected)

    def test_convert_integer(self):
        expected = np.array(
            [
                [0, 3, 0, 3, 5, 1, 0, 2, 1, 2],
                [0, 3, 0, 3, 2, 1, 5, 2, 1, 2],
                [1, 2, 2, 1, 5, 3, 2, 1, 0, 3],
                [1, 2, 2, 1, 0, 3, 2, 1, 0, 3],
                [1, 2, 2, 1, 0, 3, 2, 1, 0, 3],
            ],
            dtype=np.int8,
        )

        result = self.encoder.genotypes_int
        np.testing.assert_array_equal(result, expected)

    def tearDown(self):
        # Clean up temporary files
        for f in [self.tmp_phy_file, self.tmp_popmap_file]:
            Path(f.name).unlink(missing_ok=True)

        # Clean up the output directory if it exists
        output_dir = Path("test_read_phylip_output")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        # Clean up the temporary files
        for f in Path(".").glob("tmp*.phy"):
            if f.is_file():
                f.unlink(missing_ok=True)

        for f in Path(".").glob("tmp*.popmap"):
            if f.is_file():
                f.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
