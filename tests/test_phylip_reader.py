import unittest
from pathlib import Path
import tempfile
import shutil

from snpio.io.phylip_reader import PhylipReader


class TestPhylipReader(unittest.TestCase):
    def setUp(self):
        self.phylip_data = [
            "5 10",
            "Sample1\tACGTACGTAC",
            "Sample2\tTCGATCGATA",
            "Sample3\tGCTAGCTAGC",
            "Sample4\tATCGATCGAT",
            "Sample5\tCGATCGATCG",
        ]

        self.test_popmap_content = [
            "Sample1\tpop1",
            "Sample2\tpop1",
            "Sample3\tpop2",
            "Sample4\tpop2",
            "Sample5\tpop2",
        ]

        self.temp_phy = tempfile.NamedTemporaryFile(delete=False, suffix=".phy")
        self.temp_popmap = tempfile.NamedTemporaryFile(delete=False, suffix=".popmap")
        self.temp_output_phy = tempfile.NamedTemporaryFile(delete=False, suffix=".phy")

        with (
            open(self.temp_phy.name, "w") as f_phy,
            open(self.temp_popmap.name, "w") as f_pop,
        ):
            f_phy.write("\n".join(self.phylip_data))
            f_pop.write("\n".join(self.test_popmap_content))

    def tearDown(self):
        dir = Path("test_read_phylip_output")
        if dir.is_dir():
            shutil.rmtree(dir)
        Path(self.temp_phy.name).unlink(missing_ok=True)
        Path(self.temp_popmap.name).unlink(missing_ok=True)
        Path(self.temp_output_phy.name).unlink(missing_ok=True)

    def test_load_phylip(self):
        self.reader = PhylipReader(
            filename=self.temp_phy.name,
            popmapfile=self.temp_popmap.name,
            prefix="test_read_phylip",
            verbose=False,
        )

        self.assertEqual(self.reader.num_snps, 10)
        self.assertEqual(self.reader.num_inds, 5)
        self.assertEqual(
            self.reader.samples, ["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"]
        )
        self.assertEqual(
            self.reader.snp_data.tolist(),
            [
                ["A", "C", "G", "T", "A", "C", "G", "T", "A", "C"],
                ["T", "C", "G", "A", "T", "C", "G", "A", "T", "A"],
                ["G", "C", "T", "A", "G", "C", "T", "A", "G", "C"],
                ["A", "T", "C", "G", "A", "T", "C", "G", "A", "T"],
                ["C", "G", "A", "T", "C", "G", "A", "T", "C", "G"],
            ],
        )

    def test_write_phylip(self):

        self.reader = PhylipReader(
            filename=self.temp_phy.name,
            popmapfile=self.temp_popmap.name,
            prefix="test_read_phylip",
            verbose=False,
        )

        self.reader.write_phylip(self.temp_output_phy.name)

        with open(self.temp_output_phy.name, "r") as f:
            output_data = f.readlines()

        # Strip newline characters from each line in output_data
        output_data = [line.strip() for line in output_data]

        self.assertEqual(output_data, self.phylip_data)

        # Clean up
        self.temp_output_phy.close()
        Path(self.temp_output_phy.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
