import os
import unittest

from snpio.io.phylip_reader import PhylipReader


class TestPhylipReader(unittest.TestCase):
    def setUp(self):
        self.phylip_file = "test.phy"
        self.phylip_data = [
            "5 10",
            "Sample1\tACGTACGTAC",
            "Sample2\tTCGATCGATA",
            "Sample3\tGCTAGCTAGC",
            "Sample4\tATCGATCGAT",
            "Sample5\tCGATCGATCG",
        ]
        self.reader = PhylipReader(self.phylip_file, verbose=True)

    def tearDown(self):
        pass

    def test_load_phylip(self):
        with open(self.phylip_file, "w") as f:
            f.write("\n".join(self.phylip_data))

        self.reader._filename = self.phylip_file
        self.reader._load_aln()

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
        self.reader._filename = self.phylip_file
        self.reader._snp_data = [
            ["A", "C", "G", "T", "A", "C", "G", "T", "A", "C"],
            ["T", "C", "G", "A", "T", "C", "G", "A", "T", "A"],
            ["G", "C", "T", "A", "G", "C", "T", "A", "G", "C"],
            ["A", "T", "C", "G", "A", "T", "C", "G", "A", "T"],
            ["C", "G", "A", "T", "C", "G", "A", "T", "C", "G"],
        ]

        output_file = "output.phy"
        self.reader.write_phylip(output_file)

        with open(output_file, "r") as f:
            output_data = f.readlines()

        # Strip newline characters from each line in output_data
        output_data = [line.strip() for line in output_data]

        self.assertEqual(output_data, self.phylip_data)

        # Clean up
        os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
