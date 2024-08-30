import os
import unittest
from snpio.io.structure_reader import StructureReader

class TestStructureReader(unittest.TestCase):
    def setUp(self):
        self.structure_file = "test.str"
        self.structure_data = [
            "Sample1 1 1 2 2 3 3 4 4",  # A=1, C=2, G=3, T=4
            "Sample1 1 1 2 2 3 3 4 4",
            "Sample2 4 4 3 3 2 2 1 1",
            "Sample2 4 4 3 3 2 2 1 1",
            "Sample3 2 2 1 1 4 4 3 3",
            "Sample3 2 2 1 1 4 4 3 3",
        ]
        with open(self.structure_file, "w") as f:
            f.write("\n".join(self.structure_data))

        self.reader = StructureReader(filename=self.structure_file, has_popids=False)

    def tearDown(self):
        if os.path.exists(self.structure_file):
            os.remove(self.structure_file)

    def test_load_structure(self):
        with open(self.structure_file, "w") as f:
            f.write("\n".join(self.structure_data))

        self.reader._filename = self.structure_file
        self.reader._load_aln()

        self.assertEqual(self.reader.num_snps, 8)
        self.assertEqual(self.reader.num_inds, 3)
        self.assertEqual(
            self.reader.samples, ["Sample1", "Sample2", "Sample3"]
        )
        self.assertEqual(
            self.reader.snp_data,
            [
                ["A", "A", "C", "C", "G", "G", "T", "T"],
                ["T", "T", "G", "G", "C", "C", "A", "A"],
                ["C", "C", "A", "A", "T", "T", "G", "G"],
            ],
        )

    def test_write_structure(self):
        # Set up _snp_data with IUPAC codes
        self.reader._snp_data = [
            ["A", "A", "C", "C", "G", "G", "T", "T"],  # IUPAC for 1/1, 2/2, 3/3, 4/4
            ["T", "T", "G", "G", "C", "C", "A", "A"],  # IUPAC for 4/4, 3/3, 2/2, 1/1
            ["C", "C", "A", "A", "T", "T", "G", "G"],  # IUPAC for 2/2, 1/1, 4/4, 3/3
        ]
        self.reader._samples = ["Sample1", "Sample2", "Sample3"]

        output_file = "output.str"
        self.reader.write_structure(output_file)

        with open(output_file, "r") as f:
            output_data = f.readlines()

        output_data = [line.strip() for line in output_data]

        expected_output_data = [
            "Sample1\t1\t1\t2\t2\t3\t3\t4\t4",
            "Sample1\t1\t1\t2\t2\t3\t3\t4\t4",
            "Sample2\t4\t4\t3\t3\t2\t2\t1\t1",
            "Sample2\t4\t4\t3\t3\t2\t2\t1\t1",
            "Sample3\t2\t2\t1\t1\t4\t4\t3\t3",
            "Sample3\t2\t2\t1\t1\t4\t4\t3\t3",
        ]

        self.assertEqual(output_data, expected_output_data)

        os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
