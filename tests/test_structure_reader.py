import shutil
import tempfile
import unittest
from pathlib import Path

from snpio.io.structure_reader import StructureReader


class TestStructureReader(unittest.TestCase):
    def setUp(self):
        self.structure_file = tempfile.NamedTemporaryFile(delete=False, suffix=".str")

        self.popmap_file = tempfile.NamedTemporaryFile(delete=False, suffix=".popmap")

        self.output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".str")

        self.popmap_content = ["Sample1\tpop1", "Sample2\tpop1", "Sample3\tpop2"]

        self.structure_data = [
            "Sample1 1 1 2 2 3 3 4 4",  # A=1, C=2, G=3, T=4
            "Sample1 1 1 2 2 3 3 4 4",
            "Sample2 4 4 3 3 2 2 1 1",
            "Sample2 4 4 3 3 2 2 1 1",
            "Sample3 2 2 1 1 4 4 3 3",
            "Sample3 2 2 1 1 4 4 3 3",
        ]

        with open(self.structure_file.name, "w") as f:
            f.write("\n".join(self.structure_data))

        with open(self.popmap_file.name, "w") as f:
            f.write("\n".join(self.popmap_content))

    def tearDown(self):
        if Path(self.structure_file.name).exists():
            Path(self.structure_file.name).unlink(missing_ok=True)

        if Path(self.popmap_file.name).exists():
            Path(self.popmap_file.name).unlink(missing_ok=True)

        if Path(self.output_file.name).exists():
            Path(self.output_file.name).unlink(missing_ok=True)

        dir = Path("test_read_structure_output")
        if dir.is_dir():
            shutil.rmtree(dir)

    def test_load_structure(self):
        reader = StructureReader(
            filename=self.structure_file.name,
            popmapfile=self.popmap_file.name,
            prefix="test_read_structure",
            verbose=False,
        )

        with open(self.structure_file.name, "w") as f:
            f.write("\n".join(self.structure_data))

        self.assertEqual(reader.num_snps, 8)
        self.assertEqual(reader.num_inds, 3)
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])

        self.assertEqual(
            reader.snp_data.tolist(),
            [
                ["A", "A", "C", "C", "G", "G", "T", "T"],
                ["T", "T", "G", "G", "C", "C", "A", "A"],
                ["C", "C", "A", "A", "T", "T", "G", "G"],
            ],
        )

    def test_write_structure(self):
        reader = StructureReader(
            filename=self.structure_file.name,
            popmapfile=self.popmap_file.name,
            prefix="test_read_structure",
            verbose=False,
        )

        output_file = self.output_file.name
        reader.write_structure(output_file)

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


if __name__ == "__main__":
    unittest.main()
