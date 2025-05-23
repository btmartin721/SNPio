import shutil
import tempfile
import unittest
from pathlib import Path

from snpio.io.structure_reader import StructureReader


class TestStructureReader(unittest.TestCase):
    def setUp(self):
        self.popmap_content = ["Sample1\tpop1", "Sample2\tpop1", "Sample3\tpop2"]

    def tearDown(self):
        for ext in ("*.str", "*.popmap"):
            for fn in Path().glob(ext):
                fn.unlink(missing_ok=True)

    def _write_file(self, lines, suffix=".str"):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w")
        tf.write("\n".join(lines))
        tf.close()
        return tf.name

    def _write_popmap(self, lines):
        pf = tempfile.NamedTemporaryFile(delete=False, suffix=".popmap", mode="w")
        pf.write("\n".join(lines))
        pf.close()
        return pf.name

    def test_one_row_no_header_no_popids(self):
        struct = [
            "Sample1 1 2 2 2 3 3 4 4",
            "Sample2 4 4 3 3 2 2 1 1",
            "Sample3 2 2 1 1 4 4 3 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=False,
            has_popids=False,
            verbose=False,
        )
        # no header → marker_names None
        self.assertIsNone(reader.marker_names)
        # sample & pop info
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, ["pop1", "pop1", "pop2"])
        # now 4 loci (not 8)
        self.assertEqual(reader.num_inds, 3)
        self.assertEqual(reader.num_snps, 4)

        # genotype calls: (1/1→A), (2/2→C), (3/3→G), (4/4→T)
        expected = [
            ["M", "C", "G", "T"],
            ["T", "G", "C", "A"],
            ["C", "A", "T", "G"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_one_row_with_header_no_popids(self):
        header = " M1 M2 M3 M4"
        struct = [
            header,
            "Sample1 1 2 2 2 3 3 4 4",
            "Sample2 4 4 3 3 2 2 1 1",
            "Sample3 2 2 1 1 4 4 3 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=True,
            has_popids=False,
            verbose=False,
        )
        # header → marker names for 4 loci
        self.assertEqual(reader.marker_names, ["M1", "M2", "M3", "M4"])
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, ["pop1", "pop1", "pop2"])
        self.assertEqual(reader.num_snps, 4)

        expected = [
            ["M", "C", "G", "T"],
            ["T", "G", "C", "A"],
            ["C", "A", "T", "G"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_one_row_with_header_with_popids(self):
        header = "  M1 M2 M3 M4"
        struct = [
            header,
            "Sample1 0 1 2 2 2 3 3 4 4",
            "Sample2 0 4 4 3 3 2 2 1 1",
            "Sample3 1 2 2 1 1 4 4 3 3",
        ]
        fstr = self._write_file(struct)

        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=True,
            has_popids=True,
            verbose=False,
        )
        # header → marker names for 4 loci
        self.assertEqual(reader.marker_names, ["M1", "M2", "M3", "M4"])
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, [0, 0, 1])
        self.assertEqual(reader.num_snps, 4)

        expected = [
            ["M", "C", "G", "T"],
            ["T", "G", "C", "A"],
            ["C", "A", "T", "G"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_two_row_with_header_with_popids(self):
        header = "  M1 M2 M3 M4"
        struct = [
            header,
            "Sample1 0 1 2 3 4",
            "Sample1 0 2 2 3 4",
            "Sample2 0 4 3 2 1",
            "Sample2 0 4 3 2 1",
            "Sample3 1 2 1 4 3",
            "Sample3 1 2 1 4 3",
        ]

        fstr = self._write_file(struct)

        fpop = self._write_popmap(self.popmap_content)
        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=True,
            has_popids=True,
            verbose=False,
        )

        # header → 4 marker names
        self.assertEqual(reader.marker_names, ["M1", "M2", "M3", "M4"])
        # samples & inline populations
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, [0, 0, 1])
        # two-line input for 4 loci → num_snps == 4
        self.assertEqual(reader.num_snps, 4)

        # merged genotypes: (1/1→A), (2/2→C), (3/3→G), (4/4→T)
        expected = [
            ["M", "C", "G", "T"],
            ["T", "G", "C", "A"],
            ["C", "A", "T", "G"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_two_row_with_header_no_popids(self):
        header = " M1 M2 M3 M4"
        struct = [
            header,
            "Sample1 1 2 3 4",
            "Sample1 2 2 3 4",
            "Sample2 4 3 2 1",
            "Sample2 4 3 2 1",
            "Sample3 2 1 4 3",
            "Sample3 2 1 4 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=True,
            has_popids=False,
            verbose=False,
        )
        # header → marker names for 4 loci
        self.assertEqual(reader.marker_names, ["M1", "M2", "M3", "M4"])
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, ["pop1", "pop1", "pop2"])
        self.assertEqual(reader.num_snps, 4)

        expected = [
            ["M", "C", "G", "T"],
            ["T", "G", "C", "A"],
            ["C", "A", "T", "G"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_two_row_no_header_no_popids(self):
        struct = [
            "Sample1 1 2 3 4",
            "Sample1 2 2 3 4",
            "Sample2 4 3 2 1",
            "Sample2 4 3 2 1",
            "Sample3 2 1 4 3",
            "Sample3 2 1 4 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=False,
            has_popids=False,
            verbose=False,
        )
        self.assertEqual(reader.marker_names, None)
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, ["pop1", "pop1", "pop2"])
        self.assertEqual(reader.num_snps, 4)

        expected = [
            ["M", "C", "G", "T"],
            ["T", "G", "C", "A"],
            ["C", "A", "T", "G"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_two_row_no_header_with_popids(self):
        struct = [
            "Sample1 0 1 2 3 4",
            "Sample1 0 2 2 3 4",
            "Sample2 0 4 3 2 1",
            "Sample2 0 4 3 2 1",
            "Sample3 1 2 1 4 3",
            "Sample3 1 2 1 4 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=None,
            has_marker_names=False,
            has_popids=True,
            verbose=False,
        )
        self.assertEqual(reader.marker_names, None)
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, [0, 0, 1])
        self.assertEqual(reader.num_snps, 4)

        expected = [
            ["M", "C", "G", "T"],
            ["T", "G", "C", "A"],
            ["C", "A", "T", "G"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_custom_allele_encoding(self):
        """Test StructureReader with custom allele_encoding mapping."""
        header = " M1 M2 M3 M4"
        struct = [
            header,
            "Sample1 0 1 2 3",  # A/C/T/G → heterozygous A/C
            "Sample1 1 1 2 3",
            "Sample2 2 2 1 0",
            "Sample2 2 2 1 0",
            "Sample3 1 0 3 -9",
            "Sample3 1 0 3 -9",
        ]

        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        # Manually supply a non-default allele_encoding
        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=True,
            has_popids=False,
            verbose=False,
            allele_encoding={0: "A", 1: "C", 2: "T", 3: "G", -9: "N"},
        )

        self.assertEqual(reader.marker_names, ["M1", "M2", "M3", "M4"])
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, ["pop1", "pop1", "pop2"])
        self.assertEqual(reader.num_snps, 4)

        # Interpretations (IUPAC):
        # Sample1: A/C = M, A/T = W, A/G = R, A/G = R
        # Sample2: T/T = T, T/T = T, C/C = C, A/A = A
        # Sample3: C/C = C, A/A = A, G/G = G, N/N = N
        expected = [
            ["M", "C", "T", "G"],
            ["T", "T", "C", "A"],
            ["C", "A", "G", "N"],
        ]
        self.assertEqual(reader.snp_data.tolist(), expected)

    def test_write_structure_onerow_with_marker_names_and_popids(self):
        header = "   M1 M2 M3 M4"
        struct = [
            header,
            "Sample1 0 1 2 3 4",
            "Sample1 0 2 2 3 4",
            "Sample2 0 4 3 2 1",
            "Sample2 0 4 3 2 1",
            "Sample3 1 2 1 4 3",
            "Sample3 1 2 1 4 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=True,
            has_popids=True,
            verbose=False,
        )

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".str").name
        reader.write_structure(
            output_file=output_path,
            onerow=True,
            popids=True,
            marker_names=True,
        )

        with open(output_path) as f:
            lines = [line.strip() for line in f.readlines()]

        self.assertTrue(lines[0].startswith("M1\tM1\tM2\tM2\tM3\tM3\tM4\tM4"))
        self.assertEqual(len(lines), 4)  # 3 samples x 1 lines + header
        self.assertTrue(lines[1].startswith("Sample1\t0\t1\t2\t2\t2\t3\t3\t4\t4"))
        self.assertTrue(lines[2].startswith("Sample2\t0\t4\t4\t3\t3\t2\t2\t1\t1"))
        self.assertTrue(lines[3].startswith("Sample3\t1\t2\t2\t1\t1\t4\t4\t3\t3"))

    def test_write_structure_tworow_with_marker_names_no_popids(self):
        header = "   M1 M2 M3 M4"
        struct = [
            header,
            "Sample1 1 2 3 4",
            "Sample1 2 2 3 4",
            "Sample2 4 3 2 1",
            "Sample2 4 3 2 1",
            "Sample3 2 1 4 3",
            "Sample3 2 1 4 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=True,
            has_popids=False,
            verbose=False,
        )

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".str").name
        reader.write_structure(
            output_file=output_path,
            onerow=False,
            popids=False,
            marker_names=True,
        )

        with open(output_path) as f:
            lines = [line.strip() for line in f.readlines()]

        self.assertTrue(lines[0].startswith("M1\tM2\tM3\tM4"))
        self.assertEqual(len(lines), 7)  # 3 samples x 2 lines + header
        self.assertEqual(lines[1], "Sample1\t1\t2\t3\t4")
        self.assertEqual(lines[2], "Sample1\t2\t2\t3\t4")
        self.assertEqual(lines[3], "Sample2\t4\t3\t2\t1")
        self.assertEqual(lines[4], "Sample2\t4\t3\t2\t1")
        self.assertEqual(lines[5], "Sample3\t2\t1\t4\t3")
        self.assertEqual(lines[6], "Sample3\t2\t1\t4\t3")

    def test_write_structure_onerow_with_default_marker_names(self):
        struct = [
            "Sample1 1 2 3 4",
            "Sample1 2 2 3 4",
            "Sample2 4 3 2 1",
            "Sample2 4 3 2 1",
            "Sample3 2 1 4 3",
            "Sample3 2 1 4 3",
        ]
        fstr = self._write_file(struct)
        fpop = self._write_popmap(self.popmap_content)

        reader = StructureReader(
            filename=fstr,
            popmapfile=fpop,
            has_marker_names=False,
            has_popids=False,
            verbose=False,
        )

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".str").name
        reader.write_structure(
            output_file=output_path,
            onerow=True,
            popids=False,
            marker_names=True,
        )

        with open(output_path) as f:
            lines = [line.strip() for line in f.readlines()]

        self.assertTrue(
            lines[0].startswith(
                "locus_0\tlocus_0\tlocus_1\tlocus_1\tlocus_2\tlocus_2\tlocus_3\tlocus_3"
            )
        )
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[1].startswith("Sample1\t1\t2\t2\t2\t3\t3\t4\t4"))
        self.assertTrue(lines[2].startswith("Sample2\t4\t4\t3\t3\t2\t2\t1\t1"))
        self.assertTrue(lines[3].startswith("Sample3\t2\t2\t1\t1\t4\t4\t3\t3"))


if __name__ == "__main__":
    unittest.main()
