import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
import os
import gzip

import numpy as np

from snpio import VCFReader


class TestVCFReader(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

        self.test_vcf_content = textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##source=test
            ##contig=<ID=1,length=249250621>
            ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
            ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
            ##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2\tSample3
            1\t100\t.\tG\tA\t.\tPASS\t.\tGT:DP:GQ\t0/1:35:99\t0/0:40:90\t1/1:10:60
            1\t200\t.\tT\tC\t.\tPASS\t.\tGT:DP:GQ\t0/0:50:70\t0/1:20:85\t1/1:5:40
            1\t300\t.\tA\tG\t.\tPASS\t.\tGT:DP:GQ\t1/1:15:50\t0/0:60:95\t0/1:30:80
            """
        )

        self.temp_vcf = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf")

        with open(self.temp_vcf.name, "w") as f:
            f.write(self.test_vcf_content)

        self.test_popmap_content = "Sample1\tpop1\nSample2\tpop1\nSample3\tpop2\n"
        self.temp_popmap = tempfile.NamedTemporaryFile(delete=False, suffix=".popmap")
        with open(self.temp_popmap.name, "w") as f:
            f.write(self.test_popmap_content)

    def tearDown(self):
        Path(self.temp_vcf.name).unlink(missing_ok=True)
        Path(self.temp_popmap.name).unlink(missing_ok=True)
        Path(self.temp_vcf.name + ".tbi").unlink(missing_ok=True)

        for f in Path(".").glob("tmp*.vcf.gz"):
            if f.is_file():
                f.unlink(missing_ok=True)
        for f in Path(".").glob("tmp*.vcf.gz.tbi"):
            if f.is_file():
                f.unlink(missing_ok=True)

        dir = Path("test_read_vcf_output")
        if dir.is_dir():
            shutil.rmtree(dir)

        dir = Path("test_write_vcf_with_format_output")
        if dir.is_dir():
            shutil.rmtree(dir)
        dir = Path("test_write_vcf_without_format_output")
        if dir.is_dir():
            shutil.rmtree(dir)

    def test_read_vcf(self):
        reader = VCFReader(
            filename=self.temp_vcf.name,
            popmapfile=self.temp_popmap.name,
            chunk_size=3,
            verbose=False,
            debug=False,
            prefix="test_read_vcf",
        )

        self.assertEqual(len(reader.samples), 3)
        expected_snp_data = np.array(
            [["R", "T", "G"], ["G", "Y", "A"], ["A", "C", "R"]]
        )

        self.assertIsInstance(reader.snp_data, np.ndarray)

        np.testing.assert_array_equal(reader.snp_data, expected_snp_data)

        self.assertIsInstance(reader.num_snps, int)
        self.assertIsInstance(reader.num_inds, int)
        self.assertEqual(reader.num_snps, 3)
        self.assertEqual(reader.num_inds, 3)

        self.assertListEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertListEqual(reader.populations, ["pop1", "pop1", "pop2"])

        self.assertDictEqual(
            reader.popmap,
            {"Sample1": "pop1", "Sample2": "pop1", "Sample3": "pop2"},
        )
        self.assertEqual(
            reader.popmap_inverse,
            {"pop1": ["Sample1", "Sample2"], "pop2": ["Sample3"]},
        )

        self.assertTrue(reader.snp_data.shape[0] == reader.num_inds)
        self.assertTrue(reader.snp_data.shape[1] == reader.num_snps)

    def _open_text(self, path: str):
        """Open a file in text mode, handling optional gzip."""
        if os.path.exists(path):
            return open(path, "rt", encoding="utf-8", newline="")
        gz = path + ".gz"
        if os.path.exists(gz):
            # Properly decompress; do NOT read raw bytes.
            return gzip.open(gz, "rt", encoding="utf-8", newline="")
        self.fail(f"Neither {path} nor {gz} exists; VCF writing failed.")

    def _iter_data_lines(self, path: str):
        """Yield non-header, non-empty VCF data lines as text strings."""
        with self._open_text(path) as fh:
            for line in fh:
                if not line:
                    continue
                if line.startswith("#"):
                    # skips both '##' meta and '#CHROM' header
                    continue
                line = line.rstrip("\n\r")
                if line.strip() == "":
                    continue
                yield line

    def test_write_vcf_without_format_fields(self):
        """Tests that VCF writing correctly excludes extra FORMAT fields when store_format_fields is False."""
        reader = VCFReader(
            filename=self.temp_vcf.name,
            popmapfile=self.temp_popmap.name,
            chunk_size=3,
            verbose=False,
            debug=False,
            store_format_fields=False,
            prefix="test_write_vcf_without_format",
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".vcf"
        ) as temp_output_vcf:
            reader.write_vcf(temp_output_vcf.name, chunk_size=2)

            try:
                with open(temp_output_vcf.name, "r") as f:
                    output_lines = [
                        line.strip() for line in f if not line.startswith("##")
                    ]
            except FileNotFoundError:
                with open(temp_output_vcf.name + ".gz", "rb") as f:
                    output_lines = [
                        line.strip() for line in f if not line.startswith(b"##")
                    ]

            # Assertions for the "without" case
            for line in output_lines[1:]:
                try:
                    fields = line.split("\t")
                except TypeError:
                    fields = line.split(b"\t")
                format_field = fields[8]
                self.assertEqual(format_field, "GT")

    def test_write_vcf_with_format_fields(self):
        """Tests that VCF writing includes extra FORMAT fields when store_format_fields=True."""
        reader = VCFReader(
            filename=self.temp_vcf.name,
            popmapfile=self.temp_popmap.name,
            chunk_size=3,
            verbose=False,
            debug=False,
            store_format_fields=True,
            prefix="test_write_vcf_with_format",
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as temp_out:
            out_path = temp_out.name

        # Write the VCF
        # Some implementations may write out_path or out_path+'.gz'
        reader.write_vcf(out_path, chunk_size=2)

        # Collect data lines (text, no headers)
        data_lines = list(self._iter_data_lines(out_path))
        self.assertGreater(len(data_lines), 0, "No variant lines found in output VCF.")

        for i, line in enumerate(data_lines, start=1):
            fields = line.split("\t")
            self.assertGreaterEqual(
                len(fields), 9, f"Variant line {i} has <9 fields: {line!r}"
            )

            format_field = fields[8]  # FORMAT column
            # Basic structural checks
            self.assertIn(
                ":",
                format_field,
                f"FORMAT column missing ':' in line {i}: {format_field!r}",
            )
            self.assertGreaterEqual(
                len(format_field.split(":")),
                2,
                f"FORMAT column should list ≥2 keys in line {i}: {format_field!r}",
            )

            # CONTENT sanity: GT should be present; at least one extra FORMAT key should appear
            fmt_keys = set(format_field.split(":"))
            self.assertIn(
                "GT",
                fmt_keys,
                f"'GT' missing from FORMAT in line {i}: {format_field!r}",
            )
            self.assertTrue(
                len(fmt_keys) >= 2,
                f"Expected ≥2 FORMAT keys when store_format_fields=True, got {fmt_keys} on line {i}",
            )


if __name__ == "__main__":
    unittest.main()
