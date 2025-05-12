import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

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

    def test_write_vcf_with_and_without_format_fields(self):
        for store_format_fields in (False, True):
            reader = VCFReader(
                filename=self.temp_vcf.name,
                popmapfile=self.temp_popmap.name,
                chunk_size=3,
                verbose=False,
                debug=False,
                store_format_fields=store_format_fields,
            )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".vcf"
            ) as temp_output_vcf:
                reader.write_vcf(temp_output_vcf.name, chunk_size=2)

                with open(temp_output_vcf.name, "r") as f:
                    output_lines = [
                        line.strip() for line in f if not line.startswith("##")
                    ]

                # Check header line and FORMAT structure
                header_fields = output_lines[0].split("\t")
                self.assertEqual(
                    header_fields[:9],
                    [
                        "#CHROM",
                        "POS",
                        "ID",
                        "REF",
                        "ALT",
                        "QUAL",
                        "FILTER",
                        "INFO",
                        "FORMAT",
                    ],
                )

                for line in output_lines[1:]:
                    fields = line.split("\t")
                    format_field = fields[8]  # FORMAT column

                    if store_format_fields:
                        self.assertIn(":", format_field)
                        self.assertGreaterEqual(len(format_field.split(":")), 2)
                    else:
                        self.assertEqual(format_field, "GT")

                Path(temp_output_vcf.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
