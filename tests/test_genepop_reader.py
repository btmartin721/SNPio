import tempfile
import unittest
from pathlib import Path

from snpio.io.genepop_reader import GenePopReader


class TestGenePopReader(unittest.TestCase):
    def setUp(self):
        self.genepop_data_multiline = (
            "GenePop example file\n"
            "Locus1\n"
            "Locus2\n"
            "Locus3\n"
            "Pop\n"
            "Sample1 , 0101 0202 0304\n"
            "Sample2 , 0102 0303 0404\n"
            "Pop\n"
            "Sample3 , 0000 9999 0101\n"
        )

        self.genepop_data_oneline = (
            "GenePop example file\n"
            "Locus1, Locus2, Locus3\n"
            "Pop\n"
            "Sample1 , 001001 002002 003004\n"
            "Sample2 , 001002 003003 004004\n"
            "Pop\n"
            "Sample3 , 000000 999999 001001\n"
        )

        self.expected_snp_data_multiline = [
            ["A", "C", "K"],
            ["M", "G", "T"],
            ["N", "N", "A"],
        ]

        self.expected_snp_data_oneline = [
            ["A", "C", "K"],
            ["M", "G", "T"],
            ["N", "N", "A"],
        ]

        self.allele_encoding = {
            "01": "A",
            "02": "C",
            "03": "G",
            "04": "T",
            "001": "A",
            "002": "C",
            "003": "G",
            "004": "T",
        }

        self.popmap_content = [
            "Sample1\tPop1",
            "Sample2\tPop1",
            "Sample3\tPop2",
        ]
        self.popmap_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".popmap"
        )
        with open(self.popmap_file.name, "w") as f:
            f.write("\n".join(self.popmap_content))
        self.popmap_file.close()

    def tearDown(self):
        for temp_file in getattr(self, "temp_files", []):
            Path(temp_file).unlink(missing_ok=True)
        Path(self.popmap_file.name).unlink(missing_ok=True)

        for f in Path(".").glob("tmp*.nremover.vcf.gz"):
            f.unlink(missing_ok=True)

        for f in Path(".").glob("tmp*.nremover.vcf.gz.tbi"):
            f.unlink(missing_ok=True)

    def _write_temp_genepop(self, content):
        tf = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".gen")
        tf.write(content)
        tf.close()
        self.temp_files = getattr(self, "temp_files", [])
        self.temp_files.append(tf.name)
        return tf.name

    def test_genepop_parsing_multiline(self):
        filename = self._write_temp_genepop(self.genepop_data_multiline)
        reader = GenePopReader(
            filename=filename,
            popmapfile=self.popmap_file.name,
            allele_encoding=self.allele_encoding,
        )

        snp_data = reader.snp_data.tolist()
        self.assertEqual(snp_data, self.expected_snp_data_multiline)
        self.assertEqual(reader.num_snps, 3)
        self.assertEqual(reader.num_inds, 3)
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, ["Pop1", "Pop1", "Pop2"])
        self.assertEqual(reader.marker_names, ["Locus1", "Locus2", "Locus3"])
        self.assertEqual(
            reader.popmap,
            {
                "Sample1": "Pop1",
                "Sample2": "Pop1",
                "Sample3": "Pop2",
            },
        )

    def test_genepop_parsing_oneline(self):
        filename = self._write_temp_genepop(self.genepop_data_oneline)
        reader = GenePopReader(
            filename=filename,
            popmapfile=self.popmap_file.name,
            allele_encoding=self.allele_encoding,
        )

        snp_data = reader.snp_data.tolist()
        self.assertEqual(snp_data, self.expected_snp_data_oneline)
        self.assertEqual(reader.num_snps, 3)
        self.assertEqual(reader.num_inds, 3)
        self.assertEqual(reader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(reader.populations, ["Pop1", "Pop1", "Pop2"])
        self.assertEqual(reader.marker_names, ["Locus1", "Locus2", "Locus3"])
        self.assertEqual(
            reader.popmap,
            {
                "Sample1": "Pop1",
                "Sample2": "Pop1",
                "Sample3": "Pop2",
            },
        )

    def test_write_and_reparse_genepop(self):
        filename = self._write_temp_genepop(self.genepop_data_multiline)
        reader = GenePopReader(
            filename=filename,
            popmapfile=self.popmap_file.name,
            allele_encoding=self.allele_encoding,
        )

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gen").name
        self.temp_files = getattr(self, "temp_files", [])
        self.temp_files.append(output_file)

        reader.write_genepop(
            output_file=output_file, genotype_data=reader, title="Rewritten GenePop"
        )

        rereader = GenePopReader(
            filename=output_file,
            popmapfile=self.popmap_file.name,
            allele_encoding=self.allele_encoding,
        )

        reread_snp = rereader.snp_data.tolist()
        self.assertEqual(reread_snp, self.expected_snp_data_multiline)
        self.assertEqual(rereader.samples, ["Sample1", "Sample2", "Sample3"])
        self.assertEqual(rereader.marker_names, ["Locus1", "Locus2", "Locus3"])

    def test_interoperability_with_other_formats(self):
        filename = self._write_temp_genepop(self.genepop_data_multiline)
        reader = GenePopReader(
            filename=filename,
            popmapfile=self.popmap_file.name,
            allele_encoding=self.allele_encoding,
        )

        self.temp_files = getattr(self, "temp_files", [])

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Structure
            str_path = outdir / "interop.str"  # does NOT exist yet
            reader.write_structure(str(str_path))
            self.assertTrue(str_path.is_file(), f".str not written at {str_path}")
            self.temp_files.append(str(str_path))

            # Phylip
            phy_path = outdir / "interop.phy"
            reader.write_phylip(str(phy_path))
            self.assertTrue(phy_path.is_file(), f".phy not written at {phy_path}")
            self.temp_files.append(str(phy_path))

            # VCF file.
            vcfgz_path = outdir / "interop.vcf.gz"
            reader.write_vcf(vcfgz_path)

            # Assert exactly what you asked for
            self.assertTrue(
                vcfgz_path.is_file(), f".vcf.gz not written at {vcfgz_path}"
            )
            self.temp_files.append(vcfgz_path)

            self.assertTrue(
                (vcfgz_path.parent / (vcfgz_path.name + ".tbi")).is_file(),
                f".vcf.gz.tbi not written at {vcfgz_path}.tbi",
            )


if __name__ == "__main__":
    unittest.main()
