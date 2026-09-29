import tempfile
import unittest
from pathlib import Path

import numpy as np
from snpio import NRemover2, VCFReader

# Ten diploid samples. Each locus has a known allele count, so the minor
# allele frequency (MAF) and minor allele count (MAC) can be checked exactly.
#   locus 1: 8 x 0/0, 2 x 0/1  -> alleles 18 A : 2 T   MAF 0.10  MAC 2
#   locus 2: 9 x 0/0, 1 x 1/1  -> alleles 18 G : 2 C   MAF 0.10  MAC 2
#   locus 3: 5 x 0/0, 5 x 0/1  -> alleles 15 C : 5 G   MAF 0.25  MAC 5
#   locus 4: 9 x 0/0, 1 x 0/1  -> alleles 19 T : 1 A   MAF 0.05  MAC 1
#   locus 5: 7 x 0/0, 2 x 1/1, 1 x ./. -> 14 A : 4 G   MAF 4/18  MAC 4
GENOTYPES = [
    ("A", "T", ["0/0"] * 8 + ["0/1"] * 2),
    ("G", "C", ["0/0"] * 9 + ["1/1"]),
    ("C", "G", ["0/0"] * 5 + ["0/1"] * 5),
    ("T", "A", ["0/0"] * 9 + ["0/1"]),
    ("A", "G", ["0/0"] * 7 + ["1/1"] * 2 + ["./."]),
]
EXPECTED_MAF = np.array([0.10, 0.10, 0.25, 0.05, 4 / 18])
EXPECTED_MAC = np.array([2, 2, 5, 1, 4])


class TestAlleleCounts(unittest.TestCase):
    """Homozygotes carry two copies of an allele, heterozygotes one of each."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmpdir.name)
        samples = [f"S{i}" for i in range(1, 11)]

        vcf = tmp / "counts.vcf"
        with open(vcf, "w") as fh:
            fh.write("##fileformat=VCFv4.2\n")
            fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            fh.write(
                "\t".join(
                    ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
                    + ["FORMAT"]
                    + samples
                )
                + "\n"
            )
            for i, (ref, alt, gts) in enumerate(GENOTYPES, start=1):
                row = ["chr1", str(i * 100), ".", ref, alt, ".", "PASS", ".", "GT"]
                fh.write("\t".join(row + gts) + "\n")

        popmap = tmp / "counts.popmap"
        popmap.write_text("".join(f"{s}\tpop1\n" for s in samples))

        cls.gd = VCFReader(
            filename=str(vcf),
            popmapfile=str(popmap),
            prefix=str(tmp / "counts"),
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def kept(self, nrm):
        return [i for i, keep in enumerate(nrm.resolve().loci_indices) if keep]

    def test_maf_proportions(self):
        nrm = NRemover2(self.gd)
        maf = nrm._filtering_methods._compute_maf_proportions()
        np.testing.assert_allclose(maf, EXPECTED_MAF)

    def test_filter_maf(self):
        # MAF >= 0.08 keeps loci 1, 2, 3 and 5; locus 1's minor allele is
        # only in heterozygotes, so it is lost if homozygotes are over-counted
        self.assertEqual(self.kept(NRemover2(self.gd).filter_maf(0.08)), [0, 1, 2, 4])

    def test_filter_mac(self):
        # MAC >= 3 keeps loci 3 and 5; locus 2's minor allele is in one
        # homozygote (MAC 2), so it is kept if homozygotes are over-counted
        self.assertEqual(self.kept(NRemover2(self.gd).filter_mac(3)), [2, 4])

    def test_filter_singletons(self):
        # Only locus 4's minor allele appears once
        self.assertEqual(self.kept(NRemover2(self.gd).filter_singletons()), [0, 1, 2, 4])


if __name__ == "__main__":
    unittest.main()
