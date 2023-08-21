
import unittest
from snpio import GenotypeData

class TestGenotypeData(unittest.TestCase):

    def test_filename_options(self):
        filenames = [
            "example_data/vcf_files/phylogen.vcf.gz",
            "example_data/phylip_files/phylogen_nomx.u.snps.phy",
            "example_data/structure_files/phylogen_nomx.ustr"
        ]

        for filename in filenames:
            gd = GenotypeData(filename=filename)
            self.assertIsNotNone(gd)

if __name__ == "__main__":
    unittest.main()
