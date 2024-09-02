import unittest
import numpy as np

from snpio.io.vcf_reader import VCFReader
from snpio.utils.custom_exceptions import NoValidAllelesError


class TestCalculateRefAltAlleles(unittest.TestCase):

    def setUp(self):
        # Simulated reverse IUPAC mapping (example)
        self.reverse_iupac_mapping = {
            "R": ("A", "G"),
            "Y": ("C", "T"),
            "S": ("G", "C"),
            "W": ("A", "T"),
            "K": ("G", "T"),
            "M": ("A", "C"),
        }

    def assert_in_any(self, actual, expected_list):
        """Helper method to check that actual is in one of the expected possibilities."""
        self.assertTrue(
            any(np.array_equal(actual, expected) for expected in expected_list),
            f"{actual} is not in any of the expected results: {expected_list}",
        )

    def test_simple_case(self):
        snp_data = np.array(
            [["A", "G", "R"], ["T", "Y", "C"], ["C", "T", "Y"], ["G", "R", "A"]]
        )
        ref_alleles, alt_alleles, other_alleles = VCFReader.calculate_ref_alt_alleles(
            snp_data, self.reverse_iupac_mapping, random_seed=42
        )
        expected_ref_possibilities = [
            np.array(["A", "T", "C", "G"]),
            np.array(["G", "T", "C", "G"]),
            np.array(["A", "C", "C", "A"]),
            np.array(["G", "C", "C", "A"]),
        ]
        expected_alt_possibilities = [
            np.array(["G", "C", "T", "A"]),
            np.array(["A", "C", "T", "A"]),
            np.array(["G", "T", "T", "G"]),
            np.array(["A", "T", "T", "G"]),
        ]
        expected_other = [None, None, None, None]

        self.assert_in_any(ref_alleles, expected_ref_possibilities)
        self.assert_in_any(alt_alleles, expected_alt_possibilities)
        self.assertEqual(other_alleles, expected_other)

    def test_multi_allelic_case(self):
        snp_data = np.array(
            [["R", "Y", "S"], ["R", "Y", "W"], ["M", "K", "S"], ["M", "K", "W"]]
        )
        ref_alleles, alt_alleles, other_alleles = VCFReader.calculate_ref_alt_alleles(
            snp_data, self.reverse_iupac_mapping, random_seed=42
        )
        expected_ref_possibilities = [
            np.array(["A", "C", "G", "A"]),
            np.array(["G", "T", "C", "A"]),
            np.array(["C", "A", "C", "A"]),
        ]
        expected_alt_possibilities = [
            np.array(["G", "T", "C", "T"]),
            np.array(["A", "C", "G", "T"]),
            np.array(["G", "T", "A", "T"]),
            np.array(["G", "T", "G", "T"]),
        ]

        expected_other = [["A", "T"], ["C", "G"], ["A", "T"], ["C", "G"]]

        self.assert_in_any(ref_alleles, expected_ref_possibilities)
        self.assert_in_any(alt_alleles, expected_alt_possibilities)
        self.assertEqual(other_alleles, expected_other)

    def test_no_valid_alleles(self):
        snp_data = np.array([["N", "-", "."], [".", "-", "N"]])
        with self.assertRaises(NoValidAllelesError):
            VCFReader.calculate_ref_alt_alleles(
                snp_data, self.reverse_iupac_mapping, random_seed=42
            )


if __name__ == "__main__":
    unittest.main()
