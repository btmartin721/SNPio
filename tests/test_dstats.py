import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.popgenstats.d_statistics import DStatistics
from snpio.popgenstats.dstat import PattersonDStats

MISSING = -9


class DummyGenotypeData:
    """Stub minimal GenotypeData for Patterson's D tests."""

    def __init__(self, arr: np.ndarray):
        self.genotypes_012 = arr
        self.prefix = ""
        self.samples = [f"Sample_{i}" for i in range(arr.shape[0])]
        self.num_inds = arr.shape[0]
        self.num_snps = arr.shape[1]
        self.plot_format = "png"
        self.verbose = False
        self.debug = False
        self.ref = ["A"] * self.num_inds
        self.alt = ["T"] * self.num_inds

        arr = arr.astype(str)
        arr[arr == "0"] = "A"
        arr[arr == "1"] = "W"
        arr[arr == "2"] = "T"
        arr[arr == "-9"] = "N"
        self.snp_data = arr
        self.was_filtered = True
        self.from_vcf = False


def test_dstatistics_results_use_filtered_report_scope(tmp_path):
    analysis = object.__new__(DStatistics)
    analysis.genotype_data = SimpleNamespace(
        prefix=str(tmp_path / "dstat_case"),
        was_filtered=True,
    )

    assert analysis._make_results_dir() == (
        tmp_path
        / "dstat_case_output"
        / "reports"
        / "nremover"
        / "d_statistics"
    )


class TestPattersonD(unittest.TestCase):
    def setUp(self):
        self.n_ind = 5
        self.n_loci = 100
        self.rng = np.random.default_rng(42)

        # P1, P2, P3, OUT
        self.pops = [
            np.arange(0, 5),
            np.arange(5, 10),
            np.arange(10, 15),
            np.arange(15, 20),
        ]

        self.n_pops = 4

    def _build_perfect_abba(self):
        arr = np.zeros((20, self.n_loci), dtype=int)
        arr[self.pops[0], :] = 0  # P1 = ancestral
        arr[self.pops[1], :] = 2  # P2 = derived
        # P3 = mix of ancestral and derived
        arr[self.pops[2][:3], :] = 0
        arr[self.pops[2][3:], :] = 2
        arr[self.pops[3], :] = 0  # OUT = ancestral
        return arr

    def _build_perfect_baba(self):
        arr = np.zeros((self.n_pops * self.n_ind, self.n_loci), dtype=int)
        arr[self.pops[0], :] = 2  # P1
        arr[self.pops[1], :] = 0  # P2
        arr[self.pops[2][:3], :] = 0  # P3 partial ancestral
        arr[self.pops[2][3:], :] = 2  # P3 partial derived
        arr[self.pops[3], :] = 0  # OUT
        return arr

    def _apply_missing(self, arr, frac=0.2):
        mask = self.rng.random(arr.shape) < frac
        arr[mask] = MISSING
        return arr

    def test_positive_d(self):
        arr = self._build_perfect_abba()
        gd = DummyGenotypeData(arr)
        enc = GenotypeEncoder(gd)
        pds = PattersonDStats(gd, enc.genotypes_012)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=0)
        self.assertAlmostEqual(results["D"], 1.0, places=1)

    def test_negative_d(self):
        arr = self._build_perfect_baba()
        gd = DummyGenotypeData(arr)
        enc = GenotypeEncoder(gd)
        pds = PattersonDStats(gd, enc.genotypes_012)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=1)
        self.assertAlmostEqual(results["D"], -1.0, places=1)

    def test_zero_d(self):
        abba = self._build_perfect_abba()
        baba = self._build_perfect_baba()
        arr = np.hstack([abba[:, :50], baba[:, :50]])  # 50 ABBA, 50 BABA
        gd = DummyGenotypeData(arr)
        enc = GenotypeEncoder(gd)
        pds = PattersonDStats(gd, enc.genotypes_012)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=2)
        self.assertAlmostEqual(results["D"], 0.0, places=1)

    def test_missing_data(self):
        arr = self._apply_missing(self._build_perfect_abba(), frac=0.2)
        gd = DummyGenotypeData(arr)
        enc = GenotypeEncoder(gd)
        pds = PattersonDStats(gd, enc.genotypes_012)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=3)
        self.assertGreaterEqual(results["D"], 0.7)  # still strongly positive


class TestDStatisticIndividualSelection(unittest.TestCase):
    """Test D-statistic population sampling independently of the algorithms."""

    def setUp(self):
        samples = ["A", "B", "C", "D", "E", "F"]
        popmap_inverse = {
            "P1": ["D", "C", "B", "A"],
            "P2": ["E", "F"],
        }
        geno012 = np.array(
            [
                [0, 0, 0, 0],
                [0, MISSING, 0, 0],
                [MISSING, 0, 0, 0],
                [MISSING, 0, MISSING, MISSING],
                [0, 0, 0, 0],
                [MISSING, MISSING, 0, 0],
            ],
            dtype=int,
        )

        self.selector = DStatistics.__new__(DStatistics)
        self.selector.genotype_data = SimpleNamespace(
            samples=samples,
            popmap_inverse=popmap_inverse,
        )
        self.selector.geno012 = geno012
        self.selector.logger = Mock()

    def test_least_missing_selects_most_complete_with_stable_ties(self):
        selected = self.selector._get_single_pop_indices(
            "P1",
            max_individuals_per_pop=3,
            individual_selection="least_missing",
            seed=999,
        )

        self.assertEqual(selected, [0, 1, 2])

    def test_least_missing_is_seed_independent(self):
        selected_a = self.selector._get_single_pop_indices(
            "P1", 2, "least_missing", seed=1
        )
        selected_b = self.selector._get_single_pop_indices(
            "P1", 2, "least_missing", seed=9876
        )

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(selected_a, [0, 1])

    def test_all_selection_ignores_population_cap(self):
        selected = self.selector._get_single_pop_indices(
            "P1",
            max_individuals_per_pop=1,
            individual_selection="all",
            seed=None,
        )

        self.assertEqual(selected, [3, 2, 1, 0])

    def test_explicit_selection_is_honored_without_population_cap(self):
        selected = self.selector._get_single_pop_indices(
            "P1",
            max_individuals_per_pop=None,
            individual_selection={"P1": ["C", "A"]},
            seed=None,
        )

        self.assertEqual(selected, [2, 0])

    def test_invalid_strategy_is_rejected_without_population_cap(self):
        with self.assertRaisesRegex(ValueError, "Invalid individual_selection"):
            self.selector._get_single_pop_indices(
                "P2",
                max_individuals_per_pop=None,
                individual_selection="unknown",  # type: ignore[arg-type]
                seed=None,
            )

    def test_nonpositive_population_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.selector._get_single_pop_indices(
                "P1",
                max_individuals_per_pop=0,
                individual_selection="least_missing",
                seed=None,
            )


if __name__ == "__main__":
    unittest.main()
