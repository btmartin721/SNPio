import unittest
import numpy as np
from snpio.popgenstats.dstat import PattersonDStats  # Adjust as needed

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
        pds = PattersonDStats(gd)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=0)
        self.assertAlmostEqual(results["D"][0], 1.0, places=1)

    def test_negative_d(self):
        arr = self._build_perfect_baba()
        gd = DummyGenotypeData(arr)
        pds = PattersonDStats(gd)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=1)
        self.assertAlmostEqual(results["D"][0], -1.0, places=1)

    def test_zero_d(self):
        abba = self._build_perfect_abba()
        baba = self._build_perfect_baba()
        arr = np.hstack([abba[:, :50], baba[:, :50]])  # 50 ABBA, 50 BABA
        gd = DummyGenotypeData(arr)
        pds = PattersonDStats(gd)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=2)
        self.assertAlmostEqual(results["D"][0], 0.0, places=1)

    def test_missing_data(self):
        arr = self._apply_missing(self._build_perfect_abba(), frac=0.2)
        gd = DummyGenotypeData(arr)
        pds = PattersonDStats(gd)
        results, _ = pds.calculate(*self.pops, n_boot=200, seed=3)
        self.assertGreaterEqual(results["D"][0], 0.7)  # still strongly positive


if __name__ == "__main__":
    unittest.main()
