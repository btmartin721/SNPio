import unittest
import numpy as np
from snpio.popgenstats.dfoil import DfoilStats

MISSING = -9


class DummyGenotypeData:
    def __init__(self, arr: np.ndarray):
        self.genotypes_012 = arr
        self.prefix = ""
        self.popmap_inverse = {
            "P1": [f"Sample_{i}" for i in range(5)],
            "P2": [f"Sample_{i}" for i in range(5, 10)],
            "P3a": [f"Sample_{i}" for i in range(10, 15)],
            "P3b": [f"Sample_{i}" for i in range(15, 20)],
            "Out": [f"Sample_{i}" for i in range(20, 25)],
        }
        self.samples = [f"Sample_{i}" for i in range(25)]
        self.num_inds = arr.shape[0]
        self.num_snps = arr.shape[1]
        self.plot_format = "png"
        self.verbose = False
        self.debug = False

        arr = arr.astype(str)
        arr[arr == "0"] = "A"
        arr[arr == "1"] = "W"
        arr[arr == "2"] = "T"
        arr[arr == "-9"] = "N"
        self.snp_data = arr
        self.was_filtered = True


class TestDFOILStats(unittest.TestCase):
    def setUp(self):
        self.n_pops = 5
        self.n_ind = 5
        self.n_loci = 60
        self.pops = [
            np.arange(i * self.n_ind, (i + 1) * self.n_ind) for i in range(self.n_pops)
        ]
        self.rng = np.random.default_rng(42)

    def _build_pristine_strong_dfo(self):
        """Strong DFO-only signal → DFO=1, others ≈ 0"""
        arr = np.zeros((self.n_pops * self.n_ind, self.n_loci), int)
        for j in range(self.n_loci):
            # BABAA: P1=B, P2=A, P3a=B, P3b=A, Out=A
            arr[self.pops[0], j] = 2  # P1 = B
            arr[self.pops[1], j] = 0  # P2 = A
            arr[self.pops[2], j] = 2  # P3a = B
            arr[self.pops[3], j] = 0  # P3b = A
            arr[self.pops[4], j] = 0  # Out = A (ancestral)
        return arr

    def _build_pristine_balanced(self):
        """30 ABBAA + 30 BABAA → DFO ≈ 0"""
        arr = np.zeros((self.n_pops * self.n_ind, self.n_loci), int)
        for j in range(self.n_loci):
            if j < 30:
                arr[self.pops[0], j] = 0
                arr[self.pops[1], j] = 2
            else:
                arr[self.pops[0], j] = 2
                arr[self.pops[1], j] = 0
            arr[self.pops[2], j] = 2
            arr[self.pops[3], j] = 0
            arr[self.pops[4], j] = 0
        return arr

    def _apply_missing(self, arr, frac=0.2):
        mask = self.rng.random(arr.shape) < frac
        arr2 = arr.copy()
        arr2[mask] = MISSING
        return arr2

    def _apply_nonbiallelic(self, arr, n_non=10):
        arr2 = arr.copy()
        cols = self.rng.choice(self.n_loci, size=n_non, replace=False)
        for j in cols:
            arr2[:, j] = 1
        return arr2

    def _build_pristine_strong_dil(self):
        """Strong DIL-only signal → DIL=1, others ≈ 0"""
        arr = np.zeros((self.n_pops * self.n_ind, self.n_loci), int)
        for j in range(self.n_loci):
            if j % 4 == 0:
                # ABBAA → P1=0, P2=2, P3a=2, P3b=0, Out=0
                arr[self.pops[0], j] = 0
                arr[self.pops[1], j] = 2
                arr[self.pops[2], j] = 2
                arr[self.pops[3], j] = 0
                arr[self.pops[4], j] = 0
            elif j % 4 == 1:
                # BBBAA → P1=2, P2=2, P3a=2, P3b=0, Out=0
                arr[self.pops[0], j] = 2
                arr[self.pops[1], j] = 2
                arr[self.pops[2], j] = 2
                arr[self.pops[3], j] = 0
                arr[self.pops[4], j] = 0
            elif j % 4 == 2:
                # BAABA → P1=2, P2=0, P3a=0, P3b=2, Out=0
                arr[self.pops[0], j] = 2
                arr[self.pops[1], j] = 0
                arr[self.pops[2], j] = 0
                arr[self.pops[3], j] = 2
                arr[self.pops[4], j] = 0
            else:
                # AAABA → P1=0, P2=0, P3a=0, P3b=2, Out=0
                arr[self.pops[0], j] = 0
                arr[self.pops[1], j] = 0
                arr[self.pops[2], j] = 0
                arr[self.pops[3], j] = 2
                arr[self.pops[4], j] = 0
        return arr

    def _build_pristine_strong_dfi(self):
        """Strong DFI-only signal → DFI=1, others ≈ 0"""
        arr = np.zeros((self.n_pops * self.n_ind, self.n_loci), int)
        half = self.n_loci // 2
        for j in range(self.n_loci):
            if j < half:
                # BABBA → P1=2, P2=0, P3a=2, P3b=2, Out=0
                arr[self.pops[0], j] = 2
                arr[self.pops[1], j] = 0
                arr[self.pops[2], j] = 2
                arr[self.pops[3], j] = 2
                arr[self.pops[4], j] = 0
            else:
                # ABAAA → P1=0, P2=2, P3a=0, P3b=0, Out=0
                arr[self.pops[0], j] = 0
                arr[self.pops[1], j] = 2
                arr[self.pops[2], j] = 0
                arr[self.pops[3], j] = 0
                arr[self.pops[4], j] = 0
        return arr

    def test_dfoil_strong_signal(self):
        arr = self._build_pristine_strong_dfo()
        gd = DummyGenotypeData(arr)
        dfs = DfoilStats(gd)
        results, _ = dfs.calculate(*self.pops, n_boot=200, seed=1)

        self.assertAlmostEqual(results["DFO"], 1.0, places=1)
        self.assertAlmostEqual(results["DFI"], 1.0, places=1)
        self.assertAlmostEqual(results["DOL"], -1.0, places=1)
        self.assertAlmostEqual(results["DIL"], -1.0, places=1)

    def test_dfoil_balanced(self):
        arr = self._build_pristine_balanced()
        gd = DummyGenotypeData(arr)
        dfs = DfoilStats(gd)
        results, _ = dfs.calculate(*self.pops, n_boot=200, seed=2)
        self.assertAlmostEqual(results["DFO"], 0.0, places=1)
        self.assertAlmostEqual(results["DFI"], 0.0, places=1)
        self.assertAlmostEqual(results["DOL"], 0.0, places=1)
        self.assertAlmostEqual(results["DIL"], 0.0, places=1)

    def test_dfoil_missing_data(self):
        arr = self._apply_missing(self._build_pristine_balanced(), frac=0.2)
        gd = DummyGenotypeData(arr)
        dfs = DfoilStats(gd)
        results, _ = dfs.calculate(*self.pops, n_boot=200, seed=3)
        self.assertAlmostEqual(results["DFO"], 0.0, places=1)

    def test_dfoil_nonbiallelic(self):
        arr = self._apply_nonbiallelic(self._build_pristine_balanced(), n_non=10)
        gd = DummyGenotypeData(arr)
        dfs = DfoilStats(gd)
        results, _ = dfs.calculate(*self.pops, n_boot=200, seed=4)
        self.assertAlmostEqual(results["DFO"], 0.0, places=1)

    def test_dfoil_mixed_noise(self):
        arr = self._apply_missing(
            self._apply_nonbiallelic(self._build_pristine_balanced()), frac=0.2
        )
        gd = DummyGenotypeData(arr)
        dfs = DfoilStats(gd)
        results, _ = dfs.calculate(*self.pops, n_boot=200, seed=5)
        self.assertAlmostEqual(results["DFO"], 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
