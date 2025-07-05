import logging
from typing import List, Tuple

import numpy as np
from numba import njit, prange
from scipy.stats import norm
from tqdm import tqdm


class DStatistics:
    """Class to calculate D-statistics (Patterson's D, Partitioned D, and D-FOIL) with bootstrap support.

    This class provides methods to compute various D-statistics using genotype data, including Patterson's D, Partitioned D, and D-FOIL. It supports bootstrapping for statistical inference and can handle missing data represented as 'N' or '.' in genotype strings. The class uses Numba for efficient computation, especially for large datasets.
    """

    def __init__(
        self,
        alignment: np.ndarray,
        sample_ids: List[str],
        logger: logging.Logger,
    ) -> None:
        """
        Args:
            alignment: 2D array of genotype strings (shape: n_samples x n_snps).
            sample_ids: List of sample identifiers corresponding to alignment rows.
            logger: Logger for diagnostic messages.
        """
        self.alignment = alignment
        self.sample_ids = sample_ids
        self.logger = logger
        # Allele mapping: A,C,G,T -> 0-3; missing -> -1
        self._allele_code = {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1, ".": -1}

    def run_patterson_d(self, geno, d1, d2, d3, out, snp_idx):
        """Run Patterson's D-statistic bootstrap and return statistics.

        Args:
            geno (np.ndarray): Genotype data as a 2D array (individuals x SNPs).
            d1 (List[int]): Indices of individuals in population 1.
            d2 (List[int]): Indices of individuals in population 2.
            d3 (List[int]): Indices of individuals in population 3.
            out (List[int]): Indices of individuals in the outgroup population.
            snp_idx (np.ndarray): 2D array of SNP indices for bootstrapping.

        Returns:
            Tuple[float, float, float]: Mean D-statistic, z-score, and p-value.
        """
        boots = self._patterson_d_bootstrap(
            geno, np.array(d1), np.array(d2), np.array(d3), np.array(out), snp_idx
        )
        return self._dstat_z_and_p(boots)  # (mean, z, p)

    def run_part_d(self, geno, d1, d2, d3, d4, out, snp_idx):
        """Run Partitioned D-statistic bootstrap and return statistics.

        Args:
            geno (np.ndarray): Genotype data as a 2D array (individuals x SNPs).
            d1 (List[int]): Indices of individuals in population 1.
            d2 (List[int]): Indices of individuals in population 2.
            d3 (List[int]): Indices of individuals in population 3.
            d4 (List[int]): Indices of individuals in population 4.
            out (List[int]): Indices of individuals in the outgroup population.
            snp_idx (np.ndarray): 2D array of SNP indices for bootstrapping.

        Returns:
            Tuple[float, float, float]: Mean D-statistic, z-score, and p-value.
        """
        boots = self._partitioned_d_bootstrap(
            geno,
            np.array(d1),
            np.array(d2),
            np.array(d3),
            np.array(d4),
            np.array(out),
            snp_idx,
        )
        return self._dstat_z_and_p(boots)  # (mean, z, p)

    def run_dfoil(self, geno, d1, d2, d3, d4, out, snp_idx):
        """Run DFOIL bootstrap and return statistics.

        Args:
            geno (np.ndarray): Genotype data as a 2D array (individuals x SNPs).
            d1 (List[int]): Indices of individuals in population 1.
            d2 (List[int]): Indices of individuals in population 2.
            d3 (List[int]): Indices of individuals in population 3.
            d4 (List[int]): Indices of individuals in population 4.
            out (List[int]): Indices of individuals in the outgroup population.
            snp_idx (np.ndarray): 2D array of SNP indices for bootstrapping.

        Returns:
            List[Tuple[float, float, float]]: Statistics for DFO, DFI, DOL, DIL.
            Each tuple contains (mean, z-score, p-value).
        """
        boots = self._dfoil_bootstrap(
            geno,
            np.array(d1),
            np.array(d2),
            np.array(d3),
            np.array(d4),
            np.array(out),
            snp_idx,
        )

        # Returns [(mean, z, p) for DFO, DFI, DOL, DIL]
        return self._dfoil_z_and_p(boots)

    @staticmethod
    @njit(parallel=True)
    def _patterson_d_bootstrap(
        geno: np.ndarray,
        d1_inds: np.ndarray,
        d2_inds: np.ndarray,
        d3_inds: np.ndarray,
        out_inds: np.ndarray,
        snp_idx: np.ndarray,
    ) -> np.ndarray:
        n_boot, n_snps = snp_idx.shape
        boots = np.full(n_boot, np.nan)
        for b in prange(n_boot):
            abba = 0.0
            baba = 0.0
            for j in range(n_snps):
                idx = snp_idx[b, j]
                g1 = geno[d1_inds[np.random.randint(0, d1_inds.shape[0])], idx]
                g2 = geno[d2_inds[np.random.randint(0, d2_inds.shape[0])], idx]
                g3 = geno[d3_inds[np.random.randint(0, d3_inds.shape[0])], idx]
                go = geno[out_inds[np.random.randint(0, out_inds.shape[0])], idx]
                if g1 < 0 or g2 < 0 or g3 < 0 or go < 0:
                    continue
                if g1 != go and g2 == go and g3 == go:
                    abba += 1
                elif g1 == go and g2 != go and g3 == go:
                    baba += 1
            tot = abba + baba
            if tot > 0:
                boots[b] = (abba - baba) / tot
        return boots

    @staticmethod
    @njit(parallel=True)
    def _partitioned_d_bootstrap(
        geno: np.ndarray,
        d1_inds: np.ndarray,
        d2_inds: np.ndarray,
        d3_inds: np.ndarray,
        d4_inds: np.ndarray,
        out_inds: np.ndarray,
        snp_idx: np.ndarray,
    ) -> np.ndarray:
        n_boot, n_snps = snp_idx.shape
        boots = np.full(n_boot, np.nan)
        for b in prange(n_boot):
            abba = 0.0
            baba = 0.0
            for j in range(n_snps):
                idx = snp_idx[b, j]
                g1 = geno[d1_inds[np.random.randint(0, d1_inds.shape[0])], idx]
                g2 = geno[d2_inds[np.random.randint(0, d2_inds.shape[0])], idx]
                g3 = geno[d3_inds[np.random.randint(0, d3_inds.shape[0])], idx]
                g4 = geno[d4_inds[np.random.randint(0, d4_inds.shape[0])], idx]
                go = geno[out_inds[np.random.randint(0, out_inds.shape[0])], idx]
                if g1 < 0 or g2 < 0 or g3 < 0 or g4 < 0 or go < 0:
                    continue
                if g1 != go and g2 == go and g3 == go and g4 == go:
                    abba += 1
                elif g1 == go and g2 != go and g3 == go and g4 == go:
                    baba += 1
            tot = abba + baba
            if tot > 0:
                boots[b] = (abba - baba) / tot
        return boots

    @staticmethod
    @njit(parallel=True)
    def _dfoil_bootstrap(
        geno: np.ndarray,
        d1_inds: np.ndarray,
        d2_inds: np.ndarray,
        d3_inds: np.ndarray,
        d4_inds: np.ndarray,
        out_inds: np.ndarray,
        snp_idx: np.ndarray,
    ) -> np.ndarray:
        n_boot, n_snps = snp_idx.shape
        boots = np.full((n_boot, 4), np.nan)
        for b in prange(n_boot):
            counts = np.zeros(4)
            dens = np.zeros(4)
            for j in range(n_snps):
                idx = snp_idx[b, j]
                g1 = geno[d1_inds[np.random.randint(0, d1_inds.shape[0])], idx]
                g2 = geno[d2_inds[np.random.randint(0, d2_inds.shape[0])], idx]
                g3 = geno[d3_inds[np.random.randint(0, d3_inds.shape[0])], idx]
                g4 = geno[d4_inds[np.random.randint(0, d4_inds.shape[0])], idx]
                go = geno[out_inds[np.random.randint(0, out_inds.shape[0])], idx]
                if g1 < 0 or g2 < 0 or g3 < 0 or g4 < 0 or go < 0:
                    continue
                pat = "".join(["A" if x == go else "B" for x in (g1, g2, g3, g4, go)])
                # DFO
                if pat in ("AABBA", "AABAA"):
                    counts[0] += 1
                    dens[0] += 1
                elif pat in ("ABBAA", "ABABA"):
                    counts[0] -= 1
                    dens[0] += 1
                # DFI
                if pat in ("BAABA", "BAAAA"):
                    counts[1] += 1
                    dens[1] += 1
                elif pat in ("BBAAA", "BABAA"):
                    counts[1] -= 1
                    dens[1] += 1
                # DOL
                if pat in ("AABBA", "BAABA"):
                    counts[2] += 1
                    dens[2] += 1
                elif pat in ("ABBAA", "BBAAA"):
                    counts[2] -= 1
                    dens[2] += 1
                # DIL
                if pat in ("AABAA", "BAAAA"):
                    counts[3] += 1
                    dens[3] += 1
                elif pat in ("ABABA", "BABAA"):
                    counts[3] -= 1
                    dens[3] += 1
            for i in range(4):
                if dens[i] > 0:
                    boots[b, i] = counts[i] / dens[i]
        return boots

    def _dstat_z_and_p(self, boot: np.ndarray) -> Tuple[float, float, float]:
        boot = boot[np.isfinite(boot)]
        if boot.size == 0:
            return np.nan, np.nan, np.nan
        mean = np.mean(boot)
        std = np.std(boot, ddof=1)
        z = mean / std if std > 0 else 0.0
        p = 2 * (1 - norm.cdf(abs(z)))
        return mean, z, p

    def _dfoil_z_and_p(self, boots: np.ndarray) -> List[Tuple[float, float, float]]:
        stats: List[Tuple[float, float, float]] = []
        for i in range(4):
            stats.append(self._dstat_z_and_p(boots[:, i]))
        return stats

    def _encode_alleles(self, geno_array: np.ndarray) -> np.ndarray:
        # full IUPAC mapping to possible bases
        iupac_map = {
            "A": ["A"],
            "C": ["C"],
            "G": ["G"],
            "T": ["T"],
            "R": ["A", "G"],
            "Y": ["C", "T"],
            "S": ["G", "C"],
            "W": ["A", "T"],
            "K": ["G", "T"],
            "M": ["A", "C"],
            "B": ["C", "G", "T"],
            "D": ["A", "G", "T"],
            "H": ["A", "C", "T"],
            "V": ["A", "C", "G"],
            "N": [],
            ".": [],
        }

        n_inds, n_snps = geno_array.shape
        encoded = np.full((n_inds, n_snps), -1, dtype=np.int8)

        for i in range(n_inds):
            for j in range(n_snps):
                code = geno_array[i, j].upper()
                alleles = iupac_map.get(code, [])
                if len(alleles) == 1:
                    # unambiguous
                    encoded[i, j] = self._allele_code.get(alleles[0], -1)
                elif len(alleles) > 1:
                    # pick one of the possibilities at random
                    choice = np.random.choice(alleles)
                    encoded[i, j] = self._allele_code.get(choice, -1)
                else:
                    # missing or unknown
                    encoded[i, j] = -1

        return encoded
