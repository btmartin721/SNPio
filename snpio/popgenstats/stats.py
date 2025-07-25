from typing import Literal, Tuple

import numpy as np
from numba import njit
from scipy.stats import norm


@njit(inline="always")
def derived_freq(
    geno: np.ndarray,
    inds: np.ndarray,
    snp: int,
    anc: int,
    het_mode: int,  # 0 = ignore, 1 = include, 2 = random
) -> float:
    """Diploid derived-allele frequency (0-1); returns -1 if all calls are missing.

    This method calculates the frequency of derived alleles at a given SNP position across a set of individuals. It handles missing data and heterozygous sites according to the specified `het_mode`.

    Args:
        geno: Genotype array (shape: n_samples x n_snps).
        inds: Array of individual indices to consider.
        snp: SNP position (column index in `geno`).
        anc: Ancestral allele code (0-3) or -1 if missing.
        het_mode: Heterozygous site handling mode (0-2). If 0, heterozygous calls are ignored. If 1, they contribute +1 derived allele and +1 ancestral allele. If 2, they contribute either 0 or 2 derived alleles randomly.

    Returns:
        float: Derived allele frequency (0-1) or -1 if all calls are missing.
    """
    tot, derived = 0.0, 0.0

    for idx in inds:
        g = geno[idx, snp]

        if g == -1:  # missing
            continue

        if g == -2:  # heterozygous / ambiguous
            if het_mode == 0:  # ignore
                continue
            tot += 2.0

            if het_mode == 1:  # include = +1 derived
                if anc != -1:
                    derived += 1.0
            else:  # het_mode == 2 → random
                # Bernoulli(0.5): add either 0 or 2 derived alleles
                if np.random.rand() < 0.5 and anc != -1:
                    derived += 2.0
            continue

        # unambiguous A,C,G,T
        tot += 2.0
        if g != anc:
            derived += 2.0

    return -1.0 if tot == 0 else derived / tot


# @njit
def calc_partitioned_d_freq(
    geno: np.ndarray,
    p1_inds: np.ndarray,
    p2_inds: np.ndarray,
    p3a_inds: np.ndarray,
    p3b_inds: np.ndarray,
    out_inds: np.ndarray,
    snp_indices: np.ndarray,
    het_mode: int = 2,
) -> Tuple[float, float, float]:
    """Compute D1, D2, D12 using derived allele frequencies (Comp-D, Pease & Hahn, 2015).

    Args:
        geno: 2D array of encoded genotypes (n_samples x n_snps).
        p1_inds, p2_inds, p3a_inds, p3b_inds, out_inds: arrays of indices for each population.
        snp_indices: array of SNP indices to use.
        het_mode: how to handle heterozygotes (default: random).

    Returns:
        Tuple[float, float, float]: D1, D2, D12 statistics.
    """
    ABBAA = 0.0
    BABAA = 0.0
    ABABA = 0.0
    BAABA = 0.0
    ABBBA = 0.0
    BABBA = 0.0

    for snp in snp_indices:
        # ancestral allele = first non-missing out-group call
        anc = -1
        for io in out_inds:
            g = geno[io, snp]
            if g != -1:
                anc = g
                break
        if anc < 0:
            continue

        # Calculate derived allele frequencies for each population
        p1 = derived_freq(geno, p1_inds, snp, anc, het_mode)
        p2 = derived_freq(geno, p2_inds, snp, anc, het_mode)
        p3a = derived_freq(geno, p3a_inds, snp, anc, het_mode)
        p3b = derived_freq(geno, p3b_inds, snp, anc, het_mode)
        out = derived_freq(geno, out_inds, snp, anc, het_mode)

        if (p1 < 0) or (p2 < 0) or (p3a < 0) or (p3b < 0):
            continue

        # Comp-D pattern weights (see Pease & Hahn 2015, Table 1)
        # D1: ABBAA vs BABAA
        ABBAA += (1 - p1) * p2 * p3a * (1 - p3b) * (1 - out)
        BABAA += p1 * (1 - p2) * p3a * (1 - p3b) * (1 - out)

        # D2: ABABA vs BAABA
        ABABA += (1 - p1) * p2 * (1 - p3a) * p3b * (1 - out)
        BAABA += p1 * (1 - p2) * (1 - p3a) * p3b * (1 - out)
        # D12: ABBBA vs BABBA
        ABBBA += (1 - p1) * p2 * p3a * p3b * (1 - out)
        BABBA += p1 * (1 - p2) * p3a * p3b * (1 - out)

    D1 = (ABBAA - BABAA) / (ABBAA + BABAA) if (ABBAA + BABAA) > 0 else 0.0
    D2 = (ABABA - BAABA) / (ABABA + BAABA) if (ABABA + BAABA) > 0 else 0.0
    D12 = (ABBBA - BABBA) / (ABBBA + BABBA) if (ABBBA + BABBA) > 0 else 0.0

    return D1, D2, D12


class Stats:
    def chisq_2pattern(self, a, b):
        """Chi-squared test for two patterns (Comp-D style).
        Returns chi-squared statistic and p-value.
        """
        from scipy.stats import chi2

        total = a + b
        if total == 0:
            return float("nan"), float("nan")
        expected = total / 2
        stat = ((a - expected) ** 2 + (b - expected) ** 2) / expected
        p = chi2.sf(stat, df=1)
        return stat, p

    """Class to summarize results from population genetic statistics.

    This class provides methods to calculate and summarize population genetic statistics, such as Patterson's D-statistic and partitioned D-statistics. It includes functionality for calculating z-scores and p-values based on bootstrap samples, and it logs warnings when bootstrap samples are empty or contain no finite values.
    """

    def dstat_zscores_pvalues(
        self,
        D_obs: float | tuple[float, ...],
        boots: np.ndarray,
        use_jackknife: bool,
        pattern_counts: dict = None,
        use_chisq: bool = False,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Compute Z-scores and P-values for Patterson's D, Partitioned D, or D-FOIL.

        Args:
            D_obs (float | tuple): Observed D value(s): scalar or length 3 or 4.
            boots (np.ndarray): 1D (for scalar D) or 2D array (for multivariate D).
            use_jackknife (bool): Whether jackknife is used. If True, uses jackknife variance; otherwise, uses bootstrapped standard deviation.

        Returns:
            tuple[tuple[float, ...], tuple[float, ...]]: Z-scores and P-values for each D statistic.
        """
        # Convert single-value case into consistent tuple form
        if isinstance(D_obs, float) or isinstance(D_obs, np.floating):
            D_obs = (D_obs,)
            boots = np.asarray(boots).reshape(-1, 1)
        else:
            # If boots is a list of arrays (e.g., [array([d1,d2,d3]), ...]), stack vertically
            if isinstance(boots, list):
                # Force every bootstrap replicate to be a flat list
                flat_boots = []
                for b in boots:
                    if isinstance(b, np.ndarray):
                        flat_boots.append(b.tolist())
                    elif isinstance(b, (list, tuple)):
                        flat_boots.append(list(b))
                    else:
                        flat_boots.append([b])
                boots = np.array(flat_boots)
            else:
                boots = np.asarray(boots)
            # If boots is 1D but D_obs is tuple, reshape
            if boots.ndim == 1 and len(D_obs) > 1:
                boots = boots.reshape(-1, len(D_obs))
            # If boots is transposed (n_stats, n_boots), fix it
            if (
                boots.ndim == 2
                and boots.shape[0] == len(D_obs)
                and boots.shape[1] != len(D_obs)
            ):
                boots = boots.T
            # If boots is not 2D with correct second dim, raise error
            if boots.ndim != 2 or boots.shape[1] != len(D_obs):
                raise ValueError(
                    f"Bootstraps shape {boots.shape} does not match D_obs length {len(D_obs)}. Each bootstrap should return a tuple/list of D statistics."
                )

        z_scores, p_values = [], []

        for i in range(len(D_obs)):
            x = boots[:, i]
            x = x[np.isfinite(x)]  # Filter out non-finite values

            if len(x) == 0:
                z_scores.append(np.nan)
                p_values.append(np.nan)
                continue

            if use_chisq and pattern_counts is not None:
                # Use chi-squared test for partitioned D
                stat = None
                if i == 0:
                    # D1: ABBAA vs BABAA
                    a, b = pattern_counts.get("ABBAA", 0), pattern_counts.get(
                        "BABAA", 0
                    )
                elif i == 1:
                    # D2: ABABA vs BAABA
                    a, b = pattern_counts.get("ABABA", 0), pattern_counts.get(
                        "BAABA", 0
                    )
                elif i == 2:
                    # D12: ABBBA vs BABBA
                    a, b = pattern_counts.get("ABBBA", 0), pattern_counts.get(
                        "BABBA", 0
                    )
                else:
                    a, b = 0, 0
                stat, p = self.chisq_2pattern(a, b)
                z_scores.append(stat)
                p_values.append(p)

                print(
                    f"[DIAGNOSTIC] Stat {i}: D_obs={D_obs[i]}, Z={stat}, P={self.chisq_2pattern(a, b)})"
                )
                continue

            if use_jackknife:
                n = len(x)
                mean_jk = np.mean(x)
                jk_var = (n - 1) / n * np.sum((x - mean_jk) ** 2)
                std = np.sqrt(jk_var)
                z = (D_obs[i] - mean_jk) / std if std > 0 else 0.0
                z_scores.append(z)
                p_values.append(self.pval_from_zscore(z, two_tailed=True))
                print(
                    f"[DIAGNOSTIC] Stat {i}: D_obs={D_obs[i]}, mean_jk={mean_jk}, std={std}, Z={z}, P={self.pval_from_zscore(z, two_tailed=True)}"
                )
            else:
                std = np.std(x)  # Sample standard deviation
                z = (D_obs[i] - np.mean(x)) / std if std > 0 else 0.0
                p = self.pval_from_zscore(z, two_tailed=True)
                z_scores.append(z)
                p_values.append(p)
                print(
                    f"[DIAGNOSTIC] Stat {i}: D_obs={D_obs[i]}, mean_boot={np.mean(x)}, std={std}, Z={z}, P={p}"
                )

        return tuple(z_scores), tuple(p_values)

    def pval_from_zscore(
        self,
        z: float,
        two_tailed: bool = True,
        tail: Literal["upper", "lower"] = "upper",
    ) -> float:
        """Estimate the two-tailed p-value from a z-score.

        Args:
            z (float): The z-score to convert to a p-value.
            two_tailed (bool): Whether to calculate a two-tailed p-value. Defaults to True.
            tail (Literal["upper", "lower"]): The tail direction for one-tailed p-value. Defaults to "upper".

        Returns:
            float: The one-tailed or two-tailed p-value. If the z-score is NaN, returns NaN.
        """
        if np.isnan(z):
            return 1.0  # Return 1.0 for invalid z-scores

        if two_tailed:
            return 2 * (1 - norm.cdf(np.abs(z)))
        elif tail == "upper":
            return 1 - norm.cdf(z)
        elif tail == "lower":
            return norm.cdf(z)
        else:
            raise ValueError(
                f"Invalid tail option: {tail}. Use 'upper' or 'lower' for one-tailed p-value."
            )
