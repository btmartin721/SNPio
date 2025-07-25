from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(inline="always")
def _safe_ratio(num: float, den: float, ret_val: float = 0.0) -> float:
    """Compute a safe ratio avoiding division by zero.

    This method returns `ret_val` if the denominator is zero, otherwise it returns the ratio of `num` to `den`.

    Args:
        num (float): Numerator of the ratio.
        den (float): Denominator of the ratio.
        ret_val (float): Value to return if the denominator is zero (default is 0.0).

    Returns:
        float: The computed ratio or `ret_val` if the denominator is zero.
    """
    return num / den if np.abs(den) > 0.0 else ret_val


@njit(inline="always")
def _compute_d(boot_arr: np.ndarray) -> float:
    """Compute the D-statistics from a 4 x L matrix of per-population frequencies, where L is the number of SNPs.

    This method calculates the Patterson's D-statistic using the formula:

    .. math::

        D = \frac{ABBA - BABA}{ABBA + BABA}

    where ABBA and BABA are the weighted sums of the derived allele frequencies across the populations.

    Args:
        boot_arr (np.ndarray): 4 x L matrix of per-population allele frequencies.

    Returns:
        float: Patterson's D-statistic.
    """
    P1, P2, P3, out = boot_arr
    L1 = P1.shape[0]  # Number of SNPs
    minus_out = 1.0 - out

    ABBA = 0.0
    BABA = 0.0

    for i in range(L1):
        # Frequency-based D-statistics
        ABBA += (1 - P1[i]) * P2[i] * P3[i] * minus_out[i]
        BABA += P1[i] * (1 - P2[i]) * P3[i] * minus_out[i]

    return _safe_ratio(ABBA - BABA, ABBA + BABA, ret_val=np.nan)


@njit(parallel=True)
def _execute_bootstrap_d(
    n_boot: int, boot_arr: np.ndarray, boots: np.ndarray, n_dstats: int = 1
) -> Tuple[float, float, float, float]:
    """Compute Patterson's D-statistic from a ``(5 x L)`` allele frequency matrix using Numba's JIT compilation.

    This method performs bootstrap resampling and computes the D-statistic for each bootstrap replicate.

    Args:
        n_boot (int): Number of bootstrap replicates.
        boot_arr (np.ndarray): Frequency matrix of shape ``(5, L)`` where the rows correspond to [P1, P2, P3a, P3b, Out] and the columns correspond to SNP sites.
        boots (np.ndarray): Indices for bootstrap sampling.
        n_dstats (int): Number of D-statistics to compute.

    Returns:
        np.ndarray: Array of shape (n_boot, n_dstats) containing the computed D-statistics for each bootstrap replicate.
    """
    boot_res = np.empty((n_boot, n_dstats), dtype=float)
    for b in prange(n_boot):
        cols = boots[b]
        barr_b = boot_arr[:, cols]
        boot_res[b] = _compute_d(barr_b)
    return boot_res


@njit(parallel=True)
def _execute_bootstrap_partd(
    n_boot: int, boot_arr: np.ndarray, boots: np.ndarray, n_dstats: int = 3
) -> Tuple[float, float, float]:
    """Compute Partitioned-D statistics from a ``(5 x L)`` allele frequency matrix using Numba's JIT compilation.

    This method performs bootstrap resampling and computes the Partitioned-D statistics for each bootstrap replicate.

    Args:
        n_boot (int): Number of bootstrap replicates.
        boot_arr (np.ndarray): Frequency matrix of shape ``(5, L)`` where the rows correspond to [P1, P2, P3a, P3b, Out] and the columns correspond to SNP sites.
        boots (np.ndarray): Indices for bootstrap sampling.
        n_dstats (int): Number of D-statistics to compute.

    Returns:
        np.ndarray: Array of shape (n_boot, n_dstats) containing the computed Partitioned-D statistics for each bootstrap replicate.
    """
    boot_res = np.empty((n_boot, n_dstats), dtype=float)
    for b in prange(n_boot):
        cols = boots[b]
        barr_b = boot_arr[:, cols]
        boot_res[b] = _compute_partd(barr_b)
    return boot_res


@njit(inline="always")
def _compute_partd(boot_arr: np.ndarray) -> Tuple[float, float, float]:
    """Weighted Partitioned-D from a 5 x L freq matrix.

    This method calculates the Partitioned-D statistics using the formula:

    .. math::

        D1 = \frac{ABBAA - BABAA}{ABBAA + BABAA} \\
        D2 = \frac{ABABA - BAABA}{ABABA + BAABA} \\
        D12 = \frac{ABBBA - BABBA}{ABBBA + BABBA}

    Args:
        boot_arr (np.ndarray): Frequency matrix of shape ``(5, L)`` where the rows correspond to ``[P1, P2, P3a, P3b, Out]`` and the columns correspond to SNP sites.

    Returns:
        Tuple[float, float, float]: Partitioned-D statistics (D1, D2, D12).
    """
    P1, P2, P3a, P3b, O = boot_arr
    L1 = P1.shape[0]  # Number of SNPs

    # Initialize sums
    ABBAA = 0.0
    BABAA = 0.0
    ABABA = 0.0
    BAABA = 0.0
    ABBBA = 0.0
    BABBA = 0.0

    for i in range(L1):
        one_minus_P1 = 1.0 - P1[i]
        one_minus_P2 = 1.0 - P2[i]
        one_minus_P3a = 1.0 - P3a[i]
        one_minus_P3b = 1.0 - P3b[i]
        one_minus_O = 1.0 - O[i]
        p1 = P1[i]
        p2 = P2[i]
        p3a = P3a[i]
        p3b = P3b[i]

        # Calculate the contributions for each D-statistic
        ABBAA += one_minus_P1 * p2 * p3a * one_minus_P3b * one_minus_O
        BABAA += p1 * one_minus_P2 * p3a * one_minus_P3b * one_minus_O

        ABABA += one_minus_P1 * p2 * one_minus_P3a * p3b * one_minus_O
        BAABA += p1 * one_minus_P2 * one_minus_P3a * p3b * one_minus_O

        ABBBA += one_minus_P1 * p2 * p3a * p3b * one_minus_O
        BABBA += p1 * one_minus_P2 * p3a * p3b * one_minus_O

    d1 = _safe_ratio(ABBAA - BABAA, ABBAA + BABAA, ret_val=np.nan)
    d2 = _safe_ratio(ABABA - BAABA, ABABA + BAABA, ret_val=np.nan)
    d12 = _safe_ratio(ABBBA - BABBA, ABBBA + BABBA, ret_val=np.nan)

    return d1, d2, d12


@njit(inline="always")
def _compute_dfoil(boot_arr: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute DFO, DFI, DOL, and DIL from a 5xL allele frequency matrix using Numba's JIT compilation.

    This method calculates the DFOIL statistics using the formula:

        .. math::
            DFO = \frac{(BABAA + BBBAA + ABABA + AAABA) - (BAABA + BBABA + ABBAA + AABAA)}{(BABAA + BBBAA + ABABA + AAABA) + (BAABA + BBABA + ABBAA + AABAA)} \\
            DFI = \frac{(BABAA + BABBA + ABABA + ABAAA) - (ABBAA + ABBBA + BAABA + BAAAA)}{(BABAA + BABBA + ABABA + ABAAA) + (ABBAA + ABBBA + BAABA + BAAAA)} \\
            DOL = \frac{(BAABA + BABBA + ABBAA + ABAAA) - (ABABA + ABBBA + BABAA + BAAAA)}{(BAABA + BABBA + ABBAA + ABAAA) + (ABABA + ABBBA + BABAA + BAAAA)} \\
            DIL = \frac{(ABBAA + BBBAA + BAABA + AAABA) - (ABABA + BBABA + BABAA + AABAA)}{(ABBAA + BBBAA + BAABA + AAABA) + (ABABA + BBABA + BABAA + AABAA)}

    Args:
        boot_arr (np.ndarray): Frequency matrix of shape ``(5, L)`` where the rows correspond to ``[P1, P2, P3a, P3b, Out]`` and the columns correspond to SNP sites.

    Returns:
        Tuple[float, float, float, float]: DFO, DFI, DOL, DIL values.
    """
    P1, P2, P3a, P3b, O = boot_arr
    L1 = P1.shape[0]  # Number of SNPs

    ABBBA = 0.0
    BABBA = 0.0
    ABBAA = 0.0
    BABAA = 0.0
    ABABA = 0.0
    BAABA = 0.0
    BBBAA = 0.0
    BBABA = 0.0
    AAABA = 0.0
    AABAA = 0.0
    ABAAA = 0.0
    BAAAA = 0.0

    for i in range(L1):
        one_minus_O = 1.0 - O[i]
        one_minus_P1 = 1.0 - P1[i]
        one_minus_P2 = 1.0 - P2[i]
        one_minus_P3a = 1.0 - P3a[i]
        one_minus_P3b = 1.0 - P3b[i]

        ABBBA += one_minus_P1 * P2[i] * P3a[i] * P3b[i] * one_minus_O
        BABBA += P1[i] * one_minus_P2 * P3a[i] * P3b[i] * one_minus_O
        ABBAA += one_minus_P1 * P2[i] * P3a[i] * one_minus_P3b * one_minus_O
        BABAA += P1[i] * one_minus_P2 * P3a[i] * one_minus_P3b * one_minus_O
        ABABA += one_minus_P1 * P2[i] * one_minus_P3a * P3b[i] * one_minus_O
        BAABA += P1[i] * one_minus_P2 * one_minus_P3a * P3b[i] * one_minus_O
        BBBAA += P1[i] * P2[i] * P3a[i] * one_minus_P3b * one_minus_O
        BBABA += P1[i] * P2[i] * one_minus_P3a * P3b[i] * one_minus_O
        AAABA += one_minus_P1 * one_minus_P2 * one_minus_P3a * P3b[i] * one_minus_O
        AABAA += one_minus_P1 * one_minus_P2 * P3a[i] * one_minus_P3b * one_minus_O
        ABAAA += one_minus_P1 * P2[i] * one_minus_P3a * one_minus_P3b * one_minus_O
        BAAAA += P1[i] * one_minus_P2 * one_minus_P3a * one_minus_P3b * one_minus_O

    # Compute each D-statistic
    dfo_num = (BABAA + BBBAA + ABABA + AAABA) - (BAABA + BBABA + ABBAA + AABAA)
    dfo_den = (BABAA + BBBAA + ABABA + AAABA) + (BAABA + BBABA + ABBAA + AABAA)

    dfi_num = (BABAA + BABBA + ABABA + ABAAA) - (ABBAA + ABBBA + BAABA + BAAAA)
    dfi_den = (BABAA + BABBA + ABABA + ABAAA) + (ABBAA + ABBBA + BAABA + BAAAA)

    dol_num = (BAABA + BABBA + ABBAA + ABAAA) - (ABABA + ABBBA + BABAA + BAAAA)
    dol_den = (BAABA + BABBA + ABBAA + ABAAA) + (ABABA + ABBBA + BABAA + BAAAA)

    dil_num = (ABBAA + BBBAA + BAABA + AAABA) - (ABABA + BBABA + BABAA + AABAA)
    dil_den = (ABBAA + BBBAA + BAABA + AAABA) + (ABABA + BBABA + BABAA + AABAA)

    # Safe division
    DFO = _safe_ratio(dfo_num, dfo_den, ret_val=np.nan)
    DFI = _safe_ratio(dfi_num, dfi_den, ret_val=np.nan)
    DOL = _safe_ratio(dol_num, dol_den, ret_val=np.nan)
    DIL = _safe_ratio(dil_num, dil_den, ret_val=np.nan)

    return DFO, DFI, DOL, DIL


@njit(parallel=True)
def _execute_bootstrap_dfoil(
    n_boot: int, boot_arr: np.ndarray, boots: np.ndarray, n_dstats: int = 4
) -> np.ndarray:
    """Compute DFOIL statistics from a ``(5 x L)`` allele frequency matrix using Numba's JIT compilation.

    This method performs bootstrap resampling and computes the DFOIL statistics for each bootstrap replicate.

    Args:
        n_boot (int): Number of bootstrap replicates.
        boot_arr (np.ndarray): Frequency matrix of shape ``(5, L)`` where the rows correspond to ``[P1, P2, P3a, P3b, Out]`` and the columns correspond to SNP sites.
        boots (np.ndarray): Indices for bootstrap sampling.
        n_dstats (int): Number of D-statistics to compute.

    Returns:
        np.ndarray: Array of shape (n_boot, n_dstats) containing the computed DFOIL statistics for each bootstrap replicate.
    """
    boot_res = np.empty((n_boot, n_dstats), dtype=float)
    for b in prange(n_boot):
        cols = boots[b]
        barr_b = boot_arr[:, cols]
        boot_res[b] = _compute_dfoil(barr_b)
    return boot_res
