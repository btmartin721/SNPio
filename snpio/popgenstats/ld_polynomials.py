r"""Exact polynomials for unbiased LD estimation from unphased diploids.

This module implements the polynomial construction described by Ragsdale and
Gravel (2020), Equation 5.  A two-locus diploid genotype has one of nine
states, ordered as::

    AABB, AABb, AAbb, AaBB, AaBb, Aabb, aaBB, aaBb, aabb

The population-frequency polynomials are expanded exactly with
``fractions.Fraction``.  For a monomial

.. math::

    a \prod_j g_j^{k_j},

the unbiased finite-sample estimate is evaluated as

.. math::

    a \frac{\prod_j (n_j)_{k_j}}{(n)_{\sum_j k_j}},

where ``(x)_k`` is a falling factorial.  This is algebraically equivalent to
the multivariate-hypergeometric form in Equation 5 but avoids factorials and
large intermediate values.

References
----------
Ragsdale, A. P. & Gravel, S. (2020). Unbiased estimation of linkage
disequilibrium from unphased data. Molecular Biology and Evolution,
37(3), 923-932. https://doi.org/10.1093/molbev/msz265
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numba import njit

_N_GENOTYPE_STATES = 9
Monomial = tuple[int, ...]
Polynomial = dict[Monomial, Fraction]


@dataclass(frozen=True)
class PolynomialTerms:
    """Immutable sparse representation of a homogeneous LD polynomial."""

    name: str
    coefficients: tuple[Fraction, ...]
    exponents: tuple[Monomial, ...]
    degree: int

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError(f"Polynomial '{self.name}' has no terms.")

        if len(self.coefficients) != len(self.exponents):
            raise ValueError("Coefficient and exponent counts do not match.")

        if any(len(exponent) != _N_GENOTYPE_STATES for exponent in self.exponents):
            raise ValueError("Every monomial must contain nine genotype exponents.")

        if any(sum(exponent) != self.degree for exponent in self.exponents):
            raise ValueError(f"Polynomial '{self.name}' is not homogeneous.")


def _constant(value: int | Fraction) -> Polynomial:
    """Return a constant sparse polynomial."""

    coefficient = Fraction(value)

    if coefficient == 0:
        return {}

    return {(0,) * _N_GENOTYPE_STATES: coefficient}


def _linear(*coefficients: int | Fraction) -> Polynomial:
    """Return a linear polynomial in the nine genotype frequencies."""

    if len(coefficients) != _N_GENOTYPE_STATES:
        msg = "A genotype-frequency linear form requires nine terms."
        raise ValueError(msg)

    polynomial: Polynomial = {}

    for index, value in enumerate(coefficients):
        coefficient = Fraction(value)

        if coefficient == 0:
            continue

        exponent = [0] * _N_GENOTYPE_STATES
        exponent[index] = 1
        polynomial[tuple(exponent)] = coefficient

    return polynomial


def _add(left: Polynomial, right: Polynomial) -> Polynomial:
    """Add two sparse polynomials."""

    result = dict(left)

    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient

        if result[exponent] == 0:
            del result[exponent]

    return result


def _scale(polynomial: Polynomial, scalar: int | Fraction) -> Polynomial:
    """Multiply a sparse polynomial by an exact scalar."""

    factor = Fraction(scalar)

    if factor == 0:
        return {}

    return {
        exponent: coefficient * factor
        for exponent, coefficient in polynomial.items()
        if coefficient * factor != 0
    }


def _subtract(left: Polynomial, right: Polynomial) -> Polynomial:
    """Subtract one sparse polynomial from another."""

    return _add(left, _scale(right, -1))


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    """Multiply two sparse polynomials and combine identical monomials."""

    result: Polynomial = {}
    for left_exp, left_coef in left.items():
        for right_exp, right_coef in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))

            result[exponent] = (
                result.get(exponent, Fraction(0)) + left_coef * right_coef
            )

            if result[exponent] == 0:
                del result[exponent]

    return result


def _power(polynomial: Polynomial, exponent: int) -> Polynomial:
    """Raise a sparse polynomial to a non-negative integer power."""

    if exponent < 0:
        raise ValueError("Polynomial exponents must be non-negative.")

    result = _constant(1)
    base = polynomial
    power = exponent

    while power:
        if power & 1:
            result = _multiply(result, base)

        power >>= 1

        if power:
            base = _multiply(base, base)

    return result


def _to_terms(name: str, polynomial: Polynomial) -> PolynomialTerms:
    """Convert a sparse polynomial mapping into deterministic term arrays."""

    ordered = sorted(polynomial.items())
    degrees = {sum(exponent) for exponent, _ in ordered}

    if len(degrees) != 1:
        raise ValueError(f"Polynomial '{name}' is not homogeneous: {degrees}.")

    degree = degrees.pop()

    return PolynomialTerms(
        name=name,
        coefficients=tuple(coefficient for _, coefficient in ordered),
        exponents=tuple(exponent for exponent, _ in ordered),
        degree=degree,
    )


def _build_polynomials() -> Mapping[str, PolynomialTerms]:
    """Build the Hill-Robertson polynomials from the paper's equations."""

    half = Fraction(1, 2)
    quarter = Fraction(1, 4)

    # Naive haplotype-frequency estimates from the nine unphased diploid
    # genotype frequencies (Ragsdale & Gravel 2020, p. 930).
    x_AB = _linear(1, half, 0, half, quarter, 0, 0, 0, 0)
    x_Ab = _linear(0, half, 1, 0, quarter, half, 0, 0, 0)
    x_aB = _linear(0, 0, 0, half, quarter, 0, 1, half, 0)
    x_ab = _linear(0, 0, 0, 0, quarter, half, 0, half, 1)

    delta_x = _subtract(
        _multiply(x_AB, x_ab),
        _multiply(x_Ab, x_aB),
    )

    left_major = _add(x_AB, x_Ab)
    left_minor = _add(x_aB, x_ab)
    right_major = _add(x_AB, x_aB)
    right_minor = _add(x_Ab, x_ab)

    d_polynomial = _scale(delta_x, 2)
    d2_polynomial = _scale(_power(delta_x, 2), 4)

    dz_polynomial = _scale(
        _multiply(
            _multiply(delta_x, _subtract(left_minor, left_major)),
            _subtract(right_minor, right_major),
        ),
        2,
    )

    pi2_polynomial = _multiply(
        _multiply(left_major, left_minor),
        _multiply(right_major, right_minor),
    )

    return MappingProxyType(
        {
            "D": _to_terms("D", d_polynomial),
            "D2": _to_terms("D2", d2_polynomial),
            "Dz": _to_terms("Dz", dz_polynomial),
            "pi2": _to_terms("pi2", pi2_polynomial),
        }
    )


LD_POLYNOMIALS: Mapping[str, PolynomialTerms] = _build_polynomials()
"""Exact sparse polynomial terms for ``D``, ``D2``, ``Dz``, and ``pi2``."""


_NUMERIC_TERMS = MappingProxyType(
    {
        name: (
            np.asarray([float(value) for value in polynomial.coefficients]),
            np.asarray(polynomial.exponents, dtype=np.int8),
            polynomial.degree,
        )
        for name, polynomial in LD_POLYNOMIALS.items()
    }
)


@njit(cache=True, nogil=True)
def _evaluate_polynomial(
    counts: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Evaluate one unbiased polynomial in compiled pair-major loops."""

    n_pairs = counts.shape[0]
    values = np.empty(n_pairs, dtype=np.float64)

    for pair_index in range(n_pairs):
        sample_size = 0
        for state_index in range(_N_GENOTYPE_STATES):
            sample_size += counts[pair_index, state_index]

        if sample_size < degree:
            values[pair_index] = np.nan
            continue

        denominator = 1.0

        for offset in range(degree):
            denominator *= sample_size - offset

        estimate = 0.0

        for term_index in range(coefficients.size):
            term = coefficients[term_index]

            for state_index in range(_N_GENOTYPE_STATES):
                order = exponents[term_index, state_index]
                if order:
                    count = counts[pair_index, state_index]
                    for offset in range(order):
                        term *= count - offset
            estimate += term

        values[pair_index] = estimate / denominator

    return values


def evaluate_unbiased_ld(
    genotype_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate unbiased LD statistics from nine-state genotype counts.

    Args:
        genotype_counts: Integer array with shape ``(n_pairs, 9)``. Columns
            follow the order ``AABB, AABb, AAbb, AaBB, AaBb, Aabb, aaBB,
            aaBb, aabb``. Counts must contain only non-negative values.

    Returns:
        Dictionary mapping ``D``, ``D2``, ``Dz``, and ``pi2`` to float64 arrays with one value per locus pair. A value is ``NaN`` when fewer than the polynomial degree's number of complete diploid samples are available (two for ``D`` and four for the fourth-order statistics).

    Raises:
        ValueError: If the count array is not two-dimensional with nine
            columns, contains negative/non-integral values, or is empty.
    """

    counts = np.asarray(genotype_counts)
    if counts.ndim != 2 or counts.shape[1] != _N_GENOTYPE_STATES:
        raise ValueError("genotype_counts must have shape (n_pairs, 9).")

    if counts.shape[0] == 0:
        msg = "genotype_counts must contain at least one locus pair."
        raise ValueError(msg)

    if not np.issubdtype(counts.dtype, np.integer):
        if not np.all(np.isfinite(counts)) or not np.all(counts == np.floor(counts)):
            msg = "genotype_counts must contain finite integer values."
            raise ValueError(msg)

    if np.any(counts < 0):
        raise ValueError("genotype_counts cannot contain negative values.")

    counts_int = np.ascontiguousarray(counts, dtype=np.int64)

    return {
        name: _evaluate_polynomial(counts_int, coefficients, exponents, degree)
        for name, (coefficients, exponents, degree) in _NUMERIC_TERMS.items()
    }


__all__ = ["LD_POLYNOMIALS", "PolynomialTerms", "evaluate_unbiased_ld"]
