"""Numerical helpers shared across SNPio analyses."""

from numbers import Real

import numpy as np


def safe_divide(
    numerator: np.ndarray | Real,
    denominator: np.ndarray | Real,
) -> np.ndarray:
    """Divide finite values with positive denominators.

    Non-estimable ratios, including non-finite inputs and non-positive
    denominators, are retained as ``NaN``. Inputs follow NumPy broadcasting
    rules.

    Args:
        numerator: Scalar or array-like dividend.
        denominator: Scalar or array-like divisor.

    Returns:
        A floating-point array with the broadcast shape of the inputs.
    """
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    output = np.full(
        np.broadcast_shapes(numerator_array.shape, denominator_array.shape),
        np.nan,
        dtype=np.float64,
    )
    valid = (
        np.isfinite(numerator_array)
        & np.isfinite(denominator_array)
        & (denominator_array > 0.0)
    )
    return np.divide(
        numerator_array,
        denominator_array,
        out=output,
        where=valid,
    )
