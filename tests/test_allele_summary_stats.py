from unittest.mock import Mock

import numpy as np

from snpio.analysis.allele_summary_stats import AlleleSummaryStats


def test_safe_divide_broadcasts_and_marks_invalid_ratios() -> None:
    """Masked and broadcast divisions should be deterministic and warning-free."""
    numerator = np.array([[1.0, 1.0], [0.0, 0.0]])
    denominator = np.array([[2.0], [0.0]])

    observed = AlleleSummaryStats._safe_divide(numerator, denominator)

    np.testing.assert_allclose(observed[0], np.array([0.5, 0.5]))
    assert np.isnan(observed[1]).all()


def test_summarize_retains_missing_sample_and_locus_as_nan() -> None:
    """Fully missing rows and columns must not leak uninitialized values."""
    stats = object.__new__(AlleleSummaryStats)
    stats.alleles = (
        np.array([[0, -1], [0, -1], [-1, -1]], dtype=np.int8),
        np.array([[1, -1], [0, -1], [-1, -1]], dtype=np.int8),
    )
    stats.logger = Mock()
    stats._plot = Mock()

    summary = stats.summarize()

    assert summary["Overall Heterozygosity Prop."] == 0.5
    assert summary["Mean Sample Heterozygosity Prop."] == 0.5
    assert summary["Mean Locus Heterozygosity Prop."] == 0.5
    assert summary["Mean Effective Alleles"] == 1.6
    assert summary["MAF Mean"] == 0.25
    stats._plot.assert_called_once()
