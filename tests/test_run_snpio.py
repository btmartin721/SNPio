from argparse import Namespace
from unittest.mock import Mock

import pytest

from snpio.run_snpio import _run_fst_outlier_detection


@pytest.mark.parametrize("use_dbscan", [False, True])
def test_cli_runs_exactly_one_selected_fst_outlier_method(
    use_dbscan: bool,
) -> None:
    statistics = Mock()
    expected = object()
    statistics.detect_fst_outliers.return_value = expected
    args = Namespace(
        n_perm_fst=25,
        pvalue_correction_method="fdr_bh",
        use_dbscan=use_dbscan,
        n_jobs=4,
        min_samples_dbscan=7,
        random_seed=123,
    )

    observed = _run_fst_outlier_detection(statistics, args)

    assert observed is expected
    statistics.detect_fst_outliers.assert_called_once_with(
        n_permutations=25,
        correction_method="fdr_bh",
        use_dbscan=use_dbscan,
        n_jobs=4,
        min_samples=7,
        seed=123,
    )
