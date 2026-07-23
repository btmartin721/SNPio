from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import snpio.popgenstats.fst_outliers as fst_outliers_module
import snpio.popgenstats.pop_gen_statistics as pop_gen_statistics_module
from snpio.plotting.plotting import Plotting
from snpio.popgenstats.fst_outlier_results import (
    FST_OUTLIER_COLUMNS,
    build_fst_outlier_dataframe,
)
from snpio.popgenstats.fst_outliers import FstOutliers
from snpio.popgenstats.fst_outliers_perm import PermutationOutlierDetector
from snpio.popgenstats.pop_gen_statistics import PopGenStatistics


def test_fst_outlier_dataframe_preserves_schema_when_empty() -> None:
    result = build_fst_outlier_dataframe()

    assert result.empty
    assert tuple(result.columns) == FST_OUTLIER_COLUMNS


def test_permutation_detector_preserves_schema_when_no_outliers() -> None:
    detector = object.__new__(PermutationOutlierDetector)
    detector.logger = Mock()
    detector.pop_names = ["pop1", "pop2"]
    detector.popmap_inverse = {"pop1": ["sample1"], "pop2": ["sample2"]}
    detector.genotype_data = SimpleNamespace(
        samples=["sample1", "sample2"],
        marker_names=["locus1", "locus2"],
    )
    detector._per_locus_permutation_test = Mock(
        return_value=(
            np.array([0.1, 0.2]),
            np.array([0.8, 0.9]),
        )
    )

    result = detector.run(
        n_perm=2,
        correction_method=None,
        alpha=0.05,
        seed=123,
        alternative="upper",
    )

    assert result.empty
    assert tuple(result.columns) == FST_OUTLIER_COLUMNS


def test_dbscan_detector_preserves_schema_when_no_outliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoOutlierDBSCAN:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, _: pd.DataFrame) -> None:
            pass

        def estimate_pvalues(self) -> np.ndarray:
            return np.array([0.7, 0.8])

        def identify_outliers(
            self,
            _: np.ndarray,
            *,
            alpha: float,
            correction_method: str | None,
        ) -> tuple[pd.Series, pd.Series]:
            del alpha, correction_method
            return (
                pd.Series([False, False]),
                pd.Series([0.7, 0.8]),
            )

    monkeypatch.setattr(
        fst_outliers_module,
        "DBSCANOutlierDetector",
        NoOutlierDBSCAN,
    )

    detector = object.__new__(FstOutliers)
    detector.logger = Mock()
    detector.verbose = False
    detector.debug = False
    detector._calculate_observed_per_locus_fst_all_pairs = Mock(
        return_value={
            ("pop1", "pop2"): pd.DataFrame(
                {"Locus": ["locus1", "locus2"], "Fst": [0.1, 0.2]}
            ),
            ("pop1", "pop3"): pd.DataFrame(
                {"Locus": ["locus1", "locus2"], "Fst": [0.2, 0.3]}
            ),
            ("pop2", "pop3"): pd.DataFrame(
                {"Locus": ["locus1", "locus2"], "Fst": [0.3, 0.4]}
            ),
        }
    )

    result = detector._dbscan_fst(
        correction_method=None,
        alpha=0.05,
        n_simulations=2,
        n_jobs=1,
        seed=123,
    )

    assert result.empty
    assert tuple(result.columns) == FST_OUTLIER_COLUMNS


def test_public_fst_outlier_api_skips_plotting_for_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyFstOutliers:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def detect_fst_outliers_dbscan(
            self, *_: object, **__: object
        ) -> pd.DataFrame:
            return build_fst_outlier_dataframe()

    monkeypatch.setattr(
        pop_gen_statistics_module,
        "FstOutliers",
        EmptyFstOutliers,
    )

    statistics = object.__new__(PopGenStatistics)
    statistics.logger = Mock()
    statistics.genotype_data = SimpleNamespace()
    statistics.plotter = Mock()
    statistics.verbose = False
    statistics.debug = False

    result = statistics.detect_fst_outliers(
        use_dbscan=True,
        n_permutations=2,
        seed=123,
    )

    assert result.empty
    assert tuple(result.columns) == FST_OUTLIER_COLUMNS
    statistics.plotter.plot_fst_outliers.assert_not_called()


def test_plot_fst_outliers_returns_cleanly_for_empty_results() -> None:
    plotter = object.__new__(Plotting)
    plotter.logger = Mock()

    plotter.plot_fst_outliers(
        build_fst_outlier_dataframe(),
        method="dbscan",
    )

    plotter.logger.warning.assert_called_once()


def test_plot_fst_outliers_rejects_missing_columns() -> None:
    plotter = object.__new__(Plotting)
    plotter.logger = Mock()

    with pytest.raises(ValueError, match="missing required column"):
        plotter.plot_fst_outliers(
            pd.DataFrame({"Locus": ["locus1"], "Fst": [0.5]}),
            method="permutation",
        )


def test_plot_fst_outliers_caps_static_and_multiqc_heatmaps(tmp_path) -> None:
    plotter = object.__new__(Plotting)
    plotter.logger = Mock()
    plotter.plot_format = "png"
    plotter.show = False
    plotter.snpio_mqc = Mock()
    plotter._plot_dir = Mock(return_value=tmp_path)

    results = build_fst_outlier_dataframe(
        [
            {
                "Locus": "locus1",
                "Population_Pair": "pop1_pop2",
                "Fst": 0.6,
                "q_value": 0.01,
            },
            {
                "Locus": "locus2",
                "Population_Pair": "pop1_pop3",
                "Fst": 0.7,
                "q_value": 0.02,
            },
        ]
    )

    plotter.plot_fst_outliers(
        results,
        method="permutation",
        max_outliers_to_plot=1,
    )

    assert (tmp_path / "outlier_snps_heatmap_permutation.png").is_file()
    queued = plotter.snpio_mqc.queue_heatmap.call_args.kwargs
    assert list(queued["df"].index) == ["locus1"]
    assert queued["pconfig"]["zlab"] == "Fst"
