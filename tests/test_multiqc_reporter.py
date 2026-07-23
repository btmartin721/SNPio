"""Regression tests for SNPio's MultiQC data adapter."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from snpio import Plotting, SNPioMultiQC
from snpio.utils.multiqc_reporter import custom_box
from snpio.utils.plot_queue import queued_plots


def test_nonfinite_barplot_values_build_a_multiqc_report(tmp_path) -> None:
    """NaN and infinite values should be treated as missing, not fatal."""

    frame = pd.DataFrame(
        {"Ne": [212.17, np.nan, np.inf]},
        index=pd.Index(["DS", "TT", "overflow"], name="Population"),
    )
    queued_plots.clear()
    try:
        SNPioMultiQC.queue_barplot(
            frame,
            panel_id="nonfinite_ne",
            section="Linkage Disequilibrium",
            title="Effective Population Size",
            index_label="Population",
            value_label="Ne",
        )
        queued_data = queued_plots[-1]["data"]
        assert queued_data["DS"]["Ne"] == 212.17
        assert queued_data["TT"]["Ne"] is None
        assert queued_data["overflow"]["Ne"] is None

        report = SNPioMultiQC.build(
            prefix=tmp_path / "nonfinite_ne",
            output_dir=tmp_path / "multiqc",
            overwrite=True,
        )
    finally:
        queued_plots.clear()

    assert report.is_file()
    assert report.stat().st_size > 0
    assert np.isnan(frame.loc["TT", "Ne"])
    assert np.isposinf(frame.loc["overflow", "Ne"])


def test_custom_box_renders_grouped_long_form_data() -> None:
    """Long-form box data should render one grouped trace per statistic."""

    frame = pd.DataFrame(
        {
            "Population": ["pop1", "pop1", "pop2", "pop2"] * 2,
            "Replicate": [0, 1, 0, 1] * 2,
            "Statistic": ["r2D"] * 4 + ["rDz"] * 4,
            "Estimate": [0.010, 0.012, 0.020, 0.022, 0.001, 0.002, 0.003, 0.004],
        }
    )
    pconfig = {
        "title": "LD bootstrap distributions",
        "x": "Population",
        "y": "Estimate",
        "color": "Statistic",
        "hover_data": ["Replicate"],
        "category_orders": {
            "Population": ["pop1", "pop2"],
            "Statistic": ["r2D", "rDz"],
        },
        "boxmode": "group",
        "x_type": "category",
        "xlab": "Population",
        "ylab": "Bootstrap Estimate",
    }

    with patch(
        "snpio.utils.multiqc_reporter.pio.to_html", return_value="<div>plot</div>"
    ) as to_html:
        html = custom_box(frame, pconfig)

    figure = to_html.call_args.args[0]
    assert html == "<div>plot</div>"
    assert figure.layout.boxmode == "group"
    assert figure.layout.xaxis.type == "category"
    assert figure.layout.xaxis.title.text == "Population"
    assert figure.layout.yaxis.title.text == "Bootstrap Estimate"
    assert [trace.name for trace in figure.data] == ["r2D", "rDz"]
    assert all(set(trace.x) == {"pop1", "pop2"} for trace in figure.data)


def test_html_loader_accepts_inline_content() -> None:
    """Inline HTML should not be interpreted as a filesystem path."""

    reporter = object.__new__(SNPioMultiQC)
    content = "  <!doctype html><div>Inline panel</div>"

    rendered = reporter._add_html({"data": content, "panel_id": "inline"})

    assert rendered == {"html_content": content, "plot": None}


def test_html_loader_reads_an_existing_file(tmp_path) -> None:
    """File-backed HTML should be read as UTF-8 content."""

    reporter = object.__new__(SNPioMultiQC)
    html_path = tmp_path / "panel.html"
    html_path.write_text("<div>File-backed panel</div>", encoding="utf-8")

    rendered = reporter._add_html(
        {"data": html_path, "panel_id": "file_backed"}
    )

    assert rendered == {
        "html_content": "<div>File-backed panel</div>",
        "plot": None,
    }


def test_queue_html_rejects_a_missing_file_without_creating_directories(
    tmp_path,
) -> None:
    """A missing producer output should fail at queue time without side effects."""

    missing_path = tmp_path / "not-created" / "panel.html"

    with pytest.raises(FileNotFoundError, match="HTML file does not exist"):
        SNPioMultiQC.queue_html(
            missing_path,
            panel_id="missing",
            section="test",
            title="Missing panel",
            index_label=None,
        )

    assert not missing_path.parent.exists()


def test_html_loader_rejects_a_missing_file_without_creating_directories(
    tmp_path,
) -> None:
    """Rendering a stale queue entry should not create a phantom directory."""

    reporter = object.__new__(SNPioMultiQC)
    missing_path = tmp_path / "still-not-created" / "panel.html"

    with pytest.raises(FileNotFoundError, match="panel 'missing'"):
        reporter._add_html({"data": missing_path, "panel_id": "missing"})

    assert not missing_path.parent.exists()


def test_dstat_significance_counts_writes_html_before_queueing(tmp_path) -> None:
    """The D-statistic count panel must queue an HTML file that exists."""

    plotting = object.__new__(Plotting)
    plotting.output_dir_analysis = tmp_path
    plotting.plot_format = "png"
    plotting.show = False
    plotting.snpio_mqc = Mock()
    plotting.logger = Mock()
    results = pd.DataFrame(
        {
            "Significant (Raw)": [True, False],
            "Significant (Bonferroni)": [False, False],
            "Significant (FDR-BH)": [True, False],
        }
    )

    plotting.plot_dstat_significance_counts(results, "patterson")

    html_path = plotting.snpio_mqc.queue_html.call_args.args[0]
    assert html_path.is_file()
    assert "plotly" in html_path.read_text(encoding="utf-8").lower()


def test_dstat_heatmap_batches_significance_annotations(tmp_path) -> None:
    """Heatmap significance labels should be stored in one trace matrix."""

    plotting = object.__new__(Plotting)
    plotting.output_dir_analysis = tmp_path
    plotting.snpio_mqc = Mock()
    plotting._queued_panels = set()
    plotting.logger = logging.getLogger("test_dstat_heatmap_annotations")
    frame = pd.DataFrame(
        {
            "Method": ["dfoil", "dfoil"],
            "P_DFO": [0.01, 0.5],
            "P_DFI": [0.02, 0.6],
            "P_DOL": [0.03, 0.7],
            "P_DIL": [0.04, 0.8],
            **{
                f"Significant ({correction}) {stat}": [True, False]
                for correction in ("Raw", "Bonferroni", "FDR-BH")
                for stat in ("DFO", "DFI", "DOL", "DIL")
            },
        },
        index=pd.Index(["a-b-c-d-o", "f-g-h-i-o"], name="Quartet"),
    )

    plotting.plot_d_statistics_heatmap(frame, method_name="dfoil")

    output_path = tmp_path / "d_statistics_heatmap_dfoil.html"
    content = output_path.read_text(encoding="utf-8")
    assert '"texttemplate":"%{text}"' in content
    assert plotting.snpio_mqc.queue_html.call_count == 1
