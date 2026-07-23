from pathlib import Path
from types import SimpleNamespace

import pytest

from snpio.utils.output_paths import OutputPaths
from snpio.utils.results_exporter import ResultsExporter


def test_unfiltered_output_paths_are_organized_by_artifact_type() -> None:
    paths = OutputPaths("run")

    assert paths.root == Path("run_output")
    assert paths.plots("pca") == Path("run_output/plots/pca")
    assert paths.reports("d_statistics") == Path(
        "run_output/reports/d_statistics"
    )
    assert paths.logs == Path("run_output/logs")
    assert paths.multiqc == Path("run_output/multiqc")
    assert paths.vcf_data == Path("run_output/data/vcf")
    assert paths.popmaps == Path("run_output/data/popmaps")


def test_filtered_output_paths_nest_nremover_inside_artifact_type() -> None:
    paths = OutputPaths("run", filtered=True)

    assert paths.plots("summary_statistics") == Path(
        "run_output/plots/nremover/summary_statistics"
    )
    assert paths.reports("linkage_disequilibrium") == Path(
        "run_output/reports/nremover/linkage_disequilibrium"
    )
    assert paths.logs == Path("run_output/logs")
    assert paths.vcf_data == Path("run_output/data/vcf")


def test_output_paths_infer_and_can_force_filtering_scope() -> None:
    unfiltered = SimpleNamespace(prefix="run", was_filtered=False)
    filtered = SimpleNamespace(prefix="run", was_filtered=True)

    assert OutputPaths.from_genotype_data(unfiltered).plots("missingness") == Path(
        "run_output/plots/missingness"
    )
    assert OutputPaths.from_genotype_data(filtered).plots("missingness") == Path(
        "run_output/plots/nremover/missingness"
    )
    assert OutputPaths.from_genotype_data(
        unfiltered, force_filtered=True
    ).plots("filtering") == Path("run_output/plots/nremover/filtering")


@pytest.mark.parametrize("operation", ["", ".", "..", "a/b", "/absolute"])
def test_output_paths_reject_invalid_operation_names(operation: str) -> None:
    with pytest.raises(ValueError, match="single non-empty directory name"):
        OutputPaths("run").plots(operation)


def test_results_exporter_uses_a_named_reports_directory(tmp_path: Path) -> None:
    exporter = ResultsExporter(tmp_path, operation="summary_statistics")

    assert exporter.output_dir == tmp_path / "reports" / "summary_statistics"
