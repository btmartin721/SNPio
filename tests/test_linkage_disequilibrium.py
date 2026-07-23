"""Tests for unbiased LD estimation from unphased diploid genotypes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from snpio import PhylipReader, PopGenStatistics
from snpio.plotting.linkage_disequilibrium import LinkageDisequilibriumPlotter
from snpio.popgenstats.ld_polynomials import LD_POLYNOMIALS, evaluate_unbiased_ld
from snpio.popgenstats.linkage_disequilibrium import LinkageDisequilibrium


def _burrows_d(counts: np.ndarray) -> float:
    """Calculate the finite-sample Burrows composite-D estimator."""

    n = counts.sum()
    frequencies = counts / n
    p = frequencies[:3].sum() + 0.5 * frequencies[3:6].sum()
    q = frequencies[[0, 3, 6]].sum() + 0.5 * frequencies[[1, 4, 7]].sum()
    delta_tilde = (
        2 * frequencies[0]
        + frequencies[1]
        + frequencies[3]
        + 0.5 * frequencies[4]
        - 2 * p * q
    )
    return n / (n - 1) * delta_tilde


def _flip_left_allele(counts: np.ndarray) -> np.ndarray:
    """Swap AA and aa rows in the 3-by-3 genotype count table."""

    return counts.reshape(3, 3)[::-1].reshape(9)


def _allele_channels(genotypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 0/1/2 dosages to two biological allele channels."""

    first = np.full_like(genotypes, -1)
    second = np.full_like(genotypes, -1)
    called = genotypes >= 0
    first[called] = 0
    second[genotypes == 0] = 0
    second[genotypes == 1] = 1
    first[genotypes == 2] = 1
    second[genotypes == 2] = 1
    return first, second


def _fake_genotype_data(
    genotypes: np.ndarray,
    prefix: Path,
    *,
    populations: list[str] | None = None,
    marker_names: list[str] | None = None,
    was_filtered: bool = False,
) -> SimpleNamespace:
    """Build the GenotypeData attributes required by the LD module."""

    n_samples, n_loci = genotypes.shape
    if populations is None:
        populations = []
        has_popmap = False
        popmap = None
    else:
        has_popmap = True
        popmap = {
            f"sample_{index}": population
            for index, population in enumerate(populations)
        }
    return SimpleNamespace(
        snp_data=np.full((n_samples, n_loci), "A", dtype="U1"),
        prefix=str(prefix),
        was_filtered=was_filtered,
        marker_names=marker_names,
        samples=[f"sample_{index}" for index in range(n_samples)],
        populations=populations,
        popmap=popmap,
        has_popmap=has_popmap,
        filetype="vcf" if marker_names else "phylip",
        from_vcf=marker_names is not None,
    )


@pytest.fixture
def genotype_matrix() -> np.ndarray:
    """Return a deterministic two-population unphased genotype matrix."""

    return np.asarray(
        [
            [0, 0, 0, 1, 2, 1, 0, 2],
            [0, 1, 0, 1, 2, 2, 1, 2],
            [1, 0, 1, 1, 1, 2, 0, 1],
            [2, 1, 2, 0, 0, 1, 2, 0],
            [0, 2, 0, 1, 1, 0, 2, 1],
            [1, 2, 1, 2, 0, 0, 2, 1],
            [2, 1, 2, 2, 0, 1, 1, 0],
            [2, 0, 1, 0, 1, 2, 0, 2],
        ],
        dtype=np.int8,
    )


def test_polynomial_definitions_are_homogeneous() -> None:
    """Every stored polynomial must use one finite-sample order."""

    assert set(LD_POLYNOMIALS) == {"D", "D2", "Dz", "pi2"}
    assert LD_POLYNOMIALS["D"].degree == 2
    for statistic in ("D2", "Dz", "pi2"):
        assert LD_POLYNOMIALS[statistic].degree == 4
    for polynomial in LD_POLYNOMIALS.values():
        assert all(
            sum(exponents) == polynomial.degree for exponents in polynomial.exponents
        )


def test_unbiased_d_matches_burrows_estimator() -> None:
    """The degree-two polynomial must reduce to corrected composite D."""

    counts = np.asarray([[8, 3, 2, 4, 5, 1, 2, 3, 7]], dtype=np.int64)
    observed = evaluate_unbiased_ld(counts)["D"][0]
    assert observed == pytest.approx(_burrows_d(counts[0]), abs=1e-14)


def test_polynomials_match_paper_authors_reference_values() -> None:
    """Guard exact agreement with moments-popgen 1.6.0 reference outputs."""

    counts = np.asarray(
        [
            [8, 3, 2, 4, 5, 1, 2, 3, 7],
            [1, 0, 1, 0, 2, 0, 1, 0, 1],
            [12, 4, 0, 5, 8, 2, 0, 3, 9],
            [0, 2, 5, 1, 3, 6, 4, 2, 1],
        ],
        dtype=np.int64,
    )
    expected = {
        "D": [
            0.16008403361344536,
            0.0,
            0.24335548172757476,
            -0.17391304347826086,
        ],
        "D2": [
            0.022021422205245736,
            -0.07222222222222222,
            0.057528765902276964,
            0.026526915113871636,
        ],
        "Dz": [
            -0.0006254774637127578,
            0.044444444444444446,
            0.0007150960213921076,
            -0.0017527762092979485,
        ],
        "pi2": [
            0.06418882894066717,
            0.07777777777777778,
            0.06261234570402183,
            0.06038255223037832,
        ],
    }
    observed = evaluate_unbiased_ld(counts)
    for statistic, values in expected.items():
        np.testing.assert_allclose(observed[statistic], values, rtol=1e-13, atol=1e-14)


def test_polynomials_are_invariant_to_allele_label_swap() -> None:
    """Unpolarized fourth-order statistics must not depend on allele labels."""

    counts = np.asarray([8, 3, 2, 4, 5, 1, 2, 3, 7], dtype=np.int64)
    original = evaluate_unbiased_ld(counts[None, :])
    flipped = evaluate_unbiased_ld(_flip_left_allele(counts)[None, :])

    assert flipped["D"][0] == pytest.approx(-original["D"][0])
    for statistic in ("D2", "Dz", "pi2"):
        assert flipped[statistic][0] == pytest.approx(original[statistic][0])


def test_fourth_order_statistics_require_four_complete_samples() -> None:
    """Only D is defined when fewer than four complete diploids are present."""

    estimates = evaluate_unbiased_ld(
        np.asarray([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.int64)
    )
    assert np.isfinite(estimates["D"][0])
    assert np.isnan(estimates["D2"][0])
    assert np.isnan(estimates["Dz"][0])
    assert np.isnan(estimates["pi2"][0])


def test_invalid_genotype_counts_are_rejected() -> None:
    """Malformed count arrays must fail before polynomial evaluation."""

    with pytest.raises(ValueError, match="shape"):
        evaluate_unbiased_ld(np.ones((2, 8), dtype=int))
    with pytest.raises(ValueError, match="negative"):
        evaluate_unbiased_ld(-np.ones((1, 9), dtype=int))
    with pytest.raises(ValueError, match="integer"):
        evaluate_unbiased_ld(np.full((1, 9), 0.5))


def test_nonpositive_r2d_reports_undefined_ne_and_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A negative unbiased LD estimate must not become a negative Ne."""

    genotypes = np.asarray(
        [[0, 0], [0, 2], [1, 1], [1, 1], [2, 0], [2, 2]],
        dtype=np.int8,
    )
    genotype_data = _fake_genotype_data(genotypes, tmp_path / "negative_r2d")
    logger = logging.getLogger("test_ld_nonpositive_r2d")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = LinkageDisequilibrium(
            genotype_data,
            genotypes,
            allele_channels=_allele_channels(genotypes),
            logger=logger,
        ).run(
            assume_unlinked=True,
            n_bootstraps=0,
            n_bootstrap_blocks=2,
            max_pairs=None,
            pairwise_sample_size=0,
            seed=42,
            save_pairwise=False,
            save_plots=False,
        )

    row = result.summary.iloc[0]
    assert row["r2D"] < 0.0
    assert np.isnan(row["Ne"])
    assert "Recent Ne is not estimable" in caplog.text
    assert "Ne is reported as NaN" in caplog.text


def test_unlinked_analysis_outputs_and_ne_formula(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """The public analysis core should write complete structured outputs."""

    genotype_data = _fake_genotype_data(
        genotype_matrix,
        tmp_path / "ld_case",
        populations=["pop1"] * 4 + ["pop2"] * 4,
    )
    analysis = LinkageDisequilibrium(
        genotype_data,
        genotype_matrix,
        allele_channels=_allele_channels(genotype_matrix),
        logger=logging.getLogger("test_ld"),
    )
    result = analysis.run(
        assume_unlinked=True,
        n_bootstraps=20,
        n_bootstrap_blocks=4,
        max_pairs=None,
        pairwise_sample_size=100,
        pair_chunk_size=3,
        seed=42,
        save_pairwise=True,
        save_plots=False,
    )

    assert result.summary["Population"].tolist() == ["pop1", "pop2"]
    assert (result.summary["Pairs"] > 0).all()
    finite = result.summary.loc[result.summary["r2D"] > 0]
    np.testing.assert_allclose(finite["Ne"], 1.0 / (3.0 * finite["r2D"]))
    assert result.bootstrap.groupby("Population").size().to_dict() == {
        "pop1": 20,
        "pop2": 20,
    }
    assert not result.block_summaries.empty
    assert not result.pairwise_sample.empty
    assert result.metadata["method"].startswith("Ragsdale-Gravel")
    assert result.metadata["parallel_backend"] == "thread"

    expected_report_dir = (
        tmp_path / "ld_case_output" / "reports" / "linkage_disequilibrium"
    )
    assert {path.parent for path in result.files.values()} == {
        expected_report_dir
    }
    for path in result.files.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    with result.files["ld_metadata"].open(encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    assert metadata["doi"] == "10.1093/molbev/msz265"


def test_popgenstatistics_public_ld_api(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """The public facade should encode PHYLIP genotypes and run end to end."""

    allele_codes = np.asarray(["A", "M", "C"])
    phylip = tmp_path / "unlinked.phy"
    popmap = tmp_path / "unlinked.popmap"
    phylip_rows = [
        f"sample_{index}\t{''.join(allele_codes[row])}"
        for index, row in enumerate(genotype_matrix)
    ]
    phylip.write_text(
        "\n".join(
            [
                f"{genotype_matrix.shape[0]} {genotype_matrix.shape[1]}",
                *phylip_rows,
            ]
        ),
        encoding="utf-8",
    )
    popmap.write_text(
        "\n".join(
            f"sample_{index}\t{'pop1' if index < 4 else 'pop2'}"
            for index in range(genotype_matrix.shape[0])
        ),
        encoding="utf-8",
    )

    genotype_data = PhylipReader(
        filename=str(phylip),
        popmapfile=str(popmap),
        force_popmap=True,
        prefix=str(tmp_path / "public_api"),
        verbose=False,
    )
    result = PopGenStatistics(genotype_data).calculate_linkage_disequilibrium(
        assume_unlinked=True,
        n_bootstraps=0,
        n_bootstrap_blocks=4,
        max_pairs=None,
        pairwise_sample_size=8,
        pair_chunk_size=3,
        seed=2718,
        save_pairwise=False,
        save_plots=False,
    )

    assert result.summary["Population"].tolist() == ["pop1", "pop2"]
    assert result.metadata["n_biallelic_loci"] == genotype_matrix.shape[1]
    assert result.metadata["group_source"] == "assumed_unlinked"


def test_ld_multiqc_omits_undefined_ne_from_barplot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The LD report should label missing Ne and plot only finite estimates."""

    class RecordingMultiQC:
        tables: list[tuple[pd.DataFrame, dict[str, object]]] = []
        bars: list[tuple[pd.DataFrame, dict[str, object]]] = []
        heatmaps: list[tuple[pd.DataFrame, dict[str, object]]] = []
        linegraphs: list[tuple[pd.DataFrame, dict[str, object]]] = []

        @classmethod
        def queue_table(cls, frame: pd.DataFrame, **kwargs: object) -> None:
            cls.tables.append((frame.copy(), kwargs))

        @classmethod
        def queue_barplot(cls, frame: pd.DataFrame, **kwargs: object) -> None:
            cls.bars.append((frame.copy(), kwargs))

        @classmethod
        def queue_heatmap(cls, frame: pd.DataFrame, **kwargs: object) -> None:
            cls.heatmaps.append((frame.copy(), kwargs))

        @classmethod
        def queue_linegraph(cls, frame: pd.DataFrame, **kwargs: object) -> None:
            cls.linegraphs.append((frame.copy(), kwargs))

    statistics = object.__new__(PopGenStatistics)
    statistics.snpio_mqc = RecordingMultiQC
    statistics.logger = logging.getLogger("test_ld_multiqc_undefined_ne")
    summary = pd.DataFrame(
        {
            "Population": ["finite", "undefined"],
            "Samples": [20, 26],
            "Loci": [100, 100],
            "Pairs": [1_000, 1_000],
            "r2D": [0.01, -0.001],
            "rDz": [0.0, 0.01],
            "Ne": [1.0 / 0.03, np.nan],
        }
    )

    with caplog.at_level(logging.WARNING, logger=statistics.logger.name):
        statistics._queue_linkage_disequilibrium_multiqc(summary)

    report_table = RecordingMultiQC.tables[0][0]
    assert report_table.loc["finite", "Ne_Status"] == "Estimated"
    assert report_table.loc["undefined", "Ne_Status"] == "Not estimable (r2D <= 0)"
    ne_bars = [
        frame
        for frame, options in RecordingMultiQC.bars
        if options["panel_id"] == "linkage_disequilibrium_barplot_Ne"
    ]
    assert len(ne_bars) == 1
    assert ne_bars[0].index.tolist() == ["finite"]
    assert "Omitting non-estimable Ne values" in caplog.text


def test_ld_bootstrap_boxplots_group_replicates_by_population() -> None:
    """Each population should retain its own finite bootstrap distributions."""

    class RecordingMultiQC:
        boxes: list[tuple[pd.DataFrame, dict[str, object]]] = []

        @classmethod
        def queue_custom_boxplot(
            cls, df: pd.DataFrame, **kwargs: object
        ) -> None:
            cls.boxes.append((df.copy(), kwargs))

    statistics = object.__new__(PopGenStatistics)
    statistics.snpio_mqc = RecordingMultiQC
    statistics.logger = logging.getLogger("test_ld_bootstrap_boxplots")
    bootstrap = pd.DataFrame(
        {
            "Population": ["pop1", "pop1", "pop1", "pop2", "pop2", "pop2"],
            "Replicate": [0, 1, 2, 0, 1, 2],
            "r2D": [0.010, 0.012, 0.011, 0.020, 0.019, 0.021],
            "rDz": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
            "D": [-0.01, 0.00, 0.01, -0.02, 0.00, 0.02],
            "Dz": [0.10, 0.11, 0.12, 0.20, 0.21, 0.22],
            "Pi2": [1.0, np.inf, 1.2, 2.0, 2.1, 2.2],
            "Ne": [125.75, 130.25, np.nan, 50.50, 52.25, 48.75],
        }
    )

    statistics._queue_linkage_disequilibrium_boxplots(bootstrap)

    assert len(RecordingMultiQC.boxes) == 2
    ld_frame, ld_config = RecordingMultiQC.boxes[0]
    ne_frame, ne_config = RecordingMultiQC.boxes[1]

    assert ld_config["panel_id"] == "linkage_disequilibrium_boxplot"
    assert ne_config["panel_id"] == "linkage_disequilibrium_boxplot_Ne"
    assert ld_config["panel_id"] != ne_config["panel_id"]
    ld_description = ld_config["description"]
    ne_description = ne_config["description"]
    assert isinstance(ld_description, str)
    assert isinstance(ne_description, str)
    assert "center line is the median" in ld_description
    assert "compare populations within the same statistic" in ld_description
    assert "effective, not census" in ne_description
    ld_pconfig = ld_config["pconfig"]
    assert isinstance(ld_pconfig, dict)
    assert ld_pconfig["color"] == "Statistic"
    assert ld_pconfig["boxmode"] == "group"

    assert ld_frame.columns.tolist() == [
        "Population",
        "Replicate",
        "Statistic",
        "Estimate",
    ]
    assert set(ld_frame["Population"]) == {"pop1", "pop2"}
    assert set(ld_frame["Statistic"]) == {"r2D", "rDz", "D", "Dz", "Pi2"}
    assert "Ne" not in set(ld_frame["Statistic"])
    assert np.isfinite(ld_frame["Estimate"]).all()
    assert ld_frame.groupby(["Population", "Statistic"]).size().to_dict()[
        ("pop1", "Pi2")
    ] == 2
    pop2_replicates = ld_frame.loc[
        ld_frame["Population"] == "pop2", "Replicate"
    ].unique()
    assert sorted(pop2_replicates) == [0, 1, 2]

    assert ne_frame.columns.tolist() == ["Population", "Replicate", "Ne"]
    assert ne_frame.groupby("Population").size().to_dict() == {"pop1": 2, "pop2": 3}
    assert np.isfinite(ne_frame["Ne"]).all()
    assert ne_frame.loc[0, "Ne"] == pytest.approx(125.75)


def test_serial_and_parallel_results_match(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """Thread scheduling must not alter sampled pairs or bootstrap estimates."""

    outputs = []
    for jobs, prefix in ((1, "serial"), (2, "parallel")):
        genotype_data = _fake_genotype_data(
            genotype_matrix,
            tmp_path / prefix,
            populations=["pop1"] * 4 + ["pop2"] * 4,
        )
        result = LinkageDisequilibrium(
            genotype_data,
            genotype_matrix,
            allele_channels=_allele_channels(genotype_matrix),
        ).run(
            assume_unlinked=True,
            n_bootstraps=10,
            n_bootstrap_blocks=4,
            n_jobs=jobs,
            max_pairs=18,
            pairwise_sample_size=18,
            pair_chunk_size=2,
            seed=194,
            save_pairwise=False,
            save_plots=False,
        )
        outputs.append(result)

    pd.testing.assert_frame_equal(outputs[0].summary, outputs[1].summary)
    pd.testing.assert_frame_equal(outputs[0].bootstrap, outputs[1].bootstrap)
    sort_columns = ["Population", "locus_i", "locus_j"]
    left = outputs[0].pairwise_sample.sort_values(sort_columns).reset_index(drop=True)
    right = outputs[1].pairwise_sample.sort_values(sort_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_explicit_locus_groups_exclude_within_group_pairs(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """Explicit linkage groups must be honored by every retained pair."""

    groups = np.asarray(
        ["chr1", "chr1", "chr2", "chr2", "chr3", "chr3", "chr4", "chr4"]
    )
    genotype_data = _fake_genotype_data(genotype_matrix, tmp_path / "groups")
    result = LinkageDisequilibrium(
        genotype_data,
        genotype_matrix,
        allele_channels=_allele_channels(genotype_matrix),
    ).run(
        locus_groups=groups,
        n_bootstraps=0,
        n_bootstrap_blocks=4,
        max_pairs=None,
        pairwise_sample_size=100,
        seed=12,
        save_pairwise=False,
        save_plots=False,
    )

    locus_i = result.pairwise_sample["locus_i"].to_numpy(dtype=int)
    locus_j = result.pairwise_sample["locus_j"].to_numpy(dtype=int)
    assert np.all(groups[locus_i] != groups[locus_j])


def test_monomorphic_loci_are_excluded(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """Globally monomorphic columns must not enter SNP-pair estimates."""

    genotypes = np.column_stack(
        [genotype_matrix, np.zeros(genotype_matrix.shape[0], dtype=np.int8)]
    )
    genotype_data = _fake_genotype_data(genotypes, tmp_path / "monomorphic")
    result = LinkageDisequilibrium(
        genotype_data,
        genotypes,
        allele_channels=_allele_channels(genotypes),
    ).run(
        assume_unlinked=True,
        n_bootstraps=0,
        n_bootstrap_blocks=4,
        max_pairs=None,
        pairwise_sample_size=100,
        seed=121,
        save_pairwise=False,
        save_plots=False,
    )

    assert result.metadata["n_biallelic_loci"] == genotype_matrix.shape[1]
    assert result.metadata["n_excluded_non_biallelic_loci"] == 1
    assert genotypes.shape[1] - 1 not in set(result.pairwise_sample["locus_i"])
    assert genotypes.shape[1] - 1 not in set(result.pairwise_sample["locus_j"])


def test_locus_groups_must_be_hashable(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """Invalid group identifiers should fail with an actionable error."""

    genotype_data = _fake_genotype_data(genotype_matrix, tmp_path / "bad_groups")
    analysis = LinkageDisequilibrium(
        genotype_data,
        genotype_matrix,
        allele_channels=_allele_channels(genotype_matrix),
    )
    with pytest.raises(ValueError, match="hashable"):
        analysis.run(
            locus_groups=[{"chromosome": index} for index in range(8)],
            n_bootstraps=0,
            save_plots=False,
        )


def test_coordinate_free_data_requires_explicit_unlinked_assumption(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """SNPio must not silently declare coordinate-free loci unlinked."""

    genotype_data = _fake_genotype_data(genotype_matrix, tmp_path / "coordinate_free")
    analysis = LinkageDisequilibrium(
        genotype_data,
        genotype_matrix,
        allele_channels=_allele_channels(genotype_matrix),
    )
    with pytest.raises(ValueError, match="cannot be inferred"):
        analysis.run(n_bootstraps=0, save_plots=False)


def test_vcf_chromosomes_are_used_automatically(
    tmp_path: Path, genotype_matrix: np.ndarray
) -> None:
    """VCF marker names should supply chromosome groups without an override."""

    markers = [
        "chr1:10",
        "chr1:20",
        "chr2:10",
        "chr2:20",
        "chr3:10",
        "chr3:20",
        "chr4:10",
        "chr4:20",
    ]
    genotype_data = _fake_genotype_data(
        genotype_matrix, tmp_path / "vcf_groups", marker_names=markers
    )
    result = LinkageDisequilibrium(
        genotype_data,
        genotype_matrix,
        allele_channels=_allele_channels(genotype_matrix),
    ).run(
        n_bootstraps=0,
        n_bootstrap_blocks=4,
        max_pairs=None,
        pairwise_sample_size=100,
        seed=9,
        save_pairwise=False,
        save_plots=False,
    )
    assert result.metadata["group_source"] == "vcf_chromosome"


def test_assume_unlinked_warns_when_vcf_groups_are_available(
    tmp_path: Path,
    genotype_matrix: np.ndarray,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An explicit VCF grouping override should be visible to the user."""

    markers = [
        "chr1:10",
        "chr1:20",
        "chr2:10",
        "chr2:20",
        "chr3:10",
        "chr3:20",
        "chr4:10",
        "chr4:20",
    ]
    genotype_data = _fake_genotype_data(
        genotype_matrix,
        tmp_path / "vcf_override",
        marker_names=markers,
    )
    with caplog.at_level(logging.WARNING):
        result = LinkageDisequilibrium(
            genotype_data,
            genotype_matrix,
            allele_channels=_allele_channels(genotype_matrix),
            logger=logging.getLogger("test_ld_vcf_override"),
        ).run(
            assume_unlinked=True,
            n_bootstraps=0,
            n_bootstrap_blocks=4,
            max_pairs=None,
            pairwise_sample_size=0,
            seed=91,
            save_pairwise=False,
            save_plots=False,
        )

    assert result.metadata["group_source"] == "assumed_unlinked_vcf_override"
    assert "overrides chromosome/scaffold labels" in caplog.text


def test_ld_plotter_generates_expected_files(tmp_path: Path) -> None:
    """LD visualizations should be saved in the dedicated plot directory."""

    summary = pd.DataFrame(
        {
            "Population": ["pop1", "pop2"],
            "r2D": [0.02, 0.04],
            "r2D_CI_Lower": [0.01, 0.03],
            "r2D_CI_Upper": [0.03, 0.05],
            "rDz": [0.0, 0.01],
            "rDz_CI_Lower": [-0.01, 0.001],
            "rDz_CI_Upper": [0.01, 0.02],
            "Ne": [16.7, 8.3],
            "Ne_CI_Lower": [12.0, 7.0],
            "Ne_CI_Upper": [np.inf, 10.0],
        }
    )
    pairwise = pd.DataFrame(
        {
            "Population": ["pop1"] * 4 + ["pop2"] * 4,
            "D": np.linspace(-0.1, 0.1, 8),
            "D2": np.linspace(-0.01, 0.03, 8),
            "Dz": np.linspace(-0.02, 0.02, 8),
            "Pi2": np.linspace(0.01, 0.08, 8),
            "r2_star": np.linspace(-0.1, 0.8, 8),
        }
    )
    files = LinkageDisequilibriumPlotter(
        tmp_path / "plots" / "linkage_disequilibrium",
        plot_format="png",
        dpi=300,
        fontsize=16,
        title_fontsize=18,
        despine=True,
        show=False,
    ).plot_all(summary, pairwise)

    assert LinkageDisequilibriumPlotter._use_log_scale(np.asarray([1e-4, 1.0]))
    assert not LinkageDisequilibriumPlotter._use_log_scale(
        np.asarray([-1e-4, 1.0])
    )
    assert not LinkageDisequilibriumPlotter._use_log_scale(
        np.asarray([1.0, 50.0])
    )
    assert LinkageDisequilibriumPlotter._pairwise_uses_log_scale(
        "Pi2", np.asarray([1e-4, 1.0])
    )
    assert not LinkageDisequilibriumPlotter._pairwise_uses_log_scale(
        "D2", np.asarray([1e-4, 1.0])
    )

    ordinary_style = LinkageDisequilibriumPlotter._point_style(False)
    warning_style = LinkageDisequilibriumPlotter._point_style(True)
    assert ordinary_style == {
        "color": LinkageDisequilibriumPlotter.PRIMARY_COLOR,
        "marker": LinkageDisequilibriumPlotter.ESTIMATE_MARKER,
    }
    assert warning_style == {
        "color": LinkageDisequilibriumPlotter.WARNING_COLOR,
        "marker": LinkageDisequilibriumPlotter.WARNING_MARKER,
    }
    assert ordinary_style["color"] != warning_style["color"]
    assert ordinary_style["marker"] != warning_style["marker"]

    assert LinkageDisequilibriumPlotter._diagnostic_flags(summary).tolist() == [
        False,
        True,
    ]

    assert set(files) == {"ld_population_plot", "ld_ne_plot", "ld_distribution_plot"}

    assert all(path.is_file() and path.stat().st_size > 0 for path in files.values())
