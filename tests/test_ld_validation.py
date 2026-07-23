"""Independent mathematical and pipeline validation for SNPio LD."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
from time import sleep
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scripts.validate_ld as validate_ld_script

from snpio.popgenstats.ld_polynomials import evaluate_unbiased_ld
from snpio.popgenstats.linkage_disequilibrium import LinkageDisequilibrium
from snpio.validation.ld_simulation import (
    Fwdpy11CalibrationConfig,
    _ascertain_loci,
    summarize_calibration,
)
from snpio.validation.ld_plots import (
    plot_exact_expectation_errors,
    plot_golden_reference_errors,
    plot_pair_convergence,
    plot_published_island_fox_comparison,
    plot_simulation_calibration,
)
from snpio.validation.linkage_disequilibrium import (
    COUNT_COLUMNS,
    PUBLISHED_ISLAND_FOX_ESTIMATES,
    compare_published_estimates,
    prepare_island_fox_genepop,
    summarize_convergence,
    validate_exact_expectations,
    validate_golden_reference,
    validation_data_path,
)


def _analysis(genotypes: np.ndarray, prefix: Path) -> LinkageDisequilibrium:
    """Build a minimal LD analysis without a file reader."""

    n_samples, n_loci = genotypes.shape
    data = SimpleNamespace(
        snp_data=np.full((n_samples, n_loci), "A", dtype="U1"),
        prefix=str(prefix),
        was_filtered=False,
        marker_names=None,
        samples=[f"sample_{index}" for index in range(n_samples)],
        populations=[],
        popmap=None,
        has_popmap=False,
        filetype="phylip",
        from_vcf=False,
    )
    return LinkageDisequilibrium(
        data,
        genotypes,
        logger=logging.getLogger("test_ld_validation"),
    )


def _flip_left(counts: np.ndarray) -> np.ndarray:
    return counts.reshape(3, 3)[::-1, :].reshape(9)


def _flip_right(counts: np.ndarray) -> np.ndarray:
    return counts.reshape(3, 3)[:, ::-1].reshape(9)


def _swap_loci(counts: np.ndarray) -> np.ndarray:
    return counts.reshape(3, 3).T.reshape(9)


def test_exact_multinomial_oracle_validates_unbiasedness() -> None:
    """Every small-sample exact expectation must equal its population target."""

    results = validate_exact_expectations(sample_sizes=(4, 6), atol=1e-12)

    assert len(results) == 32
    assert results["Passed"].all()
    assert results["Absolute_Error"].max() < 2e-15


def test_golden_corpus_matches_moments_popgen_1_6_0() -> None:
    """SNPio must match all frozen outputs from the paper authors' release."""

    results = validate_golden_reference()
    fixture = pd.read_csv(validation_data_path("moments_popgen_1_6_0_golden.csv.gz"))
    provenance_path = validation_data_path("moments_popgen_1_6_0_golden.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        validation_data_path("moments_popgen_1_6_0_golden.csv.gz").read_bytes()
    ).hexdigest()

    assert fixture.shape == (1_000, 15)
    assert set(COUNT_COLUMNS).issubset(fixture.columns)
    assert provenance["reference_version"] == "1.6.0"
    assert provenance["sha256"] == digest
    assert len(results) == 4_000
    assert results["Passed"].all()


def test_all_allele_and_locus_relabelings_preserve_unpolarized_statistics() -> None:
    """D2, Dz, and pi2 must be invariant under biological relabelings."""

    counts = np.asarray([8, 3, 2, 4, 5, 1, 2, 3, 7], dtype=np.int64)
    baseline = evaluate_unbiased_ld(counts[None, :])
    transforms = (
        _flip_left(counts),
        _flip_right(counts),
        _flip_right(_flip_left(counts)),
        _swap_loci(counts),
    )
    expected_d_signs = (-1.0, -1.0, 1.0, 1.0)

    for transformed_counts, d_sign in zip(transforms, expected_d_signs):
        transformed = evaluate_unbiased_ld(transformed_counts[None, :])
        assert transformed["D"][0] == pytest.approx(d_sign * baseline["D"][0])
        for statistic in ("D2", "Dz", "pi2"):
            assert transformed[statistic][0] == pytest.approx(baseline[statistic][0])


def test_two_locus_counting_matches_literal_reference(tmp_path: Path) -> None:
    """Vectorized nine-state counts must match a sample-by-sample implementation."""

    genotypes = np.asarray(
        [
            [0, 2, 1, -1],
            [1, 1, 2, 0],
            [2, 0, -1, 1],
            [0, 2, 0, 2],
            [1, -1, 1, 1],
        ],
        dtype=np.int8,
    )
    locus_i = np.asarray([0, 0, 1, 2], dtype=np.int64)
    locus_j = np.asarray([1, 3, 3, 3], dtype=np.int64)
    observed = _analysis(genotypes, tmp_path / "counts")._two_locus_counts(
        genotypes, locus_i, locus_j
    )

    expected = np.zeros_like(observed)
    for pair_index, (left_locus, right_locus) in enumerate(zip(locus_i, locus_j)):
        for sample in genotypes:
            left = int(sample[left_locus])
            right = int(sample[right_locus])
            if left >= 0 and right >= 0:
                expected[pair_index, left * 3 + right] += 1
    np.testing.assert_array_equal(observed, expected)


def test_exhaustive_pipeline_uses_each_eligible_pair_once(tmp_path: Path) -> None:
    """Pair generation and aggregate ratios must match a literal exhaustive result."""

    genotypes = np.asarray(
        [
            [0, 0, 1, 0, 2, 1],
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 1, 0],
            [2, 1, 0, 2, 0, 1],
            [0, 2, 1, 0, 1, 2],
            [1, 2, 0, 1, 0, 2],
            [2, 1, 2, 2, 0, 1],
            [2, 0, 1, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    groups = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    result = _analysis(genotypes, tmp_path / "exhaustive").run(
        locus_groups=groups,
        n_bootstraps=0,
        n_bootstrap_blocks=genotypes.shape[1],
        max_pairs=None,
        pairwise_sample_size=100,
        pair_chunk_size=2,
        seed=481,
        save_pairwise=False,
        save_plots=False,
    )
    pairwise = result.pairwise_sample
    observed_pairs = set(zip(pairwise["locus_i"], pairwise["locus_j"]))
    expected_pairs = {
        (left, right)
        for left in range(genotypes.shape[1])
        for right in range(left + 1, genotypes.shape[1])
        if groups[left] != groups[right]
    }

    assert observed_pairs == expected_pairs
    assert len(pairwise) == len(expected_pairs)
    assert (pairwise["locus_i"] < pairwise["locus_j"]).all()
    summary = result.summary.iloc[0]
    expected_r2d = pairwise["D2"].sum() / pairwise["Pi2"].sum()
    expected_rdz = pairwise["Dz"].sum() / pairwise["Pi2"].sum()
    assert summary["r2D"] == pytest.approx(expected_r2d)
    assert summary["rDz"] == pytest.approx(expected_rdz)
    assert not np.isclose(summary["r2D"], pairwise["r2_star"].mean())


def test_explicit_bootstrap_groups_preserve_biological_units(
    tmp_path: Path,
) -> None:
    """Chromosomes can replace random locus blocks during calibration."""

    genotypes = np.asarray(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2],
            [2, 1, 0, 2, 1, 0],
            [0, 2, 1, 0, 2, 1],
            [1, 2, 0, 1, 0, 2],
            [2, 1, 2, 2, 0, 1],
            [2, 0, 1, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    groups = np.asarray(["chr1", "chr1", "chr2", "chr2", "chr3", "chr3"])

    result = _analysis(genotypes, tmp_path / "biological_blocks").run(
        locus_groups=groups,
        bootstrap_groups=groups,
        n_bootstraps=4,
        max_pairs=None,
        pairwise_sample_size=0,
        seed=718,
        save_pairwise=False,
        save_plots=False,
    )

    assert result.metadata["bootstrap_group_source"] == "explicit_biological_groups"
    assert result.metadata["bootstrap_method"] == "node_cluster"
    assert result.metadata["n_bootstrap_blocks"] == 3
    assert result.metadata["candidate_pairs"] == 12


def test_bootstrap_matches_literal_seeded_resampling(tmp_path: Path) -> None:
    """Grouped bootstrap rows must reproduce literal NumPy resampling."""

    block_frame = pd.DataFrame(
        {
            "pair_count": [3, 4, 2],
            "D_sum": [0.3, -0.1, 0.2],
            "D2_sum": [0.12, 0.08, 0.03],
            "Dz_sum": [0.01, -0.02, 0.005],
            "Pi2_sum": [0.7, 0.9, 0.4],
        }
    )
    analysis = _analysis(np.zeros((4, 2), dtype=np.int8), tmp_path / "bootstrap")
    observed = analysis._bootstrap_population(
        population="p1",
        block_df=block_frame,
        n_bootstraps=12,
        mating_system="random",
        seed=719,
    )
    rng = np.random.default_rng(719)
    expected_rows = []
    for replicate in range(12):
        sampled = block_frame.iloc[
            rng.integers(0, len(block_frame), size=len(block_frame))
        ]
        expected_rows.append(
            {
                "Population": "p1",
                "Replicate": replicate,
                **analysis._aggregate_rows(sampled, "random"),
            }
        )
    pd.testing.assert_frame_equal(observed, pd.DataFrame(expected_rows))


def test_node_cluster_bootstrap_matches_literal_seeded_weights(
    tmp_path: Path,
) -> None:
    """Chromosome multiplicities must weight every incident pair together."""

    block_frame = pd.DataFrame(
        {
            "block_a": [2, 2, 5],
            "block_b": [5, 9, 9],
            "pair_count": [3, 4, 2],
            "D_sum": [0.3, -0.1, 0.2],
            "D2_sum": [0.12, 0.08, 0.03],
            "Dz_sum": [0.01, -0.02, 0.005],
            "Pi2_sum": [0.7, 0.9, 0.4],
        }
    )
    analysis = _analysis(np.zeros((4, 2), dtype=np.int8), tmp_path / "cluster")
    observed = analysis._bootstrap_population(
        population="p1",
        block_df=block_frame,
        n_bootstraps=12,
        mating_system="random",
        seed=720,
        method="node_cluster",
    )

    block_ids = np.asarray([2, 5, 9])
    block_lookup = {value: index for index, value in enumerate(block_ids)}
    left = block_frame["block_a"].map(block_lookup).to_numpy()
    right = block_frame["block_b"].map(block_lookup).to_numpy()
    rng = np.random.default_rng(720)
    expected_rows = []
    for replicate in range(12):
        sampled = rng.integers(0, len(block_ids), size=len(block_ids))
        multiplicities = np.bincount(sampled, minlength=len(block_ids))
        weights = multiplicities[left] * multiplicities[right]
        expected_rows.append(
            {
                "Population": "p1",
                "Replicate": replicate,
                **analysis._aggregate_rows(block_frame, "random", weights=weights),
            }
        )
    pd.testing.assert_frame_equal(observed, pd.DataFrame(expected_rows))


def test_population_specific_monomorphism_is_reported_without_crashing(
    tmp_path: Path,
) -> None:
    """A population with zero diversity should produce undefined ratios, not errors."""

    genotypes = np.vstack(
        [
            np.zeros((4, 4), dtype=np.int8),
            np.asarray(
                [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], [1, 0, 1, 2]],
                dtype=np.int8,
            ),
        ]
    )
    analysis = _analysis(genotypes, tmp_path / "population_monomorphic")
    analysis.genotype_data.has_popmap = True
    analysis.genotype_data.populations = ["fixed"] * 4 + ["variable"] * 4
    result = analysis.run(
        assume_unlinked=True,
        n_bootstraps=0,
        n_bootstrap_blocks=4,
        max_pairs=None,
        pairwise_sample_size=0,
        seed=90,
        save_pairwise=False,
        save_plots=False,
    )

    fixed = result.summary.set_index("Population").loc["fixed"]
    assert fixed["Pairs"] > 0
    assert np.isnan(fixed["r2D"])
    assert np.isnan(fixed["Ne"])


def test_published_and_convergence_comparison_contracts() -> None:
    """Benchmark helpers must expose explicit, machine-readable pass criteria."""

    published = pd.DataFrame(
        {
            "Population": ["San Miguel I"],
            "Ne": [15.3],
            "Ne_CI_Lower": [14.6],
            "Ne_CI_Upper": [16.0],
        }
    )
    comparison = compare_published_estimates(published)
    assert comparison.loc[0, "Passed"]
    assert comparison.loc[1:, "Passed"].eq(False).all()

    convergence = pd.DataFrame(
        {
            "Population": ["p1"] * 4,
            "max_pairs": [100, 100, 200, 200],
            "seed": [1, 2, 1, 2],
            "r2D": [0.1, 0.2, 0.12, 0.14],
            "rDz": [0.01, -0.01, 0.0, 0.01],
            "Ne": [3.3, 1.7, 2.8, 2.4],
            "Pairs": [100, 100, 200, 200],
        }
    )
    summary = summarize_convergence(convergence)
    assert summary["max_pairs"].tolist() == [100, 200]
    assert {"r2D_CV", "rDz_CV", "Ne_CV"}.issubset(summary.columns)


def test_published_genepop_normalization_assigns_unique_sample_ids(
    tmp_path: Path,
) -> None:
    """Repeated island codes in the Dryad file must become unique sample IDs."""

    sections = []
    for island in ("smi", "sri", "sci", "sca", "scl", "sni"):
        sections.extend(["POP", f"{island}, 0101 0202", f"{island}, 0202 0101"])
    source = tmp_path / "island_fox.txt"
    source.write_text(
        "Island fox benchmark\nlocus_1\nlocus_2\n" + "\n".join(sections) + "\n",
        encoding="utf-8",
    )

    normalized, popmap = prepare_island_fox_genepop(source, tmp_path / "prepared")
    sample_ids = [line.split()[0] for line in popmap.read_text().splitlines()]

    assert len(sample_ids) == 12
    assert len(set(sample_ids)) == 12
    assert sample_ids[:2] == ["SMI_001", "SMI_002"]
    assert sample_ids[-2:] == ["SNI_001", "SNI_002"]
    assert normalized.read_text().count("POP") == 6


def test_forward_locus_ascertainment_uses_the_requested_genotype_panel() -> None:
    """Sample and census ascertainment can select different variable sites."""

    config = Fwdpy11CalibrationConfig(
        population_sizes=(10,),
        sample_sizes=(4,),
        replicates=1,
        chromosomes=2,
        chromosome_length=100.0,
        loci_per_chromosome=10,
        n_bootstraps=0,
    )
    genotypes = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [2, 2, 2, 2],
            [0, 2, 1, 1],
        ],
        dtype=np.int8,
    )
    positions = np.asarray([10.0, 20.0, 110.0, 120.0])

    selected, groups, indices = _ascertain_loci(
        genotypes,
        positions,
        config=config,
        seed=721,
        source="test panel",
    )

    np.testing.assert_array_equal(indices, [1, 3])
    np.testing.assert_array_equal(groups, [0, 1])
    np.testing.assert_array_equal(selected, genotypes[[1, 3]].T)


def test_ne_interval_inverts_full_r2d_interval(tmp_path: Path) -> None:
    """Nonpositive bootstrap r2D implies an unbounded upper Ne interval."""

    block_frame = pd.DataFrame(
        {
            "pair_count": [10],
            "D_sum": [0.1],
            "D2_sum": [0.2],
            "Dz_sum": [0.0],
            "Pi2_sum": [10.0],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "r2D": [-0.01, 0.01, 0.02, 0.03],
            "rDz": [-0.02, -0.01, 0.01, 0.02],
        }
    )
    row = _analysis(
        np.zeros((4, 2), dtype=np.int8), tmp_path / "ne_interval"
    )._summarize_population(
        population="p1",
        sample_count=4,
        locus_count=2,
        block_df=block_frame,
        bootstrap_df=bootstrap,
        mating_system="random",
        alpha=0.05,
    )

    assert row["r2D_CI_Lower"] < 0.0
    assert row["Ne_CI_Lower"] == pytest.approx(1.0 / (3.0 * row["r2D_CI_Upper"]))
    assert np.isposinf(row["Ne_CI_Upper"])


def test_forward_calibration_configuration_and_summary() -> None:
    """Optional simulation results must reduce to bias and coverage diagnostics."""

    config = Fwdpy11CalibrationConfig(
        population_sizes=(10, 25),
        sample_sizes=(6, 20),
        replicates=2,
        chromosomes=2,
        n_bootstraps=0,
    )
    assert config.burnin_multiplier == 10
    assert config.allow_residual_selfing is True
    assert config.minimum_model_population_size == 100
    assert config.minimum_coverage_sample_size == 8
    assert config.model_relative_bias_tolerance == pytest.approx(0.05)
    assert config.require_population_coverage is False

    replicates = pd.DataFrame(
        {
            "Population_Size": [10, 10, 10, 10],
            "Sample_Size": [6, 6, 6, 6],
            "Pairs": [100, 100, 100, 100],
            "D2": [0.030, 0.032, 0.034, 0.036],
            "Dz": [-0.002, -0.001, 0.001, 0.002],
            "Pi2": [1.0, 1.0, 1.0, 1.0],
            "r2D": [0.030, 0.032, 0.034, 0.036],
            "r2D_CI_Lower": [0.02, 0.02, 0.02, 0.02],
            "r2D_CI_Upper": [0.04, 0.04, 0.04, 0.04],
            "rDz": [-0.002, -0.001, 0.001, 0.002],
            "Ne": [11.1, 10.4, 9.8, 9.3],
            "Ne_CI_Lower": [8.0, 8.0, 8.0, 8.0],
            "Ne_CI_Upper": [12.0, 12.0, 12.0, 12.0],
            "Matched_Census_Pairs": [100, 100, 100, 100],
            "Matched_Census_D2": [0.030, 0.032, 0.034, 0.036],
            # Sample-based ascertainment can shift matched-census rDz, so it
            # remains diagnostic rather than a formal zero-null check.
            "Matched_Census_Dz": [0.2, 0.2, 0.2, 0.2],
            "Matched_Census_Pi2": [1.0, 1.0, 1.0, 1.0],
            "Matched_Census_r2D": [0.030, 0.032, 0.034, 0.036],
            "Census_Pairs": [100, 100, 100, 100],
            "Census_D2": [0.030, 0.032, 0.034, 0.036],
            "Census_Dz": [-0.002, -0.001, 0.001, 0.002],
            "Census_Pi2": [1.0, 1.0, 1.0, 1.0],
        }
    )
    summary = summarize_calibration(replicates)
    row = summary.iloc[0]
    assert row["Target_r2D"] == pytest.approx(1.0 / 30.0)
    assert row["Mean_rDz"] == pytest.approx(0.0)
    assert row["Matched_Census_r2D_CI_Coverage"] == pytest.approx(1.0)
    assert row["Census_Ne_CI_Coverage"] == pytest.approx(1.0)
    assert row["Aggregation"] == "ratio_of_sums"
    assert row["Sample_Matched_Census_r2D_Z"] == pytest.approx(0.0)
    assert row["Passed_Calibration"]


def test_forward_calibration_pools_moments_before_forming_ratios() -> None:
    """The validation target is a ratio of expectations, not mean ratios."""

    replicates = pd.DataFrame(
        {
            "Population_Size": [10, 10],
            "Sample_Size": [6, 6],
            "Pairs": [100, 100],
            "D2": [0.09, 0.02],
            "Dz": [0.009, -0.002],
            "Pi2": [0.9, 0.1],
            "r2D": [0.1, 0.2],
            "r2D_CI_Lower": [0.05, 0.10],
            "r2D_CI_Upper": [0.15, 0.30],
            "rDz": [0.01, -0.02],
            "Ne": [10.0 / 3.0, 5.0 / 3.0],
            "Ne_CI_Lower": [2.0, 1.0],
            "Ne_CI_Upper": [5.0, 3.0],
            "Matched_Census_Pairs": [100, 100],
            "Matched_Census_D2": [0.09, 0.02],
            "Matched_Census_Dz": [0.009, -0.002],
            "Matched_Census_Pi2": [0.9, 0.1],
            "Matched_Census_r2D": [0.1, 0.2],
            "Census_Pairs": [100, 100],
            "Census_D2": [0.09, 0.02],
            "Census_Dz": [0.009, -0.002],
            "Census_Pi2": [0.9, 0.1],
        }
    )

    row = summarize_calibration(replicates).iloc[0]

    assert row["Mean_r2D"] == pytest.approx(0.11)
    assert row["Arithmetic_Mean_r2D"] == pytest.approx(0.15)
    assert row["Mean_rDz"] == pytest.approx(0.007)


def test_forward_coverage_counts_unbounded_upper_intervals() -> None:
    """A positive target is covered when the valid Ne upper bound is infinite."""

    d2 = np.linspace(0.028, 0.038, 10)
    dz = np.linspace(-0.003, 0.003, 10)
    replicates = pd.DataFrame(
        {
            "Population_Size": [10] * 10,
            "Sample_Size": [8] * 10,
            "Pairs": [100] * 10,
            "D2": d2,
            "Dz": dz,
            "Pi2": [1.0] * 10,
            "r2D": d2,
            "r2D_CI_Lower": [-0.01] * 10,
            "r2D_CI_Upper": [0.04] * 10,
            "rDz": dz,
            "Ne": 1.0 / (3.0 * d2),
            "Ne_CI_Lower": [8.0] * 10,
            "Ne_CI_Upper": [np.inf] * 10,
            "Matched_Census_Pairs": [100] * 10,
            "Matched_Census_D2": d2,
            "Matched_Census_Dz": dz,
            "Matched_Census_Pi2": [1.0] * 10,
            "Matched_Census_r2D": d2,
            "Census_Pairs": [100] * 10,
            "Census_D2": d2,
            "Census_Dz": dz,
            "Census_Pi2": [1.0] * 10,
        }
    )

    row = summarize_calibration(replicates).iloc[0]

    assert row["Coverage_Replicates"] == 10
    assert row["Census_Ne_CI_Coverage"] == pytest.approx(1.0)
    assert row["Matched_Census_r2D_CI_Coverage"] == pytest.approx(1.0)
    assert not row["Coverage_Checked"]
    assert row["Coverage_Applicability"] == "diagnostic_chromosome_only_interval"
    assert row["Passed_Coverage_Check"]

    strict = summarize_calibration(replicates, require_population_coverage=True).iloc[0]
    assert strict["Coverage_Checked"]
    assert strict["Coverage_Applicability"] == "formal_population_target_check"
    assert strict["Passed_Coverage_Check"]

    strict_failure = summarize_calibration(
        replicates.assign(Matched_Census_r2D=0.2),
        require_population_coverage=True,
    ).iloc[0]
    assert strict_failure["Matched_Census_r2D_CI_Coverage"] == pytest.approx(0.0)
    assert not strict_failure["Passed_Coverage_Check"]
    assert not strict_failure["Passed_Calibration"]

    small_sample = summarize_calibration(
        replicates.assign(Sample_Size=6, Matched_Census_r2D=0.2)
    ).iloc[0]
    assert small_sample["Matched_Census_r2D_CI_Coverage"] == pytest.approx(0.0)
    assert not small_sample["Coverage_Checked"]
    assert small_sample["Coverage_Applicability"] == "small_sample_stress_test"
    assert small_sample["Passed_Coverage_Check"]

    formal_model = summarize_calibration(replicates.assign(Population_Size=100)).iloc[0]
    assert formal_model["Model_Checks_Applied"]
    assert not formal_model["Passed_Model_Checks"]
    assert formal_model["Sample_r2D_Material_Bias_Detected"]
    assert not formal_model["Passed_Calibration"]


def test_forward_model_gate_uses_practical_bias_evidence() -> None:
    """Tiny precise deviations are diagnostics, not material model failures."""

    target = 1.0 / 300.0
    offsets = np.linspace(-0.01, 0.01, 100) * target
    dz = np.linspace(-0.001, 0.001, 100)

    def frame(relative_bias: float) -> pd.DataFrame:
        d2 = target * (1.0 + relative_bias) + offsets
        return pd.DataFrame(
            {
                "Population_Size": [100] * len(d2),
                "Sample_Size": [20] * len(d2),
                "Pairs": [100] * len(d2),
                "D2": d2,
                "Dz": dz,
                "Pi2": [1.0] * len(d2),
                "r2D": d2,
                "r2D_CI_Lower": d2 - target * 0.2,
                "r2D_CI_Upper": d2 + target * 0.2,
                "rDz": dz,
                "Ne": 1.0 / (3.0 * d2),
                "Ne_CI_Lower": [50.0] * len(d2),
                "Ne_CI_Upper": [150.0] * len(d2),
                "Matched_Census_Pairs": [100] * len(d2),
                "Matched_Census_D2": d2,
                "Matched_Census_Dz": dz,
                "Matched_Census_Pi2": [1.0] * len(d2),
                "Matched_Census_r2D": d2,
                "Census_Pairs": [100] * len(d2),
                "Census_D2": d2,
                "Census_Dz": dz,
                "Census_Pi2": [1.0] * len(d2),
            }
        )

    small_deviation = summarize_calibration(frame(relative_bias=-0.025)).iloc[0]
    assert abs(small_deviation["r2D_Z"]) > 3.0
    assert not small_deviation["Passed_Sample_Z_Checks"]
    assert not small_deviation["Sample_r2D_Material_Bias_Detected"]
    assert small_deviation["Passed_Model_Checks"]
    assert small_deviation["Passed_Calibration"]

    material_deviation = summarize_calibration(frame(relative_bias=0.10)).iloc[0]
    assert material_deviation["Sample_r2D_Material_Bias_Detected"]
    assert material_deviation["Census_r2D_Material_Bias_Detected"]
    assert not material_deviation["Passed_Model_Checks"]
    assert not material_deviation["Passed_Calibration"]


def test_validation_plot_suite_writes_discoverable_figures(tmp_path: Path) -> None:
    """Every validation layer should write its plot beneath a plots directory."""

    exact = validate_exact_expectations(sample_sizes=(4,))
    golden = validate_golden_reference().groupby("Statistic").head(8)

    published = PUBLISHED_ISLAND_FOX_ESTIMATES.copy()
    published["Ne"] = published["Published_Ne"] * 1.01
    published["Ne_CI_Lower"] = published["Published_Ne_CI_Lower"]
    published["Ne_CI_Upper"] = published["Published_Ne_CI_Upper"]

    simulation = pd.DataFrame(
        {
            "Population_Size": [10, 25, 10, 25],
            "Sample_Size": [6, 6, 8, 8],
            "Relative_r2D_Bias": [0.1, -0.05, 0.02, 0.04],
            "Pooled_Ne": [11.0, 24.0, 10.2, 26.0],
            "Census_Relative_r2D_Bias": [0.05, -0.02, 0.01, 0.02],
            "Census_Pooled_Ne": [10.5, 24.5, 10.1, 25.5],
            "Matched_Census_r2D_CI_Coverage": [0.9, 1.0, 0.95, 0.9],
            "Census_Ne_CI_Coverage": [0.8, 0.9, 0.85, 0.8],
            "Mean_rDz": [0.01, -0.01, 0.0, 0.005],
            "SE_rDz": [0.01, 0.02, 0.01, 0.01],
        }
    )
    convergence_runs = pd.DataFrame(
        {
            "Population": ["p1"] * 4,
            "max_pairs": [100, 100, 1_000, 1_000],
            "seed": [1, 2, 1, 2],
            "r2D": [0.10, 0.12, 0.105, 0.107],
            "rDz": [0.01, -0.01, 0.001, -0.001],
            "Ne": [3.3, 2.8, 3.2, 3.1],
            "Pairs": [100, 100, 1_000, 1_000],
        }
    )
    convergence = summarize_convergence(convergence_runs)

    plotters = (
        (plot_exact_expectation_errors, exact, "exact"),
        (plot_golden_reference_errors, golden, "golden"),
        (plot_published_island_fox_comparison, published, "published"),
        (plot_simulation_calibration, simulation, "simulation"),
        (plot_pair_convergence, convergence, "convergence"),
    )
    for plotter, frame, directory in plotters:
        files = plotter(
            frame,
            tmp_path / directory / "plots",
            formats=("png",),
            dpi=100,
        )
        assert set(files) == {"png"}
        assert files["png"].parent.name == "plots"
        assert files["png"].is_file()
        assert files["png"].stat().st_size > 0


def test_simulation_calibration_plot_uses_linear_population_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Ne identity reference must remain correct on a linear x-axis."""

    summary = pd.DataFrame(
        {
            "Population_Size": [10, 25, 50, 100, 400],
            "Sample_Size": [8] * 5,
            "Relative_r2D_Bias": [0.0] * 5,
            "Pooled_Ne": [10.0, 25.0, 50.0, 100.0, 400.0],
            "Matched_Census_r2D_CI_Coverage": [0.95] * 5,
            "Mean_rDz": [0.0] * 5,
            "SE_rDz": [0.001] * 5,
        }
    )
    captured = {}

    def capture_figure(figure, *_args, **_kwargs):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr("snpio.validation.ld_plots._save_figure", capture_figure)

    plot_simulation_calibration(summary, "unused", formats=("png",))
    figure = captured["figure"]
    try:
        assert all(axis.get_xscale() == "linear" for axis in figure.axes)
        identity_lines = [
            line for line in figure.axes[1].lines if line.get_linestyle() == "--"
        ]
        assert len(identity_lines) == 1
        expected = np.asarray([10.0, 25.0, 50.0, 100.0, 400.0])
        np.testing.assert_array_equal(identity_lines[0].get_xdata(), expected)
        np.testing.assert_array_equal(identity_lines[0].get_ydata(), expected)
    finally:
        figure.clear()


def test_convergence_reports_weighted_run_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long convergence runs should expose budget-weighted progress and ETA."""

    class FakeReader:
        def __init__(self, **kwargs: object) -> None:
            self.prefix = str(kwargs["prefix"])

    calls: list[tuple[int, int]] = []

    class FakeStatistics:
        def __init__(self, reader: FakeReader) -> None:
            self.reader = reader

        def calculate_linkage_disequilibrium(self, **kwargs: object) -> SimpleNamespace:
            pair_budget = int(kwargs["max_pairs"])
            seed = int(kwargs["seed"])
            calls.append((pair_budget, seed))
            return SimpleNamespace(
                summary=pd.DataFrame(
                    {
                        "Population": ["p1"],
                        "r2D": [0.1],
                        "rDz": [0.0],
                        "Ne": [10.0 / 3.0],
                        "Pairs": [pair_budget],
                    }
                ),
                metadata={"group_source": "vcf_chromosome"},
            )

    progress_instances: list[FakeProgress] = []

    class FakeProgress:
        def __init__(self, **kwargs: object) -> None:
            self.total = int(kwargs["total"])
            self.updates: list[int] = []
            self.postfixes: list[dict[str, object]] = []
            self.messages: list[str] = []
            progress_instances.append(self)

        def __enter__(self) -> FakeProgress:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def set_postfix(self, **kwargs: object) -> None:
            kwargs.pop("refresh", None)
            self.postfixes.append(kwargs)

        def update(self, value: int) -> None:
            self.updates.append(value)

        def write(self, message: str) -> None:
            self.messages.append(message)

    vcf = tmp_path / "input.vcf.gz"
    popmap = tmp_path / "input.popmap"
    vcf.touch()
    popmap.touch()
    monkeypatch.setattr(validate_ld_script, "VCFReader", FakeReader)
    monkeypatch.setattr(validate_ld_script, "PopGenStatistics", FakeStatistics)
    monkeypatch.setattr(validate_ld_script, "tqdm", FakeProgress)
    monkeypatch.setattr(validate_ld_script, "plot_pair_convergence", lambda *a, **k: {})

    passed = validate_ld_script._convergence(
        argparse.Namespace(
            vcf=vcf,
            popmap=popmap,
            output=tmp_path / "results",
            pair_budgets=[10, 20],
            seeds=[1, 2],
            n_jobs=1,
            n_bootstraps=0,
            assume_unlinked=False,
            no_progress=False,
            plot_formats=["png"],
            plot_dpi=100,
        )
    )

    assert passed
    assert calls == [(10, 1), (10, 2), (20, 1), (20, 2)]
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.total == 60
    assert progress.updates == [10, 10, 20, 20]
    assert [postfix["run"] for postfix in progress.postfixes[:4]] == [
        "1/4",
        "2/4",
        "3/4",
        "4/4",
    ]
    assert all(postfix["run_elapsed"] == "0s" for postfix in progress.postfixes[:4])
    assert len(progress.messages) == 4
    assert "Completed convergence run 4/4" in progress.messages[-1]


def test_convergence_progress_heartbeat_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One slow convergence run should keep reporting its live elapsed time."""

    class Recorder:
        def __init__(self) -> None:
            self.postfixes: list[dict[str, object]] = []

        def set_postfix(self, **kwargs: object) -> None:
            kwargs.pop("refresh", None)
            self.postfixes.append(kwargs)

    recorder = Recorder()
    monkeypatch.setattr(validate_ld_script, "CONVERGENCE_HEARTBEAT_SECONDS", 0.001)

    with validate_ld_script._progress_heartbeat(
        recorder,
        postfix={"run": "1/1", "budget": "1,000", "seed": 101},
        enabled=True,
    ):
        sleep(0.01)

    assert recorder.postfixes
    assert recorder.postfixes[-1]["run"] == "1/1"
    assert str(recorder.postfixes[-1]["run_elapsed"]).endswith("s")


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is unavailable")
def test_robust_validation_runner_has_reproducible_dry_run(tmp_path: Path) -> None:
    """The portable runner should expose robust defaults without writing files."""

    repository = Path(__file__).resolve().parents[1]
    runner = repository / "scripts" / "run_robust_ld_validation.zsh"
    output = tmp_path / "robust_validation"
    completed = subprocess.run(
        [
            "zsh",
            str(runner),
            "--dry-run",
            "--skip-published",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--replicates 250" in completed.stdout
    assert "--sample-sizes 4 6 8 20 50" in completed.stdout
    assert "--burnin-multiplier 10" in completed.stdout
    assert "--model-relative-bias-tolerance 0.05" in completed.stdout
    assert "--require-population-coverage" not in completed.stdout
    assert "--pair-budgets 250000 1000000 4000000" in completed.stdout
    assert "--seeds 101 203 307 409 503" in completed.stdout
    assert not output.exists()
