"""Forward-time calibration of SNPio LD estimates with fwdpy11 and tskit.

This module is optional validation infrastructure. Imports of ``fwdpy11``,
``msprime``, and ``tskit`` are deliberately local so production SNPio analyses
do not require simulation libraries.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from snpio.popgenstats.linkage_disequilibrium import LinkageDisequilibrium

_MODEL_BIAS_CONFIDENCE_Z = 1.959963984540054


def _library_seed(*parts: int) -> int:
    """Derive a positive uint32 seed accepted by all simulation libraries."""

    seed = int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint32)[0])
    return max(1, seed)


@dataclass(frozen=True)
class Fwdpy11CalibrationConfig:
    """Configuration for one neutral Wright-Fisher LD calibration grid."""

    population_sizes: tuple[int, ...] = (10, 25, 50, 100, 400)
    sample_sizes: tuple[int, ...] = (6, 8, 20, 50)
    replicates: int = 100
    chromosomes: int = 8
    chromosome_length: float = 1_000_000.0
    crossovers_per_chromosome: float = 0.5
    mutation_rate: float = 2e-6
    loci_per_chromosome: int = 100
    burnin_multiplier: int = 10
    allow_residual_selfing: bool = True
    n_bootstraps: int = 200
    n_bootstrap_blocks: int = 20
    minimum_model_population_size: int = 100
    minimum_coverage_sample_size: int = 8
    model_relative_bias_tolerance: float = 0.05
    require_population_coverage: bool = False
    seed: int = 20260715
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if not self.population_sizes or any(
            value < 2 for value in self.population_sizes
        ):
            raise ValueError("population_sizes must contain values of at least two.")
        if not self.sample_sizes or any(value < 4 for value in self.sample_sizes):
            raise ValueError("sample_sizes must contain values of at least four.")
        if self.replicates < 1:
            raise ValueError("replicates must be positive.")
        if self.chromosomes < 2:
            raise ValueError("chromosomes must be at least two.")
        if self.chromosome_length <= 0.0 or self.crossovers_per_chromosome < 0.0:
            raise ValueError("Chromosome lengths/rates must be non-negative.")
        if self.mutation_rate <= 0.0 or self.loci_per_chromosome < 1:
            raise ValueError("Mutation rate and loci_per_chromosome must be positive.")
        if self.burnin_multiplier < 1:
            raise ValueError("burnin_multiplier must be positive.")
        if self.minimum_model_population_size < 2:
            raise ValueError("minimum_model_population_size must be at least two.")
        if self.minimum_coverage_sample_size < 4:
            raise ValueError("minimum_coverage_sample_size must be at least four.")
        if not 0.0 < self.model_relative_bias_tolerance < 1.0:
            raise ValueError(
                "model_relative_bias_tolerance must lie strictly between zero "
                "and one."
            )
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not any(
            sample_size <= population_size
            for population_size in self.population_sizes
            for sample_size in self.sample_sizes
        ):
            raise ValueError("The calibration grid contains no valid sample sizes.")


def _simulation_dependencies() -> tuple[Any, Any, Any]:
    """Import optional simulation dependencies with an actionable error."""

    try:
        import fwdpy11
        import msprime
        import tskit
    except ImportError as error:
        raise RuntimeError(
            "Forward LD calibration requires SNPio's 'ld-validation' extra: "
            "pip install 'snpio[ld-validation]'."
        ) from error
    return fwdpy11, msprime, tskit


def _recombination_map(config: Fwdpy11CalibrationConfig, fwdpy11: Any) -> list[Any]:
    """Build independently assorting chromosomes with within-chromosome crossing over."""

    regions = []
    for chromosome in range(config.chromosomes):
        beginning = chromosome * config.chromosome_length
        end = beginning + config.chromosome_length
        regions.append(
            fwdpy11.PoissonInterval(
                beg=beginning,
                end=end,
                mean=config.crossovers_per_chromosome,
            )
        )
        if chromosome < config.chromosomes - 1:
            regions.append(fwdpy11.BinomialPoint(end, 0.5))
    return regions


def _ascertain_loci(
    genotypes: np.ndarray,
    positions: np.ndarray,
    *,
    config: Fwdpy11CalibrationConfig,
    seed: int,
    source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select variable loci independently within each simulated chromosome."""

    if genotypes.ndim != 2 or genotypes.shape[0] != positions.size:
        raise ValueError("genotypes and positions must contain the same sites.")
    sample_size = genotypes.shape[1]
    allele_counts = genotypes.sum(axis=1)
    variable = (allele_counts > 0) & (allele_counts < 2 * sample_size)
    variable_indices = np.flatnonzero(variable)
    variable_positions = positions[variable_indices]
    variable_groups = np.floor(variable_positions / config.chromosome_length).astype(
        np.int32
    )
    rng = np.random.default_rng(seed)

    selected_offsets: list[int] = []
    for chromosome in range(config.chromosomes):
        candidates = np.flatnonzero(variable_groups == chromosome)
        if candidates.size == 0:
            continue
        target = min(config.loci_per_chromosome, candidates.size)
        selected_offsets.extend(
            rng.choice(candidates, size=target, replace=False).tolist()
        )
    if not selected_offsets:
        raise RuntimeError(f"The {source} ascertainment produced no variable loci.")

    selected_offsets_array = np.asarray(
        sorted(selected_offsets, key=lambda index: variable_positions[index]),
        dtype=np.int64,
    )
    selected_groups = variable_groups[selected_offsets_array]
    if np.unique(selected_groups).size < 2:
        raise RuntimeError(
            f"The {source} ascertainment produced variable loci on fewer than "
            "two chromosomes; increase mutation_rate, chromosome_length, or "
            "population size."
        )
    selected_indices = variable_indices[selected_offsets_array]
    return genotypes[selected_indices].T, selected_groups, selected_indices


def _sample_genotypes(
    tree_sequence: Any,
    *,
    sample_size: int,
    config: Fwdpy11CalibrationConfig,
    seed: int,
    fwdpy11: Any,
    msprime: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sample, matched-census, and census-ascertained genotypes."""

    mutated = msprime.sim_mutations(
        tree_sequence,
        rate=config.mutation_rate,
        random_seed=seed,
        model=msprime.BinaryMutationModel(),
        discrete_genome=False,
    )
    decoded = fwdpy11.tskit_tools.decode_individual_metadata(mutated)
    alive = np.asarray(
        [index for index, metadata in enumerate(decoded) if metadata.alive],
        dtype=np.int64,
    )
    if alive.size < sample_size:
        raise RuntimeError(
            f"Simulation contains {alive.size} living individuals, fewer than "
            f"the requested sample of {sample_size}."
        )
    rng = np.random.default_rng(_library_seed(seed, 1))
    selected_offsets = np.sort(rng.choice(alive.size, size=sample_size, replace=False))
    census_nodes = np.asarray(
        [
            node
            for individual in alive
            for node in mutated.individual(int(individual)).nodes
        ],
        dtype=np.int32,
    )
    haplotypes = mutated.genotype_matrix(samples=census_nodes).astype(
        np.int8, copy=False
    )
    if haplotypes.shape[1] != 2 * alive.size:
        raise RuntimeError("Tree-sequence samples could not be paired into diploids.")
    census_genotypes = haplotypes[:, 0::2] + haplotypes[:, 1::2]
    sample_site_genotypes = census_genotypes[:, selected_offsets]
    positions = np.fromiter(
        (site.position for site in mutated.sites()),
        dtype=np.float64,
        count=mutated.num_sites,
    )
    sample_genotypes, sample_groups, sample_loci = _ascertain_loci(
        sample_site_genotypes,
        positions,
        config=config,
        seed=_library_seed(seed, 2),
        source="sample",
    )
    census_ascertained, census_groups, _ = _ascertain_loci(
        census_genotypes,
        positions,
        config=config,
        seed=_library_seed(seed, 3),
        source="census",
    )
    return (
        sample_genotypes,
        census_genotypes[sample_loci].T,
        sample_groups,
        census_ascertained,
        census_groups,
    )


def _analyze_genotypes(
    genotypes: np.ndarray,
    locus_groups: np.ndarray,
    *,
    config: Fwdpy11CalibrationConfig,
    prefix: Path,
    seed: int,
    n_bootstraps: int,
) -> pd.Series:
    """Analyze one genotype matrix using chromosome-level resampling units."""

    data = SimpleNamespace(
        snp_data=np.full(genotypes.shape, "A", dtype="U1"),
        prefix=str(prefix),
        was_filtered=False,
        marker_names=None,
        samples=[f"sample_{index}" for index in range(genotypes.shape[0])],
        populations=[],
        popmap=None,
        has_popmap=False,
        filetype="simulation",
        from_vcf=False,
    )
    result = LinkageDisequilibrium(data, genotypes).run(
        locus_groups=locus_groups,
        bootstrap_groups=locus_groups,
        n_bootstraps=n_bootstraps,
        n_bootstrap_blocks=min(config.n_bootstrap_blocks, genotypes.shape[1]),
        max_pairs=None,
        pairwise_sample_size=0,
        pair_chunk_size=25_000,
        seed=seed,
        save_pairwise=False,
        save_plots=False,
    )
    return result.summary.iloc[0]


def _simulate_one(
    config: Fwdpy11CalibrationConfig,
    population_size: int,
    sample_size: int,
    replicate: int,
    seed: int,
) -> dict[str, float | int]:
    """Simulate and analyze one calibration replicate."""

    fwdpy11, msprime, _ = _simulation_dependencies()
    genome_length = config.chromosomes * config.chromosome_length
    population = fwdpy11.DiploidPopulation(population_size, genome_length)
    rng = fwdpy11.GSLrng(seed)
    simlen = config.burnin_multiplier * population_size
    demography = fwdpy11.ForwardDemesGraph.tubes(
        [population_size], burnin=simlen, burnin_is_exact=True
    )
    parameters = fwdpy11.ModelParams(
        nregions=[],
        sregions=[],
        recregions=_recombination_map(config, fwdpy11),
        rates=(0.0, 0.0, None),
        gvalue=fwdpy11.Multiplicative(2.0),
        prune_selected=False,
        demography=demography,
        simlen=simlen,
        allow_residual_selfing=config.allow_residual_selfing,
    )
    fwdpy11.evolvets(
        rng,
        population,
        parameters,
        simplification_interval=max(10, min(100, population_size)),
    )
    tree_sequence = population.dump_tables_to_tskit(model_params=parameters, seed=seed)
    mutation_seed = _library_seed(seed, 1)
    (
        genotypes,
        matched_census_genotypes,
        locus_groups,
        census_genotypes,
        census_locus_groups,
    ) = _sample_genotypes(
        tree_sequence,
        sample_size=sample_size,
        config=config,
        seed=mutation_seed,
        fwdpy11=fwdpy11,
        msprime=msprime,
    )

    with TemporaryDirectory(prefix="snpio-ld-calibration-") as temporary:
        temporary_path = Path(temporary)
        row = _analyze_genotypes(
            genotypes,
            locus_groups,
            config=config,
            prefix=temporary_path / "sample",
            seed=_library_seed(seed, 2),
            n_bootstraps=config.n_bootstraps,
        )
        matched_census_row = _analyze_genotypes(
            matched_census_genotypes,
            locus_groups,
            config=config,
            prefix=temporary_path / "matched_census",
            seed=_library_seed(seed, 3),
            n_bootstraps=0,
        )
        census_row = _analyze_genotypes(
            census_genotypes,
            census_locus_groups,
            config=config,
            prefix=temporary_path / "census",
            seed=_library_seed(seed, 4),
            n_bootstraps=0,
        )
    return {
        "Population_Size": population_size,
        "Sample_Size": sample_size,
        "Sampling_Fraction": sample_size / population_size,
        "Replicate": replicate,
        "Seed": seed,
        "Loci": int(row["Loci"]),
        "Pairs": int(row["Pairs"]),
        "D2": float(row["D2"]),
        "Dz": float(row["Dz"]),
        "Pi2": float(row["Pi2"]),
        "r2D": float(row["r2D"]),
        "r2D_CI_Lower": float(row["r2D_CI_Lower"]),
        "r2D_CI_Upper": float(row["r2D_CI_Upper"]),
        "rDz": float(row["rDz"]),
        "rDz_CI_Lower": float(row["rDz_CI_Lower"]),
        "rDz_CI_Upper": float(row["rDz_CI_Upper"]),
        "Ne": float(row["Ne"]),
        "Ne_CI_Lower": float(row["Ne_CI_Lower"]),
        "Ne_CI_Upper": float(row["Ne_CI_Upper"]),
        "Matched_Census_Loci": int(matched_census_row["Loci"]),
        "Matched_Census_Pairs": int(matched_census_row["Pairs"]),
        "Matched_Census_D2": float(matched_census_row["D2"]),
        "Matched_Census_Dz": float(matched_census_row["Dz"]),
        "Matched_Census_Pi2": float(matched_census_row["Pi2"]),
        "Matched_Census_r2D": float(matched_census_row["r2D"]),
        "Matched_Census_rDz": float(matched_census_row["rDz"]),
        "Matched_Census_Ne": float(matched_census_row["Ne"]),
        "Census_Loci": int(census_row["Loci"]),
        "Census_Pairs": int(census_row["Pairs"]),
        "Census_D2": float(census_row["D2"]),
        "Census_Dz": float(census_row["Dz"]),
        "Census_Pi2": float(census_row["Pi2"]),
        "Census_r2D": float(census_row["r2D"]),
        "Census_rDz": float(census_row["rDz"]),
        "Census_Ne": float(census_row["Ne"]),
    }


def _pooled_ratio(
    group: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    pairs: str,
) -> tuple[float, float]:
    """Estimate a ratio of expectations and its replicate-clustered SE."""

    values = group[[numerator, denominator, pairs]].replace([np.inf, -np.inf], np.nan)
    valid = values.notna().all(axis=1) & (values[pairs] > 0)
    values = values.loc[valid]
    if values.empty:
        return np.nan, np.nan

    numerator_totals = values[numerator].to_numpy(dtype=float) * values[pairs].to_numpy(
        dtype=float
    )
    denominator_totals = values[denominator].to_numpy(dtype=float) * values[
        pairs
    ].to_numpy(dtype=float)
    denominator_sum = denominator_totals.sum()
    if denominator_sum == 0.0:
        return np.nan, np.nan

    ratio = float(numerator_totals.sum() / denominator_sum)
    replicate_count = len(values)
    if replicate_count < 2:
        return ratio, np.nan

    residuals = numerator_totals - ratio * denominator_totals
    standard_error = np.sqrt(
        replicate_count * np.square(residuals).sum() / (replicate_count - 1)
    ) / abs(denominator_sum)
    return ratio, float(standard_error)


def _paired_ratio_difference(
    group: pd.DataFrame,
) -> tuple[float, float]:
    """Compare sample and locus-matched census ratios with a paired SE."""

    columns = [
        "D2",
        "Pi2",
        "Pairs",
        "Matched_Census_D2",
        "Matched_Census_Pi2",
        "Matched_Census_Pairs",
    ]
    values = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
    values = values.loc[(values["Pairs"] > 0) & (values["Matched_Census_Pairs"] > 0)]
    if values.empty:
        return np.nan, np.nan

    sample_numerator = values["D2"].to_numpy(dtype=float) * values["Pairs"].to_numpy(
        dtype=float
    )
    sample_denominator = values["Pi2"].to_numpy(dtype=float) * values["Pairs"].to_numpy(
        dtype=float
    )
    census_numerator = values["Matched_Census_D2"].to_numpy(dtype=float) * values[
        "Matched_Census_Pairs"
    ].to_numpy(dtype=float)
    census_denominator = values["Matched_Census_Pi2"].to_numpy(dtype=float) * values[
        "Matched_Census_Pairs"
    ].to_numpy(dtype=float)
    if sample_denominator.sum() == 0.0 or census_denominator.sum() == 0.0:
        return np.nan, np.nan

    sample_ratio = sample_numerator.sum() / sample_denominator.sum()
    census_ratio = census_numerator.sum() / census_denominator.sum()
    difference = float(sample_ratio - census_ratio)
    if len(values) < 2:
        return difference, np.nan

    sample_influence = (
        sample_numerator - sample_ratio * sample_denominator
    ) / sample_denominator.mean()
    census_influence = (
        census_numerator - census_ratio * census_denominator
    ) / census_denominator.mean()
    standard_error = (sample_influence - census_influence).std(ddof=1) / np.sqrt(
        len(values)
    )
    return difference, float(standard_error)


def _interval_coverage(
    lower: pd.Series,
    upper: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Return valid-interval membership without discarding +infinity."""

    lower_values = pd.to_numeric(lower, errors="coerce")
    upper_values = pd.to_numeric(upper, errors="coerce")
    target_values = pd.to_numeric(target, errors="coerce")
    valid = (
        lower_values.notna()
        & upper_values.notna()
        & target_values.notna()
        & np.isfinite(lower_values)
        & ~np.isneginf(upper_values)
        & np.isfinite(target_values)
        & (upper_values >= lower_values)
    )
    return pd.DataFrame(
        {
            "Valid": valid,
            "Covered": valid
            & (lower_values <= target_values)
            & (upper_values >= target_values),
        },
        index=lower.index,
    )


def summarize_calibration(
    replicates: pd.DataFrame,
    *,
    z_threshold: float = 3.0,
    minimum_model_population_size: int = 100,
    minimum_coverage_sample_size: int = 8,
    model_relative_bias_tolerance: float = 0.05,
    require_population_coverage: bool = False,
) -> pd.DataFrame:
    """Summarize model, estimator, null, and interval calibration by cell.

    Exact-null Z scores remain visible diagnostics. Formal forward-model
    acceptance instead rejects a cell only when its 95% confidence interval
    demonstrates an absolute relative ``r2D`` bias larger than the configured
    practical tolerance. The chromosome-only interval cannot represent all
    individual-sampling uncertainty, so matched-census population coverage is
    diagnostic unless ``require_population_coverage`` is explicitly enabled.
    """

    if minimum_model_population_size < 2:
        raise ValueError("minimum_model_population_size must be at least two.")
    if minimum_coverage_sample_size < 4:
        raise ValueError("minimum_coverage_sample_size must be at least four.")
    if not 0.0 < model_relative_bias_tolerance < 1.0:
        raise ValueError(
            "model_relative_bias_tolerance must lie strictly between zero and one."
        )

    required = {
        "Population_Size",
        "Sample_Size",
        "Pairs",
        "D2",
        "Dz",
        "Pi2",
        "r2D",
        "r2D_CI_Lower",
        "r2D_CI_Upper",
        "rDz",
        "Ne",
        "Ne_CI_Lower",
        "Ne_CI_Upper",
        "Matched_Census_Pairs",
        "Matched_Census_D2",
        "Matched_Census_Dz",
        "Matched_Census_Pi2",
        "Matched_Census_r2D",
        "Census_Pairs",
        "Census_D2",
        "Census_Dz",
        "Census_Pi2",
    }
    missing = sorted(required.difference(replicates.columns))
    if missing:
        raise ValueError(
            "Forward-calibration replicates are missing pooled-moment or "
            f"census diagnostics: {missing}. Rerun the simulation."
        )

    rows = []
    for (population_size, sample_size), group in replicates.groupby(
        ["Population_Size", "Sample_Size"], sort=True
    ):
        target_r2d = 1.0 / (3.0 * population_size)
        r2d, r2d_se = _pooled_ratio(
            group, numerator="D2", denominator="Pi2", pairs="Pairs"
        )
        rdz, rdz_se = _pooled_ratio(
            group, numerator="Dz", denominator="Pi2", pairs="Pairs"
        )
        matched_census_r2d, matched_census_r2d_se = _pooled_ratio(
            group,
            numerator="Matched_Census_D2",
            denominator="Matched_Census_Pi2",
            pairs="Matched_Census_Pairs",
        )
        matched_census_rdz, matched_census_rdz_se = _pooled_ratio(
            group,
            numerator="Matched_Census_Dz",
            denominator="Matched_Census_Pi2",
            pairs="Matched_Census_Pairs",
        )
        census_r2d, census_r2d_se = _pooled_ratio(
            group,
            numerator="Census_D2",
            denominator="Census_Pi2",
            pairs="Census_Pairs",
        )
        census_rdz, census_rdz_se = _pooled_ratio(
            group,
            numerator="Census_Dz",
            denominator="Census_Pi2",
            pairs="Census_Pairs",
        )
        ratio_difference, ratio_difference_se = _paired_ratio_difference(group)

        r2d_z = (r2d - target_r2d) / r2d_se if r2d_se > 0 else np.nan
        rdz_z = rdz / rdz_se if rdz_se > 0 else np.nan
        census_r2d_z = (
            (census_r2d - target_r2d) / census_r2d_se if census_r2d_se > 0 else np.nan
        )
        census_rdz_z = census_rdz / census_rdz_se if census_rdz_se > 0 else np.nan
        sample_matched_census_z = (
            ratio_difference / ratio_difference_se
            if ratio_difference_se > 0
            else (0.0 if ratio_difference == 0.0 else np.nan)
        )
        relative_r2d_bias = (r2d - target_r2d) / target_r2d
        census_relative_r2d_bias = (census_r2d - target_r2d) / target_r2d
        relative_r2d_bias_se = r2d_se / target_r2d
        census_relative_r2d_bias_se = census_r2d_se / target_r2d
        sample_bias_lower = (
            relative_r2d_bias - _MODEL_BIAS_CONFIDENCE_Z * relative_r2d_bias_se
        )
        sample_bias_upper = (
            relative_r2d_bias + _MODEL_BIAS_CONFIDENCE_Z * relative_r2d_bias_se
        )
        census_bias_lower = (
            census_relative_r2d_bias
            - _MODEL_BIAS_CONFIDENCE_Z * census_relative_r2d_bias_se
        )
        census_bias_upper = (
            census_relative_r2d_bias
            + _MODEL_BIAS_CONFIDENCE_Z * census_relative_r2d_bias_se
        )
        sample_material_bias = bool(
            np.isfinite(sample_bias_lower)
            and np.isfinite(sample_bias_upper)
            and (
                sample_bias_upper < -model_relative_bias_tolerance
                or sample_bias_lower > model_relative_bias_tolerance
            )
        )
        census_material_bias = bool(
            np.isfinite(census_bias_lower)
            and np.isfinite(census_bias_upper)
            and (
                census_bias_upper < -model_relative_bias_tolerance
                or census_bias_lower > model_relative_bias_tolerance
            )
        )

        census_ne_coverage_frame = _interval_coverage(
            group["Ne_CI_Lower"],
            group["Ne_CI_Upper"],
            pd.Series(population_size, index=group.index, dtype=float),
        )
        census_ne_interval_count = int(census_ne_coverage_frame["Valid"].sum())
        if census_ne_interval_count:
            census_ne_coverage = float(
                census_ne_coverage_frame.loc[
                    census_ne_coverage_frame["Valid"], "Covered"
                ].mean()
            )
        else:
            census_ne_coverage = np.nan

        matched_coverage_frame = _interval_coverage(
            group["r2D_CI_Lower"],
            group["r2D_CI_Upper"],
            group["Matched_Census_r2D"],
        )
        interval_count = int(matched_coverage_frame["Valid"].sum())
        if interval_count:
            coverage = float(
                matched_coverage_frame.loc[
                    matched_coverage_frame["Valid"], "Covered"
                ].mean()
            )
        else:
            coverage = np.nan
        coverage_se = (
            np.sqrt(0.95 * 0.05 / interval_count) if interval_count >= 10 else np.nan
        )
        coverage_z = (coverage - 0.95) / coverage_se if coverage_se > 0 else np.nan

        sample_passed = bool(
            np.isfinite(r2d_z)
            and np.isfinite(rdz_z)
            and abs(r2d_z) <= z_threshold
            and abs(rdz_z) <= z_threshold
        )
        census_passed = bool(
            np.isfinite(census_r2d_z)
            and np.isfinite(census_rdz_z)
            and abs(census_r2d_z) <= z_threshold
            and abs(census_rdz_z) <= z_threshold
        )
        sample_model_passed = bool(
            np.isfinite(sample_bias_lower)
            and np.isfinite(sample_bias_upper)
            and not sample_material_bias
            and np.isfinite(rdz_z)
            and abs(rdz_z) <= z_threshold
        )
        census_model_passed = bool(
            np.isfinite(census_bias_lower)
            and np.isfinite(census_bias_upper)
            and not census_material_bias
            and np.isfinite(census_rdz_z)
            and abs(census_rdz_z) <= z_threshold
        )
        estimator_passed = bool(
            np.isfinite(sample_matched_census_z)
            and abs(sample_matched_census_z) <= z_threshold
        )
        model_checks_applied = population_size >= minimum_model_population_size
        model_passed = bool(
            not model_checks_applied or (sample_model_passed and census_model_passed)
        )
        if interval_count < 10:
            coverage_applicability = "insufficient_replicates"
        elif sample_size < minimum_coverage_sample_size:
            coverage_applicability = "small_sample_stress_test"
        elif sample_size == population_size:
            coverage_applicability = "full_census_identity_check"
        elif require_population_coverage:
            coverage_applicability = "formal_population_target_check"
        else:
            coverage_applicability = "diagnostic_chromosome_only_interval"
        coverage_checked = coverage_applicability == "formal_population_target_check"
        coverage_passed = bool(
            not coverage_checked
            or (np.isfinite(coverage_z) and abs(coverage_z) <= z_threshold)
        )

        rows.append(
            {
                "Population_Size": int(population_size),
                "Sample_Size": int(sample_size),
                "Sampling_Fraction": sample_size / population_size,
                "Replicates": int(len(group)),
                "Aggregation": "ratio_of_sums",
                "Target_r2D": target_r2d,
                "Mean_r2D": r2d,
                "SE_r2D": r2d_se,
                "Arithmetic_Mean_r2D": group["r2D"]
                .replace([np.inf, -np.inf], np.nan)
                .mean(),
                "Relative_r2D_Bias": relative_r2d_bias,
                "Sample_r2D_Relative_Bias_CI_Lower": sample_bias_lower,
                "Sample_r2D_Relative_Bias_CI_Upper": sample_bias_upper,
                "Sample_r2D_Material_Bias_Detected": sample_material_bias,
                "r2D_Z": r2d_z,
                "Mean_rDz": rdz,
                "SE_rDz": rdz_se,
                "Arithmetic_Mean_rDz": group["rDz"]
                .replace([np.inf, -np.inf], np.nan)
                .mean(),
                "rDz_Z": rdz_z,
                "Pooled_Ne": 1.0 / (3.0 * r2d) if r2d > 0 else np.nan,
                "Median_Ne": group["Ne"].replace([np.inf, -np.inf], np.nan).median(),
                "Matched_Census_r2D": matched_census_r2d,
                "Matched_Census_SE_r2D": matched_census_r2d_se,
                "Matched_Census_rDz": matched_census_rdz,
                "Matched_Census_SE_rDz": matched_census_rdz_se,
                "Matched_Census_Pooled_Ne": (
                    1.0 / (3.0 * matched_census_r2d)
                    if matched_census_r2d > 0
                    else np.nan
                ),
                "Census_r2D": census_r2d,
                "Census_SE_r2D": census_r2d_se,
                "Census_Relative_r2D_Bias": census_relative_r2d_bias,
                "Census_r2D_Relative_Bias_CI_Lower": census_bias_lower,
                "Census_r2D_Relative_Bias_CI_Upper": census_bias_upper,
                "Census_r2D_Material_Bias_Detected": census_material_bias,
                "Census_r2D_Z": census_r2d_z,
                "Census_rDz": census_rdz,
                "Census_SE_rDz": census_rdz_se,
                "Census_rDz_Z": census_rdz_z,
                "Census_Pooled_Ne": (
                    1.0 / (3.0 * census_r2d) if census_r2d > 0 else np.nan
                ),
                "Sample_Matched_Census_r2D_Difference": ratio_difference,
                "SE_Sample_Matched_Census_r2D_Difference": ratio_difference_se,
                "Sample_Matched_Census_r2D_Z": sample_matched_census_z,
                "Matched_Census_r2D_CI_Coverage": coverage,
                "Coverage_Replicates": interval_count,
                "Coverage_Z": coverage_z,
                "Census_Ne_CI_Coverage": census_ne_coverage,
                "Census_Ne_Coverage_Replicates": census_ne_interval_count,
                "Passed_Sample_Z_Checks": sample_passed,
                "Passed_Census_Z_Checks": census_passed,
                "Model_r2D_Relative_Bias_Tolerance": (model_relative_bias_tolerance),
                "Passed_Sample_Model_Check": sample_model_passed,
                "Passed_Census_Model_Check": census_model_passed,
                "Model_Checks_Applied": model_checks_applied,
                "Passed_Model_Checks": model_passed,
                "Passed_Estimator_Z_Check": estimator_passed,
                "Coverage_Checked": coverage_checked,
                "Population_Coverage_Required": require_population_coverage,
                "Coverage_Applicability": coverage_applicability,
                "Passed_Coverage_Check": coverage_passed,
                "Passed_Acceptance_Checks": model_passed and estimator_passed,
                "Passed_Z_Checks": model_passed and estimator_passed,
                "Passed_Calibration": model_passed
                and estimator_passed
                and coverage_passed,
            }
        )
    return pd.DataFrame(rows)


def run_fwdpy11_calibration(
    config: Fwdpy11CalibrationConfig,
    *,
    output_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a forward-time grid and write replicate, summary, and config files."""

    _simulation_dependencies()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for population_size in config.population_sizes:
        for sample_size in config.sample_sizes:
            if sample_size > population_size:
                continue
            for replicate in range(config.replicates):
                seed = _library_seed(
                    config.seed, population_size, sample_size, replicate
                )
                tasks.append((population_size, sample_size, replicate, seed))

    resolved_jobs = config.n_jobs
    if resolved_jobs == -1:
        import os

        resolved_jobs = max(1, os.cpu_count() or 1)

    if resolved_jobs == 1:
        rows = [
            _simulate_one(config, *task) for task in tqdm(tasks, desc="LD calibration")
        ]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=resolved_jobs) as executor:
            futures = {
                executor.submit(_simulate_one, config, *task): task for task in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="LD calibration"
            ):
                rows.append(future.result())
    replicates = pd.DataFrame(rows)
    summary = summarize_calibration(
        replicates,
        minimum_model_population_size=config.minimum_model_population_size,
        minimum_coverage_sample_size=config.minimum_coverage_sample_size,
        model_relative_bias_tolerance=config.model_relative_bias_tolerance,
        require_population_coverage=config.require_population_coverage,
    )
    replicates.to_csv(output / "ld_calibration_replicates.csv", index=False)
    summary.to_csv(output / "ld_calibration_summary.csv", index=False)
    (output / "ld_calibration_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    inferential_checks_applied = config.replicates >= 10
    failed = summary.loc[
        inferential_checks_applied & ~summary["Passed_Calibration"],
        [
            "Population_Size",
            "Sample_Size",
            "Passed_Sample_Z_Checks",
            "Passed_Census_Z_Checks",
            "Passed_Sample_Model_Check",
            "Passed_Census_Model_Check",
            "Model_Checks_Applied",
            "Passed_Model_Checks",
            "Passed_Estimator_Z_Check",
            "Coverage_Checked",
            "Coverage_Applicability",
            "Passed_Coverage_Check",
        ],
    ]
    status = {
        "inferential_checks_applied": inferential_checks_applied,
        "acceptance_contract": {
            "model_relative_bias_tolerance": (config.model_relative_bias_tolerance),
            "model_bias_confidence_level": 0.95,
            "population_coverage_required": (config.require_population_coverage),
            "chromosome_only_population_coverage_is_diagnostic": (
                not config.require_population_coverage
            ),
        },
        "passed": (
            bool(summary["Passed_Calibration"].all())
            if inferential_checks_applied
            else None
        ),
        "cells": int(len(summary)),
        "failed_cells": failed.to_dict(orient="records"),
    }
    (output / "ld_calibration_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return replicates, summary


__all__ = [
    "Fwdpy11CalibrationConfig",
    "run_fwdpy11_calibration",
    "summarize_calibration",
]
