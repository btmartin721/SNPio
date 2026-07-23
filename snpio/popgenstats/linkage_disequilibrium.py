"""Unbiased linkage-disequilibrium statistics for unphased diploid SNPs."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd
import tqdm

from snpio.popgenstats.ld_polynomials import LD_POLYNOMIALS, evaluate_unbiased_ld
from snpio.utils.output_paths import OutputPaths

if TYPE_CHECKING:
    import logging

    from snpio.plotting.plotting import Plotting
    from snpio.read_input.genotype_data import GenotypeData


MatingSystem = Literal["random", "monogamous"]
BootstrapMethod = Literal["block_pair", "node_cluster"]


@dataclass
class LinkageDisequilibriumResult:
    """Container returned by an unbiased LD analysis."""

    summary: pd.DataFrame
    bootstrap: pd.DataFrame
    block_summaries: pd.DataFrame
    pairwise_sample: pd.DataFrame
    metadata: dict[str, Any]
    files: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class _BlockPairTask:
    """A deterministic unit of work for one pair of bootstrap blocks."""

    block_a: int
    block_b: int
    left_loci: np.ndarray
    right_loci: np.ndarray
    candidate_pairs: int
    target_pairs: int
    pairwise_sample_size: int
    seed: int


class LinkageDisequilibrium:
    """Calculate unbiased LD moments from unphased diploid genotypes.

    The implementation follows Ragsdale and Gravel (2020). It computes the
    unbiased Hill-Robertson statistics ``D2``, ``Dz``, and ``pi2`` from the
    nine possible two-locus diploid genotype counts. The normalized
    population-level statistics are ratios of pairwise sums:

    ``r2D = sum(D2) / sum(pi2)`` and ``rDz = sum(Dz) / sum(pi2)``.

    The class deliberately does not report the mean of pairwise ``D2/pi2`` as
    ``r2D``. That pairwise ratio, exposed only as ``r2_star`` in sampled
    outputs, remains biased and may fall outside ``[0, 1]``.
    """

    _PAIRWISE_COLUMNS = [
        "locus_i",
        "locus_j",
        "D",
        "D2",
        "Dz",
        "Pi2",
        "r2_star",
        "n_complete",
    ]

    def __init__(
        self,
        genotype_data: "GenotypeData",
        alignment_012: np.ndarray,
        *,
        allele_channels: tuple[np.ndarray, np.ndarray] | None = None,
        plotter: "Plotting | None" = None,
        logger: "logging.Logger | None" = None,
    ) -> None:
        """Initialize an LD analysis from SNPio's shared genotype model."""

        self.genotype_data = genotype_data
        self.plotter = plotter
        self.logger = logger

        alignment = np.asarray(alignment_012)

        if alignment.ndim != 2:
            msg = "alignment_012 must be a two-dimensional array."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if alignment.shape != genotype_data.snp_data.shape:
            msg = "alignment_012 must have the same sample-by-locus shape as  genotype_data.snp_data."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if not np.all(np.isin(alignment, (-9, -1, 0, 1, 2))):
            msg = "LD genotypes must be encoded as 0, 1, 2, or missing."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        self.alignment = alignment.astype(np.int8, copy=True)
        self.alignment[self.alignment < 0] = -1
        self.allele_channels = allele_channels
        self.n_samples, self.n_loci = self.alignment.shape
        self.locus_labels = self._resolve_locus_labels()

        self.report_dir = OutputPaths.from_genotype_data(genotype_data).reports(
            "linkage_disequilibrium"
        )

    def run(
        self,
        *,
        populations: Sequence[str | int] | str | int | None = None,
        include_overall: bool = False,
        locus_groups: Sequence[Hashable] | None = None,
        bootstrap_groups: Sequence[Hashable] | None = None,
        assume_unlinked: bool = False,
        n_bootstraps: int = 200,
        n_bootstrap_blocks: int = 20,
        n_jobs: int = 1,
        max_pairs: int | None = 1_000_000,
        pair_chunk_size: int = 25_000,
        pairwise_sample_size: int = 100_000,
        mating_system: MatingSystem = "random",
        alpha: float = 0.05,
        seed: int | None = None,
        save_pairwise: bool = True,
        save_plots: bool = True,
    ) -> LinkageDisequilibriumResult:
        """Calculate unbiased LD statistics and bootstrap confidence intervals.

        Args:
            populations: Population IDs to analyze. By default, every
                population in the population map is analyzed; without a
                population map, all samples are treated as one population.
            include_overall: Also analyze all samples together when a
                population map is present.
            locus_groups: Optional chromosome, scaffold, or independently
                segregating group for each locus. Only pairs from different
                groups are used. When omitted for VCF input, chromosome names
                are parsed from SNPio marker names.
            bootstrap_groups: Optional biological resampling group for every
                locus. When supplied, these groups replace random bootstrap
                blocks and SNPio applies a node-cluster bootstrap, weighting
                each block-pair summary by the product of the two sampled
                group multiplicities. This is useful when chromosomes, rather
                than individual SNPs or chromosome pairs, are the independent
                units. The groups must be consistent with ``locus_groups``.
            assume_unlinked: Treat all supplied loci as unlinked. This must be
                explicitly enabled for coordinate-free input.
            n_bootstraps: Number of grouped-locus bootstrap replicates.
            n_bootstrap_blocks: Number of random locus groups used by the
                Ragsdale-Gravel grouped bootstrap.
            n_jobs: Number of NumPy worker threads. ``-1`` uses all CPUs.
            max_pairs: Maximum uniformly sampled locus pairs. ``None`` uses
                all eligible pairs.
            pair_chunk_size: Maximum locus pairs evaluated in one vectorized
                chunk.
            pairwise_sample_size: Maximum pairwise rows retained for plots and
                optional output. Aggregate estimates use up to ``max_pairs``.
            mating_system: ``"random"`` uses ``Ne = 1 / (3 r2D)``;
                ``"monogamous"`` uses ``Ne = 2 / (3 r2D)``.
            alpha: Two-sided confidence-interval error rate.
            seed: Random seed controlling locus blocks, pair sampling, and
                bootstrap resampling.
            save_pairwise: Write the retained pairwise sample to CSV.GZ.
            save_plots: Generate LD summary visualizations.

        Returns:
            A :class:`LinkageDisequilibriumResult` containing summary,
            bootstrap, block-level, and sampled pairwise tables.
        """

        self._validate_run_arguments(
            n_bootstraps=n_bootstraps,
            n_bootstrap_blocks=n_bootstrap_blocks,
            n_jobs=n_jobs,
            max_pairs=max_pairs,
            pair_chunk_size=pair_chunk_size,
            pairwise_sample_size=pairwise_sample_size,
            mating_system=mating_system,
            alpha=alpha,
        )

        root_seed = self._resolve_seed(seed)

        valid_loci = self._valid_biallelic_loci()

        if valid_loci.size < 2:
            msg = "At least two biallelic loci are required for LD."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        resolved_groups, group_source = self._resolve_locus_groups(
            locus_groups=locus_groups,
            assume_unlinked=assume_unlinked,
        )
        population_indices = self._resolve_populations(populations, include_overall)

        blocks, bootstrap_group_source = self._resolve_bootstrap_blocks(
            valid_loci=valid_loci,
            bootstrap_groups=bootstrap_groups,
            n_bootstrap_blocks=n_bootstrap_blocks,
            seed=root_seed,
        )

        bootstrap_method: BootstrapMethod = (
            "node_cluster"
            if bootstrap_group_source == "explicit_biological_groups"
            else "block_pair"
        )

        tasks = self._build_tasks(
            valid_loci=valid_loci,
            blocks=blocks,
            locus_groups=resolved_groups,
            max_pairs=max_pairs,
            pairwise_sample_size=pairwise_sample_size,
            root_seed=root_seed,
        )

        if not tasks:
            msg = "No eligible unlinked locus pairs were found. Supply independent locus_groups or explicitly set assume_unlinked=True."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if self.logger is not None:
            total_candidates = sum(task.candidate_pairs for task in tasks)
            total_targets = sum(task.target_pairs for task in tasks)

            self.logger.info(
                f"Calculating unbiased LD for {len(population_indices)} population set(s), {valid_loci.size} biallelic loci, and {total_targets}/{total_candidates} eligible locus pairs.",
            )

        block_frames: list[pd.DataFrame] = []
        pairwise_frames: list[pd.DataFrame] = []
        summary_rows: list[dict[str, Any]] = []
        bootstrap_frames: list[pd.DataFrame] = []
        resolved_jobs = self._resolve_n_jobs(n_jobs)

        for population_number, (population, sample_indices) in tqdm.tqdm(
            enumerate(population_indices.items()),
            total=len(population_indices),
            desc="Populations",
            unit="population",
            leave=True,
            position=0,
        ):
            if sample_indices.size < 4:
                msg = f"Population '{population}' has {sample_indices.size} samples; the fourth-order unbiased LD estimators require at least four."
                self.logger.error(msg) if self.logger is not None else None
                raise ValueError(msg)

            genotype_matrix = self.alignment[sample_indices]

            block_df, pairwise_df = self._compute_population(
                population=str(population),
                population_number=population_number,
                genotype_matrix=genotype_matrix,
                tasks=tasks,
                locus_groups=resolved_groups,
                pair_chunk_size=pair_chunk_size,
                n_jobs=resolved_jobs,
            )

            if block_df.empty or int(block_df["pair_count"].sum()) == 0:
                msg = f"Population '{population}' has no locus pairs with at least four complete diploid genotypes."
                self.logger.error(msg) if self.logger is not None else None
                raise ValueError(msg)

            population_bootstrap = self._bootstrap_population(
                population=str(population),
                block_df=block_df,
                n_bootstraps=n_bootstraps,
                mating_system=mating_system,
                seed=self._task_seed(root_seed, population_number, 911, 353),
                method=bootstrap_method,
            )

            summary_rows.append(
                self._summarize_population(
                    population=str(population),
                    sample_count=sample_indices.size,
                    locus_count=valid_loci.size,
                    block_df=block_df,
                    bootstrap_df=population_bootstrap,
                    mating_system=mating_system,
                    alpha=alpha,
                )
            )

            block_frames.append(block_df)

            if not pairwise_df.empty:
                pairwise_frames.append(pairwise_df)

            if not population_bootstrap.empty:
                bootstrap_frames.append(population_bootstrap)

        summary = pd.DataFrame(summary_rows)

        non_estimable = summary.loc[
            ~np.isfinite(summary["Ne"].to_numpy(dtype=np.float64)),
            ["Population", "r2D"],
        ]

        if self.logger is not None and not non_estimable.empty:
            details = ", ".join(
                f"{row.Population} (r2D={row.r2D:.6g})"
                for row in non_estimable.itertuples(index=False)
            )

            self.logger.warning(
                f"Recent Ne is not estimable for {details} because aggregate r2D is nonpositive or nonfinite. The unbiased r2D estimate is retained and Ne is reported as NaN.",
            )

        block_summaries = pd.concat(block_frames, ignore_index=True)

        pairwise_sample = self._concat_or_empty(
            pairwise_frames,
            ["Population", *self._PAIRWISE_COLUMNS, "locus_i_label", "locus_j_label"],
        )

        bootstrap = self._concat_or_empty(
            bootstrap_frames,
            ["Population", "Replicate", "D", "D2", "Dz", "Pi2", "r2D", "rDz", "Ne"],
        )

        total_candidate_pairs = int(sum(task.candidate_pairs for task in tasks))

        total_requested_pairs = int(sum(task.target_pairs for task in tasks))

        metadata: dict[str, Any] = {
            "method": "Ragsdale-Gravel unbiased unphased LD",
            "doi": "10.1093/molbev/msz265",
            "statistics": ["D", "D2", "Dz", "Pi2", "r2D", "rDz", "Ne"],
            "polynomial_term_counts": {
                name: len(terms.coefficients) for name, terms in LD_POLYNOMIALS.items()
            },
            "group_source": group_source,
            "assume_unlinked": assume_unlinked,
            "n_biallelic_loci": int(valid_loci.size),
            "n_excluded_non_biallelic_loci": int(self.n_loci - valid_loci.size),
            "n_bootstrap_blocks": int(np.unique(blocks[blocks >= 0]).size),
            "bootstrap_group_source": bootstrap_group_source,
            "bootstrap_method": bootstrap_method,
            "n_bootstraps": n_bootstraps,
            "candidate_pairs": total_candidate_pairs,
            "requested_pairs_per_population": total_requested_pairs,
            "analyzed_pairs_per_population": {
                str(row["Population"]): int(row["Pairs"]) for row in summary_rows
            },
            "max_pairs": max_pairs,
            "pairwise_sample_size": pairwise_sample_size,
            "pair_chunk_size": pair_chunk_size,
            "n_jobs": resolved_jobs,
            "parallel_backend": "thread",
            "mating_system": mating_system,
            "alpha": alpha,
            "seed": root_seed,
            "pairwise_ratio_warning": (
                "r2_star is the biased pairwise D2/Pi2 ratio and may be negative "
                "or greater than one; r2D is the ratio of aggregate sums."
            ),
        }

        result = LinkageDisequilibriumResult(
            summary=summary,
            bootstrap=bootstrap,
            block_summaries=block_summaries,
            pairwise_sample=pairwise_sample,
            metadata=metadata,
        )

        result.files.update(self._write_reports(result, save_pairwise=save_pairwise))

        if save_plots and self.plotter is not None:
            plot_files = self.plotter.plot_linkage_disequilibrium(
                summary=summary, pairwise=pairwise_sample
            )

            result.files.update(plot_files)

        return result

    def _validate_run_arguments(
        self,
        *,
        n_bootstraps: int,
        n_bootstrap_blocks: int,
        n_jobs: int,
        max_pairs: int | None,
        pair_chunk_size: int,
        pairwise_sample_size: int,
        mating_system: MatingSystem,
        alpha: float,
    ) -> None:
        """Validate public run arguments before allocating analysis arrays."""

        if n_bootstraps < 0:
            msg = "n_bootstraps must be non-negative."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if n_bootstrap_blocks < 2:
            msg = "n_bootstrap_blocks must be at least two."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if n_jobs == 0 or n_jobs < -1:
            msg = "n_jobs must be -1 or a positive integer."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if max_pairs is not None and max_pairs < 1:
            msg = "max_pairs must be positive or None."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if pair_chunk_size < 1:
            msg = "pair_chunk_size must be positive."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if pairwise_sample_size < 0:
            msg = "pairwise_sample_size must be non-negative."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if mating_system not in {"random", "monogamous"}:
            msg = "mating_system must be 'random' or 'monogamous'."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if not 0.0 < alpha < 1.0:
            msg = "alpha must fall strictly between zero and one."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

    @staticmethod
    def _resolve_seed(seed: int | None) -> int:
        """Return a concrete non-negative seed and preserve reproducibility."""

        if seed is None:
            return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

        if seed < 0:
            raise ValueError("seed must be non-negative.")

        return int(seed)

    @staticmethod
    def _resolve_n_jobs(n_jobs: int) -> int:
        """Normalize SNPio's conventional ``n_jobs=-1`` setting."""

        return max(1, os.cpu_count() or 1) if n_jobs == -1 else n_jobs

    @staticmethod
    def _task_seed(root_seed: int, *parts: int) -> int:
        """Derive deterministic, order-independent child seeds."""

        state = np.random.SeedSequence([root_seed, *parts]).generate_state(
            1, dtype=np.uint32
        )

        return int(state[0])

    def _resolve_locus_labels(self) -> np.ndarray:
        """Return stable display labels for every locus."""

        marker_names = getattr(self.genotype_data, "marker_names", None)
        if marker_names is not None and len(marker_names) == self.n_loci:
            return np.asarray(marker_names, dtype=object)
        return np.asarray([f"locus_{index + 1}" for index in range(self.n_loci)])

    def _valid_biallelic_loci(self) -> np.ndarray:
        """Identify loci with exactly two observed biological alleles."""

        if self.allele_channels is None:
            heterozygous = np.any(self.alignment == 1, axis=0)
            both_homozygotes = np.any(self.alignment == 0, axis=0) & np.any(
                self.alignment == 2, axis=0
            )
            return np.flatnonzero(heterozygous | both_homozygotes)

        allele_a, allele_b = (np.asarray(channel) for channel in self.allele_channels)

        if (
            allele_a.shape != self.alignment.shape
            or allele_b.shape != self.alignment.shape
        ):
            msg = "allele_channels must match the genotype alignment shape."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        present = np.zeros((4, self.n_loci), dtype=bool)

        for allele in range(4):
            present[allele] = np.any(
                (allele_a == allele) | (allele_b == allele), axis=0
            )

        distinct = present.sum(axis=0)
        return np.flatnonzero(distinct == 2)

    def _resolve_locus_groups(
        self,
        *,
        locus_groups: Sequence[Hashable] | None,
        assume_unlinked: bool,
    ) -> tuple[np.ndarray | None, str]:
        """Resolve explicit or VCF-derived independently segregating groups."""

        if locus_groups is not None:
            groups = np.asarray(locus_groups, dtype=object)
            if groups.ndim != 1 or groups.size != self.n_loci:
                msg = "locus_groups must contain one value per locus."
                self.logger.error(msg) if self.logger is not None else None
                raise ValueError(msg)

            for group in groups.tolist():
                if group is None or (isinstance(group, float) and np.isnan(group)):
                    msg = "locus_groups cannot contain missing values."
                    self.logger.error(msg) if self.logger is not None else None
                    raise ValueError(msg)

                try:
                    hash(group)
                except TypeError as error:
                    msg = "Every locus_groups value must be hashable."
                    self.logger.error(msg) if self.logger is not None else None
                    raise ValueError(msg) from error

            return groups, "explicit"

        is_vcf = (
            bool(getattr(self.genotype_data, "from_vcf", False))
            or str(getattr(self.genotype_data, "filetype", "")).lower() == "vcf"
        )
        vcf_groups: np.ndarray | None = None
        if is_vcf and all(":" in str(label) for label in self.locus_labels):
            candidate_groups = np.asarray(
                [str(label).rsplit(":", maxsplit=1)[0] for label in self.locus_labels],
                dtype=object,
            )
            if len(set(candidate_groups.tolist())) >= 2:
                vcf_groups = candidate_groups

        if assume_unlinked:
            if vcf_groups is not None:
                message = "assume_unlinked=True overrides chromosome/scaffold labels available in the VCF; within-group locus pairs will be included. Omit assume_unlinked to restrict the analysis to cross-group pairs."

                if self.logger is not None:
                    self.logger.warning(message)

                return None, "assumed_unlinked_vcf_override"

            return None, "assumed_unlinked"

        if vcf_groups is not None:
            return vcf_groups, "vcf_chromosome"

        msg = "Unlinked loci cannot be inferred from this dataset. Supply locus_groups or explicitly set assume_unlinked=True."
        self.logger.error(msg) if self.logger is not None else None
        raise ValueError(msg)

    def _resolve_populations(
        self,
        populations: Sequence[str | int] | str | int | None,
        include_overall: bool,
    ) -> dict[str | int, np.ndarray]:
        """Map requested population IDs to sample row indices."""

        has_popmap = bool(getattr(self.genotype_data, "has_popmap", False))

        if not has_popmap:
            if populations is not None:
                msg = "populations requires a population map."
                self.logger.error(msg) if self.logger is not None else None
                raise ValueError(msg)

            return {"Overall": np.arange(self.n_samples, dtype=np.int64)}

        sample_populations = np.asarray(self.genotype_data.populations, dtype=object)

        available = list(dict.fromkeys(sample_populations.tolist()))

        if populations is None:
            requested = available
        elif isinstance(populations, (str, int)):
            requested = [populations]
        else:
            requested = list(populations)

        if not requested:
            msg = "populations cannot be empty."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        missing = [
            population for population in requested if population not in available
        ]

        if missing:
            msg = f"populations {missing} are not present in the population map."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        resolved = {
            population: np.flatnonzero(sample_populations == population)
            for population in requested
        }

        if include_overall:
            resolved["Overall"] = np.arange(self.n_samples, dtype=np.int64)

        return resolved

    def _assign_bootstrap_blocks(
        self, valid_loci: np.ndarray, n_bootstrap_blocks: int, seed: int
    ) -> np.ndarray:
        """Randomly and evenly assign valid loci to grouped-bootstrap blocks."""

        n_blocks = min(n_bootstrap_blocks, valid_loci.size)
        rng = np.random.default_rng(self._task_seed(seed, 701, n_blocks))
        permuted = rng.permutation(valid_loci)
        blocks = np.full(self.n_loci, -1, dtype=np.int32)
        blocks[permuted] = np.arange(valid_loci.size, dtype=np.int32) % n_blocks
        return blocks

    def _resolve_bootstrap_blocks(
        self,
        *,
        valid_loci: np.ndarray,
        bootstrap_groups: Sequence[Hashable] | None,
        n_bootstrap_blocks: int,
        seed: int,
    ) -> tuple[np.ndarray, str]:
        """Resolve random or explicitly supplied biological bootstrap units."""

        if bootstrap_groups is None:
            return (
                self._assign_bootstrap_blocks(valid_loci, n_bootstrap_blocks, seed),
                "random_locus_groups",
            )

        groups = np.asarray(bootstrap_groups, dtype=object)

        if groups.ndim != 1 or groups.size != self.n_loci:
            msg = "bootstrap_groups must contain one value per locus."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        blocks = np.full(self.n_loci, -1, dtype=np.int32)
        identifiers: dict[Hashable, int] = {}

        for locus in valid_loci.tolist():
            group = groups[locus]

            if group is None or (isinstance(group, float) and np.isnan(group)):
                msg = "bootstrap_groups cannot contain missing values."
                self.logger.error(msg) if self.logger is not None else None
                raise ValueError(msg)

            try:
                group_id = identifiers.setdefault(group, len(identifiers))
            except TypeError as error:
                msg = "Every bootstrap_groups value must be hashable."
                self.logger.error(msg) if self.logger is not None else None
                raise ValueError(msg) from error

            blocks[locus] = group_id

        if len(identifiers) < 2:
            msg = "bootstrap_groups must contain at least two groups among biallelic loci."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        return blocks, "explicit_biological_groups"

    @staticmethod
    def _grouped_indices(
        indices: np.ndarray, groups: np.ndarray | None
    ) -> dict[Hashable, np.ndarray]:
        """Partition locus indices without requiring sortable group labels."""

        if groups is None:
            return {"__all__": indices}
        grouped: dict[Hashable, list[int]] = {}
        for index in indices.tolist():
            grouped.setdefault(groups[index], []).append(index)

        return {
            group: np.asarray(group_indices, dtype=np.int64)
            for group, group_indices in grouped.items()
        }

    @classmethod
    def _candidate_segments(
        cls,
        left_loci: np.ndarray,
        right_loci: np.ndarray,
        locus_groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Return disjoint cross-products that satisfy the unlinked-group rule."""

        left_groups = cls._grouped_indices(left_loci, locus_groups)
        right_groups = cls._grouped_indices(right_loci, locus_groups)
        segments: list[tuple[np.ndarray, np.ndarray, int]] = []
        for left_group, left_indices in left_groups.items():
            for right_group, right_indices in right_groups.items():
                if locus_groups is not None and left_group == right_group:
                    continue

                count = int(left_indices.size * right_indices.size)

                if count:
                    segments.append((left_indices, right_indices, count))

        return segments

    @staticmethod
    def _allocate_budget(counts: Sequence[int], budget: int | None) -> np.ndarray:
        """Allocate an integer sampling budget proportionally and deterministically."""

        count_array = np.asarray(counts, dtype=np.int64)
        total = int(count_array.sum())

        if total == 0:
            return np.zeros_like(count_array)

        if budget is None or budget >= total:
            return count_array.copy()

        exact = count_array.astype(np.float64) * (budget / total)
        allocation = np.floor(exact).astype(np.int64)
        remainder = budget - int(allocation.sum())

        if remainder:
            fractional = exact - allocation
            order = np.lexsort((np.arange(count_array.size), -fractional))

            for index in order[:remainder]:
                if allocation[index] < count_array[index]:
                    allocation[index] += 1

        return allocation

    def _build_tasks(
        self,
        *,
        valid_loci: np.ndarray,
        blocks: np.ndarray,
        locus_groups: np.ndarray | None,
        max_pairs: int | None,
        pairwise_sample_size: int,
        root_seed: int,
    ) -> list[_BlockPairTask]:
        """Create block-pair tasks and proportionally allocate pair budgets."""

        block_ids = sorted(set(blocks[valid_loci].tolist()))
        task_parts: list[tuple[int, int, np.ndarray, np.ndarray, int]] = []
        for offset, block_a in enumerate(block_ids[:-1]):
            left_loci = valid_loci[blocks[valid_loci] == block_a]

            for block_b in block_ids[offset + 1 :]:
                right_loci = valid_loci[blocks[valid_loci] == block_b]
                candidate_pairs = sum(
                    segment[2]
                    for segment in self._candidate_segments(
                        left_loci, right_loci, locus_groups
                    )
                )

                if candidate_pairs:
                    task_parts.append(
                        (block_a, block_b, left_loci, right_loci, candidate_pairs)
                    )

        if not task_parts:
            return []

        pair_allocations = self._allocate_budget(
            [part[4] for part in task_parts], max_pairs
        )

        nonzero = pair_allocations > 0

        if not np.any(nonzero):
            return []

        sample_allocations = self._allocate_budget(
            pair_allocations.tolist(), pairwise_sample_size
        )

        tasks = []
        for task_number, (part, target, sample_target) in enumerate(
            zip(task_parts, pair_allocations, sample_allocations)
        ):
            if target == 0:
                continue

            block_a, block_b, left_loci, right_loci, candidate_pairs = part

            tasks.append(
                _BlockPairTask(
                    block_a=block_a,
                    block_b=block_b,
                    left_loci=left_loci,
                    right_loci=right_loci,
                    candidate_pairs=candidate_pairs,
                    target_pairs=int(target),
                    pairwise_sample_size=int(sample_target),
                    seed=self._task_seed(root_seed, task_number, block_a, block_b),
                )
            )

        return tasks

    @classmethod
    def _iter_pair_chunks(
        cls,
        task: _BlockPairTask,
        locus_groups: np.ndarray | None,
        chunk_size: int,
        rng: np.random.Generator,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield all or uniformly sampled eligible pairs in bounded arrays."""

        segments = cls._candidate_segments(
            task.left_loci, task.right_loci, locus_groups
        )

        allocations = cls._allocate_budget(
            [segment[2] for segment in segments], task.target_pairs
        )

        for (left_loci, right_loci, segment_size), target in zip(segments, allocations):
            if target == 0:
                continue

            if target == segment_size:
                flat_indices: Iterable[np.ndarray] = (
                    np.arange(start, min(start + chunk_size, segment_size))
                    for start in range(0, segment_size, chunk_size)
                )

            else:
                selected = np.sort(
                    rng.choice(segment_size, size=int(target), replace=False)
                )
                flat_indices = (
                    selected[start : start + chunk_size]
                    for start in range(0, selected.size, chunk_size)
                )

            right_size = right_loci.size

            for flat in flat_indices:
                locus_i = left_loci[flat // right_size]
                locus_j = right_loci[flat % right_size]
                # Bootstrap-block assignment is randomized, so task orientation
                # does not imply genomic/index order. Canonicalizing here makes
                # each retained unordered pair unambiguous in reports and tests.
                yield np.minimum(locus_i, locus_j), np.maximum(locus_i, locus_j)

    @staticmethod
    def _two_locus_counts(
        genotype_matrix: np.ndarray,
        locus_i: np.ndarray,
        locus_j: np.ndarray,
    ) -> np.ndarray:
        """Count the nine unphased two-locus genotypes for each locus pair."""

        left = genotype_matrix[:, locus_i].T
        right = genotype_matrix[:, locus_j].T
        complete = (left >= 0) & (right >= 0)
        codes = np.where(complete, left * 3 + right, -1)

        return np.column_stack(
            [np.count_nonzero(codes == state, axis=1) for state in range(9)]
        ).astype(np.int64, copy=False)

    def _compute_block_pair(
        self,
        *,
        population: str,
        population_number: int,
        genotype_matrix: np.ndarray,
        task: _BlockPairTask,
        locus_groups: np.ndarray | None,
        pair_chunk_size: int,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Evaluate and aggregate one population/bootstrap-block pair."""

        rng = np.random.default_rng(
            self._task_seed(task.seed, population_number, task.block_a, task.block_b)
        )

        sums = {"D": 0.0, "D2": 0.0, "Dz": 0.0, "Pi2": 0.0}
        pair_count = 0
        complete_sum = 0
        complete_min: int | None = None
        retained = np.empty((0, len(self._PAIRWISE_COLUMNS)), dtype=np.float64)
        priorities = np.empty(0, dtype=np.float64)

        for locus_i, locus_j in self._iter_pair_chunks(
            task, locus_groups, pair_chunk_size, rng
        ):
            counts = self._two_locus_counts(genotype_matrix, locus_i, locus_j)
            estimates = evaluate_unbiased_ld(counts)
            n_complete = counts.sum(axis=1)
            valid = (
                (n_complete >= 4)
                & np.isfinite(estimates["D"])
                & np.isfinite(estimates["D2"])
                & np.isfinite(estimates["Dz"])
                & np.isfinite(estimates["pi2"])
            )

            if not np.any(valid):
                continue

            d = estimates["D"][valid]
            d2 = estimates["D2"][valid]
            dz = estimates["Dz"][valid]
            pi2 = estimates["pi2"][valid]
            complete = n_complete[valid]

            with np.errstate(divide="ignore", invalid="ignore"):
                r2_star = np.divide(
                    d2,
                    pi2,
                    out=np.full_like(d2, np.nan),
                    where=pi2 != 0,
                )

            sums["D"] += float(d.sum())
            sums["D2"] += float(d2.sum())
            sums["Dz"] += float(dz.sum())
            sums["Pi2"] += float(pi2.sum())
            pair_count += int(valid.sum())
            complete_sum += int(complete.sum())
            current_min = int(complete.min())

            complete_min = (
                current_min if complete_min is None else min(complete_min, current_min)
            )

            if task.pairwise_sample_size:
                rows = np.column_stack(
                    [
                        locus_i[valid],
                        locus_j[valid],
                        d,
                        d2,
                        dz,
                        pi2,
                        r2_star,
                        complete,
                    ]
                )

                row_priorities = rng.random(rows.shape[0])
                retained = np.vstack([retained, rows])
                priorities = np.concatenate([priorities, row_priorities])

                if retained.shape[0] > task.pairwise_sample_size:
                    keep = np.argpartition(priorities, task.pairwise_sample_size - 1)[
                        : task.pairwise_sample_size
                    ]

                    retained = retained[keep]
                    priorities = priorities[keep]

        block_row: dict[str, Any] = {
            "Population": population,
            "block_a": task.block_a,
            "block_b": task.block_b,
            "candidate_pairs": task.candidate_pairs,
            "requested_pairs": task.target_pairs,
            "pair_count": pair_count,
            "D_sum": sums["D"],
            "D2_sum": sums["D2"],
            "Dz_sum": sums["Dz"],
            "Pi2_sum": sums["Pi2"],
            "mean_complete_samples": (
                complete_sum / pair_count if pair_count else np.nan
            ),
            "min_complete_samples": (
                complete_min if complete_min is not None else np.nan
            ),
        }

        return block_row, retained

    def _compute_population(
        self,
        *,
        population: str,
        population_number: int,
        genotype_matrix: np.ndarray,
        tasks: Sequence[_BlockPairTask],
        locus_groups: np.ndarray | None,
        pair_chunk_size: int,
        n_jobs: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute every block-pair task serially or with NumPy worker threads."""

        def compute(task: _BlockPairTask) -> tuple[dict[str, Any], np.ndarray]:
            return self._compute_block_pair(
                population=population,
                population_number=population_number,
                genotype_matrix=genotype_matrix,
                task=task,
                locus_groups=locus_groups,
                pair_chunk_size=pair_chunk_size,
            )

        if n_jobs == 1:
            results = [compute(task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(tasks))) as executor:
                results = list(executor.map(compute, tasks))

        block_df = pd.DataFrame([result[0] for result in results])
        pairwise_arrays = [result[1] for result in results if result[1].size]

        if not pairwise_arrays:
            return block_df, pd.DataFrame(
                columns=[
                    "Population",
                    *self._PAIRWISE_COLUMNS,
                    "locus_i_label",
                    "locus_j_label",
                ]
            )

        pairwise_values = np.vstack(pairwise_arrays)
        pairwise_df = pd.DataFrame(pairwise_values, columns=self._PAIRWISE_COLUMNS)
        pairwise_df.insert(0, "Population", population)
        pairwise_df["locus_i"] = pairwise_df["locus_i"].astype(np.int64)
        pairwise_df["locus_j"] = pairwise_df["locus_j"].astype(np.int64)
        pairwise_df["n_complete"] = pairwise_df["n_complete"].astype(np.int64)

        pairwise_df["locus_i_label"] = self.locus_labels[
            pairwise_df["locus_i"].to_numpy()
        ]

        pairwise_df["locus_j_label"] = self.locus_labels[
            pairwise_df["locus_j"].to_numpy()
        ]

        return block_df, pairwise_df

    @staticmethod
    def _effective_population_size(r2d: float, mating_system: MatingSystem) -> float:
        """Convert unlinked ``r2D`` to recent effective population size."""

        if not np.isfinite(r2d) or r2d <= 0:
            return np.nan

        numerator = 1.0 if mating_system == "random" else 2.0
        return numerator / (3.0 * r2d)

    @classmethod
    def _aggregate_rows(
        cls,
        rows: pd.DataFrame,
        mating_system: MatingSystem,
        weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Aggregate optionally weighted block-pair moment sums."""

        if weights is None:
            row_weights = np.ones(len(rows), dtype=np.float64)
        else:
            row_weights = np.asarray(weights, dtype=np.float64)

            if row_weights.ndim != 1 or row_weights.size != len(rows):
                msg = "weights must be a one-dimensional array with one value per block-pair row."
                raise ValueError(msg)

            if np.any(~np.isfinite(row_weights)) or np.any(row_weights < 0.0):
                msg = "weights must be finite and non-negative."
                raise ValueError(msg)

        pair_count = float(
            np.dot(rows["pair_count"].to_numpy(dtype=np.float64), row_weights)
        )

        if pair_count <= 0:
            return {key: np.nan for key in ("D", "D2", "Dz", "Pi2", "r2D", "rDz", "Ne")}

        d_sum = float(np.dot(rows["D_sum"].to_numpy(dtype=np.float64), row_weights))
        d2_sum = float(np.dot(rows["D2_sum"].to_numpy(dtype=np.float64), row_weights))
        dz_sum = float(np.dot(rows["Dz_sum"].to_numpy(dtype=np.float64), row_weights))
        pi2_sum = float(np.dot(rows["Pi2_sum"].to_numpy(dtype=np.float64), row_weights))

        r2d = d2_sum / pi2_sum if pi2_sum != 0 else np.nan
        rdz = dz_sum / pi2_sum if pi2_sum != 0 else np.nan

        return {
            "D": d_sum / pair_count,
            "D2": d2_sum / pair_count,
            "Dz": dz_sum / pair_count,
            "Pi2": pi2_sum / pair_count,
            "r2D": r2d,
            "rDz": rdz,
            "Ne": cls._effective_population_size(r2d, mating_system),
        }

    def _bootstrap_population(
        self,
        *,
        population: str,
        block_df: pd.DataFrame,
        n_bootstraps: int,
        mating_system: MatingSystem,
        seed: int,
        method: BootstrapMethod = "block_pair",
    ) -> pd.DataFrame:
        """Resample random block pairs or explicit biological groups."""

        if method not in {"block_pair", "node_cluster"}:
            msg = f"Invalid bootstrap method '{method}'. Must be 'block_pair' or 'node_cluster'."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if n_bootstraps == 0:
            return pd.DataFrame(
                columns=[
                    "Population",
                    "Replicate",
                    "D",
                    "D2",
                    "Dz",
                    "Pi2",
                    "r2D",
                    "rDz",
                    "Ne",
                ]
            )

        usable = block_df.loc[block_df["pair_count"] > 0].reset_index(drop=True)

        if usable.empty:
            msg = "block_df contains no block pairs with usable loci."
            self.logger.error(msg) if self.logger is not None else None
            raise ValueError(msg)

        if method == "node_cluster":
            required = {"block_a", "block_b"}
            missing = sorted(required.difference(usable.columns))

            if missing:
                msg = f"node_cluster bootstrap requires block identifiers: {missing}."
                self.logger.error(msg) if self.logger is not None else None
                raise ValueError(msg)

            block_ids = np.unique(
                np.concatenate(
                    [
                        usable["block_a"].to_numpy(dtype=np.int64),
                        usable["block_b"].to_numpy(dtype=np.int64),
                    ]
                )
            )

            block_lookup = {block_id: index for index, block_id in enumerate(block_ids)}

            left_indices = np.fromiter(
                (block_lookup[value] for value in usable["block_a"]),
                dtype=np.int64,
                count=len(usable),
            )

            right_indices = np.fromiter(
                (block_lookup[value] for value in usable["block_b"]),
                dtype=np.int64,
                count=len(usable),
            )

        else:
            block_ids = np.arange(usable.shape[0], dtype=np.int64)

            left_indices = right_indices = np.arange(usable.shape[0], dtype=np.int64)

        rng = np.random.default_rng(seed)
        rows = []

        for replicate in range(n_bootstraps):
            if method == "block_pair":
                sampled = usable.iloc[
                    rng.integers(0, usable.shape[0], size=usable.shape[0])
                ]

                aggregate = self._aggregate_rows(sampled, mating_system)

            else:
                sampled_blocks = rng.integers(0, block_ids.size, size=block_ids.size)

                multiplicities = np.bincount(sampled_blocks, minlength=block_ids.size)

                pair_weights = (
                    multiplicities[left_indices] * multiplicities[right_indices]
                )

                aggregate = self._aggregate_rows(
                    usable, mating_system, weights=pair_weights
                )

            rows.append({"Population": population, "Replicate": replicate, **aggregate})

        return pd.DataFrame(rows)

    def _summarize_population(
        self,
        *,
        population: str,
        sample_count: int,
        locus_count: int,
        block_df: pd.DataFrame,
        bootstrap_df: pd.DataFrame,
        mating_system: MatingSystem,
        alpha: float,
    ) -> dict[str, Any]:
        """Create one population summary row with percentile intervals."""

        aggregate = self._aggregate_rows(block_df, mating_system)

        row: dict[str, Any] = {
            "Population": population,
            "Samples": sample_count,
            "Loci": locus_count,
            "Pairs": int(block_df["pair_count"].sum()),
            **aggregate,
        }

        for statistic in ("r2D", "rDz"):
            if bootstrap_df.empty:
                lower = upper = np.nan

            else:
                finite = (
                    bootstrap_df[statistic].replace([np.inf, -np.inf], np.nan).dropna()
                )
                if finite.empty:
                    lower = upper = np.nan
                else:
                    lower, upper = finite.quantile([alpha / 2, 1 - alpha / 2])

            row[f"{statistic}_CI_Lower"] = lower
            row[f"{statistic}_CI_Upper"] = upper

        r2d_lower = float(row["r2D_CI_Lower"])
        r2d_upper = float(row["r2D_CI_Upper"])

        if not np.isfinite(r2d_upper) or r2d_upper <= 0.0:
            ne_lower = ne_upper = np.nan

        else:
            numerator = 1.0 if mating_system == "random" else 2.0
            ne_lower = numerator / (3.0 * r2d_upper)

            ne_upper = (
                numerator / (3.0 * r2d_lower)
                if np.isfinite(r2d_lower) and r2d_lower > 0.0
                else np.inf
            )

        row["Ne_CI_Lower"] = ne_lower
        row["Ne_CI_Upper"] = ne_upper
        return row

    @staticmethod
    def _concat_or_empty(
        frames: Sequence[pd.DataFrame], columns: Sequence[str]
    ) -> pd.DataFrame:
        """Concatenate DataFrames or return an empty table with stable columns."""

        if frames:
            return pd.concat(frames, ignore_index=True)

        return pd.DataFrame(columns=list(columns))

    def _write_reports(
        self,
        result: LinkageDisequilibriumResult,
        *,
        save_pairwise: bool,
    ) -> dict[str, Path]:
        """Write machine-readable LD reports under SNPio's analysis hierarchy."""

        self.report_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "ld_summary": self.report_dir / "linkage_disequilibrium_summary.csv",
            "ld_bootstrap": self.report_dir / "linkage_disequilibrium_bootstrap.csv.gz",
            "ld_blocks": self.report_dir
            / "linkage_disequilibrium_block_summaries.csv.gz",
            "ld_metadata": self.report_dir / "linkage_disequilibrium_metadata.json",
        }
        result.summary.to_csv(files["ld_summary"], index=False)

        result.bootstrap.to_csv(files["ld_bootstrap"], index=False, compression="gzip")

        result.block_summaries.to_csv(
            files["ld_blocks"], index=False, compression="gzip"
        )

        with files["ld_metadata"].open("w", encoding="utf-8") as output:
            json.dump(
                result.metadata, output, indent=2, sort_keys=True, allow_nan=False
            )

            output.write("\n")

        if save_pairwise and not result.pairwise_sample.empty:
            pairwise_path = (
                self.report_dir / "linkage_disequilibrium_pairwise_sample.csv.gz"
            )

            result.pairwise_sample.to_csv(
                pairwise_path, index=False, compression="gzip"
            )

            files["ld_pairwise_sample"] = pairwise_path
        return files


__all__ = ["LinkageDisequilibrium", "LinkageDisequilibriumResult"]
