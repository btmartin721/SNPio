# Validation Evidence Manifest

Last updated: 2026-07-17

This file is the authoritative registry of tracked SNPio validation evidence. Status values have the following meanings:

- **Validated** — all required layers passed and a complete, checksummed evidence snapshot is present.
- **Validation in progress** — validation is running or the completed output has not yet been reviewed and promoted.
- **Diagnostic only** — useful supporting evidence that does not establish the complete feature claim.
- **Superseded** — retained for provenance but replaced by a newer snapshot.

## Feature registry

| Feature ID | User-facing feature | Implementation | Validation entry point | Evidence snapshot | Status |
|---|---|---|---|---|---|
| `linkage_disequilibrium` | Unbiased unphased LD statistics and LD-based recent effective population size from unlinked SNPs | `snpio/popgenstats/linkage_disequilibrium.py`; `snpio/popgenstats/ld_polynomials.py` | `scripts/run_robust_ld_validation.zsh` | [`20260717T003026Z_ea14ecf`](linkage_disequilibrium/results/20260717T003026Z_ea14ecf/) | **Validated** |

The LD snapshot combines compatible source invocations from the same tested commit and configuration: the primary invocation passed steps 1–5 and the earlier invocation supplies the already-completed step-6 convergence result. Both source summaries, the dirty-tree declaration, and an authoritative combined `validation_status.tsv` are preserved. The evidence boundary and interpretation are recorded below.

## Linkage disequilibrium validation record

### Validated claim and scope

SNPio calculates finite-sample-unbiased, unphased LD moment estimators for biallelic diploid SNPs following the Ragsdale-Gravel formulation. It reports `D2`, `Dz`, and `Pi2`, their ratio statistics `r2D` and `rDz`, and recent random-mating effective population size `Ne = 1 / (3 * r2D)` for unlinked loci.

The feature is appropriate when genotypes are diploid and biallelic, at least
four complete diploid samples are available for a locus pair, analyzed pairs
are genuinely unlinked, and the selected mating-system conversion is
biologically appropriate. `assume_unlinked=True` is a user assertion rather
than an automatic test for physical linkage.

### Evidence outcomes

| Layer | Dataset or cases | Result | What it establishes |
|---|---|---|---|
| Focused tests | SNPio LD unit and integration fixtures | 38/38 passed | Polynomial evaluation, pair construction, aggregation, bootstrapping, output, plotting, and validation controls behave as tested. |
| Exact enumeration | All two-locus genotype-count states for `n=4` and `n=6` across four scenarios | 32/32 rows passed at `1e-12` tolerance | The finite-sample estimator has the intended exact expectation in exhaustively enumerable cases. |
| Frozen external reference | 1,000 archived `moments-popgen 1.6.0` cases for each of `D`, `D2`, `Dz`, and `Pi2` | 4,000/4,000 comparisons passed | SNPio's independent implementation agrees numerically with the external reference without a runtime `moments-popgen` dependency. |
| Published island fox | Six island populations from the published GenePop benchmark | 6/6 passed; all confidence intervals overlap; maximum relative point-estimate error was 1.09% | The SNPio LD-to-`Ne` workflow reproduces a published empirical benchmark. |
| Neutral forward simulation | 250 replicates per 22 population-size/sample-size cells using eight unlinked chromosomes | All 22 cells passed the finalized acceptance contract | No material model bias was demonstrated for formal `N >= 100` cells; smaller populations remain stress diagnostics. |
| Pair-budget convergence | Example VCF and population map; 250,000, 1,000,000, and 4,000,000 maximum pairs across five seeds | Completed successfully | Quantifies sensitivity of empirical LD and `Ne` estimates to random locus-pair subsampling. Near-zero or small-sample estimates can remain unstable. |

The snapshot is
[`linkage_disequilibrium/results/20260717T003026Z_ea14ecf/`](linkage_disequilibrium/results/20260717T003026Z_ea14ecf/).
Its `tables/`, `plots/`, `provenance/`, and `logs/` directories contain the machine-readable evidence and figures. `SHA256SUMS.txt` covers every promoted artifact.

### Provenance boundary

All six layers passed in recorded development runs based on Git commit `ea14ecfb14cdab75c96cc67e7253454efd4c97e5`. The tested working trees were dirty, so the commit is the shared base revision rather than a byte-exact reconstruction. The primary run `robust_20260717T003026Z` supplies steps 1–5; the compatible run `robust_20260716T193101Z` supplies the already-completed step-6 convergence evidence. The latter's superseded step-5 diagnostic is not used. A clean-tree, single-invocation run should eventually supersede this snapshot for exact release-grade source reconstruction.

### Reproducing the validation

Run the full repository driver:

```shell
scripts/run_robust_ld_validation.zsh \
    --genepop /path/to/GP_NO_grays.txt \
    --jobs 8
```

The driver writes a timestamped raw run beneath the ignored
`validation_results/linkage_disequilibrium/` tree. Detailed options are documented in `snpio/docs/source/ld_validation.rst`. The promoted forward run used 250 replicates, seed `20260715`, and the configuration archived in [`forward_simulation_config.json`](linkage_disequilibrium/results/20260717T003026Z_ea14ecf/provenance/forward_simulation_config.json). The convergence analysis used the example VCF and population map, pair budgets of 250,000, 1,000,000, and 4,000,000, and seeds 101, 203, 307, 409, and 503.

### Interpretation limits

- `Ne = 1 / (3 * r2D)` is validated for unlinked loci under the corresponding random-mating demographic assumptions, not for physically linked markers or arbitrary mating systems.
- The forward-model target is formal for `N >= 100`; smaller populations are finite-population stress diagnostics.
- Chromosome-block bootstrap intervals describe genomic pair/block uncertainty conditional on sampled individuals. Matched-census coverage is diagnostic, not a population-sampling coverage guarantee.
- Small samples or `r2D` near zero can produce substantial uncertainty that is amplified by the reciprocal conversion to `Ne`.

## Required LD snapshot artifacts

The first promoted LD snapshot must contain the following compact evidence, organized according to [`STRUCTURE.md`](STRUCTURE.md):

| Evidence | Source artifact from the completed robust run | Required |
|---|---|---|
| Overall status | `provenance/validation_status.tsv` plus source step summaries | Yes |
| Run provenance | Sanitized source-run metadata under `provenance/` | Yes |
| Human interpretation | Feature validation record in this manifest | Yes |
| Exact-enumeration results | `tables/exact_enumeration/` and `plots/exact_enumeration/` | Yes |
| Frozen-reference comparison | `tables/golden_reference/` and `plots/golden_reference/` | Yes |
| Published island-fox comparison | `tables/published_island_fox/` and `plots/published_island_fox/` | Yes |
| Forward calibration | Configuration under `provenance/`, plus `tables/forward_simulation/` and `plots/forward_simulation/` | Yes |
| Forward replicate table | `tables/forward_simulation/ld_calibration_replicates.csv` | Recommended if its size remains appropriate for Git |
| Pair-budget convergence | `tables/pair_budget_convergence/` and `plots/pair_budget_convergence/` | Yes |
| Focused test evidence | `logs/unit_tests.log` | Yes |
| Full logs | Selected failure-free logs needed to interpret the run | Optional |
| Integrity manifest | `SHA256SUMS.txt` covering every promoted file | Yes |

Do not promote `.cache/`, `.DS_Store`, copied source datasets, temporary output trees, or files containing user-specific absolute paths.

## Promotion checklist

- [x] Every source invocation used as evidence terminated normally.
- [x] Every required row in `provenance/validation_status.tsv` is `PASS`.
- [x] Machine-readable status JSON files report passing required checks.
- [x] Plots and summary tables were visually and numerically reviewed.
- [x] The tested Git commit and dirty-state information are recorded.
- [x] The feature assumptions and limitations match the implementation.
- [x] Paths in JSON, CSV, TSV, Markdown, and logs are portable or sanitized.
- [x] The compact artifact allowlist was copied into a new immutable run ID.
- [x] `SHA256SUMS.txt` was generated after all curation was complete.
- [x] This registry and its validation record link to the promoted snapshot.

A composite snapshot is acceptable only when its component runs use the same tested implementation and compatible configuration, each source status is retained, and the combined status maps every required layer to its source. Otherwise, run all required layers again under one invocation.
