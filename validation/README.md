# SNPio Validation Evidence

This directory is the tracked, reviewer- and user-facing index of evidence for new SNPio features. It is intended to answer three questions:

1. What behavior was validated?
2. Which reproducible checks support that claim?
3. Under which assumptions is the feature suitable for use?

The evidence recorded here complements, but does not replace, the test suite. A feature is described as **Validated** only after its manifest identifies the tested revision, validation configuration, acceptance criteria, complete run status, software environment, compact results, plots, and file checksums.

## Directory roles

- `validation/` contains compact, portable evidence intended for GitHub.
- `snpio/validation/` contains executable Python validation infrastructure.
- `scripts/validation/` contains validation maintenance utilities.
- `validation_results/` contains local raw runs and is intentionally ignored by Git. Active or incomplete runs must remain there until they are reviewed.
- `release_results_packages/validation_results_package/` contains the existing curated validation package for previously implemented analyses.

Do not copy an active output directory into this tracked tree. Promotion is a separate review step performed only after the run has completed.

## Current feature index

| Feature | Validation record | Evidence snapshot | Status |
|---|---|---|---|
| Unbiased LD and LD-based recent effective population size | [`MANIFEST.md`](MANIFEST.md#linkage-disequilibrium-validation-record) | [`20260717T003026Z_ea14ecf`](linkage_disequilibrium/results/20260717T003026Z_ea14ecf/) | **Validated** |

The authoritative status is maintained in [`MANIFEST.md`](MANIFEST.md). Do not infer validation status merely from the presence of files or plots.

## Evidence required for a `Validated` label

Every promoted feature snapshot must include:

- the Git commit and branch tested;
- the exact command and configuration;
- package and platform versions;
- a machine-readable per-step status showing no failed required step;
- summary tables and representative plots;
- documented acceptance criteria and applicability limits;
- a concise human-readable result interpretation;
- SHA-256 checksums for every promoted artifact; and
- repository-relative paths without user-specific absolute paths.

Large caches, transient work files, duplicate plots, raw dependency downloads, and unrelated analysis outputs must not be committed.

## Adding another feature

Every new user-facing analysis or substantial implementation should add a feature record here before release. The standard workflow is:

1. Add a feature registry row and validation-record section to [`MANIFEST.md`](MANIFEST.md), documenting purpose, assumptions, validation layers, acceptance criteria, and limitations.
2. Run the validation into a new immutable directory under `validation_results/`.
3. Review the complete status before using `rsync` to copy a compact allowlist of evidence to `validation/<feature_slug>/results/<run_id>/`. Never promote an entire raw run tree by default.
4. Add portable provenance and SHA-256 checksums to the snapshot, and record the interpreted results in `MANIFEST.md`.
5. Scan the promoted text artifacts for absolute paths and verify the checksum manifest.
6. Update [`MANIFEST.md`](MANIFEST.md) and the tree documented in [`STRUCTURE.md`](STRUCTURE.md).

Published snapshots are immutable. Corrections or reruns receive a new run ID rather than overwriting earlier evidence.

To keep this directory concise, these three Markdown files are the only documentation files used under `validation/`: this overview, the authoritative manifest, and the directory-structure standard.
