# Validation Directory Structure

The tracked validation tree is deliberately small and feature-oriented:

```text
validation/
├── README.md
├── MANIFEST.md
├── STRUCTURE.md
└── linkage_disequilibrium/
    └── results/
        └── 20260717T003026Z_ea14ecf/
            ├── SHA256SUMS.txt
            ├── provenance/
            ├── tables/
            ├── plots/
            └── logs/
```

When a complete run is promoted, its immutable snapshot should use this layout:

```text
validation/<feature_slug>/results/<run_id>/
├── SHA256SUMS.txt
├── provenance/
│   ├── run_metadata.tsv
│   ├── step_summary.tsv
│   ├── validation_status.tsv
│   └── configuration files
├── tables/
│   ├── exact and reference comparisons
│   ├── simulation summaries
│   └── convergence summaries
├── plots/
│   ├── PNG figures
│   └── PDF figures
└── logs/
    └── selected validation and test logs
```

## Naming rules

- Feature directories use stable lowercase snake-case identifiers.
- Run IDs use UTC time plus the tested short commit, for example `20260716T193101Z_a1b2c3d`.
- A run directory is immutable after publication.
- A rerun or correction receives a new run ID.

## Content rules

### `MANIFEST.md`

Serves as both the feature registry and the human-readable validation record. Each feature section summarizes the tested claim, commands, outcome of every validation layer, acceptance criteria, limitations, evidence snapshot, and the conclusion a user may safely draw.

### `SHA256SUMS.txt`

Contains repository-relative SHA-256 checksums generated only after the snapshot is finalized. It must include every file in the run directory except the checksum file itself.

### `provenance/validation_status.tsv`

Maps each required validation layer to its status, source run, and promoted evidence. It is the machine-readable snapshot-level status. A snapshot built from more than one compatible invocation must also preserve each source invocation's metadata and step summary.

### `provenance/`

Contains the tested revision, dirty-state declaration, software versions, platform, random seeds, and exact configuration. Absolute paths must be removed or replaced with repository-relative descriptions before publication.

### `tables/`

Contains compact machine-readable evidence. Prefer CSV, TSV, and JSON. Raw intermediate matrices belong in the ignored local `validation_results/` tree unless they are necessary to reproduce a published summary and remain small enough for Git.

### `plots/`

Contains the final plots used to interpret the snapshot. Preserve PNG for
quick viewing and PDF for vector-quality review when both are available.

### `logs/`

Contains only logs necessary to verify commands, test outcomes, or warnings.
Dependency caches, routine debug logs, and repeated console output are not
validation evidence.

## Separation from local runs

`validation_results/` remains ignored and may contain active, failed,
diagnostic, or extremely large runs. Its presence does not imply validation.
Only a reviewed snapshot listed as **Validated** in `MANIFEST.md` carries a
user-facing validation claim.

The existing `release_results_packages/validation_results_package/` remains a
separate curated package for legacy analyses. New feature snapshots should use
this root `validation/` framework unless a release-specific package is
explicitly required.

## Standard promotion workflow

1. Finish and review a raw run beneath `validation_results/<feature_slug>/`.
2. Create a new `validation/<feature_slug>/results/<run_id>/` tree using the
   standard `provenance/`, `tables/`, `plots/`, and `logs/` subdirectories.
3. Use `rsync -a` with an explicit artifact allowlist. Include summary and
   status tables, final plots, configuration, source metadata, and focused
   logs; exclude caches, copied inputs, and transient analysis trees.
4. Add or update the feature record in `MANIFEST.md`, sanitize machine-specific
   paths, and generate `SHA256SUMS.txt` only after curation is complete.
5. Verify all checksums and machine-readable statuses, then register the
   immutable snapshot in `MANIFEST.md`.

This sequence is the default for every new SNPio analysis or substantial
feature. Feature-specific commands, outcomes, and limitations belong in the
corresponding section of `MANIFEST.md`. Nested Markdown files are avoided so
the validation tree remains compact and easy to navigate.
