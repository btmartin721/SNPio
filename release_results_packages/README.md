# Consolidated Results Packages

This directory provides a single parent location for both deliverables:

- `benchmarking_results_package/`
- `validation_results_package/`

## Why only docs are tracked in git

The full package contents include large validation/benchmark input and output files that exceed standard GitHub file-size limits.

To keep the repository PR-friendly while preserving reproducibility, this directory tracks package structure documentation (`README.md` and `STRUCTURE.md`) and excludes large artifacts from git history.

## Full package for reviewers/repository upload

Use the local assembled `release_results_packages/` directory (including full files) when creating archives for reviewers or data repositories (e.g., Zenodo/Figshare).