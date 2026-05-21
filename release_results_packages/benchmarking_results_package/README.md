# SNPio Benchmarking Results Package

Benchmarking dataset for SNPio file I/O and filtering performance comparisons.

The package is intentionally organized around three top-level folders:

- `inputs/` - VCF inputs and population map.
- `scripts/` - All runnable analysis, benchmark, and plotting scripts.
- `outputs/` - All benchmark outputs, processed tables, and figures.

No benchmark outputs are intentionally written outside `outputs/`.

## Quick Start

Reviewer-facing figures:

- `outputs/fileIO/figures/png/fileIO_execution_time_summary.png`
- `outputs/fileIO/figures/png/fileIO_memory_usage_summary.png`
- `outputs/filtering/figures/png/benchmark_execution_time_comparison.png`
- `outputs/filtering/figures/png/benchmark_memory_usage_comparison.png`

Polished figures:

- `outputs/polished_figures/fileIO/time/`
- `outputs/polished_figures/filtering/time/`
- `outputs/polished_figures/filtering/memory/`

Processed long-form tables:

- `outputs/fileIO/processed/fileIO_execution_time_long.csv`
- `outputs/fileIO/processed/fileIO_memory_usage_long.csv`
- `outputs/filtering/processed/filtering_execution_time_long.csv`
- `outputs/filtering/processed/filtering_memory_usage_long.csv`

Reviewer fold-change and raw mean tables:

- `outputs/fold_changes/snpio_fold_changes_summary.csv`
- `outputs/fold_changes/snpio_fold_changes_reviewer_table.csv`
- `outputs/fold_changes/snpio_raw_mean_estimates_summary.csv`
- `outputs/fold_changes/snpio_raw_mean_estimates_reviewer_table.csv`

## Re-run Processing

From the repository root:

```zsh
release_results_packages/benchmarking_results_package/scripts/process_reviewer_results_package.zsh
```

This regenerates the file I/O tables and figures, filtering tables and figures, and fold-change/raw-mean tables in `release_results_packages/benchmarking_results_package/outputs/`.

To rerun each step separately:

```zsh
release_results_packages/benchmarking_results_package/scripts/process_fileio_results.zsh
release_results_packages/benchmarking_results_package/scripts/process_filtering_results.zsh
release_results_packages/benchmarking_results_package/scripts/calculate_fold_changes.zsh
```

These scripts read from `outputs/` and write regenerated CSV/PNG/PDF files back to `outputs/`.

## Re-run 1M File I/O Benchmarks

Runtime:

```zsh
benchmarking_results_package/scripts/run_fileio_runtime_1000000.zsh
```

Memory:

```zsh
benchmarking_results_package/scripts/run_fileio_memory_1000000.zsh
```

Both scripts use `inputs/` and write only to `outputs/fileIO/`.

## Included Benchmarks

File I/O:

- SNPio `VCFReader`
- vcfR
- Runtime measured with `hyperfine`
- Memory measured with Python/R memory scripts
- Loci sizes: 1,000; 10,000; 50,000; 100,000; 500,000; 1,000,000

Filtering:

- SNPio nRemover
- SNPfilter
- Runtime and memory comparisons for shared filtering operations
- Loci sizes: 1,000; 10,000; 50,000; 100,000; 500,000; 1,000,000

## Current Layout

```text
benchmarking_results_package/
  inputs/
  outputs/
    fileIO/
      time/
      memory/
      processed/
      figures/
    filtering/
      time/
      memory/
      processed/
      figures/
    fold_changes/
    polished_figures/
  scripts/
  README.md
  STRUCTURE.md
```

See `STRUCTURE.md` for a detailed map.
