# Benchmarking Results Package Structure

This package has a simplified layout: one inputs folder, one scripts folder, and one outputs folder.

## `inputs/`

Benchmark input data.

- `popmap/nloci_test.popmap.txt` - Population map used by SNPio and vcfR.
- `vcf_files/filtering_benchmarks/time/` - VCF inputs used for runtime runs.
- `vcf_files/filtering_benchmarks/memory/` - VCF inputs used for memory runs.

## `scripts/`

All analysis scripts live in this single folder.

- `run_fileio_runtime_1000000.zsh` - Run SNPio VCFReader and vcfR 1M file I/O runtime benchmarks.
- `run_vcfr_runtime_1000000.zsh` - Run only vcfR 1M file I/O runtime benchmark.
- `run_fileio_memory_1000000.zsh` - Run SNPio VCFReader and vcfR 1M file I/O memory benchmarks.
- `run_snpio_fileio_runtime.py` - SNPio file I/O runtime worker.
- `run_snpio_fileio_memory.py` - SNPio file I/O memory worker.
- `benchmark_vcfr_runtime.R` - vcfR runtime worker.
- `benchmark_vcfr_memory.R` - vcfR memory worker.
- `process_fileio_results.py` and `process_fileio_results.zsh` - Process and plot file I/O results.
- `process_filtering_results.py` and `process_filtering_results.zsh` - Process and plot filtering results.

## `outputs/`

All output files live under this folder.

### `outputs/polished_figures/`

Manually edited figure files are separated from generated outputs so rerunning the processing scripts does not overwrite them.

- `fileIO/time/png/`, `fileIO/time/pdf/`, `fileIO/time/ai/` - Edited file I/O runtime figures.
- `filtering/time/png/`, `filtering/time/pdf/`, `filtering/time/ai/` - Edited filtering runtime figures.
- `filtering/memory/png/`, `filtering/memory/pdf/` - Edited filtering memory figures.

### `outputs/fileIO/`

- `time/snpio_vcfreader/` - SNPio VCFReader hyperfine exports.
- `time/vcfr/` - vcfR hyperfine exports.
- `memory/snpio_vcfreader/` - SNPio VCFReader memory JSON files.
- `memory/vcfr/` - vcfR memory JSON files.
- `processed/` - Long-form CSVs and generated file I/O plots.
- `figures/png/` - Reviewer-facing file I/O PNG figures.
- `figures/pdf/` - Reviewer-facing file I/O PDF figures.

### `outputs/filtering/`

- `time/snpio_nremover/` - SNPio nRemover runtime JSON files.
- `time/snpfilter/` - SNPfilter runtime JSON files.
- `memory/snpio_nremover/` - SNPio nRemover memory JSON files.
- `memory/snpfilter/` - SNPfilter memory JSON files.
- `processed/` - Long-form CSVs and generated filtering plots.
- `figures/png/` - Reviewer-facing filtering PNG figures.
- `figures/pdf/` - Reviewer-facing filtering PDF figures.

## Current Figures

File I/O:

- `outputs/fileIO/figures/png/fileIO_execution_time_summary.png`
- `outputs/fileIO/figures/png/fileIO_memory_usage_summary.png`
- `outputs/fileIO/figures/pdf/fileIO_execution_time_summary.pdf`
- `outputs/fileIO/figures/pdf/fileIO_memory_usage_summary.pdf`

Filtering:

- `outputs/filtering/figures/png/benchmark_execution_time_comparison.png`
- `outputs/filtering/figures/png/benchmark_memory_usage_comparison.png`
- `outputs/filtering/figures/pdf/benchmark_execution_time_comparison.pdf`
- `outputs/filtering/figures/pdf/benchmark_memory_usage_comparison.pdf`

Edited figures:

- `outputs/polished_figures/fileIO/time/png/fileIO_execution_time_summary_COMBINED.png`
- `outputs/polished_figures/fileIO/time/pdf/fileIO_execution_time_summary_COMBINED.pdf`
- `outputs/polished_figures/filtering/time/png/benchmark_execution_time_comparison_FINAL.png`
- `outputs/polished_figures/filtering/time/pdf/benchmark_execution_time_comparison_FINAL.pdf`
- `outputs/polished_figures/filtering/memory/png/benchmark_memory_usage_comparison_FINAL.png`
- `outputs/polished_figures/filtering/memory/pdf/benchmark_memory_usage_comparison_FINAL.pdf`

## Current Processed Tables

- `outputs/fileIO/processed/fileIO_execution_time_long.csv`
- `outputs/fileIO/processed/fileIO_memory_usage_long.csv`
- `outputs/filtering/processed/filtering_execution_time_long.csv`
- `outputs/filtering/processed/filtering_memory_usage_long.csv`
