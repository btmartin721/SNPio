# Validation Results Package Structure

Last verified: 2026-05-19

## `inputs/`

Validation input data.

- `d_statistics/partitioned_d/` - Partitioned D simulated VCFs, indexes, and popmaps.
- `d_statistics/patterson_d/` - Patterson D simulated VCFs, indexes, popmaps, and demes plot.
- `fst_nei/vcf_files/` - VCF input used by the Fst/Nei validation run.
- `fst_nei/popmaps/` - Population map used by the Fst/Nei validation run.

## `scripts/`

Runnable or source scripts used for the analyses.

- `d_statistics/` - D-statistics simulation and validation scripts.
- `d_statistics/plot_d_statistics_results.py` - Rebuild D-statistics plots from saved JSON outputs without rerunning SNPio analyses.
- `d_statistics/plot_d_statistics_results.zsh` - Convenience wrapper that sets Matplotlib cache paths and calls the Python plotter.
- `fst_nei/` - SNPio runner, table-combination helper, shell wrappers, and R validation scripts.

## `outputs/d_statistics/`

D-statistics validation outputs.

`d_statistics` is the canonical Dstat directory for this package.

- `raw/snpio_outputs/` - Per-admixture-fraction SNPio output directories.
- `processed/` - Compact CSV tables rebuilt from saved D-statistics JSON outputs.
- `figures/generated/png/` - Generated PNG figures.
- `figures/generated/pdf/` - Generated PDF figures.
- `figures/generated/ai/` - Generated Illustrator files.
- `figures/edited/png/` - Manually edited reviewer PNG figures.
- `figures/edited/pdf/` - Manually edited reviewer PDF figures.
- `figures/edited/ai/` - Manually edited Illustrator files.

Current counts:
- processed tables: 3 CSV files
- generated figures: 10 files across PNG/PDF/AI
- edited reviewer figures: 3 files across PNG/PDF/AI

## `outputs/fst/`

Fst validation outputs.

- `snpio/tables/` - SNPio Fst CSV summary tables.
- `snpio/json_reports/` - SNPio Fst pairwise permutation JSON reports.
- `snpio/plots/` - SNPio Fst heatmap and permutation plots.
- `reference/hierfstat/` - Reference Fst outputs from hierfstat.
- `summaries/` - Reviewer-facing Fst comparison and Mantel summary tables.

Current counts:
- `snpio/tables/`: 3 files
- `snpio/json_reports/`: 10 files
- `snpio/plots/`: 11 files

## `outputs/nei/`

Nei's genetic distance validation outputs.

- `snpio/tables/` - SNPio Nei distance CSV summary tables.
- `snpio/json_reports/` - SNPio Nei pairwise permutation JSON reports.
- `snpio/plots/` - SNPio Nei heatmap and permutation plots.
- `reference/hierfstat/` - Reference Nei outputs from hierfstat.
- `summaries/` - Reviewer-facing Nei comparison and Mantel summary tables.

Current counts:
- `snpio/tables/`: 3 files
- `snpio/json_reports/`: 10 files
- `snpio/plots/`: 11 files

## `outputs/fst_nei/`

Shared Fst/Nei files.

- `combined_summaries/` - Combined workbook summaries and shared allele-summary JSON files.
- `logs/` - SNPio and nRemover logs from the curated validation run.
- `supporting_plots/` - Shared genotype-data and summary-statistics plots.
- `results/` - Re-run outputs from `scripts/fst_nei/run_fst.zsh`; kept separate from curated manuscript-facing Fst/Nei results.
