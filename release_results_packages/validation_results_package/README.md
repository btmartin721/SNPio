# SNPio Validation Results Package

Curated validation files for SNPio D-statistics, Weir and Cockerham Fst, and
Nei's genetic distance analyses.

Last verified: 2026-05-19

The package mirrors the benchmarking package layout:

- `inputs/` - Input VCFs, popmaps, and related validation inputs.
- `scripts/` - Analysis scripts used to generate or process the results.
- `outputs/` - Validation outputs, summaries, plots, logs, and edited figures.

## Quick Reference

D-statistics:

- Edited reviewer figure: `outputs/d_statistics/figures/edited/png/d_statistics_validation_summary_CLEAN.png`
- Generated figures: `outputs/d_statistics/figures/generated/`
- Replot existing results without rerunning analyses: `scripts/d_statistics/plot_d_statistics_results.zsh`
- Compact plotting tables: `outputs/d_statistics/processed/`
- Raw SNPio outputs: `outputs/d_statistics/raw/snpio_outputs/`

Note: D-statistics are stored under `outputs/d_statistics/` (canonical directory name for Dstat results).

Fst:

- SNPio tables: `outputs/fst/snpio/tables/`
- SNPio JSON reports: `outputs/fst/snpio/json_reports/`
- SNPio plots: `outputs/fst/snpio/plots/`
- Reference hierfstat output: `outputs/fst/reference/hierfstat/`
- Summary tables: `outputs/fst/summaries/`

Nei's genetic distance:

- SNPio tables: `outputs/nei/snpio/tables/`
- SNPio JSON reports: `outputs/nei/snpio/json_reports/`
- SNPio plots: `outputs/nei/snpio/plots/`
- Reference hierfstat output: `outputs/nei/reference/hierfstat/`
- Summary tables: `outputs/nei/summaries/`

Shared Fst/Nei outputs:

- Combined summary tables and allele summary JSON: `outputs/fst_nei/combined_summaries/`
- Logs: `outputs/fst_nei/logs/`
- Supporting genotype-data plots: `outputs/fst_nei/supporting_plots/`

## Verified File Counts (2026-05-19)

- Fst: 3 tables, 10 JSON reports, 11 plots
- Nei: 3 tables, 10 JSON reports, 11 plots
- D-statistics: 3 processed CSV tables, 10 generated figures, 3 edited reviewer figures

## Source Locations

These files were curated from:

- `Validations/Dtest/DtestValidations/simulations/`
- `Validations/Dtest/DtestValidations/simulations/reviewer_test_outputs/`
- `Validations/FstNeiValidation/`
- `scripts/run_fst.zsh`
- `scripts/run_fst.py`

## Replot D-Statistics

From the repository root:

```zsh
Validations/validation_results_package/scripts/d_statistics/plot_d_statistics_results.zsh
```

This reads the saved SNPio JSON files from
`outputs/d_statistics/raw/snpio_outputs/`, writes compact CSV tables to
`outputs/d_statistics/processed/`, and regenerates PDF/PNG plots under
`outputs/d_statistics/figures/generated/`.
