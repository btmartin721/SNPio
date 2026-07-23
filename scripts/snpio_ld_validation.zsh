#!/usr/bin/env zsh

cd /Users/btm002/Documents/Work/SHU/Research/Ongoing/Repositories/SNPio

PYTHON=/tmp/snpio-ld-diagnostic-venv/bin/python
OUT="validation_results/linkage_disequilibrium/forward_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT/.cache/matplotlib"

MPLCONFIGDIR="$OUT/.cache/matplotlib" \
"$PYTHON" scripts/validate_ld.py \
    --output "$OUT" \
    --plot-formats png pdf \
    --plot-dpi 300 \
    simulate \
    --population-sizes 10 25 100 \
    --sample-sizes 4 6 8 20 \
    --replicates 250 \
    --chromosomes 8 \
    --loci-per-chromosome 100 \
    --burnin-multiplier 10 \
    --allow-residual-selfing \
    --n-bootstraps 200 \
    --minimum-model-population-size 100 \
    --minimum-coverage-sample-size 8 \
    --n-jobs 8 \
    --seed 20260715

print -r -- "Results: $OUT"
