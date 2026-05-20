#!/bin/zsh

# # Run from repository root.
python scripts/run_fst.py \
    --input snpio/example_data/vcf_files/phylogen_subset14K.vcf.gz \
    --popmap snpio/example_data/popmaps/phylogen_nomx.popmap \
    --prefix results/results \
    --n-jobs 8 \
    --n-reps 1000 \
    --verbose \
    --plot-format pdf

python scripts/combine_tables.py \
    --fst-json results/results_fst_table_inputs.json \
    --output results/wc_fst_manuscript_table.csv \
    --table-type obs-ci-pvalues \
    --population-order EA,GU,TT,ON,OG \
    --decimals 3 \
    --pvalue-decimals 3 \
    --diagonal blank \
    --stars

# python scripts/combine_tables.py \
#     --nei-json results/results_nei_table_inputs.json \
#     --output results/nei_manuscript_table.csv \
#     --table-type obs-ci-pvalues \
#     --population-order EA,GU,TT,ON,OG \
#     --decimals 3 \
#     --pvalue-decimals 3 \
#     --diagonal blank \
#     --stars