#!/bin/zsh

set -e

# -------------
# Print usage
# -------------
usage() {
    echo "Usage: $0 -i input.vcf.gz -n NUM_LOCI -o output.vcf.gz"
    echo
    echo "Arguments:"
    echo "  -i  Input VCF file (bgzipped and indexed)"
    echo "  -n  Number of loci to randomly sample"
    echo "  -o  Output VCF file (bgzipped)"
    exit 1
}

# -------------------
# Parse command-line
# -------------------
while getopts ":i:n:o:" opt; do
  case $opt in
    i) VCF_IN="$OPTARG" ;;
    n) N="$OPTARG" ;;
    o) VCF_OUT="$OPTARG" ;;
    \?) echo "Error: Invalid option -$OPTARG" >&2; usage ;;
    :) echo "Error: Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# -------------------
# Validate arguments
# -------------------
if [[ -z "$VCF_IN" || -z "$N" || -z "$VCF_OUT" ]]; then
    usage
fi

if [[ ! -f "$VCF_IN" ]]; then
    echo "Error: Input VCF file '$VCF_IN' not found." >&2
    exit 1
fi

if ! [[ "$N" =~ '^[0-9]+$' ]]; then
    echo "Error: Number of loci (-n) must be a positive integer." >&2
    exit 1
fi

# -------------------
# Main subsetting
# -------------------
TMP_DIR="./tmp_bcftools_subset"
mkdir -p "$TMP_DIR"

echo "Extracting variant positions from: $VCF_IN"
bcftools query -f '%CHROM\t%POS\n' "$VCF_IN" > "$TMP_DIR/all_sites.txt"

echo "Sampling $N random loci..."
gshuf "$TMP_DIR/all_sites.txt" | head -n "$N" > "$TMP_DIR/sampled_sites.txt"

echo "Preparing regions list for bcftools..."
cp "$TMP_DIR/sampled_sites.txt" "$TMP_DIR/regions.txt"


echo "Subsetting original VCF..."
bcftools view -R "$TMP_DIR/regions.txt" -Oz -o "$VCF_OUT" "$VCF_IN"

echo "Indexing output VCF..."
tabix -p vcf "$VCF_OUT"

rm -r "$TMP_DIR"
echo "Subset VCF written to: $VCF_OUT"

