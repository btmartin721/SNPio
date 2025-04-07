import numpy as np


def simulate_locus_with_groups(
    n_samples: int,
    missing_rate: float,
    het_rate: float,
    group_enrichment: list = None,
    base_majority: int = 0,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate genotype data for a single locus with grouped patterns of heterozygous and alt genotypes.

    Args:
        n_samples (int): Number of individuals to simulate.
        missing_rate (float): Proportion of missing genotypes (-9).
        het_rate (float): Proportion of heterozygous genotypes (1).
        group_enrichment (list): List of tuples (start_idx, end_idx, enrichment_level),
                                 where enrichment_level increases 1s and 2s in this range.
        base_majority (int): Genotype value for majority (0 by default).
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: 1D array of genotypes (length n_samples).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    genotypes = np.full(n_samples, base_majority, dtype=int)

    # Add missing values
    n_missing = int(n_samples * missing_rate)
    missing_indices = rng.choice(n_samples, size=n_missing, replace=False)
    genotypes[missing_indices] = -9

    # Prepare indices not missing
    available_indices = np.setdiff1d(np.arange(n_samples), missing_indices)

    # Assign heterozygous genotypes
    n_het = int(n_samples * het_rate)
    if group_enrichment and n_het > 0:
        enriched_indices = []
        for start, end, enrich_level in group_enrichment:
            # More 1s and 2s for this group
            group_size = int(n_het * enrich_level)
            group_indices = np.intersect1d(np.arange(start, end), available_indices)
            group_sample = rng.choice(
                group_indices, size=min(len(group_indices), group_size), replace=False
            )
            enriched_indices.extend(group_sample)
        enriched_indices = np.unique(enriched_indices)
        genotypes[enriched_indices[:n_het]] = 1
    else:
        het_indices = rng.choice(available_indices, size=n_het, replace=False)
        genotypes[het_indices] = 1

    # Assign alt homozygotes where possible to some enriched group indices not already 1 or -9
    non_ref_indices = np.where(
        (genotypes == base_majority) & (~np.isin(np.arange(n_samples), missing_indices))
    )[0]
    alt_size = int(len(non_ref_indices) * 0.1)  # 10% of remaining will be 2s
    alt_indices = rng.choice(non_ref_indices, size=alt_size, replace=False)
    genotypes[alt_indices] = 2

    return genotypes


def simulate_genotype_matrix_with_groups(
    n_samples: int = 50,
    loci_specs: list = None,
    group_enrichment: list = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate genotype matrix with group-based enrichment for heterozygous/alt genotypes.

    Args:
        n_samples (int): Number of individuals (rows).
        loci_specs (list): List of dicts with 'missing' and 'het' proportions per locus.
        group_enrichment (list): Group enrichment for heterozygous/alt genotypes.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Genotype matrix (samples x loci).
    """
    if loci_specs is None:
        loci_specs = [
            {"missing": 0.0, "het": 0.0},
            {"missing": 0.2, "het": 0.0},
            {"missing": 0.0, "het": 0.2},
            {"missing": 0.1, "het": 0.1},
        ]

    rng = np.random.default_rng(seed)
    matrix = []

    for i, spec in enumerate(loci_specs):
        locus_seed = rng.integers(0, 1_000_000)
        locus = simulate_locus_with_groups(
            n_samples=n_samples,
            missing_rate=spec["missing"],
            het_rate=spec["het"],
            group_enrichment=group_enrichment,
            seed=locus_seed,
        )
        matrix.append(locus)

    return np.column_stack(matrix)


def write_vcf(
    genotype_matrix: np.ndarray,
    output_file: str,
    sample_prefix: str = "Sample",
    chrom: str = "1",
    pos_start: int = 1000,
    locus_prefix: str = "rs",
    ref_alleles: list = None,
    alt_alleles: list = None,
):
    """
    Write a genotype matrix to a VCF file with per-locus REF and ALT alleles.

    Args:
        genotype_matrix (np.ndarray): Genotype matrix of shape (n_samples, n_loci) with values 0 (hom ref), 1 (het), 2 (hom alt), -9 (missing).
        output_file (str): Path to output VCF file.
        sample_prefix (str): Prefix for sample IDs.
        chrom (str): Chromosome ID (default "1").
        pos_start (int): Starting base-pair position for loci.
        locus_prefix (str): Prefix for locus IDs (e.g., rs).
        ref_alleles (list): List of REF alleles (length = number of loci).
        alt_alleles (list): List of ALT alleles (length = number of loci).
    """
    n_samples, n_loci = genotype_matrix.shape
    sample_ids = [f"{sample_prefix}{i+1}" for i in range(n_samples)]

    # Default: use "A" and "T" if none provided
    if ref_alleles is None:
        ref_alleles = ["A"] * n_loci
    if alt_alleles is None:
        alt_alleles = ["T"] * n_loci

    assert (
        len(ref_alleles) == n_loci
    ), "Length of ref_alleles must match number of loci."
    assert (
        len(alt_alleles) == n_loci
    ), "Length of alt_alleles must match number of loci."

    with open(output_file, "w") as f:
        # VCF headers
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=SimulatedVCFWriter\n")
        f.write(f"##contig=<ID={chrom}>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        f.write("\t".join(sample_ids) + "\n")

        # Write loci
        for locus_idx in range(n_loci):
            pos = pos_start + locus_idx
            locus_id = f"{locus_prefix}{locus_idx+1}"
            ref = ref_alleles[locus_idx]
            alt = alt_alleles[locus_idx]
            genotypes = genotype_matrix[:, locus_idx]

            # Map numeric to diploid genotype strings
            gt_strings = []
            for g in genotypes:
                if g == 0:
                    gt_strings.append("0/0")
                elif g == 1:
                    gt_strings.append("0/1")
                elif g == 2:
                    gt_strings.append("1/1")
                else:
                    gt_strings.append("./.")

            f.write(f"{chrom}\t{pos}\t{locus_id}\t{ref}\t{alt}\t.\tPASS\t.\tGT\t")
            f.write("\t".join(gt_strings) + "\n")


def main():
    n_samples = 50
    seed = 42

    loci_specs_no_missing_no_hets = [
        {"missing": 0.0, "het": 0.0},  # Column 1: All 0s and 2s, no hets or missing
        {"missing": 0.0, "het": 0.0},  # Column 2: 20% missing, rest 0s and 2s
        {"missing": 0.0, "het": 0.0},  # Column 3: 20% hets, enriched in certain group
        {"missing": 0.0, "het": 0.0},  # Column 4: 10% missing and 10% hets, enriched
    ]

    loci_specs_with_missing_only = [
        {"missing": 0.2, "het": 0.0},  # Column 1: 20% missing, rest 0s and 2s
        {"missing": 0.2, "het": 0.0},  # Column 2: 20% hets, enriched in certain group
        {"missing": 0.2, "het": 0.0},  # Column 3: 10% missing and 10% hets, enriched
        {"missing": 0.2, "het": 0.0},  # Column 4: 10% missing and 10% hets, enriched
    ]

    loci_specs_with_hets_only = [
        {"missing": 0.0, "het": 0.2},  # Column 1: 20% missing, rest 0s and 2s
        {"missing": 0.0, "het": 0.2},  # Column 2: 20% hets, enriched in certain group
        {"missing": 0.0, "het": 0.2},  # Column 3: 10% missing and 10% hets, enriched
        {"missing": 0.0, "het": 0.2},  # Column 4: 10% missing and 10% hets, enriched
    ]

    loci_specs_with_missing_and_hets = [
        {"missing": 0.1, "het": 0.1},  # Column 1: 20% missing, rest 0s and 2s
        {"missing": 0.1, "het": 0.1},  # Column 2: 20% hets, enriched in certain group
        {"missing": 0.1, "het": 0.1},  # Column 3: 10% missing and 10% hets, enriched
        {"missing": 0.1, "het": 0.1},  # Column 4: 10% missing and 10% hets, enriched
    ]

    # Simulate VCF files with different loci characteristics
    simulate_vcf_file(
        n_samples, seed, loci_specs_no_missing_no_hets, suffix="no_missing_no_hets"
    )

    # Simulate VCF files with different loci characteristics
    simulate_vcf_file(
        n_samples, seed, loci_specs_with_missing_only, suffix="missing_only"
    )

    simulate_vcf_file(n_samples, seed, loci_specs_with_hets_only, suffix="hets_only")

    simulate_vcf_file(
        n_samples, seed, loci_specs_with_missing_and_hets, suffix="missing_and_hets"
    )


def simulate_vcf_file(n_samples, seed, loci_specs, suffix=""):
    # Define group patterns: tuples of (start_index, end_index, enrichment_factor)
    group_enrichment = [
        (0, 10, 1.5),  # Enrich genotypes in samples 0-9
        (30, 40, 1.0),  # Enrich genotypes in samples 30-39
    ]

    genotype_matrix = simulate_genotype_matrix_with_groups(
        n_samples=n_samples,
        loci_specs=loci_specs,
        group_enrichment=group_enrichment,
        seed=seed,
    )

    print("Grouped Genotype Matrix:")
    print(genotype_matrix)

    ref_alleles = ["A", "G", "T", "C"]
    alt_alleles = ["C", "A", "G", "T"]

    write_vcf(
        genotype_matrix,
        output_file=f"scripts/simulated_data_{suffix}.vcf",
        ref_alleles=ref_alleles,
        alt_alleles=alt_alleles,
    )


if __name__ == "__main__":
    main()
