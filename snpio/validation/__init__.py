"""Independent validation utilities for SNPio statistical estimators.

Validation modules are intentionally excluded from SNPio's public facade and
runtime analysis path. Optional simulation dependencies are imported only by
the validation commands that require them.
"""

from snpio.validation.linkage_disequilibrium import (
    DEFAULT_HAPLOTYPE_SCENARIOS,
    PUBLISHED_ISLAND_FOX_ESTIMATES,
    compare_published_estimates,
    exact_multinomial_expectation,
    genotype_probabilities_from_haplotypes,
    population_ld_statistics,
    prepare_island_fox_genepop,
    summarize_convergence,
    validate_exact_expectations,
    validate_golden_reference,
)

__all__ = [
    "DEFAULT_HAPLOTYPE_SCENARIOS",
    "PUBLISHED_ISLAND_FOX_ESTIMATES",
    "compare_published_estimates",
    "exact_multinomial_expectation",
    "genotype_probabilities_from_haplotypes",
    "population_ld_statistics",
    "prepare_island_fox_genepop",
    "summarize_convergence",
    "validate_exact_expectations",
    "validate_golden_reference",
]
