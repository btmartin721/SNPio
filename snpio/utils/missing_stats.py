from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class MissingStats:
    """Container for all missing-data statistics computed by `calc_missing`.

    This class holds the results of missing data calculations, including the proportion of missing calls for each locus, each individual, and optionally for each population if populations are used.

    Attributes:
        per_locus (pd.Series): Proportion of missing calls for each locus (index = locus).
        per_individual (pd.Series): Proportion of missing calls for each individual (index = sample ID).
        per_population_locus (pd.DataFrame | None): Missing proportion for every population-locus combination (rows = population, columns = locus). ``None`` if *use_pops* is False.
        per_population (pd.Series | None): Proportion of missing calls aggregated per population.
        per_individual_population (pd.DataFrame | None): Missing proportion for every population-individual combination (rows = population, columns = sample ID). ``None`` if *use_pops* is False.
    """

    per_locus: pd.Series
    per_individual: pd.Series
    per_population_locus: pd.DataFrame | None
    per_population: pd.Series | None
    per_individual_population: pd.DataFrame | None

    def summary(self) -> pd.DataFrame:
        """Return a compact, human-readable summary table.

        This method computes summary statistics for the missing data proportions, including mean, median, and maximum values for loci, individuals, and populations (if applicable).

        Returns:
            pd.DataFrame: Summary table with statistics for missing data proportions.
        """
        rows = [
            ("Loci (mean)", self.per_locus.mean()),
            ("Loci (median)", self.per_locus.median()),
            ("Loci (max)", self.per_locus.max()),
            ("Individuals (mean)", self.per_individual.mean()),
            ("Individuals (median)", self.per_individual.median()),
            ("Individuals (max)", self.per_individual.max()),
        ]

        if self.per_population is not None:
            rows.extend(
                [
                    ("Populations (mean)", self.per_population.mean()),
                    ("Populations (median)", self.per_population.median()),
                    ("Populations (max)", self.per_population.max()),
                ]
            )

        return (
            pd.DataFrame(rows, columns=["Statistic", "Missing Proportion"])
            .set_index("Statistic")
            .round(4)
        )
