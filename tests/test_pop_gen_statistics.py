import tempfile
import unittest
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from snpio import PhylipReader, PopGenStatistics

# Mapping of diploid genotypes to IUPAC codes
IUPAC_CODES = {
    "AA": "A",
    "TT": "T",
    "CC": "C",
    "GG": "G",
    "AT": "W",
    "TA": "W",
    "AC": "M",
    "CA": "M",
    "AG": "R",
    "GA": "R",
    "TC": "Y",
    "CT": "Y",
    "TG": "K",
    "GT": "K",
    "CG": "S",
    "GC": "S",
}


def generate_phylip_file(
    num_samples: int = 30,
    num_loci: int = 100,
    default_rng: np.random.Generator | None = None,
    populations: Dict[str, Dict[str, float]] | None = None,
    fst_outliers: List[int] | None = None,
    admixture: bool = False,
) -> str:
    """Generates a temporary PHYLIP-formatted file with specified allele frequencies and heterozygosity.

    Args:
        num_samples (int): Total number of samples.
        num_loci (int): Total number of loci.
        default_rng (np.random.Generator): Random number generator for reproducibility.
        populations (dict): Dictionary where keys are population IDs and values are allele frequencies per population.
        fst_outliers (list): List of SNP indices to introduce Fst outliers.
        admixture (bool): Whether to introduce admixture in population 2.

    Returns:
        str: Path to the generated temporary PHYLIP file.
    """
    if populations is None:
        # Define three populations with known frequencies for simplicity
        base_populations = {
            "pop1": {"A": 0.7, "T": 0.3},
            "pop2": {"A": 0.5, "T": 0.5},
            "pop3": {"A": 0.3, "T": 0.7},
        }
    else:
        base_populations = populations

    pop_sizes = {
        pop: num_samples // len(base_populations) for pop in base_populations.keys()
    }
    samples = []

    # Initialize genotype frequencies per SNP per population
    genotype_freqs = {}

    for pop_id in base_populations.keys():
        genotype_freqs[pop_id] = {}
        for snp_index in range(num_loci):
            if fst_outliers and snp_index in fst_outliers:
                if pop_id == "pop1":
                    p = 0.9
                else:
                    p = 0.1
            elif admixture and pop_id == "pop2":
                p = (base_populations["pop1"]["A"] + base_populations["pop3"]["A"]) / 2
            else:
                p = base_populations[pop_id]["A"]
            q = 1 - p
            freq_AA = p**2
            freq_AT = 2 * p * q
            freq_TT = q**2
            genotype_freqs[pop_id][snp_index] = {
                "AA": freq_AA,
                "AT": freq_AT,
                "TT": freq_TT,
            }

    for pop_id in base_populations.keys():
        for i in range(pop_sizes[pop_id]):
            sample_genotype = []
            for snp_index in range(num_loci):
                freqs = genotype_freqs[pop_id][snp_index]
                genotype = default_rng.choice(
                    ["AA", "AT", "TT"], p=[freqs["AA"], freqs["AT"], freqs["TT"]]
                )
                genotype_code = IUPAC_CODES[genotype]
                sample_genotype.append(genotype_code)
            samples.append(f"{pop_id}_{i+1}\t" + "".join(sample_genotype))

    temp_phylip_file = tempfile.NamedTemporaryFile(delete=False, suffix=".phylip")
    with open(temp_phylip_file.name, "w") as phylip_file:
        phylip_file.write(f"{num_samples} {num_loci}\n")
        phylip_file.write("\n".join(samples))

    return temp_phylip_file.name


def generate_population_map_file(populations: Dict[str, list] | None = None) -> str:
    """Generates a temporary population map file.

    Args:
        populations (dict): Dictionary with keys as population IDs and list of sample IDs as values.

    Returns:
        str: Path to the generated temporary population map file.
    """
    if populations is None:
        populations = {
            "pop1": [f"pop1_{i+1}" for i in range(10)],
            "pop2": [f"pop2_{i+1}" for i in range(10)],
            "pop3": [f"pop3_{i+1}" for i in range(10)],
        }

    rows = [(sample, pop) for pop, samples in populations.items() for sample in samples]
    popmap_df = pd.DataFrame(rows, columns=["SampleID", "PopulationID"])

    temp_popmap_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    popmap_df.to_csv(temp_popmap_file.name, index=False)

    return temp_popmap_file.name


class TestPopGenStatistics(unittest.TestCase):

    def setUp(self):
        """Set up common variables for tests."""
        self.verbose = False
        self.debug = False

    def test_summary_statistics(self):
        """Test PopGenStatistics.summary_statistics() for structure and biological accuracy."""
        default_rng = np.random.default_rng(42)

        phyfile = generate_phylip_file(
            num_samples=30, num_loci=100, default_rng=default_rng
        )
        popmapfile = generate_population_map_file()

        genotype_data = PhylipReader(
            filename=phyfile,
            popmapfile=popmapfile,
            prefix="test_popgen",
            verbose=self.verbose,
            debug=self.debug,
            plot_format="png",
        )

        popgen_stats = PopGenStatistics(
            genotype_data, verbose=self.verbose, debug=self.debug
        )

        summary_stats = popgen_stats.summary_statistics()

        # --- 1. Check structural validity ---
        self.assertIn("overall", summary_stats)
        self.assertIn("per_population", summary_stats)
        self.assertIn("Fst_between_populations_obs", summary_stats)

        self.assertIsInstance(summary_stats["overall"], pd.DataFrame)
        self.assertEqual(
            set(summary_stats["per_population"].keys()), {"pop1", "pop2", "pop3"}
        )

        # --- 2. Check shape ---
        self.assertEqual(summary_stats["overall"].shape, (100, 3))
        for df in summary_stats["per_population"].values():
            self.assertEqual(df.shape, (100, 3))

        # --- 3. Biological expectations ---
        # Expected He under average p = (0.7 + 0.5 + 0.3)/3 = 0.5
        expected_He_pop1 = 2 * 0.7 * 0.3  # 0.42
        expected_He_pop2 = 2 * 0.5 * 0.5  # 0.50
        expected_He_pop3 = 2 * 0.3 * 0.7  # 0.42
        expected_overall_he = np.mean(
            [expected_He_pop1, expected_He_pop2, expected_He_pop3]
        )  # ~0.4467

        expected_Pi_overall = expected_overall_he * 10 / 9  # Bias-corrected

        observed_ho = summary_stats["overall"]["Ho"]
        observed_he = summary_stats["overall"]["He"]
        observed_pi = summary_stats["overall"]["Pi"]

        # Mean values should be close to expected
        self.assertAlmostEqual(observed_he.mean(), expected_overall_he, delta=0.05)
        self.assertAlmostEqual(observed_pi.mean(), expected_Pi_overall, delta=0.05)
        self.assertTrue(np.all((observed_he >= 0) & (observed_he <= 0.55)))
        self.assertTrue(np.all((observed_ho >= 0) & (observed_ho <= 1)))

        # --- 4. Check Fst matrix symmetry and expectations ---
        fst_matrix = summary_stats["Fst_between_populations_obs"]
        self.assertIsInstance(fst_matrix, pd.DataFrame)
        self.assertEqual(set(fst_matrix.columns), {"pop1", "pop2", "pop3"})

        # Confirm symmetry
        self.assertAlmostEqual(
            fst_matrix.loc["pop1", "pop2"], fst_matrix.loc["pop2", "pop1"], delta=1e-6
        )
        self.assertAlmostEqual(
            fst_matrix.loc["pop1", "pop3"], fst_matrix.loc["pop3", "pop1"], delta=1e-6
        )
        self.assertAlmostEqual(
            fst_matrix.loc["pop2", "pop3"], fst_matrix.loc["pop3", "pop2"], delta=1e-6
        )

        # Check Fst order matches divergence (pop1 and pop3 most divergent)
        fst_12 = fst_matrix.loc["pop1", "pop2"]
        fst_23 = fst_matrix.loc["pop2", "pop3"]
        fst_13 = fst_matrix.loc["pop1", "pop3"]

        self.assertTrue(fst_13 > fst_12)
        self.assertTrue(fst_13 > fst_23)
        self.assertTrue(fst_12 < 0.1)
        self.assertTrue(fst_23 < 0.1)
        self.assertTrue(fst_13 > 0.2)

        # --- 5. Validate expected heterozygosity (He) per population ---
        per_pop = summary_stats["per_population"]

        self.assertAlmostEqual(
            per_pop["pop1"]["He"].mean(), expected_He_pop1, delta=0.03
        )
        self.assertAlmostEqual(
            per_pop["pop2"]["He"].mean(), expected_He_pop2, delta=0.03
        )
        self.assertAlmostEqual(
            per_pop["pop3"]["He"].mean(), expected_He_pop3, delta=0.03
        )

        # Ho should be close to but not exceed He significantly
        self.assertAlmostEqual(
            per_pop["pop1"]["Ho"].mean(), expected_He_pop1, delta=0.05
        )
        self.assertAlmostEqual(
            per_pop["pop2"]["Ho"].mean(), expected_He_pop2, delta=0.05
        )
        self.assertAlmostEqual(
            per_pop["pop3"]["Ho"].mean(), expected_He_pop3, delta=0.05
        )

        # --- 6. Validate Pi values per population (using bias-corrected He) ---
        expected_Pi_pop1 = expected_He_pop1 * 10 / 9
        expected_Pi_pop2 = expected_He_pop2 * 10 / 9
        expected_Pi_pop3 = expected_He_pop3 * 10 / 9

        self.assertAlmostEqual(
            per_pop["pop1"]["Pi"].mean(), expected_Pi_pop1, delta=0.03
        )
        self.assertAlmostEqual(
            per_pop["pop2"]["Pi"].mean(), expected_Pi_pop2, delta=0.03
        )
        self.assertAlmostEqual(
            per_pop["pop3"]["Pi"].mean(), expected_Pi_pop3, delta=0.03
        )

        # --- 7. Validate expected Fst hierarchy based on allele frequencies ---
        # Fst should be highest between pop1 and pop3, then pop1 and pop2,
        # Just check expected order of divergence
        self.assertTrue(fst_matrix.loc["pop1", "pop3"] > fst_matrix.loc["pop1", "pop2"])
        self.assertTrue(fst_matrix.loc["pop1", "pop3"] > fst_matrix.loc["pop2", "pop3"])

        self.assertTrue(0.01 < fst_matrix.loc["pop1", "pop2"] < 0.08)
        self.assertTrue(0.01 < fst_matrix.loc["pop2", "pop3"] < 0.08)


if __name__ == "__main__":
    unittest.main()
