import tempfile
import unittest
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from snpio import PopGenStatistics
from snpio import PhylipReader

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
    populations: Optional[Dict[str, Dict[str, float]]] = None,
    fst_outliers: Optional[List[int]] = None,
    admixture: bool = False,
) -> str:
    """
    Generates a temporary PHYLIP-formatted file with specified allele frequencies and heterozygosity.

    Args:
        num_samples (int): Total number of samples.
        num_loci (int): Total number of loci.
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
                genotype = np.random.choice(
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


def generate_population_map_file(populations: Optional[Dict[str, list]] = None) -> str:
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
        self.verbose = True
        self.debug = True

    def test_amova(self):
        """Test the AMOVA method for expected variance components and Phi_ST."""
        np.random.seed(42)
        phyfile = generate_phylip_file(num_samples=30, num_loci=100)
        popmapfile = generate_population_map_file()

        genotype_data = PhylipReader(filename=phyfile, popmapfile=popmapfile)
        _ = genotype_data.snp_data
        popgen_stats = PopGenStatistics(
            genotype_data, verbose=self.verbose, debug=self.debug
        )

        amova_result = popgen_stats.amova()

        H_s = (0.42 + 0.5 + 0.42) / 3  # 0.4467
        H_t = 0.5  # Mean allele frequency is 0.5
        sigma2_among = H_t - H_s  # 0.0533
        Phi_ST = sigma2_among / H_t  # 0.1067

        expected_amova = {
            "Among_population_variance": sigma2_among,
            "Within_population_variance": H_s,
            "Phi_ST": Phi_ST,
        }

        for key, expected_val in expected_amova.items():
            self.assertAlmostEqual(amova_result[key], expected_val, places=2)

    def test_detect_fst_outliers_bootstrap(self):
        """Test Fst outlier detection using bootstrapping with and without p-value adjustment."""
        np.random.seed(42)
        phyfile = generate_phylip_file(
            num_samples=30, num_loci=100, fst_outliers=[0, 19]
        )
        popmapfile = generate_population_map_file()

        genotype_data = PhylipReader(filename=phyfile, popmapfile=popmapfile)
        _ = genotype_data.snp_data
        popgen_stats = PopGenStatistics(
            genotype_data, verbose=self.verbose, debug=self.debug
        )

        expected_outliers = pd.DataFrame(
            {
                "SNP": [0, 19],
                "Fst": [0.8, 0.8],  # Expected high Fst values
            }
        )

        outliers, _ = popgen_stats.detect_fst_outliers(
            use_bootstrap=True, n_bootstraps=10, correction_method=None
        )

        # Adjust the DataFrame to match the expected structure
        observed_outliers = outliers.reset_index()
        observed_outliers = observed_outliers[observed_outliers["index"].isin([0, 19])]

        # Rename columns for comparison
        observed_outliers = observed_outliers.rename(columns={"index": "SNP"})
        if "Fst_value" in observed_outliers.columns:
            observed_outliers = observed_outliers.rename(columns={"Fst_value": "Fst"})
        observed_outliers = observed_outliers[["SNP", "Fst"]]

        assert_frame_equal(
            observed_outliers.reset_index(drop=True),
            expected_outliers,
            check_exact=False,
            rtol=1e-1,
        )

    def test_summary_statistics(self):
        """Test the summary_statistics method to ensure correct output format and data types."""
        np.random.seed(42)
        phyfile = generate_phylip_file(num_samples=30, num_loci=100)
        popmapfile = generate_population_map_file()

        genotype_data = PhylipReader(filename=phyfile, popmapfile=popmapfile)
        _ = genotype_data.snp_data
        popgen_stats = PopGenStatistics(
            genotype_data, verbose=self.verbose, debug=self.debug
        )

        summary_stats = popgen_stats.summary_statistics()

        # Recalculate expected He
        p_mean = (0.7 + 0.5 + 0.3) / 3
        He = 2 * p_mean * (1 - p_mean)
        expected_he_overall = pd.Series([He] * 100)

        # Allow for sampling variation in He
        assert_series_equal(
            summary_stats["overall"]["He"].reset_index(drop=True),
            expected_he_overall,
            atol=0.05,
        )

    def test_calculate_d_statistics_patterson(self):
        """Test the calculation of D-statistics using the 'patterson' method."""
        np.random.seed(42)
        phyfile = generate_phylip_file(num_samples=30, num_loci=100, admixture=True)
        popmapfile = generate_population_map_file()

        genotype_data = PhylipReader(filename=phyfile, popmapfile=popmapfile)
        _ = genotype_data.snp_data
        popgen_stats = PopGenStatistics(
            genotype_data, verbose=self.verbose, debug=self.debug
        )

        # Expected values based on admixture
        expected_overall_results = {
            "Observed D-Statistic": 0.0,  # Adjust based on actual expected value
            "Z-Score": 0.0,
            "P-Value": 1.0,
        }

        _, overall_results = popgen_stats.calculate_d_statistics(
            method="patterson",
            population1="pop1",
            population2="pop2",
            population3="pop3",
            outgroup="pop3",
            num_bootstraps=5,
            max_individuals_per_pop=5,
        )

        for key, expected_val in expected_overall_results.items():
            self.assertAlmostEqual(overall_results[key], expected_val, places=1)


if __name__ == "__main__":
    unittest.main()
