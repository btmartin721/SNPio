import numpy as np
import pandas as pd
from scipy.stats import norm
import multiprocessing
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from kneed import KneeLocator
from ete3 import Tree
from sklearn.model_selection import train_test_split

# from xgboost import XGBClassifier
# import allel
from functools import partial
import sys
from collections import Counter

from snpio.plotting.plotting import Plotting
from snpio.filtering.nremover2 import NRemover2 as NRemover
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class PopGenStatistics:
    def __init__(self, popgenio):
        raise NotImplemented("PopGenStatistics has not yet been implemented.")
        if popgenio.filtered_alignment is not None:
            self.alignment = popgenio.filtered_alignment
        else:
            self.alignment = popgenio.alignment
        self.popmap = popgenio.populations
        self.populations = popgenio.populations
        self.plotting = Plotting(popgenio)
        self.nremover = NRemover(popgenio)

    def patterson_d_statistic_permutation(self, d1, d2, d3, outgroup):
        alignment_array = np.array(
            [list(rec) for rec in self.alignment], dtype=str
        )

        d1_base = alignment_array[d1, :]
        d2_base = alignment_array[d2, :]
        d3_base = alignment_array[d3, :]
        outgroup_base = alignment_array[outgroup, :]

        not_gap_mask = ~np.logical_or.reduce(
            (
                d1_base == "-",
                d2_base == "-",
                d3_base == "-",
                outgroup_base == "-",
            )
        )

        abba_mask = np.logical_and(
            d1_base == d2_base, d3_base == outgroup_base
        )
        baba_mask = np.logical_and(
            d1_base == d3_base, d2_base == outgroup_base
        )

        abba_count = np.sum(np.logical_and(abba_mask, not_gap_mask))
        baba_count = np.sum(np.logical_and(baba_mask, not_gap_mask))

        total_count = abba_count + baba_count

        d_statistic = (abba_count - baba_count) / total_count
        return d_statistic

    def partitioned_d_statistic_permutation(self, d1, d2, d3, d4, outgroup):
        alignment_array = np.array(
            [list(rec) for rec in self.alignment], dtype=str
        )

        d1_base = alignment_array[d1, :]
        d2_base = alignment_array[d2, :]
        d3_base = alignment_array[d3, :]
        d4_base = alignment_array[d4, :]
        outgroup_base = alignment_array[outgroup, :]

        not_gap_mask = ~np.logical_or.reduce(
            (
                d1_base == "-",
                d2_base == "-",
                d3_base == "-",
                d4_base == "-",
                outgroup_base == "-",
            )
        )

        abcd_mask = np.logical_and(
            d1_base == d2_base,
            np.logical_and(d3_base == d4_base, d3_base != outgroup_base),
        )
        dcba_mask = np.logical_and(
            d1_base == d4_base,
            np.logical_and(d2_base == d3_base, d2_base != outgroup_base),
        )

        abcd_count = np.sum(np.logical_and(abcd_mask, not_gap_mask))
        dcba_count = np.sum(np.logical_and(dcba_mask, not_gap_mask))

        total_count = abcd_count + dcba_count

        partitioned_d_statistic = (abcd_count - dcba_count) / total_count
        return partitioned_d_statistic

    def calculate_z_scores_and_p_values(self, d_statistic_values):
        d_statistic_array = np.array(list(d_statistic_values.values()))

        mean = np.mean(d_statistic_array)
        std_dev = np.std(d_statistic_array, ddof=1)

        z_scores = {
            combination: (value - mean) / std_dev
            for combination, value in d_statistic_values.items()
        }
        p_values = {
            combination: norm.sf(abs(z_score))
            for combination, z_score in z_scores.items()
        }

        # Apply Bonferroni correction
        num_tests = len(p_values)
        adjusted_p_values = {
            combination: min(1, p_value * num_tests)
            for combination, p_value in p_values.items()
        }

        result = {
            combination: {
                "z_score": z_scores[combination],
                "p_value": adjusted_p_values[combination],
            }
            for combination in d_statistic_values.keys()
        }
        return result

    def calculate_site_patterns(self, p1, p2, p3, p4, outgroup):
        """
        Calculate the site patterns from the given samples.

        Args:
            p1, p2, p3, p4 (str): Sample IDs.
            outgroup (str): Outgroup sample ID.

        Returns:
            dict: Dictionary of site patterns counts.
        """
        site_patterns = {"ABBA": 0, "BABA": 0, "BBAA": 0, "AAAB": 0, "AABB": 0}
        index_map = {p1: 0, p2: 1, p3: 2, p4: 3, outgroup: 4}
        indices = list(index_map.values())

        alignment_array = np.array([list(site) for site in self.alignment])

        for site in alignment_array[:, indices]:
            alleles = site.tolist()

            # Skip sites with missing or ambiguous data
            if "-" in alleles or "N" in alleles:
                continue

            # Calculate site pattern
            pattern = "".join(
                ["A" if a == alleles[4] else "B" for a in alleles[:4]]
            )

            # Increment the corresponding site pattern count
            if pattern in site_patterns:
                site_patterns[pattern] += 1

        return site_patterns

    def bootstrap_alignment(self, n_boot=100):
        """
        Bootstrap the input alignment.

        Args:
            n_boot (int): The number of bootstrap replicates to use.
        """
        self.bootstrapped_alignments = []
        for _ in range(n_boot):
            bootstrapped_alignment = self.alignment.copy()
            bootstrapped_alignment.loc[:, :] = np.random.choice(
                self.alignment.values.flatten(),
                size=self.alignment.shape,
                replace=True,
            ).reshape(self.alignment.shape)
            self.bootstrapped_alignments.append(bootstrapped_alignment)

    def calculate_z_scores_and_p_values(self, results):
        """
        Calculate Z-scores and P-values for the D-statistic methods.

        Args:
            results (list): A list of tuples containing the D-statistic values and their corresponding
                            standard errors for each permutation.

        Returns:
            z_scores (list): A list of Z-scores for each permutation.
            p_values (list): A list of P-values for each permutation.
            corrected_p_values (list): A list of Bonferroni-corrected P-values for each permutation.
        """
        d_statistics = [result[0] for result in results]
        standard_errors = [result[1] for result in results]

        # Calculate Z-scores
        z_scores = np.array(d_statistics) / np.array(standard_errors)

        # Calculate P-values
        p_values = 2 * norm.sf(np.abs(z_scores))

        # Apply Bonferroni correction
        corrected_p_values = np.minimum(p_values * len(p_values), 1)

        return z_scores, p_values, corrected_p_values

    def calculate_dfoil_statistic(self, p1, p2, p3, p4, outgroup):
        """
        Calculate the D-foil statistic for the given samples.

        Args:
            p1, p2, p3, p4 (str): Sample IDs.
            outgroup (str): Outgroup sample ID.

        Returns:
            float: D-foil statistic.
        """
        site_patterns = self.calculate_site_patterns(p1, p2, p3, p4, outgroup)

        site_patterns_array = np.array(
            [
                site_patterns["ABBA"],
                site_patterns["BABA"],
                site_patterns["BBAA"],
                site_patterns["AAAB"],
                site_patterns["AABB"],
            ]
        )

        dfoil_stat = (site_patterns_array[0] - site_patterns_array[1]) / (
            site_patterns_array[0]
            + site_patterns_array[1]
            + site_patterns_array[2]
            + site_patterns_array[3]
            + site_patterns_array[4]
        )

        return dfoil_stat

    def perform_permutations(
        self,
        sample_combinations,
        test_type="patterson",
        n_boot=1000,
        n_processes=None,
    ):
        """
        Perform permutations for the specified test_type using the loaded alignment.

        Args:
            sample_combinations (list(tuple())): List of tuples containing all sample combinations for the Dtest.
            test_type (str): The test type to run ("patterson", "partitioned", or "dfoil").
            n_boot (int): The number of bootstrap replicates to use.
            n_processes (int): The number of processes to use for multiprocessing.

        Returns:
            results_df (pd.DataFrame): A DataFrame containing the D-statistic, Z-score, and corrected P-value
                                       for each permutation.
        """
        # Create a partial function with the fixed test_type and n_boot arguments
        run_permutations_partial = partial(
            self._run_permutations, test_type, n_boot
        )

        if n_processes is None or n_processes <= 1:
            # Run permutations sequentially
            results_df = pd.concat(
                map(run_permutations_partial, sample_combinations)
            )
        else:
            # Run permutations in parallel
            with multiprocessing.Pool(n_processes) as p:
                results_list = p.map(
                    run_permutations_partial, sample_combinations
                )
                results_df = pd.concat(results_list)

        results_df.reset_index(drop=True, inplace=True)

        return results_df

    def _run_permutations(self, test_type, n_boot, samples):
        """
        Run a single permutation for the specified test_type using the given samples and
        the number of bootstrap replicates.

        Args:
            test_type (str): The test type to run ("patterson" or "partitioned").
            n_boot (int): The number of bootstrap replicates to use.
            samples (tuple): A tuple containing the samples for the permutation.

        Returns:
            results_df (pd.DataFrame): A DataFrame containing the D-statistic, Z-score, and corrected P-value
                                       for the permutation.
        """

        self.bootstrap_alignment(n_boot)
        boot_results = []

        for bootstrapped_alignment in self.bootstrapped_alignments:
            if test_type == "patterson":
                (
                    d_statistic,
                    standard_error,
                ) = self.patterson_d_statistic_permutation(
                    *samples, alignment=bootstrapped_alignment
                )
            elif test_type == "partitioned":
                (
                    d_statistic,
                    standard_error,
                ) = self.partitioned_d_statistic_permutation(
                    *samples, alignment=bootstrapped_alignment
                )
            elif test_type == "dfoil":
                d_statistic, standard_error = self.calculate_dfoil_statistic(
                    *samples, alignment=bootstrapped_alignment
                )
            else:
                raise ValueError(
                    "Invalid test_type. Must be 'patterson', 'partitioned', or 'dfoil'."
                )

            boot_results.append((d_statistic, standard_error))

        (
            z_scores,
            p_values,
            corrected_p_values,
        ) = self.calculate_z_scores_and_p_values(boot_results)

        # Store the results for this permutation
        results = {
            "Samples": str(samples),
            "Z-score": np.mean(z_scores),
            "P-value": np.mean(p_values),
            "Corrected P-value": np.mean(corrected_p_values),
        }

        # Convert results to a pandas DataFrame
        results_df = pd.DataFrame([results])

        return results_df

    # def get_population_sequences(self, population_name):
    #     """
    #     Retrieve sequences of the samples belonging to a given population.

    #     Args:
    #         population_name (str): The name of the population to retrieve sequences for.

    #     Returns:
    #         list: A list of sequences for the samples in the specified population.
    #     """
    #     # Assuming you have a dictionary that maps population names to sample names

    #     sample_names = self.populations[population_name]
    #     population_sequences = []
    #     for record in self.alignment:
    #         if record.id in sample_names:
    #             population_sequences.append(record)

    #     return population_sequences

    def calculate_1d_sfs(self, population, filter_singletons=False):
        pop_seqs = self.nremover.get_population_sequences(population)
        num_samples = len(pop_seqs)

        allele_counts = [
            Counter(
                [
                    seq[pos]
                    for seq in pop_seqs
                    if seq[pos] != "-" and seq[pos] != "N"
                ]
            )
            for pos in range(len(pop_seqs[0]))
        ]

        sfs = np.zeros(num_samples)
        for ac in allele_counts:
            if len(ac) == 2:  # biallelic
                count = min(ac.values())
                sfs[count - 1] += 1

        if filter_singletons:
            sfs = self.nremover.filter_singletons_sfs(sfs)

        return sfs

    def calculate_2d_sfs(
        self, population1, population2, filter_singletons=False
    ):
        pop1_seqs = self.nremover.get_population_sequences(population1)
        pop2_seqs = self.nremover.get_population_sequences(population2)
        num_samples1 = len(pop1_seqs)
        num_samples2 = len(pop2_seqs)

        allele_counts1 = [
            Counter(
                [
                    seq[pos]
                    for seq in pop1_seqs
                    if seq[pos] != "-" and seq[pos] != "N"
                ]
            )
            for pos in range(len(pop1_seqs[0]))
        ]

        allele_counts2 = [
            Counter(
                [
                    seq[pos]
                    for seq in pop2_seqs
                    if seq[pos] != "-" and seq[pos] != "N"
                ]
            )
            for pos in range(len(pop2_seqs[0]))
        ]

        sfs2d = np.zeros((num_samples1, num_samples2))
        for ac1, ac2 in zip(allele_counts1, allele_counts2):
            if len(ac1) == 2 and len(ac2) == 2:  # biallelic
                count1 = min(ac1.values())
                count2 = min(ac2.values())
                sfs2d[count1 - 1, count2 - 1] += 1

        if filter_singletons:
            sfs2d = self.nremover.filter_singletons_sfs(sfs2d)

        return sfs2d

    def detect_fst_outliers(self, fst_threshold):
        """
        Detect Fst outliers from the SNP data.

        Args:
            fst_threshold (float): Fst threshold for detecting outliers.

        Returns:
            pd.DataFrame: A DataFrame containing the Fst outliers.
        """

        def calculate_fst(column):
            allele_counts = (
                column.groupby(self.popmap["PopulationID"])
                .value_counts()
                .unstack(fill_value=0)
            )
            total_allele_counts = allele_counts.sum(axis=1)

            p = allele_counts.div(total_allele_counts, axis=0)
            p_mean = p.mean(axis=0)

            # Calculate Fst using the Weir and Cockerham (1984) formula
            nc = (
                total_allele_counts
                - (total_allele_counts**2).sum() / total_allele_counts.sum()
            )
            a = (p**2).div(nc, axis=0).sum(axis=0) - p_mean**2
            b = (
                p.mul(1 - p, axis=0).div(nc, axis=0).sum(axis=0)
                - (1 - p_mean) * p_mean
            )

            fst = a / (a + b)
            return fst

        chi2_stat = -2 * np.log(1 - fst_values)
        p_values = 1 - chi2.cdf(chi2_stat, df=1)

        fst_values = self.alignment.apply(calculate_fst, axis=0)
        fst_outliers = fst_values[fst_values >= fst_threshold]
        p_outliers = p_values[fst_values >= fst_threshold]

        return pd.DataFrame({"Fst": fst_outliers, "P-value": p_outliers})

    def observed_heterozygosity(self):
        """
        Calculate observed heterozygosity (Ho) for each locus.

        Returns:
            pd.Series: A pandas Series containing the observed heterozygosity values.
        """
        ho = (self.alignment == 1).sum(axis=0) / self.alignment.shape[0]
        return ho

    def expected_heterozygosity(self):
        """
        Calculate expected heterozygosity (He) for each locus.

        Returns:
            pd.Series: A pandas Series containing the expected heterozygosity values.
        """
        allele_freqs = (
            self.alignment.replace({0: np.nan, 2: np.nan}).mean(axis=0) / 2
        )
        he = 2 * allele_freqs * (1 - allele_freqs)
        return he

    def nucleotide_diversity(self):
        """
        Calculate nucleotide diversity (Pi) for each locus.

        Returns:
            pd.Series: A pandas Series containing the nucleotide diversity values.
        """
        n = self.alignment.shape[0]
        he = self.expected_heterozygosity()
        pi = (n / (n - 1)) * he
        return pi

    def wrights_fst(self):
        """
        Calculate Wright's fixation index (Fst) for each locus.

        Returns:
            pd.Series: A pandas Series containing the Fst values.
        """
        ho = self.observed_heterozygosity()
        he = self.expected_heterozygosity()
        fst = (he - ho) / he
        return fst

    def summary_statistics(self, save_plots=True):
        """
        Calculate a suite of summary statistics for SNP data.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated summary statistics.
        """
        ho = self.observed_heterozygosity()
        he = self.expected_heterozygosity()
        pi = self.nucleotide_diversity()
        fst = self.wrights_fst()

        summary_stats = pd.DataFrame(
            {"Ho": ho, "He": he, "Pi": pi, "Fst": fst}
        )

        if save_plots:
            self.plotting.plot_summary_statistics(summary_stats)

        return summary_stats

    def perform_amova(self):
        # Convert self.alignment to allel.HaplotypeArray format
        haplotype_array = allel.HaplotypeArray(self.alignment)

        # Create a list of population indices
        pop_indices = [
            np.where(np.array(self.sample_population_map) == pop_id)[0]
            for pop_id in self.population_sample_map.keys()
        ]

        # Perform AMOVA
        amova_results = allel.stats.distance.amova(
            haplotype_array, pop_indices
        )

        return amova_results

    def perform_samova(self, n_groups):
        # Convert self.alignment to allel.HaplotypeArray format
        haplotype_array = allel.HaplotypeArray(self.alignment)

        # Create a list of population indices
        pop_indices = [
            np.where(np.array(self.sample_population_map) == pop_id)[0]
            for pop_id in self.population_sample_map.keys()
        ]

        # Perform SAMOVA
        samova_results = allel.stats.distance.samova(
            haplotype_array, pop_indices, n_groups
        )

        return samova_results

    def perform_pca(
        self, n_components="auto", save_plot=True, plot_dimensions=2
    ):
        """
        Perform Principal Components Analysis (PCA) on self.alignment.

        Args:
            n_components (str, int, or float): The number of components to keep. If "auto" (default), KneeLocator will
            determine the optimal number of components based on the explained variance.

        Returns:
            sklearn.PCA: The fitted PCA object.
        """

        # Create a mapping dictionary
        char_to_int = {
            "A": 0,
            "T": 1,
            "C": 2,
            "G": 3,
            "N": np.nan,
            "-": np.nan,
            "R": 0,  # A/G, A is more frequent
            "Y": 2,  # C/T, C is more frequent
            "S": 2,  # G/C, C is more frequent
            "W": 0,  # A/T, A is more frequent
            "K": 1,  # G/T, T is more frequent
            "M": 0,  # A/C, A is more frequent
            "B": 2,  # C/G/T, C is most frequent
            "D": 0,  # A/G/T, A is most frequent
            "H": 0,  # A/C/T, A is most frequent
            "V": 0,  # A/C/G, A is most frequent
        }

        # Convert the input alignment to a numpy array of sequences
        alignment_array = np.array(
            [list(str(record.seq)) for record in self.alignment]
        )

        alignment_array = np.array(
            [[char_to_int[char] for char in row] for row in alignment_array]
        )

        imp = SimpleImputer(strategy="most_frequent")
        X = imp.fit_transform(alignment_array)

        ohe = OneHotEncoder(sparse_output=False)
        X = ohe.fit_transform(X)

        if n_components == "auto":
            pca = PCA()
            pca.fit(X)
            kneelocator = KneeLocator(
                range(1, len(pca.explained_variance_) + 1),
                pca.explained_variance_ratio_,
                curve="convex",
                direction="decreasing",
            )
            n_components = kneelocator.elbow

        pca = PCA(n_components=n_components)
        pca.fit(X)

        if save_plot:
            self.plotting.plot_pca(
                pca, X, self.popmap, dimensions=plot_dimensions
            )

        return pca

    def perform_dapc(
        self, n_components=None, save_plot=True, plot_dimensions=2
    ):
        """
        Perform Discriminant Analysis of Principal Components (DAPC) on self.alignment.

        Args:
            n_components (int, optional): The number of components to keep. If not provided, use cross-validation with
            the root mean squared error to select the optimal number of principal components.

        Returns:
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis: The fitted DAPC object.
        """
        if n_components is None:
            pca = PCA()
            pca.fit(self.alignment)
            pca_transformed = pca.transform(self.alignment)

            min_rmse = float("inf")
            best_n_components = None

            for n in range(1, len(pca.explained_variance_) + 1):
                lda = LinearDiscriminantAnalysis(n_components=n)
                X_train, X_test, y_train, y_test = train_test_split(
                    pca_transformed[:, :n],
                    self.popmap["PopulationID"],
                    test_size=0.3,
                )
                lda.fit(X_train, y_train)
                y_pred = lda.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                if rmse < min_rmse:
                    min_rmse = rmse
                    best_n_components = n

            n_components = best_n_components

        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(self.alignment)
        dapc = LinearDiscriminantAnalysis()
        dapc.fit(pca_transformed, self.popmap["PopulationID"])

        if save_plot:
            self.plotting.plot_dapc(
                dapc, self.popmap, dimensions=plot_dimensions
            )

        return dapc

    def impute_genotypes(self, method="xgboost", tree_file=None):
        """
        Impute missing genotypes using the specified method.

        Args:
            method (str): Imputation method. Options are "phylogeny" or "xgboost". Default is "xgboost".
            tree_file (str): Path to a Newick or Nexus file containing the phylogenetic tree. Required if method="phylogeny".
        """
        if method == "phylogeny":
            self._impute_phylogeny(tree_file)
        elif method == "xgboost":
            self._impute_xgboost()
        else:
            raise ValueError(
                "Invalid imputation method. Choose 'phylogeny' or 'xgboost'."
            )

    def _impute_phylogeny(self, tree_file):
        if tree_file is None:
            raise ValueError(
                "tree_file is required when using the phylogeny imputation method."
            )

        # Load the tree
        tree = Tree(tree_file)

    def _impute_phylogeny(self, tree_file):
        if tree_file is None:
            raise ValueError(
                "tree_file is required when using the phylogeny imputation method."
            )

        # Load the tree
        tree = Tree(tree_file)

        # Impute missing genotypes using the phylogenetic tree
        for site in self.alignment.columns:
            for sample in self.alignment.index:
                if pd.isna(self.alignment.at[sample, site]):
                    # Find the nearest non-missing neighbor in the phylogenetic tree
                    node = tree.search_nodes(name=sample)[0]
                    while pd.isna(self.alignment.at[node.name, site]):
                        node = node.get_closest_leaf()[0]

                    # Impute the missing genotype with the genotype of the nearest non-missing neighbor
                    self.alignment.at[sample, site] = self.alignment.at[
                        node.name, site
                    ]
