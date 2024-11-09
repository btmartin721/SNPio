==============================
PopGenStatistics Documentation
==============================

.. class:: PopGenStatistics

The `PopGenStatistics` class is designed to perform a suite of population genetic analyses on SNP datasets, supporting methods to calculate D-statistics, Fst, heterozygosity, and more. This class is particularly useful for researchers studying population structure, diversity, and genetic variation within and between populations.

**Key Features:**
    - Calculation of Patterson’s, partitioned, and D-foil D-statistics
    - Fst outlier detection using DBSCAN or bootstrapping
    - Calculation of heterozygosity, nucleotide diversity, and summary statistics
    - Perform Analysis of Molecular Variance (AMOVA) to assess genetic variation

**Dependencies:**
PopGenStatistics is a part of the `snpio` package, which includes classes to calculate GenotypeEncoder, Plotting, and D-statistics, and depends on other packages such as `numpy`, `pandas`, `scipy`, `sklearn`, and `statsmodels`.

-----------
Quick Start
-----------

.. code-block:: python

    # Import necessary classes and initialize GenotypeData with your SNP data
    from snpio import VCFReader
    from snpio.popgenstats import PopGenStatistics

    # Load SNP data and metadata into a GenotypeData object
    genotype_data = VCFReader(
        filename="example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz,
        popmapfile="example_data/popmaps/phylogen_nomx.popmap",
        force_popmap=True, # Remove samples not in the popmap or vice versa.
        verbose=True,
    )

    # Initialize PopGenStatistics with genotype data
    pgs = PopGenStatistics(genotype_data, verbose=True)

**Methods Overview:**

* `calculate_d_statistics`: Calculates D-statistics and saves them as a CSV. Also makes a plot of the D-statistics and returns a DataFrame with the statistics.
* `detect_fst_outliers`: Identifies Fst outliers between populations. Makes a plot of the Fst values and returns outlier SNPs as a DataFrame.
* `observed_heterozygosity`, `expected_heterozygosity`, `nucleotide_diversity`, and `wrights_fst`: Calculates core population genetic metrics.
* `summary_statistics`: Calculates and summarizes key metrics across populations. Makes informative plots and returns a dictionary of pandas DataFrame and Series objects with the results.
* `amova`: Conducts an Analysis of Molecular Variance. Returns a dictionary with the AMOVA results.

------------
Core Methods
------------

**calculate_d_statistics**

Calculates Patterson's D-statistic, partitioned D-statistic, or D-foil for given populations. Bootstrap replicates and heterozygosity inclusion can be customized. The results are saved as a CSV and returned as a pandas DataFrame and a dictionary with mean overall results, and a plot of the D-statistics is generated.

.. code-block:: python

    df, stats_summary = pgs.calculate_d_statistics(
        method="patterson",
        population1="EA",
        population2="GU",
        population3="TT",
        outgroup="OG",
        num_bootstraps=1000
    )

**detect_fst_outliers**

Detects Fst outliers using bootstrapping or DBSCAN. The method returns Fst outlier SNPs along with their associated population pairs.

.. code-block:: python

    outliers, pvals_df = pgs.detect_fst_outliers(
        correction_method="bonf", # perform Bonferroni P-value adjustments.
        alpha=0.05, # significance level after P-value adjustment.
        use_bootstrap=True,
        n_bootstraps=1000
    )

**summary_statistics**

Calculates a comprehensive suite of summary statistics, including heterozygosity, nucleotide diversity, and Fst. Results can be plotted or returned as a dictionary.

.. code-block:: python

    summary = pgs.summary_statistics()

    # Access overall summary statistics
    ho_overall = summary["overall"]["Ho"]
    he_overall = summary["overall"]["He"]
    pi_overall = summary["overall"]["Pi"]

    # Access population-specific summary statistics
    ho_pops = summary["populations"]["Ho"]
    he_pops = summary["populations"]["He"]
    pi_pops = summary["populations"]["Pi"]

The per-population summary statistics are stored in a dictionary with population labels as keys and pandas DataFrames as values.

**amova**

Conducts an Analysis of Molecular Variance (AMOVA) to assess genetic variation within and among populations.

.. code-block:: python

    amova_results = pgs.amova()
    print("Phi_ST:", amova_results["Phi_ST"])
    print("within_populations:", amova_results["Within_population_variance"])
    print("among_populations:", amova_results["Among_population_variance"])

---------------
Advanced Usage
---------------

- **Bootstrap Replicates in Fst Calculation**: To estimate the variance of Fst across SNPs, use the `detect_fst_outliers` method with `use_bootstrap=True`.
- **Multiple Population Comparisons in D-statistics**: The `calculate_d_statistics` method supports extended D-statistics calculations (e.g., D-foil) across more than four populations.
- **Plotting**: By default, plots for each metric are generated and saved. Customize `plot_kwargs` within your `GenotypeData` object if specific styling or debug configurations are needed.

----------------------
Additional Information
----------------------

**Notes:**
    - SNP data must be encoded in a compatible format.
    - `genotype_data.popmap` must map samples to population labels accurately.
    - It is advised to run the Fst and D-statistic calculations with sufficient bootstraps to obtain statistically robust estimates.

    **Parallelization**:
    - Many methods support parallel computation by specifying `n_jobs=-1` to use all available CPU cores, optimizing for large SNP datasets.
