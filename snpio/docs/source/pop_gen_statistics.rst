==============================
PopGenStatistics Documentation
==============================

.. class:: PopGenStatistics

The ``PopGenStatistics`` class is designed to perform a suite of population genetic analyses on SNP datasets. It includes methods for calculating D-statistics, detecting Fst outliers, computing genetic diversity metrics, AMOVAs, and calculating population genetic summary statistics. This class is particularly useful for researchers studying population structure, diversity, and genetic variation within and between populations. The class is part of the ``snpio`` package and is designed to work with genotype data loaded using the ``GenotypeData`` class.

------------
Key Features
------------

- Comprehensive suite of population genetic analyses for unlinked SNP datasets.
- Calculation of Patterson's, partitioned, and/ or D-foil D-statistics.
- Fst outlier detection using DBSCAN or bootstrapping. Supports one-tailed and two-tailed P-values.
- Estimation of heterozygosity, nucleotide diversity, and summary statistics.
- Nei's Distance matrix estimation and visualization between populations with dynamic ordering.
- Hierarchical AMOVA with bootstrapping and parallel bootstrapping computation.

----------------
Dependencies
----------------

PopGenStatistics is a part of the ``snpio`` package, which includes classes such as ``GenotypeEncoder``, ``Plotting``, and modules for calculating D-statistics. It depends on packages such as ``numpy``, ``pandas``, ``scipy``, ``scikit-learn``, ``statsmodels``, ``seaborn``, and ``plotly``.

-----------
Quick Start
-----------

.. code-block:: python

    # Import classes and initialize GenotypeData with SNP data.
    from snpio import VCFReader
    from snpio.popgenstats import PopGenStatistics

    # Load SNP data and metadata into a GenotypeData object
    genotype_data = VCFReader(
        filename="example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz",
        popmapfile="example_data/popmaps/phylogen_nomx.popmap",
        force_popmap=True,  # Remove samples not in the popmap or alignment.
        verbose=True,
    )

    # Initialize PopGenStatistics with genotype data
    pgs = PopGenStatistics(genotype_data, verbose=True)

----------------
Methods Overview
----------------

.. list-table:: Core Statistical Methods
    :header-rows: 1
    :class: responsive-table

    * - Class Method
      - Description
      - Supported Algorithm(s)
    * - ``calculate_d_statistics``
      - Calculates D-statistics and saves them as CSV.
      - Patterson's, partitioned, and D-foil D-statistics.
    * - ``detect_fst_outliers``
      - Identifies Fst outliers. Supports one-tailed & two-tailed P-values.
      - DBSCAN clustering, Traditional bootstrapping.
    * - ``summary_statistics``
      - Calculates several population genetic summary statistics.
      - Observed heterozygosity (Ho), Expected heterozygosity (He), Nucleotide diversity (Pi), Weir and Cockerham's Fst.
    * - ``amova``
      - Conducts AMOVA with bootstrapping and parallel computation.
      - Hierarchical AMOVA, variance components, Phi statistics.
    * - ``neis_genetic_distance``
      - Computes Nei's genetic distance between population pairs.
      - Nei's genetic distance.

---------------------
Calculated Statistics
---------------------

- D-statistics (Patterson's, Partitioned, and D-foil)
- Fst outliers and pairwise Fst values
- Observed heterozygosity (Ho)
- Expected heterozygosity (He)
- Nucleotide diversity (Pi)
- Weir and Cockerham's Fst (from summary statistics)
- AMOVA results (variance components and Phi statistics)
- Nei's genetic distance matrix between populations
- Summary statistics (Ho, He, Pi, Fst)

---------------------------------
Advanced Usage and Best Practices
---------------------------------

- **Distance Matrix Sorting:**  
  The ``plot_dist_matrix`` method employs dynamic ordering based on the sum of distances, ensuring that the populations with the highest overall divergence are highlighted.

- **Parallelization and Reproducibility:**  
  Many methods, including ``amova``, now support parallel computation via setting ``n_jobs=-1`` (or any positive integer value), and users are encouraged to define a random seed for consistent results.

----------------------
Additional Information
----------------------

.. note::

    - Sufficient bootstrap replicates are recommended for robust statistical estimation.
    - For parallel processing, set ``n_jobs=-1`` to utilize all available CPU cores.
    - The ``PopGenStatistics`` class is designed for SNP datasets and may not be suitable for other types of genetic data.

We hope this documentation helps you take full advantage of the features implemented in the ``PopGenStatistics`` class. For further details, refer to the source code or the official documentation. If you encounter any issues or have suggestions for improvements, please feel free to reach out to the developers. Thank you for using the SNPio package!
