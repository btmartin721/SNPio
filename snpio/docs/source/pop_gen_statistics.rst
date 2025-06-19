==============================
PopGenStatistics Documentation
==============================

.. class:: PopGenStatistics

The ``PopGenStatistics`` class is designed to perform a suite of population genetic analyses on SNP datasets. It includes methods for computing genetic diversity metrics and calculating population genetic summary statistics. It also includes some **still-experimental** features including: D-statistics calculation, Fst outlier detection, and AMOVA. This class is particularly useful for researchers studying population structure, diversity, and genetic variation within and between populations. The class is part of the ``snpio`` package and is designed to work with genotype data loaded using the ``GenotypeData`` class.

------------
Key Features
------------

- Estimation of heterozygosity, nucleotide diversity, and summary statistics.
- Nei's Distance matrix estimation and visualization between populations with dynamic ordering.
- Weir and Cockerham's (1984) Fst estimation.

Experimental Features
~~~~~~~~~~~~~~~~~~~~~

- D-statistics calculation (Patterson's D, partitioned D, and D-foil).
- Fst outlier detection using traditional bootstrapping or DBSCAN clustering.
- Hierarchical AMOVA with bootstrapping and parallel bootstrapping computation.

-----------
Quick Start
-----------

.. code-block:: python

    # Import classes and initialize GenotypeData with SNP data.
    from snpio import VCFReader, PopGenStatistics

    filename = "example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz"
    popmapfile = "example_data/popmaps/phylogen_nomx.popmap"

    # Load SNP data and metadata into a GenotypeData object
    genotype_data = VCFReader(
        filename=filename,
        popmapfile=popmapfile,
        force_popmap=True,
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
    * - ``summary_statistics``
      - Calculates several population genetic summary statistics.
      - Observed heterozygosity (Ho), Expected heterozygosity (He), Nucleotide diversity (Pi), Weir and Cockerham's Fst.
    * - ``neis_genetic_distance``
      - Computes Nei's genetic distance between population pairs.
      - Nei's genetic distance.
    * - ``calculate_d_statistics`` (Experimental)
      - Calculates D-statistics and saves them as CSV.
      - Patterson's, partitioned, and D-foil D-statistics.
    * - ``detect_fst_outliers`` (Experimental)
      - Identifies Fst outliers. Supports one-tailed & two-tailed P-values.
      - DBSCAN clustering, Traditional bootstrapping.
    * - ``amova`` (Experimental)
      - Conducts AMOVA with bootstrapping and parallel computation.
      - Hierarchical AMOVA, variance components, Phi statistics.

---------------------
Calculated Statistics
---------------------

- Pairwise Weir and Cockerham's (1984) Fst between populations
- Pairwise Nei's (1972) genetic distance matrix between populations
- Observed heterozygosity (Ho)
- Expected heterozygosity (He)
- Nucleotide diversity (Pi)
- D-statistics (Patterson's, Partitioned, and D-foil) (Experimental)
- Fst outliers (using bootstrapping and DBSCAN) (Experimental)
- AMOVA results (variance components and Phi statistics) (Experimental)

---------------------------------
Advanced Usage and Best Practices
---------------------------------

- **Parallelization and Reproducibility:**  
  Many methods support parallel computation via setting ``n_jobs=-1`` (or any positive integer value), and users are encouraged to define a random seed for consistent results.

----------------------
Additional Information
----------------------

.. note::

    - Sufficient bootstrap replicates are recommended for robust statistical estimation.
    - For parallel processing, set ``n_jobs=-1`` to utilize all available CPU cores.
    - The ``PopGenStatistics`` class is designed for biallelic SNP datasets and may not be suitable for other types of genetic data.

We hope this documentation helps you take full advantage of the features implemented in the ``PopGenStatistics`` class. For further details, refer to the source code or the official documentation. If you encounter any issues or have suggestions for improvements, please feel free to reach out to the developers. Thank you for using the SNPio package!
