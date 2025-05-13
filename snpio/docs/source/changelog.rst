=================
Changelog
=================

This document outlines the changes made to the project with each release.

Version 1.2.3 (2025-04-08)
----------------------------

Bug Fixes
~~~~~~~~~

- Fixed issues with `summary_statistics()` where the Weir and Cockerham (1984) Fst calculation was not being performed correctly. The method now calculates pairwise Weir and Cockerham (1984) Fst between populations, saves a heatmap with the Fst values, and saves the pairwise table as a CSV file. The method also returns a pandas DataFrame with the Fst values.
- Fixed issues with `neis_genetic_distance()` where the Nei's genetic distance calculation was not being performed correctly. The method now calculates pairwise Nei's genetic distance between populations, saves a heatmap with the genetic distances, and saves the pairwise table as a CSV file. The method also returns a pandas DataFrame with the genetic distances.`

Version 1.2.1 (2025-01-06)
----------------------------

Features
~~~~~~~~~

- Improved the `PopGenStatistics` class to include new functionality to calculate genetic distances between populations:
    -  calculate genetic distances between populations using the `neis_genetic_distance()` method. The method calculates Nei's genetic distance between populations and returns a pandas DataFrame with the genetic distances.

- The `PopGenStatistics` class now has the following public (user-facing) methods:
    - `neis_genetic_distance`
    - `calculate_d_statistics`
    - `detect_fst_outliers`
    - `summary_statistics`
    - `amova`

- The AMOVA method now returns a dictionary with the AMOVA results. Its functionality has been greatly extended to follow Excoffier et al. (1992) and Excoffier et al. (1999) methods. The method now calculates the variance components (within populations, within regions among popoulations, and among regions), Phi-statistics, and p-values via bootstrapping for the AMOVA analysis. A `regionmap` dictionary is now required to map populations to regions/groups. The method also has the following new parameters:
    - `n_bootstraps`: The number of bootstraps to perform.
    - `n_jobs`: The number of jobs to run in parallel.
    - `random_seed`: The random seed for reproducibility.

Enhancements
~~~~~~~~~~~~

- Improved the `PopGenStatistics` class to include new functionality to calculate observed and expected heterozygosity per population and nucleotide diversity per population.
- Improved the `PopGenStatistics` class to include new functionality to calculate Weir and Cockerham's Fst between populations.
- Improved aesthetics of the Fst heatmap plot.
- Improved the `PopGenStatistics` class to include new functionality to plot D-statistics (Patterson's, Partitioned, and D-foil) and save them as CSV files.
- Improved the `PopGenStatistics` class to include new functionality to calculate Nei's genetic distance between populations.
- Improved the `PopGenStatistics` class to include new functionality to plot Nei's distance matrix between populations.
- Improved the `PopGenStatistics` class to include new functionality to plot Fst outliers.
    - Two ways:
        - DBSCAN clustering method
        - Bootstrapping method
- Improved the `PopGenStatistics` class to include new functionality to plot summary statistics. The method now returns a dictionary with the summary statistics.
- Improved the `PopGenStatistics` class to include new functionality to calculate AMOVA results. The method now returns a dictionary with the AMOVA results.
- Improved the `PopGenStatistics` class to include new functionality to calculate genetic distances between populations. The method calculates Nei's genetic distance between populations and returns a pandas DataFrame with the genetic distances.

Changes
~~~~~~~~

- Much of the code has been refactored to improve readability and maintainability. This includes moving the `neis_genetic_distance()` method to the `genetic_distance` module, the `amova()` method to the `amova` module, and the `fst_outliers()` method to the `fst_outliers` module. The `summary_statistics()` method has been moved to the `summary_statistics` module, and the D-statistics methods have been moved to the `d_statistics` module.

Deprecations
~~~~~~~~~~~~

The following method have been deprecated:

- `wrights_fst()`: Uses `weir_cockerham_fst_between_populations()` instead.

Bug Fixes
~~~~~~~~~~

- Fixed bug where the `PopGenStatistics` class did not have the `verbose` and `debug` attributes.
- Fixed bug where the `PopGenStatistics` class did not have the `genotype_data` attribute.
- Fixed warnings in `snpio.plotting.plotting.Plotting` class with the font family.
- Fixed bug with `VCFReader` class when a non-tabix-indexed and uncompressed VCF file was read. The bug caused an error when reading an uncompressed VCF file.

Version 1.2.0 (2024-11-07)
----------------------------

Features
~~~~~~~~~

- Added new functionality to calculate several population genetic statistics using the `PopGenStatistics` class, including:
    - Wright's Fst 
    - nucleotide diversity
    - expected and observed heterozygosity
    - Fst outliers
    - Patterson's, Partitioned, and D-Foil D-statistic tests
    - AMOVAs (Analysis of Molecular Variance)

- The `PopGenStatistics` class now has the following methods:
    - `calculate_d_statistics()`
    - `detect_fst_outliers()`
    - `observed_heterozygosity()`
    - `expected_heterozygosity()`
    - `nucleotide_diversity()`
    - `wrights_fst()`
    - `summary_statistics()`
    - `amova()`

Bootstrapping is performed for D-statistics and Fst outliers, and the results are saved as CSV files. The results are also returned as pandas DataFrames and dictionaries. The D-statistics are plotted, and the Fst outliers are plotted and saved as a CSV file. The summary statistics are plotted and returned as a dictionary.

Version 1.1.3 (2024-10-25)
----------------------------

Features
~~~~~~~~~

- Updated tree parsing functionality and added it to the ``TreeParser`` class in the ``analysis/tree_parser.py`` module to conform to refactor, and added new functionality to parse, modify, draw, and save Newick and NEXUS tree files.
- ``siterates`` and ``qmatrix`` files now dynamically determine if they are in IQ-TREE format or if they are just in a simple tab-delimited or comma-delimited format.
- ``site_rates`` and ``qmat`` are now read in as pandas DataFrames with less complex logic.
- Added unit test for tree parsing.
- Added integration test for tree parsing.
- Added documentation for tree parsing.

Bug Fixes
~~~~~~~~~~

- Fixed bug where the ``PhylipReader`` and ``StructureReader`` classes did not have the ``verbose`` and ``debug`` attributes.

Changes
~~~~~~~~

- ``q`` property is now called ``qmat`` for clarity and easier searching in files.
- Removed redundant ``siterates_iqtree`` and ``qmatrix_iqtree`` arguments attributes from the ``GenotypeData``, ``VCFReader``, ``PhylipReader``, ``StructureReader``, and ``TreeParser`` classes.
- Added error handling for tree parsing.
- Added error handling for ``siterates`` and ``qmatrix`` files.

Version 1.1.0 (2024-10-08)
----------------------------

Features
~~~~~~~~~

- Full refactor of the codebase to improve user-friendliness, maintainability and readability.
    - Method chaining: All functions now return the object itself, allowing for method chaining and custom filtering orders with ``NRemover2``.
    - Most objects now just take a ``GenotypeData`` object as input, making the code more modular and easier to maintain.
    - Improved documentation and docstrings.
    - Improved error handling.
    - Improved logging. All logging is now done with the Python logging module via the custom ``LoggerManager`` class.
    - Improved testing.
    - Improved performance.
        - Reduced memory usage.
        - Reduced disk usage.
        - Reduced CPU usage.
        - Reduced execution time, particularly for reading, loading, filtering, and processing large VCF files.
    - Improved plotting.
    - Improved data handling.
    - Improved file handling. All filenames now use pathlib.Path objects.
    - Code modularity: Many functions are now in separate modules for better organization.
    - Full unit tests for all functions.
    - Full integration tests for all functions.
    - Full documentation for all functions.

Version 1.0.5 (2023-09-16)
----------------------------

Features
~~~~~~~~~

- Added ``thin`` and ``random_subset`` options to ``nremover()`` function. ``thin`` removes loci within ``thin`` bases of the nearest locus. ``random_subset`` randomly subsets the loci using an integer or proportion.

Changes
~~~~~~~~

- Changed ``unlinked`` to ``unlinked_only`` option for clarity

Version 1.0.4 (2023-09-10)
-----------------------------

Features
~~~~~~~~~

- Added functionality to filter out linked SNPs using CHROM and POS fields from VCF file.

Performance
~~~~~~~~~~~~

- Made the Sankey plot function more modular and dynamic for easier maintainability.

Bug Fixes
~~~~~~~~~~

- Fix spacing between printed STDOUT.

Version 1.0.3.3 (2023-09-01)

Bug Fixes
~~~~~~~~~~

- Fixed bug where CHROM VCF field had strings cut off at 10 characters.

Version 1.0.3.2 (2023-08-28)
-----------------------------

Bug Fixes
~~~~~~~~~~

- Fixed copy method for pysam.VariantHeader objects.

Version 1.0.3 (2023-08-27)
--------------------------

Features
~~~~~~~~

- Performance improvements for VCF files.
- Load and write VCF file in chunks of loci to improve memory consumption.
- New output directory structure for better organization.
- VCF file attributes are now written to an HDF5 file instead of all being loaded into memory.
- Increased usage of numpy to improve VCF IO.
- Added AF INFO field when converting PHYLIP or STRUCTURE files to VCF format.
- VCF file reading uses pysam instead of cyvcf2 now.

Bug Fixes
~~~~~~~~~

- Fixed bug with `search_threshold` plots where the x-axis values would be sorted as strings instead of integers.
- Fixed bugs where sampleIDs were out of order for VCF files.
- Ensured correct order for all objects.
- Fixed bugs when subsetting with popmaps files.
- Fixed to documentation.

Version 1.0.2 (2023-08-13)
--------------------------

Bug Fixes
~~~~~~~~~~

- Fix for VCF FORMAT field being in wrong order.

Version 1.0.1 (2023-08-09)

Bug Fixes

~~~~~~~~~~

- Band-aid fix for incorrect order of sampleIDs in VCF files.

Initial Release
~~~~~~~~~~~~~~~

- Reads and writes PHYLIP, STRUCTURE, and VCF files.
- Loads data into GenotypeData object.
- Filters DNA sequence alignments using NRemover2.
    - Filters by minor allele frequence, monomorphic, and non-billelic sites
    - Filters with global (whole columns) and per-population, per-locus missing data thresholds.
- Makes informative plots.