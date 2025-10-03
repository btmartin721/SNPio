==========
Changelog
==========

This document outlines the changes made to the project with each release.

Version 1.6.8 (2025-10-03)
--------------------------

Bug Fixes - v1.6.8
~~~~~~~~~~~~~~~~~~

- Fixed bug where the ``write_vcf`` method did not include the parent path when writing the BGZipped output VCF file and associated Tabix index (.tbi). The method now correctly utilizes the full path, ensuring that the output file is saved in the intended directory.

Version 1.6.7 (2025-09-25)
--------------------------

No changes. Version bump issue.

Version 1.6.6 (2025-09-24)
--------------------------

Features - v1.6.6
~~~~~~~~~~~~~~~~~

- Added several new properties to the `GenotypeData` class:
    - `shape`: A tuple representing the alignment dimensions (n_samples, n_loci).
    - `num_pops`: The total number of unique populations.
    - `pop_sizes`: A dictionary mapping each population ID to its sample count.
    - `pop_to_indices`: A dictionary mapping each population ID to a list of its sample row indices.
    - `has_popmap`: A boolean indicating if population data is present.
    - `locus_names`: A list of concrete names for each locus.
    - `snpsdict`: A dictionary mapping sample IDs to their genotype sequences.
    - `inputs`: A dictionary of the keyword arguments used to initialize the object.
    - `is_empty`: A boolean that is `True` if the dataset has zero samples or loci.
    - `output_dir`: The root output directory path for generated files.
    - `plots_dir`: The dedicated directory path for plots.
    - `reports_dir`: The dedicated directory path for reports.
    - `missing_mask`: A boolean NumPy array where `True` marks a missing genotype.
    - `valid_mask`: A boolean NumPy array where `True` marks a non-missing genotype.
    - `het_mask`: A boolean NumPy array where `True` marks a heterozygous genotype.
    - `missing_rate`: The overall proportion of missing data in the alignment.
    - `per_locus_missing`: A pandas Series with the missing data proportion for each locus.
    - `per_individual_missing`: A pandas Series with the missing data proportion for each individual.
    - `per_locus_het_rate`: A pandas Series with the heterozygosity rate for each locus.
    - `per_individual_het_rate`: A pandas Series with the heterozygosity rate for each individual.
    - `is_missing_locus`: A boolean NumPy array that is `True` for loci missing in all samples.
    - `nbytes`: The approximate memory footprint of the `snp_data` array in bytes.

Enhancements - v1.6.6
~~~~~~~~~~~~~~~~~~~~~

- Improved the `GenotypeData` class by adding several new properties to enhance data accessibility and analysis capabilities. These properties provide quick access to key dataset characteristics, such as shape, population information, missing data metrics, and memory usage. This enhancement aims to streamline workflows and improve user experience when working with genotype data.
- Added support for when there is no available population map file.
- Added several new dataclasses to encapsulate related data and functionality, improving code organization and maintainability.

Version 1.6.5 (2025-09-24)
--------------------------

Bug Fixes - v1.6.5
~~~~~~~~~~~~~~~~~~

- Minor bug fixes in `GenotypeEncoder`.

Version 1.6.4 (2025-09-20)
--------------------------

Bug Fixes - v1.6.4
~~~~~~~~~~~~~~~~~~

- Fixed bug where the `GenotypeEncoder` would fail when decoding 0/1/2 genotypes. The `decode_012` method in the `GenotypeEncoder` class now correctly handles 0/1/2 encoded genotypes, ensuring that they are accurately converted back to their original string representations. This fix resolves issues that arose when using 0/1/2 encoded data.

Version 1.6.3 (2025-09-20)
--------------------------

Bug Fixes - v1.6.3
~~~~~~~~~~~~~~~~~~

- Fixed bug with 0/1/2 encodings where it would fail. The `GenotypeEncoder` class now correctly handles 0/1/2 encodings, ensuring that genotypes are accurately converted and processed. This fix resolves issues that arose when using 0/1/2 encoded data, allowing for seamless integration with other components of the SNPio library.

Enhancements - v1.6.3
~~~~~~~~~~~~~~~~~~~~~

- Improved the `GenotypeEncoder` class to better handle various genotype encodings, including 0/1/2 and IUPAC codes. The class now includes more consistency in methods for encoding and decoding genotypes, ensuring compatibility with a wider range of input formats. The order of "A", "C", "G", "T" has been standardized across methods to prevent confusion and errors during genotype processing.

Version 1.6.2 (2025-09-15)
--------------------------

Bug Fixes
~~~~~~~~~

- Fixed bug where the `ref` and `alt` properties of the `VCFReader` class did not exist before `NRemover2` filtering was applied. This caused errors when trying to access these properties before filtering. The properties now exist and return the correct values before and after filtering.
- Fixed bugs where the format metadata in the VCF file was not being stored in the HDF5 file when reading and writing VCF files with the `VCFReader` class. The `store_format_data` parameter now works as expected, and the format metadata is stored in the HDF5 file when set to `True`.
- Fixed edge case where the `alt` property could inadvertently set some loci to heterozygous IUPAC ambiguity codes (e.g., "R") or missing data values ("N").
- Fixed various minor bugs and edge cases that caused crashes with the example script.

Enhancements
~~~~~~~~~~~~

- Improved efficiency of the D-statistics calculations by precomputing the encodings for all individuals in the dataset once, and then reusing these encodings for each D-statistic calculation. This reduces redundant computations and speeds up the overall process, especially when dealing with large datasets or multiple population combinations.
- Cleaned up some of the MultiQC report plots for consistency. Particularly the Nei genetic distance heatmap and the Fst heatmap plots.
- General efficiency improvements to every module.
- Clarified the public-facing API of the Fst and Nei distance methods in the `PopGenStatistics` class. The methods now have clear and consistent parameter names, return types, and documentation. This makes it easier for users to understand how to use these methods and what to expect from them.
- Added tqdm progress bars to the Fst and Nei distance methods in the `PopGenStatistics` class. This provides users with visual feedback on the progress of the calculations, especially for large datasets where the computations may take a significant amount of time.
- The Fst and Nei distance methods now more clearly define how the permutation versus bootstrap methods work. The permutation method randomly shuffles individuals between populations to create a null distribution of Fst values, while the bootstrap method resamples loci with replacement to estimate the variability of Fst values. This distinction is now clearly documented in the method docstrings and user guides.

Features
~~~~~~~~

- Added multiprocessing support to the Weir & Cockerham Fst and Nei genetic distance methods in the `PopGenStatistics` class. This allows for parallel computation of pairwise Fst values between populations, significantly speeding up the process for large datasets with many populations. The number of parallel jobs can be controlled with the `n_jobs` parameter.

Version 1.6.1 (2025-09-01)
--------------------------

Bug Fixes
~~~~~~~~~

- Fixes to TreeParser class to ensure correct parsing and handling of Newick and NEXUS tree files. This includes better error handling and support for various tree formats.

Version 1.6.0 (2025-07-24)
--------------------------

Big update!

Highlights - v1.6.0
~~~~~~~~~~~~~~~~~~~

- New ``AlleleSummaryStats`` class to add new tables and visualizations.
- Fully functional and validated D-statistics (Patterson, Partitioned, and DFOIL).
- ``NRemover2`` class has been overhauled for efficiency and speed.
- New visualizations in the MultiQC report.
- Bug fixes (General)
- Documentation updates

Features - v1.6.0
~~~~~~~~~~~~~~~~~

- Added a new ``AlleleSummaryStats`` class to generate allele frequency summary statistics across populations. This class provides methods to calculate allele frequencies, visualize allele distributions, and export results in various formats.
    - The new class is called when ``PopGenStatistics(...).summary_statistics()`` is called, allowing for automatic generation of allele frequency statistics as part of the population genetics analysis workflow.
    - The ``AlleleSummaryStats`` class includes methods for:
        - Calculating allele frequencies per population
        - Visualizing allele frequency distributions
        - Exporting allele frequency data to CSV and JSON formats
        - Generating MultiQC reports with summary statistics and visualizations
- New visualizations have been added to the MultiQC report generator, including:
    -  allele frequency distributions
    -  D-tests
    -  `NRemover2` threshold searches
- New filtering method: ``filter_allele_depth()``. This method filters loci based on allele depth, allowing removal of low-quality or low-coverage loci.

Enhancements - v1.6.0
~~~~~~~~~~~~~~~~~~~~~

- Validated the D-statistics calculations against simulated datasets with known parameters to ensure accuracy and reliability of results.
    - D-statistics calculations now include:
        - Patterson's D
        - Partitioned-D
        - D-FOIL
    - These calculations are integrated into the MultiQC report generator, providing a comprehensive view of introgression.
- Improved performance of the D-statistics calculations by using ``numba`` and its ``njit`` decorator for JIT compilation in parallel, significantly speeding up the computation of large datasets.

Performance Improvements - v1.6.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The D-statistics calculations have been optimized for performance, particularly for large datasets. The use of `numba.njit` has significantly reduced computation time, making it feasible to analyze larger genomic datasets efficiently.
- The `NRemover2` class has been enhanced to handle larger datasets more efficiently, with improved memory management and reduced execution time for filtering operations. This was achieved by vectorizing operations and minimizing unnecessary data copies.

Bug Fixes - v1.6.0
~~~~~~~~~~~~~~~~~~

- Fixed a minor edge case in the `VCFReader` class that resulted in incorrect shapes when filtering with `NRemover2`. The shape of the ``loci_indices`` and ``sample_indices`` attributes is now correctly maintained after filtering operations.

Documentation Updates - v1.6.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use autosummary to generate documentation, and also limit the user-facing documentation to the public methods and attributes of the classes. This ensures that only relevant information is presented to users, making the documentation cleaner and more focused.

Version 1.5.5 (2025-07-07)
--------------------------

Bug Fixes - Docker Image
~~~~~~~~~~~~~~~~~~~~~~~~

- Docker image now correctly installs the latest version of SNPio from PyPI. Before, it was installing an older version due to a caching issue in the Docker build process. The Dockerfile has been updated to ensure that the latest version is always installed.

Version 1.5.0 (2025-07-04)
--------------------------

This major release introduces an all-new, fully interactive **MultiQC report generator** that integrates results across all SNPio modules. It also includes robust enhancements to the `PopGenStatistics` class, expanded functionality for downstream analyses, and critical bug fixes.

Features
~~~~~~~~

**MultiQC Report Integration**

- Introduced the `SNPioMultiQCReport` class to generate dynamic HTML reports for all SNPio modules, including:
    - `PopGenStatistics`, `VCFReader`, `PhylipReader`, `StructureReader`, `GenePopReader`, `NRemover2`, `SummaryStatistics`, `FstDistance`, `DStatistics`, `FstOutliers`, `Plotting`, and more.
- The report aggregates visualizations and tables across modules, offering a centralized and interactive way to explore SNPio results.

**Report Highlights**

- Summary statistics: plots and tables across modules
- Genetic distance visualizations:
    - **Weir and Cockerham's Fst (1984)** heatmap
    - **Nei's genetic distance** heatmap
- D-statistics visualizations for:
    - **Patterson's D**
    - **Partitioned D**
    - **D-FOIL D**
- Fst outlier detection plots:
    - **DBSCAN clustering method**
    - **Bootstrapping/permutation method**

**PopGenStatistics Enhancements**

- **`calculate_d_statistics()`**
    - Calculates Patterson's, Partitioned, and D-FOIL D-statistics
    - Optimized with `numba.jit` for performance
    - Returns a pandas DataFrame and CSV output
    - Automatically adds interactive plots to the MultiQC report
    - Supports per-population subsampling for targeted comparisons
- **`detect_fst_outliers()`**
    - Detects outlier loci using DBSCAN or permutation-based methods
    - Returns a DataFrame, saves plots, and integrates results with MultiQC
- **`summary_statistics()`**
    - Computes summary stats across and within populations
    - Now includes expected/observed heterozygosity, nucleotide diversity, and pairwise Fst
    - Results are returned as dictionaries and visualized interactively
- **`neis_genetic_distance()`**
    - Computes Nei's genetic distances between populations
    - Produces both distance matrices and heatmaps for the MultiQC report

Enhancements
~~~~~~~~~~~~

- Performance upgrades to D-statistic calculations using `numba.jit`
- More robust and flexible subsetting options for per-population analyses
- Improved consistency and formatting of plots and CSV outputs
- Extended support for custom pipelines via MultiQC-compatible outputs
- Updated documentation to reflect new features and usage examples
- Updated documentation for clarity and consistency, including detailed examples for the new MultiQC report generator and `PopGenStatistics` methods

Bug Fixes
~~~~~~~~~

- **VCFReader**: Fixed a critical issue related to HDF5 typing errors during VCF read/write operations
- **PopGenStatistics**: Corrected Fst P-value calculation logic when using the bootstrapping method; it now correctly applies permutation-based inference
- **Docker**: Updated Docker container setup for better dependency handling and performance

Version 1.3.21 (2025-06-16)
---------------------------

Documentation and CI/CD build fixes and updates.

Version 1.3.15 (2025-06-14)
---------------------------

Documentation and CI/CD build updates.

Version 1.3.14 (2025-06-12)
---------------------------

Fix sphinx documentation build issues that were introduced in the last release. The documentation now builds correctly without any errors or warnings.

Version 1.3.13 (2025-06-12)
---------------------------

Updated documentation to reflect the latest changes and features to the API in the last few releases. The documentation now includes detailed explanations of the new `GenePopReader` class, the `PopGenStatistics` class methods, and the overall functionality of the library.

Version 1.3.11 (2025-06-12)
---------------------------

Bug Fixes
~~~~~~~~~

- Fixed a critical bug in `VCFReader` class that caused reading and writing VCF files to fail due to a typing issue with HDF5 datasets. This bug was introduced in the previous version and has been resolved.

Version 1.3.9 (2025-06-11)
--------------------------

There have been a lot of changes since the last major release, including bug fixes, enhancements, and new features.

Bug Fixes
~~~~~~~~~

- Fixed bug where the `PopGenStatistics` class did not have the `verbose` and `debug` attributes.
- Fixed lots of bugs with VCFReader class when reading and writing VCF files.
- Fixed bugs in StructureReader and PhylipReader classes when reading and writing STRUCTURE and PHYLIP files.
- Fixed bug where the `PopGenStatistics` class did not have the `genotype_data` attribute.

Enhancements
~~~~~~~~~~~~

- VCFReader is now much faster, with benchmarks showing a 40 percent speedup when reading VCF files.
- Added optional `store_format_data` parameter to the `VCFReader` class to store FORMAT metadata in the HDF5 file. Set this to `True` to store FORMAT metadata in the HDF5 file. This can be useful if the format metadata is needed for downstream analysis, but it does drastically slow down the reading and writing of VCF files.
- Added support for reading and writing GenePop files with the `GenePopReader` class.
- `StructureReader` now supports `has_popids` and `has_marker_names` parameters to indicate whether the STRUCTURE file has population IDs column and marker names header row. This allows for more flexibility when reading STRUCTURE files.
- General improvements to code for performance and maintainability.

Features
~~~~~~~~

- Added new `GenePopReader` class to read and write GenePop files. This class can read GenePop files and convert them to any of the other supported formats. `write_genepop()` method can be used to write the data to a GenePop file from any of the supported formats (VCF, PHYLIP, STRUCTURE, GENEPOP).
- All file formats are interoperable and can be converted to and from each other. This means that you can read a VCF file, convert it to a PHYLIP file, and then convert it to a STRUCTURE file, and so on.

Version 1.2.1 (2025-01-06)
--------------------------

Features
~~~~~~~~

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
~~~~~~~

- Much of the code has been refactored to improve readability and maintainability. This includes moving the `neis_genetic_distance()` method to the `genetic_distance` module, the `amova()` method to the `amova` module, and the `fst_outliers()` method to the `fst_outliers` module. The `summary_statistics()` method has been moved to the `summary_statistics` module, and the D-statistics methods have been moved to the `d_statistics` module.

Deprecations
~~~~~~~~~~~~

The following method have been deprecated:

- `wrights_fst()`: Uses `weir_cockerham_fst_between_populations()` instead.

Bug Fixes
~~~~~~~~~

- Fixed bug where the `PopGenStatistics` class did not have the `verbose` and `debug` attributes.
- Fixed bug where the `PopGenStatistics` class did not have the `genotype_data` attribute.
- Fixed warnings in `snpio.plotting.plotting.Plotting` class with the font family.
- Fixed bug with `VCFReader` class when a non-tabix-indexed and uncompressed VCF file was read. The bug caused an error when reading an uncompressed VCF file.

Version 1.2.0 (2024-11-07)
--------------------------

Features
~~~~~~~~

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
--------------------------

Features
~~~~~~~~

- Updated tree parsing functionality and added it to the ``TreeParser`` class in the ``analysis/tree_parser.py`` module to conform to refactor, and added new functionality to parse, modify, draw, and save Newick and NEXUS tree files.
- ``siterates`` and ``qmatrix`` files now dynamically determine if they are in IQ-TREE format or if they are just in a simple tab-delimited or comma-delimited format.
- ``site_rates`` and ``qmat`` are now read in as pandas DataFrames with less complex logic.
- Added unit test for tree parsing.
- Added integration test for tree parsing.
- Added documentation for tree parsing.

Bug Fixes
~~~~~~~~~

- Fixed bug where the ``PhylipReader`` and ``StructureReader`` classes did not have the ``verbose`` and ``debug`` attributes.

Changes
~~~~~~~

- ``q`` property is now called ``qmat`` for clarity and easier searching in files.
- Removed redundant ``siterates_iqtree`` and ``qmatrix_iqtree`` arguments attributes from the ``GenotypeData``, ``VCFReader``, ``PhylipReader``, ``StructureReader``, and ``TreeParser`` classes.
- Added error handling for tree parsing.
- Added error handling for ``siterates`` and ``qmatrix`` files.

Version 1.1.0 (2024-10-08)
--------------------------

Features
~~~~~~~~

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
--------------------------

Features
~~~~~~~~

- Added ``thin`` and ``random_subset`` options to ``nremover()`` function. ``thin`` removes loci within ``thin`` bases of the nearest locus. ``random_subset`` randomly subsets the loci using an integer or proportion.

Changes
~~~~~~~

- Changed ``unlinked`` to ``unlinked_only`` option for clarity

Version 1.0.4 (2023-09-10)
--------------------------

Features
~~~~~~~~

- Added functionality to filter out linked SNPs using CHROM and POS fields from VCF file.

Performance
~~~~~~~~~~~

- Made the Sankey plot function more modular and dynamic for easier maintainability.

Bug Fixes
~~~~~~~~~

- Fix spacing between printed STDOUT.

Version 1.0.3.3 (2023-09-01)

Bug Fixes
~~~~~~~~~

- Fixed bug where CHROM VCF field had strings cut off at 10 characters.

Version 1.0.3.2 (2023-08-28)
----------------------------

Bug Fixes
~~~~~~~~~

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
~~~~~~~~~

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