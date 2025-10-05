# Changelog

This document outlines the changes made to the project with each release.

## Version 1.6.10 (2025-10-05)

### Bug Fixes - v1.6.10

- Fixed bug where the `PopGenStatistics` class's `weir_cockerham_fst_between_populations` method could produce division by zero errors when populations had no samples. The method now includes checks to ensure that populations have samples before performing calculations, preventing division by zero errors and ensuring robust handling of edge cases.
- Fixed bug with `_plot_sankey_filtering_report` function in the `Plotting` class where it could fail if there were missing thresholds for MAF or MAC. The function now correctly handles NaN values in the threshold columns, ensuring that the data types are consistent and preventing errors during plotting. This fix improves the robustness of the plotting functionality when dealing with incomplete data.

## Version 1.6.9 (2025-10-05)

### Bug Fixes - v1.6.9

- Fixed bug with docker container where it was not correctly plotting the Sankey report if there were missing thresholds for MAF or MAC. The plotting function now correctly handles NaN values in the threshold columns, ensuring that the data types are consistent and preventing errors during plotting. This fix improves the robustness of the plotting functionality when dealing with incomplete data.

## Version 1.6.8 (2025-10-03)

- Fixed bug where the ``write_vcf`` method did not include the parent path when writing the BGZipped output VCF file and associated Tabix index (.tbi). The method now correctly utilizes the full path, ensuring that the output file is saved in the intended directory.
- Fixed bug where the ``write_vcf`` method was adding an extra `.gz` suffix to the output VCF filename when the provided filename already ended with `.gz`. The method now checks for the `.gz` suffix and removes duplicated `.gz` extensions, preventing issues with file naming and ensuring compatibility with downstream tools that expect standard VCF filenames.

## Version 1.6.7 (2025-09-25)

No changes. Version bump issue.

## Version 1.6.6 (2025-09-24)

### Features - v1.6.6

- Added several new properties to the `GenotypeData` class to enhance data accessibility and analysis capabilities:
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
  - `sample_indices`: A boolean array indicating which samples are retained after filtering.
  - `loci_indices`: A boolean array indicating which loci are retained after filtering.

## Version 1.6.5 (2025-09-24)

### Bug Fixes - v1.6.5

- Minor bug fixes in `GenotypeEncoder`.

## Version 1.6.4 (2025-09-20)

### Bug Fixes - v1.6.4

- Fixed bug where the `GenotypeEncoder` would fail when decoding 0/1/2 genotypes. The `decode_012` method in the `GenotypeEncoder` class now correctly handles 0/1/2 encoded genotypes, ensuring that they are accurately converted back to their original string representations. This fix resolves issues that arose when using 0/1/2 encoded data.

## Version 1.6.3 (2025-09-20)

### Bug Fixes - v1.6.3

- Fixed bug with 0/1/2 encodings where it would fail. The `GenotypeEncoder` class now correctly handles 0/1/2 encodings, ensuring that genotypes are accurately converted and processed. This fix resolves issues that arose when using 0/1/2 encoded data, allowing for seamless integration with other components of the SNPio library.

### Enhancements - v1.6.3

- Improved the `GenotypeEncoder` class to better handle various genotype encodings, including 0/1/2 and IUPAC codes. The class now includes more consistency in methods for encoding and decoding genotypes, ensuring compatibility with a wider range of input formats. The order of "A", "C", "G", "T" has been standardized across methods to prevent confusion and errors during genotype processing.

### Enhancements - v1.6.6

- Improved the `GenotypeData` class by adding several new properties to enhance data accessibility and analysis capabilities. These properties provide quick access to key dataset characteristics, such as shape, population information, missing data metrics, and memory usage. This enhancement aims to streamline workflows and improve user experience when working with genotype data.
- Added support for when there is no available population map file.
- Added several new dataclasses to encapsulate related data and functionality, improving code organization and maintainability.

## Version 1.6.2 (2025-09-15)

### Bug Fixes - 1.6.2

- Fixed bug where the `ref` and `alt` properties of the `VCFReader` class did not exist before `NRemover2` filtering was applied. This caused errors when trying to access these properties before filtering. The properties now exist and return the correct values before and after filtering.
- Fixed bugs where the format metadata in the VCF file was not being stored in the HDF5 file when reading and writing VCF files with the `VCFReader` class. The `store_format_data` parameter now works as expected, and the format metadata is stored in the HDF5 file when set to `True`.
- Fixed edge case where the `alt` property could inadvertently set some loci to heterozygous IUPAC ambiguity codes (e.g., `"R"`) or missing data values (`"N"`).
- Fixed various minor bugs and edge cases that caused crashes with the example script.

### Enhancements - 1.6.2

- Improved efficiency of the D-statistics calculations by precomputing the encodings for all individuals in the dataset once, and then reusing these encodings for each D-statistic calculation. This reduces redundant computations and speeds up the overall process, especially when dealing with large datasets or multiple population combinations.
- Cleaned up some of the MultiQC report plots for consistency—particularly the Nei genetic distance heatmap and the Fst heatmap plots.
- General efficiency improvements to every module.
- Clarified the public-facing API of the Fst and Nei distance methods in the `PopGenStatistics` class. The methods now have clear and consistent parameter names, return types, and documentation. This makes it easier for users to understand how to use these methods and what to expect from them.
- Added `tqdm` progress bars to the Fst and Nei distance methods in the `PopGenStatistics` class. This provides users with visual feedback on the progress of the calculations, especially for large datasets where the computations may take a significant amount of time.
- The Fst and Nei distance methods now more clearly define how the permutation versus bootstrap methods work. The permutation method randomly shuffles individuals between populations to create a null distribution of Fst values, while the bootstrap method resamples loci with replacement to estimate the variability of Fst values. This distinction is now clearly documented in the method docstrings and user guides.

### Features - 1.6.2

- Added multiprocessing support to the Weir & Cockerham Fst and Nei genetic distance methods in the `PopGenStatistics` class. This allows for parallel computation of pairwise Fst values between populations, significantly speeding up the process for large datasets with many populations. The number of parallel jobs can be controlled with the `n_jobs` parameter.

---

## Version 1.6.1 (2025-09-01)

### Bug Fixes - 1.6.1

- Fixes to `TreeParser` class to ensure correct parsing and handling of Newick and NEXUS tree files. This includes better error handling and support for various tree formats.

## Version 1.6.0 (2025-07-24)

Big update!

### Highlights - v1.6.0

- New ``AlleleSummaryStats`` class to add new tables and visualizations.
- Fully functional and validated D-statistics (Patterson, Partitioned, and DFOIL).
- ``NRemover2`` class has been overhauled for efficiency and speed.
- New visualizations in the MultiQC report.
- Bug fixes (General)
- Documentation updates

### Features - v1.6.0

- Added a new ``AlleleSummaryStats`` class to generate allele frequency summary statistics across populations. This class provides methods to calculate allele frequencies, visualize allele distributions, and export results in various formats.
  - The new class is called when ``PopGenStatistics(...).summary_statistics()`` is called, allowing for automatic generation of allele frequency statistics as part of the population genetics analysis workflow.
  - The ``AlleleSummaryStats`` class includes methods for:
    - Calculating allele frequencies per population
    - Visualizing allele frequency distributions
    - Exporting allele frequency data to CSV and JSON formats
    - Generating MultiQC reports with summary statistics and visualizations
- New visualizations have been added to the MultiQC report generator, including:
  - allele frequency distributions
  - D-tests
  - `NRemover2` threshold searches
- New filtering method: ``filter_allele_depth()``. This method filters loci based on allele depth, allowing removal of low-quality or low-coverage loci.

### Enhancements - v1.6.0

- Validated the D-statistics calculations against simulated datasets with known parameters to ensure accuracy and reliability of results.
  - D-statistics calculations now include:
    - Patterson's D
    - Partitioned-D
    - D-FOIL
  - These calculations are integrated into the MultiQC report generator, providing a comprehensive view of introgression.
- Improved performance of the D-statistics calculations by using ``numba`` and its ``njit`` decorator for JIT compilation in parallel, significantly speeding up the computation of large datasets.

### Performance Improvements - v1.6.0

- The D-statistics calculations have been optimized for performance, particularly for large datasets. The use of `numba.njit` has significantly reduced computation time, making it feasible to analyze larger genomic datasets efficiently.
- The `NRemover2` class has been enhanced to handle larger datasets more efficiently, with improved memory management and reduced execution time for filtering operations. This was achieved by vectorizing operations and minimizing unnecessary data copies.

### Bug Fixes - v1.6.0

- Fixed a minor edge case in the `VCFReader` class that resulted in incorrect shapes when filtering with `NRemover2`. The shape of the ``loci_indices`` and ``sample_indices`` attributes is now correctly maintained after filtering operations.

### Documentation Updates - v1.6.0

Use autosummary to generate documentation, and also limit the user-facing documentation to the public methods and attributes of the classes. This ensures that only relevant information is presented to users, making the documentation cleaner and more focused.

## Version 1.5.5 (2025-07-07)

### Bug Fixes - Docker Image

- Docker image now correctly installs the latest version of SNPio from PyPI. Before, it was installing an older version due to a caching issue in the Docker build process. The Dockerfile has been updated to ensure that the latest version is always installed.
- Additional docker image issues fixed.

## Version 1.5.0 (2025-07-04)

This major release introduces a fully interactive MultiQC report generator that integrates results across all SNPio modules. It also includes significant upgrades to the `PopGenStatistics` class, new downstream analysis capabilities, and key bug fixes.

---

### Features

#### 📊 MultiQC Report Integration

- Introduced the `SNPioMultiQCReport` class to generate interactive HTML reports across all SNPio modules, including:
  - `PopGenStatistics`, `VCFReader`, `PhylipReader`, `StructureReader`, `GenePopReader`, `NRemover2`, `SummaryStatistics`, `FstDistance`, `DStatistics`, `FstOutliers`, `Plotting`, and more
- The MultiQC report aggregates module outputs with summary statistics and visualizations in one place.

#### MultiQC Report Visualizations

- Summary statistics plots and tables across modules
- Heatmaps:
  - **Weir and Cockerham’s Fst (1984)**
  - **Nei’s genetic distance**
- D-statistics visualizations:
  - Patterson’s D
  - Partitioned D
  - D-FOIL D
- Fst outlier plots using:
  - **DBSCAN clustering**
  - **Bootstrapping/permutation testing**

#### PopGenStatistics Enhancements

- `calculate_d_statistics()`:
  - Computes Patterson’s, Partitioned, and D-FOIL D-statistics
  - Optimized using `numba.jit` for speed
  - Returns a pandas DataFrame and exports results as CSV
  - Generates MultiQC-integrated plots
  - Supports individual subsetting per population
- `detect_fst_outliers()`:
  - Detects outliers using DBSCAN or permutation methods
  - Returns a DataFrame and produces MultiQC-compatible outputs
- `neis_genetic_distance()`:
  - Computes Nei’s genetic distances between populations
  - Produces both DataFrames and interactive heatmaps
- `summary_statistics()`:
  - Includes nucleotide diversity, observed/expected heterozygosity, and pairwise Fst
  - Visualizations now integrated directly into MultiQC

---

### 🧠 Enhancements

- Improved performance of D-statistics via `numba.jit`
- Enhanced per-population subsetting options for targeted comparison
- Unified output formatting for CSVs and plots
- Extended MultiQC support across all population-genetic analyses

---

### 🐛 Bug Fixes

- **VCFReader**: Fixed a critical bug related to HDF5 typing that caused read/write failures for VCF files
- **PopGenStatistics**: Resolved an issue where Fst P-values were incorrectly calculated in bootstrap mode; now correctly uses permutation-based inference
- **Docker**: Updated Docker container setup for compatibility with latest dependencies and improved runtime performance

## Version 1.3.21 (2025-06-16)

Documentation and CI/CD build fixes and updates.

## Version 1.3.15 (2025-06-14)

Documentation and CI/CD build updates.

## Version 1.3.14 (2025-06-12)

Fix sphinx documentation build issues that were introduced in the last release. The documentation now builds correctly without any errors or warnings.

## Version 1.3.13 (2025-06-12)

- Updated documentation for clarity and updates for new API features added or enhanced in the last few versions.

## Version 1.3.11 (2025-06-12)

Fixed a critical bug introduced in v1.3.9 where `VCFReader` would fail due to a typing issue with HDF5 file IO.

## Version 1.3.9 (2025-06-11)

There have been a lot of changes since the last major release, including bug fixes, enhancements, and new features.

### Bug Fixes (v1.3.9)

- Fixed bug where the `PopGenStatistics` class did not have the `verbose` and `debug` attributes.
- Fixed lots of bugs with VCFReader class when reading and writing VCF files.
- Fixed bugs in StructureReader and PhylipReader classes when reading and writing STRUCTURE and PHYLIP files.
- Fixed bug where the `PopGenStatistics` class did not have the `genotype_data` attribute.

### Enhancements

- VCFReader is now much faster, with benchmarks showing a 40 percent speedup when reading VCF files.
- Added optional `store_format_data` parameter to the `VCFReader` class to store FORMAT metadata in the HDF5 file. Set this to `True` to store FORMAT metadata in the HDF5 file. This can be useful if the format metadata is needed for downstream analysis, but it does drastically slow down the reading and writing of VCF files.
- Added support for reading and writing GenePop files with the `GenePopReader` class.
- `StructureReader` now supports `has_popids` and `has_marker_names` parameters to indicate whether the STRUCTURE file has population IDs column and marker names header row. This allows for more flexibility when reading STRUCTURE files.
- General improvements to code for performance and maintainability.

### Features

- Added new `GenePopReader` class to read and write GenePop files. This class can read GenePop files and convert them to any of the other supported formats. `write_genepop()` method can be used to write the data to a GenePop file from any of the supported formats (VCF, PHYLIP, STRUCTURE, GENEPOP).
- All file formats are interoperable and can be converted to and from each other. This means that you can read a VCF file, convert it to a PHYLIP file, and then convert it to a STRUCTURE file, and so on.

## Version 1.23 (2025-04-08)

### Bug Fixes

- Fixed issues with `summary_statistics()` where the Weir and Cockerham (1984) Fst calculation was not being performed correctly. The method now calculates pairwise Weir and Cockerham (1984) Fst between populations, saves a heatmap with the Fst values, and saves the pairwise table as a CSV file. The method also returns a pandas DataFrame with the Fst values.
- Fixed issues with `neis_genetic_distance()` where the Nei's genetic distance calculation was not being performed correctly. The method now calculates pairwise Nei's genetic distance between populations, saves a heatmap with the genetic distances, and saves the pairwise table as a CSV file. The method also returns a pandas DataFrame with the genetic distances.`

## Version 1.2.1 (2025-01-06)

### Features 1.2.1

- Improved the <span class="title-ref">PopGenStatistics</span> class to include new functionality to calculate genetic distances between populations:  
  - calculate genetic distances between populations using the
    <span class="title-ref">neis_genetic_distance()</span> method. The
    method calculates Nei's genetic distance between populations and
    returns a pandas DataFrame with the genetic distances.

- The <span class="title-ref">PopGenStatistics</span> class now has the following public (user-facing) methods:  
  - <span class="title-ref">neis_genetic_distance</span>
  - <span class="title-ref">calculate_d_statistics</span>
  - <span class="title-ref">detect_fst_outliers</span>
  - <span class="title-ref">summary_statistics</span>
  - <span class="title-ref">amova</span>

- The AMOVA method now returns a dictionary with the AMOVA results. Its functionality has been greatly extended to follow Excoffier et al. (1992) and Excoffier et al. (1999) methods. The method now calculates the variance components (within populations, within regions among popoulations, and among regions), Phi-statistics, and p-values via bootstrapping for the AMOVA analysis. A <span class="title-ref">regionmap</span> dictionary is now required to map populations to regions/groups. The method also has the following new parameters:  
  - \`n_bootstraps\`: The number of bootstraps to perform.
  - \`n_jobs\`: The number of jobs to run in parallel.
  - \`random_seed\`: The random seed for reproducibility.

### Enhancements 1.2.1

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to calculate observed and expected
  heterozygosity per population and nucleotide diversity per population.

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to calculate Weir and Cockerham's Fst
  between populations.

- Improved aesthetics of the Fst heatmap plot.

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to plot D-statistics (Patterson's,
  Partitioned, and D-foil) and save them as CSV files.

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to calculate Nei's genetic distance between
  populations.

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to plot Nei's distance matrix between
  populations.

- Improved the <span class="title-ref">PopGenStatistics</span> class to include new functionality to plot Fst outliers.  
  - Two ways:  
    - DBSCAN clustering method
    - Bootstrapping method

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to plot summary statistics. The method now
  returns a dictionary with the summary statistics.

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to calculate AMOVA results. The method now
  returns a dictionary with the AMOVA results.

- Improved the <span class="title-ref">PopGenStatistics</span> class to
  include new functionality to calculate genetic distances between
  populations. The method calculates Nei's genetic distance between
  populations and returns a pandas DataFrame with the genetic distances.

### Changes

- Much of the code has been refactored to improve readability and
  maintainability. This includes moving the
  <span class="title-ref">neis_genetic_distance()</span> method to the
  <span class="title-ref">genetic_distance</span> module, the
  <span class="title-ref">amova()</span> method to the
  <span class="title-ref">amova</span> module, and the
  <span class="title-ref">fst_outliers()</span> method to the
  <span class="title-ref">fst_outliers</span> module. The
  <span class="title-ref">summary_statistics()</span> method has been
  moved to the <span class="title-ref">summary_statistics</span> module,
  and the D-statistics methods have been moved to the
  <span class="title-ref">d_statistics</span> module.

### Deprecations

The following method have been deprecated:

- \`wrights_fst()\`: Uses
  <span class="title-ref">weir_cockerham_fst_between_populations()</span>
  instead.

### Bug Fixes 1.2.1

- Fixed bug where the <span class="title-ref">PopGenStatistics</span>
  class did not have the <span class="title-ref">verbose</span> and
  <span class="title-ref">debug</span> attributes.
- Fixed bug where the <span class="title-ref">PopGenStatistics</span>
  class did not have the <span class="title-ref">genotype_data</span>
  attribute.
- Fixed warnings in
  <span class="title-ref">snpio.plotting.plotting.Plotting</span> class
  with the font family.
- Fixed bug with <span class="title-ref">VCFReader</span> class when a
  non-tabix-indexed and uncompressed VCF file was read. The bug caused
  an error when reading an uncompressed VCF file.
  