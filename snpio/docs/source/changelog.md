# Changelog

This document outlines the changes made to the project with each
release.

## Version 1.5.4 (2025-07-07)

### Bug Fixes

Dockerfile to ensure that the latest SNPio version is installed into the docker image. There was a bug where it was installing an older version of SNPio.

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
  