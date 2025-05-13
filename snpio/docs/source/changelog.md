# Changelog

This document outlines the changes made to the project with each
release.

## Version 1.23 (2025-04-08)

### Bug Fixes

- Fixed issues with `summary_statistics()` where the Weir and Cockerham (1984) Fst calculation was not being performed correctly. The method now calculates pairwise Weir and Cockerham (1984) Fst between populations, saves a heatmap with the Fst values, and saves the pairwise table as a CSV file. The method also returns a pandas DataFrame with the Fst values.
- Fixed issues with `neis_genetic_distance()` where the Nei's genetic distance calculation was not being performed correctly. The method now calculates pairwise Nei's genetic distance between populations, saves a heatmap with the genetic distances, and saves the pairwise table as a CSV file. The method also returns a pandas DataFrame with the genetic distances.`

## Version 1.2.1 (2025-01-06)

### Features

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

### Enhancements

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

### Bug Fixes

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
  