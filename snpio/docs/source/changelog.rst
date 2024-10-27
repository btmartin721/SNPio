=================
Changelog
=================

This document outlines the changes made to the project with each release.

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