=================
Changelog
=================

This document outlines the changes made to the project with each release.

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

