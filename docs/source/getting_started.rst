Getting Started
====================

This guide provides an overview of how to get started with the SNPio library. It covers the basic steps to read, manipulate, and analyze genotype data using the `GenotypeData` class.

Installation
-------------------

Before using SNPio, make sure it is installed in your Python environment. You can install it using pip. In the project root directory (the directory containing setup.py), type the following command into your terminal:

.. code-block:: shell

   pip install .

Importing SNPio
--------------------

To start using SNPio, import the necessary modules:

.. code-block:: python

   from snpio.read_input.genotype_data import GenotypeData
   from snpio.plotting.plotting import Plotting

.. note::

    *Important Notes:* GenotypeData and NRemover2 treat gap ('-', '?', '.') and 'N' characters as missing data. Also, if your input file is PHYLIP or STRUCTURE, they will be forced to be biallelic. If you need more than two alleles per site, then you must use the VCF file format, and even then some of the transformations force all sites to be biallelic.


Reading Alignment with Genotype Data
----------------------------------------

The first step is to read genotype data from an alignment. The `GenotypeData` class can read and write PHYLIP, STRUCTURE, and VCF files. VCF files can be either compressed with bgzip or uncompressed. GenotypeData can also convert between these three file formats and makes some informative plots. An example script, `run_snpio.py`` is also provided.



The `GenotypeData` class provides methods to read data in various formats, such as VCF, PHYLIP, STRUCTURE, and custom formats. Here's an example of reading genotype data from a VCF file:

.. code-block:: python

    genotype_data = GenotypeData(
        filename="example_data/phylip_files/phylogen_nomx.u.snps.phy",
        popmapfile="example_data/popmaps/test.nomx.popmap",
        force_popmap=True,
        filetype="auto",
        qmatrix_iqtree="example_data/trees/test.qmat",
        siterates_iqtree="example_data/trees/test.rate",
        guidetree="example_data/trees/test.tre",
    )

   # Access basic properties
   print(genotype_data.num_snps)  # Number of SNPs in the dataset
   print(genotype_data.num_inds)  # Number of individuals in the dataset
   print(genotype_data.populations)  # Population IDs
   print(genotype_data.popmap)  # Dictionary of SampleIDs as keys and popIDs as values
   print(genotype_data.popmap_inverse) # Dictionary of PopulationIDs as keys and a lists of SampleIDs for the given population as values.
   print(genotype_data.samples)  # Sample IDs in input order
   print(genotype_data.loci_indices) # If loci were removed, will be subset.
   print(genotype_data.sample_indices) # If samples were removed, will be subset.

   # You can print the alignment as a Biopython MultipleSeqAlignment
   # This is useful for visualzation.
   print(genotype_data.alignment)

   # Or you can use the alignment as a 2D list.
   print(genotype_data.snp_data)


Data Transformation and Analysis
-------------------------------------

Once you have the genotype data, you can perform various data transformations and analysis. Here's an example of running principal component analysis (PCA) on the genotype data:

.. code-block:: python

   components, pca = Plotting.run_and_plot_pca(
       data=genotype_data,
       ubp=False,
       scale=True,
       center=True,
       plot_format="png"
   )
   explvar = pca.explained_variance_ratio_

   # Access other transformed genotype data and attributes
   genotypes_012 = genotype_data.genotypes_012 # 012-encoded genotypes.
   genotypes_onehot = genotype_data.genotypes_onehot # onehot-encoded genotypes.
   alignment = genotype_data.alignment # Biopython.MultipleSeqAlignment object.
   vcf_attributes = genotype_data.vcf_attributes # If using VCF file.

   # Set and access additional properties
   q_matrix = genotype_data.q
   site_rates = genotype_data.site_rates
   newick_tree = genotype_data.tree


Alignment Filtering
===========================

The `NRemover2` class provides methods for filtering genetic alignments based on the proportion of missing data. It allows you to filter out sequences (samples) and loci (columns) that exceed a certain missing data threshold. It offers filtering options based on global missing data proportions,  per-population missing data proportions, minor allele frequency, and removing non-biallelic, monomorphic, and singleton sites. The class also provides informative plots related to the filtering process.

Attributes:
--------------

- `alignment` (list of Bio.SeqRecord.SeqRecord): The input alignment to filter.
- `populations` (list of str): The population for each sequence in the alignment.
- `loci_indices` (list of int): Indices that were retained post-filtering.
- `sample_indices` (list of int): Indices that were retained post-filtering.
- `msa`: (MultipleSeqAlignment): BioPython MultipleSeqAlignment object.

Methods:
-------------

- `nremover()`: Runs the whole NRemover2 pipeline.
- `filter_missing()`: Filters out sequences from the alignment that exceed a given proportion of missing data.
- `filter_missing_pop()`: Filters out loci (columns) where missing data from any given population exceed a given proportion threshold.
- `filter_missing_sample()`: Filters out samples from the alignment that exceed a given proportion of missing data.
- `filter_monomorphic()`: Filters out monomorphic sites from the alignment.
- `filter_singletons()`: Filters out loci (columns) where the only variant is a singleton.
- `filter_non_biallelic()`: Filters out loci (columns) that have more than two alleles.
- `filter_minor_allele_frequency()`: Filters out loci (columns) where the minor allele frequency is below the threshold.
- `get_population_sequences()`: Returns the sequences for a specific population as a dictionary object.
- `plot_missing_data_thresholds()`: Plots the proportion of missing data against a range of filtering thresholds so you can visualize missing data proportions across multiple thresholds.
- `plot_sankey_filtering_report()`: Make a Sankey plot showing the number of loci removed at each filtering step.
- `print_filtering_report()`: Prints a summary of the filtering results.
- `print_cletus()`: Prints ASCII art of Cletus from The Simpsons (a silly inside joke).

Usage Example:
-------------------

To illustrate how to use the `NRemover2` class, here's an example:

.. code-block:: python

   from snpio.filtering.nremover2 import NRemover2

   # Create an instance of NRemover2
    nrm = nremover2.NRemover2(gd)

    # Run nremover to filter out missing data.
    # Set the thresholds as desired.
    # Returns a GenotypeData object.
    gd_filtered = nrm.nremover(
        max_missing_global=0.5,
        max_missing_pop=0.5,
        max_missing_sample=0.9,
        singletons=True,
        biallelic=True,
        monomorphic=True,
        min_maf=0.01,
        plot_missingness_report=True,
        plot_dir="plots",
    )

If you do not want to use some of the filtering options, just leave them at default for the ones you don't want to run.


Writing to File and File Conversions
=========================================

If you want to write your output to a file, just do use one of the write functions. Any of the input file types can be written with any of the write functions.

.. code-block:: python

    gd_filtered.write_phylip("example_data/phylip_files/nremover_test.phy")

    gd_filtered.write_structure("example_data/structure_files/nremover_test.str")

    gd_filtered.write_vcf("example_data/vcf_files/nmremover_test.vcf")

For detailed information about the available methods and attributes, refer to the API Reference.

That's it! You have successfully completed the basic steps to get started with SNPio. Explore the library further to discover more functionality and advanced features.

For detailed information about the available methods and attributes, refer to the API Reference.

Indices and Tables
----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

