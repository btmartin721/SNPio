<img src="https://github.com/btmartin721/SNPio/blob/master/img/snpio_logo.png" width="50%" alt="SNPio logo">

# SNPio
API to read, write, and filter PHYLIP, STRUCTURE, and VCF files using a GenotypeData object.

In addition to the below tutorial, see our [API Documentation](https://snpio.readthedocs.io/en/latest/#) for more information.

# Getting Started

This guide provides an overview of how to get started with the SNPio library. It covers the basic steps to read, manipulate, and analyze genotype data using the `GenotypeData` class.

## Installation

Before using SNPio, make sure it is installed in your Python environment. You can install it using pip. In the project root directory (the directory containing setup.py), type the following command into your terminal:

```
pip install snpio
```

## Importing SNPio

To start using SNPio, import the necessary modules:

```
from snpio import GenotypeData
from snpio import Plotting
```

> **Important Notes:** GenotypeData and NRemover2 treat gap ('-', '?', '.') and 'N' characters as missing data. Also, if your input file is PHYLIP or STRUCTURE, they will be forced to be biallelic. If you need more than two alleles per site, then you must use the VCF file format, and even then some of the transformations force all sites to be biallelic.

## The Population Map File

To use `GenotypeData` you'll need a population map (popmap) file. It is basically just a two-column tab-delimited file with SampleIDs in the first column and the corresponding PopulationIDs in the second column. 

For example:

```
Sample1\tPopulation1
Sample2\tPopulation1
Sample3\tPopulation2
Sample4\tPopulation2
...
```

## Optional Input files

There are some other optional input files you can provide as well. These include a phylogenetic tree in NEWICK format and the site rates and Q-matrix obtained from running IQ-TREE. THe latter two can be found in the output from IQ-TREE.

Currently, we don't have functionality to do any analyses on the tree, site_rates, and q-matrix objects, but we plan to implement more features that incorporate them in the future.

## Reading Alignment with Genotype Data

The first step is to read genotype data from an alignment. The `GenotypeData` class can read and write PHYLIP, STRUCTURE, and VCF files. VCF files can be either compressed with bgzip or uncompressed. GenotypeData can also convert between these three file formats and makes some informative plots. An example script, `run_snpio.py` is provided to showcase some of SNPio's functionality.

The files referenced in the code blocks below can be found in the provided `example_data/` directory.

The `GenotypeData` class provides methods to read and write data in various formats, such as VCF, PHYLIP, STRUCTURE, and custom formats. Here's an example of reading genotype data from a VCF file:

```
from snpio import GenotypeData
from snpio import Plotting

# Read the alignment, popmap, and tree files
gd = GenotypeData(
    filename="example_data/phylip_files/phylogen_nomx.u.snps.phy",
    popmapfile="example_data/popmaps/phylogen_nomx.popmap",
    force_popmap=True,
    filetype="auto",
    qmatrix_iqtree="example_data/trees/test.qmat",
    siterates_iqtree="example_data/trees/test.rate",
    guidetree="example_data/trees/test.tre",
    include_pops=["EA", "TT", "GU"], # Only include these populations. There's also an exclude_pops option that will exclude the provided


    populations.
)

# Print out the phylogenetic tree that you provided as a newick input file.
print(gd.tree)

# Access basic properties
print(gd.num_snps)  # Number of SNPs (loci) in the dataset
print(gd.num_inds)  # Number of samples in the dataset
print(gd.populations)  # List of population IDs
print(gd.popmap)  # Dictionary of SampleIDs as keys and popIDs as values

# Dictionary of PopulationIDs as keys and a lists of SampleIDs 
# for the given population as values.
print(genotype_data.popmap_inverse) 

print(gd.samples)  # Sample IDs in input order
print(gd.loci_indices) # If loci were removed, will be subset.
print(gd.sample_indices) # If samples were removed, will be subset.

# You can print the alignment as a Biopython MultipleSeqAlignment
# This is useful for visualzation.
print(gd.alignment)

# Or you can use the alignment as a 2D list.
print(gd.snp_data)

# Get a numpy array of snp_data
print(np.array(gd.snp_data))
```

Here's the alignment object:

```
Alignment with 161 rows and 6724 columns
GNNNNCNNNNRNCNTNCNANNCNCGGGGCNNNCNTNNNTNNNNN...NCN EAAL_BX1380
NNGNNCNCNRGNNGTNCCNNNCCSNNNNNNGNNNYCCATTNGKN...NNT EAAL_BX211
GAGTACNCGGRGCNTTCCACGCNCGGGGCGGTCNTCCAYTCGTN...ANT EAAL_BXEA27
GAGTACCCGRRGCGTTYCACGNCCGGGGCGGTCGTCCATTCGTR...ACT EAGA_BX301
GAGTACNCGGGGCGTTYCACGCNCNGGGCGGTNGNCCATTCGTG...ACT EAGA_BX346
GAGTACCCGGRGCGTTYCACGCCCGGGGCGGNCNTCCATTCGTG...ACT EAGA_BX472
GAGTACNNGGGGCGTTCCACNCCCGGGGCGGTCGTCCATTCNTG...ACT EAGA_BX660
GAGTACNCGGRGCGTTCCACNNNSGGRGCGGTCGNCCATTCGTG...ACT EAGA_BXEA15_654
GWGTACCCGGRGCNTTCCACRNCCGGGGCGNTCGNCCNTTCGNG...ACT EAGA_BXEA17
GAGTACCCGGGGCGTTCCACGCCCGGGGCGGNCGNCCATTYGTG...ACT EAGA_BXEA21
NAGTACCCGGGGCGTTCCANNCNNGGGGCGGTCNYCCATTCGTG...ACT EAGA_BXEA25
GNNNASNNGNRNCNTTNNNCNNNCNNNGNGGNNNNNNNTNNNTG...ANN EAGA_BXEA29_655
GAGTACCCGGRGCGTTCCACGCCNGGGGCGGTCGNCCATTCGGN...ACT EAGA_BXEA31_659
GAGTACCCGGAGCGTTCCACGNCSGNGGCGNNCGTCNATTCGTG...ACT EAGA_BXEA32_662
GWGTACNCGNGGCGTTCCACGNNNNGGGNGGTCGTCNNTNCGTG...ACT EAGA_BXEA33_663
GAGTACCCGGRGCGTTCCACGNCSGGGGCGGTCGNCNATTCGTG...ACT EAGA_BXEA34_665
GAGTACNCGGRGCGTTCCACNNNSGGGGCGGTNGNCCANNCNTG...ACT EAGA_BXEA35_666
GWGTNCCYGGRGCNTNCCACRNCCGGGGCGNTCGNCCNTTCGNG...ACT EAGA_BXEA49_564
...
NANNNCNNGGGGCNTTNCNNNCCCGGGNCNGNCNTCCATTNNNN...ANT TTTX_BX23
```

## Data Transformation and Analysis

Once you have the genotype data, you can perform various data transformations and analyses. Here's an example of running principal component analysis (PCA) on the genotype data:

```
# Generate plots to assess the amount of missing data in alignment.
gd.missingness_reports(prefix="unfiltered")

# Does a Principal Component Analysis and makes a scatterplot.
components, pca = Plotting.run_pca(
        gd # GenotypeData instance from above.
        plot_dir="plots",
        prefix="unfiltered",
        n_components=None, # If None, then uses all components.
        center=True,
        scale=False,
        n_axes=2, # Can be 2 or 3. If 3, makes a 3D plot.
        point_size=15,
        font_size=15,
        plot_format="pdf",
        bottom_margin=0,
        top_margin=0,
        left_margin=0,
        right_margin=0,
        width=1088,
        height=700,
)
explvar = pca.explained_variance_ratio_ # Can use this to make a plot.

# Access other transformed genotype data and attributes

# 012-encoded genotypes, with ref=0, heterozygous=1, alt=2
genotypes_012 = genotype_data.genotypes_012(fmt="list") # Get 012-eencoded genotypes.

# onehot-encoded genotypes.
genotypes_onehot = genotype_data.genotypes_onehot 

# Dictionary object with all the VCF file fields.
# All values will be None if VCF file wasn't the input file type.
vcf_attributes = genotype_data.vcf_attributes 

# Access optional properties
q_matrix = genotype_data.q
site_rates = genotype_data.site_rates
tree = genotype_data.tree
```

## GenotypeData Plots

There are a number of informative plots that GenotypeData makes.

Here is a plot describing the counts of each found population:

<img src="https://github.com/btmartin721/SNPio/blob/master/plots/population_counts.png" width="50%" alt="Barplot with counts per population">

Here is a plot showing the distribution of genotypes in the alignment:

<img src="https://github.com/btmartin721/SNPio/blob/master/plots/genotype_distributions.png" width="50%" alt="Plot showing IUPAC genotype distributions">

## Alignment Filtering

The `NRemover2` class provides methods for filtering genetic alignments based on the proportion of missing data, the minor allele frequency (MAF), and monomorphic, non-biallelic, and singleton sites. It allows you to filter out sequences (samples) and loci (columns) that exceed the provided thresholds. Missing data filtering options include removing loci whose columns exceed global missing and per-population thresholds and removing samples that exceed a per-sample threshold. The class also provides informative plots related to the filtering process.

### Attributes:

- `alignment` (list of Bio.SeqRecord.SeqRecord): The input alignment to filter.
- `populations` (list of str): The population for each sequence in the alignment.
- `loci_indices` (list of int): Indices that were retained post-filtering.
- `sample_indices` (list of int): Indices that were retained post-filtering.
- `msa`: (MultipleSeqAlignment): BioPython MultipleSeqAlignment object.

### Methods:

- `nremover()`: Runs the whole NRemover2 pipeline. Includes arguments for all thresholds and settings that you'll need. You can also toggle a threshold search that plots the proportion of missing data across all the filtering options across multiple thresholds.

### Usage Example:

To illustrate how to use the `NRemover2` class, here's an example:

```
from snpio import NRemover2

# Create an instance of NRemover2
# Provide it the GenotypeData instance from above.
nrm = nremover2.NRemover2(gd)

# Run nremover to filter out missing data.
# Set the thresholds as desired.
# Returns a GenotypeData object.
gd_filtered = nrm.nremover(
    max_missing_global=0.5, # Maximum global missing data threshold.
    max_missing_pop=0.5, # Maximum per-population threshold.
    max_missing_sample=0.8, # Maximum per-sample threshold.
    singletons=True, # Filter out singletons.
    biallelic=True, # Filter out non-biallelic sites.
    monomorphic=True, # Filter out monomorphic loci.
    min_maf=0.01, # Only retain loci with a MAF above this threshold.
    search_thresholds=True, # Plots against multiple thresholds.
    plot_dir="plots", # Where to save the plots to.
)

# Makes an informative plot showing missing data proportions.
gd_filtered.missingness_reports(prefix="filtered")

# Run a PCA on the filtered data and make a scatterplot.
Plotting.run_pca(gd_filtered, prefix="filtered")
```

Running the above code makes a number of informative plots. See below.

Here is a Sankey diagram showing the number of loci removed at each filtering step.

<img src="https://github.com/btmartin721/SNPio/blob/master/plots/sankey_filtering_report.png" width="75%" alt="Sankey filtering report for loci removed at each filtering step">

Here is the proportions of missing data for the filter missingness report:

<img src="https://github.com/btmartin721/SNPio/blob/master/plots/filtered_missingness.png" width="75%" alt="Missingness filtering report plot">

Here is the PCA we ran on the filtered data, with colors being a gradient corresponding to the proportion of missing data in each sample:

<img src="https://github.com/btmartin721/SNPio/blob/master/plots/filtered_pca.png" width="50%" alt="Principal Component Analysis scatterplot for filtered data">

The below two plots show the missingness proportion variance among all the thresholds if you used set `search_thresholds=True` when you ran the `nremover()` function. The first makes plots for the missing data filters, and the second for the MAF, biallelic, monomorphic, and singleton filters. they are shown for both globally and per-population.

First, the missing data filter report:

<img src="https://github.com/btmartin721/SNPio/blob/master/plots/missingness_report.png" width="75%" alt="Plots showing missingness proportion variance for each filtering step">

And now the MAF, biallelic, singleton, and monomorphic filter report:

<img src="https://github.com/btmartin721/SNPio/blob/master/plots/maf_missingness_report.png" width="50%" alt="Plots showing missingness proportion variance among the MAF thresholds and singleton, biallelic, and monomorphic filters (toggled off and on)">

If you do not want to use some of the filtering options, just leave them at default for the ones you don't want to run.

## Writing to File and File Conversions

If you want to write your output to a file, just do use one of the write functions. Any of the input file types can be written with any of the write functions.

```
gd_filtered.write_phylip("example_data/phylip_files/nremover_test.phy")

gd_filtered.write_structure("example_data/structure_files/nremover_test.str")

gd_filtered.write_vcf("example_data/vcf_files/nmremover_test.vcf")
```

For detailed information about the available methods and attributes, refer to the API Reference.

That's it! You have successfully completed the basic steps to get started with SNPio. Explore the library further to discover more functionality and advanced features.

For detailed information about the available methods and attributes, refer to the API Reference.
