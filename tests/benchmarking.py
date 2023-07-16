import sys
import os

sys.path.append(os.path.normpath(os.getcwd()))
from snpio.read_input.genotype_data import GenotypeData

# Instantiate GenotypeData object
genotype_data = GenotypeData(
    filename="example_data/vcf_files/phylogen_subset14K.vcf.gz",
    force_popmap=True,
    filetype="auto",
    popmapfile="example_data/popmaps/phylogen_nomx.popmap",
    guidetree="example_data/trees/test.tre",
    siterates_iqtree="example_data/trees/test14K.rate",
    qmatrix_iqtree="example_data/trees/test.iqtree",
)

# Access basic properties
num_snps = genotype_data.num_snps
num_inds = genotype_data.num_inds
populations = genotype_data.populations
popmap = genotype_data.popmap
genotype_data.popmap_inverse
samples = genotype_data.samples
inputs = genotype_data.inputs
ref = genotype_data.ref
alt = genotype_data.alt
loci_indices = genotype_data.loci_indices
sample_indices = genotype_data.sample_indices
snp_data = genotype_data.snp_data

# Access other transformed data
genotypes_onehot = genotype_data.genotypes_onehot
genotypes_int = genotype_data.genotypes_int
alignment = genotype_data.alignment

# Access VCF file attributes
vcf_attributes = genotype_data.vcf_attributes

# Access additional properties
q_matrix = genotype_data.q
site_rates = genotype_data.site_rates
newick_tree = genotype_data.tree

genotype_data.plot_performance()
