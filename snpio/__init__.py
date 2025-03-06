# Description: This file is the main entry point for the snpio package. It imports all the modules and classes that are part of the package. It also defines the package version number.

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("snpio")
except PackageNotFoundError:
    __version__ = "unknown"  # Default if package is not installed


from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.analysis.tree_parser import TreeParser
from snpio.filtering.nremover2 import NRemover2
from snpio.io.phylip_reader import PhylipReader
from snpio.io.structure_reader import StructureReader
from snpio.io.vcf_reader import VCFReader
from snpio.plotting.plotting import Plotting
from snpio.popgenstats.pop_gen_statistics import PopGenStatistics
