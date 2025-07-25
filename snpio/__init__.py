# Description: This file is the main entry point for the snpio package. It imports all the modules and classes that are part of the package. It also defines the package version number.

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("snpio")
except PackageNotFoundError:
    __version__ = "unknown"  # Default if package is not installed

# Defines the public API for the package
__all__ = [
    "GenotypeEncoder",
    "TreeParser",
    "NRemover2",
    "GenePopReader",
    "PhylipReader",
    "StructureReader",
    "VCFReader",
    "Plotting",
    "PopGenStatistics",
    "SNPioMultiQC",
    "__version__",
]

from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.analysis.tree_parser import TreeParser
from snpio.filtering.nremover2 import NRemover2
from snpio.io.genepop_reader import GenePopReader
from snpio.io.phylip_reader import PhylipReader
from snpio.io.structure_reader import StructureReader
from snpio.io.vcf_reader import VCFReader
from snpio.plotting.plotting import Plotting
from snpio.popgenstats.pop_gen_statistics import PopGenStatistics
from snpio.utils.multiqc_reporter import SNPioMultiQC
