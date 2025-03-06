from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.filtering.nremover2 import NRemover2
from snpio.io.phylip_reader import PhylipReader
from snpio.io.structure_reader import StructureReader
from snpio.io.vcf_reader import VCFReader
from snpio.plotting.plotting import Plotting
from snpio.analysis.tree_parser import TreeParser
from snpio.popgenstats.pop_gen_statistics import PopGenStatistics

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # for Python versions < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("snpio")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
