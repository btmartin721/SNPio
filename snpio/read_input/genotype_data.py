import copy
import gzip
import os
import random
import re
import sys
import textwrap
import warnings
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.simplefilter(action="ignore", category=FutureWarning)

# from memory_profiler import profile

import h5py
import requests

# from memory_profiler import profile

# Make sure python version is >= 3.8
if sys.version_info < (3, 8):
    raise ImportError("Python < 3.8 is not supported!")

import numpy as np
import pandas as pd
import toytree as tt
from toytree.TreeParser import TreeParser

"""
NOTE:  Monkey patching a method in toytree because
There is a bug in the method that makes it incompatible
with Python 3.11. It tries to open a file with 'rU', which
is deprecated in Python 3.11.
"""
###############################################################
# Monkey patching begin
###############################################################
original_get_data_from_intree = TreeParser.get_data_from_intree


def patched_get_data_from_intree(self):
    """
    Load data from a file or string and return as a list of strings.
    The data contents could be one newick string; a multiline NEXUS format
    for one tree; multiple newick strings on multiple lines; or multiple
    newick strings in a multiline NEXUS format. In any case, we will read
    in the data as a list on lines.

    NOTE: This method is monkey patched from the toytree package (v2.0.5) because there is a bug that appears in
    Python 11 where it tries to open a file using 'rU'. 'rU' is is deprecated in Python 11, so I changed it to just
    ``with open(self.intree, 'r')``\. This has been fixed on the GitHub version of toytree,
    but it is not at present fixed in the pip or conda versions.
    """

    # load string: filename or data stream
    if isinstance(self.intree, (str, bytes)):
        # strip it
        self.intree = self.intree.strip()

        # is a URL: make a list by splitting a string
        if any([i in self.intree for i in ("http://", "https://")]):
            response = requests.get(self.intree)
            response.raise_for_status()
            self.data = response.text.strip().split("\n")

        # is a file: read by lines to a list
        elif os.path.exists(self.intree):
            with open(self.intree, "r") as indata:
                self.data = indata.readlines()

        # is a string: make into a list by splitting
        else:
            self.data = self.intree.split("\n")

    # load iterable: iterable of newick strings
    elif isinstance(self.intree, (list, set, tuple)):
        self.data = list(self.intree)


TreeParser.get_data_from_intree = patched_get_data_from_intree
##########################################################################
# Done monkey patching.
##########################################################################

import pysam
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pysam import VariantFile

from snpio.plotting.plotting import Plotting as Plotting
from snpio.read_input.popmap_file import ReadPopmap
from snpio.utils import sequence_tools
from snpio.utils.custom_exceptions import UnsupportedFileTypeError
from snpio.utils.misc import (
    class_performance_decorator,
    get_int_iupac_dict,
    get_onehot_dict,
)

# from cyvcf2 import VCF


# Global resource data dictionary
resource_data = {}


@class_performance_decorator(measure=False)
class GenotypeData:
    """A class for handling and analyzing genotype data.

    The GenotypeData class provides methods to read, manipulate, and analyze genotype data in various formats, including VCF, Structure, and other custom formats. It allows for data preprocessing, allele encoding, and various data transformations.

    Notes:
        GenotypeData handles the following characters as missing data:
            - 'N'
            - '-'
            - '?'
            - '.'

        Thus, it treats gaps as missing data.

        If using PHYLIP or STRUCTURE formats, all sites will also be forced to be biallelic. If you need >2 alleles, you must input a VCF file.

        Please keep these things in mind when using GenotypeData.


    Args:
        filename (str or None): Path to input file containing genotypes. Defaults to None.

        filetype (str or None): Type of input genotype file. Possible values include: 'phylip', 'structure', 'vcf', or '012'. Defaults to None.

        popmapfile (str or None): Path to population map file. If supplied and filetype is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column. Defaults to None.

        force_popmap (bool): If True, then samples not present in the popmap file will be excluded from the alignment. If False, then an error is raised if samples are present in the popmap file that are not present in the alignment. Defaults to False.

        exclude_pops (List[str] or None): List of population IDs to exclude from the alignment. Defaults to None.

        include_pops (List[str] or None): List of population IDs to include in the alignment. Populations not present in the include_pops list will be excluded. Defaults to None.

        guidetree (str or None): Path to input treefile. Defaults to None.

        qmatrix_iqtree (str or None): Path to iqtree output file containing Q rate matrix. Defaults to None.

        qmatrix (str or None): Path to file containing only Q rate matrix, and not the full iqtree file. Defaults to None.

        siterates (str or None): Path to file containing per-site rates, with 1 rate per line corresponding to 1 site. Not required if genotype_data is defined with the siterates or siterates_iqtree option. Defaults to None.

        siterates_iqtree (str or None): Path to \*.rates file output from IQ-TREE, containing a per-site rate table. Cannot be used in conjunction with siterates argument. Not required if the siterates or siterates_iqtree options were used with the GenotypeData object. Defaults to None.

        plot_format (str): Format to save report plots. Valid options include: 'pdf', 'svg', 'png', and 'jpeg'. Defaults to 'png'.

        prefix (str): Prefix to use for output directory. Defaults to "gtdata".

    Attributes:
        inputs (dict): GenotypeData keyword arguments as a dictionary.

        num_snps (int): Number of SNPs in the dataset.

        num_inds (int): Number of individuals in the dataset.

        populations (List[Union[str, int]]): Population IDs.

        popmap (dict): Dictionary object with SampleIDs as keys and popIDs as values.

        popmap_inverse (dict or None): Inverse dictionary of popmap, where popIDs are keys and lists of sampleIDs are values.

        samples (List[str]): Sample IDs in input order.

        snpsdict (dict or None): Dictionary with SampleIDs as keys and lists of genotypes as values.

        snp_data (List[List[str]]): Genotype data as a 2D list.

        genotypes_012 (List[List[int]], np.ndarray, or pd.DataFrame): Encoded 012 genotypes.

        genotypes_onehot (np.ndarray): One-hot encoded genotypes.

        genotypes_int (np.ndarray): Integer-encoded genotypes.

        alignment (Bio.MultipleSeqAlignment): Genotype data as a Biopython MultipleSeqAlignment object.

        vcf_attributes (dict): Attributes read in from VCF file.

        loci_indices (List[int]): Column indices for retained loci in filtered alignment.

        sample_indices (List[int]): Row indices for retained samples in the alignment.

        ref (List[str]): List of reference alleles of length num_snps.

        alt (List[str]): List of alternate alleles of length num_snps.

        q (QMatrix or None): Q-matrix object for phylogenetic tree.

        site_rates (SiteRates or None): Site rate data for phylogenetic tree.

        tree (NewickTree or None): Newick tree object.

    Methods:
        read_structure: Read data from a Structure file.

        read_vcf: Read data from a VCF file.

        read_phylip: Read data from a Phylip file.

        read_phylip: Read data from a Phylip file.

        read_012: Read data from a custom 012-encoded file format.

        read_tree: Read data from a newick file.

        q_from_iqtree: Read Q-matrix from \*.iqtree file.

        q_from_file: Read Q-matrix from file with only Q-matrix in it.

        siterates_from_iqtree: Read site rates from \*.rate file.

        siterates_from_file: Read site rates from file with only site rates in single column.

        write_structure: Write data to a Structure file.

        write_vcf: Write data to a VCF file.

        write_phylip: Write data to a Phylip file.

        read_popmap: Read in a popmap file.

        subset_vcf_data: Subset the data based on locus and sample indices.

        encode_012: Encode genotypes as 0,1,2 integers for reference, heterozygous, alternate alleles.

        decode_012: Decode 0,1,2 integers back to original genotypes.

        convert_onehot: Convert genotypes to one-hot encoding.

        convert_int_iupac: Convert genotypes to integer encoding (0-9) with IUPAC characters.

        missingness_reports: Create missingness reports from GenotypeData object.

    Example usage:
        Instantiate GenotypeData object

        genotype_data = GenotypeData(file="data.vcf", filetype="vcf", popmapfile="popmap.txt")

        # Access basic properties

        print(genotype_data.num_snps) # Number of SNPs in the dataset

        print(genotype_data.num_inds) # Number of individuals in the dataset

        print(genotype_data.populations) # Population IDs

        print(genotype_data.popmap) # Dictionary of SampleIDs as keys and popIDs as values
        print(genotype_data.samples) # Sample IDs in input order

        # Access transformed genotype data
        genotypes_012 = genotype_data.genotypes_012 # Encoded 012 genotypes as a 2D list

        genotypes_012_array = genotype_data.genotypes_012(fmt="numpy")

        genotypes_012_df = genotype_data.genotypes_012(fmt="pandas")

        genotypes_onehot = genotype_data.genotypes_onehot # One-hot encoded genotypes as a numpy array

        genotypes_int = genotype_data.genotypes_int # Integer-encoded genotypes (0-9) as a numpy array

        alignment = genotype_data.alignment # Genotype data as a Biopython MultipleSeqAlignment object

        # Access VCF file attributes

        vcf_attributes = genotype_data.vcf_attributes # Dictionary of VCF file attributes

        # Set and access additional properties

        genotype_data.q = q_matrix # Set q-matrix for phylogenetic tree

        q_matrix = genotype_data.q # Get q-matrix object

        genotype_data.site_rates = site_rates # Set site rate data for phylogenetic tree

        site_rates = genotype_data.site_rates # Get site rate data

        genotype_data.tree = newick_tree # Set newick tree data

        newick_tree = genotype_data.tree # Get newick tree object
    """

    resource_data = {}

    def __init__(
        self,
        filename: Optional[str] = None,
        filetype: Optional[str] = "auto",
        popmapfile: Optional[str] = None,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        guidetree: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        siterates_iqtree: Optional[str] = None,
        plot_format: Optional[str] = "png",
        prefix="snpio",
        chunk_size: int = 1000,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the GenotypeData object."""

        self.filename = filename
        self.filetype = filetype.lower()
        self.popmapfile = popmapfile
        self.force_popmap = force_popmap
        self.exclude_pops = exclude_pops
        self.include_pops = include_pops
        self.guidetree = guidetree
        self.qmatrix_iqtree = qmatrix_iqtree
        self.qmatrix = qmatrix
        self.siterates = siterates
        self.siterates_iqtree = siterates_iqtree
        self.plot_format = plot_format
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.measure = kwargs.get("measure", False)
        self.supported_filetypes = ["vcf", "phylip", "structure", "auto"]

        self._kwargs = {
            "filename": filename,
            "filetype": filetype,
            "popmapfile": popmapfile,
            "force_popmap": force_popmap,
            "exclude_pops": exclude_pops,
            "include_pops": include_pops,
            "guidetree": guidetree,
            "qmatrix_iqtree": qmatrix_iqtree,
            "qmatrix": qmatrix,
            "siterates": siterates,
            "siterates_iqtree": siterates_iqtree,
            "plot_format": plot_format,
            "prefix": prefix,
            "verbose": verbose,
        }

        if "vcf_attributes" in kwargs:
            self._vcf_attributes = kwargs["vcf_attributes"]
        else:
            self._vcf_attributes = None
        is_subset = True if "is_subset" in kwargs else False

        self._samples: List[str] = []
        self._populations: List[Union[str, int]] = []
        self._ref = []
        self._alt = []
        self._q = None
        self._site_rates = None
        self._tree = None
        self._popmap = None
        self._popmap_inverse = None
        self.vcf_header = None

        if self.qmatrix is not None and self.qmatrix_iqtree is not None:
            raise TypeError(
                "qmatrix and qmatrix_iqtree cannot both be provided."
            )

        if self.siterates is not None and self.siterates_iqtree is not None:
            raise TypeError(
                "siterates and siterates_iqtree cannot both be defined"
            )

        self._loci_indices = kwargs.get("loci_indices", None)
        self._sample_indices = kwargs.get("sample_indices", None)

        if self.filetype == "auto":
            filetype = self._detect_file_format(filename)

            if not filetype:
                raise AssertionError(
                    "File type could not be automatically detected. Please check the file for formatting errors or specify the file format as either 'phylip', 'structure', 'vcf', or '012' instead of 'auto'."
                )

        if self.filetype not in self.supported_filetypes:
            raise UnsupportedFileTypeError(
                self.filetype, supported_types=self.supported_filetypes
            )

        self._read_aln(filetype, popmapfile)

        if self.loci_indices is None:
            self._loci_indices = list(range(self.num_snps))
        if self.sample_indices is None:
            self.sample_indices = list(range(self.num_inds))

        if filetype != "vcf" and self.popmapfile is not None:
            self._my_popmap = self.read_popmap(popmapfile)

            self.subset_with_popmap(
                self._my_popmap,
                self.samples,
                force=self.force_popmap,
                include_pops=self.include_pops,
                exclude_pops=self.exclude_pops,
            )

        if self.popmapfile is not None:
            if self.verbose:
                print("Found the following populations:\nPopulation\tCount\n")
            self._my_popmap.get_pop_counts(plot_dir_prefix=self.prefix)

        self._kwargs["filetype"] = self.filetype
        self._kwargs["loci_indices"] = self.loci_indices
        self._kwargs["sample_indices"] = self.sample_indices

        vcf_attr_path = os.path.join(
            f"{self.prefix}_output",
            "gtdata",
            "alignments",
            "vcf",
            "vcf_attributes.h5",
        )
        if Path(vcf_attr_path).is_file():
            # Subset VCF attributes in case samples were not in popmap file.
            if self.filetype != "vcf" and not is_subset:
                if (
                    len(self.loci_indices) != self.num_snps
                    or len(self.sample_indices) != self.num_inds
                ):
                    self.vcf_attributes = self.subset_vcf_data(
                        self.loci_indices,
                        self.sample_indices,
                        self._vcf_attributes,
                        samples=self.samples,
                        chunk_size=self.chunk_size,
                    )

    def _detect_file_format(self, filename: str) -> str:
        """
        Check the filetype and call the appropriate function to read the file format.

        Args:
            filetype (Optional[str], default=None): Filetype. Supported values include: "phylip", "structure", "vcf", and "012".
            popmapfile (Optional[str], default=None): Path to population map file.

        Raises:
            OSError: If no filetype is specified.
            OSError: If an unsupported filetype is provided.
        """
        try:
            with open(filename, "r") as fin:
                lines = fin.readlines()
        except UnicodeDecodeError:
            with gzip.open(filename, "rt") as fin:
                lines = fin.readlines()

        lines = [line.strip() for line in lines]

        # Check for VCF format
        if (
            any(line.startswith("##fileformat=VCF") for line in lines)
            or any(line.startswith("#CHROM") for line in lines)
            or any(line.startswith("##FORMAT") for line in lines)
            or any(line.startswith("##INFO") for line in lines)
        ):
            return "vcf"

        # Check for PHYLIP format
        try:
            if len(list(map(int, lines[0].split()))) == 2:
                num_samples, num_loci = map(int, lines[0].split())

                if num_samples == len(lines[1:]):
                    line = lines[1]
                    seqs = line.split()[1]
                    if num_loci == len(list(seqs)):
                        return "phylip"
        except ValueError:
            pass

        # Check for STRUCTURE or encoded 012 format
        lines = lines[1:]

        def is_integer(n):
            try:
                int(n)
                return True
            except ValueError:
                return False

        gt = [
            line
            for line in lines
            if all(is_integer(col.strip()) for col in line.split()[3:])
        ]

        if gt:
            # Check for STRUCTURE format
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()

                if any(int(x) > 2 for x in cols[3:]):
                    return "structure"
        # Check for encoded 012 format
        elif (
            sum(x == "0" for line in lines for x in line.split())
            / sum(1 for line in lines[1:] for x in line.split())
            > 0.5
        ):
            return "012"
        return False

    def _read_aln(
        self, filetype: Optional[str] = None, popmapfile: Optional[str] = None
    ) -> None:
        """
        Check the filetype and call the appropriate function to read the file format.

        Args:
            filetype (Optional[str] = None): Filetype. Supported values include: "phylip", "structure", "vcf", and "012".
            popmapfile (Optional[str] = None): Path to population map file.

        Raises:
            OSError: If no filetype is specified.
            OSError: If an unsupported filetype is provided.
        """
        if filetype is None:
            raise OSError("No filetype specified.\n")
        else:
            if filetype.lower() == "phylip":
                self.filetype = filetype
                self.read_phylip()
            elif filetype.lower() == "structure":
                self.filetype = "structure"
                has_popids = True if popmapfile is None else False
                self.read_structure(popids=has_popids)
            elif filetype == "012":
                self.filetype = filetype
                self.read_012()
            elif filetype == "vcf":
                self.filetype = filetype
                self.read_vcf()
            elif filetype is None:
                raise TypeError(
                    "filetype argument must be provided, but got NoneType."
                )
            else:
                raise OSError(f"Unsupported filetype provided: {filetype}\n")

    def _check_filetype(self, filetype: str) -> None:
        """
        Validate that the filetype is correct.

        Args:
            filetype (str): Filetype to use.

        Raises:
            TypeError: If the provided filetype does not match the current filetype.
        """
        if self.filetype is None:
            self.filetype = filetype
        elif self.filetype == filetype:
            pass
        else:
            raise TypeError(
                "GenotypeData read_XX() call does not match filetype!\n"
            )

    def read_tree(self, treefile: str) -> tt.tree:
        """
        Read Newick-style phylogenetic tree into toytree object.

        The Newick-style tree file should follow the format type 0 (see toytree documentation).

        Args:
            treefile (str): Path to the Newick-style tree file.

        Returns:
            toytree.tree object: The input tree as a toytree object.

        Raises:
            FileNotFoundError: If the tree file is not found.
            AssertionError: If the tree file is not readable.
        """
        if not os.path.isfile(treefile):
            raise FileNotFoundError(f"File {treefile} not found!")

        assert os.access(treefile, os.R_OK), f"File {treefile} isn't readable"

        return tt.tree(treefile, tree_format=0)

    def q_from_file(self, fname: str, label: bool = True) -> pd.DataFrame:
        """
        Read Q matrix from a file.

        Args:
            fname (str): Path to the Q matrix input file.

            label (bool): True if the nucleotide label order is present, False otherwise.

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas DataFrame object.

        Raises:
            FileNotFoundError: If the Q matrix file is not found.
        """
        q = self._blank_q_matrix()

        if not label:
            print(
                "Warning: Assuming the following nucleotide order: A, C, G, T"
            )

        if not os.path.isfile(fname):
            raise FileNotFoundError(f"File {fname} not found!")

        with open(fname, "r") as fin:
            header = True
            qlines = list()
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if header:
                    if label:
                        order = line.split()
                        header = False
                    else:
                        order = ["A", "C", "G", "T"]
                    continue
                else:
                    qlines.append(line.split())
        fin.close()

        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])
        qdf = pd.DataFrame(q)
        return qdf.T

    def q_from_iqtree(self, iqfile: str) -> pd.DataFrame:
        """
        Read Q matrix from an IQ-TREE (\*.iqtree) file.

        The IQ-TREE file contains the standard output of an IQ-TREE run and includes the Q-matrix.

        Args:
            iqfile (str): Path to the IQ-TREE file (\*.iqtree).

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the IQ-TREE file could not be found.
            IOError: If the IQ-TREE file could not be read.
        """

        q = self._blank_q_matrix()
        qlines = list()
        try:
            with open(iqfile, "r") as fin:
                foundLine = False
                matlinecount = 0
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    if "Rate matrix Q" in line:
                        foundLine = True
                        continue
                    if foundLine:
                        matlinecount += 1
                        if matlinecount > 4:
                            break
                        stuff = line.split()
                        qlines.append(stuff)
                    else:
                        continue
        except (IOError, FileNotFoundError):
            raise FileNotFoundError(f"Could not open IQ-TREE file {iqfile}")

        # Check that the Q matrix was found and read
        if not qlines:
            raise IOError(f"Rate matrix Q not found in IQ-TREE file {iqfile}")

        # Populate q matrix with values from the IQ-TREE file
        order = [l[0] for l in qlines]
        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])

        qdf = pd.DataFrame(q)
        return qdf.T

    def _blank_q_matrix(
        self, default: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Create a blank Q-matrix dictionary initialized with default values.

        Args:
            default (float, optional): Default value to initialize the Q-matrix cells. Defaults to 0.0.

        Returns:
            Dict[str, Dict[str, float]]: Blank Q-matrix dictionary.

        """
        q: Dict[str, Dict[str, float]] = dict()
        for nuc1 in ["A", "C", "G", "T"]:
            q[nuc1] = dict()
            for nuc2 in ["A", "C", "G", "T"]:
                q[nuc1][nuc2] = default
        return q

    def siterates_from_iqtree(self, iqfile: str) -> pd.DataFrame:
        """
        Read site-specific substitution rates from \*.rates file.

        The \*.rates file is an optional output file generated by IQ-TREE and contains a table of site-specific rates and rate categories.

        Args:
            iqfile (str): Path to \*.rates input file.

        Returns:
            List[float]: List of site-specific substitution rates.

        Raises:
            FileNotFoundError: If the rates file could not be found.
            IOError: If the rates file could not be read from.
        """
        s = []
        try:
            with open(iqfile, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("#")
                        or line.lower().startswith("site")
                    ):
                        continue
                    else:
                        stuff = line.split()
                        s.append(float(stuff[1]))

        except (IOError, FileNotFoundError):
            raise IOError(f"Could not open iqtree file {iqfile}")
        return s

    def _validate_rates(self) -> None:
        """
        Validate the number of site rates matches the number of SNPs.

        Raises:
            ValueError: If the number of site rates does not match the number of SNPs.
        """
        if len(self._site_rates) != self.num_snps:
            raise ValueError(
                "The number of site rates was not equal to the number of snps in the alignment."
            )

    def siterates_from_file(self, fname: str) -> List[float]:
        """
        Read site-specific substitution rates from a file.

        Args:
            fname (str): Path to the input file.

        Returns:
            List[float]: List of site-specific substitution rates.
        """
        s = list()
        with open(fname, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                else:
                    s.append(float(line.split()[0]))
        fin.close()
        return s

    def _validate_seq_lengths(self) -> None:
        """Validate that all sequences have the same length.

        Raises:
            ValueError: If not all sequences (rows) are the same length.
        """
        # Make sure all sequences are the same length.
        all_same = all(len(row) == self.num_snps for row in self._snp_data)

        if not all_same:
            bad_rows = [
                i
                for i, row in enumerate(self.num_snps)
                if len(row) != self.num_snps
            ]

            bad_sampleids = [self._samples[i] for i in bad_rows]
            raise ValueError(
                f"The following sequences in the alignment were of unequal lengths: {','.join(bad_sampleids)}"
            )

    def read_structure(self, popids: bool = True) -> None:
        """
        Read a structure file and automatically detect its format.

        Args:
            popids (bool, optional): True if population IDs are present as the 2nd column in the structure file, otherwise False. Defaults to True.

        Raises:
            ValueError: If sample names do not match for the two-row format.

            ValueError: If population IDs do not match for the two-row format.

            ValueError: If not all sequences (rows) are the same length.
        """
        if self.verbose:
            print(f"\nReading structure file {self.filename}...")

        # Detect the format of the structure file
        onerow = self.detect_format()

        snp_data = list()
        with open(self.filename, "r") as fin:
            if not onerow:
                firstline = None
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    if not firstline:
                        firstline = line.split()
                        continue
                    else:
                        secondline = line.split()
                        if firstline[0] != secondline[0]:
                            raise ValueError(
                                f"Two rows per individual was "
                                f"specified but sample names do not match: "
                                f"{firstline[0]} and {secondline[0]}\n"
                            )

                        ind = firstline[0]
                        pop = None
                        if popids:
                            if firstline[1] != secondline[1]:
                                raise ValueError(
                                    f"Two rows per individual was "
                                    f"specified but population IDs do not "
                                    f"match {firstline[1]} {secondline[1]}\n"
                                )
                            pop = firstline[1]
                            self._populations.append(pop)
                            firstline = firstline[2:]
                            secondline = secondline[2:]
                        else:
                            firstline = firstline[1:]
                            secondline = secondline[1:]
                        self._samples.append(ind)
                        genotypes = merge_alleles(firstline, secondline)
                        snp_data.append(genotypes)
                        firstline = None
            else:  # If onerow:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    firstline = line.split()
                    ind = firstline[0]
                    pop = None
                    if popids:
                        pop = firstline[1]
                        self._populations.append(pop)
                        firstline = firstline[2:]
                    else:
                        firstline = firstline[1:]
                    self._samples.append(ind)
                    genotypes = merge_alleles(firstline, second=None)
                    snp_data.append(genotypes)
                    firstline = None

        snp_data = [
            list(map(self._genotype_to_iupac, row)) for row in snp_data
        ]
        self._snp_data = snp_data
        self._validate_seq_lengths()

        self._ref, self._alt, self._alt2 = self._get_ref_alt_alleles(
            self._snp_data
        )

        if self.verbose:
            print(f"STRUCTURE file successfully loaded!")
            print(
                f"\nFound {self.num_snps} SNPs and {self.num_inds} "
                f"individuals...\n"
            )

    def write_structure(
        self,
        output_file: str,
        genotype_data=None,
        snp_data: List[List[str]] = None,
        samples: List[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Write a structure file.

        Args:
            output_file (str): The output filename.

            genotype_data (GenotypeData, optional): GenotypeData instance. Uses snp_data from the provided GenotypeData object to write the file. genotype_data and snp_data cannot both be provided.

            snp_data (List[List[str]], optional): snp_data object obtained from a GenotypeData object. genotype_data and snp_data cannot both be provided.

            samples (List[str], optional): List of sample IDs.

            verbose (bool, optional): If True, status updates are printed.

        Raises:
            ValueError: If genotypes are not presented as a pair for each SNP.

            TypeError: If using snp_data, samples must also be provided.
        """
        if verbose:
            print(f"\nWriting structure file {output_file}...")

        if genotype_data is not None and snp_data is not None:
            raise TypeError(
                "genotype_data and snp_data cannot both be NoneType"
            )

        elif genotype_data is None and snp_data is None:
            snp_data = self.snp_data
            samples = self.samples
            snpsdict = self.snpsdict

        elif genotype_data is not None and snp_data is None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
            snpsdict = genotype_data.snpsdict

        elif genotype_data is None and snp_data is not None:
            if samples is None:
                raise TypeError(
                    "If using snp_data, samples must also be provided."
                )
            snpsdict = self._make_snpsdict(samples=samples, snp_data=snp_data)

        with open(output_file, "w") as fout:
            for sample in samples:
                genotypes = list(
                    map(self._iupac_to_genotype, snpsdict[sample])
                )

                genotypes = [
                    allele
                    for genotype in genotypes
                    for allele in genotype.split("/")
                ]

                # The genotypes must be presented as a pair for each SNP
                if len(genotypes) % 2 != 0:
                    raise ValueError(
                        f"Genotypes for sample {sample} are not presented as pairs."
                    )

                # Split the genotypes into two lines for each sample
                # Selects every other element of the list.
                firstline_genotypes = genotypes[::2]
                secondline_genotypes = genotypes[1::2]

                fout.write(
                    sample
                    + "\t"
                    + "\t".join(map(str, firstline_genotypes))
                    + "\n"
                )
                fout.write(
                    sample
                    + "\t"
                    + "\t".join(map(str, secondline_genotypes))
                    + "\n"
                )

        if verbose:
            print("Successfully wrote STRUCTURE file!")

    def detect_format(self) -> bool:
        """
        Detect the format of the structure file (onerow or tworow).

        Returns:
            bool: True if the file is in one-row format, False if it is in two-row format.
        """
        with open(self.filename, "r") as fin:
            first_line = fin.readline().split()
            second_line = fin.readline().split()

        # If the first two lines have the same sample name, then
        # it's a two-row format
        onerow = first_line[0] != second_line[0]
        return onerow

    def read_phylip(self) -> None:
        """
        Populates GenotypeData object by parsing a Phylip file.

        Raises:
            ValueError: If all sequences are not the same length as specified in the header line.

            ValueError: If the number of individuals differs from the header line.
        """
        if self.verbose:
            print(f"\nReading phylip file {self.filename}...")

        self._check_filetype("phylip")
        snp_data = list()
        with open(self.filename, "r") as fin:
            first = True
            for line in fin:
                line = line.strip()
                if not line:  # If blank line.
                    continue
                if first:
                    first = False
                    continue
                cols = line.split()
                inds = cols[0]
                seqs = cols[1]
                snps = [snp for snp in seqs]  # Split each site.
                snp_data.append(snps)

                self._samples.append(inds)

        self._snp_data = snp_data
        self._validate_seq_lengths()

        self._ref, self._alt, self._alt2 = self._get_ref_alt_alleles(
            self._snp_data
        )

        if self.verbose:
            print(f"PHYLIP file successfully loaded!")
            print(
                f"\nFound {self.num_snps} SNPs and {self.num_inds} "
                f"individuals...\n"
            )

    def read_vcf(self) -> None:
        """
        Read a VCF file into a GenotypeData object.

        Raises:
            ValueError: If the number of individuals differs from the header line.
        """
        if self.verbose:
            print(f"\nReading VCF file {self.filename}...")

        self._check_filetype("vcf")

        # Load the VCF file using pysam
        vcf = VariantFile(self.filename, mode="r")

        if self.popmapfile is not None:
            self._my_popmap = self.read_popmap(self.popmapfile)

        (
            self._vcf_attributes,
            self._snp_data,
            self._samples,
        ) = self.get_vcf_attributes(
            vcf,
            self.sample_indices,
            self.loci_indices,
            chunk_size=self.chunk_size,
        )

        vcf.close()

        self._validate_seq_lengths()

        if self.verbose:
            print(f"VCF file successfully loaded!")
            print(
                f"\nFound {self.num_snps} SNPs and {self.num_inds} individuals...\n"
            )

    # @profile
    def get_vcf_attributes(
        self, vcf, sample_indices=None, loci_indices=None, chunk_size=1000
    ):
        """Get VCF attributes from pysam.VariantRecord object.

        Args:
            vcf (pysam.VariantFile): pysam.VariantFile object.
            sample_indices (List[int]): Sample indices to include. If ``sample_indices`` is None, then all indices will be used. Defaults to None.
            loci_indices (List[int]): Loci indices to include. if ``loci_indices`` is None, then all indices will be used. Defaults to None.
            chunk_size (int): Variant (loci) Chunk size to load VCF file. Saves the output to an HDF5 file one chunk at a time so that only ``chunk_size`` loci are loaded into memory. Defaults to 1000.

        Returns:
            str: File path to vcf_attributes.h5 HDF5 file.
            List[List[str]]: snp_data 2D list object with IUPAC nucleotides.
            List[str]: List of sampleIDss found in alignment.
        """

        if loci_indices is None and self.loci_indices is not None:
            loci_indices = self.loci_indices

        if sample_indices is None:
            sample_indices = self.sample_indices

        samples = list((vcf.header.samples))

        sample_indices = self.subset_with_popmap(
            self._my_popmap,
            samples,
            force=self.force_popmap,
            include_pops=self.include_pops,
            exclude_pops=self.exclude_pops,
            return_indices=True,
        )

        samples = [x for x in samples if x in self.samples]

        if len(samples) != len(sample_indices):
            new_header = pysam.VariantHeader()

            for record in vcf.header.records:
                new_header.add_record(record)

            for sample in samples:
                new_header.add_sample(sample)

            vcf.subset_samples(samples)
        else:
            new_header = vcf.header

        self.vcf_header = new_header

        info_fields = list((vcf.header.info))

        format_fields = list((vcf.header.formats))
        format_fields = [x for x in format_fields if x != "GT"]

        outdir = os.path.join(
            f"{self.prefix}_output", "gtdata", "alignments", "vcf"
        )
        Path(outdir).mkdir(exist_ok=True, parents=True)
        h5_outfile = os.path.join(outdir, "vcf_attributes.h5")

        with h5py.File(h5_outfile, "w") as f:
            # Create datasets for basic attributes
            chrom_dset = f.create_dataset(
                "chrom", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )
            pos_dset = f.create_dataset(
                "pos", (0,), maxshape=(None,), dtype=int
            )
            vcf_id_dset = f.create_dataset(
                "vcf_id", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )
            ref_dset = f.create_dataset(
                "ref",
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(length=3),
            )
            alt_dset = f.create_dataset(
                "alt",
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(length=5),
            )
            qual_dset = f.create_dataset(
                "qual", (0,), maxshape=(None,), dtype=float
            )
            vcf_filter_dset = f.create_dataset(
                "filter",
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(),
            )

            format_dset = f.create_dataset(
                "format",
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(),
            )

            snp_data_dset = f.create_dataset(
                "snp_data",
                (0, len(samples)),
                maxshape=(None, len(samples)),
                dtype=h5py.string_dtype(),
            )

            # Create groups for info and calldata
            info_group = f.create_group("info")
            calldata_group = f.create_group("calldata")

            # Create datasets within info and calldata groups
            info_dsets = {
                k: info_group.create_dataset(
                    k,
                    (0,),  # 1-dimensional shape
                    maxshape=(
                        None,
                    ),  # Allow expansion along the first dimension
                    dtype=h5py.string_dtype(),
                )
                for k in info_fields
            }
            calldata_dsets = {
                k: calldata_group.create_dataset(
                    k,
                    (0, len(samples)),
                    maxshape=(None, len(samples)),
                    dtype=h5py.string_dtype(),
                )
                for k in format_fields
            }

            # Define the mapping outside the function
            IUPAC_MAPPING = {
                ("A", "A"): "A",
                ("A", "C"): "M",
                ("A", "G"): "R",
                ("A", "T"): "W",
                ("C", "C"): "C",
                ("C", "G"): "S",
                ("C", "T"): "Y",
                ("G", "G"): "G",
                ("G", "T"): "K",
                ("T", "T"): "T",
                ("N", "N"): "N",
                ("A", "N"): "A",
                ("C", "N"): "C",
                ("T", "N"): "T",
                ("G", "N"): "G",
            }

            for data_type in [
                "chrom",
                "pos",
                "ref",
                "alt",
                "qual",
                "vcf_id",
                "vcf_filter",
                "info",
                "format",
                "calldata",
                "snp_data",
            ]:
                if self.verbose:
                    print(f"\nLoading {data_type}...")

                for data in self.fetch_data(
                    vcf,
                    data_type,
                    loci_indices,
                    chunk_size,
                    info_fields,
                    IUPAC_MAPPING,
                ):
                    # Resize and write to datasets
                    if data_type == "chrom":
                        chrom_dset.resize((chrom_dset.shape[0] + len(data),))
                        chrom_dset[-len(data) :] = data
                    elif data_type == "pos":
                        pos_dset.resize((pos_dset.shape[0] + len(data),))
                        pos_dset[-len(data) :] = data
                    elif data_type == "vcf_id":
                        vcf_id_dset.resize((vcf_id_dset.shape[0] + len(data),))
                        vcf_id_dset[-len(data) :] = data
                    elif data_type == "ref":
                        ref_dset.resize((ref_dset.shape[0] + len(data),))
                        ref_dset[-len(data) :] = data
                    elif data_type == "alt":
                        # Resize and write to the alt_dset dataset
                        alt_dset.resize((alt_dset.shape[0] + len(data),))
                        alt_dset[-len(data) :] = data
                    elif data_type == "qual":
                        qual_dset.resize((qual_dset.shape[0] + len(data),))
                        qual_dset[-len(data) :] = data
                    elif data_type == "vcf_filter":
                        vcf_filter_dset.resize(
                            (vcf_filter_dset.shape[0] + len(data),)
                        )

                        try:
                            vcf_filter_dset[-len(data) :] = np.squeeze(data)
                        except TypeError:
                            vcf_filter_dset[-len(data) :] = np.full(
                                (len(data),), ".", dtype=str
                            )

                    elif data_type == "format":
                        format_dset.resize((format_dset.shape[0] + len(data),))
                        format_dset[-len(data) :] = data

                    elif data_type == "info":
                        for k, v in data.items():
                            v_str = np.array(v, dtype=str)
                            info_dsets[k].resize(
                                (info_dsets[k].shape[0] + len(v_str),)
                            )
                            info_dsets[k][-len(v_str) :] = v_str

                    elif data_type == "calldata":
                        for k, v in data.items():
                            v_str = np.array(v, dtype=str)
                            calldata_dsets[k].resize(
                                (
                                    calldata_dsets[k].shape[0] + len(v_str),
                                    len(samples),
                                )
                            )
                            calldata_dsets[k][-len(v_str) :, :] = v_str[
                                :, sample_indices
                            ]

                    elif data_type == "snp_data":
                        snp_data = np.array(data, dtype=str)
                        snp_data_dset.resize(
                            (snp_data_dset.shape[0] + len(data), len(samples))
                        )
                        snp_data_dset[-len(data) :, :] = np.array(data)[
                            :, sample_indices
                        ]
                vcf.reset()

        snp_data = None
        with h5py.File(h5_outfile, "r") as f:
            # Load the entire dataset into a NumPy array
            snp_data = f["snp_data"][:]

        snp_data = np.array(snp_data, dtype=str)

        dir_path = os.path.join(
            f"{self.prefix}_output", "gtdata", "alignments", "vcf"
        )
        Path(dir_path).mkdir(exist_ok=True, parents=True)
        file_path = os.path.join(dir_path, "vcf_attributes.h5")

        return file_path, snp_data.T.tolist(), samples

    def fetch_data(
        self,
        vcf,
        data_type,
        loci_indices,
        chunk_size,
        info_fields,
        IUPAC_MAPPING,
    ):
        def transform_gt(gt, ref, alt):
            gt_array = np.array(gt)

            try:
                alt_array = [ref] + list(
                    alt
                )  # Adding the reference at the beginning
            except TypeError:
                alt_array = [ref] + list(".")

            # Function to apply transformation for each pair
            def transform_pair(pair):
                allele1, allele2 = pair
                if allele1 is None or allele2 is None:
                    return "N"
                a1, a2 = alt_array[allele1], alt_array[allele2]
                a1, a2 = sorted(
                    [a1, a2]
                )  # Sort the alleles to ensure the correct mapping
                return IUPAC_MAPPING[(a1, a2)]

            return list(map(transform_pair, gt_array))

        # Initialize the data containers for each data type
        data_containers = {
            "chrom": [],
            "pos": [],
            "vcf_id": [],
            "ref": [],
            "alt": [],
            "qual": [],
            "vcf_filter": [],
            "format": [],
            "snp_data": [],
            "info": defaultdict(list),
            "calldata": defaultdict(list),
        }

        for i, variant in enumerate(vcf.fetch()):
            # Process only the required variants if loci_indices is provided
            if loci_indices is not None and i not in loci_indices:
                continue

            # Process the specific data type
            if data_type == "chrom":
                data_containers["chrom"].append(variant.chrom)
            elif data_type == "pos":
                data_containers["pos"].append(variant.pos)
            elif data_type == "vcf_id":
                data_containers["vcf_id"].append(
                    "." if variant.id is None else variant.id
                )
            elif data_type == "ref":
                data_containers["ref"].append(variant.ref)
            elif data_type == "alt":
                if variant.alts is None:
                    data_containers["alt"].append(".")
                else:
                    data_containers["alt"].append(",".join(list(variant.alts)))
            elif data_type == "qual":
                data_containers["qual"].append(variant.qual)
            elif data_type == "vcf_filter":
                data_containers["vcf_filter"].append(variant.filter)
            elif data_type == "format":
                data_containers["format"].append(
                    ":".join(list(variant.format.keys()))
                )
            elif data_type == "snp_data":
                gt = [
                    variant.samples[sample].get("GT", "./.")
                    for sample in variant.samples
                ]

                snp_data = transform_gt(gt, variant.ref, variant.alts)
                data_containers["snp_data"].append(snp_data)

            elif data_type == "info":
                for k in info_fields:
                    value = variant.info.get(k, ".")
                    processed_value = (
                        ",".join(list(value))
                        if isinstance(value, tuple)
                        else value
                    )
                    data_containers["info"][k].append(processed_value)

            elif data_type == "calldata":
                for field in variant.format.keys():
                    if field != "GT":
                        key = field
                        value = [
                            ",".join(
                                list(variant.samples[sample].get(field, (".")))
                            )
                            if isinstance(
                                variant.samples[sample].get(field), tuple
                            )
                            else variant.samples[sample].get(field, ".")
                            for sample in variant.samples
                        ]
                        data_containers["calldata"][key].append(value)

            # If reached chunk size, yield and reset current chunk
            if (i + 1) % chunk_size == 0:
                yield data_containers[data_type]

                if data_type in ["calldata", "info"]:
                    data_containers[data_type] = defaultdict(list)
                else:
                    data_containers[data_type] = []

        # Yield remaining data if any
        if data_containers[data_type]:
            yield data_containers[data_type]
            if data_type in ["calldata", "info"]:
                data_containers[data_type] = defaultdict(list)
            else:
                data_containers[data_type] = []

    def _get_ref_alt_alleles(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the most common and second most common alleles in each column of a 2D numpy array.

        Args:
            data (np.ndarray): A 2D numpy array where each column represents different data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of three numpy arrays. The first array contains the most common alleles in each column. The second array contains the second most common alleles in each column, or None if a column doesn't have a second most common allele. The third array contains the less common alleles in each column, or None if a column doesn't have less common alleles.
        """

        iupac_codes = {
            "A": ("A", "A"),
            "C": ("C", "C"),
            "G": ("G", "G"),
            "T": ("T", "T"),
            "R": ("A", "G"),
            "Y": ("C", "T"),
            "S": ("G", "C"),
            "W": ("A", "T"),
            "K": ("G", "T"),
            "M": ("A", "C"),
        }

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        most_common_alleles = []
        second_most_common_alleles = []
        less_common_alleles_list = []

        for column in data.T:
            alleles = []
            for genotype in column:
                if genotype not in ["N", "-", "?"]:
                    alleles.extend(
                        iupac_codes.get(genotype, (genotype, genotype))
                    )
                elif genotype in ["A", "C", "G", "T"]:
                    alleles.extend([genotype, genotype])

            allele_counts = Counter(alleles)
            most_common = allele_counts.most_common(1)
            most_common_allele = most_common[0][0] if most_common else None

            if most_common_allele in ["N", "-", "?"]:
                sorted_counts = sorted(
                    allele_counts.items(), key=lambda x: x[1], reverse=True
                )
                for allele, count in sorted_counts:
                    if allele not in ["N", "-", "?"]:
                        most_common_allele = allele
                        break

            most_common_alleles.append(most_common_allele)

            # Exclude the most common allele for the subsequent calculations
            if most_common_allele:
                del allele_counts[most_common_allele]

            sorted_counts = sorted(
                allele_counts.items(), key=lambda x: x[1], reverse=True
            )

            second_most_common_allele = (
                sorted_counts[0][0] if sorted_counts else None
            )
            second_most_common_alleles.append(second_most_common_allele)

            less_common_alleles = [
                allele for allele, count in sorted_counts[1:]
            ]
            less_common_alleles_list.append(
                less_common_alleles if less_common_alleles else None
            )

        return (
            most_common_alleles,
            second_most_common_alleles,
            less_common_alleles_list,
        )

    def _snpdata2gtarray(self, snpdata):
        iupac_codes = {
            "M": ("A", "C"),
            "R": ("A", "G"),
            "W": ("A", "T"),
            "S": ("C", "G"),
            "Y": ("C", "T"),
            "K": ("G", "T"),
            "A": ("A", "A"),
            "T": ("T", "T"),
            "G": ("G", "G"),
            "C": ("C", "C"),
            "N": ("N", "N"),
            "-": ("N", "N"),
            "?": ("N", "N"),
        }

        # Convert the 2D list to a numpy array and transpose it
        snpdata_array = np.array(snpdata).T

        # Use vectorization to map the IUPAC codes to pairs of nucleotides
        snpdata_tuples = np.vectorize(iupac_codes.get)(snpdata_array)

        # Convert the tuple of arrays into a 3D array
        snpdata_tuples = np.stack(snpdata_tuples, axis=-1)

        return snpdata_tuples

    def write_phylip(
        self,
        output_file: str,
        genotype_data=None,
        snp_data: List[List[str]] = None,
        samples: List[str] = None,
        verbose: bool = False,
    ):
        """
        Write the alignment as a PHYLIP file.

        Args:
            output_file (str): Name of the output phylip file.

            genotype_data (GenotypeData, optional): GenotypeData instance. Uses snp_data from the provided GenotypeData object to write the file. genotype_data and snp_data cannot both be provided.

            snp_data (List[List[str]], optional): snp_data object obtained from a GenotypeData object. genotype_data and snp_data cannot both be provided. If snp_data is not None, then samples must also be provided.

            samples (List[str], optional): List of sample IDs. Must be provided if snp_data is not None.

            verbose (bool, optional): If True, status updates are printed.

        Raises:
            TypeError: If genotype_data and snp_data are both provided.
            TypeError: If samples is not provided when snp_data is provided.
            ValueError: If samples and snp_data are not the same length.
        """
        if verbose:
            print(f"\nWriting to PHYLIP file {output_file}...")

        if genotype_data is not None and snp_data is not None:
            raise TypeError(
                "genotype_data and snp_data cannot both be NoneType"
            )
        elif genotype_data is None and snp_data is None:
            snp_data = self.snp_data
            samples = self.samples
        elif genotype_data is not None and snp_data is None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
        elif genotype_data is None and snp_data is not None:
            if samples is None:
                raise TypeError("samples must be provided if snp_data is None")

        if len(samples) != len(snp_data):
            raise ValueError(
                f"samples and snp_data are not the same length: {len(samples)}, {len(snp_data)}"
            )

        with open(output_file, "w") as f:
            aln = pd.DataFrame(snp_data)
            n_samples, n_loci = aln.shape
            f.write(f"{n_samples} {n_loci}\n")
            for sample, sample_data in zip(samples, snp_data):
                genotype_data = "".join(str(x) for x in sample_data)
                f.write(f"{sample}\t{genotype_data}\n")

        if verbose:
            print(f"Successfully wrote PHYLIP file!")

    def calculate_ns(self, snp_data):
        ns = [
            sum(1 for nucleotide in site if nucleotide != "N")
            for site in zip(*snp_data)
        ]
        return ns

    def calculate_af(self, snp_data, alternate_alleles):
        # IUPAC ambiguity characters mapping to pairs of nucleotides
        iupac_mapping = {
            "R": ("A", "G"),
            "Y": ("C", "T"),
            "S": ("G", "C"),
            "W": ("A", "T"),
            "K": ("G", "T"),
            "M": ("A", "C"),
        }

        af = []
        # Looping through sites and corresponding alternate alleles simultaneously
        for site, alt_allele in zip(zip(*snp_data), alternate_alleles):
            count = Counter()
            for nucleotide in site:
                if nucleotide in iupac_mapping:
                    count[iupac_mapping[nucleotide][0]] += 0.5
                    count[iupac_mapping[nucleotide][1]] += 0.5
                else:
                    count[nucleotide] += 1

            # Removing missing data ("N") from the count
            count.pop("N", None)

            try:
                # Calculating the frequency of the specified alternate allele
                frequency = count[alt_allele] / sum(count.values())
            except ZeroDivisionError:
                frequency = 0.0
            af.append(frequency)
        return af

    def calculate_allele_counts(self, snp_data):
        # IUPAC ambiguity characters mapping to pairs of nucleotides
        iupac_mapping = {
            "R": ("A", "G"),
            "Y": ("C", "T"),
            "S": ("G", "C"),
            "W": ("A", "T"),
            "K": ("G", "T"),
            "M": ("A", "C"),
        }

        result = []
        # Looping through sites
        for site in zip(*snp_data):
            count = Counter()
            for nucleotide in site:
                if nucleotide in iupac_mapping:
                    # Incrementing count by 1 for both involved alleles in the case of heterozygous characters
                    count[iupac_mapping[nucleotide][0]] += 1
                    count[iupac_mapping[nucleotide][1]] += 1
                elif nucleotide != "N":  # Ignoring missing values
                    count[
                        nucleotide
                    ] += 2  # Increasing count by 2 for non-heterozygous nucleotides

            # Formatting the counts as required
            formatted_count = ",".join(
                str(count[x]) for x in ["C", "A", "T", "G"]
            )
            result.append(formatted_count)

        return result

    def write_vcf(
        self,
        output_filename: str,
        hdf5_file_path: str = None,
        chunk_size=1000,
    ) -> None:
        """
        Writes the GenotypeData object data to a VCF file.

        Args:
            output_filename (str): The name of the VCF file to write to.
            hdf5_file_path (str, optional): The path to the HDF5 file containing VCF attributes. If None, then uses the vcf_attributes property to find the file. Defaults to None.
            chunk_size (int, optional): Chunk size to process the data lines. This reduces memory consumption. You can set it higher if computation is too slow. Defaults to 1000.
        """

        if self.verbose:
            print("\nWriting vcf file...")

        if self.vcf_attributes is None:
            ns = self.calculate_ns(self.snp_data)
            af = self.calculate_af(self.snp_data, self.alt)

            vcf_attributes = {
                "chrom": [f"locus_{i}" for i in range(self.num_snps)],
                "pos": ["1" for x in range(self.num_snps)],
                "id": ["." for x in range(self.num_snps)],
                "ref": [x if x is not None else "N" for x in self.ref],
                "alt": ["." if x is None else x for x in self.alt],
                "qual": ["." for x in range(self.num_snps)],
                "filter": ["." for x in range(self.num_snps)],
                "info": {
                    "NS": [f"NS={v}" for v in ns],
                    "MAF": [f"AF={round(x, 3)}" for x in af],
                },
                "format": ["GT" for x in range(self.num_snps)],
            }

            vcf_attributes["alt"] = [
                [x] + y if y else [x]
                for x, y in zip(vcf_attributes["alt"], self._alt2)
            ]
            vcf_attributes["alt"] = [
                ",".join(x) for x in vcf_attributes["alt"]
            ]

            # IUPAC ambiguity codes mapping
            iupac_mapping = {
                "R": "0/1",
                "Y": "0/1",
                "S": "0/1",
                "W": "0/1",
                "K": "0/1",
                "M": "0/1",
                "B": "0/1",
                "D": "0/1",
                "H": "0/1",
                "V": "0/1",
            }

            def replace_alleles(row, ref, alt):
                for i in range(9, len(row)):
                    # Replace the reference allele with "0/0"
                    row[i] = row[i].replace(ref, "0/0")
                    # Replace any alternate allele with "1/1"
                    for a in alt:
                        row[i] = row[i].replace(a, "1/1")
                    # Replace IUPAC ambiguity codes with "0/1"
                    for iupac, replacement in iupac_mapping.items():
                        row[i] = row[i].replace(iupac, replacement)
                    row[i] = row[i].replace("N", "./.")
                    row[i] = row[i].replace("-", "./.")
                    row[i] = row[i].replace("?", "./.")
                return "\t".join(row) + "\n"

            with open(output_filename, "w") as fout:
                sample_header = "\t".join(self.samples)

                vcf_header = textwrap.dedent(
                    f"""\
                    ##fileformat=VCFv4.0
                    ##fileDate={datetime.now().date()}
                    ##source=SNPio
                    ##phasing=unphased
                    ##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">
                    ##INFO=<ID=VAF,Number=A,Type=Float,Description="Variant Allele Frequency">
                    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
                    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_header}\n"""
                )
                gt = np.array(self.snp_data, dtype=str).T.tolist()
                gt_joined = ["\t".join(list(map(str, x))) for x in gt]

                info = {k: f"{k}={str(v)}" for k, v in vcf_attributes.items()}
                info_joined = [
                    ";".join(list(map(str, values)))
                    for values in zip(*vcf_attributes["info"].values())
                ]
                vcf_attributes["info"] = info_joined

                vcf_attributes["calldata"] = gt_joined

                lines_data = []
                for i in range(self.num_snps):
                    line = [
                        vcf_attributes["chrom"][i],
                        vcf_attributes["pos"][i],
                        vcf_attributes["id"][i],
                        vcf_attributes["ref"][i],
                        vcf_attributes["alt"][i],
                        vcf_attributes["qual"][i],
                        vcf_attributes["filter"][i],
                        vcf_attributes["info"][i],
                        vcf_attributes["format"][i],
                        vcf_attributes["calldata"][i],
                    ]
                    lines_data.append(list(map(str, line)))

                new_lines = [
                    replace_alleles(row, ref, alt)
                    for row, ref, alt in zip(
                        lines_data,
                        vcf_attributes["ref"],
                        vcf_attributes["alt"],
                    )
                ]

                # new_lines = align_columns(new_lines, alignment="left")

                fout.write(vcf_header)
                for line in new_lines:
                    fout.write(line)

            if self.verbose:
                print("\nSuccessfully wrote VCF file!\n")

            return None

        if hdf5_file_path is None:
            hdf5_file_path = self.vcf_attributes

        def replace_alleles(row, ref, alt):
            for i in range(9, len(row)):
                # Replace the reference allele with "0"
                row[i] = row[i].replace(ref, "0")
                # Replace any alternate allele with "1"
                for a in alt:
                    row[i] = row[i].replace(a, "1")
            return ["\t".join(str(x)) + "\n" for x in row]

        # 1. Opening the HDF5 File and VCF File
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            vcf_header = self.vcf_header
            with open(output_filename, "w") as f:
                for header_record in vcf_header.records:
                    f.write(str(header_record))
                sample_header = "\t".join(self.samples)
                f.write(
                    textwrap.dedent(
                        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_header}\n"
                    )
                )

                # 2. Reading Attributes in Chunks
                for start in range(0, len(hdf5_file["chrom"]), chunk_size):
                    end = min(start + chunk_size, len(hdf5_file["chrom"]))

                    # Read the chunk from the HDF5 file
                    chrom = hdf5_file["chrom"][start:end]
                    pos = hdf5_file["pos"][start:end]
                    vcf_id = hdf5_file["vcf_id"][start:end]
                    ref = hdf5_file["ref"][start:end]
                    alt = hdf5_file["alt"][start:end]
                    qual = hdf5_file["qual"][start:end]
                    fltr = hdf5_file["filter"][start:end]
                    info_keys = list(hdf5_file["info"].keys())
                    info = defaultdict(list)
                    for k in info_keys:
                        info[k] = hdf5_file[f"info/{k}"][start:end]

                    fmt = hdf5_file["format"][start:end]

                    calldata_keys = list(hdf5_file["calldata"].keys())

                    calldata = defaultdict(list)
                    for k in calldata_keys:
                        calldata[k] = hdf5_file[f"calldata/{k}"][start:end, :]

                    fmt_keys = list(set(fmt))

                    if len(fmt_keys) > 1:
                        raise ValueError(
                            "There was a discrepancy in the FORMAT keys."
                        )

                    fmt_keys = str(fmt_keys[0])
                    fmt_keys = fmt_keys.strip().split(":")

                    # Function to process keys (you can modify this based on the specific format)
                    def process_key(key):
                        return key.strip("b'")

                    # Process fmt_keys
                    fmt_keys = [process_key(k) for k in fmt_keys]

                    # Extracting fmt_keys and removing "GT" if present
                    fmt_keys = [str(k) for k in fmt_keys if k != "GT"]

                    # Extract the values corresponding to the order in fmt_keys
                    calldata_list = [calldata[k] for k in fmt_keys]

                    # Transpose the list of lists
                    calldata_transposed = list(map(list, zip(*calldata_list)))

                    # Join the corresponding elements of the lists
                    calldata_str = [
                        [
                            ":".join(
                                [
                                    e.decode()
                                    if isinstance(e, bytes)
                                    else str(e)
                                    for e in row
                                ]
                            )
                            for row in zip(*rows)
                        ]
                        for rows in calldata_transposed
                    ]

                    # Convert to a NumPy array
                    calldata_str_array = np.array(calldata_str, dtype=str)

                    # 3. Processing the Chunk
                    # Convert the data into the required format
                    chrom = chrom.astype(str)
                    vcf_id = vcf_id.astype(str)
                    ref = ref.astype(str)
                    alt_str = alt.astype(str)
                    pos_str = pos.astype(str)
                    qual_str = qual.astype(str)
                    fltr_str = fltr.astype(str)
                    fmt_str = fmt.astype(str)

                    snp_data = np.array(self.snp_data, dtype=str)[:, start:end]

                    # Create the genotype string
                    gt = self._snpdata2gtarray(snp_data)
                    gt_0 = gt[:, :, 0].astype(str)
                    gt_1 = gt[:, :, 1].astype(str)
                    gt_joined = np.char.add(np.char.add(gt_0, "/"), gt_1)
                    gt_joined[gt_joined == "N/N"] = "./."
                    gt_joined = np.char.add(gt_joined, ":")
                    gt_joined = gt_joined.astype(str)
                    gt_joined = np.char.add(gt_joined, calldata_str_array)

                    info_arrays = {
                        key: np.char.add(f"{key}=", value.astype(str))
                        for key, value in info.items()
                    }

                    info_arrays = np.array(
                        list(info_arrays.values()), dtype=str
                    )

                    # Join the elements along the last axis
                    info_result = np.apply_along_axis(
                        lambda x: ";".join(x), 0, info_arrays
                    )

                    # Concatenate the data into lines
                    lines_data = np.stack(
                        (
                            chrom,
                            pos_str,
                            vcf_id,
                            ref,
                            alt_str,
                            qual_str,
                            fltr_str,
                            info_result,
                            fmt_str,
                        ),
                        axis=-1,
                    )
                    lines = np.hstack((lines_data, gt_joined))

                    # Replace alleles with numerical values
                    ref_alleles = lines[:, 3]
                    alt_alleles = [
                        alt_str.strip().split(",") for alt_str in lines[:, 4]
                    ]
                    new_lines = [
                        replace_alleles(row, ref, alt)
                        for row, ref, alt in zip(
                            lines, ref_alleles, alt_alleles
                        )
                    ]
                    new_lines = ["\t".join(x) + "\n" for x in lines]

                    # 4. Writing the Chunk to the VCF File
                    # Write the processed lines for this chunk to the VCF file
                    f.writelines(new_lines)

        if self.verbose:
            print("\nSuccessfully wrote VCF file!\n")

    def read_012(self) -> None:
        """
        Read 012-encoded comma-delimited file.

        Raises:
            ValueError: Sequences differ in length.
        """
        if self.verbose:
            print(f"\nReading 012-encoded file {self.filename}...")

        self._check_filetype("012")
        snp_data = list()
        num_snps = list()

        with open(self.filename, "r") as fin:
            num_inds = 0
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                cols = line.split(",")
                inds = cols[0]
                snps = cols[1:]
                num_snps.append(len(snps))
                num_inds += 1
                snp_data.append(snps)
                self._samples.append(inds)

        if len(list(set(num_snps))) > 1:
            raise ValueError(
                "All sequences must be the same length; "
                "at least one sequence differs in length from the others\n"
            )

        df = pd.DataFrame(snp_data)
        df.replace("NA", "-9", inplace=True)
        df = df.astype("int")

        # Decodes 012 and converts to self._snp_data List[List[str]]
        self.genotypes_012 = df

        self._ref = None
        self._alt = None

        if self.verbose:
            print(f"012 file successfully loaded!")
            print(
                f"\nFound {self.num_snps} SNPs and {self.num_inds} "
                f"individuals...\n"
            )

    def convert_012(
        self,
        snps: List[List[str]],
        vcf: bool = False,
        impute_mode: bool = False,
    ) -> List[List[int]]:
        """
        Encode IUPAC nucleotides as 0 (reference), 1 (heterozygous), and 2 (alternate) alleles.

        Args:
            snps (List[List[str]]): 2D list of genotypes of shape (n_samples, n_sites).

            vcf (bool, optional): Whether or not VCF file input is provided. Not yet supported. Defaults to False.

            impute_mode (bool, optional): Whether or not ``convert_012()`` is called in impute mode. If True, then returns the 012-encoded genotypes and does not set the ``self.snp_data`` property. If False, it does the opposite. Defaults to False.

        Returns:
            List[List[int]], optional: 012-encoded genotypes as a 2D list of shape (n_samples, n_sites). Only returns value if ``impute_mode`` is True.

            List[int], optional: List of integers indicating bi-allelic site indexes.

            int, optional: Number of remaining valid sites.

        Warnings:
            UserWarning: If site is monomorphic.
            UserWarning: If site has >2 alleles.

        Todo:
            skip and impute_mode are now deprecated.
        """
        warnings.formatwarning = self._format_warning

        skip = 0
        snps_012 = []
        new_snps = []
        monomorphic_sites = []
        non_biallelic_sites = []
        all_missing = []

        if impute_mode:
            imp_snps = list()

        for i in range(0, len(snps)):
            new_snps.append([])

        # TODO: valid_sites is now deprecated.
        valid_sites = np.ones(len(snps[0]))
        for j in range(0, len(snps[0])):
            loc = []
            for i in range(0, len(snps)):
                if vcf:
                    loc.append(snps[i][j])
                else:
                    loc.append(snps[i][j].upper())

            if all(x == "N" for x in loc):
                all_missing.append(j)
                continue
            num_alleles = sequence_tools.count_alleles(loc, vcf=vcf)
            if num_alleles != 2:
                # If monomorphic
                if num_alleles < 2:
                    monomorphic_sites.append(j)
                    try:
                        ref = list(
                            map(
                                sequence_tools.get_major_allele,
                                loc,
                                [vcf for x in loc],
                            )
                        )
                        ref = str(ref[0])
                    except IndexError:
                        ref = list(
                            map(
                                sequence_tools.get_major_allele,
                                loc,
                                [vcf for x in loc],
                            )
                        )
                    alt = None

                    if vcf:
                        for i in range(0, len(snps)):
                            gen = snps[i][j].split("/")
                            if gen[0] in ["-", "-9", "N"] or gen[1] in [
                                "-",
                                "-9",
                                "N",
                            ]:
                                new_snps[i].append(-9)

                            elif gen[0] == gen[1] and gen[0] == ref:
                                new_snps[i].append(0)

                            else:
                                new_snps[i].append(1)
                    else:
                        for i in range(0, len(snps)):
                            if loc[i] in ["-", "-9", "N"]:
                                new_snps[i].append(-9)

                            elif loc[i] == ref:
                                new_snps[i].append(0)

                            else:
                                new_snps[i].append(1)

                # If >2 alleles
                elif num_alleles > 2:
                    non_biallelic_sites.append(j)
                    all_alleles = sequence_tools.get_major_allele(loc, vcf=vcf)
                    all_alleles = [str(x[0]) for x in all_alleles]
                    ref = all_alleles.pop(0)
                    alt = all_alleles.pop(0)
                    others = all_alleles

                    if vcf:
                        for i in range(0, len(snps)):
                            gen = snps[i][j].split("/")
                            if gen[0] in ["-", "-9", "N"] or gen[1] in [
                                "-",
                                "-9",
                                "N",
                            ]:
                                new_snps[i].append(-9)

                            elif gen[0] == gen[1] and gen[0] == ref:
                                new_snps[i].append(0)

                            elif gen[0] == gen[1] and gen[0] == alt:
                                new_snps[i].append(2)

                            # Force biallelic
                            elif gen[0] == gen[1] and gen[0] in others:
                                new_snps[i].append(2)

                            else:
                                new_snps[i].append(1)
                    else:
                        for i in range(0, len(snps)):
                            if loc[i] in ["-", "-9", "N"]:
                                new_snps[i].append(-9)

                            elif loc[i] == ref:
                                new_snps[i].append(0)

                            elif loc[i] == alt:
                                new_snps[i].append(2)

                            # Force biallelic
                            elif loc[i] in others:
                                new_snps[i].append(2)

                            else:
                                new_snps[i].append(1)

            else:
                ref, alt = sequence_tools.get_major_allele(loc, vcf=vcf)
                ref = str(ref)
                alt = str(alt)

                if vcf:
                    for i in range(0, len(snps)):
                        gen = snps[i][j].split("/")
                        if gen[0] in ["-", "-9", "N"] or gen[1] in [
                            "-",
                            "-9",
                            "N",
                        ]:
                            new_snps[i].append(-9)

                        elif gen[0] == gen[1] and gen[0] == ref:
                            new_snps[i].append(0)

                        elif gen[0] == gen[1] and gen[0] == alt:
                            new_snps[i].append(2)

                        else:
                            new_snps[i].append(1)
                else:
                    for i in range(0, len(snps)):
                        if loc[i] in ["-", "-9", "N"]:
                            new_snps[i].append(-9)

                        elif loc[i] == ref:
                            new_snps[i].append(0)

                        elif loc[i] == alt:
                            new_snps[i].append(2)

                        else:
                            new_snps[i].append(1)

        outdir = os.path.join(f"{self.prefix}_output", "gtdata", "logs")
        Path(outdir).mkdir(exist_ok=True, parents=True)
        if monomorphic_sites:
            # TODO: Check here if column is all missing. What to do in this
            # case? Error out?
            fname = "monomorphic_sites.txt"
            outfile = os.path.join(outdir, fname)
            with open(outfile, "w") as fout:
                fout.write(",".join([str(x) for x in monomorphic_sites]))

            warnings.warn(
                f"\nMonomorphic sites detected. You can check the locus indices in the following log file: {outfile}\n"
            )

        if non_biallelic_sites:
            fname = "non_biallelic_sites.txt"
            outfile = os.path.join(outdir, fname)
            with open(outfile, "w") as fout:
                fout.write(",".join([str(x) for x in non_biallelic_sites]))

            warnings.warn(
                f"\nSNP column indices listed in the log file {outfile} had >2 "
                f"alleles and was forced to "
                f"be bi-allelic. If that is not what you want, please "
                f"fix or remove the column and re-run.\n"
            )

        if all_missing:
            fname = "all_missing.txt"
            outfile = os.path.join(outdir, fname)
            with open(outfile, "w") as fout:
                ",".join([str(x) for x in all_missing])

            warnings.warn(
                f" SNP column indices found in the log file {outfile} had all missing data and were excluded from the alignment.\n"
            )

        # TODO: skip and impute_mode are now deprecated.
        if skip > 0:
            if impute_mode:
                print(
                    f"\nWarning: Skipping {skip} non-biallelic sites following "
                    "imputation\n"
                )
            else:
                print(f"\nWarning: Skipping {skip} non-biallelic sites\n")

        for s in new_snps:
            if impute_mode:
                imp_snps.append(s)
            else:
                snps_012.append(s)

        if impute_mode:
            return (
                imp_snps,
                valid_sites,
                np.count_nonzero(~np.isnan(valid_sites)),
            )
        else:
            return snps_012

    def _convert_alleles(
        self,
        data: np.ndarray,
        ref_alleles: np.ndarray,
        alt_alleles: np.ndarray,
    ) -> np.ndarray:
        """
        Replaces the values of a 3D numpy array according to provided conditions.

        This method replaces the values in the array where the values "A", "G", "T", "C" equal to the ref allele are set to "0", the values "A", "T", "G", "C" that are equal to the alt allele should be "1", and the values equal to "N" are set to "N".

        Args:
            data (np.ndarray): A 3D numpy array holding the original data of shape (n_loci, n_samples, 2).

            ref_alleles (np.ndarray): A numpy array holding the reference alleles for each column in ``data``\.

            alt_alleles (np.ndarray): A numpy array holding the alternate alleles for each column in ``data``\.

        Returns:
            np.ndarray: A new 3D numpy array with values replaced according to the provided conditions.
        """

        # Create a new array to hold the output, initially filled with 'N'
        new_data = np.full_like(data, "N")

        # Reshape ref_alleles and alt_alleles for broadcasting
        ref_alleles = ref_alleles[:, None, None]
        alt_alleles = alt_alleles[:, None, None]

        # Set locations matching the ref allele to '0'
        new_data[data == ref_alleles] = "0"

        # Set locations matching the alt allele to '1'
        new_data[data == alt_alleles] = "1"

        return new_data

    def _make_snpsdict(
        self, samples: List[str] = None, snp_data: List[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Make a dictionary with SampleIDs as keys and a list of SNPs associated with the sample as the values.

        Args:
            samples (List[str], optional): List of sample IDs. If not provided, uses self.samples.

            snp_data (List[List[str]], optional): 2D list of genotypes. If not provided, uses self.snp_data.

        Returns:
            Dict[str, List[str]]: Dictionary with sample IDs as keys and a list of SNPs as values.
        """
        if samples is None:
            samples = self.samples
        if snp_data is None:
            snp_data = self.snp_data

        snpsdict = {}
        for ind, seq in zip(samples, snp_data):
            snpsdict[ind] = seq
        return snpsdict

    def _format_warning(
        self, message, category, filename, lineno, file=None, line=None
    ) -> str:
        """
        Set the format of warnings.warn warnings.

        This method defines the format of warning messages printed by the warnings module when using `warnings.warn()`.

        Args:
            message (str): Warning message to be printed.

            category (str): Type of warning.

            filename (str): Name of the Python file where the warning was raised.

            lineno (str): Line number where the warning occurred.

        Returns:
            str: Formatted warning message.

        Note:
            To set the format of warnings, use `warnings.formatwarning = self._format_warning`.
        """
        return f"{filename}:{lineno}: {category.__name__}:{message}"

    def convert_onehot(
        self,
        snp_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert input data to one-hot encoded format.

        Args:
            snp_data (Union[np.ndarray, List[List[int]]]): Input 012-encoded data of shape (n_samples, n_SNPs).

            encodings_dict (Optional[Dict[str, int]]): Encodings to convert structure to phylip format. Defaults to None.

        Returns:
            np.ndarray: One-hot encoded data.

        Note:
            If the data file type is "phylip" and `encodings_dict` is not provided, default encodings for nucleotides are used.

            If the data file type is "structure1row" or "structure2row" and `encodings_dict` is not provided, default encodings for alleles are used.

            Otherwise, if `encodings_dict` is provided, it will be used for conversion.

        Warnings:
            If the data file type is "phylip" or "structure" and ``encodings_dict`` is not provided, a default encoding will be used. It is recommended to provide custom encodings for accurate conversion.
        """

        if encodings_dict is None:
            onehot_dict = get_onehot_dict()
        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.tolist()
            onehot_dict = encodings_dict
        onehot_outer_list = list()

        n_rows = len(self.samples) if encodings_dict is None else len(snp_data)

        for i in range(n_rows):
            onehot_list = list()
            for j in range(len(snp_data[0])):
                onehot_list.append(onehot_dict[snp_data[i][j]])
            onehot_outer_list.append(onehot_list)

        return np.array(onehot_outer_list)

    def inverse_onehot(
        self,
        onehot_data: Union[np.ndarray, List[List[float]]],
        encodings_dict: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        Convert one-hot encoded data back to original format.
        Args:
            onehot_data (Union[np.ndarray, List[List[float]]]): Input one-hot encoded data of shape (n_samples, n_SNPs).
            encodings_dict (Optional[Dict[str, List[float]]]): Encodings to convert from one-hot encoding to original format. Defaults to None.
        Returns:
            np.ndarray: Original format data.
        """

        onehot_dict = (
            get_onehot_dict() if encodings_dict is None else encodings_dict
        )

        # Create inverse dictionary (from list to key)
        inverse_onehot_dict = {tuple(v): k for k, v in onehot_dict.items()}

        if isinstance(onehot_data, np.ndarray):
            onehot_data = onehot_data.tolist()

        decoded_outer_list = []

        for i in range(len(onehot_data)):
            decoded_list = []
            for j in range(len(onehot_data[0])):
                # Look up original key using one-hot encoded list
                decoded_list.append(
                    inverse_onehot_dict[tuple(onehot_data[i][j])]
                )
            decoded_outer_list.append(decoded_list)

        return np.array(decoded_outer_list)

    def convert_int_iupac(
        self,
        snp_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert input data to integer-encoded format (0-9) based on IUPAC codes.

        Args:
            snp_data (numpy.ndarray of shape (n_samples, n_SNPs) or List[List[int]]): Input 012-encoded data.
            encodings_dict (Dict[str, int] or None): Encodings to convert structure to phylip format.

        Returns:
            numpy.ndarray: Integer-encoded data.

        Note:
            If the data file type is "phylip" or "vcf" and `encodings_dict` is not provided, default encodings based on IUPAC codes are used.

            If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.

            Otherwise, if `encodings_dict` is provided, it will be used for conversion.
        """

        if encodings_dict is None:
            int_iupac_dict = get_int_iupac_dict()
        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.tolist()

            int_iupac_dict = encodings_dict

        outer_list = list()

        n_rows = (
            len(self._samples) if encodings_dict is None else len(snp_data)
        )

        for i in range(n_rows):
            int_iupac = list()
            for j in range(len(snp_data[0])):
                int_iupac.append(int_iupac_dict[snp_data[i][j]])
            outer_list.append(int_iupac)

        return np.array(outer_list)

    def inverse_int_iupac(
        self,
        int_encoded_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert integer-encoded data back to original format.
        Args:
            int_encoded_data (numpy.ndarray of shape (n_samples, n_SNPs) or List[List[int]]): Input integer-encoded data.
            encodings_dict (Dict[str, int] or None): Encodings to convert from integer encoding to original format.
        Returns:
            numpy.ndarray: Original format data.
        """

        int_encodings_dict = (
            get_int_iupac_dict() if encodings_dict is None else encodings_dict
        )

        # Create inverse dictionary (from integer to key)
        inverse_int_encodings_dict = {
            v: k for k, v in int_encodings_dict.items()
        }

        if isinstance(int_encoded_data, np.ndarray):
            int_encoded_data = int_encoded_data.tolist()

        decoded_outer_list = []

        for i in range(len(int_encoded_data)):
            decoded_list = []
            for j in range(len(int_encoded_data[0])):
                # Look up original key using integer encoding
                decoded_list.append(
                    inverse_int_encodings_dict[int_encoded_data[i][j]]
                )
            decoded_outer_list.append(decoded_list)

        return np.array(decoded_outer_list)

    def read_popmap(self, popmapfile: Optional[str]) -> None:
        """
        Read population map from file and associate samples with populations.

        Args:
            popmapfile (str): Path to the population map file.
        """
        self.popmapfile = popmapfile

        # Instantiate popmap object
        my_popmap = ReadPopmap(popmapfile, verbose=self.verbose)
        return my_popmap

    def subset_with_popmap(
        self,
        my_popmap,
        samples: List[str],
        force: bool,
        include_pops: List[str],
        exclude_pops: List[str],
        return_indices=False,
    ):
        """Subset popmap and samples.

        Args:
            my_popmap (ReadPopmap): ReadPopmap instance.

            samples (List[str]): List of sample IDs.

            force (bool): If True, return a subset dictionary without the keys that weren't found. If False, raise an error if not all samples are present in the population map file.

            include_pops (Optional[List[str]]): List of populations to include. If provided, only samples belonging to these populations will be included in the popmap and alignment.

            exclude_pops (Optional[List[str]]): List of populations to exclude. If provided, samples belonging to these populations will be excluded from the popmap and alignment.

            return_indices (bool, optional): If True, return sample_indices. Defaults to False.

        Returns:
            ReadPopmap: ReadPopmap object.

        Raises:
            ValueError: Samples are missing from the population map file.
            ValueError: The number of individuals in the population map file differs from the number of samples in the GenotypeData object.


        """
        # Checks if all samples present in both popmap and alignment.
        popmap_ok = my_popmap.validate_popmap(samples, force=force)

        if include_pops is not None or exclude_pops is not None:
            # Subsets based on include_pops and exclude_pops
            my_popmap.subset_popmap(samples, include_pops, exclude_pops)

        if not force and not popmap_ok:
            raise ValueError(
                f"Not all samples are present in supplied popmap "
                f"file: {my_popmap.filename}\n"
            )

        if not force and include_pops is None and exclude_pops is None:
            if len(my_popmap.popmap) != len(samples):
                raise ValueError(
                    f"The number of individuals in the popmap file "
                    f"({len(my_popmap)}) differs from the number of samples "
                    f"({len(self.samples)})\n"
                )

            for sample in samples:
                if sample in my_popmap.popmap:
                    self._populations.append(my_popmap.popmap[sample])
        else:
            popmap_keys_set = set(my_popmap.popmap.keys())
            new_samples = [x for x in samples if x in popmap_keys_set]

            new_samples_set = set(new_samples)
            new_populations = [
                p for s, p in my_popmap.popmap.items() if s in new_samples_set
            ]

            if not new_samples:
                raise ValueError(
                    "No samples in the popmap file were found in the alignment file."
                )

            self._samples = new_samples
            self._populations = new_populations

        self._popmap = my_popmap.popmap
        self._popmap_inverse = my_popmap.popmap_flipped
        self._sample_indices = my_popmap.sample_indices

        if return_indices:
            return self._sample_indices

    def write_popmap(self, filename: str) -> None:
        """Write the population map to a file.

        Args:
            filename (str): Output file path.

        Raises:
            AttributeError: If samples or populations attributes are NoneType.
        """
        if not self.samples or self.samples is None:
            raise AttributeError("samples attribute is undefined.")

        if not self.populations or self.populations is None:
            raise AttributeError("populations attribute is undefined.")

        with open(filename, "w") as fout:
            for s, p in zip(self.samples, self.populations):
                fout.write(f"{s}\t{p}\n")

    def decode_012(
        self,
        X,
        write_output=True,
        prefix="imputer",
        is_nuc=False,
    ):
        """
        Decode 012-encoded or 0-9 integer-encoded imputed data to STRUCTURE or PHYLIP format.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Imputed data to decode, encoded as 012 or 0-9 integers.

            write_output (bool, optional): If True, save the decoded output to a file. If False, return the decoded data as a DataFrame. Defaults to True.

            prefix (str, optional): Prefix to append to the output file name. Defaults to "output".

            is_nuc (bool, optional): Whether the encoding is based on nucleotides instead of 012. Defaults to False.

        Returns:
            str or pandas.DataFrame: If write_output is True, returns the filename where the imputed data was written. If write_output is False, returns the decoded data as a DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)

        nuc = {
            "A/A": "A",
            "T/T": "T",
            "G/G": "G",
            "C/C": "C",
            "A/G": "R",
            "G/A": "R",
            "C/T": "Y",
            "T/C": "Y",
            "G/C": "S",
            "C/G": "S",
            "A/T": "W",
            "T/A": "W",
            "G/T": "K",
            "T/G": "K",
            "A/C": "M",
            "C/A": "M",
            "N/N": "N",
        }

        ft = self.filetype.lower()

        is_phylip = False
        if ft == "phylip" or ft == "vcf":
            is_phylip = True

        df_decoded = df.copy()
        df_decoded = df.copy().astype(object)

        # VAE uses [A,T,G,C] encodings. The other NN methods use [0,1,2] encodings.
        if is_nuc:
            classes_int = range(10)
            classes_string = [str(x) for x in classes_int]
            if is_phylip:
                gt = ["A", "T", "G", "C", "W", "R", "M", "K", "Y", "S", "N"]
            else:
                gt = [
                    "1/1",
                    "2/2",
                    "3/3",
                    "4/4",
                    "1/2",
                    "1/3",
                    "1/4",
                    "2/3",
                    "2/4",
                    "3/4",
                    "-9/-9",
                ]
            d = dict(zip(classes_int, gt))
            dstr = dict(zip(classes_string, gt))
            d.update(dstr)
            dreplace = {col: d for col in list(df.columns)}

        else:
            dreplace = dict()
            for col, ref, alt in zip(df.columns, self._ref, self._alt):
                # if site is monomorphic, set alt and ref state the same
                if alt is None:
                    alt = ref
                ref2 = f"{ref}/{ref}"
                alt2 = f"{alt}/{alt}"
                het2 = f"{ref}/{alt}"

                if is_phylip:
                    ref2 = nuc[ref2]
                    alt2 = nuc[alt2]
                    het2 = nuc[het2]

                d = {
                    "0": ref2,
                    0: ref2,
                    "1": het2,
                    1: het2,
                    "2": alt2,
                    2: alt2,
                    "-9": "N",
                    -9: "N",
                }
                dreplace[col] = d

        df_decoded.replace(dreplace, inplace=True)

        if write_output:
            outfile = os.path.join(
                f"{self.prefix}_output", "gtdata", "alignments", "012"
            )

        if ft.startswith("structure"):
            if ft.startswith("structure2row"):
                for col in df_decoded.columns:
                    df_decoded[col] = (
                        df_decoded[col]
                        .str.split("/")
                        .apply(lambda x: list(map(int, x)))
                    )

                df_decoded.insert(0, "sampleID", self._samples)
                df_decoded.insert(1, "popID", self._populations)

                # Transform each element to a separate row.
                df_decoded = (
                    df_decoded.set_index(["sampleID", "popID"])
                    .apply(pd.Series.explode)
                    .reset_index()
                )

            elif ft.startswith("structure1row"):
                df_decoded = pd.concat(
                    [
                        df_decoded[c]
                        .astype(str)
                        .str.split("/", expand=True)
                        .add_prefix(f"{c}_")
                        for c in df_decoded.columns
                    ],
                    axis=1,
                )

            elif ft == "structure":
                for col in df_decoded.columns:
                    df_decoded[col] = (
                        df_decoded[col]
                        .str.split("/")
                        .apply(lambda x: list(map(int, x)))
                    )

                df_decoded.insert(0, "sampleID", self._samples)
                df_decoded.insert(1, "popID", self._populations)

                # Transform each element to a separate row.
                df_decoded = (
                    df_decoded.set_index(["sampleID", "popID"])
                    .apply(pd.Series.explode)
                    .reset_index()
                )

            if write_output:
                of = f"{outfile}.str"
                df_decoded.insert(0, "sampleID", self._samples)
                df_decoded.insert(1, "popID", self._populations)

                df_decoded.to_csv(
                    of,
                    sep="\t",
                    header=False,
                    index=False,
                )

        elif ft.startswith("phylip"):
            if write_output:
                of = f"{outfile}.phy"
                header = f"{self.num_inds} {self.num_snps}\n"
                with open(of, "w") as fout:
                    fout.write(header)

                lst_decoded = df_decoded.values.tolist()

                with open(of, "a") as fout:
                    for sample, row in zip(self._samples, lst_decoded):
                        seqs = "".join([str(x) for x in row])
                        fout.write(f"{sample}\t{seqs}\n")

        if write_output:
            return of
        else:
            return df_decoded.values.tolist()

    def missingness_reports(
        self,
        plot_dir_prefix="snpio",
        file_prefix=None,
        zoom=True,
        horizontal_space=0.6,
        vertical_space=0.6,
        bar_color="gray",
        heatmap_palette="magma",
        plot_format="png",
        dpi=300,
    ):
        """
        Generate missingness reports and plots.

        The function will write several comma-delimited report files:

            1) individual_missingness.csv: Missing proportions per individual.

            2) locus_missingness.csv: Missing proportions per locus.

            3) population_missingness.csv: Missing proportions per population (only generated if popmapfile was passed to GenotypeData).

            4) population_locus_missingness.csv: Table of per-population and per-locus missing data proportions.

        A file missingness.<plot_format> will also be saved. It contains the following subplots:

            1) Barplot with per-individual missing data proportions.

            2) Barplot with per-locus missing data proportions.

            3) Barplot with per-population missing data proportions (only if popmapfile was passed to GenotypeData).

            4) Heatmap showing per-population + per-locus missing data proportions (only if popmapfile was passed to GenotypeData).

            5) Stacked barplot showing missing data proportions per-individual.

            6) Stacked barplot showing missing data proportions per-population (only if popmapfile was passed to GenotypeData).

        If popmapfile was not passed to GenotypeData, then the subplots and report files that require populations are not included.

        Args:
            plot_dir_prefix (str, optional): Prefix for output directory. Defaults to "snpio".

            zoom (bool, optional): If True, zoom in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.

            horizontal_space (float, optional): Set the width spacing between subplots. If your plots are overlapping horizontally, increase horizontal_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            vertical_space (float, optional): Set the height spacing between subplots. If your plots are overlapping vertically, increase vertical_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            bar_color (str, optional): Color of the bars on the non-stacked bar plots. Can be any color supported by matplotlib. See the matplotlib.pyplot.colors documentation. Defaults to 'gray'.
            heatmap_palette (str, optional): Palette to use for the heatmap plot. Can be any palette supported by seaborn. See the seaborn documentation. Defaults to 'magma'.

            plot_format (str, optional): Format to save the plots. Can be any of the following: "pdf", "png", "svg", "ps", "eps". Defaults to "png".

            dpi (int): The resolution in dots per inch. Defaults to 300.
        """
        params = dict(
            plot_dir_prefix=plot_dir_prefix,
            file_prefix=file_prefix,
            zoom=zoom,
            horizontal_space=horizontal_space,
            vertical_space=vertical_space,
            bar_color=bar_color,
            heatmap_palette=heatmap_palette,
            plot_format=plot_format,
            dpi=dpi,
        )

        df = pd.DataFrame(self.snp_data)
        df.replace(
            ["N", "-", ".", "?"],
            [np.nan, np.nan, np.nan, np.nan],
            inplace=True,
        )

        report_path = os.path.join(
            f"{plot_dir_prefix}_output", "gtdata", "reports"
        )

        Path(report_path).mkdir(exist_ok=True, parents=True)

        loc, ind, poploc, poptotal, indpop = Plotting.visualize_missingness(
            self, df, **params
        )

        fname = (
            "individual_missingness.csv"
            if file_prefix is None
            else f"{file_prefix}_individual_missingness.csv"
        )

        self._report2file(ind, report_path, fname)

        fname = (
            "locus_missingness.csv"
            if file_prefix is None
            else f"{file_prefix}_locus_missingness.csv"
        )

        self._report2file(loc, report_path, fname)

        fname = (
            "per_pop_and_locus_missingness.csv"
            if file_prefix is None
            else f"{file_prefix}_per_pop_and_locus_missingness.csv"
        )

        if self._populations is not None:
            self._report2file(poploc, report_path, fname)

            fname = (
                "population_missingness.csv"
                if file_prefix is None
                else f"{file_prefix}_population_missingness.csv"
            )

            self._report2file(poptotal, report_path, fname)

            fname = (
                "population_locus_missingness.csv"
                if file_prefix is None
                else f"{file_prefix}_population_locus_missingness.csv"
            )

            self._report2file(indpop, report_path, fname, header=True)

    def _report2file(
        self,
        df: pd.DataFrame,
        report_path: str,
        mypath: str,
        header: bool = False,
    ) -> None:
        """
        Write a DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): DataFrame to be written to the file.

            report_path (str): Path to the report directory.

            mypath (str): Name of the file to write.

            header (bool, optional): Whether to include the header row in the file. Defaults to False.
        """
        df.to_csv(
            os.path.join(report_path, mypath), header=header, index=False
        )

    def subset_vcf_data(
        self,
        loci_indices,
        sample_indices,
        vcf_attributes_path,
        samples=None,
        chunk_size=1000,
        is_filtered=False,
    ):
        fname = "vcf_attributes"
        if is_filtered:
            outdir = os.path.join(
                f"{self.prefix}_output", "nremover", "alignments", "vcf"
            )
            fname += "_filtered.h5"
        else:
            outdir = os.path.join(
                f"{self.prefix}_output", "gtdata", "alignments", "vcf"
            )
            fname += ".h5"

        Path(outdir).mkdir(exist_ok=True, parents=True)
        outfile = os.path.join(outdir, fname)

        with h5py.File(outfile, "w") as filtered_file:
            with h5py.File(vcf_attributes_path, "r") as original_file:
                # Iterate through each attribute key and subset the data
                for key in original_file.keys():
                    if key not in ["info", "calldata"]:
                        # Handling regular datasets
                        original_shape = original_file[key].shape
                        dtype = original_file[key].dtype
                        filtered_shape = list(original_shape)
                        if len(original_shape) > 0:
                            filtered_shape[0] = len(loci_indices)
                        filtered_dataset = filtered_file.create_dataset(
                            key, shape=tuple(filtered_shape), dtype=dtype
                        )
                        # Process in chunks
                        for start in range(0, len(loci_indices), chunk_size):
                            end = min(start + chunk_size, len(loci_indices))
                            loci_chunk = loci_indices[start:end]
                            data_chunk = original_file[key][loci_chunk]
                            filtered_dataset[start:end] = data_chunk
                    else:
                        # Handling "info" and "calldata" groups
                        filtered_group = filtered_file.create_group(key)
                        for inner_key in original_file[key].keys():
                            original_shape = original_file[
                                f"{key}/{inner_key}"
                            ].shape
                            dtype = original_file[f"{key}/{inner_key}"].dtype
                            filtered_shape = list(original_shape)
                            if len(original_shape) > 0:
                                filtered_shape[0] = len(loci_indices)
                            if len(original_shape) > 1:
                                filtered_shape[1] = len(sample_indices)
                            filtered_dataset = filtered_group.create_dataset(
                                inner_key,
                                shape=tuple(filtered_shape),
                                dtype=dtype,
                            )  # Create dataset for each inner_key

                            # Process in chunks
                            for start in range(
                                0, len(loci_indices), chunk_size
                            ):
                                end = min(
                                    start + chunk_size, len(loci_indices)
                                )
                                loci_chunk = loci_indices[start:end]

                                if len(original_shape) == 1:
                                    data_chunk = original_file[
                                        f"{key}/{inner_key}"
                                    ][loci_chunk]
                                else:  # len(original_shape) == 2, so key must be "calldata" or "snp_data"
                                    data_chunk = original_file[
                                        f"{key}/{inner_key}"
                                    ][loci_chunk, :][:, sample_indices]
                                filtered_dataset[start:end] = (
                                    data_chunk
                                    if len(filtered_shape) == 1
                                    else data_chunk[:, :]
                                )

        return outfile

    def _genotype_to_iupac(self, genotype: str) -> str:
        """
        Convert a genotype string to its corresponding IUPAC code.

        Args:
            genotype (str): Genotype string in the format "x/y".

        Returns:
            str: Corresponding IUPAC code for the input genotype. Returns 'N' if the genotype is not in the lookup dictionary.
        """
        iupac_dict = {
            "0/0": "A",
            "1/1": "T",
            "2/2": "C",
            "3/3": "G",
            "0/1": "W",
            "0/2": "M",
            "0/3": "R",
            "1/2": "Y",
            "1/3": "K",
            "2/3": "S",
            "-9/-9": "N",
        }
        return iupac_dict.get(genotype, "N")

    def _iupac_to_genotype(self, iupac_code: str) -> str:
        """
        Convert an IUPAC code to its corresponding genotype string.

        Args:
            iupac_code (str): IUPAC code.

        Returns:
            str: Corresponding genotype string for the input IUPAC code. Returns '-9/-9' if the IUPAC code is not in the lookup dictionary.
        """
        genotype_dict = {
            "A": "0/0",
            "T": "1/1",
            "C": "2/2",
            "G": "3/3",
            "W": "0/1",
            "M": "0/2",
            "R": "0/3",
            "Y": "1/2",
            "K": "1/3",
            "S": "2/3",
            "N": "-9/-9",
        }
        return genotype_dict.get(iupac_code, "-9/-9")

    def calc_missing(
        self, df: pd.DataFrame, use_pops: bool = True
    ) -> Tuple[
        pd.Series,
        pd.Series,
        Optional[pd.DataFrame],
        Optional[pd.Series],
        Optional[pd.DataFrame],
    ]:
        """
        Calculate missing value statistics based on a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing genotype data.

            use_pops (bool, optional): If True, calculate statistics per population. Defaults to True.

        Returns:
            Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame]]: A tuple of missing value statistics:

            - loc (pd.Series): Missing value proportions per locus.

            - ind (pd.Series): Missing value proportions per individual.

            - poploc (Optional[pd.DataFrame]): Missing value proportions per population and locus. Only returned if use_pops=True.

            - poptot (Optional[pd.Series]): Missing value proportions per population. Only returned if use_pops=True.

            - indpop (Optional[pd.DataFrame]): Missing value proportions per individual and population. Only returned if use_pops=True.
        """
        # Get missing value counts per-locus.
        loc = df.isna().sum(axis=0) / self.num_inds
        loc = loc.round(2)

        # Get missing value counts per-individual.
        ind = df.isna().sum(axis=1) / self.num_snps
        ind = ind.round(2)

        poploc = None
        poptot = None
        indpop = None
        if use_pops:
            popdf = df.copy()
            popdf.index = self._populations
            misscnt = popdf.isna().groupby(level=0).sum()
            n = popdf.groupby(level=0).size()
            poploc = misscnt.div(n, axis=0).round(2).T
            poptot = misscnt.sum(axis=1) / self.num_snps
            poptot = poptot.div(n, axis=0).round(2)
            indpop = df.copy()

        return loc, ind, poploc, poptot, indpop

    def copy(self):
        """Create a deep copy of the GenotypeData object.

        Returns:
            GenotypeData: A new GenotypeData object with the same attributes as the original.
        """
        # Create a new instance of GenotypeData
        new_obj = GenotypeData.__new__(GenotypeData)

        # Shallow copy of the original object's __dict__
        new_obj.__dict__.update(self.__dict__)

        # Deep copy all attributes EXCEPT the problematic VariantHeader
        for name, attr in self.__dict__.items():
            if name != "vcf_header":
                setattr(new_obj, name, copy.deepcopy(attr))

        # Explicitly copy VariantHeader
        if self.vcf_header:
            new_header = pysam.VariantHeader()
            new_header = self.vcf_header.copy()
            new_obj.vcf_header = new_header

        return new_obj

    class _DataFormat012:
        def __init__(self, instance, is_structure: bool = False):
            """
            Initialize the _DataFormat012 class.

            Args:
                instance: An instance of the GenotypeData class.

                is_structure (bool, optional): Specify whether the data is in STRUCTURE format. Defaults to False.
            """
            self.instance = instance
            self.is_structure = is_structure

        def __call__(self, fmt="list"):
            """
            Convert genotype data in 012 format to the specified output format.

            Args:
                fmt (str, optional): The desired output format. Supported formats: 'list', 'numpy', 'pandas'. Defaults to 'list'.

            Returns:
                The converted genotype data in the specified format.

            Raises:
                ValueError: Invalid format supplied.
            """
            if fmt == "list":
                return list(
                    self.instance.convert_012(
                        self.instance.snp_data, vcf=self.is_structure
                    )
                )

            elif fmt == "numpy":
                return np.array(
                    self.instance.convert_012(
                        self.instance.snp_data, vcf=self.is_structure
                    )
                )

            elif fmt == "pandas":
                return pd.DataFrame.from_records(
                    self.instance.convert_012(
                        self.instance.snp_data, vcf=self.is_structure
                    )
                )

            else:
                raise ValueError(
                    "Invalid format. Supported formats: 'list', 'numpy', 'pandas'"
                )

    @classmethod
    def plot_performance(cls, fontsize=14, color="#8C56E3", figsize=(16, 9)):
        """Plots the performance metrics: CPU Load, Memory Footprint, and Execution Time.

        Takes a dictionary of performance data and plots the metrics for each of the methods. The resulting plot is saved in a .png file in the ``tests`` directory.

        Args:
            resource_data (dict): Dictionary with performance data. Keys are method names, and values are dictionaries with keys 'cpu_load', 'memory_footprint', and 'execution_time'.

            fontsize (int, optional): Font size to be used in the plot. Defaults to 14.

            color (str, optional): Color to be used in the plot. Should be a valid color string. Defaults to "#8C56E3".

            figsize (tuple, optional): Size of the figure. Should be a tuple of two integers. Defaults to (16, 9).

        Returns:
            None. The function saves the plot as a .png file.
        """
        plot_dir = os.path.join(
            f"{cls.prefix}_output", "gtdata", "plots", "performance"
        )
        Path(plot_dir).mkdir(exist_ok=True, parents=True)

        Plotting.plot_performance(
            cls.resource_data,
            fontsize=fontsize,
            color=color,
            figsize=figsize,
            plot_dir=plot_dir,
        )

    @property
    def inputs(self):
        """Get GenotypeData keyword arguments as a dictionary."""
        return self._kwargs

    @inputs.setter
    def inputs(self, value):
        """Setter method for class keyword arguments."""
        self._kwargs = value

    @property
    def num_snps(self) -> int:
        """Number of snps in the dataset.

        Returns:
            int: Number of SNPs per individual.
        """
        return len(self._snp_data[0])

    @property
    def num_inds(self) -> int:
        """Number of individuals in dataset.

        Returns:
            int: Number of individuals in input data.
        """
        return len(self._snp_data)

    @property
    def populations(self) -> List[Union[str, int]]:
        """Population Ids.

        Returns:
            List[Union[str, int]]: Population IDs.
        """
        return self._populations

    @property
    def popmap(self) -> Dict[str, str]:
        """Dictionary object with SampleIDs as keys and popIDs as values."""
        return self._popmap

    @popmap.setter
    def popmap(self, value):
        """Dictionary with SampleIDs as keys and popIDs as values."""
        if not isinstance(value, dict):
            raise TypeError(
                f"popmap must be a dictionary object, but got {type(value)}."
            )

        if not all(isinstance(v, (str, int)) for v in value.values()):
            raise TypeError(f"popmap values must be strings or integers")
        self._popmap = value

    @property
    def popmap_inverse(self) -> None:
        """Inverse popmap dictionary with populationIDs as keys and lists of sampleIDs as values."""
        return self._popmap_inverse

    @popmap_inverse.setter
    def popmap_inverse(self, value) -> Dict[str, List[str]]:
        """Setter for popmap_inverse. Should have populationIDs as keys and lists of corresponding sampleIDs as values."""
        if not isinstance(value, dict):
            raise TypeError(
                f"popmap_inverse must be a dictionary object, but got {type(value)}"
            )

        if all(isinstance(v, list) for v in value.values()):
            raise TypeError(
                f"popmap_inverse values must be lists of sampleIDs for the given populationID key"
            )

        self._popmap_inverse = value

    @property
    def samples(self) -> List[str]:
        """Sample IDs in input order.

        Returns:
            List[str]: Sample IDs in input order.
        """
        return self._samples

    @samples.setter
    def samples(self, value) -> None:
        """Get the sampleIDs as a list of strings."""
        self._samples = value

    @property
    def snpsdict(self) -> Dict[str, List[str]]:
        """
        Dictionary with Sample IDs as keys and lists of genotypes as values.
        """
        self._snpsdict = self._make_snpsdict()
        return self._snpsdict

    @snpsdict.setter
    def snpsdict(self, value):
        """Set snpsdict object, which is a dictionary with sample IDs as keys and lists of genotypes as values."""
        self._snpsdict = value

    @property
    def snp_data(self) -> List[List[str]]:
        """Get the genotypes as a 2D list of shape (n_samples, n_loci)."""
        return self._snp_data

    @snp_data.setter
    def snp_data(self, value) -> None:
        """Set snp_data. Input can be a 2D list, numpy array, pandas DataFrame, or MultipleSeqAlignment object."""
        if not isinstance(value, list):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, pd.DataFrame):
                value = value.values.tolist()
            elif isinstance(value, MultipleSeqAlignment):
                value = [list(str(record.seq)) for record in value]
            else:
                raise TypeError(
                    f"snp_data must be a list, numpy array, pandas dataframe, or MultipleSeqAlignment, but got {type(value)}"
                )
        self._snp_data = value
        self._validate_seq_lengths()

    @property
    def genotypes_012(
        self,
    ) -> Union[List[List[int]], np.ndarray, pd.DataFrame]:
        """Encoded 012 genotypes as a 2D list, numpy array, or pandas DataFrame.

        The examples below show how to return the different format types.

        Returns:
            List[List[int]], np.ndarray, or pd.DataFrame: encoded 012 genotypes.

        Examples:
            >>># Get a 2D list.
            >>>gt_list = GenotypeData.genotypes_012(fmt="list")
            >>>
            >>># Get a numpy array.
            >>>gt_array = GenotypeData.genotypes_012(fmt="numpy")
            >>>
            >>># Get a pandas DataFrame.
            >>>gt_df = GenotypeData.genotypes_012(fmt="pandas")
        """
        # TODO: Remove deprecated 'vcf' and 'is_structure' arguments.
        # # is_str = True if self.filetype.startswith("structure") else False
        return self._DataFormat012(self, is_structure=False)

    @genotypes_012.setter
    def genotypes_012(self, value) -> List[List[int]]:
        """Set the 012 genotypes. They will be decoded back to a 2D list of genotypes as ``snp_data``\.

        Args:
            value (np.ndarray): 2D numpy array with 012-encoded genotypes.
        """
        self._snp_data = self.decode_012(value, write_output=False)

    @property
    def genotypes_onehot(self) -> Union[np.ndarray, List[List[List[float]]]]:
        """One-hot encoded snps format of shape (n_samples, n_loci, 4).

        Returns:
            numpy.ndarray: One-hot encoded numpy array of shape (n_samples, n_loci, 4).
        """
        return self.convert_onehot(self._snp_data)

    @genotypes_onehot.setter
    def genotypes_onehot(self, value) -> List[List[int]]:
        """Set the onehot-encoded genotypes. They will be decoded back to a 2D list of IUPAC genotypes as ``snp_data``\."""
        if isinstance(value, pd.DataFrame):
            X = value.to_numpy()
        elif isinstance(value, list):
            X = np.array(value)
        elif isinstance(value, np.ndarray):
            X = value
        else:
            raise TypeError(
                f"genotypes_onehot must be of type pd.DataFrame, np.ndarray, or list, but got {type(value)}"
            )

        Xt = self.inverse_onehot(X)
        self._snp_data = Xt.tolist()

    @property
    def genotypes_int(self) -> np.ndarray:
        """Integer-encoded (0-9 including IUPAC characters) snps format.

        Returns:
            numpy.ndarray: 2D array of shape (n_samples, n_sites), integer-encoded from 0-9 with IUPAC characters.
        """
        arr = self.convert_int_iupac(self._snp_data)
        return arr

    @genotypes_int.setter
    def genotypes_int(
        self, value: Union[pd.DataFrame, np.ndarray, List[List[int]]]
    ) -> List[List[int]]:
        """Set the integer-encoded (0-9) genotypes. They will be decoded back to a 2D list of IUPAC genotypes as ``snp_data``\."""
        if isinstance(value, pd.DataFrame):
            X = value.to_numpy()
        elif isinstance(value, list):
            X = np.array(value)
        elif isinstance(value, np.ndarray):
            X = value
        else:
            raise TypeError(
                f"genotypes_onehot must be of type pd.DataFrame, np.ndarray, or list, but got {type(value)}"
            )

        Xt = self.inverse_int_iupac(X)
        self._snp_data = Xt.tolist()

    @property
    def alignment(self) -> List[MultipleSeqAlignment]:
        """Get alignment as a biopython MultipleSeqAlignment object.

        This is good for printing and visualizing the alignment. If you want the alignment as a 2D list object, then use the ``snp_data`` property instead.
        """
        return MultipleSeqAlignment(
            [
                SeqRecord(Seq("".join(row)), id=sample)
                for sample, row in zip(self._samples, self._snp_data)
            ]
        )

    @alignment.setter
    def alignment(
        self, value: Union[np.ndarray, pd.DataFrame, MultipleSeqAlignment]
    ) -> None:
        """
        Setter method for the alignment.

        Args:
            value (Bio.MultipleSeqAlignment, list, np.ndarray, pd.DataFrame): The MultipleSeqAlignment object to set as the alignment.

        Raises:
            TypeError: If the input value is not a MultipleSeqAlignment object, list, numpy array, or pandas DataFrame.
        """
        if isinstance(value, MultipleSeqAlignment):
            alignment_array = np.array(
                [list(str(record.seq)) for record in value]
            )
        elif isinstance(value, pd.DataFrame):
            # Convert list, numpy array, or pandas DataFrame to list
            alignment_array = value.values.tolist()
        elif isinstance(value, np.ndarray):
            alignment_array = value.tolist()
        elif isinstance(value, list):
            alignment_array = value
        else:
            raise TypeError(
                "alignment must be a MultipleSequenceAlignment object, list, numpy array, or pandas DataFrame."
            )

        self._snp_data = alignment_array

    @property
    def vcf_attributes(self) -> str:
        """Path to HDF5 file containing Attributes read in from VCF file.

        Returns:
            str: Path to HDF5 file with keys corresponding to VCF file attributes and values being either a dictionary of numpy arrays (if key == 'calldata' or key == 'info') or numpy arrays (if key != 'calldata' and key != 'info').
        Raises:
            IOError: If vcf_attributes.h5 file doesn't exist.
        """
        file_path = os.path.join(
            f"{self.prefix}_output",
            "gtdata",
            "alignments",
            "vcf",
            "vcf_attributes.h5",
        )
        if not Path(file_path).is_file():
            raise IOError(f"{file_path} could not be found.")
        return self._vcf_attributes

    @vcf_attributes.setter
    def vcf_attributes(self, value: str) -> None:
        """Setter method for VCF file attributes dictionary.

        This should be a dictionary with the 9 standard VCF file keys ("chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format") plus the calldata object. The "info" object should be another dictionary with each INFO field name as the keys and an associated numpy array as the values. The "format" object should just be a numpy array of shape (n_format_fields,). The calldata object should be another dictionary with each calldata field as keys, prepended by "calldata/{key}. The keys for calldata will be the same as in the "format" field.

        Args:
            value (str): File path to HDF5 file containing VCF attributes.
        """
        self._vcf_attributes = value

    @property
    def loci_indices(self) -> List[int]:
        """Column indices for retained loci in filtered alignment."""
        return self._loci_indices

    @loci_indices.setter
    def loci_indices(self, value) -> None:
        """Column indices for retained loci in filtered alignment."""
        self._loci_indices = value

    @property
    def sample_indices(self) -> List[int]:
        """Row indices for retained samples in alignemnt."""
        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, value: List[int]) -> None:
        """Row indices for retained samples in alignemnt.

        NOTE: This will also subset the ``samples`` property to the integer values provided in ``sample_indices``\.

        """
        self._sample_indices = value
        self._samples = [x for i, x in enumerate(self._samples) if i in value]
        self._populations = [
            p
            for i, (s, p) in zip(self._samples, self._populations)
            if s in value
        ]

    @property
    def ref(self) -> List[str]:
        """Get list of reference alleles of length num_snps."""
        return self._ref

    @ref.setter
    def ref(self, value: List[str]) -> None:
        """Setter for list of reference alleles of length num_snps."""
        self._ref = value

    @property
    def alt(self) -> List[str]:
        """Get list of alternate alleles of length num_snps."""
        return self._alt

    @alt.setter
    def alt(self, value) -> None:
        """Setter for list of alternate alleles of length num_snps."""
        self._alt = value

    @property
    def q(self):
        """Get q-matrix object for phylogenetic tree."""
        if self.qmatrix_iqtree is not None and self.qmatrix is None:
            self._q = self.q_from_iqtree(self.qmatrix_iqtree)
        elif self.qmatrix_iqtree is None and self.qmatrix is not None:
            self._q = self.q_from_file(self.qmatrix)
        elif (
            self.qmatrix is None
            and self.qmatrix_iqtree is None
            and self._q is None
        ):
            raise TypeError(
                "qmatrix or qmatrix_iqtree must be provided at class instantiation or the q property must be set to get the q object."
            )
        return self._q

    @q.setter
    def q(self, value):
        """Set q-matrix for phylogenetic tree."""
        self._q = value

    @property
    def site_rates(self):
        """Get site rate data for phylogenetic tree."""
        if self.siterates_iqtree is not None and self.siterates is None:
            self._site_rates = self.siterates_from_iqtree(
                self.siterates_iqtree
            )
            self._validate_rates()
        elif self.siterates_iqtree is None and self.siterates is not None:
            self._site_rates = self.siterates_from_file(self.siterates)
            self._validate_rates()
        elif (
            self.siterates_iqtree is None
            and self.siterates is None
            and self._site_rates is None
        ):
            raise TypeError(
                "siterates or siterates_iqtree must be provided at class instantiation or the site_rates property must be set to get the site_rates object."
            )
        self._site_rates = [
            self._site_rates[i]
            for i in range(len(self._site_rates))
            if i in self.loci_indices
        ]
        return self._site_rates

    @site_rates.setter
    def site_rates(self, value):
        """Set site_rates object."""
        self._site_rates = value

    @property
    def tree(self):
        """Get newick tree provided at class instantiation."""
        if self.guidetree is not None:
            self._tree = self.read_tree(self.guidetree)
        elif self.guidetree is None and self._tree is None:
            raise TypeError(
                "Either a guidetree file must be provided at class instantiation or the tree property must be set to get the tree object."
            )
        return self._tree

    @tree.setter
    def tree(self, value):
        """Setter for newick tree data."""
        self._tree = value


def merge_alleles(
    first: List[Union[str, int]],
    second: Optional[List[Union[str, int]]] = None,
) -> List[str]:
    """Merges first and second alleles in a structure file.

    Args:
        first (List[Union[str, int] or None): Alleles on the first line.

        second (List[Union[str, int]] or None, optional): Second row of alleles. Defaults to None.

    Returns:
        List[str]: VCF file-style genotypes (i.e., split by "/").

    Raises:
        ValueError: If the first and second lines have differing lengths.

        ValueError: If the line has a non-even number of alleles.
    """
    ret = list()
    if second is not None:
        if len(first) != len(second):
            raise ValueError(
                "First and second lines have different number of alleles\n"
            )
        else:
            for i in range(0, len(first)):
                ret.append(str(first[i]) + "/" + str(second[i]))
    else:
        if len(first) % 2 != 0:
            raise ValueError("Line has non-even number of alleles!\n")
        else:
            for i, j in zip(first[::2], first[1::2]):
                ret.append(str(i) + "/" + str(j))
    return ret
