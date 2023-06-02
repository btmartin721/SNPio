import os
import sys
import warnings
from pathlib import Path

from typing import Optional, Union, List, Dict

# Make sure python version is >= 3.8
if sys.version_info < (3, 8):
    raise ImportError("Python < 3.8 is not supported!")

import numpy as np
import pandas as pd
import toytree as tt

from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

try:
    from .popmap_file import ReadPopmap
    from ..utils.plotting import Plotting
    from ..utils import sequence_tools
except (ModuleNotFoundError, ValueError):
    from read_input.popmap_file import ReadPopmap
    from utils.plotting import Plotting
    from utils import sequence_tools


class GenotypeData:
    """Read genotype and tree data and encode genotypes.

    Reads in a PHYLIP or STRUCTURE-formatted input file and converts the genotypes to 012 or one-hot encodings.

    Args:
            filename (str or None): Path to input file containing genotypes. Defaults to None.

            filetype (str or None): Type of input genotype file. Possible ``filetype`` values include: "phylip", "structure1row", or "structure2row". VCF compatibility may be added in the future, but is not currently supported. Defaults to None.

            popmapfile (str or None): Path to population map file. If ``popmapfile`` is supplied and ``filetype`` is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column. Defaults to None.

            guidetree (str or None): Path to input treefile. Defaults to None.

            qmatrix_iqtree (str or None): Path to iqtree output file containing Q rate matrix. Defaults to None.

            qmatrix (str or None): Path to file containing only Q rate matrix, and not the full iqtree file. Defaults to None.

            siterates (str or None, optional): Path to file containing per-site rates, with 1 rate per line corresponding to 1 site. Not required if ``genotype_data`` is defined with the siterates or siterates_iqtree option. Defaults to None.

            siterates_iqtree (str or None, optional): Path to *.rates file output from IQ-TREE, containing a per-site rate table. If specified, ``ImputePhylo`` will read the site-rates from the IQ-TREE output file. Cannot be used in conjunction with ``siterates`` argument. Not required if the ``siterates`` or ``siterates_iqtree`` options were used with the ``GenotypeData`` object. Defaults to None.

            plot_format (str, optional): Format to save report plots. Valid options include: 'pdf', 'svg', 'png', and 'jpeg'. Defaults to "pdf".

            prefix (str, optional): Prefix to use for output directory. Defaults to 'imputer'.

            verbose (bool, optional): Verbosity level. Defaults to True.

    Attributes:
            samples (List[str]): List containing sample IDs of shape (n_samples,).

            snps (List[List[str]]): 2D list of shape (n_samples, n_sites) containing genotypes.

            pops (List[str]): List of population IDs of shape (n_samples,).

            onehot (List[List[List[float]]]): One-hot encoded genotypes as a 3D list of shape (n_samples, n_sites, 4). The inner-most list represents the four nucleotide bases in the order of "A", "T", "G", "C". If position 0 contains a 1.0, then the site is an "A". If position 1 contains a 1.0, then the site is a "T"...etc. Two values of 0.5 indicates a heterozygote. Missing data is encoded as four values of 0.0.

            guidetree (toytree object): Input guide tree as a toytree object.

            num_snps (int): Number of SNPs (features) present in the dataset.

            num_inds: (int): Number of individuals (samples) present in the dataset.

    Properties:
            snpcount (int): Number of SNPs (features) in the dataset.

            indcount (int): Number of individuals (samples) in the dataset.

            populations (List[str]): List of population IDs of shape (n_samples,).

            individuals (List[str]): List of sample IDs of shape (n_samples,).

            genotypes012_list (List[List[str]]): List of 012-encoded genotypes of shape (n_samples, n_sites).

            genotypes012_array (numpy.ndarray): 012-encoded genotypes of shape (n_samples, n_sites).

            genotypes012_df (pandas.DataFrame): 012-encoded genotypes of shape (n_samples, n_sites). Missing values are encoded as -9.

            genotypes_onehot (numpy.ndarray of shape (n_samples, n_SNPs, 4)): One-hot encoded numpy array. The inner-most array consists of one-hot encoded values for the four nucleotides in the order of "A", "T", "G", "C". Values of 0.5 indicate heterozygotes, and missing values contain 0.0 for all four nucleotides.

            q (pandas.DataFrame): Q-matrix of nucleotide substitution rates, if initialized with ``qmatrix`` or ``qmatrix_iqtree``

            site_rates (List[float]): Site-specific substitution rates, if initialized with ``siterates`` or ``siterates_iqtree``
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        filetype: Optional[str] = None,
        popmapfile: Optional[str] = None,
        guidetree: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        siterates_iqtree: Optional[str] = None,
        plot_format: Optional[str] = "pdf",
        prefix="imputer",
        verbose: bool = True,
    ) -> None:
        """
        Initialize the GenotypeData object.

        Args:
            filename (Optional[str], default=None): Path to input file containing genotypes.
            filetype (Optional[str], default=None): Type of input genotype file. Possible values include: "phylip", "structure1row", or "structure2row".
            popmapfile (Optional[str], default=None): Path to population map file. If supplied and filetype is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column.
            guidetree (Optional[str], default=None): Path to input treefile.
            qmatrix_iqtree (Optional[str], default=None): Path to iqtree output file containing Q rate matrix.
            qmatrix (Optional[str], default=None): Path to file containing only Q rate matrix, and not the full iqtree file.
            siterates (Optional[str], default=None): Path to file containing per-site rates, with 1 rate per line corresponding to 1 site. Not required if genotype_data is defined with the siterates or siterates_iqtree option.
            siterates_iqtree (Optional[str], default=None): Path to *.rates file output from IQ-TREE, containing a per-site rate table. Cannot be used in conjunction with siterates argument. Not required if the siterates or siterates_iqtree options were used with the GenotypeData object.
            plot_format (Optional[str], default="pdf"): Format to save report plots. Valid options include: 'pdf', 'svg', 'png', and 'jpeg'.
            prefix (str, default='imputer'): Prefix to use for output directory.
            verbose (bool, default=True): Verbosity level.
        """

        self.filename = filename
        self.filetype = filetype
        self.popmapfile = popmapfile
        self.guidetree = guidetree
        self.qmatrix_iqtree = qmatrix_iqtree
        self.qmatrix = qmatrix
        self.siterates = siterates
        self.siterates_iqtree = siterates_iqtree
        self.plot_format = plot_format
        self.prefix = prefix
        self.verbose = verbose

        self.samples: List[str] = list()
        self.snps: List[List[int]] = list()
        self.pops: List[Union[str, int]] = list()
        self.onehot: Union[np.ndarray, List[List[List[float]]]] = list()
        self.ref = list()
        self.alt = list()
        self._num_snps: int = 0
        self._num_inds: int = 0
        self.q = None
        self.site_rates = None
        self.tree = None
        self.int_iupac = None
        self._popmap = None
        self._popmap_inverse = None

        if self.qmatrix_iqtree is not None and self.qmatrix is not None:
            raise TypeError(
                "qmatrix_iqtree and qmatrix cannot both be defined"
            )

        if self.siterates_iqtree is not None and self.siterates is not None:
            raise TypeError(
                "siterates_iqtree and siterates cannot both be defined"
            )

        if self.filetype is not None:
            self._parse_filetype(filetype, popmapfile)

        if self.popmapfile is not None:
            self.read_popmap(popmapfile)

        if self.guidetree is not None:
            self.tree = self.read_tree(self.guidetree)
        elif self.guidetree is None:
            self.tree = None

        if self.qmatrix_iqtree is not None:
            self.q = self.q_from_iqtree(self.qmatrix_iqtree)
        elif self.qmatrix_iqtree is None and self.qmatrix is not None:
            self.q = self.q_from_file(self.qmatrix)
        elif self.qmatrix is None and self.qmatrix_iqtree is None:
            self.q = None

        if self.siterates_iqtree is not None:
            self.site_rates = self.siterates_from_iqtree(self.siterates_iqtree)
            self._validate_rates()
        elif self.siterates_iqtree is None and self.siterates is not None:
            self.site_rates = self.siterates_from_file(self.siterates)
            self._validate_rates()
        elif self.siterates is None and self.siterates_iqtree is None:
            self.site_rates = None

    def _parse_filetype(
        self, filetype: Optional[str] = None, popmapfile: Optional[str] = None
    ) -> None:
        """
        Check the filetype and call the appropriate function to read the file format.

        Args:
            filetype (Optional[str], default=None): Filetype. Supported values include: "phylip", "structure1row", "structure2row", "structure1rowPopID", and "structure2rowPopID".
            popmapfile (Optional[str], default=None): Path to population map file.

        Raises:
            OSError: No filetype specified.
            OSError: Filetype not supported.
        """
        if filetype is None:
            raise OSError("No filetype specified.\n")
        else:
            if filetype == "phylip":
                self.filetype = filetype
                self.read_phylip()
            elif filetype.lower().startswith("structure1row"):
                if popmapfile is not None and filetype.lower().endswith("row"):
                    self.filetype = "structure1row"
                    self.read_structure(onerow=True, popids=False)

                elif popmapfile is None and filetype.lower().endswith("popid"):
                    self.filetype = "structure1rowPopID"
                    self.read_structure(onerow=True, popids=True)

                elif popmapfile is not None and filetype.lower().endswith(
                    "popid"
                ):
                    print(
                        "WARNING: popmapfile was not None but provided "
                        "filetype was structure1rowPopID. Using populations "
                        "from 2nd column in STRUCTURE file."
                    )
                    self.filetype = "structure1rowPopID"
                    self.read_structure(onerow=True, popids=True)

                elif popmapfile is None and filetype.lower().endswith("row"):
                    raise ValueError(
                        "If popmap file is not provided, filetype must be "
                        "structure1rowPopID and the 2nd STRUCTURE file column "
                        "should contain population IDs"
                    )

                else:
                    raise ValueError(
                        f"Unsupported filetype provided: {filetype}"
                    )

            elif filetype.lower().startswith("structure2row"):
                if popmapfile is not None and filetype.lower().endswith("row"):
                    self.filetype = "structure2row"
                    self.read_structure(onerow=False, popids=False)

                elif popmapfile is None and filetype.lower().endswith("popid"):
                    self.filetype = "structure2rowPopID"
                    self.read_structure(onerow=False, popids=True)

                elif popmapfile is not None and filetype.lower().endswith(
                    "popid"
                ):
                    print(
                        "WARNING: popmapfile was not None, but provided "
                        "filetype was structure2rowPopID. Using populations "
                        "from 2nd column in STRUCTURE file."
                    )
                    self.filetype = "structure2rowPopID"
                    self.read_structure(onerow=False, popids=True)

                elif popmapfile is None and filetype.lower().endswith("row"):
                    raise ValueError(
                        "If popmap file is not provided, filetype must be "
                        "structure2rowPopID and the 2nd STRUCTURE file column "
                        "should contain population IDs"
                    )

                else:
                    raise OSError(f"Unsupported filetype provided: {filetype}")

            elif filetype == "012":
                self.filetype = filetype
                self.read_012()

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
            TypeError: Filetype does not match the validation.
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

        Format should be of type 0 (see toytree documentation).

        Args:
            treefile (str): Path to Newick-style tree file.

        Returns:
            toytree.tree object: Input tree as toytree object.
        """
        if not os.path.isfile(treefile):
            raise FileNotFoundError(f"File {treefile} not found!")

        assert os.access(treefile, os.R_OK), f"File {treefile} isn't readable"

        return tt.tree(treefile, tree_format=0)

    def q_from_file(self, fname: str, label: bool = True) -> pd.DataFrame:
        """
        Read Q matrix from file on disk.

        Args:
            fname (str): Path to Q matrix input file.
            label (bool): True if nucleotide label order is present, otherwise False.

        Returns:
            pandas.DataFrame: Q-matrix as pandas DataFrame object.
        """
        q = self._blank_q_matrix()

        if not label:
            print(
                "Warning: Assuming the following nucleotide order: A, C, G, T"
            )

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
        Read in Q-matrix from *.iqtree file.

        The *.iqtree file is one of the IQ-TREE output files and contains the standard output of the IQ-TREE run.

        Args:
            iqfile (str): Path to *.iqtree file.

        Returns:
            pandas.DataFrame: Q-matrix as pandas DataFrame.

        Raises:
            FileNotFoundError: If iqtree file could not be found.
            IOError: If iqtree file could not be read from.
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
            sys.exit(f"Could not open iqtree file {iqfile}")

        # Population q matrix with values from iqtree file
        order = [l[0] for l in qlines]
        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])

        qdf = pd.DataFrame(q)
        return qdf.T

    def _blank_q_matrix(
        self, default: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        q: Dict[str, Dict[str, float]] = dict()
        for nuc1 in ["A", "C", "G", "T"]:
            q[nuc1] = dict()
            for nuc2 in ["A", "C", "G", "T"]:
                q[nuc1][nuc2] = default
        return q

    def siterates_from_iqtree(self, iqfile: str) -> pd.DataFrame:
        """
        Read in site-specific substitution rates from *.rates file.

        The *.rates file is an optional IQ-TREE output files and contains a table of site-specific rates and rate categories.

        Args:
            iqfile (str): Path to *.rates input file.

        Returns:
            List[float]: List of rates.

        Raises:
            FileNotFoundError: If rates file could not be found.
            IOError: If rates file could not be read from.
        """
        s = list()
        try:
            with open(iqfile, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    elif line[0] == "#":
                        continue
                    else:
                        stuff = line.split()
                        if stuff[0] == "Site":
                            continue
                        else:
                            s.append(float(stuff[1]))
        except (IOError, FileNotFoundError):
            sys.exit(f"Could not open iqtree file {iqfile}")
        return s

    def _validate_rates(self) -> None:
        """
        Validate if the number of site rates matches the number of SNPs.

        Raises:
            ValueError: If the number of site rates does not match the number of SNPs.
        """

    def siterates_from_file(self, fname: str) -> List[float]:
        """
        Read site-specific substitution rates from a file on disk.

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

    def read_structure(
        self, onerow: bool = False, popids: bool = True
    ) -> None:
        """
        Read a structure file with one or two rows per individual.

        Args:
            onerow (bool, optional): True if file is in one-row format. False if two-row format. Defaults to False.
            popids (bool, optional): True if population IDs are present as 2nd column in structure file, otherwise False. Defaults to True.

        Raises:
            ValueError: Sample names do not match for two-row format.
            ValueError: Population IDs do not match for two-row format.
            AssertionError: All sequences must be the same length.
        """
        if self.verbose:
            print(f"\nReading structure file {self.filename}...")

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
                            self.pops.append(pop)
                            firstline = firstline[2:]
                            secondline = secondline[2:]
                        else:
                            firstline = firstline[1:]
                            secondline = secondline[1:]
                        self.samples.append(ind)
                        genotypes = merge_alleles(firstline, secondline)
                        snp_data.append(genotypes)
                        self.snpsdict[ind] = genotypes
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
                        self.pops.append(pop)
                        firstline = firstline[2:]
                    else:
                        firstline = firstline[1:]
                    self.samples.append(ind)
                    genotypes = merge_alleles(firstline, second=None)
                    snp_data.append(genotypes)
                    self.snpsdict[ind] = genotypes
                    firstline = None

        self._snp_data = snp_data

        if self.verbose:
            print("Done!")

        # Get number of samples and snps
        self._num_snps = len(self.snps[0])
        self._num_inds = len(self.samples)

        if self.verbose:
            print(
                f"\nFound {self._num_snps} SNPs and {self._num_inds} "
                f"individuals...\n"
            )

        # Make sure all sequences are the same length.
        for item in self._snp_data:
            try:
                assert len(item) == self._num_snps
            except AssertionError:
                sys.exit(
                    "There are sequences of different lengths in the "
                    "structure file\n"
                )

    def read_phylip(self) -> None:
        """
        Populates GenotypeData object by parsing a Phylip file.

        Raises:
            ValueError: All sequences must be the same length as specified in the header line.
            ValueError: Number of individuals differs from the header line.
        """
        if self.verbose:
            print(f"\nReading phylip file {self.filename}...")

        self._check_filetype("phylip")
        snp_data = list()
        with open(self.filename, "r") as fin:
            num_inds = 0
            num_snps = 0
            first = True
            for line in fin:
                line = line.strip()
                if not line:  # If blank line.
                    continue
                if first:
                    first = False
                    header = line.split()
                    num_inds = int(header[0])
                    num_snps = int(header[1])
                    continue
                cols = line.split()
                inds = cols[0]
                seqs = cols[1]
                snps = [snp for snp in seqs]  # Split each site.

                # Error handling if incorrect sequence length
                if len(snps) != num_snps:
                    raise ValueError(
                        "All sequences must be the same length; "
                        "at least one sequence differs from the header line\n"
                    )

                snp_data.append(snps)

                self.samples.append(inds)

        self._snp_data = snp_data

        if self.verbose:
            print("Done!")

        self._num_snps = num_snps
        self._num_inds = num_inds

        # Error handling if incorrect number of individuals and snps in header.
        if len(self.samples) != num_inds:
            raise ValueError(
                "Incorrect number of individuals listed in header\n"
            )

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

                self.snpsdict[inds] = snps
                snp_data.append(snps)

                self.samples.append(inds)

        if len(list(set(num_snps))) > 1:
            raise ValueError(
                "All sequences must be the same length; "
                "at least one sequence differs in length from the others\n"
            )

        self._num_snps = num_snps[0]
        self._num_inds = num_inds

        df = pd.DataFrame(snp_data)
        df.replace("NA", "-9", inplace=True)
        df = df.astype("int")

        self.original_snps = df.values.tolist()

        if self.verbose:
            print("Done!")

        self.snps = df.values.tolist()

        self.ref = None
        self.alt = None

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
            impute_mode (bool, optional): Whether or not convert_012() is called in impute mode. If True, then returns the 012-encoded genotypes and does not set the ``self.snps`` attribute. If False, it does the opposite. Defaults to False.

        Returns:
            List[List[int]], optional: 012-encoded genotypes as a 2D list of shape (n_samples, n_sites). Only returns value if ``impute_mode`` is True.
            List[int], optional: List of integers indicating bi-allelic site indexes.
            int, optional: Number of remaining valid sites.

        Warnings:
            warnings.warn: If site is monomorphic.
            warnings.warn: If site has >2 alleles.

        Todo:
            skip and impute_mode are now deprecated.
        """
        warnings.formatwarning = self._format_warning

        skip = 0
        snps_012 = list()
        new_snps = list()

        if impute_mode:
            imp_snps = list()

        for i in range(0, len(snps)):
            new_snps.append([])

        # TODO: valid_sites is now deprecated.
        valid_sites = np.ones(len(snps[0]))
        for j in range(0, len(snps[0])):
            loc = list()
            for i in range(0, len(snps)):
                if vcf:
                    loc.append(snps[i][j])
                else:
                    loc.append(snps[i][j].upper())

            num_alleles = sequence_tools.count_alleles(loc, vcf=vcf)
            if num_alleles != 2:
                # If monomorphic
                if num_alleles < 2:
                    warnings.warn(
                        f"Monomorphic site detected at SNP column {j+1}.\n"
                    )
                    """
                    ***TO-DO***: Check here if column is all-missing. What to
                    do in this case? Error out?
                    """
                    ref = sequence_tools.get_major_allele(loc, vcf=vcf)
                    ref = str(ref[0])
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
                    warnings.warn(
                        f" SNP column {j+1} had >2 alleles and was forced to "
                        f"be bi-allelic. If that is not what you want, please "
                        f"fix or remove the column and re-run.\n"
                    )
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

            # Set the ref and alt alleles for each column
            self.ref.append(ref)
            self.alt.append(alt)

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

    def _make_snpsdict(self) -> Dict[str, List[str]]:
        """
        Make a dicionary with SampleIDs as keys and a list of snps associated with the sample as the values.
        """
        snpsdict = dict()
        for ind, seq in zip(self.samples, self._snp_data):
            snpsdict[ind] = seq
        return snpsdict

    def _format_warning(
        self, message, category, filename, lineno, file=None, line=None
    ) -> str:
        """
        Set the format of warnings.warn warnings.

        For setting the format of warnings, use `warnings.formatwarning = self._format_warning`.

        Args:
            message (str): Warning message to print.
            category (str): Type of warning.
            filename (str): Name of python file where the warning was raised.
            lineno (str): Line number where warning occurred.
            file (None): Not used here.
            line (None): Not used here.

        Returns:
            str: Full warning message.
        """
        return f"{filename}:{lineno}: {category.__name__}:{message}"

    def convert_onehot(
        self,
        snp_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert input data to one-hot format.

        Args:
            snp_data (Union[np.ndarray, List[List[int]]]): Input 012-encoded data of shape (n_samples, n_SNPs).
            encodings_dict (Optional[Dict[str, int]]): Encodings to convert structure to phylip format. Defaults to None.

        Returns:
            np.ndarray: One-hot encoded data.
        """

        if self.filetype == "phylip" and encodings_dict is None:
            onehot_dict = {
                "A": [1.0, 0.0, 0.0, 0.0],
                "T": [0.0, 1.0, 0.0, 0.0],
                "G": [0.0, 0.0, 1.0, 0.0],
                "C": [0.0, 0.0, 0.0, 1.0],
                "N": [0.0, 0.0, 0.0, 0.0],
                "W": [0.5, 0.5, 0.0, 0.0],
                "R": [0.5, 0.0, 0.5, 0.0],
                "M": [0.5, 0.0, 0.0, 0.5],
                "K": [0.0, 0.5, 0.5, 0.0],
                "Y": [0.0, 0.5, 0.0, 0.5],
                "S": [0.0, 0.0, 0.5, 0.5],
                "-": [0.0, 0.0, 0.0, 0.0],
            }

        elif (
            self.filetype.startswith("structure1row")
            or self.filetype.startswith("structure2row")
            and encodings_dict is None
        ):
            onehot_dict = {
                "1/1": [1.0, 0.0, 0.0, 0.0],
                "2/2": [0.0, 1.0, 0.0, 0.0],
                "3/3": [0.0, 0.0, 1.0, 0.0],
                "4/4": [0.0, 0.0, 0.0, 1.0],
                "-9/-9": [0.0, 0.0, 0.0, 0.0],
                "1/2": [0.5, 0.5, 0.0, 0.0],
                "2/1": [0.5, 0.5, 0.0, 0.0],
                "1/3": [0.5, 0.0, 0.5, 0.0],
                "3/1": [0.5, 0.0, 0.5, 0.0],
                "1/4": [0.5, 0.0, 0.0, 0.5],
                "4/1": [0.5, 0.0, 0.0, 0.5],
                "2/3": [0.0, 0.5, 0.5, 0.0],
                "3/2": [0.0, 0.5, 0.5, 0.0],
                "2/4": [0.0, 0.5, 0.0, 0.5],
                "4/2": [0.0, 0.5, 0.0, 0.5],
                "3/4": [0.0, 0.0, 0.5, 0.5],
                "4/3": [0.0, 0.0, 0.5, 0.5],
            }

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

    def convert_int_iupac(
        self,
        snp_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """Convert input data to integer-encoded format (0-9) based on IUPAC codes.

        Args:
            snp_data (numpy.ndarray of shape (n_samples, n_SNPs) or List[List[int]]): Input 012-encoded data.

            encodings_dict (Dict[str, int] or None): Encodings to convert structure to phylip format.

        Returns:
            numpy.ndarray: One-hot encoded data.
        """

        if self.filetype == "phylip" and encodings_dict is None:
            onehot_dict = {
                "A": 0,
                "T": 1,
                "G": 2,
                "C": 3,
                "W": 4,
                "R": 5,
                "M": 6,
                "K": 7,
                "Y": 8,
                "S": 9,
                "-": -9,
                "N": -9,
            }

        elif (
            self.filetype.startswith("structure1row")
            or self.filetype.startswith("structure2row")
            and encodings_dict is None
        ):
            onehot_dict = {
                "1/1": 0,
                "2/2": 1,
                "3/3": 2,
                "4/4": 3,
                "1/2": 4,
                "2/1": 4,
                "1/3": 5,
                "3/1": 5,
                "1/4": 6,
                "4/1": 6,
                "2/3": 7,
                "3/2": 7,
                "2/4": 8,
                "4/2": 8,
                "3/4": 9,
                "4/3": 9,
                "-9/-9": -9,
            }

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

    def read_popmap(self, popmapfile: Optional[str]) -> None:
        """Read population map from file.

        Args:
            popmapfile (str): Path to population map file.

        Raises:
            ValueError: No samples were in the input file.
            ValueError: Samples missing from the popmap file.
            ValueError: Lengths of popmap file and samples differ.
        """
        self.popmapfile = popmapfile
        # Join popmap file with main object.
        if len(self.samples) < 1:
            raise ValueError("No samples in GenotypeData\n")

        # Instantiate popmap object
        my_popmap = ReadPopmap(popmapfile)

        popmap_ok = my_popmap.validate_popmap(self.samples)

        if not popmap_ok:
            raise ValueError(
                f"Not all samples are present in supplied popmap "
                f"file: {my_popmap.filename}\n"
            )

        if len(my_popmap) != len(self.samples):
            raise ValueError(
                f"The number of individuals in the popmap file "
                f"({len(my_popmap)}) differs from the number of samples "
                f"({len(self.samples)})\n"
            )

        for sample in self.samples:
            if sample in my_popmap:
                self.pops.append(my_popmap[sample])

        self._popmap = my_popmap.popmap
        self._popmap_inverse = my_popmap.popmap_flipped

    def decode_012(self, X, write_output=True, prefix="imputer", is_nuc=False):
        """Decode 012-encoded or 0-9 integer-encoded imputed data to STRUCTURE or PHYLIP format.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): 012-encoded imputed data to decode.

            write_output (bool, optional): If True, saves output to file on disk. Otherwise just makes a GenotypeData attribute. Defaults to True.

            prefix (str, optional): Prefix to append to output file. Defaults to "output".

            is_nuc (bool, optional): Whether using nucelotide encodings instead of 012 encodings. Defaults to False.

        Returns:
            str: Filename that imputed data was written to.
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

        if ft.startswith("phylip"):
            is_phylip = True
        else:
            is_phylip = False

        df_decoded = df.copy()

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
            for col, ref, alt in zip(df.columns, self.ref, self.alt):
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
                }
                dreplace[col] = d

        df_decoded.replace(dreplace, inplace=True)

        if write_output:
            outfile = os.path.join(
                f"{self.prefix}_output", "alignments", "imputed"
            )

        if ft.startswith("structure"):
            of = f"{outfile}.str"

            if ft.startswith("structure2row"):
                for col in df_decoded.columns:
                    df_decoded[col] = (
                        df_decoded[col]
                        .str.split("/")
                        .apply(lambda x: list(map(int, x)))
                    )

                df_decoded.insert(0, "sampleID", self.samples)
                df_decoded.insert(1, "popID", self.pops)

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

            if write_output:
                df_decoded.insert(0, "sampleID", self.samples)
                df_decoded.insert(1, "popID", self.pops)

                df_decoded.to_csv(
                    of,
                    sep="\t",
                    header=False,
                    index=False,
                )

        elif ft.startswith("phylip"):
            of = f"{outfile}.phy"
            header = f"{self._num_inds} {self._num_snps}\n"

            if write_output:
                with open(of, "w") as fout:
                    fout.write(header)

                lst_decoded = df_decoded.values.tolist()

                with open(of, "a") as fout:
                    for sample, row in zip(self.samples, lst_decoded):
                        seqs = "".join([str(x) for x in row])
                        fout.write(f"{sample}\t{seqs}\n")

        if write_output:
            return of
        else:
            return df_decoded.values.tolist()

    def missingness_reports(
        self,
        zoom=True,
        prefix="imputer",
        horizontal_space=0.6,
        vertical_space=0.6,
        bar_color="gray",
        heatmap_palette="magma",
        plot_format="pdf",
        dpi=300,
    ):
        """Generate missingness reports and plots.

        Function will write several comma-delimited report files:
            1) individual_missingness.csv: Missing proportions per-individual.
            2) locus_missingness.csv: Missing proportions per-locus.
            3) population_missingness.csv: Missing proportions per population (only generated if popmapfile was passed to GenotypeData).
            4) population_locus_missingness.csv: Table of per-population and per-locus missing data proportions.

        A file missingness.<plot_format> will also be saved.
        It contains the following subplots:
            1) Barplot with per-individual missing data proportions.
            2) Barplot with per-locus missing data proportions.
            3) Barplot with per-population missing data proportions (only if popmapfile was passed to GenotypeData.
            4) Heatmap showing per-population + per-locus missing data proportions (only if popmapfile was passed to GenotypeData).
            5) Stacked barplot showing missing data proportions per-individual.
            6) Stacked barplot showing missing data proportions per-population (only if popmapfile was passed to GenotypeData).

        If popmapfile was not passed to GenotypeData, then the subplots and report files that require populations are not included.

        The non-stacked bar plot colors can be adjusted, as can the heatmap palette, plot file format, and the spacing between subplots.

        Args:
            zoom (bool, optional): If True, zooms in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.

            prefix (str, optional): Prefix for output directory and files. Plots and files will be written to a directory called <prefix>_reports. The report directory will be created if it does not already exist. Defaults to 'imputer'.

            horizontal_space (float, optional): Set width spacing between subplots. If your plot are overlapping horizontally, increase horizontal_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            vertical_space (float, optioanl): Set height spacing between subplots. If your plots are overlapping vertically, increase vertical_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            bar_color (str, optional): Color of the bars on the non-stacked barplots. Can be any color supported by matplotlib. See matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, optional): Palette to use for heatmap plot. Can be any palette supported by seaborn. See seaborn documentation. Defaults to 'magma'.

            plot_format (str, optional): Format to save plots. Can be any of the following: "pdf", "png", "svg", "ps", "eps". Defaults to "pdf".

            dpi (int): The resolution in dots per inch. Defaults to 300.
        """
        params = dict(
            zoom=zoom,
            prefix=prefix,
            horizontal_space=horizontal_space,
            vertical_space=vertical_space,
            bar_color=bar_color,
            heatmap_palette=heatmap_palette,
            plot_format=plot_format,
            dpi=dpi,
        )

        df = pd.DataFrame(self.snps)
        df.replace(-9, np.nan, inplace=True)

        report_path = os.path.join(f"{self.prefix}_output", "reports")

        if prefix is not None:
            report_path = os.path.join(f"{self.prefix}_output", "reports")
        os.makedirs(report_path, exist_ok=True)

        loc, ind, poploc, poptotal, indpop = Plotting.visualize_missingness(
            self, df, **params
        )

        self._report2file(ind, report_path, "individual_missingness.csv")
        self._report2file(loc, report_path, "locus_missingness.csv")

        if self.pops is not None:
            self._report2file(
                poploc, report_path, "per_pop_and_locus_missingness.csv"
            )
            self._report2file(
                poptotal, report_path, "population_missingness.csv"
            )
            self._report2file(
                indpop,
                report_path,
                "population_locus_missingness.csv",
                header=True,
            )

    def _report2file(self, df, report_path, mypath, header=False):
        df.to_csv(
            os.path.join(report_path, mypath), header=header, index=False
        )

    def plot_allele_distribution(self, int_iupac: np.ndarray) -> None:
        plot_path = os.path.join(f"{self.prefix}_output", "plots")
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        plotting = Plotting()
        plotting.plot_gt_distribution(int_iupac, plot_path)

    def calc_missing(self, df: pd.DataFrame, use_pops: bool = True):
        # Get missing value counts per-locus.
        loc = df.isna().sum(axis=0) / self._num_inds
        loc = loc.round(2)

        # Get missing value counts per-individual.
        ind = df.isna().sum(axis=1) / self._num_snps
        ind = ind.round(2)

        poploc = None
        poptot = None
        indpop = None
        if use_pops:
            popdf = df.copy()
            popdf.index = self.pops
            misscnt = popdf.isna().groupby(level=0).sum()
            n = popdf.groupby(level=0).size()
            poploc = misscnt.div(n, axis=0).round(2).T
            poptot = misscnt.sum(axis=1) / self._num_snps
            poptot = poptot.div(n, axis=0).round(2)
            indpop = df.copy()

        return loc, ind, poploc, poptot, indpop

    class _DataFormat012:
        def __init__(self, instance, is_structure=False):
            self.instance = instance
            self.is_structure = is_structure

        def __call__(self, fmt="list"):
            if fmt == "list":
                return self.instance.convert_012(
                    self.instance.snp_data, vcf=self.is_structure
                )

            elif fmt == "numpy":
                return np.array(
                    self.instance.convert_012(
                        self.instance.snp_data, vcf=self.is_structure
                    )
                )

            elif fmt == "pandas":
                df = pd.DataFrame.from_records(
                    self.instance.convert_012(
                        self.instance.snp_data, vcf=self.is_structure
                    )
                )

            else:
                raise ValueError(
                    "Invalid format. Supported formats: 'list', 'numpy', 'pandas'"
                )

    @property
    def num_snps(self) -> int:
        """Number of snps in the dataset.

        Returns:
            int: Number of SNPs per individual.
        """
        return self._num_snps

    @num_snps.setter
    def num_snps(self, value) -> None:
        self._num_snps = value

    @property
    def num_inds(self) -> int:
        """Number of individuals in dataset.

        Returns:
            int: Number of individuals in input data.
        """
        return self._num_inds

    @num_inds.setter
    def num_inds(self, value) -> None:
        self._num_inds = value

    @property
    def populations(self) -> List[Union[str, int]]:
        """Population Ids.

        Returns:
            List[Union[str, int]]: Population IDs.
        """
        return self.pops

    @property
    def popmap(self) -> Dict[str, str]:
        """Dictionary object with SampleIDs as keys and popIDs as values."""
        return self._popmap

    @popmap.setter
    def popmap(self, value):
        """Dictionary with SampleIDs as keys and popIDs as values."""
        self._popmap = value

    @property
    def popmap_inverse(self) -> None:
        return self._popmap_inverse

    @popmap_inverse.setter
    def popmap_inverse(self, value) -> Dict[str, str]:
        self._popmap_inverse = value

    @property
    def individuals(self) -> List[str]:
        """Sample IDs in input order.

        Returns:
            List[str]: Sample IDs in input order.
        """
        return self.samples

    @individuals.setter
    def individuals(self, value) -> None:
        self.samples = value

    @property
    def snpsdict(self) -> Dict[str, List[str]]:
        """
        Dictionary with Sample IDs as keys and lists of genotypes as values.
        """
        return self._make_snpsdict()

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
            >>>gt_list = GenotypeData.genotypes_012
            >>>
            >>># Get a numpy array.
            >>>gt_array = GenotypeData.genotypes_012(fmt="numpy")
            >>>
            >>># Get a pandas DataFrame.
            >>>gt_df = GenotypeData.genotypes_012(fmt="pandas")
        """
        is_str = True if self.filetype.startswith("structure") else False
        return self._DataFormat012(self, is_structure=is_str)

    @genotypes_012.setter
    def genotypes_012(self, value) -> List[List[int]]:
        self._snp_data = self.decode_012(value, write_output=False)

    @property
    def genotypes_onehot(self) -> Union[np.ndarray, List[List[List[float]]]]:
        """One-hot encoded snps format.

        Returns:
            numpy.ndarray: One-hot encoded numpy array of shape (n_samples, n_sites).
        """
        return self.convert_onehot(self._snp_data)

    @property
    def genotypes_int(self) -> np.ndarray:
        """Integer-encoded (0-9 including IUPAC characters) snps format.

        Returns:
            numpy.ndarray: Array of shape (n_samples, n_sites), integer-encoded from 0-9 with IUPAC characters.
        """
        arr = self.convert_int_iupac(self._snp_data)
        self.plot_allele_distribution(arr)
        return arr

    @property
    def alignment(self) -> List[MultipleSeqAlignment]:
        is_str = True if self.filetype.startswith("structure") else False

        if is_str:
            snp_data = [
                [self._genotype_to_iupac(g) for g in row]
                for row in self._snp_data
            ]
        else:
            snp_data = self._snp_data

        return MultipleSeqAlignment(
            [
                SeqRecord(Seq("".join(row)), id=sample)
                for sample, row in zip(self.samples, snp_data)
            ]
        )

    @alignment.setter
    def alignment(self, value: MultipleSeqAlignment) -> None:
        """
        Setter method for the alignment.

        Args:
            value (Bio.MultipleSeqAlignment): The MultipleSeqAlignment object to set as the alignment.

        Raises:
            TypeError: If the input value is not a MultipleSeqAlignment object.
        """
        if not isinstance(value, MultipleSeqAlignment):
            raise TypeError(
                "alignment must be a MultipleSequenceAlignment object."
            )

        alignment_array = np.array([list(str(record.seq)) for record in value])

        self._snp_data = alignment_array.tolist()
        self.num_inds = len(self._snp_data)
        self.num_snps = len(self._snp_data[0])

    def _genotype_to_iupac(self, genotype):
        """Converts a genotype string to its corresponding IUPAC code.

        Args:
            genotype (str): A string containing genotype information in the format of "x/y", where x and y are integers that represent an allele.

        Returns:
            str: The corresponding IUPAC code for the input genotype. Returns 'N' if the genotype is not in the lookup dictionary.

        Raises:
            None.
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


def merge_alleles(
    first: List[Union[str, int]],
    second: Optional[List[Union[str, int]]] = None,
) -> List[str]:
    """Merges first and second alleles in structure file.

    Args:
        first (List[Union[str, int] or None): Alleles on first line.
        second (List[Union[str, int]] or None, optional): Second row of alleles. Defaults to None.

    Returns:
        List[str]: VCF file-style genotypes (i.e. split by "/").

    Raises:
        ValueError: First and second lines have differing lengths.
        ValueError: Line has non-even number of alleles.
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
