import copy
import random
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pysam
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from snpio.utils.logging import setup_logger  # type: ignore
from snpio.plotting.plotting import Plotting as Plotting
from snpio.read_input.popmap_file import ReadPopmap
from snpio.utils.custom_exceptions import (
    UnsupportedFileTypeError,
    SequenceLengthError,
    NoValidAllelesError,
)
from snpio.utils.misc import class_performance_decorator, get_gt2iupac, get_iupac2gt

logger = setup_logger(__name__)

# Make sure python version is >= 3.8
if sys.version_info < (3, 8):
    raise ImportError("Python < 3.8 is not supported!")

warnings.simplefilter(action="ignore", category=FutureWarning)


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

        If using PHYLIP or STRUCTURE formats, all sites will be forced to be biallelic. If you need >2 alleles, you must input a VCF file.

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

        loci_indices (List[int]): Column indices for retained loci in filtered alignment.

        sample_indices (List[int]): Row indices for retained samples in the alignment.

        ref (List[str]): List of reference alleles of length num_snps.

        alt (List[str]): List of alternate alleles of length num_snps.

    Methods:

        read_012: Read data from a custom 012-encoded file format.
        read_popmap: Read in a popmap file.
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
        self._snp_data = []

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
            raise TypeError("qmatrix and qmatrix_iqtree cannot both be provided.")

        if self.siterates is not None and self.siterates_iqtree is not None:
            raise TypeError("siterates and siterates_iqtree cannot both be defined")

        self._loci_indices = kwargs.get("loci_indices", None)
        self.sample_indices = kwargs.get("sample_indices", None)

        if self.filetype not in self.supported_filetypes:
            raise UnsupportedFileTypeError(
                self.filetype, supported_types=self.supported_filetypes
            )

        if self.loci_indices is None:
            self._loci_indices = list(range(self.num_snps))

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
                msg = "Found the following populations:\nPopulation\tCount\n"
                logger.info(msg)
            self._my_popmap.get_pop_counts(plot_dir_prefix=self.prefix)

        self._kwargs["filetype"] = self.filetype
        self._kwargs["loci_indices"] = self.loci_indices
        self._kwargs["sample_indices"] = self.sample_indices

        vcf_attr_path = Path(
            f"{self.prefix}_output",
            "gtdata",
            "alignments",
            "vcf",
            "vcf_attributes.h5",
        )
        if vcf_attr_path.is_file():
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

        self.iupac_mapping = self._iupac_from_gt_tuples()
        self.reverse_iupac_mapping = {v: k for k, v in self.iupac_mapping.items()}

    def _iupac_from_gt_tuples(self) -> Dict[Tuple[str, str], str]:
        """Returns the IUPAC code mapping."""
        return {
            ("A", "A"): "A",
            ("C", "C"): "C",
            ("G", "G"): "G",
            ("T", "T"): "T",
            ("A", "G"): "R",
            ("C", "T"): "Y",
            ("G", "C"): "S",
            ("A", "T"): "W",
            ("G", "T"): "K",
            ("A", "C"): "M",
            ("C", "G"): "S",
            ("A", "C"): "M",
            ("N", "N"): "N",
        }

    def get_reverse_iupac_mapping(self) -> Dict[str, Tuple[str, str]]:
        """Creates a reverse mapping from IUPAC codes to allele tuples."""
        return self.reverse_iupac_mapping

    def _get_ref_alt_alleles(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Determine the most common, second most common, and less common alleles in each column of a 2D numpy array, excluding 'N' and '-' alleles. The reference allele is determined by frequency and by the fewest number of heterozygous genotypes. If tied, a random allele is selected.

        Args:
            data (np.ndarray): A 2D numpy array where each column represents different SNP data.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
                - Most common alleles (likely ref)
                - Second most common alleles (likely alt)
                - Less common alleles (for potential multi-allelic sites)
        """
        # Initialize arrays to hold results
        most_common_alleles = np.full(data.shape[1], None, dtype=object)
        second_most_common_alleles = np.full(data.shape[1], None, dtype=object)
        less_common_alleles_list = []

        for i in range(data.shape[1]):
            column = data[:, i]

            # Flatten alleles and remove 'N' and '-'
            valid_alleles = []
            heterozygous_counts = {}
            for genotype in column:
                if genotype not in ["N", "-"]:
                    # Split heterozygous genotypes (e.g., 'A/G') and add each allele separately
                    alleles = genotype.split("/")
                    valid_alleles.extend(alleles)
                    # Count heterozygous genotypes for each allele
                    if len(alleles) == 2:
                        for allele in alleles:
                            heterozygous_counts[allele] = (
                                heterozygous_counts.get(allele, 0) + 1
                            )

            # Convert valid_alleles to a numpy array
            valid_alleles = np.array(valid_alleles)

            if valid_alleles.size == 0:
                # If no valid alleles, log an error and raise an exception
                logger.error(f"No valid alleles found in column {i}")
                raise NoValidAllelesError(i)

            # Use numpy's unique function with return_counts for counting
            alleles, counts = np.unique(valid_alleles, return_counts=True)

            # Sort by counts (descending order)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_alleles = alleles[sorted_indices]
            sorted_counts = counts[sorted_indices]

            # Warning for low allele counts or borderline cases
            if sorted_counts[0] <= 2:
                logger.warning(
                    f"Low allele count in column {i}: {sorted_alleles[0]} occurs only {sorted_counts[0]} times."
                )

            # If the top two alleles have the same count, choose by fewest heterozygous occurrences or randomly
            if len(sorted_alleles) > 1 and sorted_counts[0] == sorted_counts[1]:
                top_alleles = [sorted_alleles[0], sorted_alleles[1]]
                heterozygous_top_counts = [
                    heterozygous_counts.get(allele, 0) for allele in top_alleles
                ]
                if heterozygous_top_counts[0] == heterozygous_top_counts[1]:
                    # Randomly choose the reference allele if heterozygous counts are tied
                    chosen_ref_index = random.choice([0, 1])
                    most_common_alleles[i] = top_alleles[chosen_ref_index]
                    second_most_common_alleles[i] = top_alleles[1 - chosen_ref_index]
                else:
                    # Choose by fewest heterozygous occurrences
                    chosen_ref_index = np.argmin(heterozygous_top_counts)
                    most_common_alleles[i] = top_alleles[chosen_ref_index]
                    second_most_common_alleles[i] = top_alleles[1 - chosen_ref_index]
            else:
                # Assign most common and second most common alleles
                if len(sorted_alleles) > 0:
                    most_common_alleles[i] = sorted_alleles[0]
                if len(sorted_alleles) > 1:
                    second_most_common_alleles[i] = sorted_alleles[1]

            # Less common alleles are those beyond the second most common
            less_common_alleles = (
                sorted_alleles[2:] if len(sorted_alleles) > 2 else np.array([])
            )
            less_common_alleles_list.append(less_common_alleles)

        return most_common_alleles, second_most_common_alleles, less_common_alleles_list

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

    def _make_snpsdict(
        self,
        samples: Optional[List[str]] = None,
        snp_data: Optional[List[List[str]]] = None,
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

            self.samples = new_samples
            self._populations = new_populations

        self._popmap = my_popmap.popmap
        self._popmap_inverse = my_popmap.popmap_flipped
        self.sample_indices = my_popmap.sample_indices

        if return_indices:
            return self.sample_indices

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

        report_path = Path(f"{plot_dir_prefix}_output", "gtdata", "reports")

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

        self._report2file(loc, str(report_path), fname)

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
        df.to_csv(Path(report_path, mypath), header=header, index=False)

    def _genotype_to_iupac(self, genotype: str) -> str:
        """
        Convert a genotype string to its corresponding IUPAC code.

        Args:
            genotype (str): Genotype string in the format "x/y".

        Returns:
            str: Corresponding IUPAC code for the input genotype. Returns 'N' if the genotype is not in the lookup dictionary.
        """
        iupac_dict = get_gt2iupac()

        gt = iupac_dict.get(genotype)

        if gt is None:
            msg = f"Invalid Genotype: {genotype}"
            logger.error(msg)
            raise ValueError(msg)
        return gt

    def _iupac_to_genotype(self, iupac_code: str) -> str:
        """
        Convert an IUPAC code to its corresponding genotype string.

        Args:
            iupac_code (str): IUPAC code.

        Returns:
            str: Corresponding genotype string for the input IUPAC code. Returns '-9/-9' if the IUPAC code is not in the lookup dictionary.
        """
        genotype_dict = get_iupac2gt()

        gt = genotype_dict.get(iupac_code)
        if gt is None:
            msg = f"Invalid IUPAC Code: {iupac_code}"
            logger.error(msg)
            raise ValueError(msg)
        return gt

    def calc_missing(self, df: pd.DataFrame, use_pops: bool = True) -> Tuple[
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
        if self.snp_data.size > 0:
            return len(self.snp_data[0])
        return 0

    @property
    def num_inds(self) -> int:
        """Number of individuals in dataset.

        Returns:
            int: Number of individuals in input data.
        """
        if self.snp_data.size > 0:
            return len(self.snp_data)
        return 0

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
        if isinstance(self._snp_data, list):
            return np.array(self._snp_data)
        return self._snp_data

    @snp_data.setter
    def snp_data(self, value) -> None:
        """Set snp_data. Input can be a 2D list, numpy array, pandas DataFrame, or MultipleSeqAlignment object."""
        if not isinstance(value, np.ndarray):
            if isinstance(value, list):
                value = np.array(value)
            elif isinstance(value, pd.DataFrame):
                value = value.to_numpy()
            elif isinstance(value, MultipleSeqAlignment):
                value = [list(str(record.seq)) for record in value]
                value = np.array(value)
            else:
                msg = f"snp_data must be a list, numpy.ndarray, pandas.DataFrame, or Bio.MultipleSeqAlignment, but got {type(value)}"
                logger.error(msg)
                raise TypeError(msg)
        self._snp_data = value
        self._validate_seq_lengths()

    def _validate_seq_lengths(self):
        """Ensure that all SNP data rows have the same length."""
        lengths = {len(row) for row in self.snp_data}
        if len(lengths) > 1:
            n_snps = len(self.snp_data[0])
            for i, row in enumerate(self.snp_data):
                if len(row) != n_snps:
                    raise SequenceLengthError(self.samples[i])

    @property
    def alignment(self) -> List[MultipleSeqAlignment]:
        """Get alignment as a biopython MultipleSeqAlignment object.

        This is good for printing and visualizing the alignment. If you want the alignment as a 2D list object, then use the ``snp_data`` property instead.
        """
        return MultipleSeqAlignment(
            [
                SeqRecord(Seq("".join(row)), id=sample)
                for sample, row in zip(self.samples, self.snp_data)
            ]
        )

    @alignment.setter
    def alignment(
        self, value: Union[np.ndarray, pd.DataFrame, MultipleSeqAlignment, list, Any]
    ) -> None:
        """
        Setter method for the alignment.

        Args:
            value (Bio.MultipleSeqAlignment, list, np.ndarray, pd.DataFrame): The MultipleSeqAlignment object to set as the alignment.

        Raises:
            TypeError: If the input value is not a MultipleSeqAlignment object, list, numpy array, or pandas DataFrame.
        """
        if isinstance(value, MultipleSeqAlignment):
            alignment_array = np.array([list(str(record.seq)) for record in value])
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

        self.snp_data = alignment_array

    @property
    def loci_indices(self) -> List[int]:
        """Column indices for retained loci in filtered alignment."""
        return self._loci_indices

    @loci_indices.setter
    def loci_indices(self, value) -> None:
        """Column indices for retained loci in filtered alignment."""
        self._loci_indices = value

    @property
    def sample_indices(self) -> np.ndarray:
        """Row indices for retained samples in alignemnt."""
        if not isinstance(self._sample_indices, np.ndarray):
            if self._sample_indices is None or not self._sample_indices:
                self._sample_indices = np.ones_like(self.samples, dtype=bool)
            else:
                if not isinstance(self._sample_indcies, np.ndarray):
                    self._sample_indices = np.array(self._sample_indices)
                if not self._sample_indices.dtype is np.dtype(bool):
                    msg = f"'sample_indices' must by np.dtype 'bool', but got: {self._sample_indices.dtype}"
                    logger.error(msg)
                    raise TypeError(msg)

        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, value):
        if value is None:
            value = np.ones_like(self.samples, dtype=bool)
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if not value.dtype is np.dtype(bool):
            msg = f"Attempt to set 'sample_indices' to an unexpected np.dtype. Expected 'bool', but got: {value.dtype}"
            logger.error(msg)
            raise TypeError(msg)
        self._sample_indices = value

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
