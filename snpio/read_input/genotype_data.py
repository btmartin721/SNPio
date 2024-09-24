import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pysam

from snpio.plotting.plotting import Plotting
from snpio.read_input.genotype_data_base import BaseGenotypeData
from snpio.read_input.popmap_file import ReadPopmap
from snpio.utils.benchmarking import class_performance_decorator, Benchmark
from snpio.utils.custom_exceptions import SequenceLengthError, UnsupportedFileTypeError
from snpio.utils.logging import setup_logger
from snpio.utils.misc import get_gt2iupac, get_iupac2gt


@class_performance_decorator(measure=False)
class GenotypeData(BaseGenotypeData):
    """A class for handling and analyzing genotype data.

    The GenotypeData class provides methods to read, manipulate, and analyze genotype data in various formats, including VCF, Structure, and other custom formats. It allows for data preprocessing, allele encoding, and various data transformations.

    Notes:
        GenotypeData handles the following characters as missing data:
            - 'N'
            - '-'
            - '?'
            - '.'

        If using PHYLIP or STRUCTURE formats, all sites will be forced to be biallelic. If you need multiple alleles, you must use a VCF file.

        Please keep these things in mind when using GenotypeData.


    Args:
        filename (str or None): Path to input file containing genotypes. Defaults to None.

        filetype (str or None): Type of input genotype file. Possible values include: 'phylip', 'structure', 'vcf', or '012'. Defaults to None.

        popmapfile (str or None): Path to population map file. If supplied and filetype is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column. Defaults to None.

        force_popmap (bool): If True, then samples not present in the popmap file will be excluded from the alignment. If False, then an error is raised if samples are present in the popmap file that are not present in the alignment. Defaults to False.

        exclude_pops (List[str] or None): List of population IDs to exclude from the alignment. Defaults to None.

        include_pops (List[str] or None): List of population IDs to include in the alignment. Populations not present in the include_pops list will be excluded. Defaults to None.

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

        loci_indices (List[int]): Column indices for retained loci in filtered alignment.

        sample_indices (List[int]): Row indices for retained samples in the alignment.

        ref (List[str]): List of reference alleles of length num_snps.

        alt (List[str]): List of alternate alleles of length num_snps.

    Methods:
        read_popmap: Read in a popmap file.

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
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        filetype: Optional[str] = None,
        popmapfile: Optional[str] = None,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        plot_format: Optional[str] = "png",
        plot_fontsize: int = 12,
        plot_dpi: int = 300,
        plot_despine: bool = True,
        show_plots: bool = False,
        prefix: str = "snpio",
        verbose: bool = True,
        loci_indices: Optional[np.ndarray] = None,
        sample_indices: Optional[np.ndarray] = None,
        chunk_size: int = 1000,
        logger=None,
        debug: bool = False,
        benchmark: bool = False,
    ) -> None:
        """
        Initialize the GenotypeData object.

        Args:
            filename (str, optional): Path to input file containing genotypes. Defaults to None.

            filetype (str, optional): Type of input genotype file. Possible values include: 'phylip', 'structure', 'vcf', or '012'. Defaults to None.

            popmapfile (str, optional): Path to population map file. If supplied and filetype is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column. Defaults to None.

            force_popmap (bool, optional): If True, then samples not present in the popmap file will be excluded from the alignment. If False, then an error is raised if samples are present in the popmap file that are not present in the alignment. Defaults to False.

            exclude_pops (List[str], optional): List of population IDs to exclude from the alignment. Defaults to None.

            include_pops (List[str], optional): List of population IDs to include in the alignment. Populations not present in the include_pops list will be excluded. Defaults to None.

            plot_format (str, optional): Format to save report plots. Valid options include: 'pdf', 'svg', 'png', and 'jpeg'. Defaults to 'png'.

            plot_fontsize (int, optional): Font size for plots. Defaults to 12.

            plot_dpi (int, optional): Resolution in dots per inch for plots. Defaults to 300.

            plot_despine (bool, optional): If True, remove the top and right spines from plots. Defaults to True.

            show_plots (bool, optional): If True, display plots in the console. Defaults to False.

            prefix (str, optional): Prefix to use for output directory. Defaults to "gtdata".

            verbose (bool, optional): If True, display verbose output. Defaults to True.


            loci_indices (np.ndarray, optional): Column indices for retained loci in filtered alignment. Defaults to None.

            sample_indices (np.ndarray, optional): Row indices for retained samples in the alignment. Defaults to None.

            chunk_size (int, optional): Chunk size for reading in large files. Defaults to 1000.

            logger (logging, optional): Logger object. Defaults to None.

            debug (bool, optional): If True, display debug messages. Defaults to False.

            benchmark (bool, optional): If True, benchmark the class methods. Defaults to False.
        """
        filetype = filetype.lower()
        super().__init__(filename, filetype)

        self.filename = filename
        self.filetype = filetype
        self.popmapfile = popmapfile
        self.force_popmap = force_popmap
        self.exclude_pops = exclude_pops
        self.include_pops = include_pops
        self.plot_format = plot_format
        self.plot_fontsize = plot_fontsize
        self.plot_dpi = plot_dpi
        self.plot_despine = plot_despine
        self.show_plots = show_plots
        self.prefix = prefix
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.supported_filetypes = ["vcf", "phylip", "structure"]
        self._snp_data = None
        self.logger = logger
        self.debug = debug
        self.benchmark = benchmark

        if self.logger is None:
            log_file = Path(f"{prefix}_output", "logs", "snpio.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            level = "DEBUG" if self.debug else "INFO"
            self.logger = setup_logger(__name__, log_file=log_file, level=level)

        self.kwargs = {
            "filename": filename,
            "filetype": filetype,
            "popmapfile": popmapfile,
            "force_popmap": force_popmap,
            "exclude_pops": exclude_pops,
            "include_pops": include_pops,
            "plot_format": plot_format,
            "plot_fontsize": plot_fontsize,
            "plot_dpi": plot_dpi,
            "plot_despine": plot_despine,
            "show_plots": show_plots,
            "prefix": prefix,
            "verbose": verbose,
            "logger": logger,
            "debug": debug,
            "benchmark": benchmark,
        }

        self.plot_kwargs = {
            "plot_format": self.plot_format,
            "plot_fontsize": self.plot_fontsize,
            "dpi": self.plot_dpi,
            "despine": self.plot_despine,
            "show": self.show_plots,
            "verbose": self.verbose,
            "debug": self.debug,
        }

        self._samples: List[str] = []
        self._populations: List[Union[str, int]] = []
        self._ref: List[str] = []
        self._alt: List[str] = []
        self._popmap: Dict[str, Union[str, int]] = None
        self._popmap_inverse: Dict[str, List[str]] = None

        self.loci_indices = loci_indices
        self.sample_indices = sample_indices

        if self.filetype not in self.supported_filetypes:
            msg = f"Unsupported filetype provided to GenotypeData: {self.filetype}"
            self.logger.error(msg)
            raise UnsupportedFileTypeError(
                self.filetype, supported_types=self.supported_filetypes
            )

        if self.popmapfile is not None:
            self._my_popmap = self.read_popmap(popmapfile)

            self.subset_with_popmap(
                self._my_popmap,
                self.samples,
                force=self.force_popmap,
                include_pops=self.include_pops,
                exclude_pops=self.exclude_pops,
            )

            self._my_popmap.get_pop_counts(self)

        self.kwargs["filetype"] = self.filetype
        self.kwargs["loci_indices"] = self.loci_indices
        self.kwargs["sample_indices"] = self.sample_indices

        self.iupac_mapping: Dict[Tuple[str, str], str] = self._iupac_from_gt_tuples()

        self.reverse_iupac_mapping: Dict[str, Tuple[str, str]] = {
            v: k for k, v in self.iupac_mapping.items()
        }

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

    def _make_snpsdict(
        self,
        samples: Optional[List[str]] = None,
        snp_data: Optional[List[List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Make a dictionary with SampleIDs as keys and a list of SNPs associated with the sample as the values.

        Args:
            samples (List[str], optional): List of sample IDs. If not provided, uses self.samples.

            snp_data (np.ndarray, optional): 2D list of genotypes. If not provided, uses self.snp_data.

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
        my_popmap = ReadPopmap(popmapfile, self.logger, verbose=self.verbose)
        return my_popmap

    def subset_with_popmap(
        self,
        my_popmap: ReadPopmap,
        samples: List[str],
        force: bool,
        include_pops: Optional[List[str]] = None,
        exclude_pops: Optional[List[str]] = None,
        return_indices: bool = False,
    ):
        """Subset popmap and samples based on population criteria.

        Args:
            my_popmap (ReadPopmap): ReadPopmap instance.

            samples (List[str]): List of sample IDs.

            force (bool): If True, force the subsetting. If False, raise an error if the samples don't align.

            include_pops (Optional[List[str]]): List of populations to include. If provided, only samples belonging to these populations will be retained.

            exclude_pops (Optional[List[str]]): List of populations to exclude. If provided, samples belonging to these populations will be excluded.

            return_indices (bool, optional): If True, return the indices for samples. Defaults to False.

        Returns:
            Optional[np.ndarray]: Boolean array of `sample_indices` if return_indices is True.
        """
        # Validate popmap with current samples
        popmap_ok = my_popmap.validate_popmap(samples, force=force)

        if not popmap_ok:
            msg = "Popmap validation failed. Check the popmap file and try again."
            self.logger.error(msg)
            raise ValueError(msg)

        # Subset the popmap based on inclusion/exclusion criteria
        my_popmap.subset_popmap(samples, include_pops, exclude_pops)

        # Update the sample list and the populations
        new_samples = [s for s in samples if s in my_popmap.popmap]
        if len(new_samples) != len(samples):
            self.logger.warning(
                "Some samples in the alignment are absent from the popmap."
            )

        new_populations = [my_popmap.popmap[s] for s in new_samples]

        if not new_samples:
            msg = "No valid samples found after popmap subsetting."
            self.logger.error(msg)
            raise ValueError(msg)

        # Update the sample indices as a boolean array
        self.sample_indices = np.isin(self.samples, new_samples)

        # Update samples and populations based on the subset
        self.samples = new_samples
        self._populations = new_populations

        # Ensure the snp_data is filtered by the subset of samples
        self.snp_data = self.snp_data[self.sample_indices, :]

        # Update popmap and inverse popmap
        self._popmap = my_popmap.popmap
        self._popmap_inverse = my_popmap.popmap_flipped

        # Return indices if requested
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
            msg = "'samples attribute is undefined."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not self.populations or self.populations is None:
            msg = "'populations' attribute is undefined."
            self.logger.error(msg)
            raise AttributeError(msg)

        with open(filename, "w") as fout:
            for s, p in zip(self.samples, self.populations):
                fout.write(f"{s}\t{p}\n")

    def missingness_reports(
        self, prefix: Optional[str] = None, zoom: bool = True, bar_color: str = "gray"
    ) -> None:
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
            prefix (str, optional): Output file prefix for the missingness report. Defaults to None.

            zoom (bool, optional): If True, zoom in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.

            bar_color (str, optional): Color of the bars on the non-stacked bar plots. Can be any color supported by matplotlib. See the matplotlib.pyplot.colors documentation. Defaults to 'gray'.
        """
        params = dict(
            zoom=zoom,
            horizontal_space=0.6,
            vertical_space=0.6,
            bar_color=bar_color,
            heatmap_palette="magma",
        )

        df = pd.DataFrame(self.snp_data)
        df = df.replace(
            ["N", "-", ".", "?"],
            [np.nan, np.nan, np.nan, np.nan],
        )

        report_path = Path(f"{self.prefix}_output", "gtdata", "reports")
        report_path = report_path / "individual_missingness.csv"
        Path(report_path).parent.mkdir(exist_ok=True, parents=True)

        kwargs = self.plot_kwargs
        prefix = prefix if prefix is not None else self.prefix
        kwargs["prefix"] = prefix
        plotting = Plotting(self, **kwargs)

        loc, ind, poploc, poptotal, indpop = plotting.visualize_missingness(
            df, prefix=prefix, **params
        )

        self._report2file(ind, report_path)
        self._report2file(loc, report_path.with_name("locus_missingness.csv"))

        if self._populations is not None:
            self._report2file(
                poploc, report_path.with_name("population_locus_missingness.csv")
            )

            self._report2file(
                poptotal, report_path.with_name("population_missingness.csv")
            )

            self._report2file(
                indpop,
                report_path.with_name("population_locus_missingness.csv"),
                header=True,
            )

    def _report2file(
        self, df: pd.DataFrame, report_path: str, header: bool = False
    ) -> None:
        """
        Write a DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): DataFrame to be written to the file.

            report_path (str): Path to the report directory.

            header (bool, optional): Whether to include the header row in the file. Defaults to False.
        """
        df.to_csv(report_path, header=header, index=False)

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
            self.logger.error(msg)
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
            self.logger.error(msg)
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
        """Create a deep copy of the GenotypeData or VCFReader object.

        Returns:
            GenotypeData or VCFReader: A new object with the same attributes as the original.
        """
        # Determine the class type of the current object
        new_obj = self.__class__.__new__(self.__class__)

        # Shallow copy of the original object's __dict__
        new_obj.__dict__.update(self.__dict__)

        # Deep copy all attributes EXCEPT the problematic VariantHeader
        for name, attr in self.__dict__.items():
            if name not in {"vcf_header", "_vcf_attributes_fn"}:
                setattr(new_obj, name, copy.deepcopy(attr))

        # Handle VCF-specific attributes if they exist
        if hasattr(self, "vcf_header") and self.vcf_header:
            new_header = pysam.VariantHeader()
            new_header = self.vcf_header.copy()
            new_obj.vcf_header = new_header

        if hasattr(self, "_vcf_attributes_fn"):
            # Copy the VCF attributes file path as a Path object
            new_obj._vcf_attributes_fn = Path(self._vcf_attributes_fn)

        return new_obj

    @property
    def inputs(self) -> Dict[str, Any]:
        """Get GenotypeData keyword arguments as a dictionary."""
        return self.kwargs

    @inputs.setter
    def inputs(self, value: Dict[str, Any]) -> None:
        """Setter method for class keyword arguments."""
        self.kwargs = value

    @property
    def num_snps(self) -> int:
        """Number of snps in the dataset.

        Returns:
            int: Number of SNPs per individual.
        """
        if self.snp_data.size > 0:
            return len(self.snp_data[0])
        return 0

    @num_snps.setter
    def num_snps(self, value: int) -> None:
        """Set the number of SNPs in the dataset."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except ValueError:
                f"num_snps must be numeric, but got {type(value)}"
                self.logger.error(msg)
                raise TypeError(msg)
        if value < 0:
            msg = f"num_snps must be a positive integer, but got {value}"
            self.logger.error(msg)
            raise ValueError(msg)

        self._num_snps = value

    @property
    def num_inds(self) -> int:
        """Number of individuals in dataset.

        Returns:
            int: Number of individuals in input data.
        """
        if self.snp_data.size > 0:
            return len(self.snp_data)
        return 0

    @num_inds.setter
    def num_inds(self, value: int) -> None:
        """Set the number of individuals in the dataset."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except ValueError:
                f"num_inds must be numeric, but got {type(value)}"
                self.logger.error(msg)
                raise TypeError(msg)
        if value < 0:
            msg = f"num_inds must be a positive integer, but got {value}"
            self.logger.error(msg)
            raise ValueError(msg)

        self._num_inds = value

    @property
    def populations(self) -> List[Union[str, int]]:
        """Population Ids.

        Returns:
            List[Union[str, int]]: Population IDs.
        """
        return self._populations

    @populations.setter
    def populations(self, value: List[Union[str, int]]) -> None:
        """Set the population IDs."""
        self._populations = value

    @property
    def popmap(self) -> Dict[str, str]:
        """Dictionary object with SampleIDs as keys and popIDs as values."""
        return self._popmap

    @popmap.setter
    def popmap(self, value: Dict[str, str]) -> None:
        """Dictionary with SampleIDs as keys and popIDs as values."""
        if not isinstance(value, dict):
            raise TypeError(
                f"popmap must be a dictionary object, but got {type(value)}."
            )

        if not all(isinstance(v, (str, int)) for v in value.values()):
            raise TypeError(f"popmap values must be strings or integers")
        self._popmap = value

    @property
    def popmap_inverse(self) -> Dict[str, List[str]]:
        """Inverse popmap dictionary with populationIDs as keys and lists of sampleIDs as values."""
        return self._popmap_inverse

    @popmap_inverse.setter
    def popmap_inverse(self, value: Dict[str, List[str]]) -> None:
        """Setter for popmap_inverse. Should have populationIDs as keys and lists of corresponding sampleIDs as values."""
        if not isinstance(value, dict):
            msg = f"'popmap_inverse' must be a dict object: {type(value)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if not all(isinstance(v, list) for v in value.values()):
            msg = f"'popmap_inverse' values must be lists of sampleIDs per populationID key, but got: {[type(v) for v in value.values()]}"
            self.logger.error(msg)
            raise TypeError(msg)

        self._popmap_inverse = value

    @property
    def samples(self) -> List[str]:
        """Sample IDs in input order.

        Returns:
            List[str]: Sample IDs in input order.
        """
        return self._samples

    @samples.setter
    def samples(self, value: List[str]) -> None:
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
    def snpsdict(self, value: Dict[str, List[str]]):
        """Set snpsdict object, which is a dictionary with sample IDs as keys and lists of genotypes as values."""
        self._snpsdict = value

    @property
    def loci_indices(self) -> np.ndarray:
        """Column indices for retained loci in filtered alignment."""

        if not isinstance(self._loci_indices, (np.ndarray, list)):
            msg = f"'loci_indices' set to invalid type. Expected numpy.ndarray or list, but got: {type(self._loci_indices)}"
            self.logger.error(msg)
            raise TypeError(msg)

        elif isinstance(self._loci_indices, list):
            self._loci_indices = np.array(self._loci_indices)

        if self._loci_indices is None or not self._loci_indices.size > 0:
            self._loci_indices = np.ones(self.num_snps, dtype=bool)

        else:
            if not self._loci_indices.dtype is np.dtype(bool):
                msg = f"'loci_indices' must be numpy.dtype 'bool', but got: {self._loci_indices.dtype}"
                self.logger.error(msg)
                raise TypeError(msg)

        return self._loci_indices

    @loci_indices.setter
    def loci_indices(self, value) -> None:
        """Column indices for retained loci in filtered alignment."""
        if value is None:
            value = np.ones(self.num_snps, dtype=bool)
        if isinstance(value, list):
            value = np.array(value)
        if not value.dtype is np.dtype(bool):
            msg = f"Attempt to set 'sample_indices' to an unexpected np.dtype. Expected 'bool', but got: {value.dtype}"
            self.logger.error(msg)
            raise TypeError(msg)
        self._loci_indices = value

    @property
    def sample_indices(self) -> np.ndarray:
        """Row indices for retained samples in alignemnt."""
        if not isinstance(self._sample_indices, (np.ndarray, list)):
            msg = f"'sample_indices' set to invalid type. Expected numpy.ndarray or list, but got: {type(self._sample_indices)}"
            self.logger.error(msg)
            raise TypeError(msg)

        elif isinstance(self._sample_indices, list):
            self._sample_indices = np.array(self._sample_indices)

        if self._sample_indices is None or not self._sample_indices.size > 0:
            self._sample_indices = np.ones_like(self.samples, dtype=bool)

        else:
            if not self._sample_indices.dtype is np.dtype(bool):
                msg = f"'sample_indices' must be numpy.dtype 'bool', but got: {self._sample_indices.dtype}"
                self.logger.error(msg)
                raise TypeError(msg)

        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, value) -> None:
        if value is None:
            value = np.ones_like(self.samples, dtype=bool)
        if isinstance(value, list):
            value = np.array(value)
        if not value.dtype is np.dtype(bool):
            msg = f"Attempt to set 'sample_indices' to an unexpected np.dtype. Expected 'bool', but got: {value.dtype}"
            self.logger.error(msg)
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
    def alt(self, value: List[str]) -> None:
        """Setter for list of alternate alleles of length num_snps."""
        self._alt = value

    @property
    def snp_data(self) -> np.ndarray:
        """Get the genotypes as a 2D list of shape (n_samples, n_loci)."""
        is_array = True if isinstance(self._snp_data, np.ndarray) else False

        load = True if is_array and not self._snp_data.size > 0 else False
        load = (
            True
            if not is_array and isinstance(self._snp_data, list) and not self._snp_data
            else False
        )
        load = True if self._snp_data is None else False

        if load:
            self.load_aln()
            self.logger.debug(f"snp_data: {self.snp_data}")
            self.logger.debug(f"snp_data shape: {self.snp_data.shape}")

        if isinstance(self._snp_data, (np.ndarray, pd.DataFrame, list)):
            if isinstance(self._snp_data, list):
                return np.array(self._snp_data)
            elif isinstance(self._snp_data, pd.DataFrame):
                return self._snp_data.to_numpy()
            return self._snp_data  # is numpy.ndarray
        else:
            msg = f"Invalid 'snp_data' type. Expected numpy.ndarray, pandas.DataFrame, or list, but got: {type(self._snp_data)}"
            self.logger.error(msg)
            raise TypeError(msg)

    @snp_data.setter
    def snp_data(self, value) -> None:
        """Set snp_data. Input can be a 2D list, numpy array, or pandas DataFrame object."""
        if isinstance(value, (np.ndarray, pd.DataFrame, list)):
            if isinstance(value, list):
                value = np.array(value)
            elif isinstance(value, pd.DataFrame):
                value = value.to_numpy()
            self._snp_data = value
            self._validate_seq_lengths()
        else:
            msg = f"Attempt to set 'snp_data' to invalid type. Must be a list, numpy.ndarray, pandas.DataFrame, but got: {type(value)}"
            self.logger.debug(value)
            self.logger.error(msg)
            raise TypeError(msg)

    def _validate_seq_lengths(self):
        """Ensure that all SNP data rows have the same length."""
        lengths = {len(row) for row in self.snp_data}
        if len(lengths) > 1:
            n_snps = len(self.snp_data[0])
            for i, row in enumerate(self.snp_data):
                if len(row) != n_snps:
                    msg = f"Invalid sequence length for Sample {self.samples[i]}. Expected {n_snps}, but got: {len(row)}"
                    self.logger.error(msg)
                    raise SequenceLengthError(self.samples[i])

    def set_alignment(
        self,
        snp_data: np.ndarray,
        samples: List[str],
        sample_indices: np.ndarray,
        loci_indices: np.ndarray,
    ) -> None:
        """Set the alignment data and sample IDs.

        Args:
            snp_data (np.ndarray): 2D array of genotype data.

            samples (List[str]): List of sample IDs.
        """
        self.snp_data = snp_data
        self.samples = samples
        self.populations = [self.popmap[s] for s in samples]
        self.popmap = {s: self.popmap[s] for s in samples}
        self.popmap_inverse = {
            p: [s for s in samp if s in samples]
            for p, samp in self.popmap_inverse.items()
        }

        self.sample_indices = sample_indices
        self.loci_indices = loci_indices
        self.num_inds = np.count_nonzero(self.sample_indices)
        self.num_snps = np.count_nonzero(self.loci_indices)

        idx = np.where(self.loci_indices)[0].tolist()
        self.ref = [i for j, i in enumerate(self.ref) if j in idx]
        self.alt = [i for j, i in enumerate(self.alt) if j in idx]

        if self.filetype == "vcf":
            self.update_vcf_attributes(
                self.snp_data, self.sample_indices, self.loci_indices, self.samples
            )

    def __len__(self):
        """Return the number of individuals in the dataset."""
        return self.num_inds

    def __getitem__(self, index):
        """Return the genotype data for a specific individual."""
        return self.snp_data[index]

    def __iter__(self):
        """Iterate over the genotype data for each individual."""
        for i in range(self.num_inds):
            yield self.snp_data[i]

    def __contains__(self, individual):
        """Check if an individual is present in the dataset."""
        return individual in self.samples

    def __str__(self):
        """Return a string representation of the GenotypeData object."""
        return f"GenotypeData: {self.num_snps} SNPs, {self.num_inds} individuals"

    def __repr__(self):
        """Return a detailed string representation of the GenotypeData object."""
        return f"GenotypeData(filename={self.filename}, filetype={self.filetype}, popmapfile={self.popmapfile}, force_popmap={self.force_popmap}, exclude_pops={self.exclude_pops}, include_pops={self.include_pops}, plot_format={self.plot_format}, prefix={self.prefix})"
