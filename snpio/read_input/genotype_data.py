import copy
import itertools
import logging
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Self,
    Sequence,
    Tuple,
    cast,
)

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

import snpio.utils.custom_exceptions as exceptions
from snpio.plotting.plotting import Plotting
from snpio.read_input.genotype_data_base import BaseGenotypeData
from snpio.read_input.popmap_file import ReadPopmap
from snpio.utils.containers import IOConfig, PlotConfig, PopState
from snpio.utils.custom_exceptions import AlignmentFormatError
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC
from snpio.utils.missing_stats import MissingStats
from snpio.utils.multiqc_reporter import SNPioMultiQC
from snpio.utils.output_paths import OutputPaths


class GenotypeData(BaseGenotypeData):
    """A class for handling and analyzing genotype data.

    The GenotypeData class is intended as a parent class for the file reader classes, such as VCFReader, StructureReader, and PhylipReader. It provides common methods and attributes for handling genotype data, such as reading population maps, subsetting data, and generating missingness reports.

    Note:
        GenotypeData handles the following characters as missing data:
            - 'N'
            - '-'
            - '?'
            - '.'

        If using PHYLIP or STRUCTURE formats, all sites will be forced to be biallelic. If multiple alleles are needed, you must use a VCF file.

    Attributes:
        inputs (dict): GenotypeData keyword arguments as a dictionary.
        num_snps (int): Number of SNPs in the dataset.
        num_inds (int): Number of individuals in the dataset.
        populations (List[str | int]): Population IDs.
        popmap (dict): Dictionary object with SampleIDs as keys and popIDs as values.
        popmap_inverse (dict or None): Inverse dictionary of popmap, where popIDs are keys and lists of sampleIDs are values.
        samples (List[str]): Sample IDs in input order.
        snpsdict (dict or None): Dictionary with SampleIDs as keys and lists of genotypes as values.
        snp_data (List[List[str]]): Genotype data as a 2D list.
        loci_indices (List[int]): Column indices for retained loci in filtered alignment.
        sample_indices (List[int]): Row indices for retained samples in the alignment.
        ref (List[str]): List of reference alleles of length num_snps.
        alt (List[str]): List of alternate alleles of length num_snps.
        iupac_mapping (dict): Mapping of allele tuples to IUPAC codes.
        reverse_iupac_mapping (dict): Mapping of IUPAC codes to allele tuples.
        missing_vals (List[str]): List of missing value characters.
        replace_vals (List[pd.NA]): List of missing value replacements.
        logger (logging.Logger): Logger object.
        debug (bool): If True, display debug messages.
        plot_kwargs (dict): Plotting keyword arguments.
        supported_filetypes (List[str]): List of supported filetypes.
        kwargs (dict): GenotypeData keyword arguments.
        chunk_size (int): Chunk size for reading in large files.
        plot_format (str): Format to save report plots.
        plot_fontsize (int): Font size for plots.
        plot_dpi (int): Resolution in dots per inch for plots.
        plot_despine (bool): If True, remove the top and right spines from plots.
        show_plots (bool): If True, display plots in the console.
        prefix (str): Prefix to use for output directory.
        verbose (bool): If True, display verbose output.

    Methods:
        read_popmap: Read population map from file to map samples to populations.
        subset_with_popmap: Subset popmap and samples based on population criteria.
        write_popmap: Write the population map to a file.
        missingness_reports: Generate missingness reports and plots.
        _make_snpsdict: Make a dictionary with SampleIDs as keys and a list of SNPs associated with the sample as the values.
        _genotype_to_iupac: Convert a genotype string to its corresponding IUPAC code.
        _iupac_to_genotype: Convert an IUPAC code to its corresponding genotype string.
        calc_missing: Calculate missing value statistics based on a DataFrame.
        copy: Create a deep copy of the GenotypeData object.
        read_popmap: Read in a popmap file.
        missingness_reports: Create missingness reports from GenotypeData object.
        _genotype_to_iupac: Convert a genotype string to its corresponding IUPAC code.
        _iupac_to_genotype: Convert an IUPAC code to its corresponding genotype string.
        get_reverse_iupac_mapping: Create a reverse mapping from IUPAC codes to allele tuples.

    Example:
        >>> gd = GenotypeData(file="data.vcf", filetype="vcf", popmapfile="popmap.txt")
        >>> print(gd.snp_data)
        [['A', 'C', 'G', 'T'], ['A', 'C', 'G', 'T'], ['A', 'C', 'G', 'T']]
        >>> print(gd.num_snps)
        4
        >>> print(gd.num_inds)
        3
        >>> print(gd.populations)
        ['pop1', 'pop2', 'pop2']
        >>> print(gd.popmap)
        {'sample1': 'pop1', 'sample2': 'pop2', 'sample3': 'pop2'}
        >>> print(gd.samples)
        ['sample1', 'sample2', 'sample3']
    """

    def __init__(
        self,
        filename: str | None = None,
        filetype: str | None = None,
        popmapfile: str | None = None,
        force_popmap: bool = False,
        exclude_pops: List[str] | None = None,
        include_pops: List[str] | None = None,
        plot_format: Literal["png", "pdf", "jpg", "jpeg"] | None = "png",
        plot_fontsize: int = 18,
        plot_dpi: int = 300,
        plot_despine: bool = True,
        show_plots: bool = False,
        prefix: str = "snpio",
        verbose: bool = False,
        loci_indices: List[int] | np.ndarray | None = None,
        sample_indices: List[int] | np.ndarray | None = None,
        chunk_size: int = 1000,
        logger: logging.Logger | None = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the GenotypeData object.

        This class is used to read in genotype data from various file formats, such as VCF, PHYLIP, and STRUCTURE. It provides methods for reading population maps, subsetting data, and generating missingness reports.

        Args:
            filename (str): Path to input file containing genotypes. Defaults to None.
            filetype (str): Type of input genotype file. Possible values include: 'phylip', 'structure', 'vcf', 'tree', or '012'. Defaults to None.
            popmapfile (str): Path to population map file. If supplied and filetype is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column. Defaults to None.
            force_popmap (bool): If True, then samples not present in the popmap file will be excluded from the alignment. If False, then an error is raised if samples are present in the popmap file that are not present in the alignment. Defaults to False.
            exclude_pops (List[str]): List of population IDs to exclude from the alignment. Defaults to None.
            include_pops (List[str]): List of population IDs to include in the alignment. Populations not present in the include_pops list will be excluded. Defaults to None.
            plot_format (str): Format to save report plots. Valid options include: 'pdf', 'svg', 'png', and 'jpeg'. Defaults to 'png'.
            plot_fontsize (int): Font size for plots. Defaults to 12.
            plot_dpi (int): Resolution in dots per inch for plots. Defaults to 300.
            plot_despine (bool): If True, remove the top and right spines from plots. Defaults to True.
            show_plots (bool): If True, display plots in the console. Defaults to False.
            prefix (str): Prefix to use for output directory. Defaults to "gtdata".
            verbose (bool): If True, display verbose output. Defaults to False.
            loci_indices (np.ndarray): Column indices for retained loci in filtered alignment. Defaults to None.
            sample_indices (np.ndarray): Row indices for retained samples in the alignment. Defaults to None.
            chunk_size (int): Chunk size for reading in large files. Defaults to 1000.
            logger (logging.Logger): Logger object. Defaults to None.
            debug (bool): If True, display debug messages. Defaults to False.

        Raises:
            UnsupportedFileTypeError: If the filetype is not supported.

        Note:
            If using PHYLIP or STRUCTURE formats, all sites will be forced to be biallelic. If multiple alleles are needed, you must use a VCF file.
        """
        if filetype is not None:
            filetype = filetype.lower()

        self.prefix = prefix
        self.was_filtered = False
        self.verbose = verbose
        self.debug = debug

        super().__init__(self.prefix, self.verbose, self.debug, filename, filetype)

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
        self.chunk_size = chunk_size

        self.supported_filetypes: set = {
            "vcf",
            "phylip",
            "structure",
            "tree",
            "genepop",
        }

        self._snp_data: np.ndarray | None = None
        self.from_vcf = getattr(self, "from_vcf", False)

        if logger is None:
            kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
            logman = LoggerManager(__name__, **kwargs)
            self.logger: logging.Logger = logman.get_logger()
        else:
            self.logger: logging.Logger = logger

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
        }

        if self.plot_format is None:
            self.plot_format = "png"

        self.plot_config = PlotConfig(
            plot_format=self.plot_format,
            plot_fontsize=self.plot_fontsize,
            dpi=self.plot_dpi,
            despine=self.plot_despine,
            show=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
        )

        self.io_config = IOConfig(
            prefix=self.prefix,
            chunk_size=self.chunk_size,
            force_popmap=self.force_popmap,
            include_pops=self.include_pops,
            exclude_pops=self.exclude_pops,
            verbose=self.verbose,
            debug=self.debug,
        )

        self.missing_vals = ["N", "-", ".", "?"]
        self.replace_vals = [pd.NA] * len(self.missing_vals)
        self.all_missing_idx = []

        self._samples: List[str] = []
        self._populations: List[str | int] = []
        self._ref: List[str] = []
        self._alt: List[List[str]] = []
        self._popmap: Dict[str, str | int] | None = None
        self._popmap_inverse: Dict[str | int, List[str]] | None = None

        self._loci_indices = loci_indices
        self._sample_indices = sample_indices

        if self.filetype not in self.supported_filetypes and self.filetype is not None:
            msg = f"Unsupported filetype provided to GenotypeData: {self.filetype}"
            self.logger.error(msg)
            raise exceptions.UnsupportedFileTypeError(
                self.filetype, supported_types=list(self.supported_filetypes)
            )

        if self.filetype is None:
            msg = "No filetype provided to GenotypeData. Please specify a supported filetype."
            self.logger.error(msg)
            raise TypeError(msg)

        self.kwargs["filetype"] = self.filetype
        self.kwargs["loci_indices"] = self._loci_indices
        self.kwargs["sample_indices"] = self._sample_indices

        self.iupac_mapping: Dict[Tuple[str, ...], str] = self._iupac_from_gt_tuples()

        self.reverse_iupac_mapping: Dict[str, Tuple[str, ...]] = {
            v: k for k, v in self.iupac_mapping.items()
        }

        self.pop_state = PopState(samples=self._samples)

        if not hasattr(self, "iupac"):
            self.iupac = IUPAC(logger=self.logger)

        # Ensure the load_aln method is called.
        _ = self.snp_data

        self.snpio_mqc = SNPioMultiQC

        self.set_alignment(
            self.snp_data,
            self.samples,
            self.sample_indices,
            self.loci_indices,
            reset_attributes=False,
        )

    def _iupac_from_gt_tuples(self) -> Dict[Tuple[str, ...], str]:
        """Returns the IUPAC code mapping from allele tuples to IUPAC codes.

        Notes:
            - ('A', 'A') -> 'A'
            - ('C', 'C') -> 'C'
            - ('G', 'G') -> 'G'
            - ('T', 'T') -> 'T'
            - ('A', 'G') -> 'R'
            - ('C', 'T') -> 'Y'
            - ('G', 'C') -> 'S'
            - ('A', 'T') -> 'W'
            - ('G', 'T') -> 'K'
            - ('A', 'C') -> 'M'

        Returns:
            Dict[Tuple[str, ...], str]: Mapping of allele tuples to IUPAC codes. The keys are tuples of alleles (e.g., ('A', 'G')) and the values are the corresponding IUPAC codes (e.g., 'R').
        """
        return self.iupac.get_tuple_to_iupac()

    def get_reverse_iupac_mapping(self) -> Dict[str, Tuple[str, ...]]:
        """Creates a reverse mapping from IUPAC codes -> allele tuples.

        Returns:
            Dict[str, Tuple[str, ...]]: Mapping of IUPAC codes to allele tuples (e.g., 'R' -> ('A', 'G')).
        """
        return self.reverse_iupac_mapping

    def _make_snpsdict(
        self,
        samples: List[str] | None = None,
        snp_data: np.ndarray | List[List[str]] | None = None,
    ) -> Dict[str, List[str]]:
        """Make a dictionary with SampleIDs as keys and a list of SNPs associated with the sample as the values.

        This method is used to create a dictionary with sample IDs as keys and a list of SNPs as values. The dictionary is used to quickly access the SNPs associated with a sample.

        Args:
            samples (List[str] | None, optional): List of sample IDs. If not provided, uses self.samples.
            snp_data (np.ndarray | List[List[str]] | None, optional): 2D list of genotypes. If not provided, uses self.snp_data.

        Returns:
            Dict[str, List[str]]: Dictionary with sample IDs as keys and a list of SNPs as values. The keys are sample IDs (e.g., 'sample1') and the values are lists of SNPs (e.g., ['A', 'G', 'T']).
        """
        if samples is None:
            samples = self.samples
        if snp_data is None:
            snp_data = self.snp_data

        snpsdict = {ind: seq for ind, seq in zip(samples, snp_data)}
        return snpsdict

    def read_popmap(self) -> None:
        """Read population map from file to map samples to populations.

        Makes use of the ReadPopmap class to read in the popmap file and validate the samples against the alignment.

        Sets or updates the following attributes:
            - samples
            - populations
            - popmap
            - popmap_inverse
            - sample_indices

        Raises:
            PopmapFileNotFoundError: If the popmap file is not found or is of an invalid type.
            PopmapFileFormatError: If the popmap file format is invalid or does not align with the samples in the alignment.
            EmptyIterableError: If no valid samples are found after subsetting with the popmap.
        """
        self.logger.info(f"Reading population map from: {self.popmapfile}")

        if self.popmapfile is not None:
            if not isinstance(self.popmapfile, (str, Path)):
                msg = f"Invalid popmapfile type provided. Expected str or pathlib.Path, but got: {type(self.popmapfile)}"
                self.logger.error(msg)
                raise exceptions.PopmapFileNotFoundError(msg)

            # Instantiate popmap object and read in the popmap file.
            pm = ReadPopmap(
                self.popmapfile, self.logger, verbose=self.verbose, debug=self.debug
            )

            # Get the samples and populations from the popmap file.
            fc = self.force_popmap
            inc, exc = self.include_pops, self.exclude_pops
            kwargs = {"force": fc, "include_pops": inc, "exclude_pops": exc}

            # Subset the popmap and samples based on the population criteria.
            self.subset_with_popmap(pm, self.samples, **kwargs)
            pm.get_pop_counts(self)

        else:
            msg = "No popmapfile provided. Skipping population map reading."
            self.logger.info(msg)
            self._popmap = dict(zip(self.samples, ["NA"] * self.num_inds))
            self._popmap_inverse = {"NA": self.samples}
            self._populations = ["NA"] * self.num_inds
            self.sample_indices = np.ones(self.num_inds, dtype=bool)

        self.pop_state.populations = self._populations
        self.pop_state.popmap = self._popmap
        self.pop_state.popmap_inverse = self._popmap_inverse
        self.pop_state.refresh_num_pops()

        self.logger.info(f"Population map reading complete.")

    def subset_with_popmap(
        self,
        my_popmap: ReadPopmap,
        samples: List[str],
        force: bool,
        include_pops: List[str] | None = None,
        exclude_pops: List[str] | None = None,
        return_indices: bool = False,
    ) -> np.ndarray | None:
        """Subset popmap and samples based on population criteria.

        This method validates the popmap with the current samples and subsets the popmap based on inclusion/exclusion criteria. It updates the sample list and populations accordingly.

        Args:
            my_popmap (ReadPopmap): ReadPopmap instance.
            samples (List[str]): List of sample IDs.
            force (bool): If True, force the subsetting. If False, raise an error if the samples don't align.
            include_pops (List[str] | None): List of populations to include. If provided, only samples belonging to these populations will be retained.
            exclude_pops (List[str] | None): List of populations to exclude. If provided, samples belonging to these populations will be excluded.
            return_indices (bool): If True, return the indices for samples. Defaults to False.

        Returns:
            np.ndarray | None: Boolean array of `sample_indices` if return_indices is True. Otherwise, None.

        Raises:
            EmptyIterableError: If no valid samples are found after subsetting.
            PopmapFileNotFoundError: If the popmap file is not found or is of an invalid type.
            PopmapFileFormatError: If the popmap file format is invalid or does not align with the samples in the alignment.
        """
        # Validate popmap with current samples
        popmap_ok = my_popmap.validate_popmap(samples, force=force)

        if not popmap_ok:
            msg = "Popmap validation failed. Check the popmap file and try again."
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

        # Subset the popmap based on inclusion/exclusion criteria
        my_popmap.subset_popmap(samples, include_pops, exclude_pops)

        popmap = my_popmap.popmap

        # Update the sample list and the populations
        new_samples = [str(s) for s in samples if s in popmap]
        if len(new_samples) != len(samples):
            self.logger.warning(
                "Some samples in the alignment are not found in the population map."
            )
            self.logger.debug(f"Length of samples: {len(samples)}")
            self.logger.debug(f"Length of new_samples: {len(new_samples)}")

        self.logger.debug(f"samples: {samples}")
        self.logger.debug(f"new_samples: {new_samples}")

        new_populations = [my_popmap.popmap[s] for s in new_samples]

        if not new_samples:
            msg = "No valid samples found after popmap subsetting."
            self.logger.error(msg)
            raise exceptions.EmptyIterableError(msg)

        # Create a boolean mask for filtering
        sample_mask = np.isin(self.samples, new_samples)

        # Update samples and populations based on the subset
        self.samples = new_samples
        self._populations = [str(x) for x in new_populations]

        # Filter the snp_data using the mask
        self.snp_data = self.snp_data[sample_mask, :]

        # Update popmap and inverse popmap
        if my_popmap.popmap is not None:
            # Ensure all keys and values are strings
            popmap = my_popmap.popmap
            self._popmap = {str(k): str(v) for k, v in popmap.items()}

            popmap_flip = my_popmap.popmap_flipped
            self._popmap_inverse = {str(k): v for k, v in popmap_flip.items()}

        # Reset sample_indices to reflect the new state
        # (all current samples are kept)
        self.sample_indices = np.ones(self.num_inds, dtype=bool)

        # Return indices if requested
        if return_indices:
            return sample_mask  # Return the original mask used for filtering

    def write_popmap(self, filename: str | Path) -> None:
        """Write the population map to a file.

        Args:
            filename (str | Path): Output file path.

        Raises:
            IOError: If the filename is not a string or Path object.
            EmptyIterableError: If the samples attribute is not defined.
            EmptyIterableError: If the populations attribute is not defined.
            EmptyIterableError: If the popmap attribute is not defined.
        """
        self.logger.info("Writing population map to file...")

        if not isinstance(filename, (str, Path)):
            msg = f"Invalid filename type provided. Expected str or pathlib.Path, but got: {type(filename)}"
            self.logger.error(msg)
            raise IOError(msg)

        fp = Path(filename)

        if not self.samples or self.samples is None:
            msg = "'samples' attribute is undefined."
            self.logger.error(msg)
            raise exceptions.EmptyIterableError(msg)

        if not self.populations or self.populations is None:
            msg = "'populations' attribute is undefined."
            self.logger.error(msg)
            raise exceptions.EmptyIterableError(msg)

        with open(fp, "w") as fout:
            [fout.write(f"{s},{p}\n") for s, p in zip(self.samples, self.populations)]

        self.logger.info(f"Population map written to: {fp}")

    @staticmethod
    def _sanitize_vcf_id(value: Any) -> str:
        """Sanitize a VCF ID field value.

        Args:
            value: Raw ID field value.

        Returns:
            VCF-compatible ID value. Missing, empty, None-like, or invalid values are
            returned as ".".
        """
        if value is None:
            return "."

        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")

        value_str = str(value).strip()

        if value_str in {"", ".", "None", "none", "NONE", "nan", "NaN", "NA", "<NA>"}:
            return "."

        return value_str

    def write_vcf(
        self,
        output_filename: str | Path,
        hdf5_file_path: str | Path | None = None,
        chunk_size: int = 1000,
    ) -> None:
        """Writes the GenotypeData object data to a VCF file in chunks.

        This method writes the VCF data, bgzips the output file, indexes it with Tabix, and validates the output. If the original input was VCF (`self.from_vcf`), it reads from the HDF5 produced by `get_vcf_attributes_fast`. Otherwise it falls back to writing from `self.snp_data` directly.

        Args:
            output_filename (str | Path): The name of the output VCF file.
            hdf5_file_path (str | Path, optional): Path to the HDF5 file; if None uses self.vcf_attributes_fn.
            chunk_size (int, optional): Number of records per write chunk.

        Raises:
            TypeError: If the output filename is not a string or Path object.
            FileNotFoundError: If the HDF5 file is not found when reading from VCF.
            Exception: If there is an error encoding SNP data to VCF format.
        """
        self.logger.info(f"Writing VCF file to: {output_filename}")

        from_vcf = self.from_vcf

        of = Path(output_filename)
        parent_pth = of.parent
        if of.suffix == ".gz":
            of = Path(of.stem)  # Remove .gz suffix; gets added back later

        of = parent_pth / of
        of.parent.mkdir(parents=True, exist_ok=True)

        try:
            if from_vcf:
                # HDF5-backed branch
                if hdf5_file_path is None:
                    hdf5_file_path = self.vcf_attributes_fn

                if hdf5_file_path is None:
                    msg = "HDF5 file path is not provided and self.vcf_attributes_fn is not set."
                    self.logger.error(msg)
                    raise TypeError(msg)

                h5p = Path(hdf5_file_path)
                if not h5p.is_file():
                    msg = f"HDF5 file not found: {h5p}"
                    self.logger.error(msg)
                    raise FileNotFoundError(msg)

                with h5py.File(h5p, "r") as h5, open(of, "w") as f:
                    f.write(self.build_vcf_header())

                    chrom_ds = cast(h5py.Dataset, h5["chrom"])
                    pos_ds = cast(h5py.Dataset, h5["pos"])
                    id_ds = cast(h5py.Dataset, h5["id"])
                    ref_ds = cast(h5py.Dataset, h5["ref"])
                    alt_ds = cast(h5py.Dataset, h5["alt"])
                    filt_ds = cast(h5py.Dataset, h5["filt"])

                    info_grp = cast(h5py.Group, h5["info"])

                    total = chrom_ds.shape[0]
                    info_keys = list(info_grp.keys())

                    if not hasattr(self, "store_format_fields"):
                        self.store_format_fields = False

                    write_format_metadata = (
                        self.store_format_fields and "fmt_metadata" in h5
                    )

                    if write_format_metadata:
                        fmt_metadata_grp = cast(h5py.Group, h5["fmt_metadata"])
                        fmt_keys = list(fmt_metadata_grp.keys())
                    else:
                        fmt_metadata_grp = None
                        fmt_keys = []
                        self.logger.debug(
                            "Format metadata fields not being written to the VCF."
                        )

                    qual_ds = cast(h5py.Dataset, h5["qual"]) if "qual" in h5 else None

                    with tqdm(
                        total=total, desc="Writing VCF Records: ", unit=" rec"
                    ) as pbar:
                        for start in range(0, total, chunk_size):
                            end = min(start + chunk_size, total)
                            n = end - start

                            chrom = chrom_ds[start:end].astype(str)
                            pos = pos_ds[start:end]
                            vid = np.array(
                                [
                                    self._sanitize_vcf_id(value)
                                    for value in id_ds[start:end]
                                ],
                                dtype=object,
                            )
                            ref = ref_ds[start:end].astype(str)
                            alt_raw = alt_ds[start:end].astype(str)
                            filt = filt_ds[start:end].astype(str)

                            qual = (
                                qual_ds[start:end].astype(str)
                                if qual_ds is not None
                                else np.array(["."] * n)
                            )

                            info_chunk = {
                                key: cast(h5py.Dataset, info_grp[key])[
                                    start:end
                                ].astype(str)
                                for key in info_keys
                            }

                            refs = ref.tolist()
                            alts = [
                                (
                                    [
                                        a
                                        for a in s.split(",")
                                        if a not in {"", ".", "'", "(", ")", " "}
                                    ]
                                    if s not in {".", ""}
                                    else []
                                )
                                for s in alt_raw
                            ]

                            snp_chunk = self.snp_data[:, start:end].T

                            try:
                                fmt_matrix = self.encode_to_vcf_format(
                                    snp_chunk, refs, alts
                                )
                            except Exception as e:
                                self.logger.error(f"Error encoding SNP data: {e}")
                                raise

                            if write_format_metadata and fmt_metadata_grp is not None:
                                fmt_chunks = {
                                    key: cast(h5py.Dataset, fmt_metadata_grp[key])[
                                        start:end, :
                                    ].astype(str)
                                    for key in fmt_keys
                                }
                            else:
                                fmt_chunks = {}

                            for i in range(n):
                                parts = [
                                    f"{key}={info_chunk[key][i]}"
                                    for key in info_keys
                                    if info_chunk[key][i] not in {".", ""}
                                ]
                                info_str = ";".join(parts) if parts else "."
                                alt_str = ",".join(alts[i]) if alts[i] else "."

                                if write_format_metadata:
                                    format_col = "GT:" + ":".join(fmt_keys)
                                    sample_fields = []

                                    for j in range(fmt_matrix.shape[1]):
                                        extra_fields = [
                                            str(fmt_chunks[k][i][j]) for k in fmt_keys
                                        ]
                                        sample_field = ":".join(
                                            [str(fmt_matrix[i][j])] + extra_fields
                                        )
                                        sample_fields.append(sample_field)
                                else:
                                    format_col = "GT"
                                    sample_fields = fmt_matrix[i].astype(str).tolist()

                                row = [
                                    str(chrom[i]),
                                    str(pos[i]),
                                    str(vid[i]),
                                    str(ref[i]),
                                    alt_str,
                                    str(qual[i]),
                                    str(filt[i]),
                                    info_str,
                                    format_col,
                                ] + sample_fields

                                f.write("\t".join(row) + "\n")

                            pbar.update(n)
            else:
                # Raw fallback
                n_loci = self.snp_data.shape[1]
                chrom_idx = np.arange(1, n_loci + 1).astype(str)
                chrom = [f"locus{idx}" for idx in chrom_idx]
                pos = [1] * n_loci
                vid = [self._sanitize_vcf_id(f"locus{idx}") for idx in chrom_idx]
                refs, alts = self.refs_alts_from_snp_data(self.snp_data)
                qual = ["."] * n_loci
                filt = ["PASS"] * n_loci
                info = [
                    f"NS={np.count_nonzero(self.snp_data != 'N', axis=0)[i]}"
                    for i in range(n_loci)
                ]

                try:
                    fmt_matrix = self.encode_to_vcf_format(self.snp_data.T, refs, alts)
                except Exception as e:
                    self.logger.error(f"Error encoding SNP data: {e}")
                    raise

                with open(of, "w") as f:
                    f.write(self.build_vcf_header())
                    with tqdm(
                        total=n_loci, desc="Writing VCF Records: ", unit=" rec"
                    ) as pbar:
                        for i in range(n_loci):

                            alt_str = ",".join(alts[i]) if alts[i] else "."
                            row = [
                                chrom[i],
                                str(pos[i]),
                                vid[i],
                                refs[i],
                                alt_str,
                                qual[i],
                                filt[i],
                                info[i],
                                "GT",
                            ] + fmt_matrix[i].tolist()
                            f.write("\t".join(row) + "\n")
                            pbar.update(1)

        except Exception as e:
            self.logger.error(f"Error writing VCF file: {e}")
            raise

        if not hasattr(self, "_is_bgzipped"):
            self._is_bgzipped = lambda x: False

        # bgzip the file if not already bgzipped and tabix index it.
        # NOTE: This adds a .gz suffix to the output file.
        self.tabix_index(of)

        self.logger.info(f"Indexed VCF file: {of}.tbi")
        self.logger.info(f"VCF file written to: {of}")

    def write_phylip(
        self,
        output_file: str | Path,
        genotype_data: Any = None,
        snp_data: np.ndarray | None = None,
        samples: List[str] | None = None,
    ) -> None:
        """Write the stored alignment as a PHYLIP file.

        This method writes the stored alignment as a PHYLIP file. The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data.

        Args:
            output_file (str): Name of the output PHYLIP file.
            genotype_data (GenotypeData, optional): GenotypeData instance.
            snp_data (List[List[str]], optional): SNP data. Must be provided if genotype_data is None.
            samples (List[str], optional): List of sample IDs. Must be provided if snp_data is not None.

        Raises:
            TypeError: If genotype_data and snp_data are both provided.
            TypeError: If samples are not provided when snp_data is provided.
            AlignmentFormatError: If samples and snp_data are not the same length.
            Exception: If there is an error writing the PHYLIP file.

        Notes:
            - If ``genotype_data`` is provided, the ``snp_data`` and ``samples`` objects are loaded from the ``GenotypeData`` instance.
            - If ``snp_data`` is provided, ``samples`` must also be provided.
            - If ``genotype_data`` is not provided, ``snp_data`` and ``samples`` must be provided.
            - The sequence data must have the same length for each sample.
        """
        if genotype_data is not None and snp_data is not None:
            msg = "Arguments 'genotype_data' and 'snp_data' cannot both be provided."
            self.logger.error(msg)
            raise TypeError(msg)
        elif genotype_data is None and snp_data is None:
            snp_data = self.snp_data
            samples = self.samples
        elif genotype_data is not None and snp_data is None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
        elif genotype_data is None and snp_data is not None and samples is None:
            msg = "If 'snp_data' is provided, 'samples' argument must also be provided "
            self.logger.error(msg)
            raise TypeError(msg)

        self.logger.info(f"Writing PHYLIP file as: {output_file}...")

        if not isinstance(output_file, Path):
            output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        self._validate_seq_lengths()

        try:
            with open(output_file, "w") as f:
                if samples is None or snp_data is None:
                    msg = "Samples and SNP data must be provided to write PHYLIP file."
                    self.logger.error(msg)
                    raise AlignmentFormatError(msg)

                n_samples = len(samples)
                if n_samples == 0:
                    n_loci = 0
                else:
                    n_loci = len(snp_data[0])

                f.write(f"{n_samples} {n_loci}\n")

                for sample, sample_data in zip(samples, snp_data):
                    genotype_str = "".join(str(x) for x in sample_data)
                    f.write(f"{sample}\t{genotype_str}\n")

            self.logger.info(f"PHYLIP file written to: {output_file}")

        except Exception as e:
            msg = f"An error occurred while writing the PHYLIP file: {e}"
            self.logger.error(msg)
            raise e

    def write_structure(
        self,
        output_file: str | Path,
        onerow: bool = False,
        popids: bool = False,
        marker_names: bool = False,
        genotype_data: Any = None,
        snp_data: np.ndarray | None = None,
        samples: List[str] | None = None,
    ) -> None:
        """Write the stored alignment as a STRUCTURE file.

        This method writes the stored alignment as a STRUCTURE file. The STRUCTURE format is a text format used for population structure analysis. The first line contains the number of samples and the number of loci. Each subsequent line contains the sample ID, population ID (if popids is True), and genotype data.

        Args:
            output_file (str): Name of the output STRUCTURE file.
            onerow (bool, optional): If True, write genotypes in one row per sample. If False, write two rows per sample (one for each allele). Defaults to False.
            popids (bool, optional): If True, include population IDs in the output file. Defaults to False.
            marker_names (bool, optional): If True, include marker names in the header row. Defaults to False.
            genotype_data (GenotypeData, optional): GenotypeData instance. If provided, snp_data and samples are loaded from it.
            snp_data (np.ndarray | None, optional): SNP data. If provided, samples must also be provided.
            samples (List[str] | None, optional): List of sample IDs. Must be provided if snp_data is not None.

        Raises:
            TypeError: If genotype_data and snp_data are both provided.
            TypeError: If samples are not provided when snp_data is provided.
            EmptyIterableError: If samples and snp_data are not the same length.
            Exception: If there is an error writing the STRUCTURE file.

        Note:
            - If genotype_data is provided, the snp_data and samples are loaded from the GenotypeData instance.
            - If snp_data is provided, the samples must also be provided.
            - If genotype_data is not provided, the snp_data and samples must be provided.
        """
        if not isinstance(output_file, Path):
            output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        if genotype_data is not None and snp_data is None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
        elif genotype_data is None and snp_data is None:
            snp_data = self.snp_data
            samples = self.samples
        elif genotype_data is None and snp_data is not None and samples is None:
            msg = "If snp_data is provided, samples argument must also be provided."
            self.logger.error(msg)
            raise TypeError(msg)

        self.logger.info(f"Writing STRUCTURE file to: {output_file}")

        # Get marker names if requested
        if marker_names:
            marker_names_list = getattr(self, "marker_names", None)

            # If VCF-derived data lost marker_names or has stale marker_names,
            # rebuild marker names from chrom/pos stored in the VCF HDF5 attributes.
            if getattr(self, "from_vcf", False) and (
                marker_names_list is None
                or len(marker_names_list) == 0
                or len(marker_names_list) != self.num_snps
            ):
                try:
                    with h5py.File(self.vcf_attributes_fn, "r") as h5:
                        chrom_ds = cast(h5py.Dataset, h5["chrom"])
                        pos_ds = cast(h5py.Dataset, h5["pos"])
                        chrom = chrom_ds[:]
                        pos = pos_ds[:]

                    marker_names_list = [
                        f"{c.decode() if isinstance(c, bytes) else c}:{p}"
                        for c, p in zip(chrom, pos)
                    ]

                except (AttributeError, FileNotFoundError, OSError, KeyError) as e:
                    msg = "Marker names requested for STRUCTURE output, but marker names could not be recovered from the VCF attributes file."
                    self.logger.error(f"{msg} Original error: {e}")
                    raise exceptions.EmptyIterableError(msg) from e

            # Non-VCF fallback.
            if marker_names_list is None or len(marker_names_list) == 0:
                marker_names_list = [f"locus_{i + 1}" for i in range(self.num_snps)]

            marker_names_list = list(marker_names_list)

        else:
            marker_names_list = []

        allele_start_col: int = getattr(self, "allele_start_col", 2 if popids else 1)

        # Write file
        try:
            with open(output_file, "w") as fout:
                # Header row with marker names (only once)
                if marker_names and marker_names_list:
                    header_prefix = []
                    for _ in range(allele_start_col):
                        header_prefix.append("\t")
                    if onerow:
                        header_prefix += itertools.chain.from_iterable(
                            zip(marker_names_list, marker_names_list)
                        )
                    else:
                        header_prefix += marker_names_list
                    fout.write("\t".join(header_prefix) + "\n")

                if samples is None or snp_data is None:
                    msg = (
                        "Samples and SNP data must be provided to write STRUCTURE file."
                    )
                    self.logger.error(msg)
                    raise exceptions.EmptyIterableError(msg)

                for idx, (sample, sample_data) in enumerate(zip(samples, snp_data)):
                    genotypes = [
                        self._iupac_to_genotype(iupac) for iupac in sample_data
                    ]

                    if onerow:
                        genotype_pairs = [a for gt in genotypes for a in gt.split("/")]
                        row = [sample]
                        if popids:
                            row.append(str(self.populations[idx]))
                        row.extend(genotype_pairs)
                        fout.write("\t".join(row) + "\n")
                    else:
                        first_row = [gt.split("/")[0] for gt in genotypes]
                        second_row = [gt.split("/")[1] for gt in genotypes]

                        row1 = [sample]
                        row2 = [sample]
                        if popids:
                            row1.append(str(self.populations[idx]))
                            row2.append(str(self.populations[idx]))
                        row1.extend(first_row)
                        row2.extend(second_row)
                        fout.write("\t".join(row1) + "\n")
                        fout.write("\t".join(row2) + "\n")

            self.logger.info(f"STRUCTURE file written to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to write STRUCTURE file: {e}")
            raise IOError(f"Failed to write STRUCTURE file: {e}") from e

    def write_genepop(
        self,
        output_file: str | Path,
        genotype_data: Any = None,
        snp_data: np.ndarray | None = None,
        samples: List[str] | None = None,
        marker_names: List[str] | None = None,
        title: str = "GenePop export from SNPio",
    ) -> None:
        """Write the SNP data in GenePop format.

        This method writes the SNP data in GenePop format, which is a text format used for population genetics data. The first line contains a title, followed by locus names, and then the sample IDs with their corresponding genotypes.

        Args:
            output_file (str | Path): File path to write to.
            genotype_data (GenotypeData): Object with .snp_data and .samples.
            snp_data (np.ndarray | None): Optional SNP data matrix.
            samples (List[str] | None): Optional list of sample IDs.
            marker_names (List[str] | None): Optional list of locus names.
            title (str): First line of the GenePop file.

        Raises:
            TypeError: If both genotype_data and snp_data are provided, or if samples are not provided when snp_data is provided.
            AlignmentFormatError: If marker names are requested but not found in the data, or if samples and SNP data are not provided to write the file.
            IOError: If there is an error writing the GenePop file.
        """
        if not isinstance(output_file, Path):
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        if genotype_data is not None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
            marker_names = getattr(genotype_data, "marker_names", None)
        elif snp_data is not None and samples is not None:
            pass
        else:
            msg = "Must provide either genotype_data or both snp_data and samples."
            self.logger.error(msg)
            raise TypeError(msg)

        if marker_names is None:
            if snp_data is None:
                msg = "SNP data must be provided to generate marker names."
                self.logger.error(msg)
                raise AlignmentFormatError(msg)
            marker_names = [f"Locus{i+1}" for i in range(snp_data.shape[1])]

        try:
            with open(output_file, "w") as fout:
                fout.write(f"{title}\n")
                for name in marker_names:
                    fout.write(f"{name}\n")
                fout.write("Pop\n")

                if samples is None or not samples:
                    msg = "Sample IDs must be provided to write GenePop file."
                    self.logger.error(msg)
                    raise AlignmentFormatError(msg)

                if snp_data is None:
                    msg = "SNP data must be provided to write GenePop file."
                    self.logger.error(msg)
                    raise AlignmentFormatError(msg)

                for i, sample in enumerate(samples):
                    alleles = snp_data[i]
                    # Assuming genotype format is IUPAC, convert to 2-digit
                    # diploid pairs
                    pairs = [self._iupac_to_genepop(gt) for gt in alleles]
                    fout.write(f"{sample} , {' '.join(pairs)}\n")

            self.logger.info(f"GENEPOP file written to: {output_file}")

        except Exception as e:
            msg = f"Error writing GenePop file: {e}"
            self.logger.error(msg)
            raise IOError(msg) from e

    def _iupac_to_genepop(self, iupac: str) -> str:
        """Convert IUPAC code to two-digit allele encoding."""
        iupac_to_bases = {
            "A": ("01", "01"),
            "C": ("02", "02"),
            "G": ("03", "03"),
            "T": ("04", "04"),
            "M": ("01", "02"),
            "R": ("01", "03"),
            "W": ("01", "04"),
            "S": ("02", "03"),
            "Y": ("02", "04"),
            "K": ("03", "04"),
            "N": ("00", "00"),
        }
        a1, a2 = iupac_to_bases.get(iupac, ("00", "00"))
        return f"{a1}{a2}"

    def missingness_reports(
        self,
        prefix: str | None = None,
        zoom: bool = False,
        bar_color: str = "gray",
        heatmap_palette: str = "magma",
    ) -> None:
        """Generate missingness reports and plots.

        The function will write several comma-delimited report files:

            1. individual_missingness_mqc.json: Missing proportions per individual.

            2. locus_missingness_mqc.json: Missing proportions per locus.

            3. population_missingness_mqc.json: Missing proportions per population (only generated if popmapfile was passed to GenotypeData).

            4. population_locus_missingness_mqc.json: Table of per-population and per-locus missing data proportions.

        A file missingness.<plot_format> will also be saved. It contains the following subplots:

            1. Barplot with per-individual missing data proportions.

            2. Barplot with per-locus missing data proportions.

            3. Barplot with per-population missing data proportions (only if popmapfile was passed to GenotypeData).

            4. Heatmap showing per-population + per-locus missing data proportions (only if popmapfile was passed to GenotypeData).

            5. Stacked barplot showing missing data proportions per individual.

            6. Stacked barplot showing missing data proportions per population (only if popmapfile was passed to GenotypeData).

        If popmapfile was not passed to GenotypeData, then the subplots and report files that require populations are not included.

        Args:
            prefix (str, optional): Output file prefix for the missingness report. Defaults to None.

            zoom (bool, optional): If True, zoom in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to False.

            bar_color (str, optional): Color of the bars on the non-stacked bar plots. Can be any color supported by matplotlib. See the matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, optional): Color palette for the heatmap plot. Defaults to 'magma'.
        """
        self.logger.info("Generating and plotting missingness reports...")

        # Set the prefix for the missingness report files,
        # If not provided.
        prefix = self.prefix if prefix is None else prefix

        # Set the parameters for the missingness report plots.
        keys = ["prefix", "zoom", "bar_color", "heatmap_palette"]
        values = [prefix, zoom, bar_color, heatmap_palette]

        # Create a dictionary of the parameters.
        params = dict(zip(keys, values))

        # Create a DataFrame from snp_data and replace missing values
        # with NA.
        df = pd.DataFrame(self.snp_data)
        df = df.replace(to_replace=self.missing_vals, value=pd.NA)

        # Update plot_kwargs and params with the appropriate values.
        kwargs = self.plot_kwargs
        kwargs.update({"plot_fontsize": self.plot_fontsize})
        kwargs["plot_title_fontsize"] = self.plot_fontsize

        # Plot the missingness reports.
        plotting = Plotting(self, **kwargs)

        stats = plotting.visualize_missingness(df, **params)

        # Location for all missingness report artifacts.
        report_root = OutputPaths(
            prefix,
            filtered=bool(self.was_filtered),
        ).reports("missingness")
        if self.was_filtered:
            description_prefix = "Missingness Report (Post-NRemover2 Filtering): "
        else:
            description_prefix = "Missingness Report (Pre-filtering): "

        report_root.mkdir(exist_ok=True, parents=True)

        df_summary = stats.summary()

        def _format_for_multiqc(obj: pd.Series | pd.DataFrame) -> pd.DataFrame:
            """Convert Series/ DataFrame to a MultiQC-compatible format.

            MultiQC expects a DataFrame with a single column of values and an index that serves as the category labels. If the input is a Series, it is converted to a DataFrame with the Series name as the column header. If the input is already a DataFrame, it is returned as is. If the input is neither, an attempt is made to convert it to a DataFrame, and if that fails, an error is raised.

            Args:
                obj (pd.Series | pd.DataFrame): The input object to format.

            Returns:
                pd.DataFrame: A DataFrame formatted for MultiQC, with a single column of values and an index for category labels.

            Raises:
                TypeError: If the input object cannot be converted to a DataFrame.
            """
            if isinstance(obj, pd.Series):
                df = obj.to_frame("missing_prop")
                return df
            elif isinstance(obj, pd.DataFrame):
                return obj
            else:
                try:
                    return pd.DataFrame(obj)
                except Exception as e:
                    msg = f"Error formatting for MultiQC: {e}"
                    self.logger.error(msg)
                    raise TypeError(msg)

        df_miss_summary = _format_for_multiqc(df_summary)

        self.snpio_mqc.queue_barplot(
            df=[df_miss_summary, df_miss_summary.mul(100)],
            panel_id="missing_summary",
            section="missing_data",
            title=f"SNPio: {description_prefix}Missing Data Summary Statistics",
            description=f"{description_prefix.capitalize()}Missing data proportions and percentages summarized across samples (individuals), loci, and populations.",
            index_label="Category (Summary Statistic)",
            pconfig={
                "data_labels": [
                    {
                        "name": "Proportion",
                        "ylab": "Missing Data Summary Statistics (Proportion)",
                        "ymax": 1.0,
                    },
                    {
                        "name": "Percent",
                        "ylab": "Missing Data Summary Statistics (%)",
                        "ymax": 100,
                    },
                ],
                "id": "missing_summary",
                "title": "SNPio: Missing Data Summary Statistics",
                "cpswitch": False,
                "cpswitch_c_active": False,
                "tt_decimals": 0,
                "stacking": "group",
            },
        )

        df_perind = _format_for_multiqc(stats.per_individual) * 100
        df_perind = df_perind.rename(columns={"missing_prop": "Percent Missingness"})

        self.snpio_mqc.queue_table(
            df=df_perind,
            panel_id="individual_missingness",
            section="missing_data",
            title=f"SNPio: {description_prefix}Percent Missingness per Sample",
            description=f"{description_prefix.capitalize()}Missing data percentages per sample.",
            index_label="Sample Name",
            pconfig={
                "id": "individual_missingness",
                "title": f"SNPio: {description_prefix.capitalize()}Percent Missingness per Sample",
                "save_file": True,
                "col1_header": "Sample Name",
                "min": 0,
                "scale": "YlOrBr",
                "xlab": "Percent",
                "ylab": "Missingness",
            },
        )

        df_perloc = _format_for_multiqc(stats.per_locus) * 100
        df_perloc = df_perloc.rename(columns={"missing_prop": "Percent Missing"})

        self.snpio_mqc.queue_violin(
            df=df_perloc,  # Convert to percentage
            panel_id="locus_missingness",
            section="missing_data",
            title=f"SNPio: {description_prefix}Percent Per-locus Missingness",
            description=f"{description_prefix.capitalize()}Violin plot depicting missing data percentages per locus. Thicker violin sections indicate that more individuals have missing data for that locus.",
            index_label="Locus ID",
            pconfig={
                "id": "locus_missingness",
                "title": f"SNPio: {description_prefix}Percent Per-locus Missingness",
                "save_file": True,
                "min": 0,
                "xmax": 100,
                "scale": "YlOrBr",
                "xlab": "Percent Missing",
                "ylab": "Per-locus Missingness",
                "subtitle": f"{len(df_perloc)} loci",
                "save_file": True,
            },
        )

        if stats.per_population is not None:
            df_perpop = _format_for_multiqc(stats.per_population) * 100
            df_perpop = df_perpop.rename(
                columns={"missing_prop": "Percent Missingness"}
            )

            self.snpio_mqc.queue_table(
                df=df_perpop,
                panel_id="population_missingness",
                section="missing_data",
                title=f"SNPio: {description_prefix}Percent Per-population Missingness (Mean)",
                description=f"{description_prefix.capitalize()}Missing data percentages per population, averaged across all loci.",
                index_label="Population ID",
                pconfig={
                    "title": f"SNPio: {description_prefix}Percent Per-population Missingness (Mean)",
                    "id": "population_missingness",
                    "save_file": True,
                    "col1_header": "Population ID",
                    "scale": "YlOrBr",
                },
            )

        if stats.per_population_locus is not None:
            df_poploc = _format_for_multiqc(stats.per_population_locus) * 100
            df_poploc = df_poploc.rename(
                columns={"missing_prop": "Percent Missingness"}
            )

            self.snpio_mqc.queue_violin(
                df=df_poploc,
                panel_id="population_locus_missingness",
                section="missing_data",
                title=f"SNPio: {description_prefix}Percent Missingness for each Population",
                description=f"{description_prefix.capitalize()}Violin plot depicting missing data percentages per-population and per-locus. Thicker violin sections indicate that more individuals have missing data for that locus in the given population.",
                index_label="Population",
                pconfig={
                    "title": f"SNPio: {description_prefix}Percent Missingness for each Population",
                    "id": "population_locus_missingness",
                    "save_file": True,
                    "col1_header": "Population ID",
                },
            )

        self.logger.info(f"Missingness reports written to: {report_root}")

    def _genotype_to_iupac(self, genotype: str) -> str:
        """Convert a genotype string to its corresponding IUPAC code.

        Args:
            genotype (str): Genotype string in the format "x/y".

        Returns:
            str: Corresponding IUPAC code for the input genotype. Returns 'N' if the genotype is not in the lookup dictionary.

        Raises:
            InvalidGenotypeError: If the input genotype format is invalid or if there is an error during processing.
            Exception: If any unexpected error occurs during the conversion process.
        """
        try:
            iupac_dict = getattr(self, "allele_encoding", {})
            if iupac_dict:
                # Static map from nucleotide pairs to IUPAC codes
                iupac_dict = self._get_custom_gt2iupac()
            else:
                iupac_dict = self.iupac.get_gt2iupac()

            # Validate genotype format
            if not isinstance(genotype, str) or "/" not in genotype:
                self.logger.error(f"Invalid genotype format: {genotype}")
                raise exceptions.InvalidGenotypeError(f"Invalid format: {genotype}")

            gt = iupac_dict.get(str(genotype), "N")  # Default to 'N' for undefined.

            return gt

        except Exception as e:
            self.logger.error(f"Error processing genotype {genotype}: {e}")
            raise

    def _get_custom_gt2iupac(self) -> dict:
        """Generate a custom genotype to IUPAC mapping based on the allele encoding provided in the GenotypeData instance.

        This method creates a mapping from genotype strings (e.g., "1/1", "1/2") to their corresponding IUPAC codes based on the allele encoding defined in the GenotypeData instance. It uses a standard IUPAC ambiguity code dictionary to determine the appropriate IUPAC code for each genotype combination. The resulting dictionary includes mappings for both homozygous and heterozygous genotypes, as well as common missing data codes.

        Returns:
            dict: A dictionary mapping genotype strings to IUPAC codes, based on the allele encoding provided in the GenotypeData instance. The keys are genotype strings (e.g., "1/1", "1/2") and the values are the corresponding IUPAC codes (e.g., "A", "W"). Missing data genotypes are mapped to "N".
        """
        iupac_ambig_dict = {
            "AA": "A",
            "CC": "C",
            "GG": "G",
            "TT": "T",
            "AT": "W",
            "TA": "W",
            "AC": "M",
            "CA": "M",
            "AG": "R",
            "GA": "R",
            "CT": "Y",
            "TC": "Y",
            "CG": "S",
            "GC": "S",
            "GT": "K",
            "TG": "K",
            "AN": "N",
            "NA": "N",
            "TN": "N",
            "NT": "N",
            "CN": "N",
            "NC": "N",
            "GN": "N",
            "NG": "N",
            "NN": "N",
        }

        allele_enc = getattr(self, "allele_encoding")

        # Reverse: integer to base (as strings) → e.g., "1": "A"
        allele_map = {str(k): v.upper() for k, v in allele_enc.items()}

        # Homozgyotes (e.g., "1/1": "A")
        iupac_dict = {
            f"{k}/{k}": iupac_ambig_dict.get(base + base, "N")
            for k, base in allele_map.items()
        }

        # Heterozygotes (e.g., "1/2": "W")
        for (k1, b1), (k2, b2) in itertools.combinations(allele_map.items(), 2):
            key1 = f"{k1}/{k2}"
            key2 = f"{k2}/{k1}"  # ensure both orders are covered
            code = iupac_ambig_dict.get(b1 + b2, "N")
            iupac_dict[key1] = code
            iupac_dict[key2] = code

        # Add common missing data codes
        iupac_dict["-1/-1"] = "N"
        iupac_dict["-9/-9"] = "N"

        return iupac_dict

    def _iupac_to_genotype(self, iupac_code: str) -> str:
        """Convert an IUPAC code to its corresponding genotype string.

        Args:
            iupac_code (str): IUPAC encoded nucleotide character.

        Returns:
            str: Corresponding genotype string for the input IUPAC code. Returns '-9/-9' if the IUPAC code is not in the lookup dictionary.

        Raises:
            InvalidGenotypeError: If the input IUPAC code is invalid or if there is an error during processing.
        """
        genotype_dict = self.iupac.get_iupac2gt()

        gt = genotype_dict.get(iupac_code)
        if gt is None:
            msg = f"Invalid IUPAC Code: {iupac_code}"
            self.logger.error(msg)
            raise exceptions.InvalidGenotypeError(msg)
        return gt

    def calc_missing(self, df: pd.DataFrame, *, use_pops: bool = True) -> MissingStats:
        """Compute missing-value statistics with proper locus + sample names.

        Args:
            df (pd.DataFrame): DataFrame with genotype data, where columns are loci
                and rows are individuals.
            use_pops (bool, optional): If True, compute population-level missingness
                stats. Defaults to True.

        Returns:
            MissingStats: A dataclass containing missingness statistics.

        Raises:
            FileNotFoundError: If the HDF5 file with VCF attributes is not found when marker names are derived from it.
            AlignmentFormatError: If there is a mismatch between DataFrame columns and locus names, or if marker names are requested but not found in the data.
        """
        self.logger.info("Calculating missingness statistics...")

        marker_names = getattr(self, "marker_names", None)

        if marker_names is not None:
            locus_names = list(cast(Sequence[str], marker_names))

        elif getattr(self, "from_vcf", False) and getattr(
            self, "vcf_attributes_fn", None
        ):
            hdf = Path(cast(str, self.vcf_attributes_fn))

            if not hdf.is_file():
                msg = f"HDF5 file with VCF attributes not found: {hdf}"
                self.logger.error(msg)
                raise FileNotFoundError(msg)

            with h5py.File(hdf, "r") as h5:
                chrom_ds = cast(h5py.Dataset, h5["chrom"])
                pos_ds = cast(h5py.Dataset, h5["pos"])

                all_chroms = np.asarray(chrom_ds[:], dtype=str)
                all_pos = np.asarray(pos_ds[:], dtype=str)

                loci_filter = getattr(self, "loci_indices", None)

                if loci_filter is not None and loci_filter.size == all_chroms.size:
                    filtered_chroms = all_chroms[loci_filter]
                    filtered_pos = all_pos[loci_filter]
                else:
                    filtered_chroms = all_chroms
                    filtered_pos = all_pos

                locus_names = [
                    f"{c}:{p}" for c, p in zip(filtered_chroms, filtered_pos)
                ]

            df.columns = locus_names

        else:
            locus_names = [f"locus_{i + 1}" for i in range(df.shape[1])]

        if len(df.columns) != len(locus_names):
            all_missing_idx = getattr(self, "all_missing_idx", None)

            if all_missing_idx:
                missing_idx = set(all_missing_idx)
                locus_names = [
                    x for i, x in enumerate(locus_names) if i not in missing_idx
                ]
            else:
                msg = (
                    "Mismatch between DataFrame columns and locus names. "
                    "Check the input data or the locus names."
                )
                self.logger.error(msg)
                raise AlignmentFormatError(msg)

        df.columns = locus_names
        df.index = self.samples

        loc_prop = (df.isna().sum(axis=0) / self.num_inds).round(4)
        ind_prop = (df.isna().sum(axis=1) / self.num_snps).round(4)

        poploc = poptot = indpop = None

        if use_pops and self.has_popmap:
            indpop = df.isna()

            indpop.index = pd.MultiIndex.from_arrays(
                [self._populations, df.index],
                names=["population", "individual"],
            )

            df.index = pd.MultiIndex.from_arrays(
                [self._populations, df.index],
                names=["population", "individual"],
            )

            miss_cnt = indpop.groupby(level="population").sum()
            n_per_pop = indpop.groupby(level="population").size()

            poploc = miss_cnt.div(n_per_pop, axis=0).T.round(4)
            poptot = (miss_cnt.sum(axis=1) / (n_per_pop * self.num_snps)).round(4)

        self.logger.info("Missingness statistics calculated.")

        return MissingStats(
            per_locus=loc_prop,
            per_individual=ind_prop,
            per_population_locus=poploc,
            per_population=poptot,
            per_individual_population=indpop,
        )

    def copy(self) -> "GenotypeData":
        """Create a deep copy of the GenotypeData or subclass object.

        This method creates a deep copy of the GenotypeData or subclass object, ensuring that all attributes are copied appropriately. Attributes that require special handling (e.g., VCF header, logger) are copied without deep copying to avoid issues with uncopyable objects. Subclasses can implement a _post_copy_hook method to clean up any problematic state after copying.

        Returns:
            GenotypeData or subclass: A new object with identical data and structure. The copy is deep for all attributes except those that require special handling (e.g., VCF header, logger). Subclasses can implement a _post_copy_hook method to clean up any problematic state after copying.
        """
        new_obj = cast(Self, self.__class__.__new__(self.__class__))

        # Attributes to exclude from deepcopy due to special handling.
        exclude_attrs = {"vcf_header", "_vcf_attributes_fn", "logger"}

        for name, attr in vars(self).items():
            if name in exclude_attrs:
                setattr(new_obj, name, attr)
            else:
                setattr(new_obj, name, copy.deepcopy(attr))

        # Copy VCF header safely.
        vcf_hdr = getattr(self, "vcf_header", None)

        if vcf_hdr is not None:
            try:
                vcf_hdr_copy = cast(Any, vcf_hdr).copy()
            except Exception:
                vcf_hdr_copy = vcf_hdr

            setattr(new_obj, "vcf_header", vcf_hdr_copy)

        # Reinitialize logger.
        logman = LoggerManager(
            __name__,
            prefix=getattr(self, "prefix", "default"),
            verbose=getattr(self, "verbose", False),
            debug=getattr(self, "debug", False),
        )
        new_obj.logger = logman.get_logger()

        # Allow subclasses to clean up any problematic state.
        post_copy_hook = getattr(new_obj, "_post_copy_hook", None)

        if callable(post_copy_hook):
            cast(Callable[[], None], post_copy_hook)()

        return new_obj

    def load_aln(self):
        """Abstract method to load alignment data. Must be implemented in child classes.

        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        msg = "load_aln method must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def get_population_indices(self) -> Dict[str, List[int]]:
        """Create a mapping from population IDs to sample indices.

        This method creates a dictionary with population IDs as keys and lists of sample indices as values. The sample indices are used to subset the genotype data by population.

        Returns:
            Dict[str, List[int]]: Dictionary with population IDs as keys and lists of sample indices as values.

        Raises:
            AlignmentFormatError: If samples are not defined when popmap_inverse is not available.
        """
        if self.popmap_inverse is None or not self.popmap_inverse:
            samples = getattr(self, "samples", None)
            if samples is None:
                msg = "Samples must be defined to get population indices when popmap_inverse is not available."
                self.logger.error(msg)
                raise AlignmentFormatError(msg)
            return {"NA": samples}

        sample_id_to_index = {
            sample_id: idx for idx, sample_id in enumerate(self.samples)
        }
        pop_indices = {}
        for pop_id, sample_ids in self.popmap_inverse.items():
            indices = [
                sample_id_to_index[sample_id]
                for sample_id in sample_ids
                if sample_id in sample_id_to_index
            ]
            pop_indices[pop_id] = indices
        return pop_indices

    @property
    def locus_names(self) -> list[str]:
        """Concrete locus names, generating defaults if absent.

        Returns:
            list[str]: List of locus names. If marker_names attribute is present, it returns that as a list. Otherwise, it generates default locus names in the format "locus_1", "locus_2", ..., "locus_n" based on the number of SNPs.
        """
        marker_names = getattr(self, "marker_names", None)
        if marker_names is not None:
            return list(marker_names)
        return [f"locus_{i+1}" for i in range(self.num_snps)]

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask [n_samples, n_loci] where True = non-missing genotype.

        Returns:
            np.ndarray: Boolean mask with the same shape as the SNP data, where True indicates a non-missing genotype and False indicates a missing genotype. This mask is computed by taking the logical NOT of the missing_mask, which identifies the positions of missing genotypes in the SNP data. The valid_mask can be used for filtering or analysis that requires only non-missing genotypes.
        """
        return ~self.missing_mask

    @property
    def inputs(self) -> Dict[str, Any]:
        """Get GenotypeData keyword arguments as a dictionary.

        Returns:
            Dict[str, Any]: GenotypeData keyword arguments as a dictionary. This property allows access to the keyword arguments used to initialize the GenotypeData instance. It returns a dictionary where keys are the names of the keyword arguments and values are the corresponding values. This can be useful for inspecting the initial parameters or for debugging purposes.
        """
        return self.kwargs

    @inputs.setter
    def inputs(self, value: Dict[str, Any]) -> None:
        """Setter method for class keyword arguments.

        Args:
            value (Dict[str, Any]): Dictionary of keyword arguments. This method allows setting the class keyword arguments after initialization. The input should be a dictionary where keys are the names of the keyword arguments and values are the corresponding values to set. This can be useful for updating the class state with new input parameters or for resetting the inputs as needed.
        """
        self.kwargs = value

    @property
    def shape(self) -> tuple[int, int]:
        """Tuple of (n_samples, n_loci) for the SNP data.

        Returns:
            tuple[int, int]: Tuple with number of samples (individuals) and number of loci (SNPs).
        """
        return (self.num_inds, self.num_snps)

    @cached_property
    def nbytes(self) -> int:
        """Approximate RAM footprint of snp_data (bytes).

        Returns:
            int: Approximate RAM footprint of snp_data (bytes). This is calculated as the number of elements in the SNP data array multiplied by the item size of the data type. Note that this is an approximation and may not reflect the actual memory usage due to overhead from the array structure and other factors.
        """
        arr = self.snp_data
        # dtype "<U1" is variable-sized in NumPy
        # use itemsize * size as a lower bound
        return arr.size * arr.dtype.itemsize

    @property
    def num_snps(self) -> int:
        """Number of snps (loci) in the dataset.

        Returns:
            int: Number of SNPs (loci) per individual. This is determined by the number of columns in the SNP data matrix. If the SNP data is empty, it returns 0.
        """
        if self.snp_data.size > 0:
            return len(self.snp_data[0])
        return 0

    @num_snps.setter
    def num_snps(self, value: int) -> None:
        """Set the number of SNPs in the dataset.

        Args:
            value (int): Number of SNPs (loci) in the dataset. This should be a positive integer. If the value is not an integer or is negative, an error is raised.

        Raises:
            TypeError: If the value provided is not an integer or cannot be converted to an integer.
            ValueError: If the value provided is negative.
        """
        if not isinstance(value, int):
            try:
                value = int(value)
            except ValueError:
                msg = f"num_snps must be numeric, but got {type(value)}"
                self.logger.error(msg)
                raise TypeError(msg)

        if value < 0:
            msg = f"num_snps must be a positive integer, but got {value}"
            self.logger.error(msg)
            raise ValueError(msg)

        self._num_snps = value

    @property
    def num_inds(self) -> int:
        """Number of individuals (samples) in dataset.

        Returns:
            int: Number of individuals (samples) in input data. This is determined by the number of rows in the SNP data matrix. If the SNP data is empty, it returns 0.
        """
        if self.snp_data.size > 0:
            return len(self.snp_data)
        return 0

    @num_inds.setter
    def num_inds(self, value: int) -> None:
        """Set the number of individuals (samples) in the dataset.

        Args:
            value (int): Number of individuals (samples) in the dataset. This should be a positive integer. If the value is not an integer or is negative, an error is raised.

        Raises:
            TypeError: If the value provided is not an integer or cannot be converted to an integer.
            ValueError: If the value provided is negative.
        """
        if not isinstance(value, int):
            try:
                value = int(value)
            except ValueError:
                msg = f"num_inds must be numeric, but got {type(value)}"
                self.logger.error(msg)
                raise TypeError(msg)

        if value < 0:
            msg = f"num_inds must be a positive integer, but got {value}"
            self.logger.error(msg)
            raise ValueError(msg)

        self._num_inds = value

    @property
    def samples(self) -> list[str]:
        """List of sample IDs in the dataset.

        Returns:
            list[str]: List of sample IDs in the dataset. This is determined by the number of rows in the SNP data matrix. If the SNP data is empty, it returns an empty list. The sample IDs are typically strings that uniquely identify each individual (sample) in the dataset, and they correspond to the rows in the SNP data matrix.
        """
        return self.pop_state.samples

    @samples.setter
    def samples(self, value: list[str]) -> None:
        """Set the list of sample IDs in the dataset.

        Args:
            value (list[str]): List of sample IDs in the dataset. This should be a list of unique sample identifiers corresponding to the rows in the SNP data matrix. The length of the sample list should match the number of individuals (samples) in the dataset, and each entry should be a string representing the sample ID.
        """
        self.pop_state.samples = value
        self._samples = value

    @property
    def populations(self) -> list[str | int]:
        """List of populations in the dataset.

        Returns:
            list[str | int]: List of populations in the dataset. This is determined by the number of unique populations in the SNP data matrix. If the SNP data is empty, it returns an empty list. The population IDs can be either strings or integers, depending on how the population information is encoded in the dataset.
        """
        return self.pop_state.populations

    @populations.setter
    def populations(self, value: list[str | int]) -> None:
        """Set the list of populations in the dataset.

        Args:
            value (list[str | int]): List of populations in the dataset. This should be a list of population IDs corresponding to the samples in the dataset. The length of the population list should match the number of samples, and each entry should indicate the population that the corresponding sample belongs to.
        """
        self.pop_state.populations = value
        self.pop_state.refresh_num_pops()
        self._populations = value

    @property
    def popmap(self) -> dict[str, str | int] | None:
        """Dictionary mapping sample IDs to population IDs.

        Returns:
            dict[str, str | int] | None: Dictionary mapping sample IDs to population IDs. If no population map is provided, it returns None. The keys of the dictionary correspond to the sample IDs in the dataset, and the values correspond to the population IDs that each sample belongs to.
        """
        return self.pop_state.popmap

    @popmap.setter
    def popmap(self, value: dict[str, str | int] | None) -> None:
        """Set the population map, which is a dictionary mapping sample IDs to population IDs.

        Args:
            value (dict[str, str | int] | None): Dictionary mapping sample IDs to population IDs. If no population map is provided, it should be set to None. The keys of the dictionary should correspond to the sample IDs in the dataset, and the values should correspond to the population IDs that each sample belongs to.
        """
        self.pop_state.popmap = value
        self._popmap = value

    @property
    def popmap_inverse(self) -> dict[str | int, list[str]] | None:
        """Dictionary mapping population IDs to lists of sample IDs.

        Returns:
            dict[str | int, list[str]] | None: Dictionary mapping population IDs to lists of sample IDs. If no inverse population map is provided, it returns None. This is the inverse of the popmap dictionary, where each population ID maps to a list of sample IDs that belong to that population.
        """
        return self.pop_state.popmap_inverse

    @popmap_inverse.setter
    def popmap_inverse(self, value: dict[str | int, list[str]] | None) -> None:
        self.pop_state.popmap_inverse = value
        self._popmap_inverse = value

    @property
    def num_pops(self) -> int:
        """Number of populations in the dataset.

        Returns:
            int: Number of unique populations in the dataset. This is determined by the number of unique population IDs in the population list. If the population list is empty, it returns 0.
        """
        return len(set(self.populations))

    @property
    def snpsdict(self) -> Dict[str, List[str]]:
        """Dictionary with Sample IDs as keys and lists of genotypes as values.

        Returns:
            Dict[str, List[str]]: Dictionary with sample IDs as keys and lists of genotypes as values. The keys correspond to the sample IDs in the dataset, and the values are lists of genotype strings for each sample. The length of each genotype list matches the number of SNPs (loci) in the dataset. This property allows access to the snpsdict, which is a convenient format for representing the genotype data as a dictionary. If the snpsdict has not been initialized, it will be created using the _make_snpsdict method, which constructs the dictionary based on the SNP data and sample information. Once initialized, the snpsdict can be accessed directly without needing to recompute it, as it is stored as an attribute of the GenotypeData instance.
        """
        self._snpsdict = self._make_snpsdict()
        return self._snpsdict

    @snpsdict.setter
    def snpsdict(self, value: Dict[str, List[str]]) -> None:
        """Set snpsdict object, which is a dictionary with sample IDs as keys and lists of genotypes as values.

        Args:
            value (Dict[str, List[str]]): Dictionary with sample IDs as keys and lists of genotypes as values. The keys should correspond to the sample IDs in the dataset, and the values should be lists of genotype strings for each sample. The length of each genotype list should match the number of SNPs (loci) in the dataset. This setter allows for updating the snpsdict with new genotype data or for initializing it after loading the data. It is important to ensure that the structure of the input dictionary matches the expected format for proper functionality of the GenotypeData class.
        """
        self._snpsdict = value

    @property
    def loci_indices(self) -> np.ndarray:
        """Boolean array for retained loci in filtered alignment.

        Returns:
            np.ndarray: Boolean array of loci indices, with True for retained loci and False for excluded loci. This array can be used to filter the SNP data matrix, retaining only the loci that are marked as True.

        Raises:
            TypeError: If the loci_indices attribute is not a numpy.ndarray or list.
            TypeError: If the loci_indices attribute is not a numpy.dtype 'bool'.
        """

        if not isinstance(self._loci_indices, (np.ndarray, list, type(None))):
            msg = f"'loci_indices' set to invalid type. Expected numpy.ndarray or list, but got: {type(self._loci_indices)}"
            self.logger.error(msg)
            raise TypeError(msg)

        elif isinstance(self._loci_indices, list):
            self._loci_indices = np.array(self._loci_indices)

        if self._loci_indices is None or not self._loci_indices.size > 0:
            self._loci_indices = np.ones(self.num_snps, dtype=bool)

        else:
            if self._loci_indices.dtype != bool:
                msg = f"'loci_indices' must be numpy.dtype 'bool', but got: {self._loci_indices.dtype}"
                self.logger.error(msg)
                raise TypeError(msg)

        return self._loci_indices

    @loci_indices.setter
    def loci_indices(self, value: np.ndarray | list) -> None:
        """Set the loci indices as a boolean array.

        Args:
            value (np.ndarray | list): Boolean array of loci indices, with True for retained loci and False for excluded loci. This setter allows for updating the loci_indices with new filtering criteria. The input can be either a numpy array or a list, but it must ultimately be converted to a numpy array of dtype bool. If the input is None, it defaults to an array of True values, indicating that all loci are retained.

        Raises:
            TypeError: If the input value is not a numpy.ndarray or list.
            TypeError: If the input value cannot be converted to a numpy array of dtype bool.
        """
        if value is None:
            value = np.ones(self.num_snps, dtype=bool)
        if isinstance(value, list):
            value = np.array(value)
        if value.dtype != bool:
            msg = f"Attempt to set 'loci_indices' to unexpected dtype. Expected bool, got: {value.dtype}"
            self.logger.error(msg)
            raise TypeError(msg)
        self._loci_indices = value

    @property
    def sample_indices(self) -> np.ndarray:
        """Boolean array for retained samples in filtered alignment.

        Returns:
            np.ndarray: Boolean array of sample indices, with True for retained samples and False for excluded samples. This array can be used to filter the SNP data matrix, retaining only the samples that are marked as True.

        Raises:
            TypeError: If the sample_indices attribute is not a numpy.ndarray or list.
            TypeError: If the sample_indices attribute is not a numpy.dtype 'bool'.
        """
        if not isinstance(self._sample_indices, (np.ndarray, list, type(None))):
            msg = f"'sample_indices' set to invalid type. Expected numpy.ndarray, list, or None, but got: {type(self._sample_indices)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if isinstance(self._sample_indices, list):
            self._sample_indices = np.array(self._sample_indices)

        if self._sample_indices is None or self._sample_indices.size == 0:
            self._sample_indices = np.ones(len(self.samples), dtype=bool)
        else:
            if self._sample_indices.dtype != bool:
                msg = f"'sample_indices' must be dtype bool, but got: {self._sample_indices.dtype}"
                self.logger.error(msg)
                raise TypeError(msg)

        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, value: np.ndarray | list) -> None:
        """Set the sample indices as a boolean array.

        Args:
            value (np.ndarray | list): Boolean array of sample indices, with True for retained samples and False for excluded samples. This setter allows for updating the sample_indices with new filtering criteria. The input can be either a numpy array or a list, but it must ultimately be converted to a numpy array of dtype bool. If the input is None, it defaults to an array of True values, indicating that all samples are retained.

        Raises:
            TypeError: If the input value is not a numpy.ndarray or list.
            TypeError: If the input value cannot be converted to a numpy array of dtype bool.
        """
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
        """Get list of reference alleles of length num_snps.

        Returns:
            List[str]: List of reference alleles of length num_snps. This property returns the list of reference alleles for each SNP (locus) in the dataset. The length of the list should match the number of SNPs, and each entry should be a string representing the reference allele for that SNP. If the reference alleles are stored as a numpy array, they are converted to a list of strings before being returned.
        """
        if isinstance(self._ref, np.ndarray):
            return self._ref.astype(str).tolist()
        return self._ref

    # In the GenotypeData class

    @ref.setter
    def ref(self, value: List[str] | np.ndarray) -> None:
        """Setter for list of reference alleles of length num_snps.

        Args:
            value (List[str] | np.ndarray): List of reference alleles of length num_snps. This setter allows for updating the list of reference alleles for each SNP (locus) in the dataset. The input can be either a list of strings or a numpy array. If the input is a numpy array, it is converted to a list of strings before being stored. The length of the input should match the number of SNPs in the dataset, and each entry should be a string representing the reference allele for that SNP.
        """
        if isinstance(value, np.ndarray):
            # .tolist() converts numpy string types to python strings
            self._ref = value.tolist()
        else:
            self._ref = value

    @property
    def alt(self) -> List[List[str]]:
        """Get list of alternate alleles of length num_snps.

        Returns:
            List[List[str]]: List of alternate alleles of length num_snps.
        """
        if isinstance(self._alt, np.ndarray):
            return self._alt.astype(str).tolist()
        return self._alt

    @alt.setter
    def alt(self, value: List[List[str]] | np.ndarray) -> None:
        """Setter for list of alternate alleles of length num_snps.

        Args:
            value (List[List[str]] | np.ndarray): List of alternate alleles of length num_snps. This setter allows for updating the list of alternate alleles for each SNP (locus) in the dataset. The input can be either a list of lists of strings or a numpy array. If the input is a numpy array, it is converted to a list of lists of strings before being stored. The length of the input should match the number of SNPs in the dataset, and each entry should be a list of strings representing the alternate alleles for that SNP.
        """
        if isinstance(value, list) and value:
            # Recursively convert any numpy strings to python strings
            # This handles the nested list structure for multi-allelic sites
            self._alt = [[str(allele) for allele in inner_list] for inner_list in value]
        elif isinstance(value, np.ndarray):
            # Fallback for a simple numpy array
            self._alt = value.tolist()
        else:
            self._alt = value

    @property
    def snp_data(self) -> np.ndarray:
        """Get the genotypes as a 2D list of shape (n_samples, n_loci).

        Returns:
            np.ndarray: 2D array of IUPAC encoded genotype data. The shape of the array is (n_samples, n_loci), where n_samples is the number of individuals (samples) in the dataset and n_loci is the number of SNPs (loci). Each entry in the array is a string representing the genotype for a particular sample at a particular locus, encoded using IUPAC ambiguity codes. If the snp_data attribute is not already loaded, this property will trigger the loading of the alignment data and population map to populate the snp_data before returning it. The returned array is ensured to have a dtype of Unicode string ("<U1") for consistency.

        Raises:
            TypeError: If the snp_data attribute is not a numpy.ndarray, pandas.DataFrame, or list. This ensures that the snp_data is in a valid format before attempting to return it.
        """
        if (
            self._snp_data is None
            or (isinstance(self._snp_data, np.ndarray) and self._snp_data.size == 0)
            or (isinstance(self._snp_data, list) and len(self._snp_data) == 0)
            or (isinstance(self._snp_data, pd.DataFrame) and self._snp_data.empty)
        ):
            self.load_aln()
            self.read_popmap()

        if isinstance(self._snp_data, (np.ndarray, pd.DataFrame, list)):
            if isinstance(self._snp_data, list):
                return np.array(self._snp_data, dtype="<U1")
            elif isinstance(self._snp_data, pd.DataFrame):
                return self._snp_data.to_numpy(dtype="<U1")
            return self._snp_data.astype("<U1")  # Ensure Unicode string
        else:
            msg = f"Invalid 'snp_data' type. Expected numpy.ndarray, pandas.DataFrame, or list, but got: {type(self._snp_data)}"
            self.logger.error(msg)
            raise TypeError(msg)

    @snp_data.setter
    def snp_data(self, value: np.ndarray | List[List[str]] | pd.DataFrame) -> None:
        """Set snp_data attribute as a 2D NumPy array of IUPAC encoded genotype data.

        Args:
            value (np.ndarray | List[List[str]] | pd.DataFrame): 2D array of IUPAC encoded genotype data. The input can be a numpy array, a list of lists of strings, or a pandas DataFrame. The shape of the input should be (n_samples, n_loci), where n_samples is the number of individuals (samples) in the dataset and n_loci is the number of SNPs (loci). Each entry in the input should be a string representing the genotype for a particular sample at a particular locus, encoded using IUPAC ambiguity codes.

        Raises:
            TypeError: If the input value is not a numpy.ndarray, list of lists, or pandas.DataFrame. This ensures that the snp_data is set to a valid format that can be processed correctly.
        """
        if isinstance(value, list):
            value = np.array(value, dtype="<U1")
        elif isinstance(value, pd.DataFrame):
            value = value.to_numpy(dtype="<U1")
        elif isinstance(value, np.ndarray):
            value = value.astype("<U1")
        else:
            msg = (
                "Attempt to set 'snp_data' to invalid type. Must be a list, "
                f"numpy.ndarray, pandas.DataFrame, but got: {type(value)}"
            )
            self.logger.debug(value)
            self.logger.error(msg)
            raise TypeError(msg)

        self._snp_data = value
        self._validate_seq_lengths()
        self._invalidate_caches()

    @property
    def output_dir(self) -> Path:
        """Root output directory for this dataset.

        Returns:
            Path: Path to the root output directory. This is typically constructed using the prefix attribute of the GenotypeData instance, with a suffix of "_output". The output directory is used as the base location for saving any results, reports, or plots generated from the dataset. If the prefix attribute is not set, it defaults to "default_output".
        """
        return OutputPaths.from_genotype_data(self).root

    @property
    def reports_dir(self) -> Path:
        """Standardized location for reports (pre/post-filtering aware).

        Returns:
            Path: The report root for the current dataset. Filtered datasets use
            ``<prefix>_output/reports/nremover``; unfiltered datasets use
            ``<prefix>_output/reports``.
        """
        return OutputPaths.from_genotype_data(self).reports()

    @property
    def plot_kwargs(self) -> dict:
        """Backwards compatibility; convert PlotConfig to the old dict shape.

        Returns:
            dict: Dictionary of plot configuration parameters. This property provides backwards compatibility by converting the PlotConfig object to a dictionary format that was used in previous versions of the code. The returned dictionary contains key-value pairs corresponding to the configuration parameters for plotting, allowing existing code that expects a dictionary to continue functioning without modification. The conversion is done using the to_dict method of the PlotConfig class, which ensures that all relevant configuration parameters are included in the resulting dictionary.
        """
        return self.plot_config.to_dict()

    @property
    def plots_dir(self) -> Path:
        """Standardized location for plots (pre/post-filtering aware).

        Returns:
            Path: The plot root for the current dataset. Filtered datasets use
            ``<prefix>_output/plots/nremover``; unfiltered datasets use
            ``<prefix>_output/plots``.
        """
        return OutputPaths.from_genotype_data(self).plots()

    @cached_property
    def het_mask(self) -> np.ndarray:
        """[n_samples, n_loci] True if genotype is heterozygous (IUPAC ambiguity codes).

        Returns:
            np.ndarray: Boolean mask where True indicates a heterozygous genotype. This mask is computed by checking if the genotype at each position in the SNP data corresponds to one of the IUPAC ambiguity codes that represent heterozygous genotypes (e.g., "R", "Y", "M", "K", "S", "W"). The resulting boolean array has the same shape as the SNP data, with True values indicating heterozygous genotypes and False values indicating homozygous or missing genotypes.
        """
        ambig = np.array(list("WRMKYS"), dtype="<U1")
        return np.isin(self.snp_data, ambig)

    @cached_property
    def per_locus_het_rate(self) -> pd.Series:
        """Heterozygote proportion per locus (ignores missing).

        Returns:
            pd.Series: Heterozygote proportion per locus as a pandas Series indexed by locus name. This is calculated by taking the sum of heterozygous genotypes (where the het_mask is True) for each locus and dividing it by the total number of valid (non-missing) genotypes for that locus. The resulting Series provides the proportion of heterozygous genotypes at each locus, allowing for analysis of heterozygosity across the dataset. The values are rounded to four decimal places for readability.
        """
        vm = self.valid_mask
        hm = self.het_mask
        denom = vm.sum(axis=0).clip(min=1)
        vals = (hm & vm).sum(axis=0) / denom
        return pd.Series(vals, index=self.locus_names, name="het_rate").round(4)

    @cached_property
    def per_individual_het_rate(self) -> pd.Series:
        """Heterozygote proportion per individual (ignores missing).

        Returns:
            pd.Series: Heterozygote proportion per individual as a pandas Series indexed by sample name. This is calculated by taking the sum of heterozygous genotypes (where the het_mask is True) for each individual and dividing it by the total number of valid (non-missing) genotypes for that individual. The resulting Series provides the proportion of heterozygous genotypes for each individual, allowing for analysis of heterozygosity across samples in the dataset. The values are rounded to four decimal places for readability.
        """
        vm = self.valid_mask
        hm = self.het_mask
        denom = vm.sum(axis=1).clip(min=1)
        vals = (hm & vm).sum(axis=1) / denom
        return pd.Series(vals, index=self.samples, name="het_rate").round(4)

    @cached_property
    def missing_mask(self) -> np.ndarray:
        """Boolean mask [n_samples, n_loci] where True indicates a missing genotype.

        Returns:
            np.ndarray: Boolean mask where True indicates a missing genotype. This mask is computed by checking if the genotype at each position in the SNP data corresponds to one of the values in the missing_vals list. The resulting boolean array has the same shape as the SNP data, with True values indicating missing genotypes and False values indicating valid genotypes.
        """
        # build a mask by multi-compare against the small missing set
        arr = self.snp_data
        misses = np.isin(arr, np.array(self.missing_vals, dtype=arr.dtype))
        return misses

    @property
    def missing_rate(self) -> float:
        """Overall missing proportion in the alignment.

        Returns:
            float: Overall missing proportion in the alignment. This is calculated by taking the mean of the missing_mask, which gives the proportion of genotypes that are missing across the entire dataset. If there are no genotypes (i.e., the missing_mask is empty), it returns 0.0 to avoid division by zero.
        """
        mm = self.missing_mask
        if mm.size == 0:
            return 0.0
        return float(mm.mean())

    @property
    def per_individual_missing(self) -> pd.Series:
        """Missing proportion per sample as a pandas Series indexed by sample name.

        Returns:
            pd.Series: Missing proportion per sample as a pandas Series indexed by sample name. This is calculated by taking the mean of the missing_mask along the rows, which gives the proportion of genotypes that are missing for each individual. The resulting Series provides the missing proportion for each sample, allowing for analysis of missing data across individuals in the dataset. The values are rounded to four decimal places for readability.
        """
        mm = self.missing_mask
        vals = mm.mean(axis=1)
        return pd.Series(vals, index=self.samples, name="missing_prop").round(4)

    @property
    def per_locus_missing(self) -> pd.Series:
        """Missing proportion per locus as a pandas Series; uses marker names if present.

        Returns:
            pd.Series: Missing proportion per locus as a pandas Series indexed by locus name. This is calculated by taking the mean of the missing_mask along the columns, which gives the proportion of genotypes that are missing for each locus. The resulting Series provides the missing proportion for each locus, allowing for analysis of missing data across loci in the dataset. The values are rounded to four decimal places for readability. If marker names are present in the dataset, they are used as the index for the Series; otherwise, generic locus names (e.g., "locus_1", "locus_2", etc.) are generated and used as the index. This allows for more informative labeling of the loci in the resulting Series when marker names are available.
        """
        mm = self.missing_mask
        cols = (
            self.marker_names
            if getattr(self, "marker_names", None) is not None
            else [f"locus_{i+1}" for i in range(mm.shape[1])]
        )
        vals = mm.mean(axis=0)
        return pd.Series(vals, index=cols, name="missing_prop").round(4)

    @property
    def sample_index_map(self) -> Dict[str, int]:
        """Map sample ID -> row index (useful for subsetting).

        Returns:
            Dict[str, int]: Mapping of sample IDs to their corresponding row indices. This dictionary is constructed by enumerating over the list of sample IDs and creating key-value pairs where the key is the sample ID and the value is the index of that sample in the SNP data matrix. This mapping is useful for subsetting the data based on sample IDs, allowing for easy retrieval of the corresponding row indices for any given set of sample IDs.
        """
        return {s: i for i, s in enumerate(self.samples)}

    @property
    def pop_sizes(self) -> Dict[str | int, int]:
        """Population -> sample count.

        Returns:
            Dict[str | int, int]: Mapping of population names to sample counts. This dictionary is constructed by iterating over the list of populations and counting the number of samples that belong to each population. The keys of the dictionary correspond to the unique population IDs, and the values represent the count of samples for each population. This information is useful for understanding the distribution of samples across different populations in the dataset.
        """
        counts: Dict[str | int, int] = {}
        for p in self.populations:
            counts[p] = counts.get(p, 0) + 1
        return counts

    @property
    def pop_to_indices(self) -> Dict[str | int, list[int]]:
        """Population -> list of sample indices (built from current popmap_inverse).

        Returns:
            Dict[str | int, list[int]]: Mapping of population names to lists of sample indices. This dictionary is constructed by iterating over the popmap_inverse, which maps populations to sample IDs, and then converting these sample IDs to their corresponding row indices using the sample_index_map. This allows for easy retrieval of the indices of samples belonging to each population, facilitating population-specific analyses.

        """
        out: Dict[str | int, list[int]] = {}
        if self.popmap_inverse is None:
            return out
        idx_map = self.sample_index_map
        for pop, ids in self.popmap_inverse.items():
            out[pop] = [idx_map[s] for s in ids if s in idx_map]
        return out

    @property
    def is_empty(self) -> bool:
        """True if there are zero samples or loci.

        Returns:
            bool: True if there are zero samples or loci.
        """
        return self.num_inds == 0 or self.num_snps == 0

    @property
    def has_popmap(self) -> bool:
        """True if population information is present.

        Returns:
            bool: True if population information is present. This is determined by checking if the popmap attribute is not None, which indicates that a population map has been provided and population information is available for the samples in the dataset.
        """
        return self.popmapfile is not None

    @cached_property
    def is_missing_locus(self) -> npt.NDArray[np.bool_]:
        """[n_loci] True if an entire locus is missing across all samples.

        Returns:
            npt.NDArray[np.bool_]: Boolean array indicating which loci are missing across all samples. This is computed by checking the valid_mask for each locus and determining if all samples have missing genotypes at that locus. The resulting array has a length equal to the number of loci, with True values indicating loci that are completely missing across all samples and False values indicating loci that have at least one valid genotype.
        """
        return np.asarray((~self.valid_mask).all(axis=0), dtype=np.bool_)

    @cached_property
    def observed_iupac_per_locus(self) -> list[set[str]]:
        """Observed IUPAC codes per locus (excluding missing).

        Returns:
            list[set[str]]: List of sets of observed IUPAC codes for each locus, excluding missing values. This is computed by iterating over each locus and collecting the unique IUPAC codes present in the SNP data for that locus, while ignoring any genotypes that are marked as missing according to the valid_mask. The resulting list has a length equal to the number of loci, with each entry being a set of unique IUPAC codes observed at that locus across all samples.
        """
        arr = self.snp_data
        vm = self.valid_mask
        return [set(arr[vm[:, j], j]) for j in range(arr.shape[1])]

    @cached_property
    def biallelic_mask(self) -> np.ndarray:
        """[n_loci] True where locus appears biallelic (A/C/G/T plus heterozygotes of those two).

        Returns:
            np.ndarray: Boolean array indicating which loci are biallelic. A locus is considered biallelic if it has exactly two unambiguous nucleotides (A, C, G, T) observed across all samples, along with any heterozygous genotypes that can be formed from those two nucleotides. The resulting array has a length equal to the number of loci, with True values indicating biallelic loci and False values indicating loci that are not biallelic (e.g., monomorphic or multiallelic).
        """
        obs = self.observed_iupac_per_locus

        # Extract unambiguous nucleotides present
        def nucs(s):
            return {x for x in s if x in {"A", "C", "G", "T"}}

        return np.array([len(nucs(s)) == 2 for s in obs], dtype=bool)

    @property
    def has_multiallelic(self) -> bool:
        """True if any locus shows >2 unambiguous nucleotides.


        Returns:
            bool: True if any locus shows more than two unambiguous nucleotides. This is determined by iterating over the observed IUPAC codes for each locus and checking if the number of unambiguous nucleotides (A, C, G, T) exceeds two. If any locus has more than two unambiguous nucleotides, it indicates the presence of multiallelic loci in the dataset, and this property will return True. Otherwise, it returns False, indicating that all loci are either monomorphic or biallelic.
        """
        obs = self.observed_iupac_per_locus
        return any(len({x for x in s if x in {"A", "C", "G", "T"}}) > 2 for s in obs)

    def _validate_seq_lengths(self) -> None:
        """Ensure that all SNP data rows have the same length.

        Raises:
            SequenceLengthError: If the sequence lengths are not all the same. This checks that each row in the SNP data has the same number of loci (columns). If any row has a different length than the first row, it raises a SequenceLengthError with a message indicating which sample has the invalid sequence length and what the expected and actual lengths are. This validation is important to ensure that the SNP data is properly formatted and can be processed correctly in downstream analyses.
        """
        lengths = set([len(row) for row in self.snp_data])
        if len(lengths) > 1:
            n_snps = len(self.snp_data[0])

            for i, row in enumerate(self.snp_data):
                if len(row) != n_snps:
                    msg = f"Invalid sequence length for Sample {self.samples[i]}. Expected {n_snps}, but got: {len(row)}"
                    self.logger.error(msg)
                    raise exceptions.SequenceLengthError(self.samples[i])

    def _invalidate_caches(self) -> None:
        """Clear cached_property results after data changes."""
        for attr in (
            "missing_mask",
            "valid_mask",
            "is_missing_locus",
            "observed_iupac_per_locus",
            "biallelic_mask",
            "per_locus_missing",
            "per_individual_missing",
            "het_mask",
            "per_locus_het_rate",
            "per_individual_het_rate",
            "maf_per_locus",
            "nbytes",
        ):
            if attr in self.__dict__:  # cached_property stores value here
                self.__dict__.pop(attr, None)

    @staticmethod
    def _validate_filtered_matrix_and_masks(
        snp_data: np.ndarray,
        samples: Sequence[str],
        sample_indices: np.ndarray,
        loci_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate a filtered matrix against its parent-coordinate masks."""

        data = np.asarray(snp_data)
        sample_mask = np.asarray(sample_indices)
        locus_mask = np.asarray(loci_indices)

        if data.ndim != 2:
            msg = f"Filtered SNP data must be two-dimensional; got shape={data.shape}."
            raise AlignmentFormatError(msg)

        for name, mask in (
            ("sample_indices", sample_mask),
            ("loci_indices", locus_mask),
        ):
            if mask.ndim != 1:
                raise AlignmentFormatError(
                    f"{name} must be one-dimensional; got shape={mask.shape}."
                )
            if mask.dtype != bool:
                raise TypeError(
                    f"{name} must have boolean dtype; got dtype={mask.dtype}."
                )

        expected_shape = (
            int(np.count_nonzero(sample_mask)),
            int(np.count_nonzero(locus_mask)),
        )
        if data.shape != expected_shape:
            raise AlignmentFormatError(
                "Filtered SNP data shape does not match the retained-mask counts: "
                f"data shape={data.shape}, expected={expected_shape}."
            )

        if len(samples) != data.shape[0]:
            raise AlignmentFormatError(
                "Filtered sample metadata does not match the genotype rows: "
                f"samples={len(samples)}, rows={data.shape[0]}."
            )

        return sample_mask, locus_mask

    def _validate_alignment_update(
        self,
        snp_data: np.ndarray,
        samples: Sequence[str],
        sample_indices: np.ndarray,
        loci_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate a filtered alignment and its parent-coordinate masks.

        The masks describe retained rows and columns in the parent alignment,
        while ``snp_data`` and ``samples`` already contain only the retained
        data. Their nonzero counts must therefore match the filtered matrix.
        """

        sample_mask, locus_mask = self._validate_filtered_matrix_and_masks(
            snp_data,
            samples,
            sample_indices,
            loci_indices,
        )

        marker_names = getattr(self, "marker_names", None)
        if marker_names is not None and len(marker_names) != locus_mask.size:
            raise AlignmentFormatError(
                "Cannot align locus names with the filtered genotype matrix: "
                f"mask length={locus_mask.size}, marker names={len(marker_names)}."
            )

        if self.from_vcf:
            if len(self.ref) != locus_mask.size or len(self.alt) != locus_mask.size:
                raise AlignmentFormatError(
                    f"Cannot align filtered VCF REF/ALT metadata with the genotype matrix: mask length={locus_mask.size}, REF length={len(self.ref)}, ALT length={len(self.alt)}."
                )

        return sample_mask, locus_mask

    def set_alignment(
        self,
        snp_data: np.ndarray,
        samples: List[str],
        sample_indices: np.ndarray,
        loci_indices: np.ndarray,
        reset_attributes: bool = True,
    ) -> None:
        """Set the alignment data and sample IDs after filtering.

        This method is used to update the genotype data and associated metadata after applying filters to the dataset. It takes in the new SNP data matrix, the list of sample IDs, and the boolean arrays indicating which samples and loci are retained after filtering. It also has an option to reset the VCF attributes if the dataset is VCF-backed. After updating the internal state with the new data, it invalidates any cached properties to ensure that they will be recalculated based on the updated data when accessed next.

        Args:
            snp_data (np.ndarray): 2D array of genotype data.
            samples (List[str]): List of sample IDs.
            sample_indices (np.ndarray): Boolean array of sample indices.
            loci_indices (np.ndarray): Boolean array of locus indices.
            reset_attributes (bool): If True, update the VCF attributes metadata.

        Raises:
            TypeError: If the input data types are not as expected. This ensures that the method is called with the correct types of data for proper functionality.
            AlignmentFormatError: If sample information is missing when attempting to update VCF attributes. This ensures that the necessary information is available for updating VCF attributes when required.
        """
        sample_mask, locus_mask = self._validate_alignment_update(
            snp_data,
            samples,
            sample_indices,
            loci_indices,
        )
        original_was_filtered = bool(self.was_filtered)
        original_ref = list(self.ref) if self.from_vcf else []
        original_alt = list(self.alt) if self.from_vcf else []

        if self.filetype is None:
            msg = (
                "filetype is not set; cannot determine if VCF attributes "
                "need to be updated."
            )
            self.logger.error(msg)
            raise TypeError(msg)

        samps = list(samples)
        updated_vcf_attributes: Path | None = None
        if reset_attributes and self.filetype.startswith("vcf"):
            # Complete the derived metadata artifact before mutating this
            # object, so a failed HDF5 write leaves the current state intact.
            updated_vcf_attributes = self.update_vcf_attributes(
                snp_data=snp_data,
                sample_indices=sample_mask,
                loci_indices=locus_mask,
                samples=np.asarray(samps, dtype=str),
            )

        self.snp_data = snp_data
        self.samples = samps
        self.sample_indices = sample_mask
        self.loci_indices = locus_mask

        if hasattr(self, "marker_names"):
            self.marker_names = (
                np.asarray(self.marker_names)[locus_mask].tolist()
                if self.marker_names is not None
                else None
            )
        else:
            self.marker_names = None

        self.num_inds = self.snp_data.shape[0]
        self.num_snps = self.snp_data.shape[1]
        self.was_filtered = original_was_filtered or (
            not np.all(sample_mask) or not np.all(locus_mask)
        )

        if self.popmap is not None:
            self.popmap = {str(s): self.popmap[str(s)] for s in samples}
            self.populations = [self.popmap[s] for s in samples]
            self.popmap_inverse = {}
            for s, p in self.popmap.items():
                self.popmap_inverse.setdefault(p, []).append(s)

        # Determine REF and ALT from SNP data matrix if not loading from a VCF.
        # If loading VCF, self._ref and self._alt are set in get_vcf_attributes.
        if not self.from_vcf:
            ref_alleles, alt_alleles = self.get_ref_alt_alleles(self.snp_data)
            self.ref = ref_alleles
            self.alt = alt_alleles

        if updated_vcf_attributes is not None:
            self.vcf_attributes_fn = updated_vcf_attributes

        if self.from_vcf:
            retained = np.flatnonzero(locus_mask)
            self.ref = [original_ref[i] for i in retained]
            self.alt = [original_alt[i] for i in retained]

        # Encoder-generated locus indices are relative to the previous
        # alignment. Clear them so they cannot be applied to the new columns.
        self.all_missing_idx = []

        if hasattr(self, "num_records"):
            self.num_records = self.num_snps

        self.pop_state.refresh_num_pops()

        self._invalidate_caches()

    def update_vcf_attributes(
        self,
        snp_data: np.ndarray,
        sample_indices: np.ndarray,
        loci_indices: np.ndarray,
        samples: np.ndarray,
    ) -> Path | None:
        """Update VCF attributes after genotype data changes.

        Subclasses that support VCF-backed data should override this method.

        Args:
            snp_data: SNP genotype matrix.
            sample_indices: Boolean or integer sample index mask.
            loci_indices: Boolean or integer locus index mask.
            samples: Sample names.

        Returns:
            Path or string to the updated VCF attributes file.

        Raises:
            NotImplementedError: If the subclass does not support VCF attributes.
        """
        nm = self.__class__.__name__
        msg = f"{nm} does not implement 'update_vcf_attributes()'."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def __len__(self) -> int:
        """Return the number of individuals in the dataset.

        Returns:
            int: The number of individuals in the dataset. This is determined by the length of the ``snp_data`` attribute, which should correspond to the number of samples (individuals) in the dataset. If ``snp_data`` is not properly loaded or is empty, this will return 0.
        """
        return len(self.snp_data)

    def __getitem__(self, index: int | str) -> List[str]:
        """Return the genotypes for a specific sample index.

        Args:
            index (int | str): Index of the sample to retrieve. This can be either an integer index corresponding to the row in the SNP data matrix or a string representing the sample ID. If a string is provided, it will be converted to the corresponding integer index using the list of sample IDs.

        Returns:
            List[str]: The genotypes for the specified sample index. This is returned as a list of strings, where each string represents the genotype for that sample at a particular locus, encoded using IUPAC ambiguity codes. The length of the returned list should match the number of loci in the dataset.
        """
        if isinstance(index, str):
            index = self.samples.index(index)
        return self.snp_data[index]

    def __iter__(self) -> Generator[List[str], None, None]:
        """Iterate over each sample in the dataset.

        Returns:
            Generator[List[str], None, None]: A generator that yields the genotypes for each sample in the dataset. Each item yielded is a list of strings representing the genotypes for a particular sample across all loci, encoded using IUPAC ambiguity codes. The generator allows for efficient iteration over the samples without loading all data into memory at once.
        """
        for i in range(self.num_inds):
            yield self.snp_data[i]

    def __contains__(self, individual: str) -> bool:
        """Check if an individual is present in the dataset.

        Args:
            individual (str): The sample ID of the individual to check for presence in the dataset.

        Returns:
            bool: True if the individual is present in the dataset, False otherwise. This checks if the provided sample ID exists in the list of samples associated with the genotype data. If the sample ID is found in the list, it returns True; if not, it returns False.
        """
        return individual in self.samples

    def __str__(self) -> str:
        """Return a string representation of the GenotypeData object.

        Returns:
            str: A string representation of the GenotypeData object, including the number of SNPs, number of individuals, and the source file information. This provides a concise summary of the dataset, allowing users to quickly understand the key characteristics of the genotype data when printing or logging the object.
        """
        return f"GenotypeData: {self.num_snps} SNPs, {self.num_inds} individuals, from file: {self.filename} and popmapfile: {self.popmapfile}. Use 'repr()' for more details."

    def __repr__(self) -> str:
        """Return a detailed string representation of the GenotypeData object.

        Returns:
            str: A detailed string representation of the GenotypeData object, including all relevant attributes and their values. This provides a comprehensive view of the object's state, useful for debugging and logging purposes. The representation includes the filename, filetype, population map file, force_popmap flag, excluded and included populations, plot format, and prefix, giving a complete overview of the configuration and source information for the genotype data.
        """
        # All non-iterable GenotypeData properties.
        return f"GenotypeData(filename={self.filename}, filetype={self.filetype}, popmapfile={self.popmapfile}, force_popmap={self.force_popmap}, exclude_pops={self.exclude_pops}, include_pops={self.include_pops}, plot_format={self.plot_format}, prefix={self.prefix}, num_snps={self.num_snps}, num_inds={self.num_inds}, was_filtered={self.was_filtered}, has_popmap={self.has_popmap})"
