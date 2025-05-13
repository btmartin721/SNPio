import copy
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import snpio.utils.custom_exceptions as exceptions
from snpio.plotting.plotting import Plotting
from snpio.read_input.genotype_data_base import BaseGenotypeData
from snpio.read_input.popmap_file import ReadPopmap
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


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

        _report2file: Write a DataFrame to a CSV file.

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
        plot_format: Literal["png", "pdf", "jpg", "svg"] | None = "png",
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
            filename (str, optional): Path to input file containing genotypes. Defaults to None.

            filetype (str, optional): Type of input genotype file. Possible values include: 'phylip', 'structure', 'vcf', 'tree', or '012'. Defaults to None.

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

            verbose (bool, optional): If True, display verbose output. Defaults to False.

            loci_indices (np.ndarray, optional): Column indices for retained loci in filtered alignment. Defaults to None.

            sample_indices (np.ndarray, optional): Row indices for retained samples in the alignment. Defaults to None.

            chunk_size (int, optional): Chunk size for reading in large files. Defaults to 1000.

            logger (logging.Logger, optional): Logger object. Defaults to None.

            debug (bool, optional): If True, display debug messages. Defaults to False.

        Raises:
            exceptions.UnsupportedFileTypeError: If the filetype is not supported.

        Note:
            If using PHYLIP or STRUCTURE formats, all sites will be forced to be biallelic. If multiple alleles are needed, you must use a VCF file.
        """
        if filetype is not None:
            filetype = filetype.lower()

        self.prefix = prefix
        self.verbose = verbose
        self.debug = debug

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
        self.chunk_size = chunk_size

        self.supported_filetypes: List[str] = ["vcf", "phylip", "structure", "tree"]
        self._snp_data = None
        self.from_vcf = False

        self.logger: logging.Logger | None = None if logger is None else logger

        if self.logger is None:
            kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
            logman = LoggerManager(__name__, **kwargs)
            self.logger = logman.get_logger()

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

        self.plot_kwargs = {
            "plot_format": self.plot_format,
            "plot_fontsize": self.plot_fontsize,
            "dpi": self.plot_dpi,
            "despine": self.plot_despine,
            "show": self.show_plots,
            "verbose": self.verbose,
            "debug": self.debug,
        }

        self.missing_vals = ["N", "-", ".", "?"]
        self.replace_vals = [pd.NA] * len(self.missing_vals)

        self._samples: List[str] = []
        self._populations: List[str | int] = []
        self._ref: List[str] = []
        self._alt: List[str] = []
        self._popmap: Dict[str, str | int] | None = None
        self._popmap_inverse: Dict[str, List[str]] | None = None

        self._loci_indices = loci_indices
        self._sample_indices = sample_indices

        if self.filetype not in self.supported_filetypes:
            msg = f"Unsupported filetype provided to GenotypeData: {self.filetype}"
            self.logger.error(msg)
            raise exceptions.UnsupportedFileTypeError(
                self.filetype, supported_types=self.supported_filetypes
            )

        self.kwargs["filetype"] = self.filetype
        self.kwargs["loci_indices"] = self._loci_indices
        self.kwargs["sample_indices"] = self._sample_indices

        self.iupac_mapping: Dict[Tuple[str, str], str] = self._iupac_from_gt_tuples()

        self.reverse_iupac_mapping: Dict[str, Tuple[str, str]] = {
            v: k for k, v in self.iupac_mapping.items()
        }

        self.iupac = IUPAC(logger=self.logger)

        # Ensure the load_aln method is called.
        _ = self.snp_data

    def _iupac_from_gt_tuples(self) -> Dict[Tuple[str, str], str]:
        """Returns the IUPAC code mapping.

        Returns:
            Dict[Tuple[str, str], str]: Mapping of allele tuples to IUPAC codes.
        """
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
        """Creates a reverse mapping from IUPAC codes to allele tuples.

        Returns:
            Dict[str, Tuple[str, str]]: Mapping of IUPAC codes to allele tuples
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

    def read_popmap(self) -> None:
        """Read population map from file to map samples to populations.

        Makes use of the ReadPopmap class to read in the popmap file and validate the samples against the alignment.

        Sets the following attributes:
            - samples
            - populations
            - popmap
            - popmap_inverse
            - sample_indices

        Raises:
            TypeError: If the popmapfile is not a string.
            ValueError: If no valid samples are found after subsetting.
        """
        if self.popmapfile is not None:
            if not isinstance(self.popmapfile, str):
                msg = f"Invalid popmapfile type provided. Expected str, but got: {type(self.popmapfile)}"
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
            ValueError: If no valid samples are found after subsetting.
            TypeError: If the popmap file is not a string.
        """
        # Validate popmap with current samples
        popmap_ok = my_popmap.validate_popmap(samples, force=force)

        if not popmap_ok:
            msg = "Popmap validation failed. Check the popmap file and try again."
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

        # Subset the popmap based on inclusion/exclusion criteria
        my_popmap.subset_popmap(samples, include_pops, exclude_pops)

        # Update the sample list and the populations
        new_samples = [s for s in samples if s in my_popmap.popmap]
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
            AttributeError: If the samples attribute is not defined.
            AttributeError: If the popmap attribute is not defined.
        """
        if not self.samples or self.samples is None:
            msg = "'samples attribute is undefined."
            self.logger.error(msg)
            raise exceptions.EmptyIterableError(msg)

        if not self.populations or self.populations is None:
            msg = "'populations' attribute is undefined."
            self.logger.error(msg)
            raise exceptions.EmptyIterableError(msg)

        with open(filename, "w") as fout:
            for s, p in zip(self.samples, self.populations):
                fout.write(f"{s},{p}\n")

    def write_vcf(
        self,
        output_filename: str | Path,
        hdf5_file_path: str | Path | None = None,
        chunk_size: int = 1000,
    ) -> None:
        """Writes the GenotypeData object data to a VCF file in chunks.

        This method writes the VCF data, bgzips the output file, indexes it with Tabix, and validates the output.
        If the original input was VCF (`self.from_vcf`), it reads from the HDF5 produced by `get_vcf_attributes_fast`.
        Otherwise it falls back to writing from `self.snp_data` directly.

        Args:
            output_filename (str | Path): The name of the output VCF file.
            hdf5_file_path (str | Path, optional): Path to the HDF5 file; if None uses self.vcf_attributes_fn.
            chunk_size (int, optional): Number of records per write chunk.

        Raises:
            FileNotFoundError: If HDF5 is expected but missing.
        """
        self.logger.info(f"Writing VCF file to: {output_filename}")

        from_vcf = getattr(self, "from_vcf", False)
        of = Path(output_filename)

        try:
            if from_vcf:
                # HDF5-backed branch
                if hdf5_file_path is None:
                    hdf5_file_path = self.vcf_attributes_fn
                h5p = Path(hdf5_file_path)
                if not h5p.is_file():
                    raise FileNotFoundError(f"HDF5 file not found: {h5p}")

                with h5py.File(h5p, "r") as h5, open(of, "w") as f:
                    f.write(self.build_vcf_header())
                    total = h5["chrom"].shape[0]
                    info_keys = list(h5["info"].keys())
                    write_format_metadata = (
                        self.store_format_fields and "fmt_metadata" in h5
                    )

                    if write_format_metadata:
                        fmt_keys = list(h5["fmt_metadata"].keys())

                    with tqdm(
                        total=total, desc="Writing VCF Records: ", unit=" rec"
                    ) as pbar:
                        for start in range(0, total, chunk_size):
                            end = min(start + chunk_size, total)
                            n = end - start

                            chrom = h5["chrom"][start:end].astype(str)
                            pos = h5["pos"][start:end]
                            vid = h5["id"][start:end].astype(str)
                            ref = h5["ref"][start:end].astype(str)
                            alt_raw = h5["alt"][start:end].astype(str)
                            filt = h5["filt"][start:end].astype(str)
                            qual = (
                                h5["qual"][start:end].astype(str)
                                if "qual" in h5
                                else np.array(["."] * n)
                            )

                            info_chunk = {
                                key: h5["info"][key][start:end].astype(str)
                                for key in info_keys
                            }

                            refs = ref.tolist()
                            alts = [
                                s.split(",") if s not in {".", ""} else []
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

                            if write_format_metadata:
                                fmt_chunks = {
                                    key: h5["fmt_metadata"][key][start:end, :].astype(
                                        str
                                    )
                                    for key in fmt_keys
                                }

                            for i in range(n):
                                parts = [
                                    f"{key}={info_chunk[key][i]}"
                                    for key in info_keys
                                    if info_chunk[key][i] not in {".", ""}
                                ]
                                info_str = ";".join(parts) if parts else "."
                                alt_str = (
                                    alt_raw[i] if alt_raw[i] not in {".", ""} else "."
                                )

                                # Assemble FORMAT and sample fields
                                if write_format_metadata:
                                    format_col = "GT:" + ":".join(fmt_keys)
                                    sample_fields = []
                                    for j in range(fmt_matrix.shape[1]):
                                        extra_fields = [
                                            fmt_chunks[k][i][j] for k in fmt_keys
                                        ]
                                        sample_field = ":".join(
                                            [fmt_matrix[i][j]] + extra_fields
                                        )
                                        sample_fields.append(sample_field)
                                else:
                                    format_col = "GT"
                                    sample_fields = fmt_matrix[i].tolist()

                                row = [
                                    chrom[i],
                                    str(pos[i]),
                                    vid[i],
                                    ref[i],
                                    alt_str,
                                    qual[i],
                                    filt[i],
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
                vid = [f"locus{idx}" for idx in chrom_idx]
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

        # bgzip & tabix
        bgz = of if self._is_bgzipped(of) else self.bgzip_file(of)
        self.tabix_index(bgz)
        self.logger.info(f"Indexed VCF file: {bgz}.tbi")
        self.logger.info("Successfully wrote VCF file!")

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
            ValueError: If samples and snp_data are not the same length.

        Note:
            If genotype_data is provided, the snp_data and samples are loaded from the GenotypeData instance.

            If snp_data is provided, the samples must also be provided.

            If genotype_data is not provided, the snp_data and samples must be provided.

            The sequence data must have the same length for each sample.

            The PHYLIP file must have the correct number of samples and loci.
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
                n_samples, n_loci = len(samples), len(snp_data[0])
                f.write(f"{n_samples} {n_loci}\n")
                for sample, sample_data in zip(samples, snp_data):
                    genotype_str = "".join(str(x) for x in sample_data)
                    f.write(f"{sample}\t{genotype_str}\n")

            self.logger.info(f"Successfully wrote PHYLIP file {output_file}!")
        except Exception as e:
            msg = f"An error occurred while writing the PHYLIP file: {e}"
            self.logger.error(msg)
            raise e

    def write_structure(
        self,
        output_file: str | Path,
        onerow: bool = False,
        genotype_data: Any = None,
        snp_data: np.ndarray | None = None,
        samples: List[str] | None = None,
    ) -> None:
        """Write the stored alignment as a STRUCTURE file.

        This method writes the stored alignment as a STRUCTURE file. If genotype_data is provided, the SNP data and sample IDs are extracted from it. Otherwise, the SNP data and sample IDs must be provided.

        Args:
            output_file (str): Name of the output STRUCTURE file.
            onerow (bool): If True, write the STRUCTURE file in one-row format. If False, write in two-row format.
            genotype_data (GenotypeData, optional): GenotypeData instance.
            snp_data (List[List[str]], optional): SNP data in IUPAC format. Must be provided if genotype_data is None.
            samples (List[str]], optional): List of sample IDs. Must be provided if snp_data is not provided.
            verbose (bool, optional): If True, status updates are printed.

        Raises:
            TypeError: If genotype_data and snp_data are both provided.
            TypeError: If samples are not provided when snp_data is provided.
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

        self.logger.info(f"Writing STRUCTURE file as: {output_file}...")

        try:
            with open(output_file, "w") as fout:
                for sample, sample_data in zip(samples, snp_data):
                    # Convert IUPAC codes back to genotype format
                    # (e.g., 0/0, 1/1, 2/2, 0/1, etc.)
                    genotypes = [
                        self._iupac_to_genotype(iupac) for iupac in sample_data
                    ]

                    if onerow:
                        # Flatten the genotype pairs and write to file in
                        # one-row format
                        genotype_pairs = [
                            allele
                            for genotype in genotypes
                            for allele in genotype.split("/")
                        ]
                        fout.write(f"{sample}\t" + "\t".join(genotype_pairs) + "\n")
                    else:
                        # Write the two alleles in two separate rows for two-row format
                        first_row = [genotype.split("/")[0] for genotype in genotypes]
                        second_row = [genotype.split("/")[1] for genotype in genotypes]
                        fout.write(f"{sample}\t" + "\t".join(first_row) + "\n")
                        fout.write(f"{sample}\t" + "\t".join(second_row) + "\n")

            self.logger.info("Successfully wrote STRUCTURE file.")
        except Exception as e:
            msg = f"An error occurred while writing the STRUCTURE file: {e}"
            self.logger.error(msg)
            raise e

    def missingness_reports(
        self,
        prefix: str | None = None,
        zoom: bool = True,
        bar_color: str = "gray",
        heatmap_palette: str = "magma",
    ) -> None:
        """
        Generate missingness reports and plots.

        The function will write several comma-delimited report files:

            1. individual_missingness.csv: Missing proportions per individual.

            2. locus_missingness.csv: Missing proportions per locus.

            3. population_missingness.csv: Missing proportions per population (only generated if popmapfile was passed to GenotypeData).

            4. population_locus_missingness.csv: Table of per-population and per-locus missing data proportions.

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

            zoom (bool, optional): If True, zoom in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.

            bar_color (str, optional): Color of the bars on the non-stacked bar plots. Can be any color supported by matplotlib. See the matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, otpional): Color palette for the heatmap plot. Defaults to 'magma'.
        """
        # Set the prefix for the missingness report files,
        # If not provided.
        prefix = self.prefix if prefix is None else prefix

        # Set the parameters for the missingness report plots.
        keys = ["prefix", "zoom", "horizontal_space", "vertical_space"]
        keys.extend(["bar_color", "heatmap_palette"])
        values = [prefix, zoom, 0.8, 0.6, bar_color, heatmap_palette]

        # Create a dictionary of the parameters.
        params = dict(zip(keys, values))

        # Create a DataFrame from snp_data and replace missing values
        # with NA.
        df = pd.DataFrame(self.snp_data)
        df = df.replace(self.missing_vals, self.replace_vals)

        # Update plot_kwargs and params with the appropriate values.
        kwargs = self.plot_kwargs
        kwargs.update({"plot_fontsize": self.plot_fontsize})
        kwargs["plot_title_fontsize"] = self.plot_fontsize

        # Plot the missingness reports.
        plotting = Plotting(self, **kwargs)
        dfs = plotting.visualize_missingness(df, **params)
        loc, ind, poploc, poptotal, indpop = dfs

        # Write the missingness reports to file.
        report_path = Path(f"{self.prefix}_output", "gtdata", "reports")
        report_path.mkdir(exist_ok=True, parents=True)
        report_path = report_path / "individual_missingness.csv"

        # Write the individual missingness report to file.
        self._report2file(ind, report_path)

        # Write the locus missingness report to file.
        outfn = report_path.with_name("locus_missingness.csv")
        self._report2file(loc, outfn)

        if self.populations is not None:
            outfn = report_path.with_name("population_locus_missingness.csv")
            self._report2file(poploc, outfn)

            outfn = report_path.with_name("population_missingness.csv")
            self._report2file(poptotal, outfn)

            outfn = report_path.with_name("pop_individ_locus_missingness.csv")
            self._report2file(indpop, outfn, header=True)

    def _report2file(
        self, df: pd.DataFrame, report_path: str, header: bool = False
    ) -> None:
        """Write a DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): DataFrame to be written to the file.

            report_path (str): Path to the report directory.

            header (bool, optional): Whether to include the header row in the file. Defaults to False.
        """
        df.to_csv(report_path, header=header, index=False)

    def _genotype_to_iupac(self, genotype: str) -> str:
        """Convert a genotype string to its corresponding IUPAC code.

        Args:
            genotype (str): Genotype string in the format "x/y".

        Returns:
            str: Corresponding IUPAC code for the input genotype. Returns 'N' if the genotype is not in the lookup dictionary.

        Raises:
            ValueError: If the genotype is not valid.
        """
        try:
            iupac_dict = self.iupac.get_gt2iupac()

            # Validate genotype format
            if not isinstance(genotype, str) or "/" not in genotype:
                self.logger.error(f"Invalid genotype format: {genotype}")
                raise exceptions.InvalidGenotypeError(f"Invalid format: {genotype}")

            gt = iupac_dict.get(genotype, "N")  # Default to 'N' for undefined genotypes

            if gt == "N":
                self.logger.warning(
                    f"Undefined genotype: {genotype}, defaulting to 'N'"
                )

            return gt

        except Exception as e:
            self.logger.error(f"Error processing genotype {genotype}: {e}")
            raise

    def _iupac_to_genotype(self, iupac_code: str) -> str:
        """Convert an IUPAC code to its corresponding genotype string.

        Args:
            iupac_code (str): IUPAC encoded nucleotide character.

        Returns:
            str: Corresponding genotype string for the input IUPAC code. Returns '-9/-9' if the IUPAC code is not in the lookup dictionary.

        Raises:
            ValueError: If the IUPAC code is not valid.
        """
        genotype_dict = self.iupac.get_iupac2gt()

        gt = genotype_dict.get(iupac_code)
        if gt is None:
            msg = f"Invalid IUPAC Code: {iupac_code}"
            self.logger.error(msg)
            raise exceptions.InvalidGenotypeError(msg)
        return gt

    def calc_missing(self, df: pd.DataFrame, use_pops: bool = True) -> Tuple[
        pd.Series,
        pd.Series,
        pd.DataFrame | None,
        pd.Series | None,
        pd.DataFrame | None,
    ]:
        """Calculate missing value statistics based on a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing genotype data.

            use_pops (bool, optional): If True, calculate statistics per population. Defaults to True.

        Returns:
            Tuple[pd.Series, pd.Series, pd.DataFrame | None, pd.Series | None, pd.DataFrame | None]: A tuple of missing value statistics:

            - loc (pd.Series): Missing value proportions per locus.
            - ind (pd.Series): Missing value proportions per individual.
            - poploc (pd.DataFrame | None): Missing value proportions per population and locus. Only returned if use_pops=True.
            - poptot (pd.Series | None): Missing value proportions per population. Only returned if use_pops=True.
            - indpop (pd.DataFrame | None): Missing value proportions per individual and population. Only returned if use_pops=True.
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

    def copy(self) -> Any:
        """Create a deep copy of the GenotypeData or subclass object.

        Returns:
            GenotypeData or subclass: A new object with identical data and structure.
        """
        new_obj = self.__class__.__new__(self.__class__)

        # Attributes to exclude from deepcopy due to special handling
        exclude_attrs = {"vcf_header", "_vcf_attributes_fn", "logger"}

        for name, attr in vars(self).items():  # safer than __dict__
            if name in exclude_attrs:
                setattr(new_obj, name, attr)
            else:
                setattr(new_obj, name, copy.deepcopy(attr))

        # Copy header safely
        if hasattr(self, "vcf_header") and self.vcf_header:
            try:
                new_obj.vcf_header = self.vcf_header.copy()
            except Exception:
                new_obj.vcf_header = self.vcf_header

        # Reinitialize logger
        logman = LoggerManager(
            __name__,
            prefix=getattr(self, "prefix", "default"),
            verbose=getattr(self, "verbose", False),
            debug=getattr(self, "debug", False),
        )
        new_obj.logger = logman.get_logger()

        # Allow subclasses to clean up any problematic state
        if hasattr(new_obj, "_post_copy_hook"):
            new_obj._post_copy_hook()

        return new_obj

    def load_aln(self):
        msg = "load_aln method must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def get_population_indices(self) -> Dict[str, int]:
        """Create a mapping from population IDs to sample indices.

        This method creates a dictionary with population IDs as keys and lists of sample indices as values. The sample indices are used to subset the genotype data by population.

        Returns:
            Dict[str, int]: Dictionary with population IDs as keys and lists of sample indices as values.
        """
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
    def inputs(self) -> Dict[str, Any]:
        """Get GenotypeData keyword arguments as a dictionary.

        Returns:
            Dict[str, Any]: GenotypeData keyword arguments as a dictionary.
        """
        return self.kwargs

    @inputs.setter
    def inputs(self, value: Dict[str, Any]) -> None:
        """Setter method for class keyword arguments.

        Args:
            value (Dict[str, Any]): Dictionary of keyword arguments.
        """
        self.kwargs = value

    @property
    def num_snps(self) -> int:
        """Number of snps (loci) in the dataset.

        Returns:
            int: Number of SNPs (loci) per individual.
        """
        if self.snp_data.size > 0:
            return len(self.snp_data[0])
        return 0

    @num_snps.setter
    def num_snps(self, value: int) -> None:
        """Set the number of SNPs in the dataset.

        Args:
            value (int): Number of SNPs (loci) in the dataset.
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
            int: Number of individuals (samples) in input data.
        """
        if self.snp_data.size > 0:
            return len(self.snp_data)
        return 0

    @num_inds.setter
    def num_inds(self, value: int) -> None:
        """Set the number of individuals (samples) in the dataset.

        Args:
            value (int): Number of individuals (samples) in the dataset.
        """
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
    def populations(self) -> List[str | int]:
        """Population IDs as a list of strings or integers.

        Returns:
            List[str | int]: Population IDs.
        """
        return self._populations

    @populations.setter
    def populations(self, value: List[str | int]) -> None:
        """Set the population IDs.

        Args:
            value (List[str | int]): List of population IDs.
        """
        self._populations = value

    @property
    def popmap(self) -> Dict[str, str | int]:
        """Dictionary object with SampleIDs as keys and popIDs as values.

        Returns:
            Dict[str, str | int]: Dictionary with SampleIDs as keys and popIDs as values
        """
        return self._popmap

    @popmap.setter
    def popmap(self, value: Dict[str, str | int]) -> None:
        """Dictionary with SampleIDs as keys and popIDs as values.

        Args:
            value (Dict[str, str | int]): Dictionary object with SampleIDs as keys and popIDs as values.

        Raises:
            TypeError: If the value is not a dictionary object.
            TypeError: If the values are not strings or integers.
        """
        if not isinstance(value, dict):
            msg = f"'popmap' must be a dict object: {type(value)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if not all(isinstance(v, (str, int)) for v in value.values()):
            msg = f"popmap values must be strings or integers"
            self.logger.error(msg)
            raise TypeError(msg)

        self._popmap = value

    @property
    def popmap_inverse(self) -> Dict[str, List[str]]:
        """Inverse popmap dictionary with populationIDs as keys and lists of sampleIDs as values.

        Returns:
            Dict[str, List[str]]: Inverse dictionary of popmap, where popIDs are keys and lists of sampleIDs are values.
        """
        return self._popmap_inverse

    @popmap_inverse.setter
    def popmap_inverse(self, value: Dict[str, List[str]]) -> None:
        """Setter for popmap_inverse. Should have populationIDs as keys and lists of corresponding sampleIDs as values.

        Args:
            value (Dict[str, List[str]]): Inverse dictionary of popmap, where popIDs are keys and lists of sampleIDs are values.

        Raises:
            TypeError: If the value is not a dictionary object.
            TypeError: If the values are not lists of sampleIDs.
        """
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
        """List of sample IDs in input order.

        Returns:
            List[str]: Sample IDs in input order.
        """
        return self._samples

    @samples.setter
    def samples(self, value: List[str]) -> None:
        """Get the sampleIDs as a list of strings.

        Args:
            value (List[str]): List of sample IDs.
        """
        self._samples = value

    @property
    def snpsdict(self) -> Dict[str, List[str]]:
        """Dictionary with Sample IDs as keys and lists of genotypes as values.

        Returns:
            Dict[str, List[str]]: Dictionary with sample IDs as keys and lists of genotypes as values.
        """
        self._snpsdict = self._make_snpsdict()
        return self._snpsdict

    @snpsdict.setter
    def snpsdict(self, value: Dict[str, List[str]]) -> None:
        """Set snpsdict object, which is a dictionary with sample IDs as keys and lists of genotypes as values.

        Args:
            value (Dict[str, List[str]]): Dictionary with sample IDs as keys and lists of genotypes as values.
        """
        self._snpsdict = value

    @property
    def loci_indices(self) -> np.ndarray:
        """Boolean array for retained loci in filtered alignment.

        Returns:
            np.ndarray: Boolean array of loci indices, with True for retained loci and False for excluded loci.

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
            if not self._loci_indices.dtype is np.dtype(bool):
                msg = f"'loci_indices' must be numpy.dtype 'bool', but got: {self._loci_indices.dtype}"
                self.logger.error(msg)
                raise TypeError(msg)

        return self._loci_indices

    @loci_indices.setter
    def loci_indices(self, value: np.ndarray | list) -> None:
        """Column indices for retained loci in filtered alignment.

        Args:
            value (np.ndarray): Boolean array of loci indices, with True for retained loci and False for excluded loci.

        Raises:
            TypeError: If the loci_indices attribute is not a numpy.dtype 'bool'.
        """
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
        """Row indices for retained samples in alignemnt.

        Returns:
            np.ndarray: Boolean array of sample indices, with True for retained samples and False for excluded samples.

        Raises:
            TypeError: If the sample_indices attribute is not a numpy.ndarray or list.
            TypeError: If the sample_indices attribute is not a numpy.dtype 'bool'.
        """
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
    def sample_indices(self, value: np.ndarray | list) -> None:
        """Set the sample indices as a boolean array.

        Args:
            value (np.ndarray | list): Boolean array of sample indices, with True for retained samples and False for excluded samples.
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
            List[str]: List of reference alleles of length num_snps.
        """
        return self._ref

    @ref.setter
    def ref(self, value: List[str]) -> None:
        """Setter for list of reference alleles of length num_snps.

        Args:
            value (List[str]): List of reference alleles of length num_snps.
        """
        self._ref = value

    @property
    def alt(self) -> List[str]:
        """Get list of alternate alleles of length num_snps.

        Returns:
            List[str]: List of alternate alleles of length num_snps.
        """
        return self._alt

    @alt.setter
    def alt(self, value: List[str]) -> None:
        """Setter for list of alternate alleles of length num_snps.

        Args:
            value (List[str]): List of alternate alleles of length num_snps.
        """
        self._alt = value

    @property
    def snp_data(self) -> np.ndarray:
        """Get the genotypes as a 2D list of shape (n_samples, n_loci).

        Returns:
            np.ndarray: 2D array of IUPAC encoded genotype data.

        Raises:
            TypeError: If the snp_data attribute is not a numpy.ndarray, pandas.DataFrame, or list.
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
                return np.array(self._snp_data)
            elif isinstance(self._snp_data, pd.DataFrame):
                return self._snp_data.to_numpy()
            return self._snp_data  # already numpy.ndarray
        else:
            msg = f"Invalid 'snp_data' type. Expected numpy.ndarray, pandas.DataFrame, or list, but got: {type(self._snp_data)}"
            self.logger.error(msg)
            raise TypeError(msg)

    @snp_data.setter
    def snp_data(self, value: np.ndarray | List[List[str]] | pd.DataFrame) -> None:
        """Set snp_data attribute as a 2D list of IUPAC encoded genotype data.

        Input can be a 2D list, numpy array, or pandas DataFrame object.

        Args:
            value (np.ndarray | List[List[str]] | pd.DataFrame): 2D array of IUPAC encoded genotype data.

        Raises:
            TypeError: If the snp_data attribute is not a numpy.ndarray, pandas.DataFrame, or list.
        """
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

    def _validate_seq_lengths(self) -> None:
        """Ensure that all SNP data rows have the same length.

        Raises:
            SequenceLengthError: If the sequence lengths are not all the same.
        """
        lengths = set([len(row) for row in self.snp_data])
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
        reset_attributes: bool = True,
    ) -> None:
        """Set the alignment data and sample IDs after filtering.

        Args:
            snp_data (np.ndarray): 2D array of genotype data.
            samples (List[str]): List of sample IDs.
            sample_indices (np.ndarray): Boolean array of sample indices.
            loci_indices (np.ndarray): Boolean array of locus indices.
            reset_attributes (bool): If True, update the VCF attributes metadata.
        """
        self.snp_data = snp_data
        self.samples = samples
        self.sample_indices = sample_indices
        self.loci_indices = loci_indices
        self.num_inds = np.count_nonzero(self.sample_indices)
        self.num_snps = np.count_nonzero(self.loci_indices)
        self.prefix = f"{self.prefix}_filtered"

        if self.popmap is not None:
            self.popmap = {s: self.popmap[s] for s in samples}
            self.populations = [self.popmap[s] for s in samples]
            self.popmap_inverse = {}
            for s, p in self.popmap.items():
                self.popmap_inverse.setdefault(p, []).append(s)

        ref, alt, alt2 = self.get_ref_alt_alleles(self.snp_data)
        self.ref = ref
        self.alt = alt
        self.alt = list(alt)
        if alt2:
            self.alt = [a if isinstance(a, list) else [a] for a in self.alt]
            for i, extra in enumerate(alt2):
                self.alt[i].extend(
                    extra.tolist() if isinstance(extra, np.ndarray) else extra
                )

        if reset_attributes and self.filetype.startswith("vcf"):
            self.vcf_attributes_fn = self.update_vcf_attributes(
                snp_data=self.snp_data,
                sample_indices=self.sample_indices,
                loci_indices=self.loci_indices,
                samples=self.samples,
            )

    def __len__(self) -> int:
        """Return the number of individuals in the dataset.

        Returns:
            int: The number of individuals in the dataset.
        """
        return len(self.snp_data)

    def __getitem__(self, index: int | str) -> List[str]:
        """Return the genotypes for a specific sample index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            List[str]: The genotypes for the specified sample index.
        """
        if isinstance(index, str):
            index = self.samples.index(index)
        return self.snp_data[index]

    def __iter__(self) -> Generator[List[str], None, None]:
        """Iterate over each sample in the dataset."""
        for i in range(self.num_inds):
            yield self.snp_data[i]

    def __contains__(self, individual: str) -> bool:
        """Check if an individual is present in the dataset."""
        return individual in self.samples

    def __str__(self) -> str:
        """Return a string representation of the GenotypeData object."""
        return f"GenotypeData: {self.num_snps} SNPs, {self.num_inds} individuals"

    def __repr__(self) -> str:
        """Return a detailed string representation of the GenotypeData object."""
        return f"GenotypeData(filename={self.filename}, filetype={self.filetype}, popmapfile={self.popmapfile}, force_popmap={self.force_popmap}, exclude_pops={self.exclude_pops}, include_pops={self.include_pops}, plot_format={self.plot_format}, prefix={self.prefix})"
