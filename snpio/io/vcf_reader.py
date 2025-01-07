import re
import tempfile
import textwrap
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pysam
from tqdm import tqdm

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.benchmarking import Benchmark
from snpio.utils.logging import LoggerManager

measure_execution_time = Benchmark.measure_execution_time


class VCFReader(GenotypeData):
    """A class to read VCF files into GenotypeData objects and write GenotypeData objects to VCF files.

    This class inherits from GenotypeData and provides methods to read VCF files and extract the necessary attributes.

    Example:
        >>> from snpio import VCFReader
        >>>
        >>> genotype_data = VCFReader(filename="example.vcf", popmapfile="popmap.txt", verbose=True)
        >>> genotype_data.snp_data
        array([["A", "T", "T", "A"], ["A", "T", "T", "A"], ["A", "T", "T", "A"]], dtype="<U1")
        >>>
        >>> genotype_data.samples
        ["sample1", "sample2", "sample3", "sample4"]
        >>>
        >>> genotype_data.num_inds
        4
        >>>
        >>> genotype_data.num_snps
        3
        >>>
        >>> genotype_data.populations
        ["pop1", "pop1", "pop2", "pop2"]
        >>>
        >>> genotype_data.popmap
        {"sample1": "pop1", "sample2": "pop1", "sample3": "pop2", "sample4":
        "pop2"}
        >>>
        >>> genotype_data.popmap_inverse
        {"pop1": ["sample1", "sample2"], "pop2": ["sample3", "sample4"]}
        >>>
        >>> genotype_data.loci_indices
        array([True, True, True], dtype=bool)
        >>>
        >>> genotype_data.sample_indices
        array([True, True, True, True], dtype=bool)
        >>>
        >>> genotype_data.ref
        ["A", "A", "A"]
        >>>
        >>> genotype_data.alt
        ["T", "T", "T"]
        >>>
        >>> genotype_data.missingness_reports()
        >>>
        >>> genotype_data.run_pca()
        >>>
        >>> genotype_data.write_vcf("output.vcf")

    Attributes:
        filename (Optional[str]): The name of the VCF file to read.
        popmapfile (Optional[str]): The name of the population map file to read.
        chunk_size (int): The size of the chunks to read from the VCF file.
        force_popmap (bool): Whether to force the use of the population map file.
        exclude_pops (Optional[List[str]]): The populations to exclude.
        include_pops (Optional[List[str]]): The populations to include.
        plot_format (str): The format to save the plots in.
        plot_fontsize (int): The font size for the plots.
        plot_dpi (int): The DPI for the plots.
        plot_despine (bool): Whether to remove the spines from the plots.
        show_plots (bool): Whether to show the plots.
        prefix (str): The prefix to use for the output files.
        verbose (bool): Whether to print verbose output.
        sample_indices (np.ndarray): The indices of the samples to read.
        loci_indices (np.ndarray): The indices of the loci to read.
        debug (bool): Whether to enable debug mode.
        num_records (int): The number of records in the VCF file.
        filetype (str): The type of the file.
        vcf_header (Optional[pysam.libcbcf.VariantHeader]): The VCF header.
        info_fields (Optional[List[str]]): The VCF info fields.
        resource_data (dict): A dictionary to store resource data.
        logger (logging.Logger): The logger object.
        snp_data (np.ndarray): The SNP data.
        samples (np.ndarray): The sample names.

    Note:
        The VCF file is bgzipped, sorted, and indexed using Tabix to ensure efficient reading, if necessary.

        The VCF file is read in chunks to avoid memory issues.

        The VCF attributes are extracted and stored in an HDF5 file for efficient access.

        The genotype data is transformed into IUPAC codes for efficient storage and processing.

        The VCF attributes are stored in an HDF5 file for efficient access.

        The VCF attributes are extracted and stored in an HDF5 file for efficient access.

        The genotype data is transformed into IUPAC codes for efficient storage and processing.

        The VCF attributes are stored in an HDF5 file for efficient access.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        popmapfile: Optional[str] = None,
        chunk_size: int = 1000,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        plot_format: str = "png",
        plot_fontsize: int = 18,
        plot_dpi: int = 300,
        plot_despine: bool = True,
        show_plots: bool = False,
        prefix: str = "snpio",
        verbose: bool = False,
        sample_indices: Optional[np.ndarray] = None,
        loci_indices: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> None:
        """Initializes the VCFReader object.

        This class inherits from GenotypeData and provides methods to read VCF files and extract the necessary attributes.

        Args:
            filename (Optional[str], optional): The name of the VCF file to read. Defaults to None.
            popmapfile (Optional[str], optional): The name of the population map file to read. Defaults to None.
            chunk_size (int, optional): The size of the chunks to read from the VCF file. Defaults to 1000.
            force_popmap (bool, optional): Whether to force the use of the population map file. Defaults to False.
            exclude_pops (Optional[List[str]], optional): The populations to exclude. Defaults to None.
            include_pops (Optional[List[str]], optional): The populations to include. Defaults to None.
            plot_format (str, optional): The format to save the plots in. Defaults to "png".
            plot_fontsize (int, optional): The font size for the plots. Defaults to 18.
            plot_dpi (int, optional): The DPI for the plots. Defaults to 300.
            plot_despine (bool, optional): Whether to remove the spines from the plots. Defaults to True.
            show_plots (bool, optional): Whether to show the plots. Defaults to False.
            prefix (str, optional): The prefix to use for the output files. Defaults to "snpio".
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            sample_indices (Optional[np.ndarray], optional): The indices of the samples to read. Defaults to None.
            loci_indices (Optional[np.ndarray], optional): The indices of the loci to read. Defaults to None.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
        """
        outdir = Path(f"{prefix}_output") / "gtdata" / "alignments" / "vcf"
        self._vcf_attributes_fn = outdir / "vcf_attributes.h5"

        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(name=__name__, **kwargs)
        self.logger = logman.get_logger()

        self.resource_data = {}

        super().__init__(
            filename=filename,
            filetype="vcf",
            popmapfile=popmapfile,
            force_popmap=force_popmap,
            exclude_pops=exclude_pops,
            include_pops=include_pops,
            plot_format=plot_format,
            plot_fontsize=plot_fontsize,
            plot_dpi=plot_dpi,
            plot_despine=plot_despine,
            show_plots=show_plots,
            prefix=prefix,
            verbose=verbose,
            sample_indices=sample_indices,
            loci_indices=loci_indices,
            chunk_size=chunk_size,
            logger=self.logger,
            debug=debug,
        )

        self.num_records = 0
        self.filetype = "vcf"
        self.vcf_header = None
        self.info_fields = None

        outdir.mkdir(exist_ok=True, parents=True)

        # Define reverse IUPAC mapping (example, should be defined appropriately)
        self.reverse_iupac_mapping = {
            "A": ("A", "A"),
            "C": ("C", "C"),
            "G": ("G", "G"),
            "T": ("T", "T"),
            "M": ("A", "C"),
            "R": ("A", "G"),
            "W": ("A", "T"),
            "S": ("C", "G"),
            "Y": ("C", "T"),
            "K": ("G", "T"),
            "V": ("A", "C", "G"),
            "H": ("A", "C", "T"),
            "D": ("A", "G", "T"),
            "B": ("C", "G", "T"),
            "N": ("A", "C", "G", "T"),
        }

    @measure_execution_time
    def load_aln(self) -> None:
        """Loads the alignment from the VCF file into the VCFReader object.

        This method ensures that the input VCF file is bgzipped, sorted, and indexed using Tabix. It then reads the VCF file and extracts the necessary attributes. The VCF attributes are stored in an HDF5 file for efficient access. The genotype data is transformed into IUPAC codes for efficient storage and processing.
        """
        self.logger.info(f"Processing input VCF file {self.filename}...")

        vcf_path = Path(self.filename)
        bgzipped_vcf = vcf_path

        # Ensure the VCF file is bgzipped
        if not self._is_bgzipped(vcf_path):
            bgzipped_vcf = self._bgzip_file(vcf_path)
            self.logger.info(f"Bgzipped VCF file to {bgzipped_vcf}")

        # Ensure the VCF file is sorted
        if not self._is_sorted(bgzipped_vcf):
            bgzipped_vcf = self._sort_vcf_file(bgzipped_vcf)
            self.logger.info(f"Sorted VCF file to {bgzipped_vcf}")

        # Ensure the VCF file is indexed
        if not self._has_tabix_index(bgzipped_vcf):
            self._tabix_index(bgzipped_vcf)
            self.logger.info(f"Indexed VCF file {bgzipped_vcf}.tbi")

        self.filename = str(bgzipped_vcf)

        with pysam.VariantFile(self.filename, mode="r") as vcf:
            self.vcf_header = vcf.header
            self.samples = np.array(vcf.header.samples)
            self.num_records = sum(1 for _ in vcf)
            vcf.reset()

            self.vcf_attributes_fn, self.snp_data, self.samples = (
                self.get_vcf_attributes(vcf, chunk_size=self.chunk_size)
            )

            self.snp_data = self.snp_data.T

        self.logger.info(f"VCF file successfully loaded!")

        msg = (
            f"\nLoaded Alignment contains:\n"
            f"{self.snp_data.shape[0]} samples,\n"
            f"{self.snp_data.shape[1]} loci.\n"
        )
        self.logger.info(msg)

        self.logger.debug(f"snp_data: {self.snp_data}")
        self.logger.debug(f"snp_data shape: {self.snp_data.shape}")

        return self

    def get_vcf_attributes(
        self, vcf: pysam.VariantFile, chunk_size: int = 1000
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Extracts VCF attributes and returns them in an efficient manner with chunked processing.

        Args:
            vcf (pysam.VariantFile): The VCF file object to extract attributes from.
            chunk_size (int, optional): The size of the chunks to process. Defaults to 1000.

        Returns:
            Tuple[str, np.ndarray, np.ndarray]: The path to the HDF5 file containing the VCF attributes, the SNP data, and the sample names.
        """
        h5_outfile = self.vcf_attributes_fn
        h5_outfile.parent.mkdir(exist_ok=True, parents=True)

        snp_data = []
        sample_names = np.array(vcf.header.samples)

        with h5py.File(h5_outfile, "w") as f:
            chrom_dset = f.create_dataset(
                "chrom", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )
            pos_dset = f.create_dataset("pos", (0,), maxshape=(None,), dtype=np.int32)
            id_dset = f.create_dataset(
                "id", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )
            ref_dset = f.create_dataset("ref", (0,), maxshape=(None,), dtype="S1")
            alt_dset = f.create_dataset(
                "alt", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )

            qual_dset = f.create_dataset(
                "qual", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )

            filt_dset = f.create_dataset("filt", (0,), maxshape=(None,), dtype="S4")

            fmt_dset = f.create_dataset(
                "fmt", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )

            fmt_data_dset = f.create_dataset(
                "fmt_data", (0,), maxshape=(None,), dtype=h5py.string_dtype()
            )

            info_fields = list(vcf.header.info)
            self.info_fields = info_fields

            info_group = f.create_group("info")

            # Create datasets within info groups
            info_dsets = {
                k: info_group.create_dataset(
                    k, (0,), maxshape=(None,), dtype=h5py.string_dtype()
                )
                for k in info_fields
            }

            (
                chrom_chunk,
                pos_chunk,
                id_chunk,
                ref_chunk,
                alt_chunk,
                qual_chunk,
                filt_chunk,
                fmt_chunk,
                info_chunk,
                snp_chunk,
                fmt_data_chunk,
            ) = ([], [], [], [], [], [], [], [], defaultdict(list), [], [])

            for variant in tqdm(
                vcf.fetch(),
                desc="Processing VCF records: ",
                total=self.num_records,
                unit=" records",
            ):
                chrom_chunk.append(variant.chrom)
                pos_chunk.append(variant.pos)
                id_chunk.append("." if variant.id is None else variant.id)
                ref_chunk.append(variant.ref)

                if variant.alts is None:
                    alt_chunk.append(".")
                else:
                    alt_chunk.append(",".join([a for a in variant.alts]))

                qual_chunk.append("." if variant.qual is None else str(variant.qual))
                filt_chunk.append(
                    "." if not variant.filter.keys() else list(variant.filter.keys())[0]
                )
                fmt_chunk.append(":".join([k for k in variant.format.keys()]))

                for k in info_fields:
                    value = variant.info.get(k, ".")
                    processed_value = (
                        ",".join(list(value)) if isinstance(value, tuple) else value
                    )
                    info_chunk[k].append(processed_value)

                gt_data = [
                    variant.samples[sample].get("GT", (None, None))
                    for sample in sample_names
                ]

                fmt_data = []
                for sample in sample_names:
                    fmd_list = []
                    for k in variant.format.keys():
                        if k == "GT":
                            continue
                        fmd = variant.samples[sample].get(k, ".")
                        if isinstance(fmd, (tuple, list)):
                            if isinstance(fmd, tuple):
                                fmd = list(fmd)
                            fmd = ",".join([str(x) for x in fmd])
                        else:
                            fmd = str(fmd)
                        fmd_list.append(fmd)
                    fmt_data.append(":".join(fmd_list))
                fmt_data_chunk.append("\t".join(fmt_data))

                transformed_gt = self.transform_gt(
                    np.array(gt_data), variant.ref, variant.alts
                )
                snp_chunk.append(transformed_gt)

                # Process chunk if it reaches the specified chunk size
                if len(chrom_chunk) == chunk_size:
                    # Extend the datasets
                    current_size = chrom_dset.shape[0]
                    new_size = current_size + chunk_size
                    chrom_dset.resize((new_size,))
                    pos_dset.resize((new_size,))
                    id_dset.resize((new_size,))
                    ref_dset.resize((new_size,))
                    alt_dset.resize((new_size,))
                    qual_dset.resize((new_size,))
                    filt_dset.resize((new_size,))
                    fmt_dset.resize((new_size,))
                    fmt_data_dset.resize((new_size,))

                    for k in info_dsets.keys():
                        if info_chunk[k]:
                            info_dsets[k].resize((new_size,))
                            info_dsets[k][current_size:new_size] = np.array(
                                info_chunk[k], dtype=str
                            )
                            info_chunk[k] = []

                    # Write the chunk data
                    chrom_dset[current_size:new_size] = chrom_chunk
                    pos_dset[current_size:new_size] = pos_chunk
                    id_dset[current_size:new_size] = id_chunk
                    ref_dset[current_size:new_size] = ref_chunk
                    alt_dset[current_size:new_size] = alt_chunk

                    qual_dset[current_size:new_size] = qual_chunk
                    filt_dset[current_size:new_size] = filt_chunk
                    fmt_dset[current_size:new_size] = fmt_chunk
                    fmt_data_dset[current_size:new_size] = fmt_data_chunk

                    snp_data.extend(snp_chunk)

                    # Reset the chunk lists
                    (
                        chrom_chunk,
                        pos_chunk,
                        id_chunk,
                        ref_chunk,
                        alt_chunk,
                        qual_chunk,
                        filt_chunk,
                        fmt_chunk,
                        snp_chunk,
                        fmt_data_chunk,
                    ) = (
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                    )

            # Handle any remaining data in the chunks
            if chrom_chunk:
                current_size = chrom_dset.shape[0]
                new_size = current_size + len(chrom_chunk)
                chrom_dset.resize((new_size,))
                pos_dset.resize((new_size,))
                id_dset.resize((new_size,))
                ref_dset.resize((new_size,))
                alt_dset.resize((new_size,))
                qual_dset.resize((new_size,))
                filt_dset.resize((new_size,))
                fmt_dset.resize((new_size,))
                fmt_data_dset.resize((new_size,))

                for k in info_dsets.keys():
                    info_dsets[k].resize((new_size,))
                    info_dsets[k][current_size:new_size] = np.array(
                        info_chunk[k], dtype=str
                    )
                    info_chunk[k] = []

                chrom_dset[current_size:new_size] = chrom_chunk
                pos_dset[current_size:new_size] = pos_chunk
                id_dset[current_size:new_size] = id_chunk
                ref_dset[current_size:new_size] = ref_chunk
                alt_dset[current_size:new_size] = alt_chunk
                qual_dset[current_size:new_size] = qual_chunk
                filt_dset[current_size:new_size] = filt_chunk
                fmt_dset[current_size:new_size] = fmt_chunk
                fmt_data_dset[current_size:new_size] = fmt_data_chunk

                snp_data.extend(snp_chunk)

        return h5_outfile, np.array(snp_data), sample_names

    def transform_gt(self, gt: np.ndarray, ref: str, alts: List[str]) -> np.ndarray:
        """Transforms genotype tuples into their IUPAC codes or corresponding strings.

        Args:
            gt (np.ndarray): The genotype tuples to transform.
            ref (str): The reference allele.
            alts (List[str]): The alternate alleles.

        Returns:
            np.ndarray: The transformed genotype tuples.
        """
        iupac_mapping = self._iupac_code()

        # Convert alts to a list, ensuring it's mutable and indexable
        alts = list(alts) if alts else []

        # Prepare the array of possible alleles including reference
        alt_array = np.array([ref] + alts)

        transformed_gt = np.array(
            [
                (
                    iupac_mapping(
                        (
                            (
                                alt_array[allele1]
                                if 0 <= allele1 < len(alt_array)
                                else ref
                            ),
                            (
                                alt_array[allele2]
                                if 0 <= allele2 < len(alt_array)
                                else ref
                            ),
                        )
                    )
                    if allele1 is not None and allele2 is not None
                    else "N"
                )
                for allele1, allele2 in gt
            ]
        )
        return transformed_gt

    def _iupac_code(self) -> Callable:
        """Returns a callable function to get the IUPAC code for a given allele pair.

        Returns:
            Callable: A function to get the IUPAC code for a given allele pair.
        """
        iupac_mapping = self._iupac_from_gt_tuples()

        def get_code(alleles: Tuple[str]) -> str:
            return iupac_mapping.get(tuple(sorted(alleles)), "N")

        return get_code

    def _iupac_from_gt_tuples(self) -> Dict[Tuple[str], str]:
        """Creates a mapping from genotype tuples to IUPAC codes.

        Returns:
            Dict[Tuple[str], str]: A dictionary mapping genotype tuples to IUPAC codes.
        """
        # Define IUPAC codes for allele pairs
        mapping = {
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
        }
        return mapping

    def _is_bgzipped(self, filepath: Path) -> bool:
        """Checks if a file is bgzipped.

        Args:
            filepath (Path): The path to the file.

        Returns:
            bool: True if bgzipped, False otherwise.
        """
        try:
            with open(filepath, "rb") as f:
                magic = f.read(2)
            return magic == b"\x1f\x8b"
        except Exception as e:
            self.logger.error(f"Error checking bgzip status: {e}")
            return False

    def _bgzip_file(self, filepath: Path) -> Path:
        """BGZips a VCF file using pysam's BGZFile.

        Args:
            filepath (Path): The path to the VCF file.

        Returns:
            Path: The path to the bgzipped VCF file.
        """
        bgzipped_path = filepath.with_suffix(filepath.suffix + ".gz")
        try:
            with open(filepath, "rb") as f_in, pysam.BGZFile(
                str(bgzipped_path), "wb"
            ) as f_out:
                # Read and write in chunks to handle large files efficiently
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
            return bgzipped_path
        except Exception as e:
            self.logger.error(f"Error bgzipping file {filepath}: {e}")
            raise

    def _is_sorted(self, filepath: Path) -> bool:
        """Checks if the VCF file is sorted alphanumerically by chromosome and position.

        Args:
            filepath (Path): The path to the bgzipped VCF file.

        Returns:
            bool: True if sorted, False otherwise.
        """
        try:
            with pysam.VariantFile(filepath, mode="r") as vcf:
                previous_chrom = None
                previous_pos = -1
                for record in vcf:
                    if previous_chrom is not None:
                        if record.chrom < previous_chrom:
                            return False
                        elif (
                            record.chrom == previous_chrom and record.pos < previous_pos
                        ):
                            return False
                    previous_chrom = record.chrom
                    previous_pos = record.pos
            return True
        except Exception as e:
            self.logger.error(f"Error checking sort order: {e}")
            return False

    @staticmethod
    def _natural_sort_key(chrom_pos: Tuple[str, int]) -> Any:
        """Extracts a natural sort key for sorting."""
        chrom, pos = chrom_pos
        # Split chrom into alphanumeric segments to achieve natural order
        return [int(s) if s.isdigit() else s for s in re.split("([0-9]+)", chrom)], int(
            pos
        )

    def _sort_vcf_file(self, filepath: Path) -> Path:
        """Sorts a VCF file alphanumerically using custom logic, then indexes it.

        Args:
            filepath (Path): The path to the bgzipped VCF file.

        Returns:
            Path: The path to the sorted bgzipped VCF file.
        """
        sorted_path = filepath.with_name(filepath.stem + "_sorted.vcf.gz")

        try:
            # Read the VCF file and split into header and data lines
            header_lines = []
            data_lines = []
            with pysam.VariantFile(filepath, "r") as vcf_in:
                header_lines.append(str(vcf_in.header))
                for record in vcf_in:
                    data_lines.append(record)

            # Sort the data lines alphanumerically by CHROM and POS
            sorted_data_lines = sorted(
                data_lines, key=lambda r: VCFReader._natural_sort_key((r.contig, r.pos))
            )

            # Write sorted data to a temporary VCF file
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".vcf"
            ) as temp_vcf:
                # Write the header lines
                for line in header_lines:
                    temp_vcf.write(line)

                # Write sorted data lines
                for record in sorted_data_lines:
                    temp_vcf.write(str(record))

                temp_vcf_path = Path(temp_vcf.name)

            # Bgzip and tabix index the sorted VCF file
            sorted_bgzip_path = sorted_path
            pysam.tabix_compress(str(temp_vcf_path), str(sorted_bgzip_path), force=True)
            pysam.tabix_index(str(sorted_bgzip_path), preset="vcf", force=True)

            # Clean up the temporary uncompressed VCF
            temp_vcf_path.unlink()

            return sorted_bgzip_path

        except Exception as e:
            self.logger.error(f"Error sorting VCF file with custom sort: {e}")
            raise

    def _has_tabix_index(self, filepath: Path) -> bool:
        """Checks if a Tabix index exists for the given VCF file.

        Args:
            filepath (Path): The path to the bgzipped VCF file.

        Returns:
            bool: True if index exists, False otherwise.
        """
        index_path = filepath.with_suffix(filepath.suffix + ".tbi")
        return index_path.exists()

    def _tabix_index(self, filepath: Path) -> None:
        """Creates a Tabix index for a bgzipped VCF file.

        Args:
            filepath (Path): The path to the bgzipped VCF file.
        """
        try:
            pysam.tabix_index(str(filepath), preset="vcf", force=True)
        except Exception as e:
            self.logger.error(f"Error indexing VCF file {filepath}: {e}")
            raise

    def validate_input_vcf(self, filepath: Path) -> None:
        """Validates the input VCF file to ensure it meets required criteria.

        Args:
            filepath (Path): The path to the bgzipped and sorted VCF file.

        Raises:
            ValueError: If the VCF file does not meet validation criteria.
        """
        try:
            with pysam.VariantFile(filepath, mode="r") as vcf:
                required_fields = {
                    "CHROM",
                    "POS",
                    "ID",
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                }
                header_fields = set(vcf.header.records[0].keys())
                if not required_fields.issubset(header_fields):
                    missing = required_fields - header_fields
                    msg = f"VCF file is missing required fields: {missing}"
                    self.logger.error(msg)
                    raise ValueError(msg)

                if "GT" not in vcf.header.formats:
                    msg = "VCF file is missing the GT format field."
                    self.logger.error(msg)
                    raise ValueError(msg)

                if len(vcf.header.samples) == 0:
                    msg = "VCF file contains no samples."
                    self.logger.error(msg)
                    raise ValueError("VCF file contains no samples.")
        except Exception as e:
            self.logger.error(f"Input VCF validation failed: {e}")
            raise

    def validate_output_vcf(self, filepath: Path) -> None:
        """Validates the output VCF file to ensure it was written correctly.

        Args:
            filepath (Path): The path to the bgzipped and indexed output VCF file.

        Raises:
            ValueError: If the VCF file does not meet validation criteria.
        """
        try:
            with pysam.VariantFile(filepath, mode="r") as vcf:
                # Basic checks: header presence, samples match, etc.
                if self.vcf_header is None:
                    msg = "Output VCF file lacks a header."
                    self.logger.error(msg)
                    raise ValueError(msg)

                output_samples = set(vcf.header.samples)
                expected_samples = set(self.samples)
                if not expected_samples.issubset(output_samples):
                    missing = expected_samples - output_samples
                    msg = f"Output VCF is missing samples: {missing}"
                    self.logger.error(msg)
                    raise ValueError(msg)

                # Additional checks can be added as needed
        except Exception as e:
            self.logger.error(f"Output VCF validation failed: {e}")
            raise

    @measure_execution_time
    def write_vcf(
        self,
        output_filename: str,
        hdf5_file_path: Optional[str] = None,
        chunk_size: int = 1000,
    ) -> Any:
        """Writes the GenotypeData object data to a VCF file in chunks.

        This method writes the VCF data, bgzips the output file, indexes it with Tabix,
        and validates the output.

        Args:
            output_filename (str): The name of the output VCF file to write.
            hdf5_file_path (str, optional): The path to the HDF5 file containing the VCF attributes. Defaults to None.
            chunk_size (int, optional): The size of the chunks to read from the HDF5 file. Defaults to 1000.

        Returns:
            VCFReader: The current instance of VCFReader.
        """
        self.logger.info(f"Writing VCF file to: {output_filename}")

        if hdf5_file_path is None:
            hdf5_file_path = self.vcf_attributes_fn

        of = Path(output_filename)

        # Open the HDF5 file for reading and the output VCF file for writing
        with h5py.File(hdf5_file_path, "r") as hdf5_file, open(of, "w") as f:
            # Write the VCF header
            f.write(self._build_vcf_header())

            # Get the total number of loci
            total_loci = len(hdf5_file["chrom"])

            chunk_size = self.chunk_size

            with tqdm(
                total=total_loci, desc="Writing VCF Records: ", unit=" records"
            ) as pbar:
                # Iterate over data in chunks
                for start in range(0, total_loci, chunk_size):
                    end = min(start + chunk_size, total_loci)

                    # Read data in chunks
                    chrom = hdf5_file["chrom"][start:end].astype(str)
                    pos = hdf5_file["pos"][start:end]
                    vid = hdf5_file["id"][start:end].astype(str)
                    ref = hdf5_file["ref"][start:end].astype(str)
                    alt = hdf5_file["alt"][start:end].astype(str)
                    qual = hdf5_file["qual"][start:end].astype(str)
                    filt = hdf5_file["filt"][start:end].astype(str)
                    fmt = hdf5_file["fmt"][start:end].astype(str)
                    fmt_data = hdf5_file["fmt_data"][start:end].astype(str)

                    # Read info fields in chunks
                    info_keys = list(hdf5_file["info"].keys())

                    # Create a dictionary to hold the info fields data
                    info = {
                        k: hdf5_file[f"info/{k}"][start:end].astype(str)
                        for k in info_keys
                    }

                    # Format the info fields into strings
                    info_result = [
                        ";".join([f"{key}={value[i]}" for key, value in info.items()])
                        for i in range(end - start)
                    ]

                    # Write each row to the VCF file
                    for i in range(len(chrom)):
                        # Construct the ALT field by combining alternate alleles
                        alt_alleles_str = alt[i] if alt[i] != "." else "."
                        ref_str = ref[i]
                        alts = alt[i].split(",") if alt[i] != "." else []

                        # Split fmt_data[i] by tab to get individual sample data
                        fmt_data_samples = fmt_data[i].split("\t")

                        # Join fmt_data with snp_data by ":"
                        combined_data = [
                            f"{self.snp_data[j, start + i]}:{fmt_data_samples[j]}"
                            for j in range(len(self.samples))
                        ]

                        row = [
                            chrom[i],
                            str(pos[i]),
                            vid[i],
                            ref_str,
                            alt_alleles_str,
                            qual[i],
                            filt[i],
                            info_result[i],
                            fmt[i],
                        ] + combined_data

                        # Replace alleles if necessary
                        row = self._replace_alleles(row, ref_str, alts)

                        # Write the row to the file
                        f.write("\t".join(row) + "\n")

                    pbar.update(end - start)

        # Bgzip and index the output VCF file
        bgzipped_output = self._bgzip_file(of)
        self.logger.info(f"Bgzipped output VCF file to {bgzipped_output}")

        self._tabix_index(bgzipped_output)
        self.logger.info(f"Indexed output VCF file {bgzipped_output}.tbi")
        self.logger.info("Successfully wrote VCF file!")

        return self

    def _replace_alleles(self, row: List[str], ref: str, alts: List[str]) -> List[str]:
        """Replace the alleles in the VCF row with the corresponding VCF genotype codes.

        Args:
            row (List[str]): The VCF row to process.
            ref (str): The reference allele.
            alts (List[str]): The alternate alleles.

        Returns:
            List[str]: The processed VCF row.
        """

        def get_genotype_code(
            allele1: str, allele2: str, ref: str, alts: List[str]
        ) -> str:
            """Assigns the genotype code based on the decoded alleles.

            Args:
                allele1 (str): The first allele.
                allele2 (str): The second allele.
                ref (str): The reference allele.
                alts (List[str]): The alternate alleles.

            Returns:
                str: The genotype code in VCF format.
            """
            if allele1 == ref and allele2 == ref:
                return "0/0"
            elif allele1 == ref and allele2 in alts:
                return f"0/{alts.index(allele2) + 1}"
            elif allele1 in alts and allele2 == ref:
                return f"0/{alts.index(allele1) + 1}"
            elif allele1 in alts and allele2 in alts:
                index1 = alts.index(allele1) + 1
                index2 = alts.index(allele2) + 1
                return f"{min(index1, index2)}/{max(index1, index2)}"
            return "./."

        # Apply allele replacements
        for i in range(9, len(row)):
            data = row[i].split(":")
            iupac_code = data[0]
            fmt_data = data[1:] if len(data) > 1 else []

            # Decode the IUPAC code to get the alleles
            if iupac_code in self.reverse_iupac_mapping:
                alleles = self.reverse_iupac_mapping[iupac_code]
                # If multiple alleles are present in the mapping, take the first two
                if len(alleles) >= 2:
                    allele1, allele2 = alleles[:2]
                else:
                    allele1 = allele2 = alleles[0]
            else:
                # Handle cases where genotype is a single allele or missing
                if iupac_code in {"N", "-", ".", "?"}:
                    row[i] = "./."
                    continue
                else:
                    allele1 = allele2 = iupac_code

            # Assign genotype code based on decoded alleles
            row[i] = get_genotype_code(allele1, allele2, ref, alts)

            if fmt_data:
                fmt_data = ":".join(fmt_data)
                row[i] = f"{row[i]}:{fmt_data}"

        return row

    def _build_vcf_header(self) -> str:
        """Dynamically builds the VCF header based on the loaded sample IDs.

        Returns:
            str: The VCF header.
        """
        sample_headers = "\t".join(self.samples)

        vcf_header = textwrap.dedent(
            f"""\
            ##fileformat=VCFv4.2
            ##source=SNPio
            ##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">
            ##INFO=<ID=VAF,Number=A,Type=Float,Description="Variant Allele Frequency">
            ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_headers}\n"""
        )

        return vcf_header

    def update_vcf_attributes(
        self,
        snp_data: np.ndarray,
        sample_indices: np.ndarray,
        loci_indices: np.ndarray,
        samples: np.ndarray,
    ) -> None:
        """Updates the VCF attributes with new data in chunks.

        Args:
            snp_data (np.ndarray): The SNP data to update the VCF attributes with.
            sample_indices (np.ndarray): The indices of the samples to update.
            loci_indices (np.ndarray): The indices of the loci to update.
            samples (np.ndarray): The sample names to update.

        Raises:
            FileNotFoundError: If the VCF attributes file is not found.
        """
        self.snp_data = snp_data
        self.samples = samples
        self.sample_indices = sample_indices
        self.loci_indices = loci_indices

        hdf5_file_path = Path(self.vcf_attributes_fn)
        hdf5_file_filt = hdf5_file_path.with_name("vcf_attributes_filtered.h5")

        if not hdf5_file_path.exists() and not hdf5_file_path.is_file():
            msg = f"VCF attributes file {hdf5_file_path} not found."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        new_size = np.count_nonzero(loci_indices)

        # Helper function to process chunks
        def process_chunk(
            dataset_name: str, dtype: Any, fw: h5py.File, fr: h5py.File, new_size: int
        ) -> None:
            """Reads, filters, and writes a dataset in chunks.

            Args:
                dataset_name (str): The name of the dataset to process.
                dtype (object): The datatype of the dataset.
                fw (h5py.File): The filtered output file handle.
                fr (h5py.File): The input file handle.
                new_size (int): The new size of the dataset after filtering.
            """
            chunk_size_inner = self.chunk_size
            dset_size = fr[dataset_name].shape[0]
            fw.create_dataset(
                dataset_name, dtype=dtype, shape=(new_size,), maxshape=(None,)
            )

            # Reading and writing the dataset in chunks
            start_idx = 0
            write_idx = 0
            while start_idx < dset_size:
                end_idx = min(start_idx + chunk_size_inner, dset_size)
                chunk = fr[dataset_name][start_idx:end_idx]
                filtered_chunk = chunk[loci_indices[start_idx:end_idx]]
                chunk_size_to_write = filtered_chunk.size
                fw[dataset_name][
                    write_idx : write_idx + chunk_size_to_write
                ] = filtered_chunk
                write_idx += chunk_size_to_write
                start_idx = end_idx

        # Open the filtered output file for writing
        with h5py.File(hdf5_file_filt, "w") as fw:
            # Process each dataset in chunks
            with h5py.File(hdf5_file_path, "r") as fr:
                pcp = partial(process_chunk, fw=fw, fr=fr, new_size=new_size)
                pcp("chrom", h5py.string_dtype())
                pcp("pos", np.int32)
                pcp("id", h5py.string_dtype())
                pcp("ref", "S1")
                pcp("alt", h5py.string_dtype())
                pcp("qual", h5py.string_dtype())
                pcp("filt", "S4")
                pcp("fmt", h5py.string_dtype())
                pcp("fmt_data", h5py.string_dtype())
                [pcp(f"info/{k}", h5py.string_dtype()) for k in fr["info"].keys()]

        # Update the VCF file path to the new filtered file
        self.vcf_attributes_fn = str(hdf5_file_filt)

    @property
    def vcf_attributes_fn(self) -> str:
        """The path to the HDF5 file containing the VCF attributes.

        Returns:
            str: The path to the HDF5 file containing the VCF attributes.
        """
        return self._vcf_attributes_fn

    @vcf_attributes_fn.setter
    def vcf_attributes_fn(self, value: str) -> None:
        """Sets the path to the HDF5 file containing the VCF attributes.

        Args:
            value (str): The path to the HDF5 file containing the VCF attributes.
        """
        if isinstance(value, str):
            value = Path(value)

        self._vcf_attributes_fn = value
