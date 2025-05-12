from pathlib import Path
from typing import Any, List, Literal, Tuple

import h5py
import numpy as np
import pysam
from tqdm import tqdm

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.logging import LoggerManager
import snpio.utils.custom_exceptions as exceptions


class VCFReader(GenotypeData):
    """A class to read VCF files into GenotypeData objects and write GenotypeData objects to VCF files.

    VCFReader serves as the entry point to read and write VCF file formats. It is a subclass of GenotypeData that provides methods to read VCF files and extract the necessary attributes. It also provides methods to write GenotypeData objects to VCF files. The class uses the pysam library to read VCF files and the h5py library to write HDF5 files.

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
        filename (str | None): The name of the VCF file to read.
        popmapfile (str | None): The name of the population map file to read.
        chunk_size (int): The size of the chunks to read from the VCF file.
        store_format_fields (bool): Whether to store FORMAT fields. Setting to True may result in an increase in runtime and memory usage.
        force_popmap (bool): Whether to force the use of the population map file.
        exclude_pops (List[str] | None): The populations to exclude.
        include_pops (List[str] | None): The populations to include.
        plot_format (Literal["png", "pdf", "jpg", "svg"] | None): The format to save the plots in.
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
        vcf_header (pysam.libcbcf.VariantHeader | None): The VCF header.
        info_fields (List[str] | None): The VCF info fields.
        resource_data (dict): A dictionary to store resource data.
        logger (logging.Logger): The logger object.
        snp_data (np.ndarray): The SNP data.
        samples (np.ndarray): The sample names.
        vcf_attributes_fn (Path): The path to the HDF5 file containing the VCF attributes.

    Note:
        The VCF file is bgzipped, sorted, and indexed using Tabix to ensure efficient reading, if necessary.
    """

    def __init__(
        self,
        filename: str | None = None,
        popmapfile: str | None = None,
        chunk_size: int = 1000,
        store_format_fields: bool = False,
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
        sample_indices: np.ndarray | None = None,
        loci_indices: np.ndarray | None = None,
        debug: bool = False,
    ) -> None:
        """Initializes the VCFReader object.

        This method sets up the VCFReader object with the provided parameters. It initializes the logger, sets the file paths, and prepares the output directory. The VCF file is bgzipped, sorted, and indexed using Tabix to ensure efficient reading.

        Args:
            filename (str | None): The name of the VCF file to read. Defaults to None.
            popmapfile (str | None): The name of the population map file to read. Defaults to None.
            chunk_size (int): The size of the chunks to read from the VCF file. Defaults to 1000.
            store_format_fields (bool): Whether to store FORMAT fields. Defaults to False.
            force_popmap (bool): Whether to force the use of the population map file. Defaults to False.
            exclude_pops (List[str] | None): The populations to exclude. Defaults to None.
            include_pops (List[str] | None): The populations to include. Defaults to None.
            plot_format (Literal["png", "pdf", "jpg", "svg"] | None): The format to save the plots in. Defaults to "png".
            plot_fontsize (int): The font size for the plots. Defaults to 18.
            plot_dpi (int): The DPI for the plots. Defaults to 300.
            plot_despine (bool): Whether to remove the spines from the plots. Defaults to True.
            show_plots (bool): Whether to show the plots. Defaults to False.
            prefix (str): The prefix to use for the output files. Defaults to "snpio".
            verbose (bool): Whether to print verbose output. Defaults to False.
            sample_indices (np.ndarray | None): The indices of the samples to read. Defaults to None.
            loci_indices (np.ndarray | None): The indices of the loci to read. Defaults to None.
            debug (bool): Whether to enable debug mode. Defaults to False.
        """
        outdir = Path(f"{prefix}_output") / "gtdata" / "alignments" / "vcf"
        self._vcf_attributes_fn = outdir / "vcf_attributes.h5"
        self.store_format_fields = store_format_fields

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
        self.from_vcf = True

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

    def load_aln(self) -> None:
        """Loads the alignment from the VCF file into the VCFReader object.

        This method ensures that the input VCF file is bgzipped, sorted, and indexed using Tabix. It then reads the VCF file and extracts the necessary attributes. The VCF attributes are stored in an HDF5 file for efficient access. The genotype data is transformed into IUPAC codes for efficient storage and processing.
        """
        self.logger.info(f"Processing input VCF file {self.filename}...")

        vcf_path = Path(self.filename)
        bgzipped_vcf = vcf_path

        # Ensure the VCF file is bgzipped
        if not self._is_bgzipped(vcf_path):
            bgzipped_vcf = self.bgzip_file(vcf_path)
            self.logger.info(f"Bgzipped VCF file to {bgzipped_vcf}")

        # Ensure the VCF file is sorted
        if not self._is_sorted(bgzipped_vcf):
            bgzipped_vcf = self._sort_vcf_file(bgzipped_vcf)
            self.logger.info(f"Sorted VCF file to {bgzipped_vcf}")

        # Ensure the VCF file is indexed
        if not self._has_tabix_index(bgzipped_vcf):
            self.tabix_index(bgzipped_vcf)
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
    ) -> Tuple[Path, np.ndarray, np.ndarray]:
        """Extract VCF attributes with pysam, chunked HDF5 writes, and vectorized GT→IUPAC.

        Args:
            vcf (pysam.VariantFile): Open bgzipped & indexed VCF.
            chunk_size (int): Number of records per HDF5 write chunk.

        Returns:
            Tuple[Path, np.ndarray, np.ndarray]:
                - Path to the HDF5 file
                - SNP matrix of shape (n_variants, n_samples) (IUPAC codes)
                - Array of sample names
        """
        samples = np.array(vcf.header.samples)
        n_samples = samples.size
        n_vars = self.num_records

        info_keys = list(vcf.header.info.keys())
        fmt_keys = list(vcf.header.formats.keys()) if self.store_format_fields else []

        h5_path = self.vcf_attributes_fn
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        snp_matrix = np.empty((n_vars, n_samples), dtype="<U1")

        # Core buffers
        chrom_buf, pos_buf, id_buf, qual_buf, ref_buf, alt_buf = [], [], [], [], [], []
        filt_buf, fmt_buf = [], []
        info_buf = {k: [] for k in info_keys}

        # Preallocated FORMAT buffers
        if self.store_format_fields:
            fmt_data_buf = {
                k: np.empty((chunk_size, n_samples), dtype=object)
                for k in fmt_keys
                if k != "GT"
            }

        if h5_path.exists() and h5_path.is_file():
            self.logger.warning(f"File {h5_path} already exists. Overwriting.")
            h5_path.unlink()

        with h5py.File(h5_path, "w") as h5:
            # Core field datasets
            h5.create_dataset(
                "chrom",
                shape=(n_vars,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=h5py.string_dtype(),
                compression="gzip",
            )
            h5.create_dataset(
                "pos",
                shape=(n_vars,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=np.int32,
                compression="gzip",
            )
            for name in ("id", "ref", "alt", "qual", "filt", "fmt"):
                h5.create_dataset(
                    name,
                    shape=(n_vars,),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    dtype=h5py.string_dtype(),
                    compression="gzip",
                )

            # INFO datasets
            info_grp = h5.create_group("info")
            for key in info_keys:
                info_grp.create_dataset(
                    key,
                    shape=(n_vars,),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    dtype=h5py.string_dtype(),
                    compression="gzip",
                )

            # FORMAT datasets
            if self.store_format_fields:
                fmt_grp = h5.create_group("fmt_metadata")
                for key in fmt_data_buf:
                    fmt_grp.create_dataset(
                        key,
                        shape=(n_vars, n_samples),
                        maxshape=(None, n_samples),
                        chunks=(chunk_size, n_samples),
                        dtype=h5py.string_dtype(),
                        compression="gzip",
                    )

            for idx, record in enumerate(tqdm(vcf.fetch(), total=n_vars, unit="rec")):
                row_in_chunk = idx % chunk_size

                # Core VCF fields
                chrom_buf.append(record.chrom)
                pos_buf.append(record.pos)
                id_buf.append(record.id or ".")
                ref_buf.append(record.ref)
                alt_buf.append(",".join(record.alts or ["."]))
                qual_buf.append(record.qual or ".")
                filt_buf.append(next(iter(record.filter.keys()), "."))
                fmt_buf.append(":".join(record.format.keys()))

                # INFO fields
                for key in info_keys:
                    val = record.info.get(key, None)
                    if val is None:
                        info_buf[key].append(".")
                    elif isinstance(val, (tuple, list)):
                        info_buf[key].append(",".join(map(str, val)))
                    else:
                        info_buf[key].append(str(val))

                # FORMAT fields (per sample)
                if self.store_format_fields:
                    for fmt_key in fmt_data_buf:
                        for s_idx, sample in enumerate(samples):
                            val = record.samples[sample].get(fmt_key, None)
                            if val is None:
                                fmt_data_buf[fmt_key][row_in_chunk, s_idx] = "."
                            elif isinstance(val, (tuple, list)):
                                fmt_data_buf[fmt_key][row_in_chunk, s_idx] = ",".join(
                                    map(str, val)
                                )
                            else:
                                fmt_data_buf[fmt_key][row_in_chunk, s_idx] = str(val)

                # Genotypes → IUPAC codes
                alleles = [record.ref] + list(record.alts or [])
                iupac_map = self._iupac_code()

                def convert(gt_pair):
                    a1, a2 = gt_pair
                    if a1 is None or a2 is None:
                        return "N"
                    try:
                        base1, base2 = alleles[a1], alleles[a2]
                        return iupac_map.get((base1, base2), "N")
                    except Exception:
                        return "N"

                raw_gt = [
                    record.samples[sample].get("GT", (None, None)) for sample in samples
                ]
                snp_matrix[idx, :] = [convert(gt) for gt in raw_gt]

                # Write full chunk
                if (idx + 1) % chunk_size == 0:
                    start = idx + 1 - chunk_size
                    end = idx + 1

                    h5["chrom"][start:end] = chrom_buf
                    h5["pos"][start:end] = pos_buf
                    h5["id"][start:end] = id_buf
                    h5["ref"][start:end] = ref_buf
                    h5["alt"][start:end] = alt_buf
                    h5["qual"][start:end] = qual_buf
                    h5["filt"][start:end] = filt_buf
                    h5["fmt"][start:end] = fmt_buf

                    for key in info_keys:
                        h5["info"][key][start:end] = info_buf[key]
                        info_buf[key].clear()

                    if self.store_format_fields:
                        for fmt_key, buf in fmt_data_buf.items():
                            fmt_grp[fmt_key][start:end, :] = buf
                            buf[:, :] = "."  # reset buffer in-place

                    # Reset core buffers
                    chrom_buf.clear()
                    pos_buf.clear()
                    id_buf.clear()
                    ref_buf.clear()
                    alt_buf.clear()
                    qual_buf.clear()
                    filt_buf.clear()
                    fmt_buf.clear()

            # Final partial chunk
            remainder = len(chrom_buf)
            if remainder:
                start = n_vars - remainder
                end = n_vars

                h5["chrom"][start:end] = chrom_buf
                h5["pos"][start:end] = pos_buf
                h5["id"][start:end] = id_buf
                h5["ref"][start:end] = ref_buf
                h5["alt"][start:end] = alt_buf
                h5["qual"][start:end] = qual_buf
                h5["filt"][start:end] = filt_buf
                h5["fmt"][start:end] = fmt_buf

                for key in info_keys:
                    h5["info"][key][start:end] = info_buf[key]

                if self.store_format_fields:
                    for fmt_key, buf in fmt_data_buf.items():
                        fmt_grp[fmt_key][start:end, :] = buf[:remainder, :]

        return h5_path, snp_matrix, samples

    def _iupac_code(self) -> dict[tuple[str, str], str]:
        """Return a dictionary mapping base tuples to IUPAC codes.

        The IUPAC codes are used to represent ambiguous bases in DNA sequences.

        Returns:
            dict[tuple[str, str], str]: A dictionary mapping base tuples to IUPAC codes.
        """
        # Ensure all allele combinations, including reversed pairs, are covered
        iupac = {
            ("A", "A"): "A",
            ("C", "C"): "C",
            ("G", "G"): "G",
            ("T", "T"): "T",
            ("A", "G"): "R",
            ("G", "A"): "R",
            ("C", "T"): "Y",
            ("T", "C"): "Y",
            ("G", "C"): "S",
            ("C", "G"): "S",
            ("A", "C"): "M",
            ("C", "A"): "M",
            ("G", "T"): "K",
            ("T", "G"): "K",
            ("A", "T"): "W",
            ("T", "A"): "W",
            ("N", "N"): "N",
        }
        return iupac

    def transform_gt(self, gt: np.ndarray, ref: str, alts: List[str]) -> np.ndarray:
        """Transforms genotype tuples into their IUPAC codes or corresponding strings.

        Args:
            gt (np.ndarray): The genotype tuples to transform.
            ref (str): The reference allele.
            alts (List[str]): The alternate alleles.

        Returns:
            np.ndarray: The transformed genotype tuples.

        Raises:
            InvalidGenotypeError: If the genotype tuples are invalid.
        """
        iupac_mapping = self._iupac_code()

        # Convert alts to a list, ensuring it's mutable and indexable
        alts = list(alts) if alts else []

        # Prepare the array of possible alleles including reference
        alt_array = np.array([ref] + alts)

        if not alts:
            alts = []  # Default to an empty list if no alternate alleles are provided

        # Validate genotype tuples
        valid_indices = set(range(len(alt_array)))
        for g in gt:
            if g[0] not in valid_indices or g[1] not in valid_indices:
                msg = (
                    f"Invalid genotype tuple: {g}. Expected indices in {valid_indices}."
                )
                self.logger.error(msg)
                raise exceptions.InvalidGenotypeError(msg)

        transformed_gt = np.array(
            [iupac_mapping.get((alt_array[g[0]], alt_array[g[1]]), "N") for g in gt]
        )
        return transformed_gt

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

    def update_vcf_attributes(
        self,
        snp_data: np.ndarray,
        sample_indices: np.ndarray,
        loci_indices: np.ndarray,
        samples: np.ndarray,
    ) -> str:
        """Updates the VCF attributes with new data in chunks.

        This method updates the VCF attributes in the HDF5 file with new data. It processes the data in chunks to reduce memory usage and ensures that the data is written correctly.

        Args:
            snp_data (np.ndarray): The SNP data to update the VCF attributes with.
            sample_indices (np.ndarray): The indices of the samples to update.
            loci_indices (np.ndarray): The indices of the loci to update.
            samples (np.ndarray): The sample names to update.

        Returns:
            str: Path to the new filtered HDF5 file.

        Raises:
            FileNotFoundError: If the VCF attributes file is not found.
        """
        if snp_data.size == 0 or np.count_nonzero(loci_indices) == 0:
            self.logger.warning(
                "No loci left after filtering. Skipping VCF attribute update."
            )
            return

        self.snp_data = snp_data
        self.samples = samples
        self.sample_indices = sample_indices
        self.loci_indices = loci_indices

        hdf5_file_path = Path(self.vcf_attributes_fn)
        hdf5_file_filt = hdf5_file_path.with_name("vcf_attributes_filtered.h5")

        if not hdf5_file_path.exists():
            msg = f"VCF attributes file {hdf5_file_path} not found."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        new_size = np.count_nonzero(loci_indices)

        def process_chunk(
            dataset_name: str, dtype: Any, fw: h5py.File, fr: h5py.File
        ) -> None:
            """Process a chunk of data and write it to the HDF5 file.

            Args:
                dataset_name (str): The name of the dataset to process.
                dtype (Any): The data type of the dataset.
                fw (h5py.File): The HDF5 file to write to.
                fr (h5py.File): The HDF5 file to read from.
            """
            chunk_size_inner = self.chunk_size
            dset_size = fr[dataset_name].shape[0]
            fw.create_dataset(
                dataset_name,
                dtype=dtype,
                shape=(new_size,),
                maxshape=(None,),
                compression="gzip",
            )

            start_idx = 0
            write_idx = 0
            global_indices = np.arange(dset_size)

            while start_idx < dset_size:
                end_idx = min(start_idx + chunk_size_inner, dset_size)
                chunk = fr[dataset_name][start_idx:end_idx]

                chunk_indices = global_indices[start_idx:end_idx]
                mask = loci_indices[chunk_indices]
                filtered_chunk = chunk[mask]

                # Cast to proper dtype to avoid HDF5 conversion errors
                filtered_chunk = np.array(filtered_chunk, dtype=dtype)

                fw[dataset_name][
                    write_idx : write_idx + filtered_chunk.size
                ] = filtered_chunk
                write_idx += filtered_chunk.size
                start_idx = end_idx

        with h5py.File(hdf5_file_filt, "w") as fw, h5py.File(hdf5_file_path, "r") as fr:
            string_dtype = h5py.string_dtype(encoding="utf-8")

            datasets_to_process = [
                ("chrom", string_dtype),
                ("pos", np.int32),
                ("id", string_dtype),
                ("ref", string_dtype),
                ("alt", string_dtype),
                ("qual", string_dtype),
                ("filt", string_dtype),
                ("fmt", string_dtype),
            ]

            for dataset_name, dtype in datasets_to_process:
                process_chunk(dataset_name, dtype, fw, fr)

            if "info" in fr:
                fw.create_group("info")
                for key in fr["info"]:
                    process_chunk(f"info/{key}", string_dtype, fw, fr)

        self.vcf_attributes_fn = hdf5_file_filt
        return self.vcf_attributes_fn

    @property
    def vcf_attributes_fn(self) -> Path:
        """The path to the HDF5 file containing the VCF attributes.

        This property returns the path to the HDF5 file containing the VCF attributes. The file is created when the VCF file is loaded and can be used for further processing. It is stored as a Path object for easy manipulation.

        Returns:
            Path: The path to the HDF5 file containing the VCF attributes.
        """
        if not isinstance(self._vcf_attributes_fn, Path):
            self._vcf_attributes_fn = Path(self._vcf_attributes_fn)
        return self._vcf_attributes_fn

    @vcf_attributes_fn.setter
    def vcf_attributes_fn(self, value: str | Path) -> None:
        """Sets the path to the HDF5 file containing the VCF attributes.

        This setter method allows you to set the path to the HDF5 file containing the VCF attributes. It ensures that the value is a Path object.

        Args:
            value (str): The path to the HDF5 file containing the VCF attributes.
        """
        if isinstance(value, str):
            value = Path(value)

        self._vcf_attributes_fn = value
