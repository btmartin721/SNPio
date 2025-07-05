import gc
from collections import deque
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import h5py
import numpy as np
import pysam
from tqdm import tqdm

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


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
        disable_progress_bar: bool = False,
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
            store_format_fields (bool): Whether to store per-locus-per-sample FORMAT fields. Note that setting this parameter to True may result in an increase in runtime and memory usage. Defaults to False.
            disable_progress_bar (bool): Whether to disable the progress bar. If True, disables the progress bar. Defaults to False.
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

        Notes:
            - Setting `store_format_fields` to True slows down VCFReader. If you don't need the per-sample-per-locus metadata, leave this at False.

        """
        outdir = Path(f"{prefix}_output") / "alignments" / "vcf"
        outdir.mkdir(exist_ok=True, parents=True)

        self._vcf_attributes_fn = outdir / "vcf_attributes.h5"
        self.store_format_fields = store_format_fields
        self.disable_progress_bar = disable_progress_bar

        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(name=__name__, **kwargs)
        self.logger = logman.get_logger()

        self.resource_data = {}

        self.iupac = IUPAC(logger=self.logger)

        self.marker_names = []

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

        consumeall = deque(maxlen=0).extend

        def ilen(it):
            """Count number of items in an iterator without loading into memory.

            This function uses a stateful counting iterator to efficiently count items in an iterable without loading the entire iterable into memory. It uses the `zip` function to pair each item with a count, and then consumes the iterator until it is exhausted.

            Args:
                it (iterable): An iterable object (e.g., a file or generator).

            Returns:
                int: The number of items in the iterable.
            """
            # Make a stateful counting iterator
            cnt = count()

            # zip with input iterator, drain until input exhausted at C level
            # NOTE: cnt must be second zip arg to avoid advancing too far
            # Since count 0 based, the next value is the count
            consumeall(zip(it, cnt))
            return next(cnt)

        with pysam.VariantFile(self.filename, mode="r") as vcf:
            self.vcf_header = vcf.header
            self.samples = np.array(vcf.header.samples)

            # Count records without loading all into memory.
            # Faster than sum(1 for _ in vcf) or len(list(vcf))
            self.num_records = ilen(vcf)
            vcf.reset()

            self.chunk_size = min(self.chunk_size, self.num_records)

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

        # buffers for chunked write; POS as integer
        chrom_buf = np.empty((chunk_size,), dtype=object)
        pos_buf = np.empty((chunk_size,), dtype=np.int64)
        id_buf = np.empty((chunk_size,), dtype=object)
        qual_buf = np.empty((chunk_size,), dtype=object)
        ref_buf = np.empty((chunk_size,), dtype=object)
        alt_buf = np.empty((chunk_size,), dtype=object)
        filt_buf = np.empty((chunk_size,), dtype=object)
        fmt_buf = np.empty((chunk_size,), dtype=object)
        info_buf = {k: np.empty((chunk_size,), dtype=object) for k in info_keys}
        snp_matrix = np.empty((n_vars, n_samples), dtype="<U1")

        # Preallocated FORMAT buffers
        if self.store_format_fields:
            fmt_data_buf = {
                k: np.empty((chunk_size, n_samples), dtype=object)
                for k in fmt_keys
                if k != "GT"
            }

        # Overwrite existing file if present
        if h5_path.exists() and h5_path.is_file():
            self.logger.warning(f"File {h5_path} already exists. Overwriting.")
            h5_path.unlink()

        iupac_map = self.iupac.get_tuple_to_iupac()

        @lru_cache(maxsize=None)
        def cached_convert(gt_pair: Tuple[int, int], alleles: Tuple[str, ...]) -> str:
            """Convert genotype pair to IUPAC code using cached function."""
            bases = tuple(alleles[i] if i is not None else "N" for i in gt_pair)
            return iupac_map[tuple(set(bases))] if bases else "N"

        with h5py.File(h5_path, "w") as h5:
            # core fields: POS as integer, others as utf-8 strings
            for name in ("chrom", "pos", "id", "ref", "alt", "qual", "filt", "fmt"):
                if name == "pos":
                    dtype = np.int64
                elif name == "ref":
                    dtype = h5py.string_dtype(encoding="utf-8", length=1)
                elif name in ("alt", "qual"):
                    dtype = h5py.string_dtype(encoding="utf-8", length=10)
                elif name == "filt":
                    dtype = h5py.string_dtype(encoding="utf-8", length=4)
                else:
                    dtype = h5py.string_dtype(encoding="utf-8")

                h5.create_dataset(
                    name,
                    shape=(n_vars,),
                    maxshape=(n_vars,),
                    chunks=(chunk_size,),
                    dtype=dtype,
                    compression="lzf",
                )

            # INFO datasets
            info_grp = h5.create_group("info")
            for key in info_keys:
                info_grp.create_dataset(
                    key,
                    shape=(n_vars,),
                    maxshape=(n_vars,),
                    chunks=(chunk_size,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression="lzf",
                )

            # FORMAT datasets
            if self.store_format_fields:
                fmt_grp = h5.create_group("fmt_metadata")
                for key in fmt_data_buf:
                    fmt_grp.create_dataset(
                        key,
                        shape=(n_vars, n_samples),
                        maxshape=(n_vars, n_samples),
                        chunks=(chunk_size, n_samples),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                        compression="lzf",
                    )

            # Iterate and fill buffers
            for idx, record in enumerate(
                tqdm(
                    vcf.fetch(),
                    desc="Reading VCF: ",
                    total=n_vars,
                    unit=" records",
                    disable=self.disable_progress_bar,
                )
            ):
                row_in_chunk = idx % chunk_size

                chrom = str(record.chrom)
                pos = record.pos

                self.marker_names.append(f"{chrom}:{str(pos)}")

                # Core VCF fields (leave POS as an integer)
                chrom_buf[row_in_chunk] = chrom
                pos_buf[row_in_chunk] = pos
                id_buf[row_in_chunk] = str(record.id) or "."
                ref_buf[row_in_chunk] = str(record.ref)
                alt_buf[row_in_chunk] = ",".join(str(record.alts) or ["."])
                qual_buf[row_in_chunk] = str(record.qual) or "."
                filt_buf[row_in_chunk] = next(iter(record.filter.keys()), ".")
                fmt_buf[row_in_chunk] = ":".join(record.format.keys())

                # INFO fields
                for key in info_keys:
                    val = record.info.get(key, None)
                    if val is None:
                        info_buf[key][row_in_chunk] = "."
                    elif isinstance(val, (tuple, list)):
                        info_buf[key][row_in_chunk] = ",".join([str(v) for v in val])
                    else:
                        info_buf[key][row_in_chunk] = str(val)

                # FORMAT fields
                if self.store_format_fields:
                    self._store_format_fields(
                        fmt_data_buf, record, samples, row_in_chunk
                    )

                # Convert genotypes to IUPAC codes
                alleles_tuple = tuple(record.alleles)
                snp_matrix[idx, :] = np.array(
                    [
                        cached_convert(
                            record.samples[sample].get("GT", (None, None)),
                            alleles_tuple,
                        )
                        for sample in samples
                    ],
                    dtype="<U1",
                )

                # When chunk is full, write out
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
                        info_buf[key] = np.empty((chunk_size,), dtype=object)

                    if self.store_format_fields:
                        for fmt_key, buf in fmt_data_buf.items():
                            fmt_grp[fmt_key][start:end, :] = buf

                    # reset buffers
                    chrom_buf = np.empty((chunk_size,), dtype=object)
                    pos_buf = np.empty((chunk_size,), dtype=np.int64)
                    id_buf = np.empty((chunk_size,), dtype=object)
                    qual_buf = np.empty((chunk_size,), dtype=object)
                    ref_buf = np.empty((chunk_size,), dtype=object)
                    alt_buf = np.empty((chunk_size,), dtype=object)
                    filt_buf = np.empty((chunk_size,), dtype=object)
                    fmt_buf = np.empty((chunk_size,), dtype=object)
                    if self.store_format_fields:
                        fmt_data_buf = {
                            fmt_key: np.empty((chunk_size, n_samples), dtype=object)
                            for fmt_key in fmt_data_buf
                        }

            # Final partial chunk
            remainder = n_vars % chunk_size
            if remainder:
                start = n_vars - remainder
                end = n_vars

                h5["chrom"][start:end] = chrom_buf[:remainder]
                h5["pos"][start:end] = pos_buf[:remainder]
                h5["id"][start:end] = id_buf[:remainder]
                h5["ref"][start:end] = ref_buf[:remainder]
                h5["alt"][start:end] = alt_buf[:remainder]
                h5["qual"][start:end] = qual_buf[:remainder]
                h5["filt"][start:end] = filt_buf[:remainder]
                h5["fmt"][start:end] = fmt_buf[:remainder]

                for key in info_keys:
                    h5["info"][key][start:end] = info_buf[key][:remainder]

                if self.store_format_fields:
                    for fmt_key, buf in fmt_data_buf.items():
                        fmt_grp[fmt_key][start:end, :] = buf[:remainder, :]

        return h5_path, snp_matrix, samples

    def _store_format_fields(
        self,
        fmt_data_buf: Dict[str, np.ndarray],
        record: pysam.libcbcf.VariantRecord,
        samples: np.ndarray,
        row_in_chunk: int,
    ):
        """Efficiently store FORMAT fields per sample using sample index arrays.

        This method fills pre-allocated buffers for FORMAT fields in a VCF record. It processes each sample's FORMAT data and stores it in the corresponding row of the buffer. It handles different data types (strings, lists, tuples) and ensures that missing values are represented as ".".

        Args:
            fmt_data_buf (Dict[str, np.ndarray]): Pre-allocated buffers for FORMAT fields.
            record (pysam.libcbcf.VariantRecord): The current VCF record.
            samples (np.ndarray): Array of sample names.
            row_in_chunk (int): The current row index within the chunk.

        Notes:
            - This method modifies `fmt_data_buf` in-place. It does not return a value.
        """
        # Extract sample values for the current record once (dict view → list of dicts)
        sample_data = [record.samples[sample] for sample in samples]

        for fmt_key, fmt_buf in fmt_data_buf.items():
            row = fmt_buf[row_in_chunk]

            for s_idx, s_fmt in enumerate(sample_data):
                val = s_fmt.get(fmt_key)

                if val is None:
                    row[s_idx] = "."
                elif isinstance(val, str):
                    row[s_idx] = val
                elif isinstance(val, tuple):
                    # Tuples are most common in VCF FORMAT fields (e.g., GT, AD)
                    row[s_idx] = ",".join(map(str, val))
                elif isinstance(val, list):
                    # If expect lists, handle explicitly to avoid rare slowdown
                    row[s_idx] = ",".join(map(str, val))
                else:
                    row[s_idx] = str(val)

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
        new_size = np.count_nonzero(loci_indices)
        new_sample_size = np.count_nonzero(sample_indices)

        if snp_data.size == 0 or new_size == 0 or new_sample_size == 0:
            if snp_data.size == 0:
                self.logger.warning("snp_data is empty. Skipping VCF attribute update.")
            elif new_size == 0:
                self.logger.warning(
                    "No loci left after filtering. Skipping VCF attribute update."
                )
            elif new_sample_size == 0:
                self.logger.warning(
                    "No samples left after filtering. Skipping VCF attribute update."
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

        def process_chunk(
            dataset_name: str,
            dtype: Any,
            fw: h5py.File,
            fr: h5py.File,
            sample_size: int | None = None,
        ) -> None:
            """Process a chunk of data and write it to the HDF5 file.

            Args:
                dataset_name (str): The name of the dataset to process.
                dtype (Any): The data type of the dataset.
                fw (h5py.File): The HDF5 file to write to.
                fr (h5py.File): The HDF5 file to read from.
                sample_size (int | None): The size of the sample to process. Defaults to None.
            """
            new_shape = (new_size, sample_size) if sample_size else (new_size,)

            chunk_size_inner = min(self.chunk_size, new_size)
            dset_size = fr[dataset_name].shape[0]
            fw.create_dataset(
                dataset_name,
                dtype=dtype,
                shape=new_shape,
                maxshape=new_shape,
                chunks=(
                    (chunk_size_inner, sample_size)
                    if sample_size
                    else (chunk_size_inner,)
                ),
                compression="lzf",
            )

            start_idx = 0
            write_idx = 0
            global_indices = np.arange(dset_size)

            while start_idx < dset_size:
                end_idx = min(start_idx + chunk_size_inner, dset_size)

                # Get the chunk of data from the original dataset
                chunk = fr[dataset_name][start_idx:end_idx]
                chunk_indices = global_indices[start_idx:end_idx]
                mask = loci_indices[chunk_indices]

                filtered_chunk = (
                    chunk[mask]
                    if sample_size is None
                    else chunk[mask][self.sample_indices]
                )

                # Cast to proper dtype to avoid HDF5 conversion errors
                filtered_chunk = np.array(filtered_chunk, dtype=dtype)

                fw[dataset_name][
                    write_idx : write_idx + filtered_chunk.size
                ] = filtered_chunk
                write_idx += filtered_chunk.size
                start_idx = end_idx

                del chunk, filtered_chunk, mask
                gc.collect()

        with h5py.File(hdf5_file_filt, "w") as fw, h5py.File(hdf5_file_path, "r") as fr:

            datasets_to_process = [
                ("chrom", h5py.string_dtype(encoding="utf-8")),
                ("pos", np.int64),
                ("id", h5py.string_dtype(encoding="utf-8")),
                ("ref", h5py.string_dtype(encoding="utf-8", length=1)),
                ("alt", h5py.string_dtype(encoding="utf-8", length=10)),
                ("qual", h5py.string_dtype(encoding="utf-8", length=10)),
                ("filt", h5py.string_dtype(encoding="utf-8", length=4)),
                ("fmt", h5py.string_dtype(encoding="utf-8")),
            ]

            [
                process_chunk(dataset_name, dtype, fw, fr)
                for dataset_name, dtype in datasets_to_process
            ]

            if "info" in fr:
                fw.create_group("info")
                [
                    process_chunk(
                        f"info/{key}", h5py.string_dtype(encoding="utf-8"), fw, fr
                    )
                    for key in fr["info"]
                ]

            if self.store_format_fields and "fmt_metadata" in fr:
                fw.create_group("fmt_metadata")
                [
                    process_chunk(
                        f"fmt_metadata/{fmt_key}",
                        h5py.string_dtype(encoding="utf-8"),
                        fw,
                        fr,
                        sample_size=new_sample_size,
                    )
                    for fmt_key in fr["fmt_metadata"]
                ]

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
