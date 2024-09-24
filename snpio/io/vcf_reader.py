import textwrap
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from pysam import VariantFile
from tqdm import tqdm

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.benchmarking import Benchmark, measure_execution_time
from snpio.utils.logging import setup_logger


class VCFReader(GenotypeData):

    def __init__(
        self,
        filename: Optional[str] = None,
        popmapfile: Optional[str] = None,
        chunk_size: int = 1000,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        plot_format: str = "png",
        plot_fontsize: int = 12,
        plot_dpi: int = 300,
        plot_despine: bool = True,
        show_plots: bool = False,
        prefix: str = "snpio",
        verbose: bool = True,
        sample_indices: np.ndarray = None,
        loci_indices: np.ndarray = None,
        debug: bool = False,
        benchmark: bool = False,
    ) -> None:

        self._vcf_attributes_fn = (
            Path(f"{prefix}_output")
            / "gtdata"
            / "alignments"
            / "vcf"
            / "vcf_attributes.h5"
        )

        self.log_file = Path(f"{prefix}_output") / "logs" / "vcfreader.log"
        self.log_file.parent.mkdir(exist_ok=True, parents=True)
        self.log_level = "DEBUG" if debug else "INFO"

        self.logger = setup_logger(
            __name__, log_file=self.log_file, level=self.log_level
        )

        self.benchmark = benchmark
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
            benchmark=benchmark,
        )
        self.vcf_header = None
        self.num_records = 0
        self.info_fields = None
        self.filetype = "vcf"

        self.vcf_attributes_fn.parent.mkdir(exist_ok=True, parents=True)

    @measure_execution_time
    def load_aln(self) -> None:
        """Reads a VCF file into the VCFReader object."""
        if self.verbose:
            self.logger.info(f"Reading VCF file {self.filename}...")

        with VariantFile(self.filename, mode="r") as vcf:
            self.vcf_header = vcf.header
            self.samples = np.array(vcf.header.samples)
            self.num_records = sum(1 for _ in vcf)
            vcf.reset()

            self.vcf_attributes_fn, self.snp_data, self.samples = (
                self.get_vcf_attributes(vcf, chunk_size=self.chunk_size)
            )

            self.snp_data = self.snp_data.T

        if self.verbose:
            self.logger.info(
                f"VCF file successfully loaded with {self.snp_data.shape[1]} loci and {self.samples.shape[0]} samples."
            )

        self.logger.debug(f"snp_data: {self.snp_data}")

        return self

    def get_vcf_attributes(
        self, vcf: VariantFile, chunk_size: int = 1000
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Extracts VCF attributes and returns them in an efficient manner with chunked processing.

        Args:
            vcf (VariantFile): The VCF filename to extract attributes from.
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
                    k,
                    (0,),  # 1-dimensional shape
                    maxshape=(None,),  # Allow expansion along the first dimension
                    dtype=h5py.string_dtype(),
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

            for variant in tqdm(vcf.fetch(), desc="Processing VCF records: "):
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
                    "."
                    if variant.filter is None
                    else [v for v in variant.filter.keys()][0]
                )
                fmt_chunk.append(":".join([k for k in variant.format.keys()]))

                for k in info_fields:
                    value = variant.info.get(k, ".")
                    processed_value = (
                        ",".join(list(value)) if isinstance(value, tuple) else value
                    )
                    info_chunk[k].append(processed_value)

                gt_data = [
                    variant.samples[sample].get("GT", "./.") for sample in sample_names
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
                            info_chunk = defaultdict(list)

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

        snp_data = np.array(snp_data)

        return str(h5_outfile), snp_data, sample_names

    def transform_gt(self, gt: np.ndarray, ref: str, alts: List[str]) -> np.ndarray:
        """Transforms genotype tuples into their IUPAC codes or corresponding strings."""
        iupac_mapping = self._iupac_code()

        # Convert alts to a list, ensuring it's mutable and indexable
        alts = list(alts)

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

    def _iupac_code(self) -> callable:
        iupac_mapping = self._iupac_from_gt_tuples()

        def get_code(alleles):
            return iupac_mapping.get(tuple(sorted(alleles)), "N")

        return get_code

    def write_vcf(
        self,
        output_filename: str,
        hdf5_file_path: Optional[str] = None,
        chunk_size: int = 1000,
    ):
        """Writes the GenotypeData object data to a VCF file in chunks.

        Args:
            output_filename (str): The name of the output VCF file to write.
            hdf5_file_path (str, optional): The path to the HDF5 file containing the VCF attributes. Defaults to None.
            chunk_size (int, optional): The size of the chunks to read from the HDF5 file. Defaults to 1000.
        """
        if self.verbose:
            self.logger.info(f"Writing VCF file to: {output_filename}")

        if hdf5_file_path is None:
            hdf5_file_path = self.vcf_attributes_fn

        # Open the HDF5 file for reading and the output VCF file for writing
        with h5py.File(hdf5_file_path, "r") as hdf5_file, open(
            output_filename, "w"
        ) as f:
            # Write the VCF header
            f.write(self._build_vcf_header())

            # Get the total number of loci
            total_loci = len(hdf5_file["chrom"])

            chunk_size = self.chunk_size

            # Iterate over data in chunks
            for start in range(0, total_loci, chunk_size):
                end = min(start + chunk_size, total_loci)

                # Read data in chunks
                chrom = hdf5_file["chrom"][start:end].astype(str)
                pos = hdf5_file["pos"][start:end]
                vid = hdf5_file["id"][start:end]
                ref = hdf5_file["ref"][start:end]
                alt = hdf5_file["alt"][start:end]
                qual = hdf5_file["qual"][start:end]
                filt = hdf5_file["filt"][start:end]
                fmt = hdf5_file["fmt"][start:end]
                fmt_data = hdf5_file["fmt_data"][start:end]

                # Read info fields in chunks
                info_keys = list(hdf5_file["info"].keys())

                # Create a defaultdict to hold the info fields data
                info = defaultdict(list)
                info = {k: hdf5_file[f"info/{k}"][start:end] for k in info_keys}

                # Format the info fields into strings
                info_arrays = {
                    key: np.char.add(f"{key}=", info.get(key, ".").astype(str))
                    for key in info.keys()
                }

                info_arrays = np.array(list(info_arrays.values()), dtype=str)

                # Join the elements along the last axis to create the info
                # field strings
                info_result = np.apply_along_axis(lambda x: ";".join(x), 0, info_arrays)

                # Write each row to the VCF file
                for i in tqdm(range(len(chrom)), desc="Writing VCF records: "):
                    # Construct the ALT field by combining alternate alleles
                    alt_alleles_str = ",".join(alt[i].decode())
                    ref_str = ref[i].decode()
                    alts = alt[i].decode()

                    # Split fmt_data[i] by tab to get individual sample data
                    fmt_data_samples = fmt_data[i].decode().split("\t")

                    # Join fmt_data with snp_data by ":"
                    combined_data = [
                        f"{snp}:{fmt}"
                        for snp, fmt in zip(self.snp_data[:, i], fmt_data_samples)
                    ]

                    row = np.array(
                        [
                            chrom[i],
                            str(pos[i]),
                            vid[i].decode(),
                            ref[i].decode(),
                            alt_alleles_str,
                            qual[i].decode(),
                            filt[i].decode(),
                            ("." if info_result.size == 0 else info_result[i]),
                            fmt[i].decode(),
                        ]
                        + combined_data
                    )

                    # Replace alleles if necessary
                    row = self._replace_alleles(row, ref_str, [alts])

                    # Write the row to the file
                    f.write("\t".join(row) + "\n")

        if self.verbose:
            self.logger.info("Successfully wrote VCF file!")

        return self

    def _replace_alleles(
        self, row: np.ndarray, ref: str, alts: List[str]
    ) -> np.ndarray:
        """Replace the alleles in the VCF row with the corresponding VCF genotype codes."""

        def get_genotype_code(allele1, allele2, ref, alts):
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
                allele1, allele2 = self.reverse_iupac_mapping[iupac_code]
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
        """Dynamically builds the VCF header based on the loaded sample IDs."""
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
        """
        self.snp_data = snp_data
        self.samples = samples
        self.sample_indices = sample_indices
        self.loci_indices = loci_indices

        hdf5_file_path = self.vcf_attributes_fn
        hdf5_file_path_filt = hdf5_file_path.with_name("vcf_attributes_filtered.h5")

        if not hdf5_file_path.exists() and not hdf5_file_path.is_file():
            msg = f"VCF attributes file {hdf5_file_path} not found."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        new_size = np.count_nonzero(loci_indices)

        # Helper function to process chunks
        def process_chunk(dataset_name, dtype, fw, fr, new_size):
            """Reads, filters, and writes a dataset in chunks.

            Args:
                dataset_name (str): The name of the dataset to process.
                dtype (dtype): The datatype of the dataset.
                fw (h5py.File): The filtered output file handle.
                fr (h5py.File): The input file handle.
                new_size (int): The new size of the dataset after filtering.

            Returns:
                None

            Raises:
                None
            """
            chunk_size = self.chunk_size
            dset_size = fr[dataset_name].shape[0]
            fw.create_dataset(
                dataset_name, dtype=dtype, shape=(new_size,), maxshape=(None,)
            )

            # Reading and writing the dataset in chunks
            start_idx = 0
            write_idx = 0
            while start_idx < dset_size:
                end_idx = min(start_idx + chunk_size, dset_size)
                chunk = fr[dataset_name][start_idx:end_idx]
                filtered_chunk = chunk[loci_indices[start_idx:end_idx]]
                chunk_size_to_write = filtered_chunk.size
                fw[dataset_name][
                    write_idx : write_idx + chunk_size_to_write
                ] = filtered_chunk
                write_idx += chunk_size_to_write
                start_idx = end_idx

        # Open the filtered output file for writing
        with h5py.File(hdf5_file_path_filt, "w") as fw:
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
        self.vcf_attributes_fn = hdf5_file_path_filt

    @property
    def vcf_attributes_fn(self) -> str:
        return self._vcf_attributes_fn

    @vcf_attributes_fn.setter
    def vcf_attributes_fn(self, value: str) -> None:
        if isinstance(value, str):
            value = Path(value)

        self._vcf_attributes_fn = value
