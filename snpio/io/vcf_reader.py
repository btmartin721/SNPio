import random
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from pysam import VariantFile

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.custom_exceptions import NoValidAllelesError
from snpio.utils.logging import setup_logger

# Set up logger
logger = setup_logger(__name__)


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
        prefix: str = "snpio",
        verbose: bool = True,
        sample_indices: np.ndarray = None,
        loci_indices: np.ndarray = None,
        **kwargs,
    ) -> None:
        super().__init__(
            filename=filename,
            filetype="vcf",
            popmapfile=popmapfile,
            force_popmap=force_popmap,
            exclude_pops=exclude_pops,
            include_pops=include_pops,
            plot_format=plot_format,
            prefix=prefix,
            verbose=verbose,
            sample_indices=sample_indices,
            loci_indices=loci_indices,
            chunk_size=chunk_size,
        )
        self.vcf_header = None
        self.num_records = 0

        self._vcf_attributes_fn: str = kwargs.get("vcf_attributes", None)

    def load_aln(self) -> None:
        """Reads a VCF file into the VCFReader object."""
        logger.info(f"Reading VCF file {self.filename}...")

        with VariantFile(self.filename, mode="r") as vcf:
            self.vcf_header = vcf.header
            self.samples = np.array(vcf.header.samples)
            self.num_records = sum(1 for _ in vcf)
            vcf.reset()

            self._vcf_attributes_fn, self.snp_data, self.samples = (
                self.get_vcf_attributes(vcf, chunk_size=self.chunk_size)
            )

        logger.info(
            f"VCF file successfully loaded with {self.snp_data.shape[0]} loci and {self.samples.shape[0]} samples."
        )

    def get_vcf_attributes(
        self, vcf: VariantFile, chunk_size: int = 1000
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Extracts VCF attributes and returns them in an efficient manner with chunked processing."""

        h5_outfile = Path("vcf_output/gtdata/alignments/vcf/vcf_attributes.h5")
        h5_outfile.parent.mkdir(exist_ok=True, parents=True)

        snp_data = []
        sample_names = np.array(vcf.header.samples)

        with h5py.File(h5_outfile, "w") as f:
            chrom_dset = f.create_dataset("chrom", (0,), maxshape=(None,), dtype="S20")
            pos_dset = f.create_dataset("pos", (0,), maxshape=(None,), dtype=np.int32)
            ref_dset = f.create_dataset("ref", (0,), maxshape=(None,), dtype="S1")
            alt_dset = f.create_dataset("alt", (0,), maxshape=(None,), dtype="S20")

            chrom_chunk, pos_chunk, ref_chunk, alt_chunk, snp_chunk = [], [], [], [], []

            for variant in vcf.fetch():
                chrom_chunk.append(variant.chrom.encode("utf-8"))
                pos_chunk.append(variant.pos)
                ref_chunk.append(variant.ref.encode("utf-8"))
                alt_chunk.append(b",".join([a.encode("utf-8") for a in variant.alts]))

                gt_data = [variant.samples[sample]["GT"] for sample in sample_names]
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
                    ref_dset.resize((new_size,))
                    alt_dset.resize((new_size,))

                    # Write the chunk data
                    chrom_dset[current_size:new_size] = chrom_chunk
                    pos_dset[current_size:new_size] = pos_chunk
                    ref_dset[current_size:new_size] = ref_chunk
                    alt_dset[current_size:new_size] = alt_chunk

                    snp_data.extend(snp_chunk)

                    # Reset the chunk lists
                    chrom_chunk, pos_chunk, ref_chunk, alt_chunk, snp_chunk = (
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
                ref_dset.resize((new_size,))
                alt_dset.resize((new_size,))

                chrom_dset[current_size:new_size] = chrom_chunk
                pos_dset[current_size:new_size] = pos_chunk
                ref_dset[current_size:new_size] = ref_chunk
                alt_dset[current_size:new_size] = alt_chunk

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
        """Writes the GenotypeData object data to a VCF file."""
        if self.verbose:
            logger.info("Writing VCF file...")

        if hdf5_file_path is None:
            hdf5_file_path = self._vcf_attributes_fn

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            with open(output_filename, "w") as f:
                f.write(self._build_vcf_header())

                for start in range(0, len(hdf5_file["chrom"]), chunk_size):
                    chrom = hdf5_file["chrom"][start : start + chunk_size].astype(str)
                    pos = hdf5_file["pos"][start : start + chunk_size]

                    # Determine reference and alternate alleles from the data.
                    ref_alleles, alt_alleles, other_alleles = (
                        VCFReader.calculate_ref_alt_alleles(
                            self.snp_data, self.reverse_iupac_mapping
                        )
                    )

                    for i in range(len(chrom)):
                        # Construct the ALT field by combining alternate and
                        # less common alleles
                        alt_alleles_str = alt_alleles[i]
                        if other_alleles[i]:
                            alt_alleles_str = ",".join(
                                [alt_alleles[i]] + other_alleles[i]
                            )

                        row = np.array(
                            [
                                chrom[i],
                                str(pos[i]),
                                ".",
                                ref_alleles[i],  # Reference allele
                                alt_alleles_str,  # Alternate allele(s)
                                ".",
                                ".",
                                ".",
                                "GT",
                            ]
                            + self.snp_data[i, :].tolist()
                        )

                        row = self._replace_alleles(
                            row,
                            ref_alleles[i],
                            [alt_alleles[i]],
                            self._iupac_from_gt_tuples(),
                        )

                        f.write("\t".join(row) + "\n")

        if self.verbose:
            logger.info("Successfully wrote VCF file!")

    def _replace_alleles(
        self,
        row: np.ndarray,
        ref: str,
        alts: List[str],
        iupac_mapping: Dict[Tuple[str, str], str],
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
            iupac_code = row[i]

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

        return row

    @staticmethod
    def calculate_ref_alt_alleles(
        snp_data: np.ndarray,
        reverse_iupac_mapping: Dict[str, Tuple[str, str]],
        random_seed: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Optional[List[str]]]]:
        """Calculate the reference and alternate alleles from SNP data.

        This function decodes IUPAC codes into individual alleles before counting
        their frequencies to accurately determine the REF and ALT alleles.

        Args:
            snp_data (np.ndarray): A 2D array where each row is a SNP locus and each column is a sample.
            random_seed (int): Seed for the random number generator to ensure deterministic results.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[Optional[List[str]]]]:
                - Reference alleles for each locus.
                - Alternate alleles for each locus.
                - List of less common alleles for each locus.
        """
        # Set the seed for random number generation to ensure reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        ref_alleles = np.full(snp_data.shape[0], "N", dtype="<U1")
        alt_alleles = np.full(snp_data.shape[0], ".", dtype="<U1")
        other_alleles = []

        for i, locus in enumerate(snp_data):
            alleles = []
            heterozygous_counts = {}
            for genotype in locus:
                if genotype in reverse_iupac_mapping:
                    # Decode the IUPAC code into individual alleles
                    allele1, allele2 = reverse_iupac_mapping[genotype]
                    # Exclude invalid alleles
                    if allele1 not in {"N", "-", "."}:
                        alleles.append(allele1)
                    if allele2 not in {"N", "-", "."} and allele2 != allele1:
                        alleles.append(allele2)
                        # Count heterozygous genotypes
                        heterozygous_counts[allele1] = (
                            heterozygous_counts.get(allele1, 0) + 1
                        )
                        heterozygous_counts[allele2] = (
                            heterozygous_counts.get(allele2, 0) + 1
                        )
                elif genotype not in {"N", "-", "."}:
                    # If genotype is a single allele (e.g., 'G'), add it twice
                    # (homozygous)
                    alleles.append(genotype)
                    alleles.append(genotype)

            if not alleles:
                logger.error(f"No valid alleles found in locus {i}")
                raise NoValidAllelesError(i)

            # Count allele frequencies
            allele_counts = Counter(alleles)

            # Sort alleles by frequency, then by fewest heterozygous counts,
            # then alphabetically
            sorted_alleles = sorted(
                allele_counts.items(),
                key=lambda x: (-x[1], heterozygous_counts.get(x[0], 0), x[0]),
            )

            # Assign the most common and second most common alleles
            ref_alleles[i] = sorted_alleles[0][0]
            if len(sorted_alleles) > 1:
                alt_alleles[i] = sorted_alleles[1][0]

            # Collect less common alleles
            less_common = [allele for allele, _ in sorted_alleles[2:]]
            if less_common:
                other_alleles.append(less_common)
            else:
                other_alleles.append(None)

            # Warning for low allele counts
            if allele_counts[ref_alleles[i]] <= 2:
                logger.warning(
                    f"Low allele count in locus {i}: {ref_alleles[i]} occurs only {allele_counts[ref_alleles[i]]} times."
                )

        return ref_alleles, alt_alleles, other_alleles

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
