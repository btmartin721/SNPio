import logging
import random
import re
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pysam

from snpio.utils.custom_exceptions import NoValidAllelesError
from snpio.utils.logging import LoggerManager

# IUPAC decoding dictionary
IUPAC_TO_BASES = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "M": ["A", "C"],
    "R": ["A", "G"],
    "W": ["A", "T"],
    "S": ["C", "G"],
    "Y": ["C", "T"],
    "K": ["G", "T"],
    "V": ["A", "C", "G"],
    "H": ["A", "C", "T"],
    "D": ["A", "G", "T"],
    "B": ["C", "G", "T"],
    "N": [],
    "-": [],
    "?": [],
    ".": [],
}


class BaseGenotypeData:
    def __init__(self, filename: str | None = None, filetype: str | None = "auto"):
        """Base class for handling genotype data.

        This class provides methods for loading, processing, and encoding genotype data from various file formats, including VCF, STRUCTURE, and PHYLIP files. It also includes methods for handling IUPAC-encoded genotypes and generating VCF headers.

        Args:
            filename (str, optional): Path to the input file. Defaults to None.
            filetype (str, optional): Type of the input file. Defaults to "auto".
                - "auto": Automatically detect file type.
                - "vcf": VCF file.
                - "vcf.gz": gzipped VCF file.
        """
        self.filename: str | None = filename

        self.filetype: str | None = filetype if filetype is not None else None

        if filetype is not None:
            self.filetype = filetype.lower()

        # Initialize as None, load on demand
        self._snp_data: List[List[str]] | None = None
        self._samples: List[str] = []
        self._populations: List[str | int] = []
        self._ref: List[str] = []
        self._alt: List[List[str] | str] = []

        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=self.debug
        )
        self.logger: logging.Logger | None = logman.get_logger()

    def _load_data(self) -> None:
        """Method to load data from file based on filetype"""
        raise NotImplementedError("Subclasses should implement this method")

    def get_ref_alt_alleles(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Determine ref/alt alleles for each locus (column) of IUPAC-encoded genotypes.

        Args:
            data: shape (n_samples, n_variants), dtype=object of single-char IUPAC codes.

        Notes:
            - The most common allele is chosen as the reference (REF).
            - The second most common allele is chosen as the alternate (ALT).
            - If there are more than two alleles, any further alleles are stored in a list.
            - If there are ties in counts, the one with the fewest heterozygotes is chosen.
            - If still tied, one is chosen at random.

        Returns:
            Tuple of
                - ref_alleles: shape (n_variants,) most‐common allele per locus
                - alt_alleles: shape (n_variants,) second‐most‐common allele or None
                - extra_alts: list of arrays of any further alleles per locus
        """
        # Handle edge cases for loci with no valid alleles or only one allele
        if data.size == 0:
            return np.array([]), np.array([]), []

        ref_alleles = []
        alt_alleles = []
        extra_alts = []

        for locus in data.T:
            unique, counts = np.unique(locus, return_counts=True)
            sorted_alleles = sorted(zip(unique, counts), key=lambda x: (-x[1], x[0]))

            ref_alleles.append(sorted_alleles[0][0])
            alt_alleles.append(
                sorted_alleles[1][0] if len(sorted_alleles) > 1 else None
            )
            extra_alts.append([allele for allele, _ in sorted_alleles[2:]])

        return np.array(ref_alleles), np.array(alt_alleles), extra_alts

    def refs_alts_from_snp_data(
        self, snp_matrix: np.ndarray
    ) -> Tuple[list[str], List[List[str]]]:
        """Determine REF/ALT per locus from a (samples x loci) IUPAC matrix.

        This method processes the SNP matrix to extract reference and alternate alleles for each locus.

        Args:
            snp_matrix (np.ndarray): (n_samples x n_loci) matrix of IUPAC codes

        Returns:
            tuple: (ref_alleles, alt_lists)
                - ref_alleles: List of REF alleles for each locus
                - alt_lists: List of ALT alleles for each locus
        """
        n_samples, n_loci = snp_matrix.shape
        ref_alleles = []
        alt_lists = []

        for j in range(n_loci):
            col = snp_matrix[:, j]
            # Expand IUPAC codes into pairs
            allele_pairs = [self.reverse_iupac_mapping.get(code, ()) for code in col]
            flat = [
                b
                for pair in allele_pairs
                for b in pair
                if b not in {"N", ".", "-", "?"}
            ]
            counts = Counter(flat)

            if not counts:
                ref_alleles.append("N")
                alt_lists.append([])
                continue

            # Choose reference as the most common allele
            common = counts.most_common()
            common.sort(key=lambda x: x[1], reverse=True)
            top1 = common[0][0]
            ref = top1

            # Include *all other* alleles as ALT
            alt = [a for a in counts if a != ref]

            ref_alleles.append(ref)
            alt_lists.append(sorted(alt))  # sort for determinism

        return ref_alleles, alt_lists

    def encode_to_vcf_format(
        self,
        snp_data: np.ndarray,
        ref_alleles: List[str],
        alt_alleles: List[List[str]],
    ) -> np.ndarray:
        """Vectorized encoding of IUPAC codes into VCF GT strings.

        Args:
            snp_data (np.ndarray): (n_variants x n_samples) matrix of IUPAC codes
            ref_alleles (List[str]): List of REF alleles for each locus
            alt_alleles (List[List[str]]): List of ALT allele(s) per locus

        Returns:
            np.ndarray: (n_variants x n_samples) VCF GT strings
        """
        n_vars, n_samp = snp_data.shape
        out = np.empty((n_vars, n_samp), dtype=object)

        for v in range(n_vars):
            ref = ref_alleles[v]
            alts = alt_alleles[v] or []
            alleles = [ref] + alts

            # Sanity check: if no valid REF (e.g., "."), treat as fully missing
            if ref in {".", "N", ""} or all(a in {".", "N", ""} for a in alleles):
                out[v, :] = ["./."] * n_samp
                continue

            lookup = {}
            for code, bases in IUPAC_TO_BASES.items():
                if len(bases) == 1 and bases[0] in alleles:
                    idx = alleles.index(bases[0])
                    lookup[code] = f"{idx}/{idx}"
                elif len(bases) == 2 and all(b in alleles for b in bases):
                    i1, i2 = sorted(alleles.index(b) for b in bases)
                    lookup[code] = f"{i1}/{i2}"
                else:
                    lookup[code] = "./."

            row = snp_data[v, :]
            out[v, :] = [lookup.get(c, "./.") for c in row]

        return out

    def build_vcf_header(self) -> str:
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
            ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_headers}\n"""
        )

        return vcf_header

    def replace_alleles(self, row: List[str], ref: str, alts: List[str]) -> List[str]:
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
            elif not alts:
                return "0/0" if allele1 == allele2 and allele1 != "N" else "./."
            elif (allele1 == ref or allele2 == ref) and alts[0] == ".":
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
                # If multiple alleles are present in the mapping, take the
                # first two
                if len(alleles) >= 2:
                    allele1, allele2 = alleles[:2]
                else:
                    if alleles[0] not in {"N", "-", ".", "?"}:
                        allele1 = allele2 = alleles[0]
                    else:
                        allele1 = allele2 = "N"
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
                fmt_data = ":".join(fmt_data) if len(fmt_data) > 1 else fmt_data[0]
                row[i] = f"{row[i]}:{fmt_data}"

        return row

    def bgzip_file(self, filepath: Path) -> Path:
        """BGZips a VCF file using pysam's BGZFile, avoiding overwriting the input file.

        Args:
            filepath (Path): The path to the input VCF file (uncompressed).

        Returns:
            Path: The path to the bgzipped output VCF file with a .vcf.gz suffix.
        """
        if not filepath.exists():
            msg = f"Input VCF file not found: {filepath}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Skip if already has correct suffix
        if filepath.name.endswith(".nremover.vcf.gz"):
            self.logger.warning(
                f"File already appears bgzipped with expected name: {filepath}"
            )
            return filepath

        name_parts = filepath.name.split(".")
        while name_parts and name_parts[-1] in {"vcf", "gz", "nremover"}:
            name_parts.pop()
        base = ".".join(name_parts)
        bgzipped_path = f"{base}.nremover.vcf.gz"
        bgzipped_path = Path(bgzipped_path)

        # Safety check: don't overwrite input file
        if bgzipped_path.resolve() == filepath.resolve():
            msg = f"Output path {bgzipped_path} would overwrite input file!"
            self.logger.error(msg)
            raise ValueError(msg)

        try:
            with (
                open(filepath, "rb") as f_in,
                pysam.BGZFile(str(bgzipped_path), "wb") as f_out,
            ):
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)

            return bgzipped_path

        except Exception as e:
            self.logger.error(f"Error bgzipping file {filepath}: {e}")
            raise e

    def _is_sorted(self, filepath: Path) -> bool:
        """Checks if the VCF file is sorted alphanumerically by chromosome and position.

        Args:
            filepath (Path): The path to the VCF file to check.

        Returns:
            bool: True if the file is sorted, False otherwise.
        """
        try:
            with pysam.VariantFile(filepath, mode="r") as vcf:
                previous_key = None
                for record in vcf:
                    current_key = self._natural_sort_key((record.chrom, record.pos))
                    if previous_key is not None and current_key < previous_key:
                        return False
                    previous_key = current_key
            return True
        except Exception as e:
            self.logger.error(f"Error checking sort order: {e}")
            return False

    @staticmethod
    def _natural_sort_key(chrom_pos: Tuple[str, int]) -> tuple:
        """Extracts a natural sort key for sorting.

        Args:
            chrom_pos (Tuple[str, int]): A tuple containing the chromosome and position.

        Returns:
            tuple: A tuple that can be used for natural sorting.
        """
        chrom, pos = chrom_pos
        return [
            int(s) if s.isdigit() else s for s in re.split(r"([0-9]+)", chrom)
        ], int(pos)

    def _sort_vcf_file(self, filepath: Path) -> Path:
        """Sorts a VCF file alphanumerically using custom logic, then indexes it.

        Args:
            filepath (Path): The path to the VCF file to sort.

        Returns:
            Path: The path to the sorted VCF file.
        """
        sorted_path = filepath.with_name(filepath.stem + "_sorted.vcf.gz")

        try:
            header_lines = []
            data_lines = []
            with pysam.VariantFile(filepath, "r") as vcf_in:
                header_lines.extend(str(vcf_in.header).splitlines())
                for record in vcf_in:
                    data_lines.append(record)

            # Sort data lines by CHROM and POS
            sorted_data_lines = sorted(
                data_lines,
                key=lambda r: self._natural_sort_key((r.contig, r.pos)),
            )

            # Write to temporary unsorted VCF file
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".vcf"
            ) as temp_vcf:
                for line in header_lines:
                    temp_vcf.write(line + "\n")
                for record in sorted_data_lines:
                    temp_vcf.write(str(record))  # record already ends in newline
            temp_vcf_path = Path(temp_vcf.name)

            # Compress and index
            pysam.tabix_compress(str(temp_vcf_path), str(sorted_path), force=True)
            pysam.tabix_index(str(sorted_path), preset="vcf", force=True)

            temp_vcf_path.unlink()  # Clean up temp file
            return sorted_path

        except Exception as e:
            self.logger.error(f"Error sorting VCF file: {e}")
            raise e

    def _has_tabix_index(self, filepath: Path) -> bool:
        """Checks if a Tabix index exists for the given VCF file.

        This method checks for the existence of a .tbi index file corresponding to the provided VCF file. The index file is typically created by the Tabix tool and is used for fast random access to the VCF file.

        Args:
            filepath (Path): The path to the bgzipped VCF file.

        Returns:
            bool: True if index exists, False otherwise.
        """
        index_path = filepath.with_suffix(filepath.suffix + ".tbi")
        return index_path.exists()

    def tabix_index(self, filepath: Path) -> None:
        """Creates a Tabix index for a bgzipped VCF file.

        Args:
            filepath (Path): The path to the bgzipped VCF file.
        """
        try:
            pysam.tabix_index(str(filepath), preset="vcf", force=True)
        except Exception as e:
            self.logger.error(f"Error indexing VCF file {filepath}: {e}")
            raise e
