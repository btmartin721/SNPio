from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from snpio.utils import misc
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC, validate_input_type
from snpio.utils.output_paths import OutputPaths

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class GenotypeEncoder:
    """Encode genotypes to various formats suitable for machine learning.

    This class provides methods to encode genotypes to various formats suitable for machine learning, including 012, one-hot, and integer encodings, as well as the inverse operations.

    Example:
        >>> # Import necessary modules
        >>> from snpio import VCFReader, GenotypeEncoder
        >>>
        >>> # Initialize VCFReader and GenotypeEncoder objects
        >>> gd = VCFReader(filename="my_vcf.vcf", popmapfile="my_popmap.txt")
        >>> ge = GenotypeEncoder(gd)
        >>>
        >>> # Encode genotypes to 012, one-hot, and integer formats
        >>> gt_012 = ge.genotypes_012
        >>> gt_onehot = ge.genotypes_onehot(gt_012)
        >>> gt_int = ge.genotypes_int(gt_012)
        >>>
        >>> # Inverse operations
        >>> ge.genotypes_012 = gt_012
        >>> ge.genotypes_onehot = gt_onehot
        >>> ge.genotypes_int = gt_int

    Attributes:
        plot_format (str): Plot format for the data.
        prefix (str): Prefix for the output directory.
        verbose (bool): If True, display verbose output.
        snp_data (List[List[str]]): List of lists of SNPs.
        samples (List[str]): List of sample IDs.
        filetype (str): File type of the data.
        missing_vals (List[str]): List of missing values.
        replace_vals (List[str]): List of values to replace missing values with.
    """

    def __init__(self, genotype_data: "GenotypeData") -> None:
        """Initialize the GenotypeEncoder object.

        This class provides methods to encode genotypes to various formats suitable for machine learning, including 012, one-hot, and integer encodings, as well as the inverse operations.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object.

        Note:
            The GenotypeData object must be initialized before creating an instance of this class.
        """

        self.genotype_data = genotype_data
        self.plot_format = genotype_data.plot_format
        self.prefix = genotype_data.prefix
        self.verbose = genotype_data.verbose
        self.snp_data = genotype_data.snp_data
        self.samples = genotype_data.samples
        self.filetype = "encoded"
        debug = genotype_data.debug

        self.missing_vals: List[str] = ["N", "-", ".", "?"]
        self.replace_vals: List[str] = ["-9"] * len(self.missing_vals)

        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=debug
        )
        self.logger: Logger = logman.get_logger()

        self.iupac = IUPAC(logger=self.logger)

    _BASES_012: frozenset[str] = frozenset({"A", "C", "G", "T"})

    _IUPAC_TO_ALLELES_012: dict[str, frozenset[str]] = {
        "A": frozenset({"A"}),
        "C": frozenset({"C"}),
        "G": frozenset({"G"}),
        "T": frozenset({"T"}),
        "R": frozenset({"A", "G"}),
        "Y": frozenset({"C", "T"}),
        "S": frozenset({"G", "C"}),
        "W": frozenset({"A", "T"}),
        "K": frozenset({"G", "T"}),
        "M": frozenset({"A", "C"}),
        "B": frozenset({"C", "G", "T"}),
        "D": frozenset({"A", "G", "T"}),
        "H": frozenset({"A", "C", "T"}),
        "V": frozenset({"A", "C", "G"}),
        "N": frozenset(),
        "-": frozenset(),
        ".": frozenset(),
        "?": frozenset(),
    }

    _MISSING_SCALARS_012: frozenset[str] = frozenset(
        {"", "N", "-", ".", "?", "-9", "NA", "NAN", "NONE", "NULL", "<NA>"}
    )

    _MISSING_GENOTYPES_012: frozenset[str] = frozenset(
        {"./.", ".|.", "N/N", "N|N", "-/-", "-|-", "?/?", "?|?"}
    )

    @staticmethod
    def _normalize_012_token(value: object) -> str:
        """Normalize a scalar genotype or metadata token.

        Args:
            value (object): The genotype or metadata token to normalize, which may be a string, bytes, numpy scalar, or other type.

        Returns:
            str: The normalized token as a string, with leading/trailing whitespace removed and converted to uppercase. If the input value is None or cannot be converted to a string, an empty string is returned.
        """
        if value is None:
            return ""

        if isinstance(value, (bytes, np.bytes_)):
            return bytes(value).decode("utf-8", errors="ignore").strip().upper()

        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return ""

        try:
            return str(value).strip().upper()
        except Exception:
            return ""

    @classmethod
    def _is_missing_012_token(cls, token: object) -> bool:
        """Return True if a genotype token should be treated as missing.

        Args:
            token (object): The genotype token to check for missingness.

        Returns:
            bool: True if the token should be treated as missing, False otherwise.
        """
        tok = cls._normalize_012_token(token)

        if tok in cls._MISSING_SCALARS_012 or tok in cls._MISSING_GENOTYPES_012:
            return True

        if "/" in tok or "|" in tok:
            sep = "/" if "/" in tok else "|"
            parts = [cls._normalize_012_token(part) for part in tok.split(sep)]
            return all(part in cls._MISSING_SCALARS_012 for part in parts)

        return False

    @staticmethod
    def _unique_preserve_order_012(values: list[str]) -> list[str]:
        """Return unique values while preserving first-seen order.

        Args:
            values (list[str]): A list of strings from which to extract unique values while preserving the order in which they were first seen.

        Returns:
            list[str]: A list of unique strings from the input list, in the order they were first seen.
        """
        seen: set[str] = set()
        out: list[str] = []

        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)

        return out

    @classmethod
    def _extract_metadata_alleles_012(cls, value: object) -> list[str]:
        """Extract A/C/G/T alleles from REF/ALT metadata.

        Args:
            value (object): The metadata value to extract alleles from, which may be a string, list, tuple, set, numpy array, or other type.

        Returns:
            list[str]: A list of unique A/C/G/T alleles extracted from the metadata value, preserving the order they were first seen in the input.
        """
        if value is None:
            return []

        if isinstance(value, np.ndarray) and value.ndim == 0:
            return cls._extract_metadata_alleles_012(value.item())

        if isinstance(value, (list, tuple, set, np.ndarray)):
            out: list[str] = []
            for item in value:
                out.extend(cls._extract_metadata_alleles_012(item))
            return cls._unique_preserve_order_012(out)

        tok = cls._normalize_012_token(value)

        if tok in cls._MISSING_SCALARS_012 or tok in cls._MISSING_GENOTYPES_012:
            return []

        tokens = [part.strip() for part in tok.split(",")] if "," in tok else [tok]

        alleles: list[str] = []
        for token in tokens:
            if token in cls._MISSING_SCALARS_012:
                continue

            if token in cls._BASES_012:
                alleles.append(token)
                continue

            expanded = cls._IUPAC_TO_ALLELES_012.get(token, frozenset())
            alleles.extend(
                allele for allele in sorted(expanded) if allele in cls._BASES_012
            )

        return cls._unique_preserve_order_012(alleles)

    @staticmethod
    def _metadata_at_012(metadata: object, idx: int) -> object | None:
        """Safely retrieve locus-specific metadata by position.

        Args:
            metadata (object): The metadata object to retrieve from, which may be a dict, list, pandas Series/DataFrame, or other indexable type.
            idx (int): The index of the site (0-based) to retrieve metadata for.

        Returns:
            object | None: The metadata value corresponding to the given index, or None if it cannot be retrieved.
        """
        if metadata is None:
            return None

        if isinstance(metadata, dict):
            return metadata.get(idx)

        if isinstance(metadata, (str, bytes, np.bytes_)):
            return metadata if idx == 0 else None

        if hasattr(metadata, "iloc"):
            if not isinstance(metadata, (pd.Series, pd.DataFrame)):
                return None
            try:
                return metadata.iloc[idx]
            except (IndexError, TypeError):
                return None

        try:
            if len(metadata) <= idx:  # type: ignore[arg-type]
                return None
            return metadata[idx]  # type: ignore[index]
        except (TypeError, IndexError, KeyError):
            return None

    @classmethod
    def _alleles_from_iupac_or_base_012(cls, token: object) -> list[str] | None:
        """Convert a single nucleotide/IUPAC token to diploid allele copies

        Args:
            token (object): The genotype token to convert.

        Returns:
            list[str] | None: A list of two allele copies corresponding to the genotype token, or None if the token is missing or invalid.
        """
        tok = cls._normalize_012_token(token)

        if tok in cls._MISSING_SCALARS_012:
            return None

        allele_set = cls._IUPAC_TO_ALLELES_012.get(tok)

        if allele_set is None or len(allele_set) == 0:
            return None

        if len(allele_set) == 1:
            allele = next(iter(allele_set))
            if allele not in cls._BASES_012:
                return None
            return [allele, allele]

        if len(allele_set) == 2:
            allele_list = sorted(allele_set)
            if all(allele in cls._BASES_012 for allele in allele_list):
                return allele_list

        return None

    @classmethod
    def _allele_from_vcf_index_012(
        cls,
        token: object,
        ref_base: str,
        alt_alleles_ordered: list[str],
    ) -> str | None:
        """Convert a VCF allele-index token to its nucleotide allele.

        Args:
            token (object): The genotype token to convert.
            ref_base (str): The reference allele for the site.
            alt_alleles_ordered (list[str]): A list of alternate alleles in the order they should be indexed for VCF allele-index parsing.

        Returns:
            str | None: The nucleotide allele corresponding to the VCF allele-index token, or None if the token is missing or invalid.
        """
        tok = cls._normalize_012_token(token)

        if tok in cls._MISSING_SCALARS_012 or not tok.isdigit():
            return None

        allele_idx = int(tok)

        if allele_idx == 0:
            return ref_base

        alt_idx = allele_idx - 1
        if 0 <= alt_idx < len(alt_alleles_ordered):
            return alt_alleles_ordered[alt_idx]

        return None

    @classmethod
    def _token_to_allele_copies_012(
        cls,
        token: object,
        ref_base: str,
        alt_alleles_ordered: list[str],
        *,
        allow_vcf_indices: bool,
    ) -> list[str] | None:
        """Convert a genotype token to two allele copies.

        Args:
            token (object): The genotype token to convert.
            ref_base (str): The reference allele for the site.
            alt_alleles_ordered (list[str]): A list of alternate alleles in the order they should be indexed for VCF allele-index parsing.
            allow_vcf_indices (bool): If True, allow parsing of VCF allele-index tokens (e.g., "0", "1", "2") according to the provided REF and ALT metadata. If False, treat all tokens as literal alleles or IUPAC codes.

        Returns:
            list[str] | None: A list of two allele copies corresponding to the genotype token, or None if the token is missing or invalid.
        """
        tok = cls._normalize_012_token(token)

        if cls._is_missing_012_token(tok):
            return None

        if "/" in tok or "|" in tok:
            sep = "/" if "/" in tok else "|"
            parts = [cls._normalize_012_token(part) for part in tok.split(sep)]

            if len(parts) != 2 or any(
                part in cls._MISSING_SCALARS_012 for part in parts
            ):
                return None

            allele_copies: list[str] = []

            for part in parts:
                allele: str | None = None

                if allow_vcf_indices:
                    allele = cls._allele_from_vcf_index_012(
                        part,
                        ref_base,
                        alt_alleles_ordered,
                    )

                if allele is None:
                    parsed = cls._alleles_from_iupac_or_base_012(part)

                    if parsed is None or len(set(parsed)) != 1:
                        return None

                    allele = parsed[0]

                if allele not in cls._BASES_012:
                    return None

                allele_copies.append(allele)

            return allele_copies

        if allow_vcf_indices and tok.isdigit():
            allele = cls._allele_from_vcf_index_012(
                tok,
                ref_base,
                alt_alleles_ordered,
            )
            if allele is None:
                return None
            return [allele, allele]

        return cls._alleles_from_iupac_or_base_012(tok)

    @classmethod
    def _count_observed_bases_without_metadata_012(
        cls,
        column: np.ndarray,
    ) -> dict[str, int]:
        """Count observed A/C/G/T bases from nucleotide/IUPAC genotypes.

        Args:
            column (np.ndarray): The genotype column for the site.

        Returns:
            dict[str, int]: A dictionary mapping each base (A/C/G/T) to the count of observed copies in the column, based on the genotype tokens and without using any REF/ALT metadata.
        """
        counts = {base: 0 for base in sorted(cls._BASES_012)}

        for value in column:
            tok = cls._normalize_012_token(value)

            if cls._is_missing_012_token(tok):
                continue

            if "/" in tok or "|" in tok:
                sep = "/" if "/" in tok else "|"
                parts = [cls._normalize_012_token(part) for part in tok.split(sep)]

                if len(parts) != 2:
                    continue

                for part in parts:
                    parsed = cls._alleles_from_iupac_or_base_012(part)

                    if parsed is None or len(set(parsed)) != 1:
                        continue

                    allele = parsed[0]
                    if allele in counts:
                        counts[allele] += 1

                continue

            parsed = cls._alleles_from_iupac_or_base_012(tok)

            if parsed is None:
                continue

            unique_alleles = set(parsed)

            if len(unique_alleles) == 1:
                allele = next(iter(unique_alleles))
                if allele in counts:
                    counts[allele] += 2
            elif len(unique_alleles) == 2:
                for allele in unique_alleles:
                    if allele in counts:
                        counts[allele] += 1

        return counts

    @classmethod
    def _count_heterozygous_genotypes_for_alleles_012(
        cls,
        column: np.ndarray,
        alleles: list[str],
    ) -> dict[str, int]:
        """Count heterozygous genotypes containing each candidate allele.

        Args:
            column (np.ndarray): The genotype column for the site.
            alleles (list[str]): A list of candidate alleles to count heterozygous genotypes for.

        Returns:
            dict[str, int]: A dictionary mapping each candidate allele to the count of heterozygous genotypes containing that allele in the column.
        """
        het_counts = {allele: 0 for allele in alleles}

        for value in column:
            tok = cls._normalize_012_token(value)

            if cls._is_missing_012_token(tok):
                continue

            parsed: list[str] | None = None

            if "/" in tok or "|" in tok:
                sep = "/" if "/" in tok else "|"
                parts = [cls._normalize_012_token(part) for part in tok.split(sep)]

                if len(parts) != 2 or any(
                    part in cls._MISSING_SCALARS_012 for part in parts
                ):
                    continue

                allele_copies: list[str] = []

                for part in parts:
                    allele_parts = cls._alleles_from_iupac_or_base_012(part)

                    if allele_parts is None or len(set(allele_parts)) != 1:
                        allele_copies = []
                        break

                    allele_copies.append(allele_parts[0])

                parsed = allele_copies if len(allele_copies) == 2 else None
            else:
                parsed = cls._alleles_from_iupac_or_base_012(tok)

            if parsed is None or len(parsed) != 2 or len(set(parsed)) != 2:
                continue

            for allele in set(parsed):
                if allele in het_counts:
                    het_counts[allele] += 1

        return het_counts

    def _choose_random_ref_candidate_012(self, candidates: list[str]) -> str:
        """Randomly choose among equally valid reference allele candidates.

        Args:
            candidates (list[str]): A list of candidate reference alleles that are equally supported by the data.

        Returns:
            str: The chosen reference allele from the list of candidates.
        """
        if not candidates:
            msg = "No reference allele candidates were available for 012 encoding."
            self.logger.error(msg)
            raise ValueError(msg)

        if len(candidates) == 1:
            return candidates[0]

        rng = getattr(self, "rng", None)

        if rng is not None and hasattr(rng, "choice"):
            return str(rng.choice(candidates))

        rng = getattr(self, "_convert_012_rng", None)

        if rng is None:
            seed = getattr(self, "random_seed", None)
            try:
                rng = np.random.default_rng(seed)
            except TypeError:
                rng = np.random.default_rng()

            try:
                setattr(self, "_convert_012_rng", rng)
            except Exception:
                pass

        return str(rng.choice(np.asarray(candidates, dtype=object)))

    def _infer_ref_alt_from_column_012(
        self,
        column: np.ndarray,
    ) -> tuple[str, list[str]]:
        """Infer reference and alternate alleles for non-VCF data.

        Args:
            column (np.ndarray): The genotype column for the site.

        Returns:
            tuple[str, list[str]]: A tuple containing the reference allele (str) and a list of alternate alleles (list[str]) for the site.
        """
        counts = self._count_observed_bases_without_metadata_012(column)
        observed = [allele for allele, count in counts.items() if count > 0]

        if not observed:
            return "N", []

        max_count = max(counts[allele] for allele in observed)
        ref_candidates = [allele for allele in observed if counts[allele] == max_count]

        if len(ref_candidates) > 1:
            het_counts = self._count_heterozygous_genotypes_for_alleles_012(
                column,
                ref_candidates,
            )
            min_het_count = min(het_counts[allele] for allele in ref_candidates)
            ref_candidates = [
                allele
                for allele in ref_candidates
                if het_counts[allele] == min_het_count
            ]

        ref_base = self._choose_random_ref_candidate_012(ref_candidates)
        alt_alleles = [allele for allele in sorted(observed) if allele != ref_base]

        return ref_base, alt_alleles

    @classmethod
    def _column_has_nonref_signal_012(cls, column: np.ndarray, ref_base: str) -> bool:
        """Return True if a VCF column contains a non-reference signal.

        Args:
            column (np.ndarray): The genotype column for the site.
            ref_base (str): The reference allele for the site.

        Returns:
            bool: True if any genotype in the column contains a non-reference allele, False otherwise.
        """
        for value in column:
            tok = cls._normalize_012_token(value)

            if cls._is_missing_012_token(tok):
                continue

            parts = [tok]
            if "/" in tok or "|" in tok:
                sep = "/" if "/" in tok else "|"
                parts = [cls._normalize_012_token(part) for part in tok.split(sep)]

            for part in parts:
                if part in cls._MISSING_SCALARS_012:
                    continue

                if part.isdigit():
                    if int(part) > 0:
                        return True
                    continue

                parsed = cls._alleles_from_iupac_or_base_012(part)
                if parsed is not None and any(allele != ref_base for allele in parsed):
                    return True

        return False

    def _get_ref_alt_for_012_site(
        self,
        column: np.ndarray,
        site_idx: int,
        genotype_data: object,
        *,
        from_vcf: bool,
    ) -> tuple[str, list[str]]:
        """Return REF and ALT alleles for one site.

        Args:
            column (np.ndarray): The genotype column for the site.
            site_idx (int): The index of the site (0-based).
            genotype_data (object): The original genotype data object, which may contain REF/ALT metadata.
            from_vcf (bool): If True, use VCF-derived REF/ALT metadata. If False, infer REF/ALT from observed genotypes.

        Returns:
            tuple[str, list[str]]: A tuple containing the reference allele (str) and a list of alternate alleles (list[str]) for the site.
        """
        if not from_vcf:
            return self._infer_ref_alt_from_column_012(column)

        ref_metadata = getattr(genotype_data, "ref", None)
        alt_metadata = getattr(genotype_data, "alt", None)

        if ref_metadata is None:
            ref_metadata = getattr(self, "_ref", None)

        if alt_metadata is None:
            alt_metadata = getattr(self, "_alt", None)

        ref_value = self._metadata_at_012(ref_metadata, site_idx)
        alt_value = self._metadata_at_012(alt_metadata, site_idx)

        ref_candidates = self._extract_metadata_alleles_012(ref_value)
        alt_candidates = self._extract_metadata_alleles_012(alt_value)

        if not ref_candidates:
            msg = (
                "VCF-derived 012 encoding requires locus-specific REF metadata, "
                f"but no valid REF allele was found for site index {site_idx}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        ref_base = ref_candidates[0]
        alt_alleles = [
            allele
            for allele in self._unique_preserve_order_012(alt_candidates)
            if allele != ref_base
        ]

        if not alt_alleles and self._column_has_nonref_signal_012(column, ref_base):
            msg = (
                "VCF-derived 012 encoding found non-reference genotypes but "
                f"no valid ALT allele metadata for site index {site_idx}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return ref_base, alt_alleles

    @classmethod
    def _encode_012_token(
        cls,
        token: object,
        ref_base: str,
        alt_alleles_ordered: list[str],
        *,
        allow_vcf_indices: bool,
    ) -> int:
        """Encode one genotype token as 0/1/2/-9.

        Args:
            token (object): The genotype token to encode.
            ref_base (str): The reference allele for the site.
            alt_alleles_ordered (list[str]): A list of alternate alleles in the order they should be indexed for VCF allele-index parsing.
            allow_vcf_indices (bool): If True, allow parsing of VCF allele-index tokens (e.g., "0", "1", "2") according to the provided REF and ALT metadata. If False, treat all tokens as literal alleles or IUPAC codes.

        Returns:
            int: The encoded genotype as 0 (homozygous reference), 1 (heterozygous), 2 (homozygous alternate), or -9 (missing/invalid).
        """
        if ref_base not in cls._BASES_012:
            return -9

        allele_copies = cls._token_to_allele_copies_012(
            token,
            ref_base,
            alt_alleles_ordered,
            allow_vcf_indices=allow_vcf_indices,
        )

        if allele_copies is None or len(allele_copies) != 2:
            return -9

        alt_set = set(alt_alleles_ordered)
        allowed_alleles = {ref_base} | alt_set

        if any(allele not in allowed_alleles for allele in allele_copies):
            return -9

        return int(sum(allele != ref_base for allele in allele_copies))

    @classmethod
    def _observed_bases_from_012_column(
        cls,
        column: np.ndarray,
        ref_base: str,
        alt_alleles_ordered: list[str],
        *,
        allow_vcf_indices: bool,
    ) -> set[str]:
        """Return observed biological alleles in a column.

        Args:
            column (np.ndarray): The genotype column for the site.
            ref_base (str): The reference allele for the site.
            alt_alleles_ordered (list[str]): A list of alternate alleles in the order they should be indexed for VCF allele-index parsing.
            allow_vcf_indices (bool): If True, allow parsing of VCF allele-index tokens (e.g., "0", "1", "2") according to the provided REF and ALT metadata. If False, treat all tokens as literal alleles or IUPAC codes.

        Returns:
            set[str]: A set of observed biological alleles (A/C/G/T) in the column, based on the provided REF and ALT metadata and the genotype tokens.
        """
        observed: set[str] = set()

        for value in column:
            allele_copies = cls._token_to_allele_copies_012(
                value,
                ref_base,
                alt_alleles_ordered,
                allow_vcf_indices=allow_vcf_indices,
            )

            if allele_copies is None:
                continue

            for allele in allele_copies:
                if allele in cls._BASES_012:
                    observed.add(allele)

        return observed

    def _encode_012_site(
        self,
        column: np.ndarray,
        site_idx: int,
        genotype_data: object,
        *,
        from_vcf: bool,
    ) -> tuple[list[int], set[str], list[str]]:
        """Encode one SNP column and return site-level allele summaries.

        Args:
            column (np.ndarray): The genotype column for the site.
            site_idx (int): The index of the site (0-based).
            genotype_data (object): The original genotype data object, which may contain REF/ALT metadata.
            from_vcf (bool): If True, use VCF-derived REF/ALT metadata. If False, infer REF/ALT from observed genotypes.

        Returns:
            tuple[list[int], set[str], list[str]]: A tuple containing the encoded genotype column as a list of integers (0/1/2/-9), a set of observed biological alleles (A/C/G/T) in the column, and a list of alternate alleles in the order they were indexed for VCF allele-index parsing.
        """
        ref_base, alt_alleles_ordered = self._get_ref_alt_for_012_site(
            column,
            site_idx,
            genotype_data,
            from_vcf=from_vcf,
        )

        encoded_column = [
            self._encode_012_token(
                value,
                ref_base,
                alt_alleles_ordered,
                allow_vcf_indices=from_vcf,
            )
            for value in column
        ]

        observed_bases = self._observed_bases_from_012_column(
            column,
            ref_base,
            alt_alleles_ordered,
            allow_vcf_indices=from_vcf,
        )

        return encoded_column, observed_bases, alt_alleles_ordered

    def _validate_012_input(self, snps: list[list[str]]) -> np.ndarray:
        """Validate and coerce input SNP data to a 2D object array.

        Args:
            snps (list[list[str]]): Genotypes as a 2D list with shape (n_samples, n_sites).

        Returns:
            np.ndarray: The input SNP data coerced to a 2D numpy array of dtype object.
        """
        if snps is None:
            msg = "convert_012 expected a 2D genotype matrix, but got None."
            self.logger.error(msg)
            raise ValueError(msg)

        if isinstance(snps, (list, tuple)) and len(snps) == 0:
            return np.empty((0, 0), dtype=object)

        snp_array = np.asarray(snps, dtype=object)

        if snp_array.ndim != 2:
            msg = (
                "convert_012 expected a 2D genotype matrix with shape "
                f"(n_samples, n_sites), but got array with shape {snp_array.shape}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return snp_array

    def _get_012_report_outdir(self, genotype_data: object) -> Path:
        """Return the output directory for 012 site-classification reports.

        Args:
            genotype_data (object): The original genotype data object, which may contain a "was_filtered" attribute indicating whether the data were filtered.

        Returns:
            Path: The output directory for 012 site-classification reports.
        """
        was_filtered = bool(
            getattr(genotype_data, "was_filtered", getattr(self, "was_filtered", False))
        )
        prefix = getattr(self, "prefix", "snpio")

        return OutputPaths(prefix, filtered=was_filtered).reports(
            "genotype_encoding"
        )

    @staticmethod
    def _write_012_indices_file(path: Path, indices: list[int]) -> None:
        """Write an index report, removing stale reports when empty.

        Args:
            path (Path): The file path to write the index report to.
            indices (list[int]): A list of integer indices to write to the report. If empty, any existing report at the given path will be removed.
        """
        if indices:
            path.write_text(",".join(map(str, indices)))
        elif path.exists():
            path.unlink()

    def _write_012_site_reports(
        self,
        genotype_data: object,
        *,
        monomorphic_sites: list[int],
        non_biallelic_sites: list[int],
        all_missing_sites: list[int],
    ) -> None:
        """Write 012 conversion site reports and update all-missing metadata.

        Args:
            genotype_data (object): The original genotype data object, which may be updated with an "all_missing_idx" attribute containing the indices of all-missing sites.
            monomorphic_sites (list[int]): A list of site indices classified as monomorphic.
            non_biallelic_sites (list[int]): A list of site indices classified as non-biallelic.
            all_missing_sites (list[int]): A list of site indices classified as all-missing.
        """
        outdir = self._get_012_report_outdir(genotype_data)
        outdir.mkdir(exist_ok=True, parents=True)

        monomorphic_path = outdir / "monomorphic_sites_mqc.txt"
        non_biallelic_path = outdir / "non_biallelic_sites_mqc.txt"
        all_missing_path = outdir / "all_missing_sites_mqc.txt"

        self._write_012_indices_file(monomorphic_path, monomorphic_sites)
        self._write_012_indices_file(non_biallelic_path, non_biallelic_sites)
        self._write_012_indices_file(all_missing_path, all_missing_sites)

        if monomorphic_sites:
            self.logger.info(
                f"Monomorphic sites detected; indices written to: {monomorphic_path}"
            )

        if non_biallelic_sites:
            self.logger.info(
                ">2-allele columns detected/collapsed to ALT dosage class; "
                f"indices written to: {non_biallelic_path}"
            )

        setattr(genotype_data, "all_missing_idx", all_missing_sites)

        if all_missing_sites:
            self.logger.warning(
                "All-missing columns detected; indices written to: "
                f"{all_missing_path}"
            )

    def convert_012(self, snps: list[list[str]]) -> list[list[int]]:
        """Convert genotype strings to 012 encoding.

        Notes:
            For VCF-derived data, this method encodes genotypes relative to the original VCF REF/ALT metadata stored in ``self.genotype_data.ref`` and ``self.genotype_data.alt``:

            - 0 = homozygous reference
            - 1 = heterozygous reference/alternate
            - 2 = homozygous alternate
            - -9 = missing or unencodable genotype

            For non-VCF-derived data, where no external REF/ALT metadata exist, the reference allele is inferred from observed allele counts in each SNP column, and all other observed alleles are treated as alternate alleles. If multiple alleles are tied for highest count, the allele found in the fewest heterozygous genotypes is preferred. If a tie still remains, one tied allele is selected randomly.

            Multiallelic VCF loci are collapsed into a single ALT dosage class. Under this convention, any non-reference allele contributes to ALT dosage. For example, with REF=A and ALT=C,G:

            - A/A -> 0
            - A/C or A/G -> 1
            - C/C, G/G, or C/G -> 2

        Args:
            snps (list[list[str]]): Genotypes as a 2D list with shape
                ``(n_samples, n_sites)``. Values may be IUPAC nucleotide codes,
                diploid nucleotide strings such as ``"A/G"`` or ``"A|G"``,
                missing tokens such as ``"N"``, ``"./."``, or ``".|."``, or raw
                VCF allele-index genotypes such as ``"0/1"`` if VCF REF/ALT
                metadata are available.

        Returns:
            list[list[int]]: 012-encoded genotypes with missing or unencodable
                genotypes encoded as ``-9``.

        Raises:
            ValueError: If the input matrix is not two-dimensional, rows have
                unequal lengths, or VCF-derived data lack required REF/ALT metadata.
        """
        self.logger.info("Converting genotype strings to 012 encoding...")

        genotype_data = getattr(self, "genotype_data", self)
        from_vcf = bool(
            getattr(genotype_data, "from_vcf", getattr(self, "from_vcf", False))
        )

        snp_array = self._validate_012_input(snps)
        n_samples, n_sites = snp_array.shape

        if n_samples == 0 or n_sites == 0:
            setattr(genotype_data, "all_missing_idx", [])
            return [[] for _ in range(n_samples)]

        new_snps: list[list[int]] = [[] for _ in range(n_samples)]
        monomorphic_sites: list[int] = []
        non_biallelic_sites: list[int] = []
        all_missing_sites: list[int] = []

        for site_idx in range(n_sites):
            column = snp_array[:, site_idx]
            encoded_column, observed_bases, alt_alleles_ordered = self._encode_012_site(
                column,
                site_idx,
                genotype_data,
                from_vcf=from_vcf,
            )

            if all(code == -9 for code in encoded_column):
                all_missing_sites.append(site_idx)

            if len(observed_bases) == 1:
                monomorphic_sites.append(site_idx)

            if len(observed_bases) > 2 or len(alt_alleles_ordered) > 1:
                non_biallelic_sites.append(site_idx)

            for sample_idx, code in enumerate(encoded_column):
                new_snps[sample_idx].append(code)

        self._write_012_site_reports(
            genotype_data,
            monomorphic_sites=monomorphic_sites,
            non_biallelic_sites=non_biallelic_sites,
            all_missing_sites=all_missing_sites,
        )

        self.logger.info("Genotype conversion to 012 encoding complete.")

        return new_snps

    def convert_onehot(
        self,
        snp_data: np.ndarray | List[List[int]],
        encodings_dict: Dict[str, int] | None = None,
    ) -> np.ndarray:
        """Convert input data to one-hot encoded format.

        This method converts input data to one-hot encoded format. The one-hot encoding is determined by the provided ``encodings_dict`` or defaults to IUPAC-based encodings if ``encodings_dict`` is not provided.

        Args:
            snp_data (np.ndarray | List[List[int]]): Input 012-encoded data of shape (n_samples, n_SNPs).

            encodings_dict (Dict[str, int] | None): Encodings to convert structure to phylip format. Defaults to None.

        Returns:
            np.ndarray: One-hot encoded data.

        Note:
            - If the data file type is "phylip" and `encodings_dict` is not provided, default encodings for nucleotides are used.
            - If the data file type is "structure1row" or "structure2row" and `encodings_dict` is not provided, default encodings for alleles are used.
            - Otherwise, if `encodings_dict` is provided, it will be used for conversion.

        Warning:
            If the data file type is "structure1row" or "structure2row" and `encodings_dict` is not provided, default encodings for alleles are used.
            If the data file type is "phylip" and `encodings_dict` is not provided, default encodings for nucleotides are used.
            If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.
        """
        self.logger.info("Converting genotype data to one-hot encoding...")

        if encodings_dict is None:
            onehot_dict = self.iupac.onehot_dict
        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.astype(str).tolist()
            onehot_dict = encodings_dict
        onehot_outer_list = list()

        n_rows = len(self.samples) if encodings_dict is None else len(snp_data)

        for i in range(n_rows):
            onehot_list = list()
            for j in range(len(snp_data[0])):
                onehot_list.append(onehot_dict[str(snp_data[i][j])])
            onehot_outer_list.append(onehot_list)

        return np.array(onehot_outer_list)

    def inverse_onehot(
        self,
        onehot_data: np.ndarray | List[List[float]],
        encodings_dict: Dict[str, List[float]] | None = None,
    ) -> np.ndarray:
        """Convert one-hot encoded data back to original format.

        Args:
            onehot_data (np.ndarray | List[List[float]]): Input one-hot encoded data of shape (n_samples, n_SNPs).

            encodings_dict (Dict[str, List[float]] | None): Encodings to convert from one-hot encoding to original format. Defaults to None.

        Returns:
            np.ndarray: Original format data.

        Notes:
            - If the data file type is "phylip" or "vcf" and `encodings_dict` is not provided, default encodings based on IUPAC codes are used.
            - If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.
            - Otherwise, if `encodings_dict` is provided, it will be used for conversion.
            - If the input data is a numpy array, it will be converted to a list of lists before decoding.
        """
        onehot_dict = (
            self.iupac.onehot_dict if encodings_dict is None else encodings_dict
        )

        # Create inverse dictionary (from list to key)
        inverse_onehot_dict = {tuple(v): k for k, v in onehot_dict.items()}

        onehot_data_list: List[List[float]]

        if isinstance(onehot_data, np.ndarray):
            onehot_data_list = onehot_data.astype(float).tolist()
        else:
            onehot_data_list = onehot_data

        decoded_outer_list = []

        for i in range(len(onehot_data_list)):
            decoded_list: List[str] = []
            for j in range(len(onehot_data_list[i])):
                # Look up original key using one-hot encoded list
                element = onehot_data_list[i][j]
                if isinstance(element, (list, tuple)):
                    key = tuple(element)
                else:
                    key = tuple([element])
                decoded_list.append(inverse_onehot_dict[key])
            decoded_outer_list.append(decoded_list)

        self.logger.info("One-hot encoding to genotype string conversion complete.")
        return np.array(decoded_outer_list)

    def convert_int_iupac(
        self,
        snp_data: np.ndarray | List[List[int]],
        encodings_dict: Dict[str, int] | None = None,
    ) -> np.ndarray:
        """Convert input data to integer-encoded format (0-9) based on IUPAC codes.

        This method converts input data to integer-encoded format (0-9) based on IUPAC codes. The integer encoding is as follows: A=0, T=1, G=2, C=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9.

        Args:
            snp_data (numpy.ndarray | List[List[int]]): Input 012-encoded data of shape (n_samples, n_SNPs).

            encodings_dict (Dict[str, int] | None): Encodings to convert structure to phylip format.

        Returns:
            numpy.ndarray: Integer-encoded data.

        Notes:
            - If the data file type is "phylip" or "vcf" and ``encodings_dict`` is not provided, default encodings based on IUPAC codes are used.
            - If the data file type is "structure" and ``encodings_dict`` is not provided, default encodings for alleles are used.
            - Otherwise, if ``encodings_dict`` is provided, it will be used for conversion.
        """
        self.logger.info(
            "Converting genotype data to integer-encoded format based on IUPAC codes..."
        )

        if encodings_dict is None:
            int_iupac_dict = self.iupac.int_iupac_dict
        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.astype(str).tolist()

            int_iupac_dict = encodings_dict

        outer_list = list()

        n_rows = len(self.samples) if encodings_dict is None else len(snp_data)

        for i in range(n_rows):
            int_iupac = list()
            for j in range(len(snp_data[0])):
                int_iupac.append(int_iupac_dict[str(snp_data[i][j])])
            outer_list.append(int_iupac)

        return np.array(outer_list)

    def inverse_int_iupac(
        self,
        int_encoded_data: np.ndarray | List[List[int]],
        encodings_dict: Dict[str, int] | None = None,
    ) -> np.ndarray:
        """Convert integer-encoded data back to original format.

        This method converts integer-encoded data back to the original format based on IUPAC codes. The integer encoding is as follows: A=0, T=1, G=2, C=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9.

        Args:
            int_encoded_data (numpy.ndarray | List[List[int]]): Input integer-encoded data of shape (n_samples, n_SNPs).

            encodings_dict (Dict[str, int] | None): Encodings to convert from integer encoding to original format.

        Returns:
            numpy.ndarray: Original format data.

        Notes:
            - If the data file type is "phylip" or "vcf" and `encodings_dict` is not provided, default encodings based on IUPAC codes are used.
            - If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.
            - Otherwise, if `encodings_dict` is provided, it will be used for conversion.
        """
        int_encodings_dict = (
            self.iupac.int_iupac_dict if encodings_dict is None else encodings_dict
        )

        # Create inverse dictionary (from integer to key)
        inverse_int_encodings_dict = {v: k for k, v in int_encodings_dict.items()}

        if isinstance(int_encoded_data, np.ndarray):
            int_encoded_data = int_encoded_data.tolist()

        decoded_outer_list = []

        for i in range(len(int_encoded_data)):
            decoded_list: List[str] = []
            for j in range(len(int_encoded_data[0])):
                # Look up original key using integer encoding
                decoded_list.append(inverse_int_encodings_dict[int_encoded_data[i][j]])
            decoded_outer_list.append(decoded_list)

        self.logger.info("Integer encoding to genotype string conversion complete.")

        return np.array(decoded_outer_list)

    _BASES_TO_IUPAC_012: dict[frozenset[str], str] = {
        frozenset({"A"}): "A",
        frozenset({"C"}): "C",
        frozenset({"G"}): "G",
        frozenset({"T"}): "T",
        frozenset({"A", "G"}): "R",
        frozenset({"C", "T"}): "Y",
        frozenset({"C", "G"}): "S",
        frozenset({"A", "T"}): "W",
        frozenset({"G", "T"}): "K",
        frozenset({"A", "C"}): "M",
        frozenset({"C", "G", "T"}): "B",
        frozenset({"A", "G", "T"}): "D",
        frozenset({"A", "C", "T"}): "H",
        frozenset({"A", "C", "G"}): "V",
    }

    _DIRECT_NUC_012_CODES: np.ndarray = np.array(
        ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"],
        dtype="<U1",
    )

    @classmethod
    def _normalize_decode_iupac_012(cls, value: object) -> str | None:
        """Normalize an input value into a single IUPAC token.

        Args:
            value (object): Input scalar, bytes object, or list-like metadata value.

        Returns:
            str | None: Normalized IUPAC token, or None if missing or unresolvable.
        """
        if value is None:
            return None

        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode("utf-8", errors="ignore")

        if isinstance(value, np.ndarray) and value.ndim == 0:
            return cls._normalize_decode_iupac_012(value.item())

        if isinstance(value, pd.Series):
            value = value.to_numpy()

        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None

            for item in value:
                code = cls._normalize_decode_iupac_012(item)
                if code is not None:
                    return code

            return None

        try:
            if pd.isna(str(value)):
                return None
        except (TypeError, ValueError):
            pass

        token = str(value).upper().strip()

        if (
            not token
            or token in cls._MISSING_SCALARS_012
            or token in cls._MISSING_GENOTYPES_012
        ):
            return None

        if "," in token:
            for part in (part.strip() for part in token.split(",")):
                if (
                    part
                    and part not in cls._MISSING_SCALARS_012
                    and part in cls._IUPAC_TO_ALLELES_012
                ):
                    return part

            return None

        return token if token in cls._IUPAC_TO_ALLELES_012 else None

    @classmethod
    def _extract_decode_iupac_candidates_012(cls, value: object) -> list[str]:
        """Extract candidate IUPAC tokens from scalar or list-like metadata.

        Args:
            value (object): REF or ALT metadata value.

        Returns:
            list[str]: Candidate IUPAC tokens with missing values removed.
        """
        if value is None:
            return []

        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode("utf-8", errors="ignore")

        if isinstance(value, np.ndarray) and value.ndim == 0:
            return cls._extract_decode_iupac_candidates_012(value.item())

        if isinstance(value, pd.Series):
            value = value.to_numpy()

        if isinstance(value, (list, tuple, np.ndarray)):
            out: list[str] = []
            for item in value:
                out.extend(cls._extract_decode_iupac_candidates_012(item))
            return cls._unique_preserve_order_012(out)

        try:
            if pd.isna(value):  # type: ignore
                return []
        except (TypeError, ValueError):
            pass

        token = str(value).upper().strip()

        if (
            not token
            or token in cls._MISSING_SCALARS_012
            or token in cls._MISSING_GENOTYPES_012
        ):
            return []

        parts = [part.strip() for part in token.split(",")] if "," in token else [token]

        out: list[str] = []
        for part in parts:
            if (
                part
                and part not in cls._MISSING_SCALARS_012
                and part in cls._IUPAC_TO_ALLELES_012
            ):
                out.append(part)

        return cls._unique_preserve_order_012(out)

    @classmethod
    def _empty_base_counts_012(cls) -> dict[str, int]:
        """Return an initialized A/C/G/T count dictionary.

        Returns:
            dict[str, int]: Zero-filled base-count dictionary.
        """
        return {base: 0 for base in sorted(cls._BASES_012)}

    @classmethod
    def _choose_single_base_for_decode_012(
        cls,
        token: str | None,
        counts: dict[str, int],
    ) -> str | None:
        """Resolve an IUPAC token to one A/C/G/T base.

        Ambiguous IUPAC tokens are resolved using observed source-column base
        counts. Ties are deterministic in A/C/G/T order.

        Args:
            token (str | None): IUPAC token.
            counts (dict[str, int]): Source-column base counts.

        Returns:
            str | None: Resolved base, or None if unresolvable.
        """
        if token is None:
            return None

        bases = cls._IUPAC_TO_ALLELES_012.get(token, frozenset())

        if not bases:
            return None

        valid_bases = [base for base in bases if base in cls._BASES_012]

        if not valid_bases:
            return None

        if len(valid_bases) == 1:
            return valid_bases[0]

        order = {"A": 0, "C": 1, "G": 2, "T": 3}
        return max(valid_bases, key=lambda base: (counts.get(base, 0), -order[base]))

    @classmethod
    def _choose_alt_from_decode_candidates_012(
        cls,
        ref_base: str | None,
        alt_candidates: list[str],
        counts: dict[str, int],
    ) -> str | None:
        """Choose an ALT base from candidate metadata tokens.

        Args:
            ref_base (str | None): Resolved REF base.
            alt_candidates (list[str]): Candidate ALT IUPAC tokens.
            counts (dict[str, int]): Source-column base counts.

        Returns:
            str | None: Resolved ALT base, or None if unavailable.
        """
        base_candidates: set[str] = set()

        for token in alt_candidates:
            bases = cls._IUPAC_TO_ALLELES_012.get(token, frozenset())
            base_candidates.update(base for base in bases if base in cls._BASES_012)

        if ref_base in base_candidates:
            base_candidates.remove(ref_base)

        if not base_candidates:
            return None

        order = {"A": 0, "C": 1, "G": 2, "T": 3}
        return max(
            base_candidates, key=lambda base: (counts.get(base, 0), -order[base])
        )

    @classmethod
    def _ordered_alt_bases_for_decode_counts_012(
        cls,
        alt_candidates: list[str],
        counts: dict[str, int],
        ref_base: str | None,
    ) -> list[str]:
        """Resolve candidate ALT tokens into ordered bases for VCF-index counting.

        Args:
            alt_candidates (list[str]): Candidate ALT metadata tokens.
            counts (dict[str, int]): Source-column base counts.
            ref_base (str | None): Resolved REF base.

        Returns:
            list[str]: Ordered ALT bases, excluding REF.
        """
        out: list[str] = []

        for token in alt_candidates:
            base = cls._choose_single_base_for_decode_012(token, counts)

            if base is None or base == ref_base:
                continue

            if base not in out:
                out.append(base)

        return out

    @classmethod
    def _base_from_decode_vcf_index_012(
        cls,
        token: str,
        ref_base: str | None,
        alt_bases_ordered: list[str],
    ) -> str | None:
        """Convert a VCF allele-index token into a nucleotide base.

        Args:
            token (str): Allele-index token.
            ref_base (str | None): REF base.
            alt_bases_ordered (list[str]): Ordered ALT bases.

        Returns:
            str | None: Resolved base, or None if unresolvable.
        """
        token = token.strip()

        if not token.isdigit():
            return None

        allele_idx = int(token)

        if allele_idx == 0:
            return ref_base if ref_base in cls._BASES_012 else None

        alt_idx = allele_idx - 1
        if 0 <= alt_idx < len(alt_bases_ordered):
            alt_base = alt_bases_ordered[alt_idx]
            return alt_base if alt_base in cls._BASES_012 else None

        return None

    @classmethod
    def _base_counts_from_decode_source_column_012(
        cls,
        column: np.ndarray,
        *,
        ref_base: str | None = None,
        alt_bases_ordered: list[str] | None = None,
        max_scan: int = 5000,
    ) -> dict[str, int]:
        """Count A/C/G/T bases from a source SNP column.

        Homozygous single-base calls contribute +2. IUPAC ambiguity calls
        contribute +1 to each constituent base. Delimited diploid calls such as
        ``A/G`` or ``A|G`` contribute +1 per allele copy.

        Args:
            column (np.ndarray): Source SNP column.
            ref_base (str | None): Optional REF base for VCF-indexed calls.
            alt_bases_ordered (list[str] | None): Optional ALT bases for VCF-indexed calls.
            max_scan (int): Maximum number of non-missing rows to scan.

        Returns:
            dict[str, int]: A/C/G/T base counts.
        """
        counts = cls._empty_base_counts_012()
        alt_bases_ordered = alt_bases_ordered or []

        n_seen = 0

        for value in column:
            token = cls._normalize_012_scalar_for_decode_counts(value)

            if token is None:
                continue

            counted = False

            if "/" in token or "|" in token:
                sep = "/" if "/" in token else "|"
                parts = [part.strip().upper() for part in token.split(sep)]

                if len(parts) != 2:
                    continue

                for part in parts:
                    if part in cls._MISSING_SCALARS_012:
                        continue

                    base: str | None = None

                    if part.isdigit():
                        base = cls._base_from_decode_vcf_index_012(
                            part,
                            ref_base,
                            alt_bases_ordered,
                        )
                    else:
                        part_code = cls._normalize_decode_iupac_012(part)
                        part_bases = (
                            cls._IUPAC_TO_ALLELES_012.get(part_code, frozenset())
                            if part_code is not None
                            else frozenset()
                        )

                        if len(part_bases) == 1:
                            candidate = next(iter(part_bases))
                            base = candidate if candidate in cls._BASES_012 else None

                    if base in counts:
                        counts[base] += 1
                        counted = True

            else:
                # Do not interpret scalar numeric values as VCF allele indices here.
                # They are too easy to confuse with already-encoded 012 data.
                if token.isdigit():
                    continue

                code = cls._normalize_decode_iupac_012(token)

                if code is None:
                    continue

                bases = cls._IUPAC_TO_ALLELES_012.get(code, frozenset())

                if not bases:
                    continue

                if len(bases) == 1:
                    base = next(iter(bases))
                    if base in counts:
                        counts[base] += 2
                        counted = True
                else:
                    for base in bases:
                        if base in counts:
                            counts[base] += 1
                            counted = True

            if counted:
                n_seen += 1

            if n_seen >= max_scan:
                break

        return counts

    @classmethod
    def _normalize_012_scalar_for_decode_counts(cls, value: object) -> str | None:
        """Normalize a source-column scalar for base counting.

        Args:
            value (object): Source genotype value.

        Returns:
            str | None: Uppercase token, or None if missing.
        """
        if value is None:
            return None

        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode("utf-8", errors="ignore")

        if isinstance(value, np.ndarray) and value.ndim == 0:
            return cls._normalize_012_scalar_for_decode_counts(value.item())

        try:
            if pd.isna(value):  # type: ignore
                return None
        except (TypeError, ValueError):
            pass

        token = str(value).upper().strip()

        if (
            not token
            or token in cls._MISSING_SCALARS_012
            or token in cls._MISSING_GENOTYPES_012
        ):
            return None

        return token

    @classmethod
    def _rank_decode_bases_by_count_012(cls, counts: dict[str, int]) -> list[str]:
        """Rank bases by descending count, then A/C/G/T order.

        Args:
            counts (dict[str, int]): Base-count dictionary.

        Returns:
            list[str]: Ranked bases.
        """
        order = {"A": 0, "C": 1, "G": 2, "T": 3}
        return sorted(counts, key=lambda base: (-counts[base], order[base]))

    def _resolve_decode_ref_alt_012(
        self,
        ref_value: object,
        alt_value: object,
        source_column: np.ndarray | None,
    ) -> tuple[str, str | None]:
        """Resolve REF and ALT bases for one decoded 012 column.

        Args:
            ref_value (object): REF metadata value for the site.
            alt_value (object): ALT metadata value for the site.
            source_column (np.ndarray | None): Original SNP column, if available.

        Returns:
            tuple[str, str | None]: Resolved ``(ref_base, alt_base)``. ALT may be None if unresolvable or unavailable.
        """
        zero_counts = self._empty_base_counts_012()

        ref_token = self._normalize_decode_iupac_012(ref_value)
        alt_candidates = self._extract_decode_iupac_candidates_012(alt_value)

        preliminary_ref = self._choose_single_base_for_decode_012(
            ref_token,
            zero_counts,
        )
        preliminary_alt_bases = self._ordered_alt_bases_for_decode_counts_012(
            alt_candidates,
            zero_counts,
            preliminary_ref,
        )

        counts = zero_counts
        if source_column is not None:
            try:
                counts = self._base_counts_from_decode_source_column_012(
                    source_column,
                    ref_base=preliminary_ref,
                    alt_bases_ordered=preliminary_alt_bases,
                )
            except Exception:
                counts = self._empty_base_counts_012()

        ref_base = self._choose_single_base_for_decode_012(ref_token, counts)

        alt_base: str | None = None
        if alt_candidates:
            alt_base = self._choose_alt_from_decode_candidates_012(
                ref_base,
                alt_candidates,
                counts,
            )

            if alt_base is None and len(alt_candidates) == 1:
                alt_base = self._choose_single_base_for_decode_012(
                    alt_candidates[0],
                    counts,
                )

        if (ref_base is None or alt_base is None) and any(
            count > 0 for count in counts.values()
        ):
            ranked = self._rank_decode_bases_by_count_012(counts)

            if ref_base is None:
                ref_base = ranked[0]

            if alt_base is None:
                alt_base = next(
                    (base for base in ranked if base != ref_base and counts[base] > 0),
                    None,
                )

        if ref_base is None and alt_base is None:
            ref_base = "N"
            alt_base = "N"
        elif ref_base is None:
            ref_base = alt_base if alt_base is not None else "N"
        elif alt_base is None:
            alt_base = ref_base

        return ref_base, alt_base

    @classmethod
    def _het_iupac_from_ref_alt_012(cls, ref_base: str, alt_base: str | None) -> str:
        """Return the IUPAC heterozygote code for REF/ALT.

        Args:
            ref_base (str): REF base.
            alt_base (str | None): ALT base.

        Returns:
            str: IUPAC heterozygote code, or ``"N"`` if unrepresentable.
        """
        if ref_base == alt_base or alt_base is None:
            return ref_base

        if ref_base not in cls._BASES_012 or alt_base not in cls._BASES_012:
            return "N"

        return cls._BASES_TO_IUPAC_012.get(frozenset({ref_base, alt_base}), "N")

    @classmethod
    def _decode_012_ref_alt_column(
        cls,
        column_codes: np.ndarray,
        ref_base: str,
        alt_base: str | None,
    ) -> np.ndarray:
        """Decode one 012 column using resolved REF and ALT bases.

        Args:
            column_codes (np.ndarray): Numeric 012 column.
            ref_base (str): Resolved REF base.
            alt_base (str | None): Resolved ALT base.

        Returns:
            np.ndarray: Decoded IUPAC column.
        """
        out = np.full(column_codes.shape[0], "N", dtype="<U1")
        het_code = cls._het_iupac_from_ref_alt_012(ref_base, alt_base)

        if ref_base != "N":
            out[column_codes == 0] = ref_base

        if het_code != "N":
            out[column_codes == 1] = het_code
        elif ref_base != "N":
            out[column_codes == 1] = ref_base

        if alt_base != "N" and alt_base is not None:
            out[column_codes == 2] = alt_base
        elif ref_base != "N":
            out[column_codes == 2] = ref_base

        return out

    @classmethod
    def _decode_direct_nuc_012(cls, codes: np.ndarray) -> np.ndarray:
        """Decode direct 0..9 nucleotide/IUPAC integer codes.

        Args:
            codes (np.ndarray): Integer code matrix.

        Returns:
            np.ndarray: IUPAC string matrix.
        """
        out = np.full(codes.shape, "N", dtype="<U1")
        mask = (codes >= 0) & (codes <= 9)
        out[mask] = cls._DIRECT_NUC_012_CODES[codes[mask]]
        return out

    def _coerce_decode_012_codes(
        self,
        X: np.ndarray | pd.DataFrame | list[list[int]],
    ) -> np.ndarray:
        """Validate and coerce input data to an integer code matrix.

        Non-numeric and non-integer values are treated as missing.

        Args:
            X (np.ndarray | pd.DataFrame | list[list[int]]): Input 012 matrix.

        Returns:
            np.ndarray: Integer code matrix with missing/unusable values set to -1.

        Raises:
            ValueError: If input cannot be converted to a pandas DataFrame.
        """
        df = validate_input_type(X, return_type="df")

        if not isinstance(df, pd.DataFrame):
            msg = f"Expected a pandas.DataFrame in 'decode_012', but got: {type(df)}."
            self.logger.error(msg)
            raise ValueError(msg)

        numeric_df = df.apply(pd.to_numeric, errors="coerce")

        try:
            numeric = numeric_df.to_numpy(dtype=float, na_value=np.nan)
        except TypeError:
            numeric = numeric_df.to_numpy(dtype=float)

        finite_mask = np.isfinite(numeric)
        integer_mask = finite_mask & np.isclose(numeric, np.round(numeric))

        codes = np.full(numeric.shape, -1, dtype=np.int16)
        codes[integer_mask] = np.round(numeric[integer_mask]).astype(np.int16)

        return codes

    @classmethod
    def _metadata_usable_for_decode_012(cls, metadata: object, n_cols: int) -> bool:
        """Return True if metadata can plausibly be indexed by site.

        Args:
            metadata (object): REF or ALT metadata object.
            n_cols (int): Number of SNP columns.

        Returns:
            bool: True if metadata is usable.
        """
        if metadata is None:
            return False

        if isinstance(metadata, dict):
            return True

        if isinstance(metadata, (str, bytes, np.bytes_)):
            return n_cols == 1

        if isinstance(metadata, np.ndarray) and metadata.ndim == 0:
            return n_cols == 1

        try:
            return len(metadata) == n_cols  # type: ignore[arg-type]
        except TypeError:
            return n_cols == 1

    @staticmethod
    def _metadata_at_decode_012(metadata: object, site_idx: int) -> object | None:
        """Safely retrieve site-specific metadata.

        Args:
            metadata (object): Metadata container.
            site_idx (int): Site index.

        Returns:
            object | None: Site-specific metadata value, or None if unavailable.
        """
        if metadata is None:
            return None

        if isinstance(metadata, dict):
            return metadata.get(site_idx)

        if isinstance(metadata, (str, bytes, np.bytes_)):
            return metadata if site_idx == 0 else None

        if isinstance(metadata, np.ndarray) and metadata.ndim == 0:
            return metadata.item() if site_idx == 0 else None

        if hasattr(metadata, "iloc"):
            try:
                return metadata.iloc[site_idx]  # type: ignore[attr-defined]
            except (IndexError, TypeError):
                return None

        try:
            if len(metadata) <= site_idx:  # type: ignore[arg-type]
                return None
            return metadata[site_idx]  # type: ignore[index]
        except (TypeError, IndexError, KeyError):
            return None

    def _get_decode_012_metadata_sources(
        self,
        genotype_data: object,
        n_cols: int,
    ) -> tuple[object, object]:
        """Return REF and ALT metadata sources for decoding.

        Args:
            genotype_data (object): Genotype data object or self fallback.
            n_cols (int): Number of SNP columns.

        Returns:
            tuple[object, object]: REF and ALT metadata containers.
        """
        ref_metadata = getattr(genotype_data, "ref", None)
        alt_metadata = getattr(genotype_data, "alt", None)

        if not self._metadata_usable_for_decode_012(ref_metadata, n_cols):
            ref_metadata = getattr(self, "_ref", None)

        if not self._metadata_usable_for_decode_012(alt_metadata, n_cols):
            alt_metadata = getattr(self, "_alt", None)

        if not self._metadata_usable_for_decode_012(ref_metadata, n_cols):
            ref_metadata = [None] * n_cols

        if not self._metadata_usable_for_decode_012(alt_metadata, n_cols):
            alt_metadata = [None] * n_cols

        return ref_metadata, alt_metadata

    @staticmethod
    def _get_decode_012_source_snp_data(genotype_data: object) -> np.ndarray | None:
        """Return source SNP data as an array, if available.

        Args:
            genotype_data (object): Genotype data object.

        Returns:
            np.ndarray | None: Source SNP data array, or None.
        """
        source = getattr(genotype_data, "snp_data", None)

        if source is None:
            return None

        try:
            return np.asarray(source, dtype=object)
        except Exception:
            return None

    @staticmethod
    def _source_column_for_decode_012(
        source_snp_data: np.ndarray | None,
        site_idx: int,
    ) -> np.ndarray | None:
        """Return one source SNP column if available.

        Args:
            source_snp_data (np.ndarray | None): Source SNP matrix.
            site_idx (int): Site index.

        Returns:
            np.ndarray | None: Source SNP column, or None.
        """
        if (
            source_snp_data is None
            or source_snp_data.ndim != 2
            or source_snp_data.shape[1] <= site_idx
        ):
            return None

        return source_snp_data[:, site_idx]

    def decode_012(
        self,
        X: np.ndarray | pd.DataFrame | list[list[int]],
        is_nuc: bool = False,
    ) -> np.ndarray:
        """Decode 012 encodings to IUPAC characters with metadata repair.

        Notes:
            - ``is_nuc=True`` uses direct 0..9 to IUPAC decoding.
            - ``is_nuc=False`` uses REF/ALT-aware decoding with metadata repair.
            - Multiallelic ALT metadata are allowed, but decoding is necessarily lossy because 012 stores only one collapsed ALT dosage class.
            - If REF/ALT metadata are missing or ambiguous, alleles are inferred from observed base counts in the source SNP column when available.

        Args:
            X (np.ndarray | pd.DataFrame | list[list[int]]): Matrix of encoded
                genotypes.
            is_nuc (bool): If True, decode direct nucleotide/IUPAC integer codes
                using the 0..9 nucleotide map. If False, decode as REF/ALT-based
                012 genotypes.

        Returns:
            np.ndarray: IUPAC strings as a 2D array of shape
                ``(n_samples, n_snps)``. Missing or unresolved genotypes are
                represented as ``"N"``.

        Raises:
            ValueError: If input cannot be converted to a DataFrame.
        """
        self.logger.info("Decoding 012-encoded genotypes to IUPAC strings...")

        codes = self._coerce_decode_012_codes(X)

        if is_nuc:
            out = self._decode_direct_nuc_012(codes)
            self.logger.info("012 decoding to IUPAC string conversion complete.")
            return out

        n_rows, n_cols = codes.shape
        out = np.full((n_rows, n_cols), "N", dtype="<U1")

        genotype_data = getattr(self, "genotype_data", self)
        ref_metadata, alt_metadata = self._get_decode_012_metadata_sources(
            genotype_data,
            n_cols,
        )
        source_snp_data = self._get_decode_012_source_snp_data(genotype_data)

        for site_idx in range(n_cols):
            ref_value = self._metadata_at_decode_012(ref_metadata, site_idx)
            alt_value = self._metadata_at_decode_012(alt_metadata, site_idx)
            source_column = self._source_column_for_decode_012(
                source_snp_data,
                site_idx,
            )

            ref_base, alt_base = self._resolve_decode_ref_alt_012(
                ref_value,
                alt_value,
                source_column,
            )

            out[:, site_idx] = self._decode_012_ref_alt_column(
                codes[:, site_idx],
                ref_base,
                alt_base,
            )

        self.logger.info("012 decoding to IUPAC string conversion complete.")

        return out

    def encode_alleles_two_channel(
        self, snp_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert IUPAC genotypes to two integer allele matrices.

        This method encodes the SNP data into two separate matrices representing the two alleles for each sample and locus. Each matrix will have shape (N_samples, N_loci), where each entry is an integer representing one of the two alleles (reference or alternate).

        Args:
            snp_data (np.ndarray): An (n_samples x n_loci) numpy array of IUPAC-encoded genotypes, where each entry is a single character string representing the genotype (e.g., "A", "C", "G", "T", "N", "-", etc.). Heterozygous genotypes are represented by ambiguity codes (e.g., "W", "S", "M", "K", "R", "Y").

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two matrices where each row corresponds to a sample and each column to a locus. The first matrix contains the first allele (allele1) and the second matrix contains the second allele (allele2).
        """
        self.logger.info("Encoding IUPAC genotypes into two-channel allele matrices...")

        IUPAC_MAP = self.iupac.get_two_channel_iupac()
        n_samples, n_loci = snp_data.shape
        allele1 = np.full((n_samples, n_loci), -1, dtype=np.int8)
        allele2 = np.full((n_samples, n_loci), -1, dtype=np.int8)
        for i in range(n_samples):
            for j in range(n_loci):
                a1, a2 = IUPAC_MAP.get(snp_data[i, j].upper(), (-1, -1))
                allele1[i, j], allele2[i, j] = a1, a2
        self.logger.info("Two-channel allele encoding complete.")
        return allele1, allele2

    def decode_alleles_two_channel(
        self, allele1: np.ndarray, allele2: np.ndarray
    ) -> np.ndarray:
        """Convert two integer allele matrices back into IUPAC-encoded genotypes.

        This is the inverse of ``encode_alleles_two_channel``: given allele1 and allele2 (each shape (n_samples, n_loci), values in {0,1,2,3} or -1 for missing), reconstruct the original SNP matrix of IUPAC codes: A/C/G/T for homozygotes, ambiguity codes (W,S,M,K,R,Y) for heterozygotes, and "N" for missing.

        Args:
            allele1: An (n_samples x n_loci) int array of first alleles.
            allele2: An (n_samples x n_loci) int array of second alleles.

        Returns:
            An (n_samples x n_loci) numpy array of dtype '<U1' with IUPAC codes.
        """
        self.logger.info(
            "Decoding two-channel allele matrices back to IUPAC genotypes..."
        )

        # 1) Build inverse map from numeric pairs → IUPAC letter
        IUPAC_MAP = self.iupac.get_two_channel_iupac()  # str → (int,int)
        inv_map: dict[tuple[int, int], str] = {}
        for base, pair in IUPAC_MAP.items():
            inv_map[pair] = base
            # also allow the reversed order for heterozygotes
            if pair[0] != pair[1]:
                inv_map[(pair[1], pair[0])] = base
        # ensure missing maps to "N"
        inv_map[(-1, -1)] = "N"

        # 2) Allocate output array of single‐character strings
        n_samples, n_loci = allele1.shape
        genotypes = np.full((n_samples, n_loci), "N", dtype="<U1")

        # 3) Fill by lookup
        for i in range(n_samples):
            for j in range(n_loci):
                key = (int(allele1[i, j]), int(allele2[i, j]))
                genotypes[i, j] = inv_map.get(key, "N")

        self.logger.info("Two-channel allele decoding complete.")

        return genotypes

    @property
    def genotypes_012(self) -> np.ndarray:
        """Encode 012 genotypes as a numpy array.

        This method encodes genotypes as 0 (reference), 1 (heterozygous), and 2 (alternate) alleles. The encoded genotypes are returned as a 2D list, numpy array, or pandas DataFrame.

        Returns:
            List[List[int]], np.ndarray, or pd.DataFrame: encoded 012 genotypes.

        Example:
            >>> gd = VCFReader(filename="snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz", popmapfile="snpio/example_data/popmaps/phylogen_nomx.popmap", force_popmap=True, chunk_size=5000, verbose=False)
            >>> ge = GenotypeEncoder(gd)
            >>> gt012 = ge.genotypes_012
            >>> print(gt012)
            [["0", "1", "2"], ["0", "1", "2"], ["0", "1", "2"]]
        """
        if isinstance(self.snp_data, np.ndarray):
            snp_data_list = self.snp_data.tolist()
        elif isinstance(self.snp_data, pd.DataFrame):
            snp_data_list = self.snp_data.to_numpy().tolist()
        else:
            snp_data_list = self.snp_data

        g012 = self.convert_012(snp_data_list)

        if not isinstance(g012, np.ndarray):
            try:
                g012 = np.array(g012, dtype=int)
            except Exception as e:
                msg = f"Failed to convert genotypes to numpy array of integers: {e}"
                self.logger.error(msg)
                raise TypeError(msg)

        self.logger.debug(f"Genotypes 012: {g012}")
        return g012

    @genotypes_012.setter
    def genotypes_012(self, value: np.ndarray | pd.DataFrame | List[List[int]]) -> None:
        """Set the 012 genotypes. They will be decoded back to a 2D list of genotypes as ``snp_data`` object.

        012-encoded genotypes are returned as a 2D numpy array of shape (n_samples, n_sites). The encoding is as follows: 0=reference, 1=heterozygous, 2=alternate allele.

        Args:
            value (np.ndarray | pd.DataFrame | List[List[int]]): 2D numpy array with 012-encoded genotypes.
        """
        self.snp_data = self.decode_012(value, is_nuc=False)
        self.logger.debug(f"Decoded 012 genotypes: {self.snp_data}")

    @property
    def genotypes_onehot(self) -> np.ndarray:
        """One-hot encoded snps format of shape (n_samples, n_loci, 4).

        One-hot encoded genotypes are returned as a 3D numpy array of shape (n_samples, n_loci, 4).  The one-hot encoding is as follows: A=[1, 0, 0, 0], T=[0, 1, 0, 0], G=[0, 0, 1, 0], C=[0, 0, 0, 1]. Missing values are encoded as [0, 0, 0, 0]. The one-hot encoding is based on the IUPAC ambiguity codes. Heterozygous sites are encoded as 0.5 for each allele.

        Returns:
            numpy.ndarray: One-hot encoded numpy array of shape (n_samples, n_loci, 4).
        """
        if isinstance(self.snp_data, np.ndarray):
            snp_data_list = self.snp_data.tolist()
        elif isinstance(self.snp_data, pd.DataFrame):
            snp_data_list = self.snp_data.to_numpy().tolist()
        else:
            snp_data_list = self.snp_data

        gohe = self.convert_onehot(snp_data_list)
        if not isinstance(gohe, np.ndarray):
            try:
                gohe = np.array(gohe, dtype=float)
            except Exception as e:
                msg = f"Failed to convert genotypes to numpy array of floats: {e}"
                self.logger.error(msg)
                raise TypeError(msg)
        self.logger.debug(f"Genotypes one-hot encoded: {gohe}")
        return gohe

    @genotypes_onehot.setter
    def genotypes_onehot(
        self, value: np.ndarray | List[List[List[int]]] | pd.DataFrame
    ) -> None:
        """Set the onehot-encoded genotypes. They will be decoded back to a 2D list of IUPAC genotypes as ``snp_data``.

        One-hot encoded genotypes are returned as a 3D numpy array of shape (n_samples, n_loci, 4).  The one-hot encoding is as follows: A=[1, 0, 0, 0], T=[0, 1, 0, 0], G=[0, 0, 1, 0], C=[0, 0, 0, 1]. Missing values are encoded as [0, 0, 0, 0]. The one-hot encoding is based on the IUPAC ambiguity codes. Heterozygous sites are encoded as 0.5 for each allele.

        Args:
            value (np.ndarray | List[List[List[int]]] | pd.DataFrame): 3D numpy array with one-hot encoded genotypes.

        Raises:
            TypeError: If `value` is not of type pd.DataFrame, np.ndarray, or list.
        """
        if isinstance(value, pd.DataFrame):
            X = value.to_numpy()
        elif isinstance(value, np.ndarray):
            X = value.astype(float)
        elif isinstance(value, list):
            X = np.array(value, dtype=float)
        else:
            msg = f"Expected input type of pd.DataFrame, np.ndarray, or list for 'genotypes_onehot' setter, but got: {type(value)}"
            self.logger.error(msg)
            raise TypeError(msg)
        Xt = self.inverse_onehot(X)
        self.snp_data = Xt
        self.logger.debug(f"Decoded one-hot genotypes: {Xt}")

    @property
    def genotypes_int(self) -> np.ndarray:
        """Integer-encoded (0-9 including IUPAC characters) snps format.

        Integer-encoded genotypes are returned as a 2D numpy array of shape (n_samples, n_sites). The integer encoding is as follows: A=0, T=1, G=2, C=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9. Missing values are encoded as -9.

        Returns:
            numpy.ndarray: 2D array of shape (n_samples, n_sites), integer-encoded from 0-9 with IUPAC characters.
        """
        gint = self.convert_int_iupac(self.snp_data)
        if not isinstance(gint, np.ndarray):
            try:
                gint = np.array(gint, dtype=int)
            except Exception as e:
                msg = f"Failed to convert genotypes to numpy array of integers: {e}"
                self.logger.error(msg)
                raise TypeError(msg)

        self.logger.debug(f"Genotypes integer-encoded: {gint}")
        return gint

    @genotypes_int.setter
    def genotypes_int(
        self, value: pd.DataFrame | np.ndarray | List[List[int]] | Any
    ) -> None:
        """Set the integer-encoded (0-9) genotypes. They will be decoded back to a 2D list of IUPAC genotypes as a ``snp_data`` object.

        Integer-encoded genotypes are returned as a 2D numpy array of shape (n_samples, n_sites). The integer encoding is as follows: A=0, T=1, G=2, C=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9. Missing values are encoded as -9.

        Args:
            value (pd.DataFrame | np.ndarray | List[List[int]] | Any): 2D numpy array with integer-encoded genotypes.
        """
        if isinstance(value, pd.DataFrame):
            X = value.to_numpy()
        elif isinstance(value, np.ndarray):
            X = value.astype(int)
        elif isinstance(value, list):
            X = np.array(value, dtype=int)
        else:
            msg = f"Expected input type of pd.DataFrame, np.ndarray, or list for 'genotypes_int' setter, but got: {type(value)}"
            self.logger.error(msg)
            raise TypeError(msg)

        Xt = self.inverse_int_iupac(X)
        self.snp_data = Xt
        self.logger.debug(f"Decoded integer-encoded genotypes: {Xt}")

    @property
    def two_channel_alleles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Two-channel allele matrices.

        This property encodes the SNP data into two separate matrices representing the two alleles for each sample and locus. Each matrix will have shape (N_samples, N_loci), where each entry is an integer representing one of the two alleles (reference or alternate).

        Warning:
            This method forces the SNP data to be bi-allelic. Genotypes represented by IUPAC ambiguity codes representing more than two alleles will be set to missing values (-1).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two matrices where each row corresponds to a sample and each column to a locus. The first matrix contains the first allele (allele1) and the second matrix contains the second allele (allele2).
        """
        alleles = self.encode_alleles_two_channel(self.snp_data)
        self.logger.debug(f"Alleles (first channel): {alleles[0]}")
        self.logger.debug(f"Alleles (second channel): {alleles[1]}")
        return alleles

    @two_channel_alleles.setter
    def two_channel_alleles(self, value: Tuple[np.ndarray, np.ndarray]) -> None:
        """Set the two-channel allele matrices.

        This method decodes the two-channel allele matrices back to a 2D list of IUPAC genotypes as ``snp_data``.

        Args:
            value (Tuple[np.ndarray, np.ndarray]): Two matrices where each row corresponds to a sample and each column to a locus. The first matrix contains the first allele (allele1) and the second matrix contains the second allele (allele2).
        """
        if not isinstance(value, tuple):
            msg = f"Value must be a tuple of two numpy arrays, but got: {type(value)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if len(value) != 2:
            msg = f"Value must be a tuple of two numpy arrays, but got: {len(value)} elements"
            self.logger.error(msg)
            raise ValueError(msg)

        allele1 = misc.validate_input_type(value[0], return_type="array")
        allele2 = misc.validate_input_type(value[1], return_type="array")

        if not isinstance(allele1, np.ndarray) or not isinstance(allele2, np.ndarray):
            msg = f"Both elements of the tuple must be numpy arrays, but got: {type(allele1)} and {type(allele2)}"
            self.logger.error(msg)
            raise TypeError(msg)

        self.snp_data = self.decode_alleles_two_channel(allele1, allele2)
        self.logger.debug(
            f"Decoded two-channel alleles: {self.snp_data} with shape {self.snp_data.shape}"
        )
