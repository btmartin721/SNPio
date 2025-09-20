from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import snpio.utils.custom_exceptions as exceptions
from snpio.utils import misc, sequence_tools
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC

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

        kwargs: Dict[str, bool] = {"verbose": self.verbose, "debug": debug}
        logman = LoggerManager(__name__, prefix=self.prefix, **kwargs)
        self.logger: Logger = logman.get_logger()

        self.iupac = IUPAC(logger=self.logger)

    def convert_012(self, snps: List[List[str]]) -> List[List[int]]:
        """Convert IUPAC/diploid genotype strings to 012 encoding (0=REF, 1=HET, 2=ALT, -9=missing).

        This encoder is robust to: Diploids written as "A/G" or "A|G" IUPAC ambiguity codes (e.g., R= A/G). Multi-ALT columns (forces biallelic by collapsing all ALTs to one ALT class). Common missing tokens ("N", "-", ".", "?", "./.", "-9"). It also logs monomorphic, non-biallelic, and all-missing columns (and writes indices).

        Args:
            snps (List[List[str]]): Genotypes as 2D list of shape (n_samples, n_sites). Elements may be single bases ("A") or diploids ("A/G", "A|G"), or IUPAC single-letter codes.

        Returns:
            List[List[int]]: 012-encoded genotypes with missing as -9.

        Notes:
            - If a column is monomorphic, everything non-missing becomes 0 (REF); heterozygous forms or non-REF tokens map to 1 only when necessary during forced decisions.
            - If a column has >2 alleles, all non-REF non-missing are collapsed to ALT (=2).
            - Membership checks are performed against an explicit set of ALT alleles (no iterator pitfalls).
        """
        # ---- Local helpers  ----
        IUPAC_TO_ALLELES = {
            "A": {"A"},
            "C": {"C"},
            "G": {"G"},
            "T": {"T"},
            "R": {"A", "G"},
            "Y": {"C", "T"},
            "S": {"G", "C"},
            "W": {"A", "T"},
            "K": {"G", "T"},
            "M": {"A", "C"},
            "N": set(),
            "-": set(),
            ".": set(),
            "?": set(),
        }

        MISSING = {"-9", "N", "-", ".", "?", "./."}

        def _split_diploid(tok: str) -> list[str]:
            t = tok.strip()
            if "/" in t:
                return t.split("/")
            if "|" in t:
                return t.split("|")
            if len(t) == 1 and t.upper() in IUPAC_TO_ALLELES:
                return list(IUPAC_TO_ALLELES[t.upper()])
            return [t]  # fallback

        def _flatten_alts(mc) -> tuple[str, set[str]]:
            # mc expected like ("A","G") or ("A", ["C","G"])
            if not mc:
                return "N", set()
            ref = str(mc[0])
            alts = set()
            for a in mc[1:]:
                if a is None:
                    continue
                if isinstance(a, (list, tuple, set)):
                    for x in a:
                        alts.add(str(x))
                else:
                    alts.add(str(a))
            return ref, alts

        def _encode_token(g: str, ref: str, alts: set[str]) -> int:
            if g in MISSING or g.strip() == "":
                return -9
            alleles = _split_diploid(g)
            if not alleles:  # from IUPAC N/ambiguous empty
                return -9

            ref_u = ref.upper()
            alts_u = {a.upper() for a in alts} if alts else set()
            a_u = [a.upper() for a in alleles]

            # Homozygous REF
            if len(a_u) == 1 and a_u[0] == ref_u:
                return 0
            if len(a_u) == 2 and a_u[0] == ref_u and a_u[1] == ref_u:
                return 0

            # Homozygous ALT (same ALT allele twice)
            if len(a_u) == 1 and a_u[0] in alts_u:
                return 2
            if (
                len(a_u) == 2
                and a_u[0] in alts_u
                and a_u[1] in alts_u
                and a_u[0] == a_u[1]
            ):
                return 2

            # Mixed ref/alt → heterozygote
            has_ref = any(a == ref_u for a in a_u)
            has_alt = any(a in alts_u for a in a_u)
            if has_ref and has_alt:
                return 1

            # Ambiguity spanning multiple bases (e.g., IUPAC) → heterozygote
            if len(set(a_u)) > 1:
                return 1

            # Unrecognized but non-REF → treat as heterozygote conservatively
            return 1

        # ---- Body ----
        n_samples = len(snps)
        n_sites = 0 if n_samples == 0 else len(snps[0])

        new_snps: list[list[int]] = [[] for _ in range(n_samples)]
        monomorphic_sites: list[int] = []
        non_biallelic_sites: list[int] = []
        all_missing: list[int] = []

        for j in range(n_sites):
            # column j
            loc = [str(snps[i][j]).upper() for i in range(n_samples)]

            # All missing?
            if all(x in {"N", "-", ".", "?", "./.", "-9"} for x in loc):
                all_missing.append(j)
                # still need to append -9 to keep rectangular shape
                for i in range(n_samples):
                    new_snps[i].append(-9)
                continue

            # Count unique biological alleles ignoring missing
            num_alleles = sequence_tools.count_alleles(loc)

            if num_alleles < 2:
                # Monomorphic (or effectively so)
                monomorphic_sites.append(j)
                mc = sequence_tools.get_major_allele(
                    loc, vcf=self.genotype_data.from_vcf
                )
                ref = "N" if not mc else str(mc[0])
                for i in range(n_samples):
                    g = loc[i]
                    if g in {"-", "-9", "N", ".", "?", "./."}:
                        new_snps[i].append(-9)
                    elif g == ref or _encode_token(g, ref, set()) == 0:
                        new_snps[i].append(0)
                    else:
                        # Any non-missing non-REF gets 1 (defensive)
                        new_snps[i].append(1)

            elif num_alleles > 2:
                # Force biallelic: collapse all non-REF to ALT
                non_biallelic_sites.append(j)
                all_alleles = sequence_tools.get_major_allele(
                    loc, vcf=self.genotype_data.from_vcf
                )
                # make a flat ordered list like [ref, alt1, alt2, ...]
                flat = []
                for a in all_alleles:
                    if isinstance(a, (list, tuple)):
                        flat.extend([str(x) for x in a])
                    else:
                        flat.append(str(a))
                if not flat:
                    ref = "N"
                    alts = set()
                else:
                    ref = flat[0]
                    alts = set(flat[1:])  # everything else is ALT class
                for i in range(n_samples):
                    g = loc[i]
                    if g in {"-", "-9", "N", ".", "?", "./."}:
                        new_snps[i].append(-9)
                    else:
                        code = _encode_token(g, ref, alts)
                        # ensure collapse: any non-REF non-missing that isn't het becomes ALT
                        if code == 1:
                            new_snps[i].append(1)
                        elif code == 0:
                            new_snps[i].append(0)
                        else:
                            new_snps[i].append(2)
            else:
                # Properly biallelic
                mc = sequence_tools.get_major_allele(
                    loc, vcf=self.genotype_data.from_vcf
                )
                ref, alts = _flatten_alts(mc)
                for i in range(n_samples):
                    new_snps[i].append(_encode_token(loc[i], ref, alts))

        # ---- Logging side-effects preserved ----
        if self.genotype_data.was_filtered:
            outdir = Path(f"{self.prefix}_output", "nremover", "logs")
        else:
            outdir = Path(f"{self.prefix}_output", "logs")
        outdir.mkdir(exist_ok=True, parents=True)

        if monomorphic_sites:
            (outdir / "monomorphic_sites_mqc.txt").write_text(
                ",".join(map(str, monomorphic_sites))
            )
            self.logger.info(
                f"Monomorphic sites detected; indices written to: {outdir/'monomorphic_sites_mqc.txt'}"
            )

        if non_biallelic_sites:
            (outdir / "non_biallelic_sites_mqc.txt").write_text(
                ",".join(map(str, non_biallelic_sites))
            )
            self.logger.info(
                f">2-allele columns collapsed to biallelic; indices written to: {outdir/'non_biallelic_sites_mqc.txt'}"
            )

        if any(idx in all_missing for idx in range(n_sites)):
            self.genotype_data.all_missing_idx = all_missing
            (outdir / "all_missing_sites_mqc.txt").write_text(
                ",".join(map(str, all_missing))
            )
            self.logger.warning(
                f"All-missing columns excluded; indices written to: {outdir/'all_missing_sites_mqc.txt'}"
            )

        return [row for row in new_snps]

    def convert_onehot(
        self,
        snp_data: np.ndarray | List[List[int]],
        encodings_dict: Dict[str, int] | None = None,
    ) -> np.ndarray:
        """Convert input data to one-hot encoded format.

        This method converts input data to one-hot encoded format.

        Args:
            snp_data (np.ndarray | List[List[int]]): Input 012-encoded data of shape (n_samples, n_SNPs).

            encodings_dict (Dict[str, int] | None): Encodings to convert structure to phylip format. Defaults to None.

        Returns:
            np.ndarray: One-hot encoded data.

        Note:
            If the data file type is "phylip" and `encodings_dict` is not provided, default encodings for nucleotides are used.

            If the data file type is "structure1row" or "structure2row" and `encodings_dict` is not provided, default encodings for alleles are used.

            Otherwise, if `encodings_dict` is provided, it will be used for conversion.

        Warning:
            If the data file type is "structure1row" or "structure2row" and `encodings_dict` is not provided, default encodings for alleles are used.

            If the data file type is "phylip" and `encodings_dict` is not provided, default encodings for nucleotides are used.

            If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.
        """

        if encodings_dict is None:
            onehot_dict = self.iupac.onehot_dict
        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.tolist()
            onehot_dict = encodings_dict
        onehot_outer_list = list()

        n_rows = len(self.samples) if encodings_dict is None else len(snp_data)

        for i in range(n_rows):
            onehot_list = list()
            for j in range(len(snp_data[0])):
                onehot_list.append(onehot_dict[snp_data[i][j]])
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

        Note:
            If the data file type is "phylip" or "vcf" and `encodings_dict` is not provided, default encodings based on IUPAC codes are used.

            If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.

            Otherwise, if `encodings_dict` is provided, it will be used for conversion.

            If the input data is a numpy array, it will be converted to a list of lists before decoding.
        """

        onehot_dict = (
            self.iupac.onehot_dict if encodings_dict is None else encodings_dict
        )

        # Create inverse dictionary (from list to key)
        inverse_onehot_dict = {tuple(v): k for k, v in onehot_dict.items()}

        if isinstance(onehot_data, np.ndarray):
            onehot_data = onehot_data.tolist()

        decoded_outer_list = []

        for i in range(len(onehot_data)):
            decoded_list = []
            for j in range(len(onehot_data[0])):
                # Look up original key using one-hot encoded list
                decoded_list.append(inverse_onehot_dict[tuple(onehot_data[i][j])])
            decoded_outer_list.append(decoded_list)

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

        Note:
            If the data file type is "phylip" or "vcf" and ``encodings_dict`` is not provided, default encodings based on IUPAC codes are used.

            If the data file type is "structure" and ``encodings_dict`` is not provided, default encodings for alleles are used.

            Otherwise, if ``encodings_dict`` is provided, it will be used for conversion.
        """

        if encodings_dict is None:
            int_iupac_dict = self.iupac.int_iupac_dict
        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.tolist()

            int_iupac_dict = encodings_dict

        outer_list = list()

        n_rows = len(self.samples) if encodings_dict is None else len(snp_data)

        for i in range(n_rows):
            int_iupac = list()
            for j in range(len(snp_data[0])):
                int_iupac.append(int_iupac_dict[snp_data[i][j]])
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

        Note:
            If the data file type is "phylip" or "vcf" and `encodings_dict` is not provided, default encodings based on IUPAC codes are used.

            If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.

            Otherwise, if `encodings_dict` is provided, it will be used for conversion
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
            decoded_list = []
            for j in range(len(int_encoded_data[0])):
                # Look up original key using integer encoding
                decoded_list.append(inverse_int_encodings_dict[int_encoded_data[i][j]])
            decoded_outer_list.append(decoded_list)

        return np.array(decoded_outer_list)

    def decode_012(
        self,
        X: np.ndarray | pd.DataFrame | List[List[int]],
        write_output: bool = True,
        is_nuc: bool = False,
    ):
        """Decode 012 or 0-9 integer encodings back to STRUCTURE/PHYLIP/IUPAC.

        For standard 012 decoding, we require per-locus REF/ALT from the backing
        ``GenotypeData``. We output "ref/ref", "ref/alt", "alt/alt" (or their
        IUPAC single-letter equivalents if PHYLIP/VCF-like output is requested).

        If ``is_nuc`` is True, treats inputs as 0-9 IUPAC integers instead
        (A=0, C=1, G=2, T=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9).

        Args:
            X (np.ndarray | pd.DataFrame | List[List[int]]): Matrix of 012 or 0-9 integers.
            write_output (bool): If True, write to disk using current filetype; else return decoded matrix.
            is_nuc (bool): Decode using 0-9 IUPAC integer scheme instead of 012 scheme.

        Returns:
            str | pd.DataFrame | np.ndarray:
                - If ``write_output=True``: output filename (str).
                - If ``write_output=False``: decoded data matrix (np.ndarray).

        Raises:
            ValueError: When REF/ALT are missing for 012 decoding.
        """
        df = misc.validate_input_type(X, return_type="df")
        ft = self.filetype.lower()

        # Map diploid pairs --> IUPAC single letter for PHYLIP-like output
        nuc = {
            "A/A": "A",
            "C/C": "C",
            "G/G": "G",
            "T/T": "T",
            "A/G": "R",
            "G/A": "R",
            "C/T": "Y",
            "T/C": "Y",
            "G/C": "S",
            "C/G": "S",
            "A/T": "W",
            "T/A": "W",
            "G/T": "K",
            "T/G": "K",
            "A/C": "M",
            "C/A": "M",
            "N/N": "N",
        }
        is_phylip_like = ft in {"phylip", "phylip-relaxed", "vcf"}

        df_decoded = df.copy().astype(object)

        if is_nuc:
            # 0–9 integer decoding (IUPAC integers)
            classes_int = list(range(10)) + [-9]
            if is_phylip_like:
                gt = ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S", "N"]
            else:
                gt = [
                    "1/1",  # A
                    "2/2",  # C
                    "3/3",  # G
                    "4/4",  # T
                    "1/4",  # A/T
                    "1/3",  # A/G
                    "1/2",  # A/C
                    "3/4",  # G/T
                    "2/4",  # C/T
                    "2/3",  # C/G
                    "-9/-9",  # N/N
                ]
            d = dict(zip(classes_int, gt)) | {
                str(k): v for k, v in zip(classes_int, gt)
            }
            dreplace = {col: d for col in df_decoded.columns}

        else:
            # Standard 012 decoding using REF/ALT
            ref_alleles = getattr(self.genotype_data, "ref", None)
            alt_alleles = getattr(self.genotype_data, "alt", None)
            if not ref_alleles or not alt_alleles:
                msg = "Reference and alternate alleles are not available in GenotypeData; cannot decode 012 matrix."
                self.logger.error(msg)
                raise ValueError(msg)

            dreplace = {}
            for i, col in enumerate(df_decoded.columns):
                ref = str(ref_alleles[i])
                alt = alt_alleles[i]
                # Some pipelines may carry None or multi-ALT; choose first ALT or fallback to REF
                if isinstance(alt, (list, tuple)) and len(alt) > 0:
                    alt = str(alt[0])
                elif alt is None:
                    alt = ref
                else:
                    alt = str(alt)

                ref2, alt2, het2 = f"{ref}/{ref}", f"{alt}/{alt}", f"{ref}/{alt}"
                if is_phylip_like:
                    ref2 = nuc.get(ref2, ref)  # if not A/C/G/T, keep token
                    alt2 = nuc.get(alt2, alt)
                    het2 = nuc.get(het2, "N")

                d = {
                    0: ref2,
                    "0": ref2,
                    1: het2,
                    "1": het2,
                    2: alt2,
                    "2": alt2,
                    -9: "N",
                    "-9": "N",
                }
                dreplace[col] = d

        df_decoded = df_decoded.replace(dreplace)
        return df_decoded.to_numpy()

    def encode_alleles_two_channel(
        self, snp_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert IUPAC genotypes to two integer allele matrices.

        This method encodes the SNP data into two separate matrices representing the two alleles for each sample and locus.

        Each matrix will have shape (N_samples, N_loci), where each entry is an integer representing one of the two alleles (reference or alternate).

        Args:
            snp_data (np.ndarray): An (n_samples x n_loci) numpy array of IUPAC-encoded genotypes, where each entry is a single character string representing the genotype (e.g., "A", "C", "G", "T", "N", "-", etc.). Heterozygous genotypes are represented by ambiguity codes (e.g., "W", "S", "M", "K", "R", "Y").

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two matrices where each row corresponds to a sample and each column to a locus. The first matrix contains the first allele (allele1) and the second matrix contains the second allele (allele2).
        """
        IUPAC_MAP = self.iupac.get_two_channel_iupac()
        n_samples, n_loci = snp_data.shape
        allele1 = np.full((n_samples, n_loci), -1, dtype=np.int8)
        allele2 = np.full((n_samples, n_loci), -1, dtype=np.int8)
        for i in range(n_samples):
            for j in range(n_loci):
                a1, a2 = IUPAC_MAP.get(snp_data[i, j].upper(), (-1, -1))
                allele1[i, j], allele2[i, j] = a1, a2
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
        g012 = self.convert_012(self.snp_data)
        g012 = misc.validate_input_type(g012, return_type="array")
        self.logger.debug(f"Genotypes 012: {g012}")
        return g012

    @genotypes_012.setter
    def genotypes_012(self, value: np.ndarray | pd.DataFrame | List[List[int]]) -> None:
        """Set the 012 genotypes. They will be decoded back to a 2D list of genotypes as ``snp_data`` object.

        012-encoded genotypes are returned as a 2D numpy array of shape (n_samples, n_sites). The encoding is as follows: 0=reference, 1=heterozygous, 2=alternate allele.

        Args:
            value (np.ndarray | pd.DataFrame | List[List[int]]): 2D numpy array with 012-encoded genotypes.
        """
        self.snp_data = self.decode_012(value, write_output=False)
        self.logger.debug(f"Decoded 012 genotypes: {self.snp_data}")

    @property
    def genotypes_onehot(self) -> np.ndarray:
        """One-hot encoded snps format of shape (n_samples, n_loci, 4).

        One-hot encoded genotypes are returned as a 3D numpy array of shape (n_samples, n_loci, 4).  The one-hot encoding is as follows: A=[1, 0, 0, 0], T=[0, 1, 0, 0], G=[0, 0, 1, 0], C=[0, 0, 0, 1]. Missing values are encoded as [0, 0, 0, 0]. The one-hot encoding is based on the IUPAC ambiguity codes. Heterozygous sites are encoded as 0.5 for each allele.

        Returns:
            numpy.ndarray: One-hot encoded numpy array of shape (n_samples, n_loci, 4).
        """
        gohe = self.convert_onehot(self.snp_data)
        gohe = misc.validate_input_type(gohe, return_type="array")

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
        X = misc.validate_input_type(value, return_type="array")
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
        gint = misc.validate_input_type(gint, return_type="array")
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
        X = misc.validate_input_type(value, return_type="array")
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
        self.snp_data = self.decode_alleles_two_channel(allele1, allele2)
        self.logger.debug(
            f"Decoded two-channel alleles: {self.snp_data} with shape {self.snp_data.shape}"
        )
