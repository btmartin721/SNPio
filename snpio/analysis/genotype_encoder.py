from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import snpio.utils.custom_exceptions as exceptions
from snpio.utils import misc, sequence_tools
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


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

    def __init__(self, genotype_data: Any) -> None:
        """Initialize the GenotypeEncoder object.

        This class provides methods to encode genotypes to various formats suitable for machine learning, including 012, one-hot, and integer encodings, as well as the inverse operations.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object.

        Note:
            The GenotypeData object must be initialized before creating an instance of this class.
        """

        self.plot_format = genotype_data.plot_format
        self.prefix = genotype_data.prefix
        self.verbose = genotype_data.verbose
        self.snp_data = genotype_data.snp_data
        self.samples = genotype_data.samples
        debug = genotype_data.debug
        self.filetype = "encoded"

        self.missing_vals: List[str] = ["N", "-", ".", "?"]
        self.replace_vals: List[str] = ["-9"] * len(self.missing_vals)

        kwargs: Dict[str, bool] = {"verbose": self.verbose, "debug": debug}
        logman = LoggerManager(__name__, prefix=self.prefix, **kwargs)
        self.logger: Logger = logman.get_logger()

        self.iupac = IUPAC(logger=self.logger)

    def read_012(self) -> None:
        """Read 012-encoded comma-delimited file.

        This method reads a 012-encoded comma-delimited file and stores the data in the GenotypeEncoder object.

        Raises:
            ValueError: Sequences differ in length.
        """
        self.logger.info(f"Reading 012-encoded file: {self.filename}...")

        self._check_filetype("encoded")
        snp_data, num_snps = [], []

        with open(self.filename, "r") as fin:
            num_inds = 0
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                cols = line.split(",")
                inds = cols[0]
                snps = cols[1:]
                num_snps.append(len(snps))
                num_inds += 1
                snp_data.append(snps)
                self.samples.append(inds)

        if len(list(set(num_snps))) > 1:
            msg = "All sequences must be the same length. At least one sequence differs in length from the others"
            self.logger.error(msg)
            raise exceptions.SequenceLengthError(msg)

        miss_vals = ["NA", np.nan, pd.NA, "N", ".", "-", "?"]
        df = pd.DataFrame(snp_data)
        df = df.replace(miss_vals, ["-9"] * len(miss_vals)).astype(int)

        # Decodes 012 and converts to self.snp_data
        self.genotypes_012 = df

        self.ref = None
        self.alt = None

        self.logger.info("012 file successfully loaded!")
        self.logger.info(
            f"\nFound:\n"
            f"\t{self.num_inds} individuals...\n"
            f"\t{self.num_snps} SNPs...\n"
        )

    def convert_012(self, snps: List[List[str]]) -> List[List[int]]:
        """Encode IUPAC nucleotides as 0 (reference), 1 (heterozygous), and 2 (alternate) alleles.

        This method encodes IUPAC nucleotides as 0 (reference), 1 (heterozygous), and 2 (alternate) alleles.

        Args:
            snps (List[List[str]]): 2D list of genotypes of shape (n_samples, n_sites).

        Returns:
            List[List[int]]: Encoded 012 genotypes.

        Warning:
            Monomorphic sites are detected and encoded as 0 (reference).

            Non-biallelic sites are detected and forced to be bi-allelic.

            Sites with all missing data are detected and excluded from the alignment.
        """
        snps_012 = []
        new_snps = []
        monomorphic_sites = []
        non_biallelic_sites = []
        all_missing = []

        for i in range(0, len(snps)):
            new_snps.append([])

        for j in range(0, len(snps[0])):
            loc = []
            for i in range(0, len(snps)):
                loc.append(snps[i][j].upper())

            if all(x == "N" for x in loc):
                all_missing.append(j)
                continue
            num_alleles = sequence_tools.count_alleles(loc)
            if num_alleles != 2:
                # If monomorphic
                if num_alleles < 2:
                    monomorphic_sites.append(j)
                    try:
                        ref = list(
                            map(
                                sequence_tools.get_major_allele,
                                loc,
                                [False for x in loc],
                            )
                        )
                        ref = str(ref[0])
                    except IndexError:
                        ref = list(
                            map(
                                sequence_tools.get_major_allele,
                                loc,
                                [False for x in loc],
                            )
                        )
                    alt = None

                    for i in range(0, len(snps)):
                        if loc[i] in ["-", "-9", "N"]:
                            new_snps[i].append(-9)

                        elif loc[i] == ref:
                            new_snps[i].append(0)

                        else:
                            new_snps[i].append(1)

                # If >2 alleles
                elif num_alleles > 2:
                    non_biallelic_sites.append(j)
                    all_alleles = sequence_tools.get_major_allele(loc)
                    all_alleles = [str(x[0]) for x in all_alleles]
                    ref = all_alleles.pop(0)
                    alt = all_alleles.pop(0)
                    others = all_alleles

                    for i in range(0, len(snps)):
                        if loc[i] in ["-", "-9", "N"]:
                            new_snps[i].append(-9)

                        elif loc[i] == ref:
                            new_snps[i].append(0)

                        elif loc[i] == alt:
                            new_snps[i].append(2)

                        # Force biallelic
                        elif loc[i] in others:
                            new_snps[i].append(2)

                        else:
                            new_snps[i].append(1)

            else:
                ref, alt = sequence_tools.get_major_allele(loc)
                ref = str(ref)
                alt = str(alt)

                for i in range(0, len(snps)):
                    if loc[i] in ["-", "-9", "N"]:
                        new_snps[i].append(-9)

                    elif loc[i] == ref:
                        new_snps[i].append(0)

                    elif loc[i] == alt:
                        new_snps[i].append(2)

                    else:
                        new_snps[i].append(1)

        outdir = Path(f"{self.prefix}_output", "gtdata", "logs")
        outdir.mkdir(exist_ok=True, parents=True)
        if monomorphic_sites:
            # TODO: Check here if column is all missing.
            # TODO: What to do in this case? Error out?
            outfile = outdir / "monomorphic_sites.txt"
            with open(outfile, "w") as fout:
                mono_sites = [str(x) for x in monomorphic_sites]
                fout.write(",".join(mono_sites))

            self.logger.warning(
                f"Monomorphic sites detected. You can check the locus indices in the following log file: {outfile}"
            )

        if non_biallelic_sites:
            outfile = outdir / "non_biallelic_sites.txt"
            with open(outfile, "w") as fout:
                nba = [str(x) for x in non_biallelic_sites]
                fout.write(",".join(nba))

            self.logger.warning(
                f"SNP column indices listed in the log file {outfile} had >2 alleles and was forced to be bi-allelic. If that is not desired, please fix or remove the column and re-run."
            )

        if all_missing:
            outfile = outdir / "all_missing_sites.txt"
            with open(outfile, "w") as fout:
                ",".join([str(x) for x in all_missing])

            self.logger.warning(
                f"SNP column indices found in the log file {outfile} had all missing data and were excluded from the alignment."
            )

        snps_012 = [s for s in new_snps]
        return snps_012

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
        """Decode 012-encoded or 0-9 integer-encoded imputed data to STRUCTURE or PHYLIP format.

        This method decodes 012-encoded or 0-9 integer-encoded imputed data to IUPAC format. The decoded data can be saved to a file or returned as a DataFrame.

        Args:
            X (pandas.DataFrame | numpy.ndarray | List[List[int]]): Imputed data to decode, encoded as 012 or 0-9 integers.

            write_output (bool): If True, save the decoded output to a file. If False, return the decoded data as a DataFrame. Defaults to True.

            is_nuc (bool): Whether the encoding is based on nucleotides instead of 012. Defaults to False.

        Returns:
            str | pandas.DataFrame: If write_output is True, returns the filename where the imputed data was written. If write_output is False, returns the decoded data as a DataFrame.
        """
        df = misc.validate_input_type(X, return_type="df")

        nuc = {
            "A/A": "A",
            "T/T": "T",
            "G/G": "G",
            "C/C": "C",
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

        ft = self.filetype.lower()

        is_phylip = False
        if ft == "phylip" or ft == "vcf":
            is_phylip = True

        df_decoded = df.copy()
        df_decoded = df.copy().astype(object)

        if is_nuc:
            classes_int = range(10)
            classes_string = [str(x) for x in classes_int]
            if is_phylip:
                gt = ["A", "T", "G", "C", "W", "R", "M", "K", "Y", "S", "N"]
            else:
                gt = [
                    "1/1",
                    "2/2",
                    "3/3",
                    "4/4",
                    "1/2",
                    "1/3",
                    "1/4",
                    "2/3",
                    "2/4",
                    "3/4",
                    "-9/-9",
                ]
            d = dict(zip(classes_int, gt))
            dstr = dict(zip(classes_string, gt))
            d.update(dstr)
            dreplace = {col: d for col in list(df.columns)}

        else:
            dreplace = dict()
            for col, ref, alt in zip(df.columns, self._ref, self._alt):
                # if site is monomorphic, set alt and ref state the
                # same
                if alt is None:
                    alt = ref
                ref2 = f"{ref}/{ref}"
                alt2 = f"{alt}/{alt}"
                het2 = f"{ref}/{alt}"

                if is_phylip:
                    ref2 = nuc[ref2]
                    alt2 = nuc[alt2]
                    het2 = nuc[het2]

                d = {
                    "0": ref2,
                    0: ref2,
                    "1": het2,
                    1: het2,
                    "2": alt2,
                    2: alt2,
                    "-9": "N",
                    -9: "N",
                }
                dreplace[col] = d

        df_decoded = df_decoded.replace(dreplace)

        if write_output:
            outfile = Path(f"{self.prefix}_output", "gtdata", "alignments", "012")

        if ft.startswith("structure"):
            if ft.startswith("structure2row"):
                for col in df_decoded.columns:
                    df_decoded[col] = (
                        df_decoded[col]
                        .str.split("/")
                        .apply(lambda x: list(map(int, x)))
                    )

                df_decoded.insert(0, "sampleID", self._samples)
                df_decoded.insert(1, "popID", self._populations)

                # Transform each element to a separate row.
                df_decoded = (
                    df_decoded.set_index(["sampleID", "popID"])
                    .apply(pd.Series.explode)
                    .reset_index()
                )

            elif ft.startswith("structure1row"):
                df_decoded = pd.concat(
                    [
                        df_decoded[c]
                        .astype(str)
                        .str.split("/", expand=True)
                        .add_prefix(f"{c}_")
                        for c in df_decoded.columns
                    ],
                    axis=1,
                )

            elif ft == "structure":
                for col in df_decoded.columns:
                    df_decoded[col] = (
                        df_decoded[col]
                        .str.split("/")
                        .apply(lambda x: list(map(int, x)))
                    )

                df_decoded.insert(0, "sampleID", self._samples)
                df_decoded.insert(1, "popID", self._populations)

                # Transform each element to a separate row.
                df_decoded = (
                    df_decoded.set_index(["sampleID", "popID"])
                    .apply(pd.Series.explode)
                    .reset_index()
                )

            if write_output:
                of = outfile.with_suffix(".str")
                df_decoded.insert(0, "sampleID", self._samples)
                df_decoded.insert(1, "popID", self._populations)

                df_decoded.to_csv(of, sep="\t", header=False, index=False)

        elif ft.startswith("phylip"):
            if write_output:
                of = outfile.with_suffix(".phy")
                header = f"{self.num_inds} {self.num_snps}\n"
                with open(of, "w") as fout:
                    fout.write(header)

                lst_decoded = df_decoded.values.tolist()

                with open(of, "a") as fout:
                    for sample, row in zip(self._samples, lst_decoded):
                        seqs = "".join([str(x) for x in row])
                        fout.write(f"{sample}\t{seqs}\n")

        if write_output:
            return of
        return df_decoded.to_numpy()

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
