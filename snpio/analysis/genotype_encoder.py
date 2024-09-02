import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils import misc, sequence_tools


class GenotypeEncoder(GenotypeData):

    def __init__(
        self,
        filename: Optional[str] = None,
        popmapfile: Optional[str] = None,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        guidetree: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        siterates_iqtree: Optional[str] = None,
        plot_format: Optional[str] = "png",
        prefix="snpio",
        verbose: bool = True,
        **kwargs,
    ) -> None:

        # Initialize the parent class GenotypeData
        super().__init__(
            filename=filename,
            filetype="012",
            popmapfile=popmapfile,
            force_popmap=force_popmap,
            exclude_pops=exclude_pops,
            include_pops=include_pops,
            guidetree=guidetree,
            qmatrix_iqtree=qmatrix_iqtree,
            qmatrix=qmatrix,
            siterates=siterates,
            siterates_iqtree=siterates_iqtree,
            plot_format=plot_format,
            prefix=prefix,
            verbose=verbose,
            **kwargs,
        )

    def read_012(self) -> None:
        """
        Read 012-encoded comma-delimited file.

        Raises:
            ValueError: Sequences differ in length.
        """
        if self.verbose:
            print(f"\nReading 012-encoded file {self.filename}...")

        self._check_filetype("012")
        snp_data = list()
        num_snps = list()

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
                self._samples.append(inds)

        if len(list(set(num_snps))) > 1:
            raise ValueError(
                "All sequences must be the same length; "
                "at least one sequence differs in length from the others\n"
            )

        df = pd.DataFrame(snp_data)
        df.replace("NA", "-9", inplace=True)
        df = df.astype("int")

        # Decodes 012 and converts to self.snp_data
        self.genotypes_012 = df

        self._ref = None
        self._alt = None

        if self.verbose:
            print(f"012 file successfully loaded!")
            print(
                f"\nFound {self.num_snps} SNPs and {self.num_inds} " f"individuals...\n"
            )

    def convert_012(
        self,
        snps: List[List[str]],
    ) -> List[List[int]]:
        """
        Encode IUPAC nucleotides as 0 (reference), 1 (heterozygous), and 2 (alternate) alleles.

        Args:
            snps (List[List[str]]): 2D list of genotypes of shape (n_samples, n_sites).

        Returns:
            List[List[int]], optional: 012-encoded genotypes as a 2D list of shape (n_samples, n_sites). Only returns value if ``impute_mode`` is True.

            List[int], optional: List of integers indicating bi-allelic site indexes.

            int, optional: Number of remaining valid sites.
        """
        warnings.formatwarning = misc.format_warning

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
            # TODO: Check here if column is all missing. What to do in this
            # case? Error out?
            fname = "monomorphic_sites.txt"
            outfile = outdir / fname
            with open(outfile, "w") as fout:
                fout.write(",".join([str(x) for x in monomorphic_sites]))

            warnings.warn(
                f"\nMonomorphic sites detected. You can check the locus indices in the following log file: {outfile}\n"
            )

        if non_biallelic_sites:
            fname = "non_biallelic_sites.txt"
            outfile = Path(outdir, fname)
            with open(outfile, "w") as fout:
                fout.write(",".join([str(x) for x in non_biallelic_sites]))

            warnings.warn(
                f"\nSNP column indices listed in the log file {outfile} had >2 "
                f"alleles and was forced to "
                f"be bi-allelic. If that is not what you want, please "
                f"fix or remove the column and re-run.\n"
            )

        if all_missing:
            fname = "all_missing.txt"
            outfile = Path(outdir, fname)
            with open(outfile, "w") as fout:
                ",".join([str(x) for x in all_missing])

            warnings.warn(
                f" SNP column indices found in the log file {outfile} had all "
                f"missing data and were excluded from the alignment.\n"
            )

        snps_012 = [s for s in new_snps]
        return snps_012

    def convert_onehot(
        self,
        snp_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert input data to one-hot encoded format.

        Args:
            snp_data (Union[np.ndarray, List[List[int]]]): Input 012-encoded data of shape (n_samples, n_SNPs).

            encodings_dict (Optional[Dict[str, int]]): Encodings to convert structure to phylip format. Defaults to None.

        Returns:
            np.ndarray: One-hot encoded data.

        Note:
            If the data file type is "phylip" and `encodings_dict` is not provided, default encodings for nucleotides are used.

            If the data file type is "structure1row" or "structure2row" and `encodings_dict` is not provided, default encodings for alleles are used.

            Otherwise, if `encodings_dict` is provided, it will be used for conversion.

        Warnings:
            If the data file type is "phylip" or "structure" and ``encodings_dict`` is not provided, a default encoding will be used. It is recommended to provide custom encodings for accurate conversion.
        """

        if encodings_dict is None:
            onehot_dict = misc.get_onehot_dict()
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
        onehot_data: Union[np.ndarray, List[List[float]]],
        encodings_dict: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        Convert one-hot encoded data back to original format.
        Args:
            onehot_data (Union[np.ndarray, List[List[float]]]): Input one-hot encoded data of shape (n_samples, n_SNPs).
            encodings_dict (Optional[Dict[str, List[float]]]): Encodings to convert from one-hot encoding to original format. Defaults to None.
        Returns:
            np.ndarray: Original format data.
        """

        onehot_dict = (
            misc.get_onehot_dict() if encodings_dict is None else encodings_dict
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
        snp_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert input data to integer-encoded format (0-9) based on IUPAC codes.

        Args:
            snp_data (numpy.ndarray of shape (n_samples, n_SNPs) or List[List[int]]): Input 012-encoded data.
            encodings_dict (Dict[str, int] or None): Encodings to convert structure to phylip format.

        Returns:
            numpy.ndarray: Integer-encoded data.

        Note:
            If the data file type is "phylip" or "vcf" and `encodings_dict` is not provided, default encodings based on IUPAC codes are used.

            If the data file type is "structure" and `encodings_dict` is not provided, default encodings for alleles are used.

            Otherwise, if `encodings_dict` is provided, it will be used for conversion.
        """

        if encodings_dict is None:
            int_iupac_dict = misc.get_int_iupac_dict()
        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.tolist()

            int_iupac_dict = encodings_dict

        outer_list = list()

        n_rows = len(self._samples) if encodings_dict is None else len(snp_data)

        for i in range(n_rows):
            int_iupac = list()
            for j in range(len(snp_data[0])):
                int_iupac.append(int_iupac_dict[snp_data[i][j]])
            outer_list.append(int_iupac)

        return np.array(outer_list)

    def inverse_int_iupac(
        self,
        int_encoded_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Convert integer-encoded data back to original format.
        Args:
            int_encoded_data (numpy.ndarray of shape (n_samples, n_SNPs) or List[List[int]]): Input integer-encoded data.
            encodings_dict (Dict[str, int] or None): Encodings to convert from integer encoding to original format.
        Returns:
            numpy.ndarray: Original format data.
        """

        int_encodings_dict = (
            misc.get_int_iupac_dict() if encodings_dict is None else encodings_dict
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
        X,
        write_output=True,
        is_nuc=False,
    ):
        """
        Decode 012-encoded or 0-9 integer-encoded imputed data to STRUCTURE or PHYLIP format.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Imputed data to decode, encoded as 012 or 0-9 integers.

            write_output (bool, optional): If True, save the decoded output to a file. If False, return the decoded data as a DataFrame. Defaults to True.

            is_nuc (bool, optional): Whether the encoding is based on nucleotides instead of 012. Defaults to False.

        Returns:
            str or pandas.DataFrame: If write_output is True, returns the filename where the imputed data was written. If write_output is False, returns the decoded data as a DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)

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

        # VAE uses [A,T,G,C] encodings. The other NN methods use [0,1,2] encodings.
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
                # if site is monomorphic, set alt and ref state the same
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

        df_decoded.replace(dreplace, inplace=True)

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

                df_decoded.to_csv(
                    of,
                    sep="\t",
                    header=False,
                    index=False,
                )

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
        else:
            return df_decoded.values.tolist()

    @property
    def genotypes_012(
        self,
    ) -> Union[List[List[int]], np.ndarray, pd.DataFrame]:
        """Encoded 012 genotypes as a 2D list, numpy array, or pandas DataFrame.

        The examples below show how to return the different format types.

        Returns:
            List[List[int]], np.ndarray, or pd.DataFrame: encoded 012 genotypes.

        Examples:
            >>># Get a 2D list.
            >>>gt_list = GenotypeData.genotypes_012(fmt="list")
            >>>
            >>># Get a numpy array.
            >>>gt_array = GenotypeData.genotypes_012(fmt="numpy")
            >>>
            >>># Get a pandas DataFrame.
            >>>gt_df = GenotypeData.genotypes_012(fmt="pandas")
        """
        return self.convert_012(self.snp_data)

    @genotypes_012.setter
    def genotypes_012(self, value) -> List[List[int]]:
        """Set the 012 genotypes. They will be decoded back to a 2D list of genotypes as ``snp_data``\.

        Args:
            value (np.ndarray): 2D numpy array with 012-encoded genotypes.
        """
        self.snp_data = self.decode_012(value, write_output=False)

    @property
    def genotypes_onehot(self) -> Union[np.ndarray, List[List[List[float]]]]:
        """One-hot encoded snps format of shape (n_samples, n_loci, 4).

        Returns:
            numpy.ndarray: One-hot encoded numpy array of shape (n_samples, n_loci, 4).
        """
        return self.convert_onehot(self.snp_data)

    @genotypes_onehot.setter
    def genotypes_onehot(self, value) -> List[List[int]]:
        """Set the onehot-encoded genotypes. They will be decoded back to a 2D list of IUPAC genotypes as ``snp_data``\."""
        if isinstance(value, pd.DataFrame):
            X = value.to_numpy()
        elif isinstance(value, list):
            X = np.array(value)
        elif isinstance(value, np.ndarray):
            X = value
        else:
            raise TypeError(
                f"genotypes_onehot must be of type pd.DataFrame, np.ndarray, or list, but got {type(value)}"
            )

        Xt = self.inverse_onehot(X)
        self.snp_data = Xt

    @property
    def genotypes_int(self) -> np.ndarray:
        """Integer-encoded (0-9 including IUPAC characters) snps format.

        Returns:
            numpy.ndarray: 2D array of shape (n_samples, n_sites), integer-encoded from 0-9 with IUPAC characters.
        """
        arr = self.convert_int_iupac(self.snp_data)
        return arr

    @genotypes_int.setter
    def genotypes_int(
        self, value: Union[pd.DataFrame, np.ndarray, List[List[int]], Any]
    ) -> List[List[int]]:
        """Set the integer-encoded (0-9) genotypes. They will be decoded back to a 2D list of IUPAC genotypes as ``snp_data``\."""
        if isinstance(value, pd.DataFrame):
            X = value.to_numpy()
        elif isinstance(value, list):
            X = np.array(value)
        elif isinstance(value, np.ndarray):
            X = value
        else:
            raise TypeError(
                f"genotypes_onehot must be of type pd.DataFrame, np.ndarray, "
                f"or list, but got {type(value)}"
            )

        Xt = self.inverse_int_iupac(X)
        self.snp_data = Xt
