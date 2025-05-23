from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.custom_exceptions import (
    AlignmentError,
    AlignmentFileNotFoundError,
    AlignmentFormatError,
    StructureAlignmentSampleMismatch,
)
from snpio.utils.logging import LoggerManager


class StructureReader(GenotypeData):
    """Read STRUCTURE file into a GenotypeData object.

    This class reads STRUCTURE files, which can be in one-row or two-row format. In one-row format, each genotype is represented by pairs of consecutive alleles on the same line. In two-row format, each genotype is represented by two lines, with the first line containing the first allele and the second line containing the second allele (e.g., "1" and "1" on separate lines). Each sample ID and population ID (if `has_popids=True`) should be repeated for each row of alleles if the file is in two-row format.

    The first column is always the sample name, and the second column is the population ID if `has_popids=True`. If `has_marker_names=True`, the first line of the file contains the marker names, which are stored in `self.marker_names`. The `allele_start_col` parameter specifies the zero-based index where the alleles begin. The rest of the columns are genotypes, which are converted to IUPAC codes.

    The `allele_start_col` parameter specifies the zero-based index where the alleles begin. If `has_popids=True`, the second column must be the population IDs. If `has_marker_names=True`, the first line must be the marker names.

    If no popmap filename is provided and `has_popids=True`, the class will create a default population map based on the population IDs in the STRUCTURE file, saved to `{prefix}_output/gtdata/popmap.txt`.
    """

    def __init__(
        self,
        filename: str | None = None,
        popmapfile: str | None = None,
        has_popids: bool = False,
        has_marker_names: bool = False,
        allele_start_col: int | None = None,
        allele_encoding: Dict[int | str, str] | None = None,
        force_popmap: bool = False,
        exclude_pops: List[str] | None = None,
        include_pops: List[str] | None = None,
        plot_format: Literal["png", "pdf", "jpg", "svg"] = "png",
        prefix: str = "snpio",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Read STRUCTURE file into a GenotypeData object.

        This class reads STRUCTURE files, which can be in one-row or two-row format. In one-row format, each genotype is represented by pairs of consecutive alleles on the same line. In two-row format, each genotype is represented by two lines, with the first line containing the first allele and the second line containing the second allele (e.g., "1" and "1" on separate lines). Each sample ID and population ID (if `has_popids=True`) should be repeated for each row of alleles if the file is in two-row format.

        The first column is always the sample name, and the second column is the population ID if `has_popids=True`. If `has_marker_names=True`, the first line of the file contains the marker names, which are stored in `self.marker_names`. The `allele_start_col` parameter specifies the zero-based index where the alleles begin. The rest of the columns are genotypes, which are converted to IUPAC codes.

        The `allele_start_col` parameter specifies the zero-based index where the alleles begin. If `has_popids=True`, the second column must be the population IDs. If `has_marker_names=True`, the first line must be the marker names.

        If no popmap filename is provided and `has_popids=True`, the class will create a default population map based on the population IDs in the STRUCTURE file, saved to `{prefix}_output/gtdata/popmap.txt`.

        Args:
            filename (str): path to STRUCTURE file.
            popmapfile (str): path to popmap file.
            has_popids (bool): if True, file’s second column is a popID (skipped automatically).
            has_marker_names (bool): if True, first line is marker names.
            allele_start_col (int): zero-based index where alleles begin;
                if None, defaults to 1 + (1 if has_popids else 0).
            allele_encoding (dict): dictionary for allele encoding.
                e.g., {1: "A", 2: "C", 3: "G", 4: "T"}.
                If None, defaults to {1: "A", 2: "C", 3: "G", 4: "T"}. 1/1 → A, 2/2 → C, 3/3 → G, 4/4 → T. 1/2 → M, 1/3 → R, 1/4 → W, 2/3 → S, 2/4 → Y, 3/4 → K.
            force_popmap (bool): if True, force popmap even if not needed
            exclude_pops (list[str]): list of populations to exclude.
            include_pops (list[str]): list of populations to include.
            plot_format (str): format for plots (png, pdf, jpg, svg).
            prefix (str): prefix for log files.
            verbose (bool): if True, print verbose messages.
            debug (bool): if True, print debug messages.

        Raises:
            AlignmentFileNotFoundError: if the file does not exist.
            AlignmentFormatError: if the file format is incorrect.
            StructureAlignmentSampleMismatch: if the number of samples does not match the number of genotypes.

        Example:
            >>> gd = StructureReader(
            ...    filename="path/to/structure_file.txt",
            ...    popmapfile="path/to/popmap_file.txt",
            ...    has_popids=True,
            ...    has_marker_names=False,
            ...    allele_start_col=2,
            ...    force_popmap=False,
            ...    exclude_pops=["pop1"],
            ...    include_pops=["pop2"],
            ...    plot_format="png",
            ...    prefix="snpio",
            ...    verbose=True,
            ...    debug=False,
            )

            >>> print(gd.snp_data)
            [['A', 'C', 'G', 'T'],
             ['T', 'G', 'C', 'A'],
             ['C', 'A', 'T', 'G']]

            >>> print(gd.marker_names)
            ['M1', 'M2', 'M3', 'M4']

            >>> print(gd.samples)
            ['Sample1', 'Sample2', 'Sample3']

            >>> print(gd.populations)
            ['pop1', 'pop2', 'pop3']

            >>> print(gd.num_snps)
            4

            >>> print(gd.num_inds)
            3

            >>> print(gd.popmap)
            {'Sample1': 'pop1', 'Sample2': 'pop2', 'Sample3': 'pop3'}

            >>> print(gd.popmap_inverse)
            {'pop1': ['Sample1', 'Sample2'], 'pop2': ['Sample2']}
        """
        # logger setup
        kwargs = dict(prefix=prefix, verbose=verbose, debug=debug)
        self.logger = LoggerManager(name=__name__, **kwargs).get_logger()

        self.resource_data: Dict[str, Any] = {}

        self.verbose = verbose
        self.debug = debug
        self._has_popids = has_popids
        self._has_marker_names = has_marker_names
        self.allele_encoding = allele_encoding

        if self.allele_encoding is not None:
            self.allele_encoding = {str(k): v for k, v in allele_encoding.items()}

        self._validate_allele_encoding()

        # decide where alleles start
        default_start = 1 + (1 if has_popids else 0)
        self.allele_start_col = (
            allele_start_col if allele_start_col is not None else default_start
        )

        # will hold header markers or None
        self.marker_names: list[str] | None = None
        self._onerow = False

        super().__init__(
            filename=filename,
            filetype="structure",
            popmapfile=popmapfile,
            force_popmap=force_popmap,
            exclude_pops=exclude_pops,
            include_pops=include_pops,
            plot_format=plot_format,
            prefix=prefix,
            verbose=verbose,
            logger=self.logger,
            debug=debug,
        )

    def _validate_allele_encoding(self):
        if self.allele_encoding is not None:
            if not isinstance(self.allele_encoding, dict):
                msg = f"allele_encoding must be a dictionary, not {type(self.allele_encoding)}"
                self.logger.error(msg)
                raise TypeError(msg)
            if not all(isinstance(k, (int, str)) for k in self.allele_encoding.keys()):
                msg = f"allele_encoding keys must be int or str, not {type(k)}"
                self.logger.error(msg)
                raise TypeError(msg)
            if not all(isinstance(v, str) for v in self.allele_encoding.values()):
                msg = f"allele_encoding values must be str, not {type(v)}"
                self.logger.error(msg)
                raise TypeError(msg)

            allele_encoding_values = list(self.allele_encoding.values())

            if not all(a in {"A", "C", "G", "T", "N"} for a in allele_encoding_values):
                msg = (
                    "allele_encoding values must be A, C, G, T, or N, but got:"
                    + ",".join(np.unique(allele_encoding_values).tolist())
                )
                self.logger.error(msg)
                raise ValueError(msg)

    def load_aln(self) -> None:
        """Load STRUCTURE file, parse optional header, and convert genotypes.
        The STRUCTURE file format is a tab-delimited text file with the following structure:
        - First line (optional): marker names (if `has_marker_names=True`)
        - Second line (optional): population IDs (if `has_popids=True`)
        - Subsequent lines: sample names and genotypes
        - Genotypes are represented as pairs of alleles (e.g., "1/1", "1/2", "2/2")
        - In one-row format, each genotype is represented by pairs of consecutive alleles on the same line.
        - In two-row format, each genotype is represented by two lines, with the first line containing the first allele and the second line containing the second allele (e.g., "1" and "1" on separate lines).
        - Each sample ID and population ID (if `has_popids=True`) should be repeated for each row of alleles if the file is in two-row format.
        - The first column is always the sample name, and the second column is the population ID if `has_popids=True`.
        - The `allele_start_col` parameter specifies the zero-based index where the alleles begin. The rest of the columns are genotypes, which are converted to IUPAC codes.
        - The `allele_start_col` parameter specifies the zero-based index where the alleles begin. If `has_popids=True`, the second column must be the population IDs. If `has_marker_names=True`, the first line must be the marker names.

        Raises:
            AlignmentFileNotFoundError: if the file does not exist.
            AlignmentFormatError: if the file format is incorrect.
            StructureAlignmentSampleMismatch: if the number of samples does not match the number of genotypes.
        """
        if not self.filename or not Path(self.filename).is_file():
            raise AlignmentFileNotFoundError(self.filename)

        self.logger.info(f"Reading STRUCTURE file {self.filename}...")

        with open(self.filename, "r") as fin:
            lines = fin.readlines()

        # 1) strip off marker-name header if present
        if self._has_marker_names:
            hdr = lines[0].strip().split()
            self.marker_names = hdr
            data_lines = lines[1:]
        else:
            self.marker_names = None
            data_lines = lines

        # 2) detect one-row vs two-row
        first = data_lines[0].split()
        second = data_lines[1].split() if len(data_lines) > 1 else []
        self._onerow = first[0] != (second[0] if second else None)

        samples = []
        populations = []
        raw_snps = []

        try:
            if self._onerow:
                # pair every two alleles into one genotype
                for i, ln in enumerate(data_lines):
                    toks = ln.strip().split()
                    samples.append(toks[0])

                    if self._has_popids:
                        populations.append(toks[1])

                    flat = toks[self.allele_start_col :]

                    if self.allele_encoding:
                        # check for valid alleles
                        if not all(str(a) in self.allele_encoding for a in flat):
                            msg = f"Invalid allele in line {i + 1}. Expected {np.unique(list(self.allele_encoding.keys()))}, but got: {np.unique(flat)}"
                            self.logger.error(msg)
                            raise AlignmentFormatError(msg)
                    else:
                        if not all(str(a) in {"1", "2", "3", "4", "-9"} for a in flat):
                            msg = f"Invalid allele in line {i + 1}. Expected {{'1', '2', '3', '4', '-9'}}, but got: {np.unique(flat)}"
                            self.logger.error(msg)
                            raise AlignmentFormatError(msg)

                    if len(flat) % 2 != 0:
                        msg = f"Expected even number of alleles, got {len(flat)}"
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    merged = np.array(
                        [f"{flat[i]}/{flat[i+1]}" for i in range(0, len(flat), 2)]
                    )
                    raw_snps.append(merged)
            else:
                # zip loci across two lines
                for i in range(0, len(data_lines), 2):
                    a = data_lines[i].split()
                    b = data_lines[i + 1].split()

                    if len(a) != len(b):
                        msg = f"Unequal number of alleles in lines {i + 1} and {i + 2}"
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    if a[0] != b[0]:
                        msg = f"Sample mismatch: {a[0]} vs {b[0]}"
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    samples.append(a[0])

                    if self._has_popids:
                        if a[1] != b[1]:
                            msg = f"Population mismatch: {a[1]} vs {b[1]}"
                            self.logger.error(msg)
                            raise AlignmentFormatError(msg)

                        populations.append(a[1])

                    alleles1 = a[self.allele_start_col :]
                    alleles2 = b[self.allele_start_col :]

                    if self.allele_encoding:
                        if not all(str(a) in self.allele_encoding for a in alleles1):
                            msg = f"Invalid allele in line {i + 1}. Expected {np.unique(list(self.allele_encoding.keys()))}, but got: {np.unique(alleles1)}"
                            self.logger.error(msg)
                            raise AlignmentFormatError(msg)

                        if not all(str(a) in self.allele_encoding for a in alleles2):
                            msg = f"Invalid allele in line {i + 2}. Expected {np.unique(list(self.allele_encoding.keys()))}, but got: {np.unique(alleles2)}"
                            self.logger.error(msg)
                            raise AlignmentFormatError(msg)
                    else:
                        if not all(
                            str(a) in {"1", "2", "3", "4", "-9"} for a in alleles1
                        ):
                            msg = f"Invalid allele in line {i + 1}. Expected {{'1', '2', '3', '4', '-9'}}, but got: {np.unique(alleles1)}"
                            self.logger.error(msg)
                            raise AlignmentFormatError(msg)

                        if not all(
                            str(a) in {"1", "2", "3", "4", "-9"} for a in alleles2
                        ):
                            msg = f"Invalid allele in line {i + 2}. Expected {{'1', '2', '3', '4', '-9'}}, but got: {np.unique(alleles2)}"
                            self.logger.error(msg)
                            raise AlignmentFormatError(msg)

                    merged = np.array([f"{x}/{y}" for x, y in zip(alleles1, alleles2)])
                    raw_snps.append(merged)

            # dedupe samples
            self.samples = np.unique(samples).tolist()

            if self._has_popids and len(self.samples) != len(populations):
                msg = f"Mismatch between samples and populations: {len(self.samples)} vs {len(populations)}"
                self.logger.error(msg)
                raise AlignmentFormatError(msg)

            if populations and self._has_popids:
                of = Path(f"{self.prefix}_output", "gtdata")
                of.mkdir(parents=True, exist_ok=True)

                of = of / "popmap.txt"

                with open(of, "w") as fout:
                    for sample, pop in zip(self.samples, populations):
                        fout.write(f"{sample}\t{pop}\n")

                self.popmapfile = str(of)

            if len(self.samples) != len(raw_snps):
                raise StructureAlignmentSampleMismatch(len(self.samples), len(raw_snps))

            # map "n/m" → IUPAC
            self.snp_data = np.array(
                [list(map(self._genotype_to_iupac, row)) for row in raw_snps],
                dtype="<U1",
            )

            self.logger.info("STRUCTURE file successfully loaded!")
            self.logger.info(
                f"Found {self.num_snps} SNPs and {self.num_inds} individuals."
            )

        except (AlignmentError, Exception) as e:
            self.logger.error(f"Error reading STRUCTURE file: {e}")
            raise
