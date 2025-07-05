from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.custom_exceptions import (
    AlignmentFileNotFoundError,
    AlignmentFormatError,
    StructureAlignmentSampleMismatch,
)
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


class StructureReader(GenotypeData):
    """Read STRUCTURE file into a GenotypeData object.

    This class reads STRUCTURE files, which can be in one-row or two-row format. In one-row format, each genotype is represented by pairs of consecutive alleles on the same line. In two-row format, each genotype is represented by two lines, with the first line containing the first allele and the second line containing the second allele (e.g., "1" and "1" on separate lines). Each sample ID and population ID (if `has_popids=True`) should be repeated for each row of alleles if the file is in two-row format.

    The first column is always the sample name, and the second column is the population ID if `has_popids=True`. If `has_marker_names=True`, the first line of the file contains the marker names, which are stored in `self.marker_names`. The `allele_start_col` parameter specifies the zero-based index where the alleles begin. The rest of the columns are genotypes, which are converted to IUPAC codes.

    The `allele_start_col` parameter specifies the zero-based index where the alleles begin. If `has_popids=True`, the second column must be the population IDs. If `has_marker_names=True`, the first line must be the marker names.

    If no popmap filename is provided and `has_popids=True`, the class will create a default population map based on the population IDs in the STRUCTURE file, saved to `{prefix}_output/alignments/popmap.txt` or `{prefix}_output/nremover/alignments/popmap.txt`.
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

        If no popmap filename is provided and `has_popids=True`, the class will create a default population map based on the population IDs in the STRUCTURE file, saved to `{prefix}_output/alignments/popmap.txt` or `{prefix}_output/nremover/alignments/popmap.txt`.

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
        self.iupac = IUPAC(logger=self.logger)

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

    def _validate_allele_encoding(self) -> None:
        """Validate the allele_encoding dictionary.

        Ensures:
            - allele_encoding is a dictionary (if not None).
            - Keys are int or str.
            - Values are str and valid IUPAC bases: A, C, G, T, N.

        Raises:
            TypeError: If allele_encoding is not a dict or has invalid key/value types.
            ValueError: If values are not valid IUPAC nucleotides.
        """
        if self.allele_encoding is None:
            return

        if not isinstance(self.allele_encoding, dict):
            msg = f"allele_encoding must be a dictionary, not {type(self.allele_encoding).__name__}"
            self.logger.error(msg)
            raise TypeError(msg)

        keys = self.allele_encoding.keys()
        values = self.allele_encoding.values()

        invalid_keys = [k for k in keys if not isinstance(k, (int, str))]
        if invalid_keys:
            msg = f"allele_encoding keys must be int or str, but got: {[type(k).__name__ for k in invalid_keys]}"
            self.logger.error(msg)
            raise TypeError(msg)

        invalid_values = [v for v in values if not isinstance(v, str)]
        if invalid_values:
            msg = f"allele_encoding values must be str, but got: {[type(v).__name__ for v in invalid_values]}"
            self.logger.error(msg)
            raise TypeError(msg)

        valid_nucleotides = {"A", "C", "G", "T", "N"}
        unique_values = set(values)
        non_iupac = unique_values - valid_nucleotides
        if non_iupac:
            msg = (
                "allele_encoding values must be one of A, C, G, T, N. "
                f"Invalid values found: {sorted(non_iupac)}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

    def _validate_alleles(
        self, flat: List[str], line_number: int, check_ploidy: bool
    ) -> None:
        """Validate allele codes for a single STRUCTURE line.

        Ensures:
            - All alleles are valid according to the allele encoding.
            - The number of alleles is even (for ploidy).

        Args:
            flat (List[str]): Flattened list of allele values as strings.
            line_number (int): Line number in the file (1-indexed). Used for printing error reports.
            check_ploidy (bool): If True, checks that the number of alleles is even.

        Raises:
            AlignmentFormatError: If any allele is invalid or the number of alleles is odd.
        """
        flat_str = [str(a) for a in flat]
        unique_alleles = set(flat_str)

        # Validate allele encodings
        if self.allele_encoding:
            valid_alleles = set(map(str, self.allele_encoding.keys()))
            invalid_alleles = unique_alleles - valid_alleles
            if invalid_alleles:
                msg = (
                    f"Invalid allele(s) in line {line_number}. "
                    f"Expected {sorted(valid_alleles)}, but got: "
                    f"{sorted(invalid_alleles)}"
                )
                self.logger.error(msg)
                raise AlignmentFormatError(msg)
        else:
            default_valid = {"1", "2", "3", "4", "-9"}
            invalid_alleles = unique_alleles - default_valid
            if invalid_alleles:
                msg = (
                    f"Invalid allele(s) in line {line_number}. "
                    f"Expected {sorted(default_valid)}, but got: {sorted(invalid_alleles)}"
                )
                self.logger.error(msg)
                raise AlignmentFormatError(msg)

        # Check ploidy (expect even number of alleles)
        if check_ploidy and len(flat_str) % 2 != 0:
            msg = f"Expected even number of alleles in line {line_number}, got {len(flat_str)}"
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

    def load_aln(self) -> None:
        """Efficiently load a STRUCTURE file with optional header and population IDs.

        Reads STRUCTURE files in one-row or two-row format, validating alleles and converting them to IUPAC codes.

        Uses lazy streaming and vectorized NumPy operations to optimize performance and memory usage.

        Raises:
            AlignmentFileNotFoundError: If the STRUCTURE file does not exist.
            AlignmentFormatError: If the STRUCTURE file format is incorrect.
            StructureAlignmentSampleMismatch: If the number of samples does not match the number of genotypes.
        """
        path = Path(self.filename)
        if not path.is_file():
            raise AlignmentFileNotFoundError(self.filename)

        self.logger.info(f"Reading STRUCTURE file {self.filename}...")

        samples = []
        populations = []
        raw_snps = []

        # Read file lazily
        with open(path, "r") as fin:
            lines = iter(fin)

            # Marker name header (optional)
            if self._has_marker_names:
                self.marker_names = next(lines).strip().split()
            else:
                self.marker_names = None

            # Peek at first two data lines to determine format
            peek_lines = deque()
            while len(peek_lines) < 2:
                line = next(lines).strip()
                if line:
                    peek_lines.append(line)

            first = peek_lines[0].split()
            second = peek_lines[1].split() if len(peek_lines) > 1 else []
            self._onerow = first[0] != (second[0] if second else None)

            # Reinsert peeked lines back into the iterator
            data_lines = iter(peek_lines + deque(lines))

            if self._onerow:
                for i, line in enumerate(data_lines):
                    toks = line.strip().split()
                    if not toks:
                        continue

                    samples.append(toks[0])
                    if self._has_popids:
                        populations.append(toks[1])

                    flat = toks[self.allele_start_col :]
                    self._validate_alleles(flat, line_number=i + 1, check_ploidy=True)

                    flat_arr = np.asarray(flat, dtype=str)
                    merged = np.char.add(flat_arr[::2], "/")
                    merged = np.char.add(merged, flat_arr[1::2])
                    raw_snps.append(merged)

            else:
                line_buffer = []
                for i, line in enumerate(data_lines):
                    line_buffer.append(line.strip())
                    if len(line_buffer) < 2:
                        continue  # wait for the second line

                    a = line_buffer[0].split()
                    b = line_buffer[1].split()
                    line_buffer.clear()

                    if len(a) != len(b):
                        msg = f"Unequal number of alleles in lines {i} and {i+1}"
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    if a[0] != b[0]:
                        msg = (
                            f"Sample mismatch in lines {i} and {i+1}: {a[0]} vs {b[0]}"
                        )
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

                    self._validate_alleles(alleles1, line_number=i, check_ploidy=False)
                    self._validate_alleles(
                        alleles2, line_number=i + 1, check_ploidy=False
                    )

                    arr1 = np.asarray(alleles1, dtype=str)
                    arr2 = np.asarray(alleles2, dtype=str)

                    merged = np.char.add(arr1, "/")
                    merged = np.char.add(merged, arr2)
                    raw_snps.append(merged)

        # Validate sample count and deduplicate
        self.samples = np.unique(samples).tolist()

        if self._has_popids:
            if len(self.samples) != len(populations):
                msg = f"Mismatch between samples and populations: {len(self.samples)} vs {len(populations)}"
                self.logger.error(msg)
                raise AlignmentFormatError(msg)

            # Save popmap

            if self.was_filtered:
                out_path = Path(
                    f"{self.prefix}_output", "nremover", "alignments", "popmap.txt"
                )
            else:
                # Default path for popmap
                # If not filtered, save to alignments directory
                # If filtered, save to nremover/alignments directory
                out_path = Path(f"{self.prefix}_output", "alignments", "popmap.txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fout:
                for sample, pop in zip(self.samples, populations):
                    fout.write(f"{sample}\t{pop}\n")
            self.popmapfile = str(out_path)

        if len(self.samples) != len(raw_snps):
            raise StructureAlignmentSampleMismatch(len(self.samples), len(raw_snps))

        # Convert to IUPAC codes with a vectorized or list-based approach
        self.snp_data = np.array(
            [list(map(self._genotype_to_iupac, row)) for row in raw_snps], dtype="<U1"
        )

        self.logger.info("STRUCTURE file successfully loaded!")
        self.logger.info(f"Found {self.num_snps} SNPs and {self.num_inds} individuals.")
