from pathlib import Path
from typing import List, Literal

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
    """This class reads STRUCTURE files and stores the SNP data, sample IDs, and populations.

    It is a subclass of GenotypeData and inherits its attributes and methods. The STRUCTURE file can be written to a new file using the write_structure method. SNP data can be accessed in the IUPAC format using the `snp_data` attribute. The class also provides methods for filtering, PCA, and plotting. The class is initialized with the filename of the STRUCTURE file, the population map file, and other optional parameters.


    STRUCTURE files are used in population genetics to represent genetic data for individuals across multiple loci. The format is commonly used in STRUCTURE software for inferring population structure and admixture.

    The STRUCTURE file format consists of a header line followed by lines of data. Each line represents an individual and contains the individual's name, population ID (if present), and the genotypes for each locus. The genotypes are represented as alleles, with missing data represented by a specific character (e.g., "0" or "N"). The file can be in one-row or two-row format, depending on whether the genotypes are presented in a single line or two lines per individual.

    Here is an example of a STRUCTURE file in one-row format:
    ```
    Sample1 0 0 0 0
    Sample2 0 0 0 0
    Sample3 0 0 0 0
    Sample4 0 0 0 0
    ```

    In this example, there are 4 individuals and 4 loci. The first line contains the number of individuals and loci, followed by lines for each individual with their genotypes.

    Here is an example of a STRUCTURE file in two-row format:
    ```
    Sample1 0 0 0 0
    Sample1 0 0 0 0
    Sample2 0 0 0 0
    Sample2 0 0 0 0
    Sample3 0 0 0 0
    Sample3 0 0 0 0
    Sample4 0 0 0 0
    Sample4 0 0 0 0
    Sample4 0 0 0 0
    ```

    In this example, there are 4 individuals and 4 loci. Each individual has two lines of data, with the first line containing the individual's name and the second line containing the genotypes.

    Example:
        >>> from snpio import StructureReader
        >>>
        >>> genotype_data = StructureReader(filename="data.structure", popmapfile="example.popmap", verbose=True)
        >>>
        >>> genotype_data.snp_data
        array([["A", "T", "T", "A"], ["C", "G", "G", "C"], ["A", "T", "T", "A"]], dtype="<U1")
        >>>
        >>> genotype_data.samples
        ["Sample1", "Sample2", "Sample3", "Sample4"]
        >>>
        >>> genotype_data.populations
        ["Pop1", "Pop1", "Pop2", "Pop2"]
        >>>
        >>> genotype_data.num_snps
        3
        >>>
        >>> genotype_data.num_inds
        4
        >>>
        >>> genotype_data.popmap
        >>> {"Sample1": "Pop1", "Sample2": "Pop1", "Sample3": "Pop2", "Sample4": "Pop2"}
        >>>
        >>> genotype_data.popmap_inverse
        {"Pop1": ["Sample1", "Sample2"], "Pop2": ["Sample3", "Sample4"]}
        >>>
        >>> genotype_data.ref
        ["A", "C", "A"]
        >>>
        >>> genotype_data.alt
        ["T", "G", "T"]
        >>>
        >>> genotype_data.missingness_reports()
        >>>
        >>> genotype_data.run_pca()
        >>>
        >>> genotype_data.write_structure("output.str")

    Attributes:
        logger (LoggerManager): Logger object.
        verbose (bool): If True, status updates are printed.
        debug (bool): If True, debug messages are printed.
        _has_popids (bool): If True, the STRUCTURE file includes population IDs.
        _onerow (bool): If True, the STRUCTURE file is in one-row format.
    """

    def __init__(
        self,
        filename: str | None = None,
        popmapfile: str | None = None,
        has_popids: bool = False,
        force_popmap: bool = False,
        exclude_pops: List[str] | None = None,
        include_pops: List[str] | None = None,
        plot_format: Literal["png", "pdf", "jpg", "svg"] = "png",
        prefix: str = "snpio",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the StructureReader class.

        This class reads STRUCTURE files and stores the SNP data, sample IDs, and populations. It is a subclass of GenotypeData and inherits its attributes and methods. The STRUCTURE file can be written to a new file using the write_structure method. SNP data can be accessed in the IUPAC format using the `snp_data` attribute. The class also provides methods for filtering, PCA, and plotting.

        Args:
            filename (str, optional): Name of the STRUCTURE file.
            popmapfile (str, optional): Name of the population map file.
            has_popids (bool, optional): If True, the STRUCTURE file includes population IDs.
            force_popmap (bool, optional): If True, force the use of the popmap file.
            exclude_pops (List[str], optional): List of populations to exclude.
            include_pops (List[str], optional): List of populations to include.
            plot_format (str, optional): Format for saving plots (e.g., 'png', 'svg').
            prefix (str, optional): Prefix for output files.
            verbose (bool, optional): If True, status updates are printed.
            debug (bool, optional): If True, debug messages are printed.

        Note:
            - The STRUCTURE file format consists of a header line followed by lines of data. Each line represents an individual and contains the individual's name, population ID (if present), and the genotypes for each locus. The genotypes are represented as alleles, with missing data represented by a specific character (e.g., "0" or "N"). The file can be in one-row or two-row format, depending on whether the genotypes are presented in a single line or two lines per individual.
            - The STRUCTURE file can be in one-row or two-row format. In one-row format, each line contains the individual's name and the genotypes for each locus. In two-row format, each individual has two lines of data, with the first line containing the individual's name and the second line containing the genotypes.

            - The class provides methods for filtering, PCA, and plotting. The class is initialized with the filename of the STRUCTURE file, the population map file, and other optional parameters.
        """
        # Set up logger
        kwargs = dict(prefix=prefix, verbose=verbose, debug=debug)
        logman = LoggerManager(name=__name__, **kwargs)
        self.logger = logman.get_logger()

        self.verbose = verbose
        self.debug = debug

        self._has_popids: bool = has_popids
        self._onerow: bool = False

        self.resource_data = {}

        # Initialize the parent class GenotypeData
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

    def load_aln(self) -> None:
        """Load the STRUCTURE file and populate SNP data, samples, and populations.

        This method reads the STRUCTURE file and populates the SNP data, sample IDs, and populations. It also sets the number of SNPs and individuals.

        Raises:
            AlignmentNotFoundError: If the STRUCTURE file is not found.
            AlignmentFormatError: If the STRUCTURE file has an invalid format.
            StructureAlignmentSampleMismatch: If the number of samples in the STRUCTURE file does not match the number of samples in the population map file.
            AlignmentError: If there is an error while reading the STRUCTURE file.

        Note:
            This method should be called after initializing the StructureReader object.

            The SNP data, sample IDs, and populations are stored in the snp_data, samples, and populations attributes, respectively.4

            The number of SNPs and individuals are stored in the num_snps and num_inds attributes, respectively.

            The STRUCTURE file can be written to a new file using the write_structure method.

            The SNP data can be accessed in the IUPAC format using the `snp_data` attribute.
        """
        if not self.filename:
            raise AlignmentFileNotFoundError(self.filename)

        if not Path(self.filename).is_file():
            raise AlignmentFileNotFoundError(self.filename)

        self.logger.info(f"Reading STRUCTURE file {self.filename}...")

        samples = []
        snp_data = []

        try:
            self._onerow = self._detect_format()

            with open(self.filename, "r") as fin:
                lines = fin.readlines()

            if self._onerow:
                for line in lines:
                    line = line.strip().split()
                    samples.append(line[0])
                    alleles = line[1:] if not self._has_popids else line[2:]
                    snp_data.append(alleles)
            else:
                for i in range(0, len(lines), 2):
                    firstline = lines[i].strip().split()
                    secondline = lines[i + 1].strip().split()

                    if firstline[0] != secondline[0]:
                        raise AlignmentFormatError(
                            f"Sample names do not match between lines: "
                            f"{firstline[0]} and {secondline[0]}"
                        )

                    samples.append(firstline[0])
                    alleles1 = firstline[1:] if not self._has_popids else firstline[2:]
                    alleles2 = (
                        secondline[1:] if not self._has_popids else secondline[2:]
                    )

                    merged_alleles = np.array(
                        [f"{a1}/{a2}" for a1, a2 in zip(alleles1, alleles2)]
                    )
                    snp_data.append(merged_alleles)

            self.samples = np.unique(samples).tolist()

            if len(self.samples) != len(snp_data):
                self.logger.error("Unexpected number of samples found.")
                raise StructureAlignmentSampleMismatch(len(self.samples), len(snp_data))

            self.snp_data = [
                list(map(self._genotype_to_iupac, row)) for row in snp_data
            ]

            self.logger.info(f"STRUCTURE file successfully loaded!")
            self.logger.info(
                f"Found {self.num_snps} SNPs and {self.num_inds} individuals."
            )

        except (AlignmentError, Exception) as e:
            msg = f"An error occurred while reading the STRUCTURE file: {e}"
            self.logger.error(msg)
            raise e

    def _detect_format(self) -> bool:
        """Detect the format of the STRUCTURE file (onerow or tworow).

        This method reads the first two lines of the STRUCTURE file and compares the sample names. If the sample names are the same, the file is in one-row format; otherwise, it is in two-row format.

        Returns:
            bool: True if the file is in one-row format, False if it is in two-row format.

        Raises:
            AlignmentFileNotFoundError: If the STRUCTURE file is not found.
            AlignmentFormatError: If the STRUCTURE file has an invalid format.
        """
        if not Path(self.filename).is_file():
            msg = f"STRUCTURE file {self.filename} not found."
            self.logger.error(msg)
            raise AlignmentFileNotFoundError(self.filename)

        with open(self.filename, "r") as fin:
            first_line = fin.readline().split()
            second_line = fin.readline().split()

        # If the first two lines have the same sample name, then it's a two-row format
        onerow = first_line[0] != second_line[0]
        return onerow
