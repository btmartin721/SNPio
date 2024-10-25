from pathlib import Path
from typing import List, Optional

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

    ``StructureReader`` is a subclass of GenotypeData and inherits its attributes and methods. It reads STRUCTURE files and stores the SNP data, sample IDs, and populations, as well as various other attributes.

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
        filename: Optional[str] = None,
        popmapfile: Optional[str] = None,
        has_popids: bool = False,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        plot_format: Optional[str] = "png",
        prefix="snpio",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the StructureReader class.

        This class reads STRUCTURE files and stores the SNP data, sample IDs, and populations. It is a subclass of GenotypeData and inherits its attributes and methods.

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

        Raises:
            AlignmentFileNotFoundError: If the STRUCTURE file is not found.

        Note:
            If the filename is provided, the STRUCTURE file is loaded immediately.

            If the filename is not provided, the STRUCTURE file can be loaded later using the load_aln method.

            The STRUCTURE file can be written to a new file using the write_structure method.
        """

        # Set up logger
        kwargs = dict(prefix=prefix, verbose=verbose, debug=debug)
        logman = LoggerManager(name=__name__, **kwargs)
        self.logger = logman.get_logger()

        self.verbose = verbose
        self.debug = debug

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

        self._has_popids: bool = has_popids
        self._onerow: bool = False

    def load_aln(self) -> None:
        """Load the STRUCTURE file and populate SNP data, samples, and populations.

        This method reads the STRUCTURE file and populates the SNP data, sample IDs, and populations. It also sets the number of SNPs and individuals.

        Raises:
            AlignmentNotFoundError: If the STRUCTURE file is not found.
            AlignmentFormatError: If the STRUCTURE file has an invalid format.

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
            raise

    def write_structure(
        self,
        output_file: str,
        genotype_data=None,
        snp_data: Optional[List[List[str]]] = None,
        samples: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        """Write the stored alignment as a STRUCTURE file.

        This method writes the stored alignment as a STRUCTURE file. If genotype_data is provided, the SNP data and sample IDs are extracted from it. Otherwise, the SNP data and sample IDs must be provided.

        Args:
            output_file (str): Name of the output STRUCTURE file.
            genotype_data (GenotypeData, optional): GenotypeData instance.
            snp_data (List[List[str]], optional): SNP data in IUPAC format. Must be provided if genotype_data is None.
            samples (List[str]], optional): List of sample IDs. Must be provided if snp_data is not provided.
            verbose (bool, optional): If True, status updates are printed.

        Raises:
            TypeError: If genotype_data and snp_data are both provided.
            TypeError: If samples are not provided when snp_data is provided.
        """
        self.logger.info(f"\nWriting STRUCTURE file {output_file}...")

        if genotype_data is not None and snp_data is None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
        elif genotype_data is None and snp_data is None:
            snp_data = self.snp_data
            samples = self.samples
        elif genotype_data is None and snp_data is not None:
            if samples is None:
                raise TypeError("samples must be provided if snp_data is provided.")

        try:
            with open(output_file, "w") as fout:
                for sample, sample_data in zip(samples, snp_data):
                    # Convert IUPAC codes back to genotype format (e.g., 1/1, 2/2)
                    genotypes = [
                        self._iupac_to_genotype(iupac) for iupac in sample_data
                    ]

                    if self._onerow:
                        # Flatten the genotype pairs and write to file in one-row format
                        genotype_pairs = [
                            allele
                            for genotype in genotypes
                            for allele in genotype.split("/")
                        ]
                        fout.write(f"{sample}\t" + "\t".join(genotype_pairs) + "\n")
                    else:
                        # Write the two alleles in two separate rows for two-row format
                        first_row = [genotype.split("/")[0] for genotype in genotypes]
                        second_row = [genotype.split("/")[1] for genotype in genotypes]
                        fout.write(f"{sample}\t" + "\t".join(first_row) + "\n")
                        fout.write(f"{sample}\t" + "\t".join(second_row) + "\n")

            self.logger.info("Successfully wrote STRUCTURE file!")
        except Exception as e:
            msg = f"An error occurred while writing the STRUCTURE file: {e}"
            self.logger.error(msg)
            raise

    def _detect_format(self) -> bool:
        """Detect the format of the STRUCTURE file (onerow or tworow).

        This method reads the first two lines of the STRUCTURE file and compares the sample names. If the sample names are the same, the file is in one-row format; otherwise, it is in two-row format.

        Returns:
            bool: True if the file is in one-row format, False if it is in two-row format.
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
