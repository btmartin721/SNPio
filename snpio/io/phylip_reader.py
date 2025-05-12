from pathlib import Path
from typing import List

import numpy as np

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.custom_exceptions import (
    AlignmentError,
    AlignmentFileNotFoundError,
    AlignmentFormatError,
    PhylipAlignmentSampleMismatch,
    SequenceLengthError,
)
from snpio.utils.logging import LoggerManager


class PhylipReader(GenotypeData):
    """Class to read and write PHYLIP files.

    This class inherits from the GenotypeData class and provides methods to read and write PHYLIP files. The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data.

    Example:
        >>> from snpio import PhylipReader
        >>>
        >>> phylip = PhylipReader(filename="example.phy", popmapfile="example.popmap", verbose=True)
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
        >>> genotype_data.write_phylip("output.str")

    Attributes:
        filename (str): Name of the PHYLIP file.
        popmapfile (str): Name of the population map file.
        force_popmap (bool): If True, the population map file is required.
        exclude_pops (List[str]): List of populations to exclude.
        include_pops (List[str]): List of populations to include.
        plot_format (str): Format for saving plots. Default is 'png'.
        prefix (str): Prefix for output files.
        verbose (bool): If True, status updates are printed.
        samples (List[str]): List of sample IDs.
        snp_data (List[List[str]]): List of SNP data.
        num_inds (int): Number of individuals.
        num_snps (int): Number of SNPs.
        logger (Logger): Logger instance.
        debug (bool): If True, debug messages are printed.
    """

    def __init__(
        self,
        filename: str | None = None,
        popmapfile: str | None = None,
        force_popmap: bool = False,
        exclude_pops: List[str] | None = None,
        include_pops: List[str] | None = None,
        plot_format: str | None = "png",
        prefix: str = "snpio",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the PhylipReader class.

        This method sets up the logger and initializes the list of missing values. It also takes a filename and a population map file to read the data. The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data.

            For example:

            ```
            4 4
            Sample1 ATTA
            Sample2 CGGC
            Sample3 ATTA
            Sample4 CGGC
            ```

        Args:
            filename (str | None): Name of the PHYLIP file. Defaults to None.
            popmapfile (str | None): Name of the population map file. Defaults to None.
            force_popmap (bool): If True, the population map file is required. Defaults to False.
            exclude_pops (List[str] | None): List of populations to exclude. Defaults to None.
            include_pops (List[str] | None): List of populations to include. Defaults to None.
            plot_format (str | None): Format for saving plots. Default is 'png'. Defaults to 'png'.
            prefix (str): Prefix for output files. Defaults to 'snpio'.
            verbose (bool): If True, status updates are printed. Defaults to False.
            debug (bool): If True, debug messages are printed. Defaults to False.

        Note:
            The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data.
        """
        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(name=__name__, **kwargs)
        self.logger = logman.get_logger()

        self.verbose = verbose
        self.debug = debug

        self.resource_data = {}

        # Initialize the parent class GenotypeData
        super().__init__(
            filename=filename,
            filetype="phylip",
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
        """Load the PHYLIP file and populate SNP data, samples, and alleles.

        This method reads the PHYLIP file and populates the SNP data, samples, and alleles. The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data. The sequence data can be in any format, but it is typically a string of nucleotides or amino acids.

        Raises:
            AlignmentFileNotFoundError: If the PHYLIP file is not found.
            AlignmentFormatError: If the PHYLIP file has an invalid format.
            SequenceLengthError: If the sequence length does not match the expected length.
            PhylipAlignmentSampleMismatch: If the number of samples in the PHYLIP file does not match the number of samples in the population map file.
        """
        if not self.filename or self.filename is None:
            msg = "No filename provided for PHYLIP file."
            self.logger.error(msg)
            raise AlignmentFileNotFoundError(msg)

        if not Path(self.filename).is_file() or not Path(self.filename).exists():
            msg = f"PHYLIP file {self.filename} not found."
            self.logger.error(msg)
            raise AlignmentFileNotFoundError(self.filename)

        self.logger.info(f"Reading PHYLIP file {self.filename}...")

        snp_data = []
        try:
            with open(self.filename, "r") as fin:
                header = fin.readline().strip()
                if not header:
                    msg = "PHYLIP file header is missing or the file is empty."
                    self.logger.error(msg)
                    raise AlignmentFileNotFoundError(msg)

                try:
                    n_samples, n_loci = map(int, header.split())
                except ValueError:
                    msg = "Invalid PHYLIP header provided."
                    self.logger.error(msg)
                    raise AlignmentFormatError(msg)

                for line in fin:
                    line = line.strip()
                    if not line:  # Skip blank lines
                        continue
                    cols = line.split()
                    if len(cols) < 2:
                        msg = f"PHYLIP file contains too many columns. Expected 2, but got: {len(cols)}. Ensure that your PHYLIP file is in the correct format. The first column should contain the sample ID, and the second column should contain the sequence data with no delimiter."
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    if len(cols) > 2:
                        msg = f"PHYLIP file contains too many columns. Expected 2, but got: {len(cols)}. Ensure that your PHYLIP file is in the correct format. The first column should contain the sample ID, and the second column should contain the sequence data with no delimiter."
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    inds, seqs = cols[0], cols[1]
                    if len(list(seqs)) != n_loci:
                        self.logger.error(
                            f"Sequence length mismatch for sample {inds}: {len(list(seqs))} != {n_loci}"
                        )
                        raise SequenceLengthError(inds)

                    snp_data.append(list(seqs))
                    if inds not in self.samples:
                        self.samples.append(inds)

            self.snp_data = snp_data

            if (
                n_samples != len(snp_data)
                or len(self.samples) != len(snp_data)
                or len(self.samples) != n_samples
            ):
                msg = "Unexpected number of samples encountered."
                self.logger.error(msg)
                PhylipAlignmentSampleMismatch(
                    n_samples, len(self.samples), len(snp_data)
                )
            self._ref, self._alt, self._alt2 = self.get_ref_alt_alleles(
                np.array(snp_data)
            )

            self.logger.info(f"PHYLIP file successfully loaded!")
            self.logger.info(
                f"Found {self.num_snps} SNPs and {self.num_inds} individuals."
            )
        except (AlignmentError, Exception) as e:
            msg = f"An error occurred while reading the PHYLIP file: {e}"
            self.logger.error(msg)
            raise e

        self._validate_seq_lengths()
