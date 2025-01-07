from pathlib import Path
from typing import Any, List, Optional

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

    This class provides methods to read and write PHYLIP files. The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data. The sequence data can be in any format, but it is typically a string of nucleotides or amino acids.

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
        filename: Optional[str] = None,
        popmapfile: Optional[str] = None,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        plot_format: Optional[str] = "png",
        prefix: str = "snpio",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the PhylipReader class.

        This class inherits from the GenotypeData class and provides methods to read and write PHYLIP files. The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data. The sequence data can be in any format, but it is typically a string of nucleotides or amino acids.

        Args:
            filename (str, optional): Name of the PHYLIP file. Defaults to None.
            popmapfile (str, optional): Name of the population map file. Defaults to None.
            force_popmap (bool, optional): If True, the population map file is required. Defaults to False.
            exclude_pops (List[str], optional): List of populations to exclude. Defaults to None.
            include_pops (List[str], optional): List of populations to include. Defaults to None.
            plot_format (str, optional): Format for saving plots. Default is 'png'. Defaults to 'png'.
            prefix (str, optional): Prefix for output files. Defaults to 'snpio'.
            verbose (bool, optional): If True, status updates are printed. Defaults to False.
            debug (bool, optional): If True, debug messages are printed. Defaults to False.

        Raises:
            AlignmentFormatError: If the PHYLIP file has an invalid format.
            AlignmentFileNotFoundError: If the PHYLIP file is not found.

        Note:
            The PHYLIP file format is a simple text format for representing multiple sequence alignments.

            The first line of a PHYLIP file contains the number of samples and the number of loci.

            Each subsequent line contains the sample ID followed by the sequence data.

            The sequence data can be in any format, but it is typically a string of nucleotides or amino acids.

            The PHYLIP file must have the correct number of samples and loci.

            The sequence data must have the same length for each sample.
        """
        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(name=__name__, **kwargs)
        self.logger = logman.get_logger()

        self.verbose = verbose
        self.debug = debug

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
        """
        if not self.filename or self.filename is None:
            msg = "No filename provided for PHYLIP file."
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

        if not Path(self.filename).is_file():
            raise AlignmentFileNotFoundError(self.filename)

        self.logger.info(f"Reading PHYLIP file {self.filename}...")

        snp_data = []
        try:
            with open(self.filename, "r") as fin:
                header = fin.readline().strip()
                if not header:
                    msg = "PHYLIP file header is missing or empty."
                    self.logger.error(msg)
                    raise AlignmentFormatError(msg)

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

    def write_phylip(
        self,
        output_file: str,
        genotype_data: Any = None,
        snp_data: Optional[List[List[str]]] = None,
        samples: Optional[List[str]] = None,
    ) -> None:
        """Write the stored alignment as a PHYLIP file.

        This method writes the stored alignment as a PHYLIP file. The PHYLIP format is a simple text format for representing multiple sequence alignments. The first line of a PHYLIP file contains the number of samples and the number of loci. Each subsequent line contains the sample ID followed by the sequence data.

        Args:
            output_file (str): Name of the output PHYLIP file.
            genotype_data (GenotypeData, optional): GenotypeData instance.
            snp_data (List[List[str]], optional): SNP data. Must be provided if genotype_data is None.
            samples (List[str], optional): List of sample IDs. Must be provided if snp_data is not None.

        Raises:
            TypeError: If genotype_data and snp_data are both provided.
            TypeError: If samples are not provided when snp_data is provided.
            ValueError: If samples and snp_data are not the same length.

        Note:
            If genotype_data is provided, the snp_data and samples are loaded from the GenotypeData instance.

            If snp_data is provided, the samples must also be provided.

            If genotype_data is not provided, the snp_data and samples must be provided.

            The sequence data must have the same length for each sample.

            The PHYLIP file must have the correct number of samples and loci.
        """
        self.logger.info(f"Writing to PHYLIP file {output_file}...")

        if genotype_data is not None and snp_data is not None:
            msg = "'genotype_data' and 'snp_data' cannot both be provided."
            self.logger.error(msg)
            raise TypeError(msg)
        elif genotype_data is None and snp_data is None:
            snp_data = self.snp_data
            samples = self.samples
        elif genotype_data is not None and snp_data is None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
        elif genotype_data is None and snp_data is not None:
            if samples is None:
                msg = "'samples' must be provided if 'snp_data' is provided."
                self.logger.error(msg)
                raise TypeError(msg)

        self._validate_seq_lengths()

        try:
            with open(output_file, "w") as f:
                n_samples, n_loci = len(samples), len(snp_data[0])
                f.write(f"{n_samples} {n_loci}\n")
                for sample, sample_data in zip(samples, snp_data):
                    genotype_str = "".join(str(x) for x in sample_data)
                    f.write(f"{sample}\t{genotype_str}\n")

            self.logger.info(f"Successfully wrote PHYLIP file {output_file}!")
        except Exception as e:
            msg = f"An error occurred while writing the PHYLIP file: {e}"
            self.logger.error(msg)
            raise

    @property
    def ref(self) -> List[str]:
        """List of reference alleles."""
        return self._ref

    @property
    def alt(self) -> List[str]:
        """List of alternate alleles."""
        return self._alt

    @property
    def alt2(self) -> List[List[str]]:
        """List of second alternate alleles."""
        return self._alt2
