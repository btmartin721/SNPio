from pathlib import Path
from typing import List, Optional

import numpy as np

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.custom_exceptions import (
    AlignmentError,
    AlignmentFileNotFoundError,
    AlignmentFormatError,
    PhylipAlignmentSampleMismatch,
    SequenceLengthError,
)
from snpio.utils.logging import setup_logger

# Configure logging
logger = setup_logger(__name__)


class PhylipReader(GenotypeData):
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
            filetype="phylip",
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

        # Load file data if filename is provided
        if self.filename:
            self._load_aln()

    def _load_aln(self) -> None:
        """
        Load the PHYLIP file and populate SNP data, samples, and alleles.

        Raises:
            AlignmentFileNotFoundError: If the PHYLIP file is not found.
            AlignmentFormatError: If the PHYLIP file has an invalid format.
        """
        if not self.filename:
            msg = "No filename provided for PHYLIP file."
            logger.error(msg)
            raise AlignmentFormatError(msg)

        if not Path(self.filename).is_file():
            raise AlignmentFileNotFoundError(self.filename)

        if self.verbose:
            logger.info(f"Reading PHYLIP file {self.filename}...")

        snp_data = []
        try:
            with open(self.filename, "r") as fin:
                header = fin.readline().strip()
                if not header:
                    msg = "PHYLIP file header is missing or empty."
                    logger.error(msg)
                    raise AlignmentFormatError(msg)

                try:
                    n_samples, n_loci = map(int, header.split())
                except ValueError:
                    msg = "Invalid PHYLIP header provided."
                    logger.error(msg)
                    raise AlignmentFormatError(msg)

                for line in fin:
                    line = line.strip()
                    if not line:  # Skip blank lines
                        continue
                    cols = line.split()
                    if len(cols) < 2:
                        msg = f"PHYLIP file line does not contain enough columns. Expected 2, but got: {len(cols)}"
                        logger.error(msg)
                        raise AlignmentFormatError(msg)

                    if len(cols) > 2:
                        msg = f"PHYLIP file contains too many columns. Expected 2, but got: {len(cols)}"
                        logger.error(msg)
                        raise AlignmentFormatError(msg)

                    inds, seqs = cols[0], cols[1]
                    if len(seqs) != n_loci:
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
                logger.error(msg)
                PhylipAlignmentSampleMismatch(
                    n_samples, len(self.samples), len(snp_data)
                )
            self._ref, self._alt, self._alt2 = self._get_ref_alt_alleles(
                np.array(snp_data)
            )

            if self.verbose:
                logger.info(f"PHYLIP file successfully loaded!")
                logger.info(
                    f"Found {self.num_snps} SNPs and {self.num_inds} individuals."
                )
        except (AlignmentError, Exception) as e:
            msg = f"An error occurred while reading the PHYLIP file: {e}"
            logger.error(msg)
            raise

        self._validate_seq_lengths()

    def write_phylip(
        self,
        output_file: str,
        genotype_data=None,
        snp_data: Optional[List[List[str]]] = None,
        samples: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Write the stored alignment as a PHYLIP file.

        Args:
            output_file (str): Name of the output PHYLIP file.
            genotype_data (GenotypeData, optional): GenotypeData instance.
            snp_data (List[List[str]], optional): SNP data. Must be provided if genotype_data is None.
            samples (List[str], optional): List of sample IDs. Must be provided if snp_data is not None.
            verbose (bool, optional): If True, status updates are printed.

        Raises:
            TypeError: If genotype_data and snp_data are both provided.
            TypeError: If samples are not provided when snp_data is provided.
            ValueError: If samples and snp_data are not the same length.
        """
        if verbose:
            logger.info(f"\nWriting to PHYLIP file {output_file}...")

        if genotype_data is not None and snp_data is not None:
            msg = "'genotype_data' and 'snp_data' cannot both be provided."
            logger.error(msg)
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
                logger.error(msg)
                raise TypeError(msg)

        self._validate_seq_lengths()

        try:
            with open(output_file, "w") as f:
                n_samples, n_loci = len(samples), len(snp_data[0])
                f.write(f"{n_samples} {n_loci}\n")
                for sample, sample_data in zip(samples, snp_data):
                    genotype_str = "".join(str(x) for x in sample_data)
                    f.write(f"{sample}\t{genotype_str}\n")

            if verbose:
                logger.info(f"Successfully wrote PHYLIP file {output_file}!")
        except Exception as e:
            msg = f"An error occurred while writing the PHYLIP file: {e}"
            logger.error(msg)
            raise

    @property
    def ref(self) -> List[str]:
        return self._ref

    @property
    def alt(self) -> List[str]:
        return self._alt

    @property
    def alt2(self) -> List[str]:
        return self._alt2
