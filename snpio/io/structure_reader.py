from pathlib import Path

import numpy as np
from typing import Optional, List

from snpio.utils.logging import setup_logger
from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.custom_exceptions import AlignmentError, AlignmentFileNotFoundError, AlignmentFormatError, StructureAlignmentSampleMismatch

logger = setup_logger(__name__)

class StructureReader(GenotypeData):
    def __init__(
        self,
        filename: Optional[str] = None,
        popmapfile: Optional[str] = None,
        has_popids: bool = False,
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
            filetype="structure",
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

        self._has_popids: bool = has_popids
        self._onerow: bool = False

        # Load file data if filename is provided
        if self.filename:
            self._load_aln()

    def _detect_format(self) -> bool:
        """
        Detect the format of the STRUCTURE file (onerow or tworow).

        Returns:
            bool: True if the file is in one-row format, False if it is in two-row format.
        """
        with open(self.filename, "r") as fin:
            first_line = fin.readline().split()
            second_line = fin.readline().split()

        # If the first two lines have the same sample name, then it's a two-row format
        onerow = first_line[0] != second_line[0]
        return onerow

    def _load_aln(self) -> None:
        """
        Load the STRUCTURE file and populate SNP data, samples, and populations.

        Raises:
            AlignmentNotFoundError: If the STRUCTURE file is not found.
            AlignmentFormatError: If the STRUCTURE file has an invalid format.
        """
        if not self.filename:
            raise AlignmentFileNotFoundError(self.filename)

        if not Path(self.filename).is_file():
            raise AlignmentFileNotFoundError(self.filename)

        if self.verbose:
            logger.info(f"Reading STRUCTURE file {self.filename}...")

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
                    alleles2 = secondline[1:] if not self._has_popids else secondline[2:]

                    merged_alleles = np.array([f"{a1}/{a2}" for a1, a2 in zip(alleles1, alleles2)])
                    snp_data.append(merged_alleles)

            self.samples = np.unique(samples).tolist()
            
            if len(self.samples) != len(snp_data):
                logger.error("Unexpected number of samples found.")
                raise StructureAlignmentSampleMismatch(len(self.samples), len(snp_data))
            
            self._snp_data = [list(map(self._genotype_to_iupac, row)) for row in snp_data]
            self._validate_seq_lengths()

            if self.verbose:
                logger.info(f"STRUCTURE file successfully loaded!")
                logger.info(f"Found {self.num_snps} SNPs and {self.num_inds} individuals.")

        except (AlignmentError, Exception) as e:
            msg = f"An error occurred while reading the STRUCTURE file: {e}"
            logger.error(msg)
            raise

    def write_structure(
        self,
        output_file: str,
        genotype_data=None,
        snp_data: Optional[List[List[str]]] = None,
        samples: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Write the stored alignment as a STRUCTURE file.

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
        if verbose:
            logger.info(f"\nWriting STRUCTURE file {output_file}...")

        if genotype_data is not None and snp_data is None:
            snp_data = genotype_data.snp_data
            samples = genotype_data.samples
        elif genotype_data is None and snp_data is None:
            snp_data = self._snp_data
            samples = self.samples
        elif genotype_data is None and snp_data is not None:
            if samples is None:
                raise TypeError("samples must be provided if snp_data is provided.")


        try:
            with open(output_file, "w") as fout:
                for sample, sample_data in zip(samples, snp_data):
                    # Convert IUPAC codes back to genotype format (e.g., 1/1, 2/2)
                    genotypes = [self._iupac_to_genotype(iupac) for iupac in sample_data]

                    if self._onerow:
                        # Flatten the genotype pairs and write to file in one-row format
                        genotype_pairs = [allele for genotype in genotypes for allele in genotype.split("/")]
                        fout.write(f"{sample}\t" + "\t".join(genotype_pairs) + "\n")
                    else:
                        # Write the two alleles in two separate rows for two-row format
                        first_row = [genotype.split("/")[0] for genotype in genotypes]
                        second_row = [genotype.split("/")[1] for genotype in genotypes]
                        fout.write(f"{sample}\t" + "\t".join(first_row) + "\n")
                        fout.write(f"{sample}\t" + "\t".join(second_row) + "\n")

            if verbose:
                logger.info("Successfully wrote STRUCTURE file!")
        except Exception as e:
            msg = f"An error occurred while writing the STRUCTURE file: {e}"
            logger.error(msg)
            raise
