from pathlib import Path
from typing import List, Optional

import numpy as np

from snpio.read_input.genotype_data import GenotypeData
from snpio.utils.custom_exceptions import (
    AlignmentFileNotFoundError,
    AlignmentFormatError,
    SequenceLengthError,
)
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC


class GenePopReader(GenotypeData):
    """Reads GenePop-formatted files into a GenotypeData object.

    Supports:
    - 2-digit and 3-digit allele codings
    - Mixed ploidy: haploid and diploid entries
    - Missing data encoded as 0000, 000000, or partial (e.g., 1000, 0010)
    - Flexible locus headers (comma-separated or newline-separated)

    Example:
        >>> gp = GenePopReader("example.gen", popmapfile="pops.txt")
        >>> gp.snp_data
        array([...])
    """

    def __init__(
        self,
        filename: str,
        popmapfile: Optional[str] = None,
        allele_encoding: dict[str, str] | None = None,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        plot_format: str = "png",
        prefix: str = "snpio",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the GenePopReader.

        This class reads a GenePop file and extracts genotype data, populations, and sample information.

        Args:
            filename (str): Path to the GenePop file.
            popmapfile (Optional[str]): Path to the population map file.
            allele_encoding (Optional[dict[str, str]]): Mapping of allele codes to IUPAC symbols.
            force_popmap (bool): Whether to enforce population mapping.
            exclude_pops (Optional[List[str]]): List of populations to exclude.
            include_pops (Optional[List[str]]): List of populations to include.
            plot_format (str): Format for output plots (default: "png").
            prefix (str): Prefix for output files (default: "snpio").
            verbose (bool): Whether to enable verbose logging (default: False).
            debug (bool): Whether to enable debug mode (default: False).
        """
        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(name=__name__, **kwargs)
        self.logger = logman.get_logger()
        self.iupac = IUPAC(logger=self.logger)

        self.verbose = verbose
        self.debug = debug

        self.allele_encoding = allele_encoding or {
            "01": "A",
            "02": "C",
            "03": "G",
            "04": "T",
            "001": "A",
            "002": "C",
            "003": "G",
            "004": "T",
        }

        self.resource_data = {}

        super().__init__(
            filename=filename,
            filetype="genepop",
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
        path = Path(self.filename)
        if not path.exists():
            msg = f"GenePop file {self.filename} not found."
            self.logger.error(msg)
            raise AlignmentFileNotFoundError(msg)

        self.logger.info(f"Reading GenePop file {self.filename}...")

        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            msg = f"GenePop file {self.filename} is empty or malformed."
            self.logger.error(msg)
            raise AlignmentFormatError(msg)

        # Skip title line
        lines = lines[1:]

        # Extract marker names
        self.marker_names = []
        while lines and not lines[0].upper().startswith("POP"):
            line = lines.pop(0)
            self.marker_names.extend(
                [m.strip() for m in line.replace(",", " ").split() if m.strip()]
            )

        # Parse genotypes
        samples, populations, genotypes = [], [], []
        current_pop = None
        pop_count = 0

        for line in lines:
            if line.upper() == "POP":
                pop_count += 1
                current_pop = f"Pop{pop_count}"
                continue

            try:
                sid, gts = line.split(",", 1)
                sid = sid.strip()
                allele_strs = gts.strip().split()

                if not allele_strs:
                    msg = f"Sample {sid} has no genotype data."
                    self.logger.error(msg)
                    raise AlignmentFormatError(msg)

                samples.append(sid)
                populations.append(current_pop)

                gt_calls = []
                for code in allele_strs:
                    code = code.strip()
                    if not code or not code.isdigit():
                        msg = f"Invalid genotype format: '{code}'"
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    if len(code) not in {2, 4, 6}:
                        msg = f"Unexpected allele length for sample {sid}: {code}"
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    # Determine ploidy and convert to IUPAC
                    if len(code) == 2:  # haploid 1 allele
                        a1 = code
                        a2 = code
                    elif len(code) == 4:
                        a1, a2 = code[:2], code[2:]
                    elif len(code) == 6:
                        a1, a2 = code[:3], code[3:]
                    else:
                        msg = f"Unrecognized allele length: {code}"
                        self.logger.error(msg)
                        raise AlignmentFormatError(msg)

                    gt_calls.append(self._convert_to_iupac(a1, a2))

                genotypes.append(gt_calls)
            except ValueError:
                msg = f"Malformed line in GenePop file: '{line}'"
                self.logger.error(msg)
                raise AlignmentFormatError(msg)

        self.samples = samples
        self.populations = populations
        self.snp_data = np.array(genotypes, dtype="<U1")

        if len(set(map(len, genotypes))) > 1:
            msg = "Inconsistent number of loci across samples."
            self.logger.error(msg)
            raise SequenceLengthError(msg)

        self._ref, self._alt, self._alt2 = self.get_ref_alt_alleles(self.snp_data)

        self.logger.info(
            f"Loaded {len(samples)} samples across {len(set(populations))} populations."
        )
        self.logger.info(f"Found {self.num_snps} SNPs.")

    def _convert_to_iupac(self, allele1: str, allele2: str) -> str:
        """Convert two alleles to IUPAC code."""
        missing_vals = {"00", "000", "99", "999"}
        if allele1 in missing_vals or allele2 in missing_vals:
            return "N"
        try:
            return self.iupac.encode_genepop_pair(
                allele1, allele2, allele_map=self.allele_encoding
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to convert alleles {allele1}/{allele2} to IUPAC: {e}"
            )
            return "N"
