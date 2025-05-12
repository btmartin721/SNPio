from typing import List


class SNPioError(Exception):
    """Base exception class for all SNPio errors."""

    def __init__(self, message: str = "An SNPio-related error occurred.") -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class EmptyIterableError(SNPioError):
    """Raised when a subset of an iterable is empty."""

    pass


class PopmapKeyError(SNPioError):
    """Raised when a key is not found in the population map."""

    def __init__(self, key: str, message: str = None) -> None:
        self.key = key
        msg = (
            message
            or f"Population map sample '{key}' not found in the alignment or population map file."
        )
        super().__init__(msg)


class PopmapFileFormatError(SNPioError):
    """Raised when there is an error reading the population map file."""

    pass


class PopmapFileNotFoundError(SNPioError):
    """Raised when the population map file is not found."""

    def __init__(self, filename: str, message: str = None) -> None:
        self.filename = filename
        msg = message or f"Population map file '{filename}' not found."
        super().__init__(msg)


class UnsupportedFileTypeError(SNPioError):
    """Raised when the given file type is unsupported."""

    def __init__(self, filetype: str, supported_types: List[str] | None = None) -> None:
        if supported_types is None:
            supported_types = ["vcf", "phylip", "structure"]
        self.filetype = filetype
        self.supported_types = supported_types
        message = f"Alignment file type '{filetype}' is not supported. Supported types are {', '.join(supported_types)}."
        super().__init__(message)


class AlignmentError(SNPioError):
    """Base class for alignment file-related exceptions."""

    pass


class AlignmentFormatError(AlignmentError):
    """Raised when an alignment file has an invalid format."""

    pass


class SequenceLengthError(AlignmentFormatError):
    """Raised when sequences in the alignment file have unequal lengths."""

    def __init__(self, sample_name: str) -> None:
        msg = f"Sequences have unequal lengths. Error at sample: {sample_name}"
        super().__init__(msg)


class PhylipAlignmentSampleMismatch(AlignmentFormatError):
    """Raised when samples in the PHYLIP alignment file don't match header."""

    def __init__(
        self, header_samples: int, sample_list_len: int, snp_data_len: int
    ) -> None:
        msg = (
            f"Unexpected number of samples encountered. Expected {header_samples}, "
            f"but got: {sample_list_len} samples across {snp_data_len} genotypes."
        )
        super().__init__(msg)


class StructureAlignmentSampleMismatch(AlignmentFormatError):
    """Raised when STRUCTURE sample list and genotype rows don't align."""

    def __init__(self, sample_list_len: int, snp_data_len: int) -> None:
        msg = (
            f"Unexpected number of samples encountered. Expected {sample_list_len}, "
            f"but got: {snp_data_len} genotype rows."
        )
        super().__init__(msg)


class AlignmentFileNotFoundError(AlignmentError):
    """Raised when an alignment file is not found."""

    def __init__(self, filename: str, message: str = None) -> None:
        self.filename = filename
        msg = message or f"Alignment file '{filename}' not found."
        super().__init__(msg)


class NoValidAllelesError(SNPioError):
    """Raised when no valid alleles are found in a SNP column."""

    def __init__(self, column_index: int, message: str = None) -> None:
        self.column_index = column_index
        msg = message or f"No valid alleles found in column {column_index}"
        super().__init__(msg)


class InvalidGenotypeError(SNPioError):
    """Raised when a genotype string is invalid or unrecognized."""

    def __init__(self, message: str = None) -> None:
        msg = message or "Invalid or unrecognized genotype string."
        super().__init__(msg)


class MissingPopulationMapError(SNPioError):
    """Raised when a required population map is missing."""

    def __init__(self, message: str = None) -> None:
        msg = message or "Missing required population map file or object."
        super().__init__(msg)


class NonBiallelicSiteError(SNPioError):
    """Raised when a non-biallelic SNP site is encountered where only biallelic sites are allowed."""

    def __init__(self, message: str = None) -> None:
        msg = (
            message
            or "Encountered a non-biallelic SNP site where only biallelic sites are allowed."
        )
        super().__init__(msg)


class EmptyLocusSetError(SNPioError):
    """Raised when no loci remain after filtering or subsetting."""

    def __init__(self, message: str = None) -> None:
        msg = message or "No loci remain after filtering or subsetting."
        super().__init__(msg)


class InvalidVCFHeaderError(SNPioError):
    """Raised when the VCF header is malformed or missing required fields."""

    pass


class EncodingMismatchError(SNPioError):
    """Raised when the genotype encoding is incompatible with the data format."""

    pass


class StatisticalComputationError(SNPioError):
    """Raised when a statistical calculation fails due to data or input issues."""

    pass


class PermutationInferenceError(SNPioError):
    """Raised when bootstrapping or permutation-based inference fails."""

    pass


class InvalidChunkSizeError(SNPioError):
    """Raised when the chunk size for I/O streaming is invalid."""

    pass


class UnsupportedFileFormatError(SNPioError):
    """Raised when an unsupported file format is encountered."""

    pass


class InvalidThresholdError(SNPioError):
    """Raised when an invalid threshold is provided for filtering."""

    def __init__(self, threshold: float, message: str = None) -> None:
        self.threshold = threshold
        msg = (
            message or f"Invalid threshold value: {threshold}. Must be between 0 and 1."
        )
        super().__init__(msg)
