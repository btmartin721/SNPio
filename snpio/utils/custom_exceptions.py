from typing import List, Optional


class UnsupportedFileTypeError(Exception):
    """Custom exception to check if the given file type is supported.

    Attributes:
        filetype (str): The type of the file being checked.
        message (str): Explanation of the error.
        supported_types (list): List of supported file types.
    """

    def __init__(
        self, filetype: str, supported_types: Optional[List[str]] = None
    ) -> None:
        """Initialize UnsupportedFileTypeException.

        This exception is raised when the given file type is not supported.

        Args:
            filetype (str): The type of the file being checked.
            supported_types (list, optional): List of supported file types.
                Defaults to ["vcf", "phylip", "structure"].
        """
        if supported_types is None:
            supported_types = ["vcf", "phylip", "structure"]

        self.filetype = filetype
        self.supported_types = supported_types
        self.message = f"Alignment file type '{self.filetype}' is not supported. Supported types are {', '.join(self.supported_types)}."
        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation of the exception.

        Returns:
            str: The message associated with the exception.
        """
        return self.message


class AlignmentError(Exception):
    """Base class for exceptions related to alignment file handling."""

    pass


class AlignmentFormatError(AlignmentError):
    """Raised when an alignment file has an invalid format."""

    def __init__(self, message: str) -> None:
        """Initialize AlignmentFormatError.

        Args:
            message (str): The error message.
        """
        super().__init__(message)


class SequenceLengthError(AlignmentFormatError):
    """Raised when sequences in the alignment file have unequal lengths."""

    def __init__(self, sample_name: str) -> None:
        """Initialize SequenceLengthError.

        Args:
            sample_name (str): The name of the sample where the error occurred.
        """
        msg = f"Sequences have unequal lengths. Error at sample: {sample_name}"
        super().__init__(msg)


class PhylipAlignmentSampleMismatch(AlignmentFormatError):
    """Raised when samples in the alignment file do not match the expected number of samples."""

    def __init__(
        self, header_samples: int, sample_list_len: int, snp_data_len: int
    ) -> None:
        """Initialize PhylipAlignmentSampleMismatch.

        Args:
            header_samples (int): The number of samples in the header.
            sample_list_len (int): The number of samples in the sample list.
            snp_data_len (int): The number of genotypes in the SNP data.
        """
        super().__init__(
            f"Unexpected number of samples encountered. Expected {header_samples}, but got: {sample_list_len} samples across {snp_data_len} genotypes."
        )


class StructureAlignmentSampleMismatch(AlignmentFormatError):
    """Raised when samples in the alignment file do not match the expected number of samples."""

    def __init__(self, sample_list_len: int, snp_data_len: int) -> None:
        """Initialize StructureAlignmentSampleMismatch.

        Args:
            sample_list_len (int): The number of samples in the sample list.
            snp_data_len (int): The number of genotypes in the SNP data.
        """
        super().__init__(
            f"Unexpected number of samples encountered. Expected {sample_list_len}, but got: {snp_data_len} genotype rows."
        )


class AlignmentFileNotFoundError(AlignmentError):
    """Raised when an alignment file is not found."""

    def __init__(self, filename: str) -> None:
        """Initialize AlignmentFileNotFoundError.

        Args:
            filename (str): The name of the alignment file that was not found.
        """
        super().__init__(f"Alignment file '{filename}' not found.")


class NoValidAllelesError(Exception):
    """Custom exception raised when no valid alleles are found in a SNP column."""

    def __init__(self, column_index: int) -> None:
        """Initialize NoValidAllelesError.

        Args:
            column_index (int): The index of the column where no valid alleles were found.
        """
        super().__init__(f"No valid alleles found in column {column_index}")
        self.column_index = column_index
