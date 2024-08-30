class UnsupportedFileTypeError(Exception):
    """
    Custom exception to check if the given file type is supported.

    Attributes:
        filetype (str): The type of the file being checked.
        message (str): Explanation of the error.
        supported_types (list): List of supported file types.
    """

    def __init__(self, filetype, supported_types=None):
        """
        Initialize UnsupportedFileTypeException.

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

    def __str__(self):
        """
        String representation of the exception.

        Returns:
            str: The message associated with the exception.
        """
        return self.message
    
    
class AlignmentError(Exception):
    """Base class for exceptions related to alignment file handling."""
    pass

class AlignmentFormatError(AlignmentError):
    """Raised when an alignment file has an invalid format."""
    def __init__(self, message):
        super().__init__(message)
        
class SequenceLengthError(AlignmentFormatError):
    """Raised when sequences in the alignment file have unequal lengths."""
    def __init__(self, sample_name):
        super().__init__(f"Sequences have unequal lengths. Error at sample: {sample_name}")
        
class PhylipAlignmentSampleMismatch(AlignmentFormatError):
    """Raised when samples in the alignment file do not match the expected number of samples."""
    def __init__(self, header_samples, sample_list_len, snp_data_len):
        super().__init__(f"Unexpected number of samples encountered. Expected {header_samples}, but got: {sample_list_len} samples across {snp_data_len} genotypes.")
        
class StructureAlignmentSampleMismatch(AlignmentFormatError):
    """Raised when samples in the alignment file do not match the expected number of samples."""
    def __init__(self, sample_list_len, snp_data_len):
        super().__init__(f"Unexpected number of samples encountered. Expected {sample_list_len}, but got: {snp_data_len} genotype rows.")

class AlignmentFileNotFoundError(AlignmentError):
    """Raised when an alignment file is not found."""
    def __init__(self, filename):
        super().__init__(f"Alignment file '{filename}' not found.")

