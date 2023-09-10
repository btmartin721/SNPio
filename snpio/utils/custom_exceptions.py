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
                Defaults to ["vcf", "phylip", "structure", "auto"].
        """
        if supported_types is None:
            supported_types = ["vcf", "phylip", "structure", "auto"]

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
