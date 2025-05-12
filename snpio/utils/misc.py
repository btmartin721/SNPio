import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from snpio.utils.logging import LoggerManager


class IUPAC:
    """A class to store IUPAC ambiguity data and provide helper functions.

    This class stores IUPAC ambiguity data and provides helper functions to convert between genotype and IUPAC ambiguity codes.

    Attributes:
        gt2iupac (Dict[str, str]): A dictionary of genotype to IUPAC ambiguity codes.

        iupac2gt (Dict[str, str]): A dictionary of IUPAC ambiguity codes to genotype.

        int_iupac_dict (Dict[str, int]): A dictionary of IUPAC ambiguity codes to integers.

        onehot_dict (Dict[str, List[float]]): A dictionary of IUPAC ambiguity codes to one-hot encoded vectors.

        ambiguous_dna_values (Dict[str, str]): A dictionary of IUPAC ambiguity values.

        multilab_value (float): The value to use for multilabel data.

        logger (logging.Logger | None): A logger object for logging messages.

    Example:
        >>> iupac_data = IUPAC()
        >>> print(iupac_data.gt2iupac)
        >>> # Outputs: {'1/1': 'A', '2/2': 'C', '3/3': 'G', '4/4': 'T', '1/2': 'M', '1/3': 'R', '1/4': 'W', '2/3': 'S', '2/4': 'Y', '3/4': 'K', '-9/-9': 'N', '-1/-1': 'N'}
    """

    def __init__(
        self, multilab_value: float = 1.0, logger: logging.Logger | None = None
    ) -> None:
        """Initialize the IUPACData class.

        Args:
            multilab_value (float, optional): The value to use for multilabel data. Defaults to 1.0.

            logger (logging.Logger | None): A logger object for logging messages. Defaults to None.

        Raises:
            ValueError: If `multilab_value` is not between 0 and 1.
        """

        self.multilab_value: float = multilab_value
        self.gt2iupac: Dict[str, str] = self.get_gt2iupac()
        self.iupac2gt: Dict[str, str] = self.get_iupac2gt()
        self.int_iupac_dict: Dict[str, int] = self.get_int_iupac_dict()
        self.onehot_dict: Dict[str, List[float]] = self.get_onehot_dict()
        self.ambiguous_dna_values: Dict[str, str] = self.get_iupac_ambiguous()

        if logger is not None:
            self.logger = logger
        else:
            logman = LoggerManager(__name__, prefix="snpio", verbose=False, debug=False)
            self.logger: logging.Logger = logman.get_logger()

        if multilab_value <= 0 or multilab_value > 1:
            msg: str = (
                f"Invalid multilabel value: {multilab_value}. Must be > 0 and <= 1."
            )
            self.logger.error(msg)
            raise ValueError(msg)

    def get_iupac_int_map(self) -> Dict[str, int]:
        """Get a dictionary of IUPAC ambiguity codes to integers.

        This method returns a dictionary of IUPAC ambiguity codes to integers. The keys are the IUPAC ambiguity codes and the values are the corresponding integers. The integer values are used to encode the IUPAC ambiguity codes. The integer values are assigned as follows: A=1, C=2, G=3, T=4, R=5, Y=6, S=7, W=8, K=9, M=10, B=11, D=12, H=13, V=14, N=15, -=16, ?=16, .=16.

        Returns:
            Dict[str, int]: A dictionary of IUPAC ambiguity codes to integers.

        Example:
            >>> iupac_data = IUPAC()
            >>> print(iupac_data.get_iupac_int_map())
            >>> # Outputs: {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'R': 5, 'Y': 6, 'S': 7, 'W': 8, 'K': 9, 'M': 10, 'B': 11, 'D': 12, 'H': 13, 'V': 14, 'N': 15, '-': 16, '?': 16, '.': 16}
        """
        return {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "R": 5,
            "Y": 6,
            "S": 7,
            "W": 8,
            "K": 9,
            "M": 10,
            "B": 11,
            "D": 12,
            "H": 13,
            "V": 14,
            "N": 15,
            "-": 16,
            "?": 16,
            ".": 16,
        }

    def get_iupac_ambiguous(self) -> Dict[str, str]:
        """Get a dictionary of IUPAC ambiguity values.

        This dictionary contains the IUPAC ambiguity values for DNA sequences. The keys are the IUPAC ambiguity codes and the values are the corresponding DNA bases or ambiguity codes as strings.

        Returns:
            Dict[str, str]: A dictionary of IUPAC ambiguity values.


        Example:
            >>> iupac_data = IUPAC()
            >>> print(iupac_data.ambiguous_dna_values)
            >>> # Outputs: {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'M': 'AC', 'R': 'AG', 'W': 'AT', 'S': 'CG', 'Y': 'CT', 'K': 'GT', 'V': 'ACG', 'H': 'ACT', 'D': 'AGT', 'B': 'CGT', 'X': 'GATC', 'N': 'GATC'}
        """
        return {
            "A": "A",
            "C": "C",
            "G": "G",
            "T": "T",
            "M": "AC",
            "R": "AG",
            "W": "AT",
            "S": "CG",
            "Y": "CT",
            "K": "GT",
            "V": "ACG",
            "H": "ACT",
            "D": "AGT",
            "B": "CGT",
            "X": "GATC",
            "N": "GATC",
        }

    def get_gt2iupac_0based(self) -> Dict[str, str]:
        """Get a dictionary of genotype to IUPAC ambiguity codes.

        This method returns a dictionary of genotype to IUPAC ambiguity codes. The keys are the genotypes and the values are the corresponding IUPAC ambiguity codes. The genotype values are 0-based, meaning that the first genotype is represented by 0.

        Returns:
            Dict[str, str]: A dictionary of genotype to IUPAC ambiguity codes.

        """
        return {
            "0/0": "A",
            "1/1": "C",
            "2/2": "G",
            "3/3": "T",
            "0/1": "M",  # A/C
            "1/0": "M",
            "0/2": "R",  # A/G
            "2/0": "R",
            "0/3": "W",  # A/T
            "3/0": "W",
            "1/2": "S",  # C/G
            "2/1": "S",
            "1/3": "Y",  # C/T
            "3/1": "Y",
            "2/3": "K",  # G/T
            "3/2": "K",
            "-9/-9": "N",  # Missing data
            "-1/-1": "N",
        }

    def get_gt2iupac(self) -> Dict[str, str]:
        """Get a dictionary of genotype to IUPAC ambiguity codes."

        Returns:
            Dict[str, str]: A dictionary of genotype to IUPAC ambiguity codes.
        """
        return {
            "1/1": "A",
            "2/2": "C",
            "3/3": "G",
            "4/4": "T",
            "1/2": "M",  # A/C
            "1/3": "R",  # A/G
            "1/4": "W",  # A/T
            "2/3": "S",  # C/G
            "2/4": "Y",  # C/T
            "3/4": "K",  # G/T
            "-9/-9": "N",  # Missing data
            "-1/-1": "N",  # Missing data
        }

    def get_iupac2gt(self) -> Dict[str, str]:
        """Get a dictionary of IUPAC ambiguity codes to genotype.

        Returns:
            Dict[str, str]: A dictionary of IUPAC ambiguity codes to genotype.
        """
        return {
            "A": "1/1",
            "C": "2/2",
            "G": "3/3",
            "T": "4/4",
            "M": "1/2",  # A/C
            "R": "1/3",  # A/G
            "W": "1/4",  # A/T
            "S": "2/3",  # C/G
            "Y": "2/4",  # C/T
            "K": "3/4",  # G/T
            "N": "-9/-9",  # Missing data
        }

    def get_int_iupac_dict(self) -> Dict[str, int]:
        """Get a dictionary of IUPAC ambiguity codes to integers.

        This dictionary is used to convert IUPAC ambiguity codes to integers. It differs from the `get_iupac_int_map` function in that it only includes the main IUPAC codes and not the extended codes.

        Returns:
            Dict[str, int]: A dictionary of IUPAC ambiguity codes to integers.

        Example:
            >>> iupac_data = IUPAC()
            >>> print(iupac_data.int_iupac_dict)
            >>> # Outputs: {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'W': 4, 'R': 5, 'M': 6, 'K': 7, 'Y': 8, 'S': 9, 'N': -9}
        """
        return {
            "A": 0,
            "T": 1,
            "G": 2,
            "C": 3,
            "W": 4,
            "R": 5,
            "M": 6,
            "K": 7,
            "Y": 8,
            "S": 9,
            "N": -9,
        }

    def get_onehot_dict(self) -> Dict[str, List[float]]:
        """Get a dictionary of IUPAC ambiguity codes to one-hot encoded vectors.

        Returns:
            Dict[str, List[float]]: A dictionary of IUPAC ambiguity codes to one-hot encoded vectors
        """
        val: float = self.multilab_value

        return {
            "A": [1.0, 0.0, 0.0, 0.0],
            "T": [0.0, 1.0, 0.0, 0.0],
            "G": [0.0, 0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 0.0, 1.0],
            "W": [val, val, 0.0, 0.0],
            "R": [val, 0.0, val, 0.0],
            "M": [val, 0.0, 0.0, val],
            "K": [0.0, val, val, 0.0],
            "Y": [0.0, val, 0.0, val],
            "S": [0.0, 0.0, val, val],
            "N": [0.0, 0.0, 0.0, 0.0],
        }

    def __getitem__(self, key: str) -> str | List[float]:
        """Retrieve attribute by dictionary-like access.

        Args:
            key (str): The attribute to retrieve.

        Returns:
            str | List[float]: The attribute value.
        """
        if hasattr(self, key):
            return getattr(self, key)
        msg: str = f"{key} is not a valid attribute of IUPAC."
        self.logger.error(msg)
        raise KeyError(msg)

    def __setitem__(self, key: str, value: str | List[float]) -> None:
        """Set attribute by dictionary-like access.

        Args:
            key (str): The attribute to set.
            value (str | List[float]): The value to set the attribute to.

        Raises:
            KeyError: If the attribute is not a valid attribute of IUPAC.

        Example:
            >>> iupac_data = IUPAC()
            >>> iupac_data["multilab_value"] = 0.5
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            msg: str = f"{key} is not a valid attribute of IUPAC."
            self.logger.error(msg)
            raise KeyError(msg)

    def __repr__(self) -> str:
        """Unambiguous summary for debugging.

        Returns:
            str: A string representation of the IUPAC class.
        """
        return f"IUPAC(gt2iupac={self.gt2iupac}, iupac2gt={self.iupac2gt}, int_iupac_dict={self.int_iupac_dict}, onehot_dict={self.onehot_dict}, ambiguous_dna_values={self.ambiguous_dna_values})"

    def __str__(self) -> str:
        """User-friendly representation of the IUPAC dictionaries.

        Returns:
            str: A string describing the IUPAC class.
        """
        return f"IUPAC class with dictionaries for genotype and ambiguity data."

    def __contains__(self, key: str) -> bool:
        """Check if the attribute exists for dictionary-like access.

        Args:
            key (str): The attribute to check for.

        Returns:
            bool: True if the attribute exists, False otherwise.
        """
        return hasattr(self, key)

    def __len__(self) -> List[int]:
        """Return the number of main data dictionaries.

        Returns:
            List[int]: The number of main data dictionaries.
        """
        return len(
            [attr for attr in vars(self) if isinstance(getattr(self, attr), dict)]
        )

    def __iter__(self):
        """Iterate over main dictionary attributes.

        Returns:
            Generator[str, Dict[str, Dict[str, str | List[float]]]]: A generator of main dictionary attributes
        """
        for attr, value in vars(self).items():
            if isinstance(value, dict):
                yield attr, value


def validate_input_type(
    X: np.ndarray | pd.DataFrame | List[List[int]], return_type: str = "array"
) -> np.ndarray | pd.DataFrame | List[List[int]]:
    """Validates the input type and returns it as a specified type.

    This function checks if the input `X` is a pandas DataFrame, numpy array, or a list of lists. It then converts `X` to the specified `return_type` and returns it.

    Args:
        X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): The input data to validate and convert.

        return_type (str, optional): The type of the returned object. Supported options include: "df" (DataFrame), "array" (numpy array), and "list". Defaults to "array".

    Returns:
        pandas.DataFrame, numpy.ndarray, or List[List[int]]: The input data converted to the desired return type.

    Raises:
        TypeError: If `X` is not of type pandas.DataFrame, numpy.ndarray, or List[List[int]].

        ValueError: If an unsupported `return_type` is provided. Supported types are "df", "array", and "list".

    Example:
        >>> X = [[1, 2, 3], [4, 5, 6]]
        >>> print(validate_input_type(X, "df"))  4
        >>> # Outputs: a DataFrame with the data from `X`.
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray, list)):
        raise TypeError(
            f"X must be of type pandas.DataFrame, numpy.ndarray, "
            f"or List[List[int]], but got {type(X)}"
        )

    if return_type == "array":
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        elif isinstance(X, list):
            return np.array(X)
        elif isinstance(X, np.ndarray):
            return X.copy()

    elif return_type == "df":
        if isinstance(X, pd.DataFrame):
            return X.copy()
        elif isinstance(X, (np.ndarray, list)):
            return pd.DataFrame(X)

    elif return_type == "list":
        if isinstance(X, list):
            return X
        elif isinstance(X, np.ndarray):
            return X.tolist()
        elif isinstance(X, pd.DataFrame):
            return X.values.tolist()

    else:
        raise ValueError(
            f"Unsupported return type provided: {return_type}. Supported types "
            f"are 'df', 'array', and 'list'"
        )
