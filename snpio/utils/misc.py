import logging
from collections.abc import Mapping, Sequence
from itertools import permutations
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
        if logger is None:
            msg = "Logger is a required argument for IUPAC class."
            self.logger.error(msg)
            raise TypeError(msg)

        self.multilab_value: float = multilab_value
        self.gt2iupac: Dict[str, str] = self.get_gt2iupac()
        self.iupac2gt: Dict[str, str] = self.get_iupac2gt()
        self.int_iupac_dict: Dict[str, int] = self.get_int_iupac_dict()
        self.onehot_dict: Dict[str, List[float]] = self.get_onehot_dict()
        self.ambiguous_dna_values: Dict[str, str] = self.get_iupac_ambiguous()

        self.logger = logger

        if multilab_value <= 0 or multilab_value > 1:
            msg: str = (
                f"Invalid multilabel value: {multilab_value}. Must be > 0 and <= 1."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # IUPAC ambiguity codes for unordered heterozygous pairs
        self.iupac_ambiguity = {
            frozenset(["A", "C"]): "M",
            frozenset(["A", "G"]): "R",
            frozenset(["A", "T"]): "W",
            frozenset(["C", "G"]): "S",
            frozenset(["C", "T"]): "Y",
            frozenset(["G", "T"]): "K",
        }

    def get_phased_encoding(self) -> Dict[str, str]:
        """Get a dictionary of phased encoding for IUPAC codes.

        This method returns a dictionary that maps IUPAC codes to their phased encoding. The phased encoding is used to represent the alleles in a phased manner, where each allele is represented as "A/A", "T/T", etc.

        Returns:
            Dict[str, str]: A dictionary mapping IUPAC codes to their phased encoding.
        """
        phased_encoding = {
            "A": "A/A",
            "T": "T/T",
            "C": "C/C",
            "G": "G/G",
            "N": "N/N",
            ".": "N/N",
            "?": "N/N",
            "-": "N/N",
            "W": "A/T",
            "S": "C/G",
            "Y": "C/T",
            "R": "A/G",
            "K": "G/T",
            "M": "A/C",
            "H": "A/C",
            "B": "A/G",
            "D": "C/T",
            "V": "A/G",
        }

        return phased_encoding

    def get_tuple_to_iupac(self) -> dict[tuple[str, str], str]:
        """Return a dictionary mapping base tuples to IUPAC codes.

        The IUPAC codes are used to represent ambiguous bases in DNA sequences.

        Returns:
            dict[tuple[str, str], str]: A dictionary mapping base tuples to IUPAC codes.
        """
        iupac_base_map = {
            frozenset(["A", "G"]): "R",
            frozenset(["C", "T"]): "Y",
            frozenset(["G", "C"]): "S",
            frozenset(["A", "C"]): "M",
            frozenset(["G", "T"]): "K",
            frozenset(["A", "T"]): "W",
            frozenset(["A", "C", "G"]): "V",
            frozenset(["A", "C", "T"]): "H",
            frozenset(["A", "G", "T"]): "D",
            frozenset(["C", "G", "T"]): "B",
            frozenset(["A", "C", "G", "T"]): "N",
            frozenset(["A"]): "A",
            frozenset(["C"]): "C",
            frozenset(["G"]): "G",
            frozenset(["T"]): "T",
            frozenset(["N"]): "N",
        }

        return {
            perm: code
            for bases, code in iupac_base_map.items()
            for perm in permutations(bases, len(bases))
        }

    def encode_genepop_pair(
        self, allele1: str, allele2: str, allele_map: dict[str, str]
    ) -> str:
        """Convert a pair of allele codes to IUPAC code.

        Args:
            allele1 (str): Encoded allele (e.g., "01", "001").
            allele2 (str): Encoded allele (e.g., "02", "003").
            allele_map (dict[str, str]): Mapping of allele strings to bases (e.g., "01" → "A").

        Returns:
            str: IUPAC code (A, C, G, T, M, R, W, S, Y, K) or "N" if invalid/missing.
        """
        base1 = allele_map.get(allele1, "N")
        base2 = allele_map.get(allele2, "N")

        if base1 == "N" or base2 == "N":
            if hasattr(self, "logger"):
                self.logger.debug(f"Unknown alleles: {allele1}/{allele2} → N")
            return "N"

        if base1 == base2:
            return base1

        return self.iupac_ambiguity.get(frozenset([base1, base2]), "N")

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

        This dictionary contains the IUPAC ambiguity values for DNA sequences. The keys are the IUPAC ambiguity codes and the values are the corresponding DNA bases or ambiguity codes as strings with no delimiter.

        Returns:
            Dict[str, str]: A dictionary of IUPAC ambiguity values with no delimiter.


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

    def get_gt2iupac(self) -> Dict[str, str]:
        """Get a dictionary of genotype to IUPAC ambiguity codes.

        This method returns a dictionary of genotype to IUPAC ambiguity codes. The keys are the genotypes in the format "1/1", "2/2", etc., and the values are the corresponding IUPAC ambiguity codes. The IUPAC ambiguity codes are used to represent ambiguous bases in DNA sequences.

        Example:
            >>> iupac_data = IUPAC()
            >>> print(iupac_data.gt2iupac)
            >>> # Outputs: {'1/1': 'A', '2/2': 'C', '3/3': 'G', '4/4': 'T', '1/2': 'M', '1/3': 'R', '1/4': 'W', '2/3': 'S', '2/4': 'Y', '3/4': 'K', '-9/-9': 'N', '-1/-1': 'N'}

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

        This method returns a dictionary of IUPAC ambiguity codes to genotype. The keys are the IUPAC ambiguity codes and the values are the corresponding genotypes in the format "1/1", "2/2", etc. The IUPAC ambiguity codes are used to represent ambiguous bases in DNA sequences.

        Example:
            >>> iupac_data = IUPAC()
            >>> print(iupac_data.iupac2gt)
            >>> # Outputs: {'A': '1/1', 'C': '2/2', 'G': '3/3', 'T': '4/4', 'M': '1/2', 'R': '1/3', 'W': '1/4', 'S': '2/3', 'Y': '2/4', 'K': '3/4', 'N': '-9/-9'}

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

        The integer values are assigned as follows: `A=0, T=1, G=2, C=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9`.

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

        This method returns a dictionary where each key is an IUPAC ambiguity code and the value is a one-hot encoded vector representing that code. The one-hot encoding is a list of floats where the index corresponding to the IUPAC code is set to `multilab_value` and all other indices are set to 0.0.

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


def build_dataframe(  # feel free to make this @staticmethod inside your class
    data: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    index_col: str | None = None,
    transpose: bool = False,
    index_labels: Sequence[str] | None = None,
    index_name: str | None = None,
    default_index_prefix: str = "row_",
    column_renames: Mapping[str, str] | None = None,
    column_order: Sequence[str] | None = None,
    columns_name: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build a flexibly named DataFrame from heterogeneous input.

    This function can ingest a variety of data formats and produce a well-formed
    DataFrame with appropriate labels and structure.

    Args:
        data: An existing DataFrame **or** an iterable of mapping objects
            (e.g. the ``"data"`` field you get from a JSON export).
        index_col: Optional column to promote to the index **before** transposition.
            Set to ``None`` to keep the default integer index.
        transpose: If ``True``, perform ``df.T`` *after* any index-column promotion.
        index_labels: A sequence of labels to use **after** transposition (or not),
            overriding whatever index is present at that point.
        index_name: Name to attach to the index (appears in HTML/CSV and
            downstream plots). Ignored when ``index_labels`` is ``None``.
        default_index_prefix: Prefix to use when we fall back to a synthetic index
            because the supplied ``index_labels`` length mismatches the data.
        column_renames: Optional mapping of ``old_name -> new_name`` applied with
            ``df.rename`` *after* all other operations.
        column_order: Optional explicit order for columns (ignored if ``None``).
        columns_name: Name to set on ``df.columns`` (useful for MultiQC tables,
            plot axes, etc.).
        logger: Logger for warning messages. Defaults to ``logging.getLogger(__name__)``.

    Returns:
        A well-formed, fully-labelled ``pd.DataFrame`` ready for plotting, saving, or feeding into downstream pipelines.

    Notes:
        - Any mismatch between ``index_labels`` and the resulting DataFrame shape is logged **once** and falls back to an autogenerated index (``{default_index_prefix}{i}``).
        - Column renaming happens *after* ordering so your ``column_order`` may use either the original or new names -- whichever is clearer for the use case.

    Example:
        >>> df_fst = build_dataframe(
        ...     json_fst_locus["data"],
        ...     index_col="Population",
        ...     transpose=True,
        ...     index_labels=self.genotype_data.marker_names,
        ...     index_name="Locus (CHROM:POS)",
        ...     column_renames=None,
        ... )
    """
    log = logger or logging.getLogger(__name__)

    # 1. Ingest
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()

    # 2. Promote column → index
    if index_col is not None:
        if index_col not in df.columns:
            raise KeyError(
                f"index_col '{index_col}' not found in columns: {list(df.columns)}"
            )
        df = df.set_index(index_col)

    # 3. Transpose if requested
    if transpose:
        df = df.T

    # 4. Apply user-supplied index labels
    if index_labels is not None:
        if len(index_labels) != len(df):
            log.warning(
                "Length mismatch between supplied index_labels (%d) and DataFrame "
                "rows (%d); using synthetic index '%s{N}'.",
                len(index_labels),
                len(df),
                default_index_prefix,
            )
            df.index = [f"{default_index_prefix}{i}" for i in range(len(df))]
        else:
            df.index = list(index_labels)

    # 5. Set axis names
    if index_name is not None:
        df.index.name = index_name
    if columns_name is not None:
        df.columns.name = columns_name

    # 6. Rename & re-order columns
    if column_renames:
        df = df.rename(columns=column_renames)
    if column_order:
        missing = set(column_order) - set(df.columns)
        if missing:
            raise KeyError(f"column_order contains missing columns: {missing}")
        df = df.loc[:, column_order]

    return df
