import functools
import logging
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil


def format_warning(message, category, filename, lineno) -> str:
    """
    Set the format of warnings.warn warnings.

    This method defines the format of warning messages printed by the warnings module when using `warnings.warn()`.

    Args:
        message (str): Warning message to be printed.

        category (str): Type of warning. See Python stdlib warnings module.

        filename (str): Name of the Python file where the warning was raised.

        lineno (str): Line number where the warning occurred.

    Returns:
        str: Formatted warning message.

    Examples:
        >>>warnings.formatwarning = format_warning

    Note:
        To set the format of warnings, use `warnings.formatwarning = self._format_warning`.
    """
    return f"{filename}:{lineno}: {category.__name__}:{message}"


def get_gt2iupac():
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


def get_iupac2gt():
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


def get_int_iupac_dict():
    int_iupac_dict = {
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

    return int_iupac_dict


def get_onehot_dict():
    onehot_dict = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "T": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "C": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 0.0],
        "W": [0.5, 0.5, 0.0, 0.0],
        "R": [0.5, 0.0, 0.5, 0.0],
        "M": [0.5, 0.0, 0.0, 0.5],
        "K": [0.0, 0.5, 0.5, 0.0],
        "Y": [0.0, 0.5, 0.0, 0.5],
        "S": [0.0, 0.0, 0.5, 0.5],
        "N": [0.0, 0.0, 0.0, 0.0],
    }

    return onehot_dict


def validate_input_type(X, return_type="array"):
    """
    Validates the input type and returns it as a specified type.

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
        X = [[1, 2, 3], [4, 5, 6]]
        print(validate_input_type(X, "df"))  # Outputs: a DataFrame with the data from `X`.
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
