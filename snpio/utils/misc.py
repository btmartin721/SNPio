import sys
import os
import functools
import psutil
import time
import subprocess
import re
import logging
import platform

import numpy as np
import pandas as pd


# from skopt import BayesSearchCV


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


def generate_random_dataset(
    min_value=0,
    max_value=2,
    nrows=35,
    ncols=20,
    min_missing_rate=0.15,
    max_missing_rate=0.5,
):
    """
    Generate a random integer dataset that can be used for testing.

    This function generates a 2D numpy array of random integers between `min_value` and `max_value` (inclusive). It also adds randomly missing values of random proportions between `min_missing_rate` and `max_missing_rate`.

    Args:
        min_value (int, optional): Minimum value to use. Defaults to 0.
        max_value (int, optional): Maximum value to use. Defaults to 2.
        nrows (int, optional): Number of rows to use. Defaults to 35.
        ncols (int, optional): Number of columns to use. Defaults to 20.
        min_missing_rate (float, optional): Minimum proportion of missing data per column. Defaults to 0.15.
        max_missing_rate (float, optional): Maximum proportion of missing data per column. Defaults to 0.5.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (nrows, ncols) containing the randomly generated dataset.

    Raises:
        AssertionError: If any of the input parameters are out of their expected ranges.

    Example:
        print(generate_random_dataset(0, 5, 5, 5, 0.1, 0.3))
        # Outputs: a 5x5 numpy array with random integers between 0 and 5 and some missing values.
    """
    assert (
        min_missing_rate >= 0 and min_missing_rate < 1.0
    ), f"min_missing_rate must be >= 0 and < 1.0, but got {min_missing_rate}"

    assert (
        max_missing_rate > 0 and max_missing_rate < 1.0
    ), f"max_missing_rate must be > 0 and < 1.0, but got {max_missing_rate}"

    assert nrows > 1, f"nrows must be > 1, but got {nrows}"
    assert ncols > 1, f"ncols must be > 1, but got {ncols}"

    try:
        min_missing_rate = float(min_missing_rate)
        max_missing_rate = float(max_missing_rate)
    except TypeError:
        sys.exit(
            "min_missing_rate and max_missing_rate must be of type float or "
            "must be cast-able to type float"
        )

    X = np.random.randint(
        min_value, max_value + 1, size=(nrows, ncols)
    ).astype(float)
    for i in range(X.shape[1]):
        drop_rate = int(
            np.random.choice(
                np.arange(min_missing_rate, max_missing_rate, 0.02), 1
            )[0]
            * X.shape[0]
        )

        rows = np.random.choice(np.arange(0, X.shape[0]), size=drop_rate)
        X[rows, i] = np.nan

    return X


def generate_012_genotypes(
    nrows=35,
    ncols=20,
    max_missing_rate=0.5,
    min_het_rate=0.001,
    max_het_rate=0.3,
    min_alt_rate=0.001,
    max_alt_rate=0.3,
):
    """
    Generates a 2D numpy array of random 012-encoded genotypes.

    Allows users to control the rate of reference (0's), heterozygote (1's), and alternate alleles (2's). Will insert a random proportion between `min_het_rate` and `max_het_rate` and `min_alt_rate` and `max_alt_rate` and from no missing data to a proportion of `max_missing_rate`.

    Args:
        nrows (int, optional): Number of rows to generate. Defaults to 35.

        ncols (int, optional): Number of columns to generate. Defaults to 20.

        max_missing_rate (float, optional): Maximum proportion of missing data to use. Defaults to 0.5.

        min_het_rate (float, optional): Minimum proportion of heterozygotes (1's) to insert. Defaults to 0.001.

        max_het_rate (float, optional): Maximum proportion of heterozygotes (1's) to insert. Defaults to 0.3.

        min_alt_rate (float, optional): Minimum proportion of alternate alleles (2's) to insert. Defaults to 0.001.

        max_alt_rate (float, optional): Maximum proportion of alternate alleles (2's) to insert. Defaults to 0.3.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (nrows, ncols) containing the generated 012-encoded genotypes.

    Raises:
        AssertionError: If any of the input parameters are out of their expected ranges.

    Example:
        print(generate_012_genotypes(5, 5, 0.2, 0.1, 0.3, 0.1, 0.3))
        # Outputs: a 5x5 numpy array with 012-encoded genotypes.
    """
    assert (
        min_het_rate > 0 and min_het_rate <= 1.0
    ), f"min_het_rate must be > 0 and <= 1.0, but got {min_het_rate}"

    assert (
        max_het_rate > 0 and max_het_rate <= 1.0
    ), f"max_het_rate must be > 0 and <= 1.0, but got {max_het_rate}"

    assert (
        min_alt_rate > 0 and min_alt_rate <= 1.0
    ), f"min_alt_rate must be > 0 and <= 1.0, but got {min_alt_rate}"

    assert (
        max_alt_rate > 0 and max_alt_rate <= 1.0
    ), f"max_alt_rate must be > 0 and <= 1.0, but got {max_alt_rate}"

    assert nrows > 1, f"The number of rows must be > 1, but got {nrows}"

    assert ncols > 1, f"The number of columns must be > 1, but got {ncols}"

    assert (
        max_missing_rate > 0 and max_missing_rate < 1.0
    ), f"max_missing rate must be > 0 and < 1.0, but got {max_missing_rate}"

    try:
        min_het_rate = float(min_het_rate)
        max_het_rate = float(max_het_rate)
        min_alt_rate = float(min_alt_rate)
        max_alt_rate = float(max_alt_rate)
        max_missing_rate = float(max_missing_rate)
    except TypeError:
        sys.exit(
            "max_missing_rate, min_het_rate, max_het_rate, min_alt_rate, and "
            "max_alt_rate must be of type float, or must be cast-able to type "
            "float"
        )

    X = np.zeros((nrows, ncols))
    for i in range(X.shape[1]):
        het_rate = int(
            np.ceil(
                np.random.choice(
                    np.arange(min_het_rate, max_het_rate, 0.02), 1
                )[0]
                * X.shape[0]
            )
        )

        alt_rate = int(
            np.ceil(
                np.random.choice(
                    np.arange(min_alt_rate, max_alt_rate, 0.02), 1
                )[0]
                * X.shape[0]
            )
        )

        het = np.sort(
            np.random.choice(
                np.arange(0, X.shape[0]), size=het_rate, replace=False
            )
        )

        alt = np.sort(
            np.random.choice(
                np.arange(0, X.shape[0]), size=alt_rate, replace=False
            )
        )

        sidx = alt.argsort()
        idx = np.searchsorted(alt, het, sorter=sidx)
        idx[idx == len(alt)] = 0
        het_unique = het[alt[sidx[idx]] != het]

        X[alt, i] = 2
        X[het_unique, i] = 1

        drop_rate = int(
            np.random.choice(np.arange(0.15, max_missing_rate, 0.02), 1)[0]
            * X.shape[0]
        )

        missing = np.random.choice(np.arange(0, X.shape[0]), size=drop_rate)

        X[missing, i] = np.nan

    print(
        f"Created a dataset of shape {X.shape} with {np.isnan(X).sum()} total missing values"
    )

    return X


def unique2D_subarray(a):
    """
    Returns unique subarrays for each column from a 2D numpy array.

    Args:
        a (numpy.ndarray): The 2D numpy array to process.

    Returns:
        numpy.ndarray: A 2D numpy array containing only the unique subarrays from the input array.

    Example:
        a = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        print(unique2D_subarray(a))  # Outputs: [[1, 2, 3], [4, 5, 6]]

    Note:
        The function first views the input array as a 1D array with a custom data type, then uses numpy's unique function to find the unique elements.
    """
    dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    b = np.ascontiguousarray(a.reshape(a.shape[0], -1)).view(dtype1)
    return a[np.unique(b, return_index=1, axis=-1)[1]]


def get_indices(l):
    """
    Takes a list and returns a dictionary where keys are the unique elements in the list and values are lists of indices where each element appears.

    Args:
        l (List[Any]): The list to process.

    Returns:
        dict: A dictionary where keys are the unique elements in the list and values are lists of indices where each element appears.

    Example:
        print(get_indices([0, 1, 1, 0, 0]))  # Outputs: {0: [0, 3, 4], 1: [1, 2]}

    Note:
        The function uses a set to get the unique elements in the list, and then iterates over the list to get the indices.
    """
    ret = dict()
    for member in set(l):
        ret[member] = list()
    i = 0
    for el in l:
        ret[el].append(i)
        i += 1
    return ret


def all_zero(l):
    """
    Checks whether a list consists of all zeros.

    This function returns True if the supplied list contains only zeros (integer, float, or string representations).
    It returns False if the list contains any non-zero values or if the list is empty.

    Args:
        l (List[Union[int, float, str]]): The list to check.

    Returns:
        bool: True if all elements in the list are zeros, False otherwise.

    Example:
        print(all_zero([0, 0.0, '0', '0.0']))  # Outputs: True
        print(all_zero([0, 1, 2]))  # Outputs: False
        print(all_zero([]))  # Outputs: False
    """
    values = set(l)
    if len(values) > 1:
        return False
    elif len(values) == 1 and l[0] in [0, 0.0, "0", "0.0"]:
        return True
    else:
        return False


def weighted_draw(d, num_samples=1):
    """
    Draws samples from a dictionary where keys are choices and values are their corresponding weights.

    Args:
        d (dict): The dictionary from which to draw samples. Keys are the choices and values are the corresponding weights.
        num_samples (int, optional): The number of samples to draw. Defaults to 1.

    Returns:
        numpy.ndarray: An array of drawn samples.

    Example:
        d = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        print(weighted_draw(d, 10))  # Outputs: array(['a', 'b', 'a', 'a', 'b', 'a', 'c', 'a', 'a', 'b'])

    Note:
        The function uses numpy's random.choice function for drawing samples.
    """
    choices = list(d.keys())
    weights = list(d.values())
    return np.random.choice(choices, num_samples, p=weights)


def get_attributes(cls):
    """
    Retrieves the attributes of a class or an instance.

    Args:
        cls (object): The class or instance from which to retrieve attributes.

    Returns:
        dict: A dictionary where the keys are the attribute names and the values are the attribute values. Only includes attributes that do not start with "__" and are not callable.

    Example:
        class MyClass:
            x = 1
            y = 2

        print(get_attributes(MyClass))  # Outputs: {'x': 1, 'y': 2}
    """
    return {
        k: v
        for k, v in cls.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }


def measure_performance_for_instance_method(func):
    """
    Decorator for measuring the performance of an instance method. The performance metrics include CPU load, memory footprint, and execution time.

    Args:
        func (callable): The instance method to be measured.

    Returns:
        callable: The decorated instance method that measures its performance when called.

    Note:
        The performance data is stored in the `resource_data` attribute of the instance, under a key with the name of the method. The data is a dictionary with keys 'cpu_load', 'memory_footprint', and 'execution_time'.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create nested dictionary for function's resource data
        self.resource_data[func.__name__] = {}

        # Measure CPU load
        cpu_load = psutil.cpu_percent()
        self.resource_data[func.__name__]["cpu_load"] = cpu_load

        # Measure memory footprint
        process = psutil.Process(os.getpid())
        memory_footprint = process.memory_info().rss
        memory_footprint_mb = memory_footprint / (
            1024 * 1024
        )  # Convert bytes to megabytes
        self.resource_data[func.__name__][
            "memory_footprint"
        ] = memory_footprint_mb

        # Measure execution time
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        self.resource_data[func.__name__]["execution_time"] = execution_time
        return result

    return wrapper


def measure_performance_for_class_method(func):
    """
    Decorator for measuring the performance of a class method. The performance metrics include CPU load, memory footprint, and execution time.

    Args:
        func (callable): The class method to be measured.

    Returns:
        callable: The decorated class method that measures its performance when called.
    """

    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        # Create nested dictionary for function's resource data
        cls.resource_data[func.__name__] = {}

        # Measure CPU load
        cpu_load = psutil.cpu_percent()
        cls.resource_data[func.__name__]["cpu_load"] = cpu_load

        # Measure memory footprint
        process = psutil.Process(os.getpid())
        memory_footprint = process.memory_info().rss
        memory_footprint_mb = memory_footprint / (
            1024 * 1024
        )  # Convert bytes to megabytes
        cls.resource_data[func.__name__][
            "memory_footprint"
        ] = memory_footprint_mb

        # Measure execution time
        start_time = time.time()
        result = func(cls, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        cls.resource_data[func.__name__]["execution_time"] = execution_time
        return result

    return wrapper


def class_performance_decorator(measure=True):
    """
    Decorator for applying performance measurement to all callable attributes of a class that do not start with an underscore. The performance metrics include CPU load, memory footprint, and execution time.

    Args:
        measure (bool, optional): If True, apply performance measurement. If False, return the class unchanged. Defaults to True.

    Returns:
        callable: The decorated class with performance measurement applied to its callable attributes, or the original class if measure is False.
    """

    def measure_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value) and not attr_name.startswith("_"):
                if isinstance(attr_value, (staticmethod, classmethod)):
                    decorated_func = measure_performance_for_class_method(
                        attr_value.__func__
                    )
                    setattr(cls, attr_name, classmethod(decorated_func))
                else:
                    decorated_func = measure_performance_for_instance_method(
                        attr_value
                    )
                    setattr(cls, attr_name, decorated_func)
        return cls

    return measure_decorator if measure else lambda cls: cls


def progressbar(it, prefix="", size=60, f=sys.stdout):
    """
    Generator that prints a progress bar to the console for an iterable.

    Args:
        it (iterable): The iterable to iterate over.

        prefix (str, optional): The prefix string to be printed before the progress bar. Defaults to an empty string.

        size (int, optional): The total width of the progress bar in characters. Defaults to 60.

        f (file-like object, optional): The file-like object to which the progress bar is printed. Defaults to sys.stdout.

    Yields:
        The next item from the iterable.

    Note:
        The progress bar is printed in the format: "{prefix}[{# * progress}{. * (size - progress)}] {progress}/{total}"
    """
    count = len(it)

    def show(j):
        x = int(size * j / count)
        f.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count)
        )
        f.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    f.write("\n")
    f.flush()


def get_processor_name():
    """
    Retrieves the name of the processor of the system.

    Returns:
        str: The name of the processor. If the system is not recognized (not Windows, Darwin, or Linux), an empty string is returned.

    Note:
        For Windows, it uses the platform.processor() function.

        For Darwin (Mac OS), it returns 'Intel' if the architecture starts with 'i', otherwise it returns the architecture.

        For Linux, it reads from /proc/cpuinfo to get the model name of the processor.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        # os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        arch = platform.processor()
        if arch[0] == "i":
            return "Intel"
        else:
            return arch
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        all_info = all_info.decode("utf-8")
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


class HiddenPrints:
    """
    Context manager for suppressing print statements within its scope.

    This class redirects stdout to os.devnull within its scope, effectively suppressing all print statements. When the scope is exited, stdout is restored to its original state.

    Example:
        with HiddenPrints():
            print("This will not be printed.")

    Attributes:
        _original_stdout (file-like object): The original stdout stream, stored before it is redirected.

    Note:
        This class does not suppress output from sys.stderr.
    """

    def __enter__(self):
        """
        Enter the runtime context. Redirects stdout to os.devnull.

        Returns:
            self
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context. Restores stdout to its original state.

        Args:
            exc_type (Type[BaseException], optional): The type of the exception that caused the context to be exited, if any.
            exc_val (BaseException, optional): The instance of the exception that caused the context to be exited, if any.
            exc_tb (traceback, optional): A traceback object encapsulating the call stack at the point where the exception was raised, if any.

        Returns:
            None
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout


class StreamToLogger(object):
    """
    File-like stream object that redirects writes to a logger instance.

    This class is designed to redirect stdout or stderr to a logging object. It has a write method that splits input on newlines and logs each line separately, and a flush method that logs any remaining input.

    Attributes:
        logger (logging.Logger): The logger instance to which writes are redirected.
        log_level (int, optional): The log level at which messages are logged. Defaults to logging.INFO.
        linebuf (str): A buffer for storing partial lines.

    Example:
        logger = logging.getLogger('my_logger')
        sys.stdout = StreamToLogger(logger, logging.INFO)

    Note:
        This class does not close the logger when it is garbage collected.
    """

    def __init__(self, logger, log_level=logging.INFO):
        """
        Initialize the StreamToLogger instance.

        Args:
            logger (logging.Logger): The logger instance to which writes are redirected.
            log_level (int, optional): The log level at which messages are logged. Defaults to logging.INFO.
        """
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        """
        Write the specified string to the stream.

        Args:
            buf (str): The string to write to the stream.
        """
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        """
        Flush the stream.

        If the stream buffer is not empty, logs its contents and then clears it.
        """
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""
