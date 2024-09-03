import functools
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import psutil

from snpio.plotting.plotting import Plotting


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
        self.resource_data[func.__name__]["memory_footprint"] = memory_footprint_mb

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
        cls.resource_data[func.__name__]["memory_footprint"] = memory_footprint_mb

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
    Decorator for applying performance measurement to all callable attributes of a class that do not start with an underscore.

    The performance metrics include CPU load, memory footprint, and execution time.

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
                    decorated_func = measure_performance_for_instance_method(attr_value)
                    setattr(cls, attr_name, decorated_func)
        return cls

    return measure_decorator if measure else lambda cls: cls


class Benchmark:
    @staticmethod
    def plot_performance(
        genotype_data,
        resource_data: Dict[str, Any],
        fontsize: int = 14,
        plot_type="png",
        color: str = "#8C56E3",
        figsize: Tuple[int] = (16, 9),
    ) -> None:
        """Plots the performance metrics: CPU Load, Memory Footprint, and Execution Time.

        Takes a dictionary of performance data and plots the metrics for each of the methods. The resulting plot is saved in a .png file in the ``tests`` directory.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object to use.

            resource_data (dict): Dictionary with performance data. Keys are method names, and values are dictionaries with keys 'cpu_load', 'memory_footprint', and 'execution_time'.

            fontsize (int, optional): Font size to be used in the plot. Defaults to 14.

            plot_type (str): Plot type to use. One of: 'png', 'jpg', or 'pdf'. Defaults to 'png'.

            color (str, optional): Color to be used in the plot. Should be a valid color string. Defaults to "#8C56E3".

            figsize (tuple, optional): Size of the figure. Should be a tuple of two integers. Defaults to (16, 9).

        Returns:
            None. The function saves the plot to a file.
        """
        plot_dir = Path(f"{genotype_data.prefix}_output")
        plot_dir = plot_dir / "gtdata" / "plots" / "performance"
        plot_dir.mkdir(exist_ok=True, parents=True)

        Plotting.plot_performance(
            resource_data,
            fontsize=fontsize,
            color=color,
            figsize=figsize,
            plot_dir=plot_dir,
            plot_type=plot_type,
        )
