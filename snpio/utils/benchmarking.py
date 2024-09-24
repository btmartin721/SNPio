import functools
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import psutil

from snpio.plotting.plotting import Plotting

import functools
import time
import psutil  # For CPU and memory monitoring
from memory_profiler import memory_usage  # To measure memory footprint


def measure_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Monitor start time, CPU, and memory usage before execution
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = memory_usage(-1, interval=0.1, timeout=1)

        result = func(*args, **kwargs)

        # Monitor end time, CPU, and memory usage after execution
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = memory_usage(-1, interval=0.1, timeout=1)

        # Calculate performance metrics
        execution_time = end_time - start_time
        avg_cpu_load = (start_cpu + end_cpu) / 2
        memory_footprint = max(end_memory) - min(start_memory)

        # Log the performance metrics
        logger = args[0].logger if hasattr(args[0], "logger") else None
        if logger:
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds.")
            logger.info(f"Average CPU Load: {avg_cpu_load:.2f}%")
            logger.info(f"Memory Footprint: {memory_footprint:.2f} MB")

        # Add the performance data to the resource_data dictionary
        if hasattr(args[0], "resource_data"):
            # Initialize an empty list if the key doesn't exist
            if func.__name__ not in args[0].resource_data:
                args[0].resource_data[func.__name__] = []

            # Append the new performance data to the list
            args[0].resource_data[func.__name__].append(
                {
                    "cpu_load": avg_cpu_load,
                    "memory_footprint": memory_footprint,
                    "execution_time": execution_time,
                }
            )

        return result

    return wrapper


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
        color: str = "#8C56E3",
        figsize: Tuple[int] = (18, 10),
    ) -> None:
        """Plots the performance metrics: CPU Load, Memory Footprint, and Execution Time.

        Takes a dictionary of performance data and plots the metrics for each of the methods. The resulting plot is saved in a .png file in the ``tests`` directory.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object to use.

            resource_data (dict): Dictionary with performance data. Keys are method names, and values are dictionaries with keys 'cpu_load', 'memory_footprint', and 'execution_time'.

            color (str, optional): Color to be used in the plot. Should be a valid color string. Defaults to "#8C56E3".

            figsize (tuple, optional): Size of the figure. Should be a tuple of two integers. Defaults to (16, 9).

        Returns:
            None. The function saves the plot to a file.
        """
        plot_dir = Path(f"{genotype_data.prefix}_output")
        plot_dir = plot_dir / "gtdata" / "plots" / "performance"
        plot_dir.mkdir(exist_ok=True, parents=True)

        plotting = Plotting(genotype_data=genotype_data, **genotype_data.plot_kwargs)

        plotting.plot_performance(resource_data, color=color, figsize=figsize)
