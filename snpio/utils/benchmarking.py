import functools
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from snpio.plotting.plotting import Plotting

try:
    import psutil
    from memory_profiler import memory_usage

    MEMORY_PROFILING_AVAILABLE = True

except (ImportError, ModuleNotFoundError):
    MEMORY_PROFILING_AVAILABLE = False


class Benchmark:
    @staticmethod
    def plot_performance(
        genotype_data: Any,
        resource_data: Dict[str, Any],
        color: str = "#8C56E3",
        figsize: Tuple[int, int] = (18, 10),
    ) -> None:
        """Plots the performance metrics: CPU Load, Memory Footprint, and Execution Time.

        Takes a dictionary of performance data and plots the metrics for each of the methods. The resulting plot is saved in a .png file in the ``<prefix_output/gtdata/plots/performance`` directory.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object to use.

            resource_data (Dict[str, Any]): Dictionary with performance data. Keys are method names, and values are dictionaries with keys 'cpu_load', 'memory_footprint', and 'execution_time'.

            color (str, optional): Color to be used in the plot. Should be a valid color string. Defaults to "#8C56E3".

            figsize (Tuple[int, int], optional): Size of the figure. Should be a tuple of two integers. Defaults to (18, 10).

        Returns:
            None. The function saves the plot to a file.

        Note:
            The plot is saved in the ``<prefix_output/gtdata/plots/performance`` directory.

            The plot is saved in `genotype_data.plot_format` format.
        """
        plot_dir = Path(f"{genotype_data.prefix}_output")
        plot_dir = plot_dir / "gtdata" / "plots" / "performance"
        plot_dir.mkdir(exist_ok=True, parents=True)

        plotting = Plotting(genotype_data=genotype_data, **genotype_data.plot_kwargs)

        plotting.plot_performance(resource_data, color=color, figsize=figsize)

    @staticmethod
    def measure_execution_time(func: Callable) -> Callable:
        """Decorator to measure the execution time of a function.

        This method is a decorator that measures the execution time of a function and adds the performance data to the resource_data dictionary of the object that the function is called on. The decorator also measures the CPU load and memory footprint of the function.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            Callable: The decorated function. The wrapper function measures the execution time of the decorated function.
        """
        if not MEMORY_PROFILING_AVAILABLE:
            # If memory profiler is not available, return the original function
            # without modification
            return func

        @functools.wraps(func)
        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
            """Wrapper function to measure the execution time of the decorated function.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Any: The result of the decorated function.
            """
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
