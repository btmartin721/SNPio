from pathlib import Path
from typing import Any, Dict, Tuple

from snpio.plotting.plotting import Plotting
from snpio.read_input.genotype_data import GenotypeData


class Benchmark:
    @staticmethod
    def plot_performance(genotype_data: GenotypeData, resource_data: Dict[str, Any], fontsize: int=14, plot_type="png", color: str="#8C56E3", figsize: Tuple[int]=(16, 9)) -> None:
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