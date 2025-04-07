import functools
import json
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


class ResultsExporter:
    """Handles exporting statistical results to JSON and/or CSV files.

    This class can save results to JSON or CSV files. If a dictionary contains nested sub-dictionaries or DataFrames, it properly separates them, ensuring JSON remains valid and DataFrames are saved as separate CSVs.
    """

    def __init__(self, output_dir: str | Path = "snpio_output"):
        """Initialize the ResultsExporter object.

        This class can save results to JSON or CSV files. If a dictionary contains nested sub-dictionaries or DataFrames, it properly separates them, ensuring JSON remains valid and DataFrames are saved as separate CSVs.

        Args:
            output_dir (str | Path): Directory where results will be saved.
        """
        self._output_dir = (
            Path(output_dir, "analysis")
            if not isinstance(output_dir, Path)
            else output_dir / "analysis"
        )

    def _flatten_and_save(self, data: Dict, filename_prefix: str) -> dict:
        """Recursively processes nested dictionaries, saving DataFrames as CSVs while preserving non-DataFrame data for JSON export.

        Args:
            data (Dict): Dictionary containing mixed types (DataFrames, dicts, scalars).
            filename_prefix (str): Base name for output files.

        Returns:
            dict: Processed data suitable for JSON (excluding DataFrames).
        """
        json_data = {}  # Holds non-DataFrame values for JSON export

        for key, value in data.items():
            # Convert tuple keys to strings (JSON does not allow tuple keys)
            if isinstance(key, tuple):
                # Convert (pop1, pop2) → "pop1_pop2"
                key = "_".join(map(str, key))

            # Construct hierarchical filenames
            full_key = f"{filename_prefix}_{key}"

            if isinstance(value, (pd.DataFrame, pd.Series)):
                # Save DataFrame or Series as CSV
                value.to_csv(self.output_dir / f"{full_key}.csv", index=False)

            elif isinstance(value, np.ndarray):
                # Save numpy array as CSV
                pd.DataFrame(value).to_csv(
                    self.output_dir / f"{full_key}.csv", index=False
                )

            elif isinstance(value, dict):
                # Recursively process sub-dictionaries
                nested_json_data = self._flatten_and_save(value, full_key)

                if isinstance(nested_json_data, np.ndarray):
                    # Save numpy array as CSV
                    pd.DataFrame(nested_json_data).to_csv(
                        self.output_dir / f"{full_key}.csv", index=False
                    )

                if nested_json_data:  # Avoid empty dicts in JSON
                    json_data[key] = nested_json_data

            else:
                # Store non-DataFrame values for JSON export
                json_data[key] = value

        return json_data

    def save_results(self, data: Any, filename: str) -> None:
        """Save results to an appropriate format (JSON or CSV).

        Args:
            data (Any): Data to be saved (DataFrame, dictionary, list of tuples, etc.).
            filename (str): Base filename without extension.
        """
        file_path_json = self.output_dir / f"{filename}.json"
        file_path_csv = self.output_dir / f"{filename}.csv"

        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Save DataFrame or Series as CSV
            data.to_csv(file_path_csv, index=False)

        elif isinstance(data, dict):
            # Recursively process nested dictionaries, separating DataFrames
            json_content = self._flatten_and_save(data, filename)

            if json_content:
                with open(file_path_json, "w") as f:
                    json.dump(json_content, f, indent=4)

        elif isinstance(data, list) and all(
            isinstance(item, (tuple, list)) for item in data
        ):
            # Convert list of tuples/lists to DataFrame
            df = pd.DataFrame(data)
            df.to_csv(file_path_csv, index=False)

        else:
            # Generic JSON export for any other data types
            with open(file_path_json, "w") as f:
                json.dump(data, f, indent=4)

    def capture_results(self, func: Callable) -> Callable:
        """Decorator to capture function results and save them to files.

        Args:
            func (Callable): The function whose results should be captured.

        Returns:
            Callable: Wrapped function that automatically saves its output.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            filename = func.__name__

            if isinstance(result, tuple):
                if "detect_fst_outliers" in filename:
                    part_names = ("fst", "pvalues")
                elif "neis_genetic_distance" in filename:
                    part_names = ("observed", "pvalues")
                elif "calculate_d_statistics" in filename:
                    part_names = ("per_sample_dstats", "summarized_dstats")
                elif "tajimas_d" in filename:
                    part_names = ("tajimas_d", "pvalues")
                else:
                    part_names = None

                for idx, res in enumerate(result):
                    if part_names is not None:
                        self.save_results(res, f"{filename}_{part_names[idx]}")
                    else:
                        self.save_results(res, f"{filename}_part{idx+1}")
            else:
                self.save_results(result, filename)

            return result

        return wrapper

    @property
    def output_dir(self) -> Path:
        """Returns the directory where results are saved."""

        if not isinstance(self._output_dir, Path):
            self._output_dir = Path(self._output_dir)

        if not self._output_dir.name == "analysis":
            self._output_dir = self._output_dir / "analysis"

        self._output_dir.mkdir(parents=True, exist_ok=True)

        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str | Path) -> None:
        """Sets the directory where results will be saved.

        Args:
            value (str | Path): Directory where results will be saved.
        """

        if not isinstance(value, Path):
            value = Path(value)

        if not value.name == "analysis":
            value = value / "analysis"

        value.mkdir(parents=True, exist_ok=True)

        self._output_dir = value
