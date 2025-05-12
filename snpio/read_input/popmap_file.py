import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import snpio.utils.custom_exceptions as exceptions
from snpio.plotting.plotting import Plotting


class ReadPopmap:
    """Class to read and parse a population map file.

    Population map file should contain two comma or whitespace-delimited columns, with one being the SampleIDs and the other being the associated populationIDs. There should either not be a header line in the popmap file, in which case the column order should be sampleIDs and then populationIDs.

    Alternatively, tthe header line should contain exactly one of the accepted sampleID column names ('sampleid' or 'sampleids') and exactly one of the accepted populationID column names ('populationid', 'populationids', 'popid', or 'popids').

    The population map file should not contain any duplicate SampleIDs.

    Population map file format:

        ```
        sampleID,populationID
        Sample1,Population1
        Sample2,Population1
        Sample3,Population2
        Sample4,Population2
        ```

        Example:
            >>> from snpio.read_input.popmap_file import ReadPopmap
            >>> pm = ReadPopmap("popmap.txt", logger, verbose=True, debug=False)
            >>> pm.get_pop_counts(genotype_data)
            >>> pm.validate_popmap(samples, force=True)
            >>> pm.subset_popmap(samples, include=["Population1"])
            >>> pm.write_popmap("subset_popmap.txt")
            >>> print(pm.popmap)
            {'Sample1': 'Population1', 'Sample2': 'Population1', 'Sample3': 'Population2', 'Sample4': 'Population2'}
            >>> print(pm.inverse_popmap):
            {'Population1': ['Sample1', 'Sample2'], 'Population2': ['Sample3', 'Sample4']}

    Attributes:
        filename (str): Filename for the population map.
        verbose (bool): Verbosity setting (True or False). If True, enables verbose output. If False, suppresses verbose output.
        _popdict (Dict[str, str]): Dictionary with SampleIDs as keys and the corresponding population ID as values.
        _sample_indices (np.ndarray): Boolean array representing the subset samples.
        logger (logging): Logger object.

    Methods:
        read_popmap: Read a population map file from disk into a dictionary object.

        write_popmap: Write the population map dictionary to a file.

        get_pop_counts: Print out unique population IDs and their counts.

        validate_popmap: Validate that all alignment sample IDs are present in the population map.

        subset_popmap: Subset the population map based on inclusion and exclusion criteria.

        _infer_delimiter: Infer the delimiter of a given file.

        _infer_header: Infer whether the file has a header.

        _is_numeric: Check if a string can be converted to a float.

        _validate_pop_subset_lists: Validates the elements in the given list to ensure they are all of type str.

        _flip_dictionary: Flip the keys and values of a dictionary.
    """

    def __init__(
        self, filename: str, logger: Any, verbose: bool = False, debug: bool = False
    ) -> None:
        """Initialize the ReadPopmap object.

            This class reads and parses a population map file. The population map file should contain two comma or whitespace-delimited columns, with one being the SampleIDs and the other being the associated populationIDs. There should either not be a header line in the popmap file, in which case the column order should be sampleIDs and then populationIDs. Alternatively, the header line should contain exactly one of the accepted sampleID column names ('sampleid' or 'sampleids') and exactly one of the accepted populationID column names ('populationid', 'populationids', 'popid', or 'popids'). The population map file should not contain any duplicate SampleIDs.

        Args:
            filename (str): Filename for the population map. The population map file to be read and parsed.

            logger (logging): Logger object.

            verbose (bool): Verbosity setting (True or False). If True, enables verbose output. If False, suppresses verbose output.

            debug (bool): Debugging setting (True or False). If True, enables debugging output. If False, suppresses debugging output.

        Notes:
            Initializing the ``ReadPopmap`` object will read the population map file from disk into a dictionary object.

            This class will be used to read and parse a population map file.

            The population map file should contain two comma or whitespace-delimited columns, with one being the SampleIDs and the other being the associated populationIDs.

            There should either not be a header line in the popmap file, in which case the column order should be sampleIDs and then populationIDs.

            Alternatively, the header line should contain exactly one of the accepted sampleID column names ('sampleid' or 'sampleids') and exactly one of the accepted populationID column names ('populationid', 'populationids', 'popid', or 'popids').

            The population map file should not contain any duplicate SampleIDs.

            The dictionary will have SampleIDs as keys and the associated population ID as the values.
        """
        self.filename: str = filename
        self.verbose: bool = verbose
        self._popdict: Dict[str, str] = dict()
        self._sample_indices = None
        self.logger: logging.Logger = logger

        self.read_popmap()

    def read_popmap(self) -> None:
        """Read a population map file from disk into a dictionary object.

        The dictionary will have SampleIDs as keys and the associated population ID as the values. The population map file should contain two comma or whitespace-delimited columns, with one being the SampleIDs and the other being the associated populationIDs. There should either not be a header line in the popmap file, in which case the column order should be sampleIDs and then populationIDs. Alternatively, the header line should contain exactly one of the accepted sampleID column names ('sampleid' or 'sampleids') and exactly one of the accepted populationID column names ('populationid', 'populationids', 'popid', or 'popids'). The population map file should not contain any duplicate SampleIDs.

        Raises:
            PopmapFileNotFoundError: Raises an exception if the population map file is not found on disk.

            PopmapFileFormatError: Raises an exception if the population map file is empty or if the data cannot be correctly loaded from the file.

        Note:
            This method will be executed upon initialization of the ReadPopmap object.

            The population map file should contain two comma or whitespace-delimited columns, with one being the SampleIDs and the other being the associated populationIDs.

            There should either not be a header line in the popmap file, in which case the column order should be sampleIDs and then populationIDs.

            Alternatively, the header line should contain exactly one of the accepted sampleID column names ('sampleid' or 'sampleids') and exactly one of the accepted populationID column names ('populationid', 'populationids', 'popid', or 'popids').

            The population map file should not contain any duplicate SampleIDs.

            The dictionary will have SampleIDs as keys and the associated population ID as the values.
        """
        fn = Path(self.filename)
        if not fn.is_file() or not fn.exists():
            msg = f"Population map file not found: {fn}"
            self.logger.error(msg)
            raise exceptions.PopmapFileNotFoundError(msg)

        delim = self._infer_delimiter(str(fn))
        header = self._infer_header(str(fn), delim)

        if not fn.is_file() and not fn.exists():
            msg = f"Population map file not found: {fn}"
            self.logger.error(msg)
            raise exceptions.PopmapFileNotFoundError(msg)

        # Read the file with inferred delimiter and header
        try:
            df = pd.read_csv(fn, sep=delim, header=header, engine="python")
        except Exception as e:
            msg = f"Error reading the popmap file '{fn}': {e}"
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

        if df.empty or df.size == 0:
            msg = f"Empty or incorrectly loaded popmap file: '{fn}'. Please ensure the popmap file is not empty and is in the correct format."
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

        if len(df.columns) != 2:
            msg = f"The popmap file '{fn}' must have exactly two columns, but found {len(df.columns)} columns."
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

        # Validate columns. Must have exactly two columns or contain the
        # correct headers.
        required_columns = {
            "population",
            "populations",
            "populationid",
            "populationids",
            "popid",
            "popids",
            "pop",
            "pops",
            "sample",
            "samples",
            "sampleid",
            "sampleids",
            "sampid",
            "sampids",
            "samp",
            "samps",
        }

        if all(isinstance(col, int) for col in df.columns):
            df = df.rename(columns={0: "samples", 1: "populations"})
            cols = set(df.columns)
        else:
            cols = set(df.columns.str.lower())

        if required_columns & cols:
            pop = {
                "population",
                "populations",
                "populationid",
                "populationids",
                "popid",
                "popids",
                "pop",
                "pops",
            }
            df = df.rename(columns=lambda x: ("p" if x.lower() in pop else "s"))

        elif df.shape[1] != 2:
            msg = f"The popmap file '{fn}' must have exactly two columns or must contain two of the following header names (one for samples and one for populations): {required_columns}, but found {df.shape[1]} columns without the two required headers."
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

        try:
            df["samples"] = df["samples"].astype(str)
            df["populations"] = df["populations"].astype(str)
            self._popdict = dict(zip(df["samples"], df["populations"]))
        except KeyError as e:
            self._popdict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        self.logger.debug(f"popdict: {self._popdict}")

        if not self._popdict:
            msg = f"Found empty popmap file or could not correctly load data from popmap file: {fn}. Please ensure the popmap file is in the correct format."
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

    def _infer_delimiter(self, file_path: str) -> str:
        """Infer the delimiter of a given file.

        This method reads the first 1024 bytes of the file to infer the delimiter used in the file. It returns ',' for comma-separated files, a whitespace regex pattern character for whitespace-separated files, or raises an error for unsupported delimiters.

        Args:
            file_path (str): Path to the population map file.

        Returns:
            str: The inferred delimiter. Returns ',' for comma-separated files,
                whitespace regex pattern for whitespace-separated files.
                Raises an error for unsupported delimiters.
        """

        with open(file_path, "r") as f:
            sample = f.readline()
            f.seek(0)

        if "," in sample:
            return ","
        elif re.search(r"\s+", sample):
            return r"\s+"
        else:
            msg = f"Unable to infer delimiter for population map file '{file_path}'. Supported delimiters are ',' or whitespace."
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

    def _infer_header(self, file_path: str, delimiter: str) -> int | None:
        """Infer whether the file has a header.

        Args:
            file_path (str): Path to the population map file.
            delimiter (str): The delimiter used in the file.

        Returns:
            int | None: Row number to use as header (0 for first row),
                        or None if there is no header.

        Raises:
            ValueError: Raises an exception if the file has a header with numeric values.
        """
        with open(file_path, "r") as f:
            first_line = f.readline()
            f.seek(0)

            if delimiter == ",":
                parts = first_line.strip().split(",")
            else:
                parts = re.split(delimiter, first_line.strip())

            # Check for known header keywords
            header_keywords = {
                "sampleid",
                "sampleids",
                "samples",
                "sampid",
                "sampids",
                "samps",
                "populationid",
                "populationids",
                "popid",
                "popids",
                "populations",
                "pop",
                "pops",
                "popid",
                "popids",
            }
            if any(part.lower() in header_keywords for part in parts):
                return 0
            else:
                return None

    def _is_numeric(self, s: str) -> bool:
        """Check if a string can be converted to a float.

        Args:
            s (str): The string to check.

        Returns:
            bool: True if the string is numeric, False otherwise.
        """
        try:
            float(s)
            return True
        except ValueError:
            return False

    def write_popmap(self, output_file: str) -> None:
        """Write the population map dictionary to a file.

        Writes the population map dictionary, where SampleIDs are keys and the associated population ID are values, to the specified output file.

        Args:
            output_file (str): The filename of the output file to write the population map.
        """
        with open(output_file, "w") as f:
            sorted_dict = dict(sorted(self._popdict.items(), key=lambda item: item[1]))

            for key, value in sorted_dict.items():
                f.write(f"{key}: {value}\n")

    def get_pop_counts(self, genotype_data: Any) -> None:
        """Print out unique population IDs and their counts.

        Prints the unique population IDs along with their respective counts. It also generates a plot of the population counts.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing the alignment data.
        """
        # Count the occurrences of each unique value
        value_counts = Counter(self._popdict.values())

        msg = "\n\nFound the following populations:\nPopulation\tCount\n"
        for value, count in value_counts.items():
            msg += f"\n{value:<10}{count:<10}"
        msg += "\n"
        self.logger.info(msg)

        plotting = Plotting(genotype_data, **genotype_data.plot_kwargs)
        plotting.plot_pop_counts(list(self._popdict.values()))

    def validate_popmap(
        self, samples: List[str], force: bool = False
    ) -> bool | Dict[str, str]:
        """Validate that all alignment sample IDs are present in the population map.

        Args:
            samples (List[str]): List of SampleIDs present in the alignment.
                The list of SampleIDs to be validated against the population map.

            force (bool, optional): If True, return a subset dictionary without the keys that weren't found. If False, return a boolean indicating whether all keys were found. Defaults to False.

        Returns:
            bool | Dict[str, str]: If force is False, returns True if all alignment samples are present in the population map and all population map samples are present in the alignment. Returns False otherwise. If force is True, returns a subset of the population map containing only the samples present in the alignment.

        Raises:
            ValueError: Raises an exception if duplicate sample IDs are found in the population map file.
        """
        if len(set(samples)) != len(samples):
            counter = Counter(samples)
            duplicates = [item for item, count in counter.items() if count > 1]
            msg = (
                f"Duplicate sample IDs found in the popmapfile: {','.join(duplicates)}"
            )
            self.logger.error(msg)
            raise exceptions.PopmapFileFormatError(msg)

        # Sort by alignment order.
        self._popdict = {
            str(k): self._popdict[k] for k in samples if k in self._popdict
        }

        if not all(isinstance(k, str) for k in self._popdict.keys()):
            try:
                self._popdict = {str(k): v for k, v in self._popdict.items()}
            except Exception as e:
                msg = f"Error converting popmap keys to string: {e}"
                self.logger.error(msg)
                raise exceptions.PopmapFileFormatError(msg)

        if force:
            # Create a boolean array where True indicates presence in popmap
            self._sample_indices = np.isin(samples, list(self._popdict.keys()))

            if not self._sample_indices.any():
                msg = "No samples found in the population map. Check the population map file format or path supplied to `popmapfile`."
                self.logger.error(msg)
                raise exceptions.PopmapFileFormatError(msg)

            if np.count_nonzero(self._sample_indices) != len(self._popdict):
                self.logger.warning(
                    "Not all samples in the alignment are present in the population map. `force` is set to True, so only the samples present in the population map, and vice versa, will be returned."
                )

            else:
                invalid_samp_popdict = [
                    False if samp in self._popdict else True for samp in samples
                ]

                invalid_popdict_samp = [
                    False if samp in samples else True for samp in self._popdict.keys()
                ]
                if any(invalid_samp_popdict):
                    for samp in samples:
                        if samp not in self._popdict:
                            msg = f"SampleID missing from the population map: {samp}"
                            self.logger.error(msg)
                            raise exceptions.PopmapKeyError(samp)

                if any(invalid_popdict_samp):
                    for samp in self._popdict.keys():
                        if samp not in samples:
                            msg = f"PopulationID missing from the alignment: {samp}"
                            self.logger.error(msg)
                            raise exceptions.PopmapKeyError(samp)
        return True

    def subset_popmap(
        self,
        samples: List[str],
        include: List[str] | None,
        exclude: List[str] | None,
    ) -> None:
        """Subset the population map based on inclusion and exclusion criteria.

        Subsets the population map by including only the specified populations (include) and excluding the specified populations (exclude).

        Args:
            samples (List[str]): List of samples from alignment.

            include (List[str] | None): List of populations to include in the subset.
                The populations to include in the subset of the population map.

            exclude (List[str] | None): List of populations to exclude from the subset of the population map.

        Raises:
            ValueError: Raises an exception if populations are present in both include and exclude lists.

            TypeError: Raises an exception if include or exclude arguments are not lists.

            ValueError: Raises an exception if the population map is empty after subsetting.

        """
        if include is None and exclude is None:
            return None

        if include is not None and exclude is not None:
            include_set = set(include)
            exclude_set = set(exclude)
            common = ",".join(list(include_set & exclude_set))
            if common:
                msg = (
                    f"Populations found in both include_pops and exclude_pops: {common}"
                )
                self.logger.error(msg)
                raise ValueError(msg)

        if include is not None:
            self._validate_pop_subset_lists(include)

            popmap = {k: v for k, v in self._popdict.items() if v in include}
            inc_idx = np.isin(samples, list(popmap.keys()))  # Boolean array
        else:
            inc_idx = np.ones(len(samples), dtype=bool)  # All True

        if exclude is not None:
            self._validate_pop_subset_lists(exclude)

            if include is None:
                popmap = self._popdict

            popmap = {k: v for k, v in popmap.items() if v not in exclude}
            exc_idx = np.isin(samples, list(popmap.keys()))  # Boolean array
        else:
            exc_idx = np.ones(len(samples), dtype=bool)  # All True

        if not popmap:
            msg = "popmap was empty after subseting with 'include_pops' and/ or 'exclude_pops' arguments."
            self.logger.error(msg)
            raise ValueError(msg)

        # Boolean intersection of inclusion and exclusion conditions
        indices = np.logical_and(inc_idx, exc_idx)

        if self._sample_indices is None:
            self._sample_indices = indices
        else:
            self._sample_indices = np.logical_and(indices, self._sample_indices)

        self.popmap = popmap

    def _validate_pop_subset_lists(
        self, l: List[str], lname: str = "include_pops"
    ) -> None:
        """Validates the elements in the given list `l` to ensure they are all of type `str`.

        Args:
            l (List[str]): The list to be validated and type checked.
            lname (str, optional): The name of the list being validated. Defaults to "include_pops".

        Raises:
            TypeError: If any element in the list is not of type `str`.

        """
        if not all(isinstance(x, str) for x in l):
            all_types = set([type(x) for x in l])
            msg = f"Invalid type encountered in '{lname}'. Expected str, but got: {all_types}"
            self.logger.error(msg)
            raise TypeError(msg)

    def _flip_dictionary(self, input_dict: Dict[str, str]) -> Dict[str, List[str]]:
        """Flip the keys and values of a dictionary.

        Flips the keys and values of the input dictionary, where the original keys become values and the original values become keys.

        Args:
            input_dict (Dict[str, str]): The input dictionary to be flipped.

        Returns:
            Dict[str, List[str]]: The flipped dictionary with the original values as keys and lists of original keys as values.

        """
        flipped_dict = defaultdict(list)
        for sample_id, population_id in input_dict.items():
            if population_id in flipped_dict:
                flipped_dict[population_id].append(sample_id)
            else:
                flipped_dict[population_id] = [sample_id]
        return flipped_dict

    @property
    def popmap(self) -> Dict[str, str]:
        """Get the population dictionary.

        Returns:
            Dict[str, str]: Dictionary with SampleIDs as keys and the corresponding population ID as values.
        """
        return self._popdict

    @popmap.setter
    def popmap(self, value: Dict[str, str | int]) -> None:
        """Setter for the population map dictionary.

        Args:
            value (Dict[str, str | int]): Dictionary object with SampleIDs as keys and the associated population ID as the value.

        Raises:
            TypeError: Raises an exception if the value is not a dictionary object.

        """
        self._popdict = value

    @property
    def sample_indices(self) -> np.ndarray:
        """Get the indices of the subset samples from the population map as a boolean array.

        Returns:
            np.ndarray: Boolean array representing the subset samples.
        """
        if self._sample_indices is None:
            return np.ones(len(self._popdict), dtype=bool)
        return self._sample_indices

    @property
    def popmap_flipped(self) -> Dict[str, List[str]]:
        """Associate unique populations with lists of SampleIDs.

        Returns:
            Dict[str, List[str]]: Dictionary with unique populations as keys and lists of associated SampleIDs as values.

        """
        return self._flip_dictionary(self._popdict)

    def __len__(self):
        return len(list(self._popdict.keys()))

    def __getitem__(self, idx: str) -> str:
        """Get the population ID associated with a given SampleID.

        Args:
            idx (str): The SampleID for which to retrieve the associated population ID.

        Returns:
            str: The population ID associated with the given SampleID.

        Raises:
            KeyError: If the SampleID is not found in the population map.
        """
        if idx in self._popdict:
            return self._popdict[idx]
        else:
            msg = f"Sample {idx} not in popmap: {self.filename}"
            self.logger.error(msg)
            raise KeyError(msg)

    def __contains__(self, idx: str) -> bool:
        """Check if a SampleID is present in the population map.

        Args:
            idx (str): The SampleID to check for presence in the population map.

        Returns:
            bool: True if the SampleID is present in the population map, False otherwise.
        """
        return idx in self._popdict

    def __repr__(self):
        return f"ReadPopmap(filename={self.filename}, verbose={self.verbose})"

    def __str__(self):
        output = ""
        for key, value in self._popdict.items():
            output += f"{key}\t{value}\n"
        return output.strip()

    def __iter__(self):
        return iter(self._popdict)
