from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union

import numpy as np

from snpio.plotting.plotting import Plotting


class ReadPopmap:
    """Class to read and parse a population map file.

    Population map file should contain two tab-delimited columns, with the first being the SampleIDs and the second being the associated population ID. There should not be a header line in the popmap file.

    Args:
        filename (str): Filename for the population map.
            The population map file to be read and parsed.

        logger (logging): Logger object.

        verbose (bool): Verbosity setting (True or False).
            If True, enables verbose output. If False, suppresses verbose output.

    Examples:
        Example population map file:

        ```
        Sample1\tPopulation1
        Sample2\tPopulation1
        Sample3\tPopulation2
        Sample4\tPopulation2
        ```
    """

    def __init__(self, filename: str, logger: object, verbose: bool = True) -> None:
        """Class constructor."""
        self.filename: str = filename
        self.verbose = verbose
        self._popdict: Dict[str, str] = dict()
        self._sample_indices = None
        self.logger = logger

        self.read_popmap()

    def read_popmap(self) -> None:
        """Read a population map file from disk into a dictionary object.

        The dictionary will have SampleIDs as keys and the associated population ID as the values.

        Raises:
            AssertionError: Raises an exception if the population map file does not have two columns or is not whitespace-delimited.

            AssertionError: Raises an exception if the dictionary object is empty after reading the population map file.
        """
        with open(self.filename, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()

                if len(cols) > 2:
                    msg = f"Invalid number of columns in whitespace-delimited popmap file. Expected 2, but got {len(cols)}"
                    self.logger.error(msg)
                    raise AssertionError(msg)

                ind = cols[0]
                pop = cols[1]
                self._popdict[ind] = pop

        if not self._popdict:
            raise AssertionError(
                "Found empty popmap file. Please check to see if the popmap "
                "file is a two-column, whitespace-delimited file with no header line."
            )

    def write_popmap(self, output_file: str) -> None:
        """Write the population map dictionary to a file.

        Writes the population map dictionary, where SampleIDs are keys and the associated population ID are values, to the specified output file.

        Args:
            output_file (str): The filename of the output file to write the population map.

        Raises:
            IOError: Raises an exception if there is an error writing to the output file.
        """
        with open(output_file, "w") as f:
            sorted_dict = dict(sorted(self._popdict.items(), key=lambda item: item[1]))

            for key, value in sorted_dict.items():
                f.write(f"{key}: {value}\n")

    def get_pop_counts(self, genotype_data) -> None:
        """Print out unique population IDs and their counts.

        Prints the unique population IDs along with their respective counts. It also generates a plot of the population counts.

        Args:
            genotype_data (GenotypeData): GenotypeData object containing the alignment data.
        """
        # Count the occurrences of each unique value
        value_counts = Counter(self._popdict.values())

        if self.verbose:
            msg = "\n\nFound the following populations:\nPopulation\tCount\n"
            for value, count in value_counts.items():
                msg += f"\n{value:<10}{count:<10}"
            msg += "\n"
            self.logger.info(msg)

        plotting = Plotting(genotype_data, **genotype_data.plot_kwargs)
        plotting.plot_pop_counts(list(self._popdict.values()))

    def validate_popmap(
        self, samples: List[str], force: bool = False
    ) -> Union[bool, Dict[str, str]]:
        """Validate that all alignment sample IDs are present in the population map.

        Args:
            samples (List[str]): List of SampleIDs present in the alignment.
                The list of SampleIDs to be validated against the population map.

            force (bool, optional): If True, return a subset dictionary without the keys that weren't found. If False, return a boolean indicating whether all keys were found. Defaults to False.

        Returns:
            Union[bool, Dict[str, str]]: If force is False, returns True if all alignment samples are present in the population map and all population map samples are present in the alignment. Returns False otherwise. If force is True, returns a subset of the population map containing only the samples present in the alignment.

        """
        if len(set(samples)) != len(samples):
            counter = Counter(samples)
            duplicates = [item for item, count in counter.items() if count > 1]
            msg = (
                f"Duplicate sample IDs found in the popmapfile: {','.join(duplicates)}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Sort by alignment order.
        self._popdict = {k: self._popdict[k] for k in samples if k in self._popdict}

        if force:
            # Create a boolean array where True indicates presence in popmap
            self._sample_indices = np.isin(samples, list(self._popdict.keys()))
        else:
            for samp in samples:
                if samp not in self._popdict:
                    return False
            for samp in self._popdict.keys():
                if samp not in samples:
                    return False
        return True

    def subset_popmap(
        self,
        samples: List[str],
        include: Optional[List[str]],
        exclude: Optional[List[str]],
    ) -> None:
        """Subset the population map based on inclusion and exclusion criteria.

        Subsets the population map by including only the specified populations (include) and excluding the specified populations (exclude).

        Args:
            samples (List[str]): List of samples from alignment.

            include (List[str]): List of populations to include in the subset.
                The populations to include in the subset of the population map.

            exclude (List[str]): List of populations to exclude from the subset of the population map.

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
            msg = "popmap was empty after subseting with 'include_pops' and 'exclude_pops'"
            self.logger.error(msg)
            raise ValueError(msg)

        # Boolean intersection of inclusion and exclusion conditions
        indices = np.logical_and(inc_idx, exc_idx)

        if self._sample_indices is None:
            self._sample_indices = indices
        else:
            self._sample_indices = np.logical_and(indices, self._sample_indices)

        self._popdict = popmap

    def _validate_pop_subset_lists(self, l, lname="include_pops"):
        """
        Validates the elements in the given list `l` to ensure they are all of type `str`\.

        Args:
            l (list): The list to be validated.
            lname (str, optional): The name of the list being validated. Defaults to "include_pops".

        Raises:
            TypeError: If any element in the list is not of type `str`.

        """
        if not all(isinstance(x, str) for x in l):
            all_types = set([type(x) for x in l])
            msg = f"Invalid type encountered in '{lname}'. Expected str, but got: {all_types}"
            self.logger.error(msg)
            raise TypeError(msg)

    def _flip_dictionary(self, input_dict):
        """Flip the keys and values of a dictionary.

        Flips the keys and values of the input dictionary, where the original keys become values and the original values become keys.

        Args:
            input_dict (dict): The input dictionary to be flipped.

        Returns:
            dict: The flipped dictionary with the original values as keys and lists of original keys as values.

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
    def popmap(self, value: Dict[str, Union[str, int]]) -> None:
        """Setter for the population map dictionary.

        Args:
            value (Dict[str, Union[str, int]]): Dictionary object with SampleIDs as keys and the associated population ID as the value.

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

    def __getitem__(self, idx):
        if idx in self._popdict:
            return self._popdict[idx]
        else:
            msg = f"Sample {idx} not in popmap: {self.filename}"
            self.logger.error(msg)
            raise KeyError(msg)

    def __contains__(self, idx):
        if idx in self._popdict:
            return True
        else:
            return False

    def __repr__(self):
        return f"ReadPopmap(filename={self.filename}, verbose={self.verbose})"

    def __str__(self):
        output = ""
        for key, value in self._popdict.items():
            output += f"{key}\t{value}\n"
        return output.strip()

    def __iter__(self):
        return iter(self._popdict)
