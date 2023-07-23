import sys

from collections import Counter
from typing import List, Dict, Union
from pathlib import Path

from snpio.plotting.plotting import Plotting


class ReadPopmap:
    """Class to read and parse a population map file.

    Population map file should contain two tab-delimited columns, with the first being the SampleIDs and the second being the associated population ID. There should not be a header line in the popmap file.

    Args:
        filename (str): Filename for the population map.
            The population map file to be read and parsed.

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

    def __init__(self, filename: str, verbose: bool = True) -> None:
        """Class constructor."""
        self.filename: str = filename
        self.verbose = verbose
        self._popdict: Dict[str, str] = dict()
        self._sample_indices = None
        self.read_popmap()

    def read_popmap(self) -> None:
        """Read a population map file from disk into a dictionary object.

        The dictionary will have SampleIDs as keys and the associated population ID as the values.

        Raises:
            AssertionError: Raises an exception if the population map file does not have two columns or is not tab-delimited.

            AssertionError: Raises an exception if the dictionary object is empty after reading the population map file.
        """
        with open(self.filename, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()

                if len(cols) > 2:
                    raise AssertionError(
                        f"Invalid number of columns in tab-delimited popmap "
                        "file. Expected 2, but got {len(cols)}"
                    )

                ind = cols[0]
                pop = cols[1]
                self._popdict[ind] = pop

        if not self._popdict:
            raise AssertionError(
                "Found empty popmap file. Please check to see if the popmap "
                "file is a two-column, tab-delimited file with no header line."
            )

        if self.verbose:
            print("Found the following populations:\nPopulation\tCount\n")
        self.get_pop_counts()

    def write_popmap(self, output_file: str) -> None:
        """Write the population map dictionary to a file.

        Writes the population map dictionary, where SampleIDs are keys and the associated population ID are values, to the specified output file.

        Args:
            output_file (str): The filename of the output file to write the population map.

        Raises:
            IOError: Raises an exception if there is an error writing to the output file.
        """
        with open(output_file, "w") as f:
            sorted_dict = dict(
                sorted(self._popdict.items(), key=lambda item: item[1])
            )

            for key, value in sorted_dict.items():
                f.write(f"{key}: {value}\n")

    def get_pop_counts(self) -> None:
        """Print out unique population IDs and their counts.

        Prints the unique population IDs along with their respective counts. It also generates a plot of the population counts.
        """
        # Count the occurrences of each unique value
        value_counts = Counter(self._popdict.values())

        if self.verbose:
            for value, count in value_counts.items():
                print(f"{value:<10}{count:<10}")

        Path("plots").mkdir(exist_ok=True, parents=True)
        Plotting.plot_pop_counts(list(self._popdict.values()), "plots")

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
        # Make sure all sampleIDs are unique.
        samples = list(set(samples))
        if force:
            # Create a subset dictionary containing only the samples present in the alignment
            subset_dict = {
                samp: self._popdict[samp]
                for samp in samples
                if samp in self._popdict
            }

            sample_indices = [
                i
                for i, x in enumerate(self._popdict.keys())
                if x in self._popdict
            ]

            self._popdict = subset_dict
            self._sample_indices = sample_indices
            return True
        else:
            for samp in samples:
                if samp not in self._popdict:
                    return False
            for samp in self._popdict.keys():
                if samp not in samples:
                    return False
            return True

    def subset_popmap(self, include: List[str], exclude: List[str]) -> None:
        """Subset the population map based on inclusion and exclusion criteria.

        Subsets the population map by including only the specified populations (include) and excluding the specified populations (exclude).

        Args:
            include (List[str]): List of populations to include in the subset.
                The populations to include in the subset of the population map.

            exclude (List[str]): List of populations to exclude from the subset.
                The populations to exclude from the subset of the population map.

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
            common = include_set & exclude_set
            if common:
                raise ValueError(
                    f"populations {common} were in both include_pops and exclude_pops"
                )

        if include is not None:
            if not isinstance(include, list):
                raise TypeError(
                    f"include_pops must be a list of strings, but got {type(include)}"
                )

            popmap = {k: v for k, v in self._popdict.items() if v in include}
            inc_idx = [
                i for i, x in enumerate(self._popdict.values()) if x in include
            ]
        else:
            inc_idx = list(range(len(self._popdict)))

        if exclude is not None:
            if not isinstance(exclude, list):
                raise TypeError(
                    f"include_pops must be a list of strings, but got {type(include)}"
                )

            if include is None:
                popmap = self._popdict

            popmap = {k: v for k, v in popmap.items() if v not in exclude}

            exc_idx = [
                i for i, x in enumerate(self._popdict.values()) if x in exclude
            ]
        else:
            exc_idx = list(range(len(self._popdict)))

        if not popmap:
            raise ValueError(
                "popmap was empty after subseting with 'include_pops' and 'exclude_pops'"
            )

        indices = inc_idx + exc_idx
        indices = list(set(indices))
        indices.sort()

        self._popdict = popmap
        if self._sample_indices is None:
            self._sample_indices = indices
        else:
            indices += self._sample_indices
            indices = list(set(indices))
            indices.sort()
            self._sample_indices = indices

    def _flip_dictionary(self, input_dict):
        """Flip the keys and values of a dictionary.

        Flips the keys and values of the input dictionary, where the original keys become values and the original values become keys.

        Args:
            input_dict (dict): The input dictionary to be flipped.

        Returns:
            dict: The flipped dictionary with the original values as keys and lists of original keys as values.

        """
        flipped_dict = {}
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
    def popmap(self, value) -> Dict[str, str]:
        """Setter for the population map dictionary.

        Args:
            value (Dict[str, str]): Dictionary object with SampleIDs as keys and the associated population ID as the value.
                The dictionary representing the population map to be set.

        Raises:
            TypeError: Raises an exception if the value is not a dictionary object.

        """
        if not isinstance(value, dict):
            raise TypeError(
                f"popmap must be a dictionary object, but got {type(value)}"
            )

        self._popdict = value

    @property
    def sample_indices(self) -> List[int]:
        """Get the indices of the subset samples from the population map.

        Returns:
            List[int]: List of indices representing the subset samples.

        """
        if self._sample_indices is None:
            return list(range(len(self._popdict)))
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
            sys.exit(f"\nSample {idx} not in popmap: {self.filename}\n")

    def __contains__(self, idx):
        if idx in self._popdict:
            return True
        else:
            return False
