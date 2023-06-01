import sys

from collections import Counter
from typing import List, Dict

class ReadPopmap:
	"""Class to read and parse a population map file.

	Population map file should contain two tab-delimited columns, with the first being the SampleIDs and the second being the associated population ID. There should not be a header line in the popmap file.

	Examples:
		Sample1\tPopulation1
		Sample2\tPopulation1
		Sample3\tPopulation2
		Sample4\tPopulation2
	"""

	def __init__(self, filename: str, verbose: bool=True) -> None:
		"""Class constructor.

		Args:
			filename (str): Filename for population map.
		"""
		self.filename: str = filename
		self.verbose = verbose
		self._popdict: Dict[str, str] = dict()
		self.read_popmap()

	def read_popmap(self) -> None:
		"""Read a population map file from disk into a dictionary object.

		The dictionary will have SampleIDs as keys and the associated population ID as the values.

		Raises:
			AssertionError: Ensures that popmap file has two columns and is tab-delimited.
			AssertionError: Ensures that the dictionary object is not empty after reading the popmap file.
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
			self._get_pop_counts()

	def _get_pop_counts(self) -> None:
		"""Print out unique population IDs and their counts."""
		# Count the occurrences of each unique value
		value_counts = Counter(self._popdict.values())

		for value, count in value_counts.items():
			print(f"{value:<10}{count:<10}")

	def validate_popmap(self, samples: List[str]) -> bool:
		"""Validate that all alignment sample IDs are present in the popmap.
		
		Args:
			samples (List[str]): List of SampleIDs present in the alignment.

		Returns:
			bool: True if all alignment samples are present in the popmap and all popmap samples are present in the alignment. False otherwise.
		
		"""
		# Make sure all sampleIDs are unique.
		samples = list(set(samples))
		for samp in samples:
			if samp not in self._popdict:
				return False
		for samp in self._popdict.keys():
			if samp not in samples:
				return False
		return True

	def _flip_dictionary(self, input_dict):
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
